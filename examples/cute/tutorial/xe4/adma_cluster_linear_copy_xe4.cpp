/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// The purpose of this tutorial is to demonstrate the use of ADMA for
// cluster-based linear copies, where multiple WGs in a cluster cooperatively
// load contiguous tiles from global memory into shared memory
// using the ASYNC_LINEAR_LOAD_MULTICAST operation.
// The algorithm is elaborated below in the kernel comments.

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gpu_generics.h>

#include <cute/arch/xe4_util.hpp>
#include "cute/arch/xe4_inline_pisa.hpp"
#include <cute/atom/copy_traits_xe4_adma.hpp>
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "../../../common/sycl_cute_common.hpp"
#include "../../../xe4/utils/validation.hpp"

#define SWAP_STORE 0

using namespace cute;
using sycl::ext::oneapi::this_work_item::get_nd_item;

constexpr static size_t SmemAlignment = 256;

//=============================================================================
// SharedStorage
// Holds FullTileElements = ClusterX * ChunkElements per WG.
// After the multicast load, all ClusterX WGs' SLMs are identical replicas
// of the tile.
//=============================================================================
template <class ElementA, class SmemLayoutA>
struct SharedStorage : cute::aligned_struct<SmemAlignment, _0>
{
  cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>, SmemAlignment> smem_A;
};

inline void xe4_syncthreads() {
  auto group = get_nd_item<1>().get_group();
  sycl::group_barrier(group);
}

//=============================================================================
// adma_cluster_linear_copy_device — kernel
//=============================================================================
template <class ProblemShape,
          int   ClusterSizeX,
          int   ChunkElements,
          class TA,
          class SmemLayoutA,
          class ADMA_A_LOAD, 
          class ADMA_A_STORE,
          class TC,
          class CStride>
void
adma_cluster_linear_copy_device(
    ProblemShape  problemSize,
    TA const*     A,
    SmemLayoutA   sA_layout,
    ADMA_A_LOAD   adma_load_a,
    ADMA_A_STORE  adma_store_c,
    TC*           C,
    CStride       dC,
    sycl::nd_item<3> item)
{
  static_assert(rank(problemSize) == 1);
  static_assert(is_static<SmemLayoutA>::value);
  static_assert(size<0>(SmemLayoutA{}) == ClusterSizeX * ChunkElements,
                "SmemLayoutA must hold the full cluster tile (ClusterSizeX * ChunkElements)");

  constexpr int FullTileElements = ClusterSizeX * ChunkElements;
  using SmemLayoutChunk = decltype(make_layout(Int<ChunkElements>{}, Int<1>{}));

  // ── Cluster coordinates ──────────────────────────────────────────────────
  //  cluster_id : position of this WG within its cluster
  //  tile_idx   : which full-tile (cluster) this WG belongs to
  uint32_t cluster_id = get_cluster_wgid<0>();
  uint32_t tile_idx   = BlockIdxX() / static_cast<uint32_t>(ClusterSizeX);

  // SLM allocation - Each WG allocates sizeof(TA) * FullTileElements bytes of SLM.
  using SharedStorageType = SharedStorage<TA, SmemLayoutA>;
  auto  ptr  = alloc_slm_buffer<uint8_t, sizeof(SharedStorageType)>(item.get_group());
  auto& smem = *reinterpret_cast<SharedStorageType*>(ptr);

  // Pointer to this WG's SLM chunk located at cluster-id offset
  TA* slm_base  = smem.smem_A.begin();
  TA* slm_chunk = slm_base + cluster_id * ChunkElements;

  // Tensor for this WG's SLM chunk (ChunkElements elements at slm_chunk offset)
  Tensor sA_chunk = make_tensor(make_smem_ptr(slm_chunk), SmemLayoutChunk{});

  // GMEM pointers
  //  tile_gmem_base  : start of the full tile in global memory
  //  chunk_gmem_base : start of this WG's GMEM slice within that tile
  uint32_t tile_gmem_base  = tile_idx * static_cast<uint32_t>(FullTileElements);
  uint32_t chunk_gmem_base = tile_gmem_base + cluster_id * static_cast<uint32_t>(ChunkElements);

  Tensor gA_chunk = make_tensor(make_gmem_ptr(A + chunk_gmem_base), SmemLayoutChunk{});
  Tensor gC_chunk = make_tensor(make_gmem_ptr(C + chunk_gmem_base), SmemLayoutChunk{});

  //  Multicast mask
  //  Bit i is set when WG i in the cluster should receive the payload.
  //  For an all-to-all broadcast, set all ClusterSizeX bits.
  constexpr uint32_t mcast_mask = (1u << ClusterSizeX) - 1u;

  // Transaction byte budgets
  //
  //  chunk_bytes     : bytes issued by THIS WG's single copy() call (CopyBytes of atom)
  //  full_tile_bytes : bytes THIS WG's barrier must track before firing
  //
  //  The multicast instruction delivers chunk_bytes to EVERY masked WG's SLM.
  //  Each WG's barrier therefore receives:
  //    from own load              : chunk_bytes  (direct)
  //    from each peer's multicast : chunk_bytes  (via shared_cluster)
  //    total = ClusterSizeX * chunk_bytes = full_tile_bytes for "tx bytes count" of barrier
  //
  constexpr uint32_t chunk_bytes      = static_cast<uint32_t>(ChunkElements * sizeof(TA));
  constexpr uint32_t full_tile_bytes  = static_cast<uint32_t>(FullTileElements * sizeof(TA));

  // ── Subgroup assignment ──────────────────────────────────────────────────
  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t warp_idx      = get_sg_id();

  auto& load_a_abar  = cutlass::arch::allocate_cluster_tx_barrier();
  auto& store_c_abar = cutlass::arch::allocate_cluster_tx_barrier();

  if (elect_one_thr && warp_idx == 0) {
    load_a_abar.init(1 /*numThreads*/);
  } else if (elect_one_thr && warp_idx == 1) {
    store_c_abar.init(1 /*numThreads*/);
  }
  xe4_syncthreads();

  // Load phase  (SG-0)
  if (warp_idx == 0 && elect_one_thr) {
    load_a_abar.arrive_and_expect_tx(full_tile_bytes);
    copy(adma_load_a.with(reinterpret_cast<uint64_t*>(&load_a_abar), mcast_mask),
         coalesce(gA_chunk),
         coalesce(sA_chunk));
  }
  sycl::group_barrier(item.get_sub_group());

  // Compute phase  (SG-1)
  if (warp_idx == 1 && elect_one_thr) {
    load_a_abar.try_wait(0 /*phase*/);

    for (int i = 0; i < ChunkElements; ++i) {
      float temp = 0;
      for (int j = cluster_id, k = 1; j < (cluster_id + ClusterSizeX); ++j, k++) {
        int wrap_cluster = j % ClusterSizeX;
        temp = temp + static_cast<float>(slm_base[i + wrap_cluster * ChunkElements]) * static_cast<float>(k);
      }
      sA_chunk[i] = temp;
    }
  }

  sycl::group_barrier(item.get_sub_group());

  // Store phase  (SG-1)
  if (warp_idx == 1 && elect_one_thr) {
    store_c_abar.arrive_and_expect_tx(chunk_bytes);
    copy(adma_store_c.with(reinterpret_cast<uint64_t*>(&store_c_abar)),
         coalesce(sA_chunk),
         coalesce(gC_chunk));
    store_c_abar.try_wait(0 /*phase*/);
  }
  sycl::group_barrier(item.get_sub_group());
}

//=============================================================================
// adma_cluster_linear_copy_host — host launcher
//=============================================================================

// Algorithm: 
// Problem is divided into chunks - each chunk of size ElementsPerChunk
// Every chunk is processed by a WG -> #WGs = #chunks
// Each cluster has ClusterSizeX WGs, so each cluster processes ClusterSizeX chunks = FullTile
// Whole GMEM is divided into #Clusters each of size FullTile
// Each WG loads its FullTile from the start of start of its cluster's tile in GMEM
// Thus, every WG in the cluster loads the same FullTile chunk from GMEM into its SLM, 
// which is performed by ADMA_LD_MULTICAST mechanism such that each WG's load contributes to filling every other WG's SLM as well

// Once every WG loads its FullTile - based on its id within its cluster
// it multiplies its native chunk matching with its id within its cluster
// with 1 and moves right and multiplies the next chunk with 2 and adds to the previous result
// WHile moving if the chunk hits the end, it wraps around to the start of the FullTile and
// continues the same process until it processes all the chunks in its cluster and
// stores the final result in its native chunk in SLM

template <int ClusterSizeX, int ElementsPerChunk, class TensorA, class TensorC>
void
adma_cluster_linear_copy_host(
    uint32_t      probSize,
    TensorA const& A,
    TensorC&       C,
    sycl::queue&   queue)
{
  using TA = typename TensorA::element_type;
  using TC = typename TensorC::element_type;

  TA const* A_ptr = &*A.data();
  TC*       C_ptr = &*C.data();
  auto      dC    = C.stride();

  constexpr int ClusterX       = ClusterSizeX;
  constexpr int ChunkElements  = ElementsPerChunk;
  constexpr int FullTile       = ClusterX * ChunkElements;

  static_assert(ChunkElements % 64 == 0,
                "ChunkElements should be a multiple of 64 for alignment");
  assert(probSize % FullTile == 0 && "probSize must be divisible by FullTile");

  constexpr int ChunkBytesA = ChunkElements * sizeof(TA);
  constexpr int ChunkBytesC = ChunkElements * sizeof(TC);
  using GmemTiledCopyA = cute::Copy_Traits<cute::XE4_ADMA_LINEAR_LOAD_MULTICAST_CLUSTER, cute::Int<ChunkBytesA>>;
  using GmemTiledCopyC = cute::Copy_Traits<cute::XE4_ADMA_LINEAR_STORE<>, cute::Int<ChunkBytesC>>;

  // ── SMEM layouts ──────────────────────────────────────────────────────────
  //
  //  SmemLayoutA : flat 1-D, holds FullTile
  //  SmemLayoutC : alias for store (same backing SLM, chunk slice only).
  //
  using SmemLayoutA = decltype(make_layout(Int<FullTile>{},   Int<1>{}));
  using SmemLayoutC = decltype(make_layout(Int<ChunkElements>{}, Int<1>{}));

  SmemLayoutA sA{};
  SmemLayoutC sC{};

  auto adma_load_a = make_tiled_copy(
      Copy_Atom<GmemTiledCopyA, TA>{},
      Layout<_1>{},
      Layout<Int<ChunkElements>>{}
  );

  auto adma_store_c = make_tiled_copy(
      Copy_Atom<GmemTiledCopyC, TC>{},
      Layout<_1>{},
      Layout<Int<ChunkElements>>{}
  );

  // ── Cluster launch configuration ─────────────────────────────────────────
  //
  //  ClusterShape for SYCL = (Z=1, Y=1, X=ClusterX).
  //  Total work-groups in X = ceil_div(probSize, ChunkElements).
  
  constexpr int NumCtrlWarps      = 16;
  constexpr int NumThreadsPerWarp = 32;

  namespace syclexp = sycl::ext::oneapi::experimental;
  sycl::range<3> clusterSize(1, 1, ClusterX);  // (Z,Y,X)
  syclexp::properties Props{syclexp::work_groups_per_cluster<3>(clusterSize)};

  uint32_t total_wgs_x = probSize / ChunkElements;
  sycl::range<3> local_range(1, NumCtrlWarps, NumThreadsPerWarp);
  sycl::range<3> group_range(1, 1, total_wgs_x);
  sycl::nd_range<3> Range(group_range * local_range, local_range);

  auto launch_cfg = syclexp::launch_config(Range, Props);
  syclexp::submit_with_event(queue, [&](sycl::handler& handler) {
    syclexp::nd_launch(handler, launch_cfg, [=](sycl::nd_item<3> item) ALWAYS_INLINE {
      adma_cluster_linear_copy_device<
          decltype(make_shape(probSize)),
          ClusterX,
          ChunkElements,
          TA,
          SmemLayoutA,
          decltype(adma_load_a),
          decltype(adma_store_c),
          TC,
          decltype(dC)
      >(
          make_shape(probSize),
          A_ptr, sA,
          adma_load_a, adma_store_c,
          C_ptr, dC,
          item);
    });
  }).wait();
}

//=============================================================================
// main
//=============================================================================
int main(int argc, char** argv)
{
  sycl::queue queue{sycl::gpu_selector_v};

  std::cout << "Running on device: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  int m = 512;
  int k = 64;

  constexpr int ClusterX = 4;
  constexpr int ChunkElements = 512;

  using TA = fp16;
  using TC = fp16;

  auto prob_shape = make_shape(m, k);
  uint32_t sizeA  = size(prob_shape);  // total elements

  // ── Allocate USM tensors ─────────────────────────────────────────────────
  auto A = make_shared_usm_tensor<TA, 'R'>(queue, 1, sizeA);
  auto C = make_shared_usm_tensor<TC, 'R'>(queue, 1, sizeA);

  random_fill(A);
  zero_fill(C);

  // Keep a reference copy before subbyte_pack for verification
  auto A_ref = make_shared_usm_tensor<TA, 'R'>(queue, 1, sizeA);
  copy(A, A_ref);
  subbyte_pack(A);

  auto C_ref = make_shared_usm_tensor<TC, 'R'>(queue, 1, sizeA);
  copy(A_ref, C_ref);

  int num_chunks = sizeA / ChunkElements;
  int num_clusters = num_chunks / ClusterX;

  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    int num_chunks_per_cluster = ClusterX;
    int cluster_num = chunk / num_chunks_per_cluster;
    int offset = cluster_num * num_chunks_per_cluster;

    for (int i = 0; i < ChunkElements; ++i) {
      int cluster_id = chunk % num_chunks_per_cluster;
      float temp = 0;
      for (int j = cluster_id, k = 1; j < (cluster_id + num_chunks_per_cluster); ++j, k++) {
        int wrap_cluster = j % num_chunks_per_cluster;
        temp = temp + static_cast<float>(A_ref[(offset + wrap_cluster) * ChunkElements + i]) * static_cast<float>(k);
      }
      C_ref[chunk * ChunkElements + i] = temp;
    }
  }

  // ── Warmup
  adma_cluster_linear_copy_host<ClusterX, ChunkElements>(sizeA, A, C, queue);
  queue.wait_and_throw();

  // ── Verification: C_ptr[i] must equal C_ref_ptr[i] for all i
  TA* C_ref_ptr = &*C_ref.data();
  TC* C_ptr     = &*C.data();
  TA* A_ref_ptr = &*A_ref.data();

  int err_cnt = 0;
  for (uint32_t i = 0; i < sizeA; ++i) {
    if (C_ref_ptr[i] != C_ptr[i]) {
      printf("Mismatch at index %u: C_ref = %f, C = %f\n",
             i,
             static_cast<float>(C_ref_ptr[i]),
             static_cast<float>(C_ptr[i]));
      ++err_cnt;
    }
  }
  printf("Verification: %s\n", (err_cnt == 0) ? "PASSED" : "FAILED");

  if (err_cnt != 0) {
    throw std::runtime_error(
        "Cluster linear copy verification failed: output does not match input.");
  }

  return 0;
}
