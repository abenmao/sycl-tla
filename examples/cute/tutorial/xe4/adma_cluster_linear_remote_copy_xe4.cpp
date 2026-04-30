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
// Cluster-based ADMA linear copy local-to-remote SLM kernel example (Xe4)
//
// This file demonstrates the use of ADMA linear copy operations in a cluster
// of work-groups, highlighting local-to-remote-slm copy

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>
#include <iostream>

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gpu_generics.h>

#include <cute/arch/xe4_util.hpp>
#include <cute/arch/cluster_xe4.hpp>
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
// Holds FullClusterElements = ClusterX * ChunkElements per WG.
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
// adma_cluster_linear_remote_copy_device — kernel
//
// Ring-reduce algorithm (ClusterSizeX − 1 rotation steps):
//
//  SLM layout per WG (3 × ChunkElements used within the FullTile allocation):
//    slm_local [ChunkElements] @ offset 0              — native chunk
//    slm_recv  [ChunkElements] @ offset ChunkElements  — landing zone for right neighbor's push
//    slm_accum [ChunkElements] @ offset 2*ChunkElements — running accumulator
//
//  Push direction (every step, simultaneously to a different WG's SLM in the cluster)
//    WG-k pushes local chunk → neighbor's slm_recv   (ring: WG-0 → WG-(N-1))
//    push_mask = (1u << right_id)  — only right neighbor, NOT self
//    tx_bytes  = chunk_bytes      — one remote copy, nothing written locally
//  After ClusterSizeX−1 steps every WG's slm_accum holds the sum of all rows
//  in its plane
//
//  Final step: each WG stores its slm_accum back to GMEM (per-WG, no mask).
//=============================================================================
template <class ProblemShape,
          int   ClusterSizeX,
          int   ChunkElements,
          class TA,
          class SmemLayoutA,
          class ADMA_A_LOAD_LOCAL,
          class ADMA_A_LOAD_REMOTE,
          class ADMA_C_STORE,
          class TC,
          class CStride>
void
adma_cluster_linear_remote_copy_device(
    ProblemShape        problemSize,
    TA const*           A,
    SmemLayoutA         sA_layout,
    ADMA_A_LOAD_LOCAL   adma_load_local,
    ADMA_A_LOAD_REMOTE  adma_load_remote,
    ADMA_C_STORE        adma_store_c,
    TC*                 C,
    CStride             dC,
    sycl::nd_item<3>    item)
{
  static_assert(rank(problemSize) == 1);
  static_assert(is_static<SmemLayoutA>::value);
  static_assert(size<0>(SmemLayoutA{}) == 3 * ChunkElements,
                "SmemLayoutA must hold the full cluster tile (ClusterX * Chunk)");

  constexpr int      FullClusterElements = ClusterSizeX * ChunkElements;
  constexpr uint32_t chunk_bytes      = static_cast<uint32_t>(ChunkElements * sizeof(TA));

  using SmemLayoutChunk = decltype(make_layout(Int<ChunkElements>{}, Int<1>{}));

  // ── Identity ──────────────────────────────────────────────────────────────
  uint32_t cluster_id = get_cluster_wgid<0>();
  uint32_t tile_idx   = BlockIdxX() / static_cast<uint32_t>(ClusterSizeX);

  // ── SLM allocation ────────────────────────────────────────────────────────
  //    [0 .. ChunkElements)           → native chunk (local chunk)
  //    [ChunkElements .. 2*ChunkElements) → slm_recv   (incoming push landing zone)
  //    [2*ChunkElements .. 3*ChunkElements) → slm_accum   (running accumulator)
  using SharedStorageType = SharedStorage<TA, SmemLayoutA>;
  auto  ptr  = alloc_slm_buffer<uint8_t, sizeof(SharedStorageType)>(item.get_group());
  auto& smem = *reinterpret_cast<SharedStorageType*>(ptr);

  TA* slm_base  = smem.smem_A.begin();
  TA* slm_local        = slm_base;                     // local chunk
  TA* slm_from_remote  = slm_base + ChunkElements;     // landing zone for right neighbor
  TA* slm_accum        = slm_base + 2 * ChunkElements; // running accumulator

  Tensor sA_local = make_tensor(make_smem_ptr(slm_local), SmemLayoutChunk{});
  Tensor sA_from_remote = make_tensor(make_smem_ptr(slm_from_remote), SmemLayoutChunk{});
  Tensor sA_accum = make_tensor(make_smem_ptr(slm_accum), SmemLayoutChunk{});
  
  // ── GMEM tensors ──────────────────────────────────────────────────────────
  uint32_t tile_gmem_base  = tile_idx * static_cast<uint32_t>(FullClusterElements);
  uint32_t chunk_gmem_base = tile_gmem_base + cluster_id * static_cast<uint32_t>(ChunkElements);

  Tensor gA_chunk = make_tensor(make_gmem_ptr(A + chunk_gmem_base), SmemLayoutChunk{});
  Tensor gC_chunk = make_tensor(make_gmem_ptr(C + chunk_gmem_base), SmemLayoutChunk{});

  // ── Push routing ──────────────────────────────────────────────────────────
  //  push_mask has only the right_id bit set (NOT self — LOCAL_TO_REMOTE writes).
  
  // ── SG / thread identity ──────────────────────────────────────────────────
  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t warp_idx      = get_sg_id();

  auto load_abar  = allocate_abar<0, 1>();
  auto store_abar = allocate_abar<1, 1>();
  auto push_abar  = allocate_abar<2, ClusterSizeX - 1>();

  if (elect_one_thr && warp_idx == 0) {
    xe4_initialize_barrier(load_abar[0], 1 /*numThreads*/);
    for (int s = 0; s < ClusterSizeX - 1; ++s) {
      xe4_initialize_barrier(push_abar[s], 1 /*numThreads*/);
    }
    xe4_initialize_barrier(store_abar[0], 1 /*numThreads*/);
  }
  cbar_arrive();
  cbar_wait();

  // ── Phase 1: Load native row GMEM → slm_local (per-WG, no multicast) ─────
  if (warp_idx == 0 && elect_one_thr) {
    xe4_set_barrier_transaction_bytes(load_abar[0], chunk_bytes);
    copy(adma_load_local.with(&load_abar[0]),
         coalesce(gA_chunk),
         coalesce(sA_local));    
  }

  sycl::group_barrier(item.get_sub_group());

  if (warp_idx == 1 && elect_one_thr)
  {
    xe4_wait_barrier(load_abar[0], 0 /*phase*/);
    for (int i = 0; i < ChunkElements; ++i) {
      slm_accum[i] = slm_local[i];
    }
  }

    for (int step = 0; step < ClusterSizeX - 1; ++step) 
    {
      if (warp_idx == 1 && elect_one_thr)
      {
        uint32_t right_id  = (cluster_id + step + 1) % static_cast<uint32_t>(ClusterSizeX);
        uint32_t push_mask = (1u << right_id);  // only right neighbor — NOT self

        copy(adma_load_remote.with(&push_abar[step], push_mask),
              coalesce(sA_local),         // src: this WG's native chunk in SLM
              coalesce(sA_from_remote));  // dst offset in neighbor's SLM same as offset for receiving from remote for this WG
          
        auto* remote_abar = get_remote_abar_address(&push_abar[step], right_id);
        abarrier_cluster_arrive_expect_tx(remote_abar, chunk_bytes);
     }

     if (warp_idx == 0 && elect_one_thr)
     {
        xe4_wait_barrier(push_abar[step], 0 /*phase*/);
        // Accumulate the incoming chunk from the right neighbor.
        for (int i = 0; i < ChunkElements; ++i) {
          slm_accum[i] = static_cast<TA>(
              static_cast<float>(slm_accum[i]) + static_cast<float>(slm_from_remote[i]));
        }
     }
      xe4_syncthreads();
      cbar_arrive();
      cbar_wait();
    }

  if (warp_idx == 0 && elect_one_thr)
  {
    xe4_set_barrier_transaction_bytes(store_abar[0], chunk_bytes);
    copy(adma_store_c.with(&store_abar[0]),
        coalesce(sA_accum),
        coalesce(gC_chunk));
    xe4_wait_barrier(store_abar[0], 0 /*phase*/);
  }

    
  sycl::group_barrier(item.get_sub_group());

  // ── Phase 3: Store slm_accum → GMEM (per-WG, no multicast) ──────────────
  
  sycl::group_barrier(item.get_sub_group());

}

//=============================================================================
// adma_cluster_linear_remote_copy_host — host launcher
//=============================================================================
//
// 3D Problem partitioning:
//
//   Total WGs    = num_planes × rows_per_plane
//   Total clusters = num_planes
//   Flat buffer size = num_planes × rows_per_plane × cols
//   Volume: num_planes × rows_per_plane × cols
//   Hardware mapping:
//     One cluster  ←→ one plane   (ClusterSizeX WGs, one per row)
//     One WG       ←→ one row     (processes ElementsPerChunk elements)
//     Total WGs    = num_planes × ClusterSizeX
//     Total clusters = num_planes
//
//   Ring-reduce algorithm within each plane:
//     Each WG loads its own row from GMEM into native SLM (via ADMA_LINEAR_LOAD,
//     no multicast, per-WG only).  Then, over ClusterSizeX-1 rotation steps,
//     every WG pushes its native chunk into all of its neighbour's SLM, one neighbor in a step,
//     (via LOCAL_TO_REMOTE_SLM) and accumulates the incoming data from its
//     neighbour.  After all rotations every WG holds the sum of all rows
//     in its plane.  Finally each WG writes its result chunk back to GMEM.
//     Every WG maintains 3 buffers - local native chunk + remote recv chunk + accumulation buffer, all of size ElementsPerChunk.

template <int ClusterSizeX, int ElementsPerChunk, class TensorA, class TensorC>
void
adma_cluster_linear_remote_copy_host(
    uint32_t      num_planes,
    TensorA const& A,
    TensorC&       C,
    sycl::queue&   queue)
{
  using TA = typename TensorA::element_type;
  using TC = typename TensorC::element_type;

  TA const* A_ptr = &*A.data();
  TC*       C_ptr = &*C.data();
  auto      dC    = C.stride();

  constexpr int ClusterX      = ClusterSizeX;
  constexpr int ChunkElements = ElementsPerChunk;
  constexpr int ChunkContextTile = 3 * ChunkElements; // local + remote + accum per WG
  constexpr int FullClusterElements = ClusterX * ChunkElements;

  static_assert(ChunkElements % 64 == 0,
                "ChunkElements (cols) must be a multiple of 64 for ADMA alignment");

  uint32_t probSize = num_planes * static_cast<uint32_t>(FullClusterElements);

  constexpr int ChunkBytesA = ChunkElements * sizeof(TA);
  constexpr int ChunkBytesC = ChunkElements * sizeof(TC);
  
  using GmemTiledCopyToLocal = cute::Copy_Traits<cute::XE4_ADMA_LINEAR_LOAD<>,
                                            cute::Int<ChunkBytesA>>;
  using GmemTiledCopyToRemote = cute::Copy_Traits<cute::XE4_ADMA_LINEAR_LOAD_LOCAL_TO_REMOTE_SLM_CLUSTER,
                                            cute::Int<ChunkBytesA>>;
  using GmemTiledCopyC = cute::Copy_Traits<cute::XE4_ADMA_LINEAR_STORE<>,
                                            cute::Int<ChunkBytesC>>;

  // ── SMEM layouts ──────────────────────────────────────────────────────────
  using SmemLayoutA = decltype(make_layout(Int<ChunkContextTile>{}, Int<1>{}));
  using SmemLayoutC = decltype(make_layout(Int<ChunkElements>{}, Int<1>{}));

  SmemLayoutA sA{};
  SmemLayoutC sC{};

  auto adma_load_local = make_tiled_copy(
      Copy_Atom<GmemTiledCopyToLocal, TA>{},
      Layout<_1>{},
      Layout<Int<ChunkElements>>{}
  );

  auto adma_load_remote = make_tiled_copy(
      Copy_Atom<GmemTiledCopyToRemote, TA>{},
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
  //  Grid (X-axis):
  //    cluster_size_x = ClusterX              (rows_per_plane WGs per cluster)
  //    total_wgs_x    = num_planes × ClusterX  (one WG per row across all planes)
  //    num_clusters   = num_planes             (one cluster per plane)
  //
  constexpr int NumCtrlWarps      = 16;
  constexpr int NumThreadsPerWarp = 32;

  namespace syclexp = sycl::ext::oneapi::experimental;
  sycl::range<3> clusterSize(1, 1, ClusterX);   // (Z=1, Y=1, X=ClusterX)
  syclexp::properties Props{syclexp::work_groups_per_cluster<3>(clusterSize)};

  // One WG per row; rows = num_planes × rows_per_plane
  uint32_t total_wgs_x = num_planes * static_cast<uint32_t>(ClusterX);
  sycl::range<3> local_range(1, NumCtrlWarps, NumThreadsPerWarp);
  sycl::range<3> group_range(1, 1, total_wgs_x);
  sycl::nd_range<3> Range(group_range * local_range, local_range);

  auto launch_cfg = syclexp::launch_config(Range, Props);
  syclexp::submit_with_event(queue, [&](sycl::handler& handler) {
    syclexp::nd_launch(handler, launch_cfg, [=](sycl::nd_item<3> item) ALWAYS_INLINE {
      adma_cluster_linear_remote_copy_device<
          decltype(make_shape(probSize)),
          ClusterX,
          ChunkElements,
          TA,
          SmemLayoutA,
          decltype(adma_load_local),
          decltype(adma_load_remote),
          decltype(adma_store_c),
          TC,
          decltype(dC)
      >(
          make_shape(probSize),
          A_ptr, sA,
          adma_load_local, adma_load_remote, adma_store_c,
          C_ptr, dC,
          item);
    });
  }).wait();
}

//=============================================================================
// main
//=============================================================================
//
// 3D Volume:  num_planes  ×  rows_per_plane  ×  cols
//
//   num_planes     — number of planes in the volume.  Each plane is
//                    dispatched to one cluster.
//   rows_per_plane — number of row-vectors per plane.  Must equal ClusterX
//                    so that one cluster (ClusterX WGs) covers one plane,
//                    with one WG per row.
//   cols           — number of elements per row-vector.  Must equal
//                    ChunkElements so that one WG processes exactly one row.
int main(int argc, char** argv)
{
  sycl::queue queue{sycl::gpu_selector_v};

  std::cout << "Running on device: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  constexpr int ClusterX      = 4;  
  constexpr int ChunkElements = 512;

  int num_planes = 4;

  uint32_t sizeA = static_cast<uint32_t>(num_planes)
                 * static_cast<uint32_t>(ClusterX)
                 * static_cast<uint32_t>(ChunkElements);

  std::cout << "Problem shape : num_planes=" << num_planes
            << "  rows_per_plane=" << ClusterX
            << "  cols=" << ChunkElements
            << "  total_elements=" << sizeA << std::endl;

  using TA = fp16;
  using TC = fp16;

  // ── Allocate USM tensors (flat, row-major volume layout) ─────────────────
  auto A = make_shared_usm_tensor<TA, 'R'>(queue, 1, sizeA);
  auto C = make_shared_usm_tensor<TC, 'R'>(queue, 1, sizeA);

  // Fill A with random integers for fp16 lossless accumulation.
  {
    constexpr int max_fp16_exact_int = 2048;
    constexpr int max_val = max_fp16_exact_int / ClusterX;
    static_assert(max_val >= 1, "ClusterX too large for lossless fp16 integers");
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(1, max_val);
    for (uint32_t i = 0; i < sizeA; ++i)
      A(i) = static_cast<TA>(dist(rng));
  }
  zero_fill(C);

  // Pack A into the sub-byte format expected by the hardware copy engine.
  // Keep A_ref for any future reference verification.
  auto A_ref = make_shared_usm_tensor<TA, 'R'>(queue, 1, sizeA);
  copy(A, A_ref);
  subbyte_pack(A);

  // ── CPU Reference ─────────────────────────────────────────────────────────
  auto C_ref = make_shared_usm_tensor<TC, 'R'>(queue, 1, sizeA);
  zero_fill(C_ref);

  {
    TA const* A_ref_ptr = &*A_ref.data();
    TC*       C_ref_ptr = &*C_ref.data();

    for (int p = 0; p < num_planes; ++p) {
      // Step 1: accumulate all rows of plane p into a temporary row buffer.
      std::vector<float> acc(ChunkElements, 0.0f);
      for (int row = 0; row < ClusterX; ++row) {
        int base = (p * ClusterX + row) * ChunkElements;
        for (int col = 0; col < ChunkElements; ++col) {
          acc[col] += static_cast<float>(A_ref_ptr[base + col]);
        }
      }

      // Step 2: replicate acc into every row of plane p in C_ref.
      for (int row = 0; row < ClusterX; ++row) {
        int base = (p * ClusterX + row) * ChunkElements;
        for (int col = 0; col < ChunkElements; ++col) {
          C_ref_ptr[base + col] = static_cast<TC>(acc[col]);
        }
      }
    }
  }

  adma_cluster_linear_remote_copy_host<ClusterX, ChunkElements>(
      static_cast<uint32_t>(num_planes), A, C, queue);
  queue.wait_and_throw();

  // ── Verification ──────────────────────────────────────────────────────────
  {
    TC const* C_ref_ptr = &*C_ref.data();
    TC const* C_ptr     = &*C.data();
    TA const* A_ref_ptr = &*A_ref.data();

    int err_cnt = 0;
    for (uint32_t i = 0; i < sizeA; ++i) {
      float ref = static_cast<float>(C_ref_ptr[i]);
      float got = static_cast<float>(C_ptr[i]);
      if (ref != got) {
        if (err_cnt < 16) {
          printf("Mismatch at [plane=%u row=%u col=%u]: ref=%f  got=%f\n",
                 i / (ClusterX * ChunkElements),
                 (i / ChunkElements) % ClusterX,
                 i % ChunkElements,
                 ref, got);
        }
        ++err_cnt;
      }
    }
    printf("Verification: %s  (errors=%d / %u)\n",
           (err_cnt == 0) ? "PASSED" : "FAILED", err_cnt, sizeA);
  }

  return 0;
}
