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

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// CuTe Tutorial: AMMA + ADMA pipelined GEMM with 2D Cluster Multicast on Xe4
//
// This tutorial extends amma_adma_xe4_cluster.cpp (which uses ClusterShape<2,1,1>)
// to support a 2D cluster shape: ClusterShape<2,2,1>.
//
// With ClusterShape<2,2,1>, four CTAs are grouped in a 2×2 grid:
//   - 2 CTAs along M and 2 CTAs along N.
//   - A tiles are multicast across all N-CTAs at the same M-row.
//   - B tiles are multicast across all M-CTAs at the same N-column.
//   - This reduces global memory traffic for both A and B operands.
//
// Key difference from the single-dimension cluster tutorial:
//   1. ClusterShape_MNK = Shape<_2, _2, _1>  →  4 CTAs per cluster
//   2. Multicast masks must correctly account for 2D cluster linearization
//   3. SYCL dimension ordering follows production convention:
//        clusterSize = (K, M, N)  →  N is dim-2 (fastest, X)
//      This ensures hardware linearization: rank = M_pos * N_size + N_pos
//   4. Both A and B benefit from multicast (unlike single-dim where only one does)
//
// Hardware linearization for ClusterShape<2,2,1> (SYCL cluster range (1,2,2)):
//
//   rank 0 = (M=0, N=0)    Multicast groups:
//   rank 1 = (M=0, N=1)      A row M=0: ranks {0,1} → mask_a = 0x3
//   rank 2 = (M=1, N=0)      A row M=1: ranks {2,3} → mask_a = 0xC
//   rank 3 = (M=1, N=1)      B col N=0: ranks {0,2} → mask_b = 0x5
//                              B col N=1: ranks {1,3} → mask_b = 0xA
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <string>
#include <type_traits>

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gpu_generics.h>

#include "cute/arch/mma_xe4_amma.hpp"
#include <cute/arch/mma_xe4.hpp>
#include <cute/atom/mma_traits_xe4_amma.hpp>
#include <cute/atom/copy_traits_xe4_adma.hpp>
#include <cute/arch/xe4_util.hpp>
#include <cute/arch/cluster_xe4.hpp>
#include "cutlass/arch/barrier.h"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "../../../common/sycl_cute_common.hpp"
#include "../../../xe4/utils/validation.hpp"

using namespace cute;

using sycl::ext::oneapi::this_work_item::get_nd_item;

constexpr static size_t SmemAlignment = 256;

template <class ElementA,
          class ElementB,
          class ElementC,
          class ElementAcc,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB,  // (N,K,P)
          class SmemLayoutC,  // (M,N)
          bool SameCD = std::is_same_v<ElementC, ElementAcc>>
struct SharedStorage;

// Specialization when ElementC == ElementAcc (e.g., both float): D aliases C
template <class ElementA, class ElementB, class ElementC, class ElementAcc,
          class SmemLayoutA, class SmemLayoutB, class SmemLayoutC>
struct SharedStorage<ElementA, ElementB, ElementC, ElementAcc,
                     SmemLayoutA, SmemLayoutB, SmemLayoutC, true>
  : cute::aligned_struct<SmemAlignment, _0>
{
  cute::array_aligned<ElementA,   cute::cosize_v<SmemLayoutA>, SmemAlignment> smem_A;
  cute::array_aligned<ElementB,   cute::cosize_v<SmemLayoutB>, SmemAlignment> smem_B;
  cute::array_aligned<ElementAcc, cute::cosize_v<SmemLayoutC>, SmemAlignment> smem_C;
};

// Specialization when ElementC != ElementAcc (e.g., fp16 output, float accumulator):
// separate smem_D buffer for the dtype-converted output
template <class ElementA, class ElementB, class ElementC, class ElementAcc,
          class SmemLayoutA, class SmemLayoutB, class SmemLayoutC>
struct SharedStorage<ElementA, ElementB, ElementC, ElementAcc,
                     SmemLayoutA, SmemLayoutB, SmemLayoutC, false>
  : cute::aligned_struct<SmemAlignment, _0>
{
  cute::array_aligned<ElementA,   cute::cosize_v<SmemLayoutA>, SmemAlignment> smem_A;
  cute::array_aligned<ElementB,   cute::cosize_v<SmemLayoutB>, SmemAlignment> smem_B;
  cute::array_aligned<ElementAcc, cute::cosize_v<SmemLayoutC>, SmemAlignment> smem_C;
  cute::array_aligned<ElementC,   cute::cosize_v<SmemLayoutC>, SmemAlignment> smem_D;
};

inline void xe4_syncthreads() {
  auto group = get_nd_item<1>().get_group();
  sycl::group_barrier(group);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Compute multicast masks for 2D cluster arrangement.
//
// SYCL cluster range is set as (K, M, N), so:
//   - get_cluster_wgid<0>() = X = dim-2 = N position within cluster
//   - get_cluster_wgid<1>() = Y = dim-1 = M position within cluster
//
// Hardware linearization: rank = M_pos * N_size + N_pos (N-fast)
//
// For ClusterShape<2,2,1> (2×2 grid, 4 CTAs total):
//   - A (shared across N at same M):
//       M=0: ranks 0,1 → mask = 0x3
//       M=1: ranks 2,3 → mask = 0xC
//     Formula: mask_a = ((1 << N_size) - 1) << (M_pos * N_size)
//
//   - B (shared across M at same N):
//       N=0: ranks 0,2 → mask = 0x5
//       N=1: ranks 1,3 → mask = 0xA
//     Formula: mask_b = Σ (1 << (i * N_size + N_pos)) for i=0..M_size-1
//
// This matches the production code in xe4_mma_warpspecialized.hpp (AlongM path).
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class ClusterShape_MNK>
CUTE_DEVICE auto
compute_cluster_masks(ClusterShape_MNK cluster_shape)
{
  auto [cluster_size_m, cluster_size_n, cluster_size_k] = cluster_shape;

  // With SYCL cluster range = (K, M, N):
  //   get_cluster_wgid<0>() = X = dim-2 = N position
  //   get_cluster_wgid<1>() = Y = dim-1 = M position
  uint32_t cluster_wgid_n = get_cluster_wgid<0>();  // N position (SYCL X / dim-2)
  uint32_t cluster_wgid_m = get_cluster_wgid<1>();  // M position (SYCL Y / dim-1)

  uint32_t cluster_mask_a = 0;
  uint32_t cluster_mask_b = 0;

  // Cooperative groups:
  //   coop_num_a = how many CTAs cooperate on loading A = cluster_size_n
  //   coop_num_b = how many CTAs cooperate on loading B = cluster_size_m
  uint32_t coop_num_a = cluster_size_n;
  uint32_t coop_num_b = cluster_size_m;

  // A tile at (m, k) is multicast to all N-CTAs sharing the same M-row.
  // With N-fast linearization: ranks at same M are contiguous.
  // mask_a = N_size consecutive bits starting at M_pos * N_size
  cluster_mask_a = ((1u << coop_num_a) - 1) << (cluster_wgid_m * coop_num_a);

  // B tile at (n, k) is multicast to all M-CTAs sharing the same N-column.
  // With N-fast linearization: ranks at same N are strided by N_size.
  // mask_b = bits at positions N_pos, N_pos + N_size, N_pos + 2*N_size, ...
  uint32_t cluster_mask_b_base = 1u << cluster_wgid_n;
  #pragma unroll
  for (uint32_t i = 0; i < coop_num_b; i++) {
    cluster_mask_b |= cluster_mask_b_base << (i * coop_num_a);
  }

  return make_tuple(cluster_mask_a, cluster_mask_b);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Device kernel: pipelined GEMM with 2D cluster multicast
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class ProblemShape, class CtaTiler, class TileShape, class ClusterShape,
          class TA, class SmemLayoutA, class SmemLayoutAcc, class ADMA_A,
          class TB, class SmemLayoutB, class ADMA_B, class ADMA_C,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler, TileShape tile_shape, ClusterShape cluster_shape,
            TA const* A, SmemLayoutA sA_layout, SmemLayoutAcc sC_layout, ADMA_A adma_load_a,
            TB const* B, SmemLayoutB sB_layout, ADMA_B adma_load_b, ADMA_C adma_store_c,
            TC* C, CStride dC,
            Alpha alpha, Beta beta,
            sycl::nd_item<3> item)
{
  // Preconditions
  static_assert(rank(shape_MNK) == 3);
  static_assert(rank(cta_tiler) == 3);

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  static_assert(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));         // BLK_M
  static_assert(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));         // BLK_N
  static_assert(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));         // BLK_K
  static_assert(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));         // BLK_K

  // ========================================================================
  // Step 1: Allocate tensor descriptors and shared memory
  // ========================================================================
  auto tdesc_a = allocate_tdesc<0>();
  auto tdesc_b = allocate_tdesc<1>();
  auto tdesc_c = allocate_tdesc<2>();

  adma_load_a.set_tensor_desc(tdesc_a);
  adma_load_b.set_tensor_desc(tdesc_b);
  adma_store_c.set_tensor_desc(tdesc_c);

  using SharedStorageType = SharedStorage<TA, TB, TC, float, SmemLayoutA, SmemLayoutB, SmemLayoutAcc>;
  auto ptr = alloc_slm_buffer<uint8_t, sizeof(SharedStorageType)>(item.get_group());
  auto& smem = *reinterpret_cast<SharedStorageType*>(ptr);

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.begin()), SmemLayoutA{});   // (BLK_M, BLK_K, PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.begin()), SmemLayoutB{});   // (BLK_N, BLK_K, PIPE)
  Tensor sC = make_tensor(make_smem_ptr(smem.smem_C.begin()), SmemLayoutAcc{}); // (BLK_M, BLK_N)

  // sD: when ElementC == float (same as accumulator), D aliases C.
  // When ElementC is fp16, D points to separate smem_D buffer.
  [[maybe_unused]] auto sD_tensor = [&]() {
    if constexpr (std::is_same_v<TC, float>) {
      return make_tensor(make_smem_ptr(reinterpret_cast<TC*>(smem.smem_C.begin())), SmemLayoutAcc{});
    } else {
      return make_tensor(make_smem_ptr(smem.smem_D.begin()), SmemLayoutAcc{});
    }
  }();
  auto sD = sD_tensor;

  // ========================================================================
  // Step 2: Create full TMA tensors and tile them
  // ========================================================================
  auto [M, N, K] = shape_MNK;
  auto mA = adma_load_a.get_tma_tensor(make_shape(M, K));   // (m, k)
  auto mB = adma_load_b.get_tma_tensor(make_shape(N, K));   // (n, k)
  auto mC = adma_store_c.get_tma_tensor(make_shape(M, N));  // (m, n)

  // ========================================================================
  // Step 3: Cluster-aware tile partitioning
  //
  // With SYCL range (K, M, N) → dim-1=M, dim-2=N:
  //   BlockIdxY() = get_group(1) = M tile index (global)
  //   BlockIdxX() = get_group(2) = N tile index (global)
  //   cta_coord = (M_tile, N_tile, _) for local_tile
  // ========================================================================
  auto cta_coord = make_coord(BlockIdxY(), BlockIdxX(), _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});  // (BLK_M, BLK_K, m, k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{}); // (BLK_N, BLK_K, n, k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1,  X>{}); // (BLK_M, BLK_N)

  // ========================================================================
  // Step 4: Cluster-aware TMA partition
  //
  // With 2D clusters, each CTA loads a portion of A and B cooperatively.
  // tma_partition uses the CTA's position within the cluster to determine
  // which portion of the shared memory this CTA is responsible for.
  //
  // For A: projected along N-modes (N CTAs cooperatively load A)
  // For B: projected along M-modes (M CTAs cooperatively load B)
  // ========================================================================
  TiledMma mma{};
  auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));

  // Block rank within the cluster: consistent with SYCL N-fast linearization
  // cluster_layout_mn: shape=(M,N), stride=(1,M_size) → M-fast CuTe ordering
  auto cluster_layout_mn = make_layout(select<0,1>(cluster_shape),
                                       make_stride(_1{}, get<0>(cluster_shape)));
  // With SYCL range (K,M,N): get_cluster_wgid<1>()=M_pos(Y), get_cluster_wgid<0>()=N_pos(X)
  uint32_t block_rank = cluster_layout_mn(make_coord(get_cluster_wgid<1>(), get_cluster_wgid<0>()));
  auto cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(block_rank);

  // Partition A: project along N-modes (CTAs along N share the same A tile)
  auto [tAgA, tAsA] = tma_partition(adma_load_a,
                                    get<2>(cta_coord_vmnk), make_layout(size<2>(cluster_layout_vmnk)),
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));

  // Partition B: project along M-modes (CTAs along M share the same B tile)
  auto [tBgB, tBsB] = tma_partition(adma_load_b,
                                    get<1>(cta_coord_vmnk), make_layout(size<1>(cluster_layout_vmnk)),
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));

  // Total stages
  auto K_PIPE_MAX = size<1>(tAsA);
  // Total count of tiles
  int K_TILE_MAX = size<1>(tAgA);
  // Current tile index in gmem to read from
  int k_tile = 0;

  // ========================================================================
  // Step 5: Compute pipeline transaction bytes and multicast masks
  // ========================================================================
  constexpr int dma_transaction_bytesA = (cosize(sA_layout) * sizeof(TA)) / K_PIPE_MAX;
  constexpr int dma_transaction_bytesB = (cosize(sB_layout) * sizeof(TB)) / K_PIPE_MAX;
  constexpr int dma_transaction_bytesC = (cosize(sC_layout) * sizeof(TC));

  // Compute multicast masks based on 2D cluster position
  auto [mcast_mask_a, mcast_mask_b] = compute_cluster_masks(cluster_shape);

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t warp_idx = get_sg_id();

  // ========================================================================
  // Step 6: Allocate and initialize async barriers
  // ========================================================================
  auto load_a_abar = allocate_abar<0, K_PIPE_MAX>();
  auto load_b_abar = allocate_abar<1, K_PIPE_MAX>();
  auto mma_abar    = allocate_abar<2, K_PIPE_MAX>();
  auto store_c_abar = allocate_abar<3>();

  if (elect_one_thr && warp_idx == 0) {
    for (int i = 0; i < K_PIPE_MAX; ++i) {
      xe4_initialize_barrier(load_a_abar[i], 1);
      xe4_initialize_barrier(load_b_abar[i], 1);
    }
  } else if (elect_one_thr && warp_idx == 1) {
    for (int i = 0; i < K_PIPE_MAX; ++i) {
      xe4_initialize_barrier(mma_abar[i], 1);
    }
    xe4_initialize_barrier(store_c_abar[0], 1);
  }
  xe4_syncthreads();

  // Phase bit for store C barrier
  int store_c_barrier_phase_bit = 0;

  // ========================================================================
  // Step 7: MMA partition and fragment setup
  // ========================================================================
  ThrMMA thr_mma = mma.get_thread_slice(ThreadIdxX());
  Tensor tCsA = thr_mma.partition_A(sA);   // (MMA, MMA_M, MMA_K, PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);   // (MMA, MMA_N, MMA_K, PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCsC = thr_mma.partition_C(sC);   // (MMA, MMA_M, MMA_N)

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCsC);

  Tensor tCsD = thr_mma.partition_C(sD);
  Tensor tCrD = thr_mma.make_fragment_C(tCsD);

  // Clear accumulator in SLM
  clear(tCsC);
  xe4_syncthreads();

  // ========================================================================
  // Step 8: Cluster sync before starting the pipeline
  //
  // All 4 CTAs in the 2×2 cluster must have their SLM initialized before any
  // multicast DMA can write to peer SLM via shared_cluster.
  // ========================================================================
  if constexpr (size(ClusterShape{}) > 1) {
    cluster_sync();
  }

  uint64_t mma_ctrl = 0x100;

  // Pipeline state management
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();

  // ========================================================================
  // Step 9: Warp-specialized mainloop
  //
  // Warp 0 (ADMA): loads A and B tiles from global → SLM with multicast
  //   - A loads use mcast_mask_a: delivers to all CTAs in same M-row
  //   - B loads use mcast_mask_b: delivers to all CTAs in same N-column
  // Warp 1 (AMMA): performs matrix multiply-accumulate from SLM
  // ========================================================================
  if (warp_idx == 0)
  {
    if (elect_one_thr)
    {
      // ---- Prologue: Fill all pipeline stages ----
      for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
      {
        xe4_set_barrier_transaction_bytes(load_a_abar[pipe], dma_transaction_bytesA);
        xe4_set_barrier_transaction_bytes(load_b_abar[pipe], dma_transaction_bytesB);
        copy(adma_load_a.with(&load_a_abar[pipe], mcast_mask_a), tAgA(_, k_tile), tAsA(_, pipe));
        copy(adma_load_b.with(&load_b_abar[pipe], mcast_mask_b), tBgB(_, k_tile), tBsB(_, pipe));
        ++k_tile;
      }

      // ---- Mainloop: Wait for MMA to consume, then load next tile ----
      for (int k_tile_next = k_tile; k_tile_next < K_TILE_MAX; ++k_tile_next)
      {
        int write_pipe = write_state.index();
        xe4_wait_barrier(mma_abar[write_pipe], write_state.phase());
        xe4_set_barrier_transaction_bytes(load_a_abar[write_pipe], dma_transaction_bytesA);
        xe4_set_barrier_transaction_bytes(load_b_abar[write_pipe], dma_transaction_bytesB);
        copy(adma_load_a.with(&load_a_abar[write_pipe], mcast_mask_a), tAgA(_, k_tile_next), tAsA(_, write_pipe));
        copy(adma_load_b.with(&load_b_abar[write_pipe], mcast_mask_b), tBgB(_, k_tile_next), tBsB(_, write_pipe));
        ++write_state;
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }
  else if (warp_idx == 1)
  {
    if (elect_one_thr)
    {
      // ---- Mainloop: Wait for load, then MMA ----
      for (int k_tile_next = 0; k_tile_next < K_TILE_MAX - 1; ++k_tile_next)
      {
        int read_pipe = read_state.index();
        xe4_wait_barrier(load_a_abar[read_pipe], read_state.phase());
        xe4_wait_barrier(load_b_abar[read_pipe], read_state.phase());
        xe4_set_barrier_transaction_bytes(mma_abar[read_pipe], 2);
        auto new_mma = mma.with(AMMA::TrackMethod<AMMA::Tracking::AB>{},
                                mma_ctrl,
                                &mma_abar[read_pipe], &mma_abar[read_pipe],
                                mcast_mask_a, mcast_mask_b);
        cute::gemm(new_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
        mma_ctrl = 0x000;
        ++read_state;
      }

      // ---- Last iteration: Track D+A+B completion for store barrier ----
      {
        int read_pipe = read_state.index();
        xe4_wait_barrier(load_a_abar[read_pipe], read_state.phase());
        xe4_wait_barrier(load_b_abar[read_pipe], read_state.phase());
        xe4_set_barrier_transaction_bytes(store_c_abar[0], 1);       // D barrier: 1 byte
        xe4_set_barrier_transaction_bytes(mma_abar[read_pipe], 2);   // A+B barriers: 2 bytes
        if constexpr (std::is_same_v<TC, float>) {
          // Same type: no dtype change, 3-operand DAB tracking
          auto new_mma = mma.with(AMMA::TrackMethod<AMMA::Tracking::DAB>{},
                                  mma_ctrl,
                                  &store_c_abar[0],                            // D → dedicated epilogue barrier
                                  &mma_abar[read_pipe], &mma_abar[read_pipe],  // A,B → pipeline slot barrier
                                  mcast_mask_a, mcast_mask_b);
          cute::gemm(new_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
        } else {
          // Dtype change: D(FP16) to separate buffer, 4-operand DAB tracking
          auto new_mma = mma.with(TC{},
                                  AMMA::TrackMethod<AMMA::Tracking::DAB>{},
                                  mma_ctrl,
                                  &store_c_abar[0],                            // D → dedicated epilogue barrier
                                  &mma_abar[read_pipe], &mma_abar[read_pipe],  // A,B → pipeline slot barrier
                                  mcast_mask_a, mcast_mask_b);
          cute::gemm(new_mma, tCrD, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
        }
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }

  xe4_syncthreads();

  // ========================================================================
  // Step 10: Epilogue — Store result back to GMEM
  // ========================================================================
  if (warp_idx == 0)
  {
    if (elect_one_thr)
    {
      xe4_wait_barrier(store_c_abar[0], 0);
      xe4_set_barrier_transaction_bytes(store_c_abar[0], dma_transaction_bytesC);
      if constexpr (std::is_same_v<TC, float>) {
        copy(adma_store_c.with(&store_c_abar[0]), tCsC, tCgC);
      } else {
        copy(adma_store_c.with(&store_c_abar[0]), tCsD, tCgC);
      }
      xe4_wait_barrier(store_c_abar[0], 1);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host-side setup for TN GEMM with 2D cluster
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class TensorA, class TensorB, class TensorC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TensorA const& A,
        TensorB const& B,
        Beta beta,
        TensorC& C,
        sycl::queue& queue)
{
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  using TA = typename TensorA::element_type;
  using TB = typename TensorB::element_type;
  using TC = typename TensorC::element_type;

  TA const* A_ptr = &*A.data();
  TB const* B_ptr = &*B.data();
  TC* C_ptr = &*C.data();

  auto dA = A.stride();
  auto dB = B.stride();
  auto dC = C.stride();

  Tensor mA = make_tensor(make_gmem_ptr(A_ptr), make_layout(make_shape(M,K), dA));
  Tensor mB = make_tensor(make_gmem_ptr(B_ptr), make_layout(make_shape(N,K), dB));
  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), make_layout(make_shape(M,N), dC));

  // ========================================================================
  // Tile and Cluster configuration
  //
  // ClusterShape<2,2,1>: 4 CTAs in a 2×2 grid.
  //   - A tile is multicast to 2 CTAs along N (same M row)
  //   - B tile is multicast to 2 CTAs along M (same N column)
  //   - Both operands benefit from reduced global memory traffic
  // ========================================================================
  using TileShape_MNK = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape_MNK = cute::Shape<cute::_2, cute::_2, cute::_1>;

  constexpr auto bM = get<0>(TileShape_MNK{});
  constexpr auto bN = get<1>(TileShape_MNK{});
  constexpr auto bK = get<2>(TileShape_MNK{});
  auto cta_tiler = make_shape(bM, bN, bK);
  constexpr auto bP = Int<3>{};  // Pipeline stages

  static constexpr auto majorA = cute::AMMA::Major::K;
  static constexpr auto majorB = cute::AMMA::Major::K;

  using ElementA = TA;
  using ElementB = TB;
  using ElementAccumulator = float;
  using ElementC = TC;

  // ========================================================================
  // MMA atom: ss_op_selector picks XE4_AMMA_AB_CLUSTER when cluster > 1
  // ========================================================================
  using TiledMma = decltype(cute::make_tiled_mma(
    cute::AMMA::ss_op_selector<
      ElementAccumulator,
      ElementA, ElementB,
      ElementAccumulator,
      decltype(cute::product_each(TileShape_MNK{})),
      ClusterShape_MNK, majorA, majorB>()
  ));

  TiledMma tiled_mma{};

  // SLM layouts
  using SmemLayoutAtomA =
    cute::conditional_t<
      majorA == cute::AMMA::Major::K,
      decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenRowMajor{})),
      decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenColMajor{}))
    >;
  using SmemLayoutAtomB =
    cute::conditional_t<
      majorB == cute::AMMA::Major::K,
      decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenRowMajor{})),
      decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenColMajor{}))
    >;
  using SmemLayoutAtomC =
    cute::conditional_t<
      majorA == cute::AMMA::Major::K,
      decltype(make_layout(cute::select<0, 1>(TileShape_MNK{}), GenRowMajor{})),
      decltype(make_layout(cute::select<0, 1>(TileShape_MNK{}), GenColMajor{}))
    >;

  using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
    cute::conditional_t<majorA == cute::AMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{},
    make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
    cute::conditional_t<majorB == cute::AMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  using SmemLayoutAcc = decltype(tile_to_shape(
    SmemLayoutAtomC{},
    make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), Step<_2,_1>{}));

  SmemLayoutA sA{};
  SmemLayoutB sB{};
  SmemLayoutAcc sC{};

  // ========================================================================
  // DMA atoms: XE4_ADMA_LOAD_MULTICAST for cluster multicast loads
  // ========================================================================
  using GmemTiledCopyA =
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::XE4_ADMA_LOAD,
      cute::XE4_ADMA_LOAD_MULTICAST
    >;

  using GmemTiledCopyB =
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::XE4_ADMA_LOAD,
      cute::XE4_ADMA_LOAD_MULTICAST
    >;

  using GmemTiledCopyC = cute::XE4_ADMA_STORE;

  // Cluster layout for TMA partition projection
  auto cluster_layout_vmnk = tiled_divide(make_layout(ClusterShape_MNK{}),
                                          make_tile(typename TiledMma::AtomThrID{}));

  // ADMA atoms for A/B copy (cluster-aware)
  auto adma_load_a = make_adma_atom_A_xe4(
      GmemTiledCopyA{},
      mA,
      SmemLayoutA{}(_,_,cute::Int<0>{}),
      TileShape_MNK{},
      TiledMma{},
      cluster_layout_vmnk);

  auto adma_load_b = make_adma_atom_B_xe4(
      GmemTiledCopyB{},
      mB,
      SmemLayoutB{}(_,_,cute::Int<0>{}),
      TileShape_MNK{},
      TiledMma{},
      cluster_layout_vmnk);

  // ADMA store atom for C (no multicast needed for stores)
  auto cta_tiler_mn = make_shape(bM, bN);
  auto adma_store_c = make_adma_copy<TC>(GmemTiledCopyC{}, mC, SmemLayoutAcc{}, cta_tiler_mn, Int<1>{});

  // ========================================================================
  // Launch configuration with 2D cluster support
  //
  // SYCL dimension mapping (matches production code convention):
  //   dim-0 (Z) = K cluster = 1  (unused)
  //   dim-1 (Y) = M cluster & M tiles
  //   dim-2 (X) = N cluster & N tiles
  //
  // This gives N-fast hardware linearization:
  //   rank = M_pos * N_size + N_pos
  //
  // BlockIdxY() = get_group(1) = M tile index
  // BlockIdxX() = get_group(2) = N tile index
  // ========================================================================
  constexpr int NumOfControlWarps = 2;
  constexpr int NumOfThreadsPerWarp = 32;

  namespace syclexp = sycl::ext::oneapi::experimental;
  auto [cluster_size_m, cluster_size_n, cluster_size_k] = ClusterShape_MNK{};
  // Map: dim-0=K, dim-1=M, dim-2=N (production convention: N is fastest)
  sycl::range<3> clusterSize(cluster_size_k, cluster_size_m, cluster_size_n);
  syclexp::properties Props{syclexp::work_groups_per_cluster<3>(clusterSize)};

  auto num_groups = ceil_div(prob_shape, TileShape_MNK{});
  sycl::range<3> local_range(1, NumOfControlWarps, NumOfThreadsPerWarp);
  // group_range: (Z=1, Y=M_tiles, X=N_tiles) matching cluster dim mapping
  sycl::range<3> group_range(1, get<0>(num_groups), get<1>(num_groups));
  sycl::nd_range<3> Range(group_range * local_range, local_range);

  auto launch_cfg = syclexp::launch_config(Range, Props);
  syclexp::submit_with_event(queue, [&](sycl::handler& handler) {
    syclexp::nd_launch(handler, launch_cfg, [=](sycl::nd_item<3> item) ALWAYS_INLINE {
      gemm_device<decltype(prob_shape), decltype(cta_tiler), decltype(TileShape_MNK{}), decltype(ClusterShape_MNK{}),
                  TA, decltype(sA), decltype(sC), decltype(adma_load_a),
                  TB, decltype(sB), decltype(adma_load_b), decltype(adma_store_c),
                  TC, decltype(dC), TiledMma,
                  Alpha, Beta>(
                  prob_shape, cta_tiler, TileShape_MNK{}, ClusterShape_MNK{},
                  A_ptr, sA, sC, adma_load_a,
                  B_ptr, sB, adma_load_b, adma_store_c,
                  C_ptr, dC,
                  alpha, beta, item);
    });
  }).wait();
}

template <class TensorA, class TensorB, class TensorC,
          class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     TensorA const& A,
     TensorB const& B,
     Beta beta,
     TensorC& C,
     sycl::queue& queue)
{
  if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, B, beta, C, queue);
  }
  assert(false && "Not implemented");
}

template <class TC>
int run_gemm(int m, int n, int k, char transA, char transB)
{
  sycl::queue queue{sycl::gpu_selector_v};

  std::cout << "Running on device: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  using TA = fp16;
  using TB = fp16;
  using TI = float;

  TI alpha = TI(1.0f);
  TI beta  = TI(0.0f);

  auto prob_shape = make_shape(m, n, k);

  auto A = make_shared_usm_tensor<TA, 'R'>(queue, m, k);
  auto B = make_shared_usm_tensor<TB, 'R'>(queue, n, k);
  auto C = make_shared_usm_tensor<TC, 'R'>(queue, m, n);

  random_fill(A);
  random_fill(B);
  zero_fill(C);

  auto A_ref = make_shared_usm_tensor<TA, 'R'>(queue, m, k);
  auto B_ref = make_shared_usm_tensor<TB, 'R'>(queue, n, k);

  copy(A, A_ref);
  copy(B, B_ref);

  subbyte_pack(A);
  subbyte_pack(B);

  // Warmup
  gemm(transA, transB, m, n, k, alpha, A, B, beta, C, queue);
  queue.wait_and_throw();

  // Validation
  mem_layout layout_a = mem_layout::row_major;
  mem_layout layout_b = mem_layout::col_major;

  TA* A_ref_ptr = &*A_ref.data();
  TB* B_ref_ptr = &*B_ref.data();
  TC* C_ptr = &*C.data();

  int err_cnt = validate_gemm_result(A_ref_ptr, B_ref_ptr, C_ptr, m, n, k, layout_a, layout_b);
  printf("Verification: %s\n", (err_cnt == 0) ? "PASSED" : "FAILED");

  if (err_cnt != 0) {
    throw std::runtime_error("2D Cluster GEMM verification failed! Output does not match expected results.");
  }

  return 0;
}

int main(int argc, char** argv)
{
  int m = 512;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 512;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 512;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  char transA = 'T';
  if (argc >= 5)
    sscanf(argv[4], "%c", &transA);

  char transB = 'N';
  if (argc >= 6)
    sscanf(argv[5], "%c", &transB);

  // Output dtype: "FP32" (default) or "FP16"
  std::string out_dtype = "FP32";
  if (argc >= 7)
    out_dtype = argv[6];

  std::cout << "Output dtype: " << out_dtype << std::endl;

  if (out_dtype == "fp16") {
    return run_gemm<sycl::half>(m, n, k, transA, transB);
  } else {
    return run_gemm<float>(m, n, k, transA, transB);
  }
}
