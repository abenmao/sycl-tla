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

// =============================================================================
//  CuTe-only Xe4 GEMM with LDSM warp-row epilogue
// =============================================================================
//
//  This example mirrors the FMHA4 forward kernel's softmax epilogue pattern:
//  the per-warp MMA accumulator lives in SLM, and the **register round-trip**
//  (SLM → register → optional element-wise compute → SLM) is performed via
//  the within-warp-row LDSM factories
//
//      cute::make_ldsm_copy_warp_row_C<NumWarps>(slm_acc)   // S2R load
//      cute::make_ldsm_copy_warp_row_D<NumWarps>(slm_acc)   // R2S store
//
//  from `cute/atom/copy_traits_xe4_ldsm.hpp` — the same factories the FMHA4
//  softmax epilogue uses for sS/sP and sOacc/sO transfers.
//
//  HIGH-LEVEL STRUCTURE
//  --------------------
//    Kernel = 4 sub-groups (warps), 32 work-items each.
//      warp_idx == 0 : ADMA producer (loads A and B tiles into SLM pipes).
//      warp_idx == 1 : AMMA consumer (drives `cute::gemm` over the K loop;
//                       writes the accumulator into SLM `sC`).
//      All four warps cooperate in the LDSM warp-row epilogue:
//          sC --(make_ldsm_copy_warp_row_C)--> rAcc registers
//          rD = alpha * rAcc + bias[n]                     (FMHA-style row op)
//          rD --(make_ldsm_copy_warp_row_D)--> sC
//      Then ADMA-stores sC out to GMEM.
//
//  SHAPES (mirroring FMHA4's QK / PV tile shapes)
//    BLK_M = BLK_N = BLK_K = 128.
//    fp16 inputs, fp32 accumulator+output (matches sOacc in FMHA4).
//    With NumEpilogueWarps = 4, EuCount = 4, RowsPerWi = 2, NumThreadPerRow = 16:
//      EuSgCount         = NumWarps / EuCount           = 1
//      numRowsPerIter    = RowsPerWi · NumWarps         = 8
//      NumValPerWIPerIter = TileN / NumThreadPerRow      = 128/16 = 8
//      TotalRowsPerThread = TileM / numRowsPerIter      = 128/8  = 16
//      coop_vlen<32>      = 8                            → UnorderedVector,
//                                                         Vlen=8, Alen=1.
//    This selects exactly the same LDSM hardware path FMHA4's sOacc load uses:
//      ld_matrix.unordered.al1.cooprow.32b
//
//  WHY warp-row over MMA-aware?  The FMHA softmax loop needs row-mates within
//  one warp so within-warp `fred.max` (mask 0x55555555) can reduce a row
//  without inter-warp traffic.  Even though this GEMM does not run a softmax,
//  using the same factory lets us showcase a CuTe-only example that exercises
//  the warp-row TV layout end-to-end.
// =============================================================================

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gpu_generics.h>

#include "cute/arch/mma_xe4_amma.hpp"
#include <cute/arch/mma_xe4.hpp>
#include <cute/atom/mma_traits_xe4_amma.hpp>
#include <cute/atom/copy_traits_xe4_adma.hpp>
#include <cute/atom/copy_traits_xe4_ldsm.hpp>          // <-- LDSM warp-row factory
#include <cute/arch/xe4_util.hpp>
#include "cutlass/arch/barrier.h"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/pipeline/pipeline.hpp"

#include "../../../common/sycl_cute_common.hpp"
#include "../../../xe4/utils/validation.hpp"

using namespace cute;
using sycl::ext::oneapi::this_work_item::get_nd_item;

constexpr static size_t SmemAlignment = 256;

template <class ElementA, class ElementB, class ElementC,
          class SmemLayoutA, class SmemLayoutB, class SmemLayoutC>
struct SharedStorage : cute::aligned_struct<SmemAlignment, _0>
{
  cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>, SmemAlignment> smem_A;
  cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>, SmemAlignment> smem_B;
  cute::array_aligned<ElementC, cute::cosize_v<SmemLayoutC>, SmemAlignment> smem_C;
};

inline void xe4_syncthreads() {
  auto group = get_nd_item<1>().get_group();
  sycl::group_barrier(group);
}

// -----------------------------------------------------------------------------
// gemm_device — producer/consumer kernel
// -----------------------------------------------------------------------------
//
// warp_idx mapping:
//   warp_idx == 0 : single elected lane drives the ADMA loads of A and B.
//   warp_idx == 1 : single elected lane drives the AMMA over the K loop.
//   warp_idx == {0,1,2,3} : all four warps execute the LDSM warp-row epilogue
//                            and the final ADMA store.
//
// The mainloop is intentionally identical in shape to amma_adma_xe4.cpp so the
// only novelty is the epilogue.
template <class ProblemShape, class CtaTiler, class TileShape,
          class TA, class SmemLayoutA, class SmemLayoutAcc, class ADMA_A,
          class TB, class SmemLayoutB, class ADMA_B, class ADMA_C,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta,
          int NumEpilogueWarps>
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler, TileShape tile_shape,
            TA const* A, SmemLayoutA sA_layout, SmemLayoutAcc sC_layout, ADMA_A adma_load_a,
            TB const* B, SmemLayoutB sB_layout, ADMA_B adma_load_b, ADMA_C adma_store_c,
            TC* C, CStride dC,
            Alpha alpha, Beta /*beta — unused, kept for API symmetry*/,
            TC const* Bias_ptr,
            sycl::nd_item<3> item)
{
  static_assert(rank(shape_MNK) == 3);
  static_assert(rank(cta_tiler) == 3);
  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);
  static_assert(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));
  static_assert(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));
  static_assert(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));
  static_assert(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));

  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  auto tdesc_a = allocate_tdesc<0>();
  auto tdesc_b = allocate_tdesc<1>();
  auto tdesc_c = allocate_tdesc<2>();
  adma_load_a.set_tensor_desc(tdesc_a);
  adma_load_b.set_tensor_desc(tdesc_b);
  adma_store_c.set_tensor_desc(tdesc_c);

  using SharedStorageType = SharedStorage<TA, TB, TC, SmemLayoutA, SmemLayoutB, SmemLayoutAcc>;
  auto ptr = alloc_slm_buffer<uint8_t, sizeof(SharedStorageType)>(item.get_group());
  auto& smem = *reinterpret_cast<SharedStorageType*>(ptr);

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.begin()), SmemLayoutA{});  // (M,K,P)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.begin()), SmemLayoutB{});  // (N,K,P)
  Tensor sC = make_tensor(make_smem_ptr(smem.smem_C.begin()), SmemLayoutAcc{});// (M,N)

  auto [M, N, K] = shape_MNK;
  auto mA = adma_load_a .get_tma_tensor(make_shape(M, K));
  auto mB = adma_load_b .get_tma_tensor(make_shape(N, K));
  auto mC = adma_store_c.get_tma_tensor(make_shape(M, N));
  Tensor mBias = make_tensor(make_gmem_ptr(Bias_ptr),
                             make_shape(M, N), make_stride(_0{}, _1{}));

  auto cta_coord = make_coord(BlockIdxX(), BlockIdxY(), _);
  Tensor gA    = local_tile(mA,    cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB    = local_tile(mB,    cta_tiler, cta_coord, Step< X, _1, _1>{});
  Tensor gC    = local_tile(mC,    cta_tiler, cta_coord, Step<_1, _1, X>{});
  Tensor gBias = local_tile(mBias, cta_tiler, cta_coord, Step<_1, _1, X>{});

  auto [tAgA, tAsA] = tma_partition(adma_load_a, group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto [tBgB, tBsB] = tma_partition(adma_load_b, group_modes<0,2>(sB), group_modes<0,2>(gB));

  auto K_PIPE_MAX = size<1>(tAsA);
  int  K_TILE_MAX = size<1>(tAgA);
  int  k_tile     = 0;

  constexpr int dma_transaction_bytesA = (cosize(sA_layout) * sizeof(TA)) / K_PIPE_MAX;
  constexpr int dma_transaction_bytesB = (cosize(sB_layout) * sizeof(TB)) / K_PIPE_MAX;
  constexpr int dma_transaction_bytesC = (cosize(sC_layout) * sizeof(TC));

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t warp_idx      = get_sg_id();
  uint32_t lane_id       = item.get_local_id(2);

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

  int store_c_barrier_phase_bit = 0;

  // -------- MMA partitioning (only consumer warp uses it) --------
  TiledMma mma{};
  ThrMMA thr_mma = mma.get_thread_slice(ThreadIdxX());
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCsC = thr_mma.partition_C(sC);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCsC);

  clear(tCsC);
  xe4_syncthreads();

  uint64_t mma_ctrl = 0x100;

  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();

  // ===========================================================================
  // Mainloop — producer (warp 0) and AMMA consumer (warp 1).
  // ===========================================================================
  if (warp_idx == 0) {
    if (elect_one_thr) {
      for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
        xe4_set_barrier_transaction_bytes(load_a_abar[pipe], dma_transaction_bytesA);
        xe4_set_barrier_transaction_bytes(load_b_abar[pipe], dma_transaction_bytesB);
        copy(adma_load_a.with(&load_a_abar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
        copy(adma_load_b.with(&load_b_abar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
        ++k_tile;
      }
      for (int k_tile_next = k_tile; k_tile_next < K_TILE_MAX; ++k_tile_next) {
        int write_pipe = write_state.index();
        xe4_wait_barrier(mma_abar[write_pipe], write_state.phase());
        xe4_set_barrier_transaction_bytes(load_a_abar[write_pipe], dma_transaction_bytesA);
        xe4_set_barrier_transaction_bytes(load_b_abar[write_pipe], dma_transaction_bytesB);
        copy(adma_load_a.with(&load_a_abar[write_pipe]), tAgA(_,k_tile_next), tAsA(_,write_pipe));
        copy(adma_load_b.with(&load_b_abar[write_pipe]), tBgB(_,k_tile_next), tBsB(_,write_pipe));
        ++write_state;
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }
  else if (warp_idx == 1) {
    if (elect_one_thr) {
      for (int k_tile_next = 0; k_tile_next < K_TILE_MAX - 1; ++k_tile_next) {
        int read_pipe = read_state.index();
        xe4_wait_barrier(load_a_abar[read_pipe], read_state.phase());
        xe4_wait_barrier(load_b_abar[read_pipe], read_state.phase());
        xe4_set_barrier_transaction_bytes(mma_abar[read_pipe], 2);
        auto new_mma = mma.with(AMMA::TrackMethod<AMMA::Tracking::AB>{}, mma_ctrl,
                                 &mma_abar[read_pipe], &mma_abar[read_pipe], 0u, 0u);
        cute::gemm(new_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
        mma_ctrl = 0x000;
        ++read_state;
      }
      // Last K-tile: track D so the consumer can wait on it before the epilogue.
      {
        int read_pipe = read_state.index();
        xe4_wait_barrier(load_a_abar[read_pipe], read_state.phase());
        xe4_wait_barrier(load_b_abar[read_pipe], read_state.phase());
        xe4_set_barrier_transaction_bytes(mma_abar[read_pipe], 1);
        auto new_mma = mma.with(AMMA::TrackMethod<AMMA::Tracking::D>{}, mma_ctrl,
                                 &mma_abar[read_pipe]);
        cute::gemm(new_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }

  xe4_syncthreads();

  // ===========================================================================
  // LDSM warp-row epilogue — all NumEpilogueWarps participate.
  // ---------------------------------------------------------------------------
  // 1. Warp 1's elected lane waits on the AMMA-D barrier so the SLM accumulator
  //    `sC` is fully written before any warp loads it.
  // 2. All warps build a TiledCopy via make_ldsm_copy_warp_row_C, partition
  //    sC, and bulk-load (CPY, CPY_M, CPY_N) into a register fragment rAcc.
  // 3. Apply alpha * rAcc + bias[n] (broadcast a row bias across M like the
  //    FMHA softmax `final_rescale_O` cast).
  // 4. Bulk-store rD back to sC via make_ldsm_copy_warp_row_D.
  // 5. Warp 0 issues the ADMA store sC → GMEM.
  // ===========================================================================
  if (warp_idx == 1 && elect_one_thr) {
    int read_pipe = read_state.index();
    xe4_wait_barrier(mma_abar[read_pipe], read_state.phase());
  }
  xe4_syncthreads();

  {
    constexpr int NumWarps = NumEpilogueWarps;
    int worker_id = ThreadIdxX();  // 0..NumWarps*32-1

    auto tc_load  = cute::make_ldsm_copy_warp_row_C<NumWarps>(sC);
    auto tc_store = cute::make_ldsm_copy_warp_row_D<NumWarps>(sC);

    constexpr int TileM = decltype(size<0>(SmemLayoutAcc{}))::value;
    constexpr int TileN = decltype(size<1>(SmemLayoutAcc{}))::value;
    auto coord = make_identity_tensor(make_shape(Int<TileM>{}, Int<TileN>{}));

    auto thr_load  = tc_load .get_thread_slice(worker_id);
    auto thr_store = tc_store.get_thread_slice(worker_id);

    auto tXsX     = thr_load .partition_S(coord);
    auto tXrAcc   = thr_load .partition_fragment_D(coord);
    auto thr_dst  = thr_store.partition_D(coord);
    auto thr_src  = thr_store.partition_fragment_S(coord);

    clear(tXrAcc);
    copy(tc_load, tXsX, tXrAcc);
    sycl::group_barrier(item.get_sub_group());

    // Optional row-wise epilogue: rD = alpha * rAcc + bias[n].
    // The `bias` tensor has stride (0, 1) so every row sees the same bias[n].
    // We compute (m, n) per element by reading the coord tensor partitioned
    // identically; rAcc / coord_view share the (CPY, CPY_M, CPY_N) shape.
    auto coord_view = thr_load.partition_S(coord);  // (CPY, CPY_M, CPY_N)
    static_assert(size(coord_view) == size(tXrAcc),
                  "Coord view and accumulator fragment must have the same size");

    CUTE_UNROLL
    for (int idx = 0; idx < size(tXrAcc); ++idx) {
      auto crd = coord_view(idx);   // (m, n)
      int m_idx = get<0>(crd);
      int n_idx = get<1>(crd);
      TC bias_v = (Bias_ptr != nullptr) ? gBias(m_idx, n_idx) : TC(0);
      tXrAcc(idx) = TC(alpha) * tXrAcc(idx) + bias_v;
    }

    // Register-to-register bridge load → store fragment.
    copy(tXrAcc, thr_src);
    sycl::group_barrier(item.get_sub_group());

    copy(tc_store, thr_src, thr_dst);
    sycl::group_barrier(item.get_sub_group());
  }

  xe4_syncthreads();

  // ADMA store sC → GMEM (single elected lane in warp 0).
  // tCsC = thr_mma.partition_C(sC) is just a view of sC under the MMA's
  // C-fragment shape — it reflects the values the LDSM warp-row epilogue
  // wrote back to sC.
  if (warp_idx == 0 && elect_one_thr) {
    xe4_set_barrier_transaction_bytes(store_c_abar[0], dma_transaction_bytesC);
    copy(adma_store_c.with(&store_c_abar[0]), tCsC, tCgC);
    xe4_wait_barrier(store_c_abar[0], store_c_barrier_phase_bit);
    store_c_barrier_phase_bit ^= 1;
  }
}

// -----------------------------------------------------------------------------
// Host-side launcher
// -----------------------------------------------------------------------------
template <class TensorA, class TensorB, class TensorC, class TensorBias,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TensorA const& A,
        TensorB const& B,
        Beta beta,
        TensorC& C,
        TensorBias const& Bias,
        sycl::queue& queue)
{
  auto M = int(m), N = int(n), K = int(k);
  auto prob_shape = make_shape(M, N, K);

  using TA = typename TensorA::element_type;
  using TB = typename TensorB::element_type;
  using TC = typename TensorC::element_type;

  TA const* A_ptr = &*A.data();
  TB const* B_ptr = &*B.data();
  TC* C_ptr       = &*C.data();
  TC const* Bias_ptr = &*Bias.data();

  auto dA = A.stride();
  auto dB = B.stride();
  auto dC = C.stride();

  Tensor mA = make_tensor(make_gmem_ptr(A_ptr), make_layout(make_shape(M,K), dA));
  Tensor mB = make_tensor(make_gmem_ptr(B_ptr), make_layout(make_shape(N,K), dB));
  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), make_layout(make_shape(M,N), dC));

  using TileShape_MNK    = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;
  constexpr auto bM = get<0>(TileShape_MNK{});
  constexpr auto bN = get<1>(TileShape_MNK{});
  constexpr auto bK = get<2>(TileShape_MNK{});
  auto cta_tiler = make_shape(bM, bN, bK);
  constexpr auto bP = Int<3>{};

  static constexpr auto majorA = cute::AMMA::Major::K;
  static constexpr auto majorB = cute::AMMA::Major::K;

  using ElementA = TA;
  using ElementB = TB;
  using ElementAccumulator = float;
  using ElementC = float;

  using TiledMma = decltype(cute::make_tiled_mma(
    cute::AMMA::ss_op_selector<
      ElementAccumulator, ElementA, ElementB, ElementC,
      decltype(cute::product_each(TileShape_MNK{})),
      ClusterShape_MNK, majorA, majorB>()
  ));
  TiledMma tiled_mma{};

  using SmemLayoutAtomA = decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenRowMajor{}));
  using SmemLayoutAtomB = decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenRowMajor{}));
  using SmemLayoutAtomC = decltype(make_layout(cute::select<0, 1>(TileShape_MNK{}), GenRowMajor{}));

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
      Step<_2,_1,_3>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
      Step<_2,_1,_3>{}));
  using SmemLayoutAcc = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
      Step<_2,_1>{}));

  SmemLayoutA  sA{};
  SmemLayoutB  sB{};
  SmemLayoutAcc sC{};

  using GmemTiledCopyA = cute::XE4_ADMA_LOAD;
  using GmemTiledCopyB = cute::XE4_ADMA_LOAD;
  using GmemTiledCopyC = cute::XE4_ADMA_STORE;

  auto cluster_layout_vmnk = tiled_divide(
      make_layout(ClusterShape_MNK{}), make_tile(typename TiledMma::AtomThrID{}));

  auto adma_load_a = make_adma_atom_A_xe4(
      GmemTiledCopyA{}, mA, SmemLayoutA{}(_,_,cute::Int<0>{}),
      TileShape_MNK{}, TiledMma{}, cluster_layout_vmnk);
  auto adma_load_b = make_adma_atom_B_xe4(
      GmemTiledCopyB{}, mB, SmemLayoutB{}(_,_,cute::Int<0>{}),
      TileShape_MNK{}, TiledMma{}, cluster_layout_vmnk);

  auto cta_tiler_mn = make_shape(bM, bN);
  auto adma_store_c = make_adma_copy<TC>(GmemTiledCopyC{}, mC, SmemLayoutAcc{},
                                          cta_tiler_mn, Int<1>{});

  // 4 warps total: warp 0 = ADMA producer, warp 1 = AMMA consumer, all 4 = epilogue.
  constexpr int NumEpilogueWarps  = 4;
  constexpr int NumOfThreadsPerWarp = 32;

  namespace syclexp = sycl::ext::oneapi::experimental;
  auto [cluster_size_y, cluster_size_x, cluster_size_z] = ClusterShape_MNK{};
  sycl::range<3> clusterSize(cluster_size_z, cluster_size_y, cluster_size_x);
  syclexp::properties Props{ syclexp::work_groups_per_cluster<3>(clusterSize) };

  auto num_groups = ceil_div(prob_shape, TileShape_MNK{});
  sycl::range<3> local_range(1, NumEpilogueWarps, NumOfThreadsPerWarp);
  sycl::range<3> group_range(1, get<1>(num_groups), get<0>(num_groups));
  sycl::nd_range<3> Range(group_range * local_range, local_range);

  auto launch_cfg = syclexp::launch_config(Range, Props);
  syclexp::submit_with_event(queue, [&](sycl::handler& handler) {
    syclexp::nd_launch(handler, launch_cfg, [=](sycl::nd_item<3> item) ALWAYS_INLINE {
      gemm_device<decltype(prob_shape), decltype(cta_tiler), decltype(TileShape_MNK{}),
                  TA, decltype(sA), decltype(sC), decltype(adma_load_a),
                  TB, decltype(sB), decltype(adma_load_b), decltype(adma_store_c),
                  TC, decltype(dC), TiledMma,
                  Alpha, Beta, NumEpilogueWarps>(
                  prob_shape, cta_tiler, TileShape_MNK{},
                  A_ptr, sA, sC, adma_load_a,
                  B_ptr, sB, adma_load_b, adma_store_c,
                  C_ptr, dC,
                  alpha, beta,
                  Bias_ptr,
                  item);
    });
  }).wait();
}

int main(int argc, char** argv)
{
  sycl::queue queue{ sycl::gpu_selector_v };
  std::cout << "Running on device: "
            << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

  int m = 256, n = 256, k = 256;
  if (argc >= 2) sscanf(argv[1], "%d", &m);
  if (argc >= 3) sscanf(argv[2], "%d", &n);
  if (argc >= 4) sscanf(argv[3], "%d", &k);

  using TA = fp16;
  using TB = fp16;
  using TC = float;
  using TI = float;

  TI alpha = TI(1.0f);
  TI beta  = TI(0.0f);

  auto A = make_shared_usm_tensor<TA, 'R'>(queue, m, k);
  auto B = make_shared_usm_tensor<TB, 'R'>(queue, n, k);
  auto C = make_shared_usm_tensor<TC, 'R'>(queue, m, n);

  // Per-column bias broadcast across rows.  Use stride (0, 1) so reading
  // gBias(m, n) yields bias[n] for every m.
  auto Bias = make_shared_usm_tensor<TC, 'R'>(queue, 1, n);

  random_fill(A);
  random_fill(B);
  zero_fill(C);
  random_fill(Bias);

  // Snapshot A and B for host validation before any in-place modifications.
  auto A_ref = make_shared_usm_tensor<TA, 'R'>(queue, m, k);
  auto B_ref = make_shared_usm_tensor<TB, 'R'>(queue, n, k);
  copy(A, A_ref);
  copy(B, B_ref);

  // The kernel expects sub-byte-packed inputs; subbyte_pack is a no-op for fp16.
  subbyte_pack(A);
  subbyte_pack(B);

  gemm_tn(m, n, k, alpha, A, B, beta, C, Bias, queue);
  queue.wait_and_throw();

  TA* A_ref_ptr = &*A_ref.data();
  TB* B_ref_ptr = &*B_ref.data();
  TC* C_ptr     = &*C.data();
  TC* Bias_ptr  = &*Bias.data();

  // Validate via examples/xe4/utils/validation.hpp's validate_gemm_result.
  //
  // - layout_a = row_major: A is (M, K) row-major.
  // - layout_b = col_major: validate_gemm_result expects B in (K, N) form;
  //   our B is (N, K) row-major which is the same memory as (K, N) col-major.
  // - Epilogue lambda: D = alpha * gold_acc + bias[n] (bias broadcast across M).
  //
  // The lambda receives `gold_acc` indexed in row-major (M, N) order, so the
  // column index is `i % N`.
  mem_layout layout_a = mem_layout::row_major;
  mem_layout layout_b = mem_layout::col_major;

  auto bias_post_op = [=](auto&& gold_acc) {
    std::vector<TC> result(gold_acc.size());
    for (size_t i = 0; i < gold_acc.size(); ++i) {
      int col = int(i % size_t(n));
      result[i] = TC(float(alpha) * float(gold_acc[i]) + float(Bias_ptr[col]));
    }
    return result;
  };

  int err_cnt = validate_gemm_result(A_ref_ptr, B_ref_ptr, C_ptr,
                                     uint32_t(m), uint32_t(n), uint32_t(k),
                                     layout_a, layout_b,
                                     NoOp{}, bias_post_op);
  std::cout << "Verification: " << (err_cnt == 0 ? "PASSED" : "FAILED") << std::endl;
  return err_cnt == 0 ? 0 : 1;
}
