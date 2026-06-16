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
//  CuTe-only Xe4 GEMM with row-wise softmax epilogue (FMHA-style online softmax)
// =============================================================================
//
//  Builds on examples/cute/tutorial/xe4/gemm_ldsm_warp_row_epilogue_xe4.cpp.
//  The mainloop is unchanged: 4 warps, ADMA producer + AMMA consumer, fp32
//  accumulator in SLM tile `sC`.
//
//  The epilogue performs a row-wise softmax exactly the way the FMHA forward
//  kernel does inside CollectiveSoftmaxEpilogue::update<Init=true>():
//
//      for each iter i in [0, TotalRowsPerThread):
//          (1) load this WI's per-iter slice from SLM via warp_row_C
//          (2) intra-thread max over the slice
//          (3) cross-lane fred.max / XOR-permute over the 16 row-mates → row max
//          (4) acc <- exp(acc - row_max)        (row-stable shift)
//          (5) intra-thread sum over the slice
//          (6) cross-lane sum over the 16 row-mates → row sum
//          (7) acc <- acc / row_sum
//          (8) store this WI's per-iter slice back to SLM via warp_row_D
//
//  Cross-lane reductions are done over the 16 lanes that share a row in the
//  warp-row factory's TV layout (the row-mates).  Because warp_row_CD_int
//  guarantees row-mates live in one warp, this reduces with no inter-warp
//  traffic — same invariant FMHA softmax relies on.
//
//  Output is then stored back to SLM and ADMA-pumped to GMEM.  The final D
//  matrix is row-stochastic: every row sums to 1 (within fp32 rounding).
// =============================================================================

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gpu_generics.h>

#include "cute/arch/mma_xe4_amma.hpp"
#include <cute/arch/mma_xe4.hpp>
#include <cute/atom/mma_traits_xe4_amma.hpp>
#include <cute/atom/copy_traits_xe4_adma.hpp>
#include <cute/atom/copy_traits_xe4_ldsm.hpp>
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
// Within-warp row reduction over the 16 lanes that share an `m` coordinate.
// The warp-row factory packs `wi_lo` (the row-id bit) at stride 1, so the
// 16 row-mates differ in lanes by XOR mask {2, 4, 8, 16}.  Mirrors FMHA's
// reduce_sum:
//
//     for (int i = numRowsPerSubgroup; i < NumThreadsPerWarp; i *= 2)
//         row_sum += permute_group_by_xor(sg, row_sum, i);
// -----------------------------------------------------------------------------
inline float warp_row_max(sycl::sub_group sg, float v) {
  CUTE_UNROLL
  for (unsigned mask = 2; mask < 32; mask *= 2) {
    v = sycl::fmax(v, permute_group_by_xor(sg, v, mask));
  }
  return v;
}
inline float warp_row_sum(sycl::sub_group sg, float v) {
  CUTE_UNROLL
  for (unsigned mask = 2; mask < 32; mask *= 2) {
    v += permute_group_by_xor(sg, v, mask);
  }
  return v;
}
// DEBUG variant: reduce over ALL 32 lanes (combines both row-mate groups).
// Used only to confirm hypothesis; row math is wrong but row_sum is doubled.
inline float warp_full_sum(sycl::sub_group sg, float v) {
  CUTE_UNROLL
  for (unsigned mask = 1; mask < 32; mask *= 2) {
    v += permute_group_by_xor(sg, v, mask);
  }
  return v;
}

// -----------------------------------------------------------------------------
// gemm_softmax_device — producer/consumer GEMM with row-softmax epilogue
// -----------------------------------------------------------------------------
template <class ProblemShape, class CtaTiler, class TileShape,
          class TA, class SmemLayoutA, class SmemLayoutAcc, class ADMA_A,
          class TB, class SmemLayoutB, class ADMA_B, class ADMA_C,
          class TC, class CStride, class TiledMma,
          class Alpha,
          int NumEpilogueWarps,
          // Warp-row factory knobs — chosen in main(), threaded down here.
          int LdsmEuCount, int LdsmRowsPerWi, LDSMMode LdsmPreferredMode>
void
gemm_softmax_device(ProblemShape shape_MNK, CtaTiler cta_tiler, TileShape /*tile_shape*/,
                    TA const* A, SmemLayoutA sA_layout, SmemLayoutAcc sC_layout, ADMA_A adma_load_a,
                    TB const* B, SmemLayoutB sB_layout, ADMA_B adma_load_b, ADMA_C adma_store_c,
                    TC* C, CStride /*dC*/,
                    Alpha alpha,
                    sycl::nd_item<3> item)
{
  static_assert(rank(shape_MNK) == 3);
  static_assert(rank(cta_tiler) == 3);
  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

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

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.begin()), SmemLayoutB{});
  Tensor sC = make_tensor(make_smem_ptr(smem.smem_C.begin()), SmemLayoutAcc{});

  auto [M, N, K] = shape_MNK;
  auto mA = adma_load_a .get_tma_tensor(make_shape(M, K));
  auto mB = adma_load_b .get_tma_tensor(make_shape(N, K));
  auto mC = adma_store_c.get_tma_tensor(make_shape(M, N));

  auto cta_coord = make_coord(BlockIdxX(), BlockIdxY(), _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

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
  // Mainloop — same as gemm_ldsm_warp_row_epilogue_xe4.cpp
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

  if (warp_idx == 1 && elect_one_thr) {
    int read_pipe = read_state.index();
    xe4_wait_barrier(mma_abar[read_pipe], read_state.phase());
  }
  xe4_syncthreads();

  // ===========================================================================
  // Online softmax epilogue — all NumEpilogueWarps participate.
  // Mirrors FMHA's CollectiveSoftmaxEpilogue::update<Init=true>():
  //   * partition_S/D produce a (CPY, CPY_M=TotalRowsPerThread, CPY_N=1) tensor.
  //   * Iter `i` in [0, CPY_M) covers one independent set of rows per WI.
  //   * Within an iter, all 16 row-mates (lanes that share `m`) exchange data
  //     across the warp via permute_group_by_xor for max and sum reductions.
  // ===========================================================================
  {
    constexpr int NumWarps = NumEpilogueWarps;
    // worker_id MUST be the flat 0..NumWarps*32-1 within the work-group.
    // ThreadIdxX() returns only the SIMD lane id 0..31 (mapping local_id(2)),
    // which collapses all warps into warp-0's partition and produces an LDSM
    // descriptor whose lane-id mapping no longer matches the hardware dispatch
    // The fix is to use get_local_linear_id.
    uint32_t worker_id = item.get_local_linear_id();
    sycl::sub_group sg = item.get_sub_group();

    // Warp-row factory's template knobs (NumWarps, EuCount, RowsPerWi,
    // PreferredMode) are chosen host-side and threaded through `LdsmEuCount`
    // / `LdsmRowsPerWi` / `LdsmPreferredMode` (and `NumWarps` from the
    // workgroup config).  The kernel only owns the device-side
    // descriptor-build, which is what `make_ldsm_copy_warp_row_C/D` does
    // internally via `make_ldsm_matrix_descriptor(sC, …)`.
    auto tc_load  = cute::make_ldsm_copy_warp_row_C<
        NumWarps, LdsmEuCount, LdsmRowsPerWi, LdsmPreferredMode>(sC);
    auto tc_store = cute::make_ldsm_copy_warp_row_D<
        NumWarps, LdsmEuCount, LdsmRowsPerWi, LdsmPreferredMode>(sC);

    constexpr int TileM = decltype(size<0>(SmemLayoutAcc{}))::value;
    constexpr int TileN = decltype(size<1>(SmemLayoutAcc{}))::value;
    auto coord = make_identity_tensor(make_shape(Int<TileM>{}, Int<TileN>{}));

    auto thr_load  = tc_load .get_thread_slice(worker_id);
    auto thr_store = tc_store.get_thread_slice(worker_id);

    auto tXsX  = thr_load .partition_S(coord);   // (CPY, CPY_M, CPY_N)
    auto tXrA  = thr_load .partition_fragment_D(coord);
    auto tXdst = thr_store.partition_D(coord);
    auto tXrD  = thr_store.partition_fragment_S(coord);

    constexpr int CPY     = decltype(size<0>(tXrA))::value;
    constexpr int CPY_M   = decltype(size<1>(tXrA))::value;
    constexpr int CPY_N   = decltype(size<2>(tXrA))::value;
    static_assert(CPY_N == 1, "warp-row factory keeps CPY_N == 1");

    clear(tXrA);
    copy(tc_load, tXsX, tXrA);
    sycl::group_barrier(sg);

    // Per-iter softmax loop — each iter `i` is an independent row-band.
    CUTE_UNROLL
    for (int i = 0; i < CPY_M; ++i) {
      float my_max = -INFINITY;
      CUTE_UNROLL
      for (int v = 0; v < CPY; ++v) {
        my_max = sycl::fmax(my_max, float(tXrA(v, i, _0{})));
      }
      float row_max = warp_row_max(sg, my_max);
      float my_sum = 0.f;
      CUTE_UNROLL
      for (int v = 0; v < CPY; ++v) {
        float scaled = float(alpha) * (float(tXrA(v, i, _0{})) - row_max);
        float ex = sycl::native::exp(scaled);
        tXrD(v, i, _0{}) = TC(ex);
        my_sum += ex;
      }
      float row_sum = warp_row_sum(sg, my_sum);
      float inv_sum = 1.f / row_sum;

      // (e) Normalize.
      CUTE_UNROLL
      for (int v = 0; v < CPY; ++v) {
        tXrD(v, i, _0{}) = TC(float(tXrD(v, i, _0{})) * inv_sum);
      }
    }
    sycl::group_barrier(sg);

    copy(tc_store, tXrD, tXdst);
    sycl::group_barrier(sg);
  }

  xe4_syncthreads();

  if (warp_idx == 0 && elect_one_thr) {
    xe4_set_barrier_transaction_bytes(store_c_abar[0], dma_transaction_bytesC);
    copy(adma_store_c.with(&store_c_abar[0]), tCsC, tCgC);
    xe4_wait_barrier(store_c_abar[0], store_c_barrier_phase_bit);
    store_c_barrier_phase_bit ^= 1;
  }
}

// -----------------------------------------------------------------------------
// Host launcher
// -----------------------------------------------------------------------------
template <int LdsmEuCount, int LdsmRowsPerWi, LDSMMode LdsmPreferredMode,
          class TensorA, class TensorB, class TensorC, class Alpha>
void gemm_softmax_tn(int m, int n, int k, Alpha alpha,
                     TensorA const& A, TensorB const& B, TensorC& C,
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

  auto dA = A.stride();
  auto dB = B.stride();
  auto dC = C.stride();

  Tensor mA = make_tensor(make_gmem_ptr(A_ptr), make_layout(make_shape(M,K), dA));
  Tensor mB = make_tensor(make_gmem_ptr(B_ptr), make_layout(make_shape(N,K), dB));
  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), make_layout(make_shape(M,N), dC));

  // Tile shape sized for the warp-row factory's m_tiler with NumWarps=4:
  //   m_tiler shape (RowsPerWi=2, EuCount=4, EuSgCount=1) covers 8 rows per
  //   call iteration with codomain {0,1,8,9,16,17,24,25}.  Setting BLK_M=32
  //   makes the iter axis cover the remaining 24 rows at row-stride 2 with
  //   no aliasing.  Larger TileM (e.g. 128) would re-enter the m_tiler's
  //   codomain and produce double writes — see fmha4 which uses NumWarps=16
  //   (EuSgCount=4) to fit a 128-row tile cleanly.
  using TileShape_MNK    = cute::Shape<cute::_32, cute::_128, cute::_128>;
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

  constexpr int NumEpilogueWarps    = 4;
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
      gemm_softmax_device<decltype(prob_shape), decltype(cta_tiler), decltype(TileShape_MNK{}),
                          TA, decltype(sA), decltype(sC), decltype(adma_load_a),
                          TB, decltype(sB), decltype(adma_load_b), decltype(adma_store_c),
                          TC, decltype(dC), TiledMma,
                          Alpha, NumEpilogueWarps,
                          LdsmEuCount, LdsmRowsPerWi, LdsmPreferredMode>(
                          prob_shape, cta_tiler, TileShape_MNK{},
                          A_ptr, sA, sC, adma_load_a,
                          B_ptr, sB, adma_load_b, adma_store_c,
                          C_ptr, dC,
                          alpha,
                          item);
    });
  }).wait();
}

int main(int argc, char** argv)
{
  sycl::queue queue{ sycl::gpu_selector_v };
  std::cout << "Running on device: "
            << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

  // Problem must satisfy n == BLK_N (= 128) so the row-softmax is contained
  // within a single work-group's column band.  M and K can be any multiples
  // of their tile sizes (32 and 128 respectively).  TileM is set to 32 to
  // keep the warp-row factory's m_tiler tiling clean at NumWarps=4
  // (EuSgCount=1) — see the kernel comment.
  int m = 128, n = 128, k = 256;
  if (argc >= 2) sscanf(argv[1], "%d", &m);
  if (argc >= 3) sscanf(argv[2], "%d", &n);
  if (argc >= 4) sscanf(argv[3], "%d", &k);

  if (n != 128) {
    std::cerr << "ERROR: this example requires n == BLK_N == 128 so the "
                 "row-softmax stays within one work-group's column band.\n";
    return 2;
  }
  if ((m % 32) != 0 || (k % 128) != 0) {
    std::cerr << "ERROR: m must be a multiple of 32 and k of 128.\n";
    return 2;
  }

  using TA = fp16;
  using TB = fp16;
  using TC = float;
  using TI = float;

  // The test fills A and B with values in [0, 256).  For K=128 dot products
  // reach ~K · 128 · 128 ≈ 2M.  We want exp(alpha · (acc - row_max)) to span
  // a useful dynamic range without underflowing the smallest values.  With
  // alpha = 1 / (K · 64) the per-element scaled difference between any
  // element and the row max is at most ~50 (still moderate).
  TI alpha = TI(1.0f / (64.0f * float(k)));

  // ----- Warp-row LDSM factory configuration (host-side decision) -----
  //
  // The factory's NumWarps comes from the kernel's NumEpilogueWarps; the
  // remaining knobs (EuCount, RowsPerWi, PreferredMode) are chosen here
  // and threaded down through `gemm_softmax_tn` → `gemm_softmax_device` →
  // `make_ldsm_copy_warp_row_C/D` template arguments.
  constexpr int      kLdsmEuCount        = 4;
  constexpr int      kLdsmRowsPerWi      = 2;
  constexpr LDSMMode kLdsmPreferredMode  = LDSMMode::UnorderedVector;
  std::cout << "Warp-row LDSM config: EuCount=" << kLdsmEuCount
            << " RowsPerWi=" << kLdsmRowsPerWi
            << " PreferredMode=UnorderedVector\n";

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

  gemm_softmax_tn<kLdsmEuCount, kLdsmRowsPerWi, kLdsmPreferredMode>(
      m, n, k, alpha, A, B, C, queue);
  queue.wait_and_throw();

  TA* A_ref_ptr = &*A_ref.data();
  TB* B_ref_ptr = &*B_ref.data();
  TC* C_ptr     = &*C.data();

  // ----- Validation -----
  // Use validate_gemm_result with a row-softmax post-op:
  //   gold_acc[i*N + j] = sum_k A[i,k] * B[j,k]   (TN, supplied by validator)
  //   gold[i*N + j]     = exp(alpha * gold_acc[i*N + j] - row_max[i]) / row_sum[i]
  //
  // Tolerances are looser than a plain GEMM because the softmax kernel runs
  // at fp32 register precision but with a row-max shift computed from the
  // local 16-lane row-mate reduction (matching FMHA's invariant).
  mem_layout layout_a = mem_layout::row_major;
  mem_layout layout_b = mem_layout::col_major;

  auto softmax_post_op = [=](auto&& gold_acc) {
    int M = int(m), N = int(n);
    std::vector<TC> gold(gold_acc.size());
    for (int row = 0; row < M; ++row) {
      double rmax = -std::numeric_limits<double>::infinity();
      for (int col = 0; col < N; ++col) {
        double v = double(alpha) * double(gold_acc[row * N + col]);
        if (v > rmax) rmax = v;
      }
      double rsum = 0.0;
      for (int col = 0; col < N; ++col) {
        double v = double(alpha) * double(gold_acc[row * N + col]) - rmax;
        rsum += std::exp(v);
      }
      double inv_rsum = 1.0 / rsum;
      for (int col = 0; col < N; ++col) {
        double v = double(alpha) * double(gold_acc[row * N + col]) - rmax;
        gold[row * N + col] = TC(std::exp(v) * inv_rsum);
      }
    }
    return gold;
  };

  // Slightly loosen ULP tolerance because exp() rounding may add a few ULPs;
  // abs / rel tolerances stay near fp32 precision.
  tolerance<TC> tol{};
  tol.ulp_tolerance = 64;

  // Quick row-stochasticity sanity check: every row of D should sum to ~1.
  {
    double max_dev = 0.0;
    int worst_row = 0;
    for (int row = 0; row < m; ++row) {
      double s = 0.0;
      for (int col = 0; col < n; ++col) s += double(C_ptr[row * n + col]);
      double dev = std::abs(s - 1.0);
      if (dev > max_dev) { max_dev = dev; worst_row = row; }
    }
    std::cout << "Row-sum sanity: max |row_sum - 1| = " << max_dev
              << " (worst row " << worst_row << ")\n";
  }

  int err_cnt = validate_gemm_result(A_ref_ptr, B_ref_ptr, C_ptr,
                                     uint32_t(m), uint32_t(n), uint32_t(k),
                                     layout_a, layout_b,
                                     NoOp{}, softmax_post_op,
                                     /*negative_axb=*/false, tol);
  std::cout << "Verification: " << (err_cnt == 0 ? "PASSED" : "FAILED") << std::endl;
  return err_cnt == 0 ? 0 : 1;
}
