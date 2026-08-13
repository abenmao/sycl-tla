/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Double-buffer MoE kernels (uniform-M): the operand-tensor helper
           make_moe_tensor(), the tile-scheduler type aliases, and BOTH kernels --
           MoEGEMMDoubleBuffer (plain bf16) and MoEGEMMDoubleBufferScaled
           (fp8/mxfp8/mxfp4). The greedy path lives in xe_moe_gemm_greedy.hpp.
*/
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/platform/platform.h"
#include "moe_grouped_gemm/kernel/xe_moe_tile_scheduler.hpp"
#include "moe_grouped_gemm/moe_scale_layout.hpp"
#include <cute/util/compat.hpp>
#include "cutlass/gemm/collective/xe_mma_blockscaled_scale_traits.hpp"  // scaled DB kernel

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

using ProblemShapeMNKL = Shape<int, int, int, int>;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using TileScheduler = typename MoE::PersistentTileSchedulerXeMoE<ProblemShape>;
using RasterOrderOptions = typename TileScheduler::RasterOrderOptions;

// Build a 2D gmem tensor (r, c) using the CUTLASS *operand* layout convention
// (same as TagToStride / examples 50/51), not textbook row/col-major. B is
// passed (N,K) but is the K×N operand, so it inverts the A/D stride mapping:
//   A/D: RowMajor -> stride (c,1);  ColumnMajor -> stride (1,r)
//   B  : RowMajor -> (1,r) N-contig (ldb=N);  ColumnMajor -> (c,1) K-contig (ldb=K)
template <typename T, class Layout, bool IsBOperand = false>
CUTE_DEVICE auto make_moe_tensor(T *ptr, int r, int c) {
  auto shape = make_shape(r, c);
  constexpr bool is_row = cute::is_same_v<Layout, cutlass::layout::RowMajor>;
  // Is mode r the unit-stride dim? A/D: only ColumnMajor; B: only RowMajor.
  constexpr bool unit_first = IsBOperand ? is_row : !is_row;
  if constexpr (cute::is_subbyte_v<T>) {
    // Sub-byte types need const pointers to avoid subbyte_iterator issues with
    // 2D block loads.
    auto const_ptr = const_cast<T const*>(ptr);
    if constexpr (unit_first)
      return make_tensor(make_gmem_ptr(const_ptr),
                         make_layout(shape, make_stride(_1{}, r)));
    else
      return make_tensor(make_gmem_ptr(const_ptr),
                         make_layout(shape, make_stride(c, _1{})));
  } else {
    if constexpr (unit_first)
      return make_tensor(make_gmem_ptr<T>(ptr),
                         make_layout(shape, make_stride(_1{}, r)));
    else
      return make_tensor(make_gmem_ptr<T>(ptr),
                         make_layout(shape, make_stride(c, _1{})));
  }
}

///////////////////////////////////////////////////////////////////////////////
// Plain (bf16) double-buffer MoE kernel.
///////////////////////////////////////////////////////////////////////////////

using ProblemShapeMNKL = Shape<int, int, int, int>;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          class LayoutKindA, class LayoutKindB, class LayoutKindD, class TiledMMA,
          typename ElementA, typename ElementB, typename ElementS,
          typename ElementD>
CUTE_DEVICE void
MoEGEMMDoubleBuffer(const ElementA *Activations, const ElementB *Weights,
        const ElementS *Scales, ElementD *Outputs, TiledMMA const &mma,
        const int32_t uniform_M, const int32_t num_experts, const int32_t N,
        const int32_t K,
        PersistentTileSchedulerSm90GroupParams<ProblemShape> scheduler_params) {

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto local_id = item.get_local_linear_id();
  int sg_id = (int)item.get_sub_group().get_group_id()[0];

  auto wg_tile = mma.tile_mnk();
  int32_t tile_m = get<0>(wg_tile);
  int32_t tile_n = get<1>(wg_tile);

  int32_t m_tiles_in_expert = (uniform_M + tile_m - 1) / tile_m;
  int32_t n_tiles           = (N + tile_n - 1) / tile_n;
  int64_t mn_per_expert     = int64_t(m_tiles_in_expert) * n_tiles;
  int64_t total_work        = int64_t(num_experts) * mn_per_expert;

  uint64_t wg_id;
  if (scheduler_params.raster_order_ ==
      PersistentTileSchedulerSm90GroupParams<ProblemShape>::RasterOrder::AlongN) {
    wg_id = uint64_t(BlockIdxX()) + uint64_t(BlockIdxY()) * uint64_t(GridDimX());
  } else {
    wg_id = uint64_t(BlockIdxX()) * uint64_t(GridDimY()) + uint64_t(BlockIdxY());
  }
  int64_t grid_size = int64_t(GridDimX()) * int64_t(GridDimY()) * int64_t(GridDimZ());

  // Both accumulator banks live at outermost scope so the compiler keeps them
  // across the entire persistent loop.
  auto D_tensor_init = make_moe_tensor<ElementD, LayoutKindD>(
      Outputs, uniform_M, N);

  Tensor cD_init = make_identity_tensor(D_tensor_init.shape());

  Tensor gD_even_init = local_tile(cD_init, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});
  Tensor gD_odd_init  = local_tile(cD_init, wg_tile, make_coord(0, 1, 0), Step<_1, _1, X>{});

  auto thr_mma = mma.get_slice(local_id);

  auto tCrD_even = thr_mma.partition_sg_fragment_C(gD_even_init);
  auto tCrD_odd  = thr_mma.partition_sg_fragment_C(gD_odd_init);
  // cute::gemm accumulates over every K-tile; in-loop clears only run after a
  // store, so the first accumulation needs these fragments cleared up front.
  clear(tCrD_even);
  clear(tCrD_odd);

  bool is_first_wave  = true;
  ElementD *ptr_D_prev    = Outputs;
  int32_t   wg_m_prev     = 0;
  int32_t   wg_n_odd_prev = 1;

  constexpr int prefetch_dist = 2;
  int k_tile_count = ceil_div(K, get<2>(wg_tile));
  constexpr uint8_t NUM_SUBGROUPS = 32;
  uint8_t k_tile_subgroup_store_ratio = k_tile_count >= NUM_SUBGROUPS ? 1 : ceil_div(NUM_SUBGROUPS, k_tile_count);

  // ── Strided pair loop ────────────────────────────────────────────────────
  // Each WG owns pair slots wg_id, wg_id+grid_size, ...; each slot covers two
  // consecutive flat MN indices (even=idx, odd=idx+1). Expert setup rebuilt only
  // on expert change.
  int32_t cur_expert_id = -1;
  ElementD *ptr_D = Outputs;

  auto A_tensor = make_moe_tensor<ElementA, LayoutKindA>(
      const_cast<ElementA *>(Activations), uniform_M, K);
  auto B_tensor = make_moe_tensor<ElementB, LayoutKindB, /*IsBOperand=*/true>(
      const_cast<ElementB *>(Weights), N, K);
  auto D_tensor = make_moe_tensor<ElementD, LayoutKindD>(Outputs, uniform_M, N);

  Tensor cA = make_identity_tensor(A_tensor.shape());
  Tensor cB = make_identity_tensor(B_tensor.shape());
  Tensor cD = make_identity_tensor(D_tensor.shape());

  auto tiled_copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A_tensor);
  auto tiled_copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B_tensor);
  auto tiled_copy_d = get_block_2d_copy_D<GmemTiledCopyD>(mma, D_tensor);

  auto thr_copy_a = tiled_copy_a.get_slice(local_id);
  auto thr_copy_b = tiled_copy_b.get_slice(local_id);
  auto thr_copy_d = tiled_copy_d.get_slice(local_id);

  auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
  auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);
  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  int32_t wg_m_cur = -1;
  auto gA_cur   = local_tile(cA, select<0, 2>(wg_tile), make_coord(0, _));
  auto tArA_cur = thr_copy_a.partition_sg_fragment_D(gA_cur(_, _, 0));
  auto tCrA_cur = thr_mma.partition_sg_fragment_A(gA_cur(_, _, 0));
  auto tAgA_cur = thr_copy_a.partition_S(gA_cur);
  auto pAgA_cur = thr_prefetch_A.partition_S(gA_cur);

  auto rebuild_A = [&](int32_t wg_m) {
    auto gA   = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
    tArA_cur  = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    tCrA_cur  = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    tAgA_cur  = thr_copy_a.partition_S(gA);
    pAgA_cur  = thr_prefetch_A.partition_S(gA);
    wg_m_cur  = wg_m;
  };

  for (int64_t idx = int64_t(wg_id) * 2; idx < total_work; idx += grid_size * 2) {

    int32_t expert_id   = int32_t(idx / mn_per_expert);
    int64_t mn_even     = idx % mn_per_expert;
    int32_t wg_m_even   = int32_t(mn_even / n_tiles);
    int32_t wg_n_even   = int32_t(mn_even % n_tiles);

    int64_t idx_odd   = idx + 1;
    int64_t mn_odd    = idx_odd % mn_per_expert;
    int32_t wg_m_odd  = int32_t(mn_odd / n_tiles);
    int32_t wg_n_odd  = int32_t(mn_odd % n_tiles);

    // Rebuild expert-level state only on expert change.
    if (expert_id != cur_expert_id) {
      int64_t cumulative_M = int64_t(expert_id) * uniform_M;

      ElementA *ptr_A = const_cast<ElementA *>(Activations) + cumulative_M * K;
      ElementB *ptr_B = const_cast<ElementB *>(Weights) + int64_t(expert_id) * K * N;
      ptr_D = Outputs + cumulative_M * N;

      A_tensor = make_moe_tensor<ElementA, LayoutKindA>(ptr_A, uniform_M, K);
      B_tensor = make_moe_tensor<ElementB, LayoutKindB, /*IsBOperand=*/true>(ptr_B, N, K);
      D_tensor = make_moe_tensor<ElementD, LayoutKindD>(ptr_D, uniform_M, N);

      cA = make_identity_tensor(A_tensor.shape());
      cB = make_identity_tensor(B_tensor.shape());
      cD = make_identity_tensor(D_tensor.shape());

      tiled_copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A_tensor);
      tiled_copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B_tensor);
      tiled_copy_d = get_block_2d_copy_D<GmemTiledCopyD>(mma, D_tensor);

      thr_copy_a = tiled_copy_a.get_slice(local_id);
      thr_copy_b = tiled_copy_b.get_slice(local_id);
      thr_copy_d = tiled_copy_d.get_slice(local_id);

      prefetch_a = make_block_2d_prefetch(tiled_copy_a);
      prefetch_b = make_block_2d_prefetch(tiled_copy_b);
      thr_prefetch_A = prefetch_a.get_slice(local_id);
      thr_prefetch_B = prefetch_b.get_slice(local_id);

      wg_m_cur    = -1;
      cur_expert_id = expert_id;
    }

    if (wg_m_even != wg_m_cur) rebuild_A(wg_m_even);
    auto tArA_even = tArA_cur;
    auto tCrA_even = tCrA_cur;
    auto tAgA_even = tAgA_cur;
    auto pAgA_even = pAgA_cur;

    Tensor gB_even   = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n_even, _));
    Tensor gD_even   = local_tile(cD, wg_tile, make_coord(wg_m_even, wg_n_even, 0), Step<_1, _1, X>{});
    auto tBrB_even   = thr_copy_b.partition_sg_fragment_D(gB_even(_, _, 0));
    auto tCrB_even   = thr_mma.partition_sg_fragment_B(gB_even(_, _, 0));
    Tensor tBgB_even = thr_copy_b.partition_S(gB_even);
    auto tCgD_even   = thr_copy_d.partition_D(gD_even);
    auto pBgB_even   = thr_prefetch_B.partition_S(gB_even);

    // Fire even tile prefetch warm-up here so odd tile setup provides latency hiding.
    int prefetch_k = 0;
    CUTE_UNROLL
    for (; prefetch_k < prefetch_dist; ++prefetch_k) {
      prefetch(prefetch_a, pAgA_even(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB_even(_, _, _, prefetch_k));
    }

    if (wg_m_odd != wg_m_even) rebuild_A(wg_m_odd);
    auto tArA_odd = tArA_cur;
    auto tCrA_odd = tCrA_cur;
    auto tAgA_odd = tAgA_cur;
    auto pAgA_odd = pAgA_cur;

    Tensor gB_odd   = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n_odd, _));
    Tensor gD_odd   = local_tile(cD, wg_tile, make_coord(wg_m_odd, wg_n_odd, 0), Step<_1, _1, X>{});
    auto tBrB_odd   = thr_copy_b.partition_sg_fragment_D(gB_odd(_, _, 0));
    auto tCrB_odd   = thr_mma.partition_sg_fragment_B(gB_odd(_, _, 0));
    Tensor tBgB_odd = thr_copy_b.partition_S(gB_odd);
    auto tCgD_odd   = thr_copy_d.partition_D(gD_odd);
    auto pBgB_odd   = thr_prefetch_B.partition_S(gB_odd);

    // ── Even k-loop: gemm even tile, store previous odd result (deferred) ──
    int prefetch_k_odd = 0;
    for (int k = 0; k < k_tile_count; ++k, ++prefetch_k) {
      if (!is_first_wave && sg_id / k_tile_subgroup_store_ratio == k) {
        auto D_tensor_prev     = make_moe_tensor<ElementD, LayoutKindD>(ptr_D_prev, uniform_M, N);
        auto cD_prev           = make_identity_tensor(D_tensor_prev.shape());
        auto tiled_copy_d_prev = get_block_2d_copy_D<GmemTiledCopyD>(mma, D_tensor_prev);
        auto thr_copy_d_prev   = tiled_copy_d_prev.get_slice(local_id);
        auto gD_odd_prev       = local_tile(cD_prev, wg_tile, make_coord(wg_m_prev, wg_n_odd_prev, 0), Step<_1, _1, X>{});
        auto tCrD_final_odd    = thr_copy_d_prev.partition_sg_fragment_S(gD_odd_prev);
        auto tCgD_odd_prev     = thr_copy_d_prev.partition_D(gD_odd_prev);
        reorder(tCrD_odd, tCrD_final_odd);
        copy(tiled_copy_d_prev, tCrD_final_odd, tCgD_odd_prev);
        clear(tCrD_odd);
      }
      copy(tiled_copy_a, tAgA_even(_, _, _, k), tArA_even);
      copy(tiled_copy_b, tBgB_even(_, _, _, k), tBrB_even);
      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA_even(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB_even(_, _, _, prefetch_k));
      } else {
        prefetch(prefetch_a, pAgA_odd(_, _, _, prefetch_k_odd));
        prefetch(prefetch_b, pBgB_odd(_, _, _, prefetch_k_odd));
        ++prefetch_k_odd;
      }
      reorder(tArA_even, tCrA_even);
      reorder(tBrB_even, tCrB_even);
      cute::gemm(mma, tCrA_even, tCrB_even, tCrD_even);
    }

    // ── Odd k-loop: gemm odd tile, store even result (deferred) ──────────
    for (int k = 0; k < k_tile_count; ++k, ++prefetch_k_odd) {
      if (sg_id / k_tile_subgroup_store_ratio == k) {
        auto tCrD_final_even = thr_copy_d.partition_sg_fragment_S(gD_even);
        reorder(tCrD_even, tCrD_final_even);
        copy(tiled_copy_d, tCrD_final_even, tCgD_even);
        clear(tCrD_even);
      }
      copy(tiled_copy_a, tAgA_odd(_, _, _, k), tArA_odd);
      copy(tiled_copy_b, tBgB_odd(_, _, _, k), tBrB_odd);
      if (prefetch_k_odd < k_tile_count) {
        prefetch(prefetch_a, pAgA_odd(_, _, _, prefetch_k_odd));
        prefetch(prefetch_b, pBgB_odd(_, _, _, prefetch_k_odd));
      }
      reorder(tArA_odd, tCrA_odd);
      reorder(tBrB_odd, tCrB_odd);
      cute::gemm(mma, tCrA_odd, tCrB_odd, tCrD_odd);
    }

    ptr_D_prev    = ptr_D;
    wg_m_prev     = wg_m_odd;
    wg_n_odd_prev = wg_n_odd;
    is_first_wave = false;
  } // strided pair loop

  // Flush the final odd result.
  if (!is_first_wave) {
    auto D_tensor_last     = make_moe_tensor<ElementD, LayoutKindD>(ptr_D_prev, uniform_M, N);
    auto cD_last           = make_identity_tensor(D_tensor_last.shape());
    auto tiled_copy_d_last = get_block_2d_copy_D<GmemTiledCopyD>(mma, D_tensor_last);
    auto thr_copy_d_last   = tiled_copy_d_last.get_slice(local_id);
    auto gD_odd_last       = local_tile(cD_last, wg_tile, make_coord(wg_m_prev, wg_n_odd_prev, 0), Step<_1, _1, X>{});
    auto tCrD_final_odd    = thr_copy_d_last.partition_sg_fragment_S(gD_odd_last);
    auto tCgD_odd_last     = thr_copy_d_last.partition_D(gD_odd_last);
    reorder(tCrD_odd, tCrD_final_odd);
    copy(tiled_copy_d_last, tCrD_final_odd, tCgD_odd_last);
  }
}


///////////////////////////////////////////////////////////////////////////////
// Block-scaled (fp8/mxfp8/mxfp4) double-buffer MoE kernel.
///////////////////////////////////////////////////////////////////////////////



// Double-buffer uniform-M kernel for scaled (low-precision) dtypes. Mirrors
// MoEGEMMDoubleBuffer but replaces the plain gemm k-loop bodies with scaled gemm
// (TENSOR: CfgGroupK==0, or BLOCK/MX: CfgGroupK>0).
//
// Scale storage conventions match MoEGEMM / moe_gemm_scaled:
//   TENSOR: ScalesA contiguous (total_M, scale_k) stride (_1{}, total_M);
//           ScalesB per-expert (scale_n, scale_k) stride (scale_k, _1{})
//   BLOCK : ScalesA per-expert padded (M, scale_k, 1) stride (_1{}, round_up_M, ..);
//           ScalesB per-expert padded (scale_n, scale_k, 1) stride (_1{}, padded_scale_n, ..)
template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          class LayoutKindA, class LayoutKindB, class LayoutKindD,
          int CfgGroupN, int CfgGroupK,
          class TiledMMA,
          typename ElementA, typename ElementB, typename ElementS,
          typename ElementD>
CUTE_DEVICE void
MoEGEMMDoubleBufferScaled(const ElementA *Activations, const ElementB *Weights,
                     const ElementS *ScalesA, const ElementS *ScalesB,
                     ElementD *Outputs, TiledMMA const &mma,
                     const int32_t uniform_M, const int32_t num_experts,
                     const int32_t N, const int32_t K,
                     const int32_t GroupN, const int32_t GroupK,
                     PersistentTileSchedulerSm90GroupParams<ProblemShape> scheduler_params) {

  static_assert(!cute::is_void_v<ElementS>, "Use MoEGEMMDoubleBuffer for plain BF16");
  // Shared with the host packer via moe_scale_layout.hpp so padding cannot drift.
  constexpr int kScaleAlign = cutlass::moe::kScaleAlign;
  constexpr int kTensorPaddedScaleN = cutlass::moe::kTensorPaddedScaleN;
  constexpr int kTensorScaleK = cutlass::moe::kTensorScaleK;

  namespace coll = cutlass::gemm::collective;

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto local_id = item.get_local_linear_id();
  int sg_id = (int)item.get_sub_group().get_group_id()[0];

  auto wg_tile = mma.tile_mnk();
  int32_t tile_m = get<0>(wg_tile);
  int32_t tile_n = get<1>(wg_tile);

  constexpr int BLK_M = decltype(get<0>(wg_tile))::value;
  constexpr int BLK_N = decltype(get<1>(wg_tile))::value;
  constexpr int BLK_K = decltype(get<2>(wg_tile))::value;
  constexpr int SG_NUMS_M = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  constexpr int SG_NUMS_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  constexpr int SG_NUMS_K = get<3>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  constexpr int SG_M = ceil_div(BLK_M, SG_NUMS_M);
  constexpr int SG_N = ceil_div(BLK_N, SG_NUMS_N);
  constexpr int SG_K = ceil_div(BLK_K, SG_NUMS_K);

  int32_t m_tiles_in_expert = (uniform_M + tile_m - 1) / tile_m;
  int32_t n_tiles           = (N + tile_n - 1) / tile_n;
  int64_t mn_per_expert     = int64_t(m_tiles_in_expert) * n_tiles;
  int64_t total_work        = int64_t(num_experts) * mn_per_expert;

  uint64_t wg_id;
  if (scheduler_params.raster_order_ ==
      PersistentTileSchedulerSm90GroupParams<ProblemShape>::RasterOrder::AlongN) {
    wg_id = uint64_t(BlockIdxX()) + uint64_t(BlockIdxY()) * uint64_t(GridDimX());
  } else {
    wg_id = uint64_t(BlockIdxX()) * uint64_t(GridDimY()) + uint64_t(BlockIdxY());
  }
  int64_t grid_size = int64_t(GridDimX()) * int64_t(GridDimY()) * int64_t(GridDimZ());

  auto D_tensor_init = make_moe_tensor<ElementD, LayoutKindD>(Outputs, uniform_M, N);
  Tensor cD_init = make_identity_tensor(D_tensor_init.shape());
  Tensor gD_even_init = local_tile(cD_init, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});
  Tensor gD_odd_init  = local_tile(cD_init, wg_tile, make_coord(0, 1, 0), Step<_1, _1, X>{});

  auto thr_mma = mma.get_slice(local_id);
  auto tCrD_even = thr_mma.partition_sg_fragment_C(gD_even_init);
  auto tCrD_odd  = thr_mma.partition_sg_fragment_C(gD_odd_init);
  // Every K-tile uses gemm<false> (accumulate); in-loop clears only run after a
  // store, so the first accumulation needs these fragments cleared up front.
  clear(tCrD_even);
  clear(tCrD_odd);

  bool is_first_wave  = true;
  ElementD *ptr_D_prev    = Outputs;
  int32_t   wg_m_prev     = 0;
  int32_t   wg_n_odd_prev = 1;

  constexpr int prefetch_dist = 2;
  int k_tile_count = ceil_div(K, get<2>(wg_tile));
  constexpr uint8_t NUM_SUBGROUPS = 32;
  uint8_t k_tile_subgroup_store_ratio = k_tile_count >= NUM_SUBGROUPS ? 1 : ceil_div(NUM_SUBGROUPS, k_tile_count);

  const int scale_n = ceil_div(int(N), int(GroupN > 0 ? GroupN : 1));

  // Each WG owns pair slots wg_id, wg_id+grid_size, wg_id+2*grid_size, ...
  // Each slot covers two consecutive flat MN indices (even=idx, odd=idx+1).
  // Expert-level setup is rebuilt only when the expert changes.
  int32_t cur_expert_id = -1;
  int64_t cumulative_M  = 0;
  ElementD *ptr_D       = Outputs;

  // Dummy-initialised; overwritten on first expert entry.
  auto A_tensor = make_moe_tensor<ElementA, LayoutKindA>(
      const_cast<ElementA *>(Activations), uniform_M, K);
  auto B_tensor = make_moe_tensor<ElementB, LayoutKindB, /*IsBOperand=*/true>(
      const_cast<ElementB *>(Weights), N, K);
  auto D_tensor = make_moe_tensor<ElementD, LayoutKindD>(Outputs, uniform_M, N);

  Tensor cA = make_identity_tensor(A_tensor.shape());
  Tensor cB = make_identity_tensor(B_tensor.shape());
  Tensor cD = make_identity_tensor(D_tensor.shape());

  auto tiled_copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A_tensor);
  auto tiled_copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B_tensor);
  auto tiled_copy_d = get_block_2d_copy_D<GmemTiledCopyD>(mma, D_tensor);

  auto thr_copy_a = tiled_copy_a.get_slice(local_id);
  auto thr_copy_b = tiled_copy_b.get_slice(local_id);
  auto thr_copy_d = tiled_copy_d.get_slice(local_id);

  auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
  auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);
  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  int32_t wg_m_cur = -1;
  auto gA_cur   = local_tile(cA, select<0, 2>(wg_tile), make_coord(0, _));
  auto tArA_cur = thr_copy_a.partition_sg_fragment_D(gA_cur(_, _, 0));
  auto tCrA_cur = thr_mma.partition_sg_fragment_A(gA_cur(_, _, 0));
  auto tAgA_cur = thr_copy_a.partition_S(gA_cur);
  auto pAgA_cur = thr_prefetch_A.partition_S(gA_cur);

  auto rebuild_A = [&](int32_t wg_m) {
    auto gA   = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
    tArA_cur  = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    tCrA_cur  = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    tAgA_cur  = thr_copy_a.partition_S(gA);
    pAgA_cur  = thr_prefetch_A.partition_S(gA);
    wg_m_cur  = wg_m;
  };

  using ElementScaleA = ElementS;
  using ElementScaleB = ElementS;

  [[maybe_unused]] int padded_scale_n = cute::round_up(scale_n, kScaleAlign);

  for (int64_t idx = int64_t(wg_id) * 2; idx < total_work; idx += grid_size * 2) {

    int32_t expert_id  = int32_t(idx / mn_per_expert);
    int64_t mn_even    = idx % mn_per_expert;
    int32_t wg_m_even  = int32_t(mn_even / n_tiles);
    int32_t wg_n_even  = int32_t(mn_even % n_tiles);

    int64_t idx_odd   = idx + 1;
    int64_t mn_odd    = idx_odd % mn_per_expert;
    int32_t wg_m_odd  = int32_t(mn_odd / n_tiles);
    int32_t wg_n_odd  = int32_t(mn_odd % n_tiles);

    // Rebuild expert-level state only on expert change.
    if (expert_id != cur_expert_id) {
      cumulative_M = int64_t(expert_id) * uniform_M;

      auto byte_offset_A = cumulative_M * K * cute::sizeof_bits_v<ElementA> / 8;
      auto byte_offset_B = int64_t(expert_id) * K * N * cute::sizeof_bits_v<ElementB> / 8;
      ElementA *ptr_A = reinterpret_cast<ElementA *>(
          reinterpret_cast<uint8_t *>(const_cast<ElementA *>(Activations)) + byte_offset_A);
      ElementB *ptr_B = reinterpret_cast<ElementB *>(
          reinterpret_cast<uint8_t *>(const_cast<ElementB *>(Weights)) + byte_offset_B);
      ptr_D = Outputs + cumulative_M * N;

      A_tensor = make_moe_tensor<ElementA, LayoutKindA>(ptr_A, uniform_M, K);
      B_tensor = make_moe_tensor<ElementB, LayoutKindB, /*IsBOperand=*/true>(ptr_B, N, K);
      D_tensor = make_moe_tensor<ElementD, LayoutKindD>(ptr_D, uniform_M, N);

      cA = make_identity_tensor(A_tensor.shape());
      cB = make_identity_tensor(B_tensor.shape());
      cD = make_identity_tensor(D_tensor.shape());

      tiled_copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A_tensor);
      tiled_copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B_tensor);
      tiled_copy_d = get_block_2d_copy_D<GmemTiledCopyD>(mma, D_tensor);

      thr_copy_a = tiled_copy_a.get_slice(local_id);
      thr_copy_b = tiled_copy_b.get_slice(local_id);
      thr_copy_d = tiled_copy_d.get_slice(local_id);

      prefetch_a = make_block_2d_prefetch(tiled_copy_a);
      prefetch_b = make_block_2d_prefetch(tiled_copy_b);
      thr_prefetch_A = prefetch_a.get_slice(local_id);
      thr_prefetch_B = prefetch_b.get_slice(local_id);

      wg_m_cur = -1;
      cur_expert_id = expert_id;
    }

    // BLOCK-path scale tensors (CfgGroupK > 0), rebuilt fresh each iteration:
    // they have runtime strides, so storing them as mutable vars would trip a
    // CuTe static-vs-runtime type mismatch.
    const int scale_k_block = (CfgGroupK > 0) ? ceil_div(int(K), int(GroupK)) : 1;
    const int round_up_M    = (uniform_M + (kScaleAlign - 1)) & ~(kScaleAlign - 1);
    // Per-expert stride expert_id * round_up_M, NOT ceil64(cumulative_M):
    // ceil64(i * uM) != i * ceil64(uM) when uM % 64 != 0, drifting scaleA into
    // the wrong expert's data from expert 4 onward.
    [[maybe_unused]] const int64_t padded_cumulative_M_i =
        int64_t(expert_id) * round_up_M;

    auto sA_block = [&]() {
      if constexpr (CfgGroupK > 0)
        // Padded M extent (round_up_M) for DWord-aligned MXFP scale load.
        return make_tensor(
            make_gmem_ptr(const_cast<ElementS *>(ScalesA) +
                          padded_cumulative_M_i * scale_k_block),
            make_layout(make_shape(round_up_M, scale_k_block, 1),
                        make_stride(_1{}, round_up_M,
                                    int64_t(round_up_M) * scale_k_block)));
      else return make_tensor(make_gmem_ptr(const_cast<ElementS *>(ScalesA)),
                              make_layout(make_shape(1)));
    }();
    auto sB_block = [&]() {
      if constexpr (CfgGroupK > 0)
        return make_tensor(
            make_gmem_ptr(const_cast<ElementS *>(ScalesB) +
                          int64_t(expert_id) * padded_scale_n * scale_k_block),
            make_layout(make_shape(scale_n, scale_k_block, 1),
                        make_stride(_1{}, padded_scale_n,
                                    int64_t(padded_scale_n) * scale_k_block)));
      else return make_tensor(make_gmem_ptr(const_cast<ElementS *>(ScalesB)),
                              make_layout(make_shape(1)));
    }();

      if (wg_m_even != wg_m_cur) rebuild_A(wg_m_even);
      auto tArA_even = tArA_cur;
      auto tCrA_even = tCrA_cur;
      auto tAgA_even = tAgA_cur;
      auto pAgA_even = pAgA_cur;

      Tensor gB_even = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n_even, _));
      Tensor gD_even = local_tile(cD, wg_tile, make_coord(wg_m_even, wg_n_even, 0), Step<_1, _1, X>{});
      auto tBrB_even   = thr_copy_b.partition_sg_fragment_D(gB_even(_, _, 0));
      auto tCrB_even   = thr_mma.partition_sg_fragment_B(gB_even(_, _, 0));
      Tensor tBgB_even = thr_copy_b.partition_S(gB_even);
      auto tCgD_even   = thr_copy_d.partition_D(gD_even);
      auto pBgB_even   = thr_prefetch_B.partition_S(gB_even);

      // Fire even tile prefetch warm-up here so odd tile setup provides latency hiding.
      int prefetch_k = 0;
      CUTE_UNROLL
      for (; prefetch_k < prefetch_dist; ++prefetch_k) {
        prefetch(prefetch_a, pAgA_even(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB_even(_, _, _, prefetch_k));
      }

      if (wg_m_odd != wg_m_even) rebuild_A(wg_m_odd);
      auto tArA_odd = tArA_cur;
      auto tCrA_odd = tCrA_cur;
      auto tAgA_odd = tAgA_cur;
      auto pAgA_odd = pAgA_cur;

      Tensor gB_odd = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n_odd, _));
      Tensor gD_odd = local_tile(cD, wg_tile, make_coord(wg_m_odd, wg_n_odd, 0), Step<_1, _1, X>{});
      auto tBrB_odd   = thr_copy_b.partition_sg_fragment_D(gB_odd(_, _, 0));
      auto tCrB_odd   = thr_mma.partition_sg_fragment_B(gB_odd(_, _, 0));
      Tensor tBgB_odd = thr_copy_b.partition_S(gB_odd);
      auto tCgD_odd   = thr_copy_d.partition_D(gD_odd);
      auto pBgB_odd   = thr_prefetch_B.partition_S(gB_odd);

      const int m_coord_even = wg_m_even * BLK_M + (sg_id / SG_NUMS_N) * SG_M;
      const int n_coord_even = wg_n_even * BLK_N + (sg_id % SG_NUMS_N) * SG_N;
      const int m_coord_odd  = wg_m_odd  * BLK_M + (sg_id / SG_NUMS_N) * SG_M;
      const int n_coord_odd  = wg_n_odd  * BLK_N + (sg_id % SG_NUMS_N) * SG_N;

      constexpr int l_coord = 0;

      int prefetch_k_odd = 0;

      if constexpr (CfgGroupK == 0) {
        // ── TENSOR scale path: HW BDPAS ───────────────────────────────────
        // Fixed surface geometry from moe_scale_layout.hpp — the same constants
        // the host packer uses: height 2 (BDPAS offset scheme) and one
        // kScaleAlign-wide N stripe per expert. The static_assert enforces that
        // one stripe covers the whole subgroup N extent; the 2D load walks in
        // kScaleAlign-wide steps, so a larger SG_N would read past the host
        // allocation. Scale is fused into each DPAS via make_zip_tensor (same
        // structure as the BLOCK path, different scale layout).
        static_assert(cute::sizeof_bits_v<ElementA> == 8,
                      "ScaleKind::Tensor scale surface geometry is fp8-only.");
        constexpr int scale_k_tensor = kTensorScaleK;
        const int round_up_M_tensor  = (uniform_M + (kScaleAlign - 1)) & ~(kScaleAlign - 1);
        static_assert(SG_N <= kTensorPaddedScaleN,
                      "Tensor-scale surface is one kScaleAlign-wide N stripe per "
                      "expert; SG_N (BLK_N/SG_NUMS_N) must not exceed it.");
        constexpr int padded_scale_n_tensor = kTensorPaddedScaleN;
        // Same per-expert-stride fix as the BLOCK path above.
        [[maybe_unused]] const int64_t padded_cumM_tensor =
            int64_t(expert_id) * round_up_M_tensor;

        auto sA_block_t = make_tensor(
            make_gmem_ptr(const_cast<ElementS *>(ScalesA) +
                          padded_cumM_tensor * scale_k_tensor),
            // Padded M extent (round_up_M_tensor) for DWord-aligned MXFP scale load.
            make_layout(make_shape(round_up_M_tensor, scale_k_tensor, 1),
                        make_stride(_1{}, round_up_M_tensor,
                                    int64_t(round_up_M_tensor) * scale_k_tensor)));
        auto sB_block_t = make_tensor(
            make_gmem_ptr(const_cast<ElementS *>(ScalesB) +
                          int64_t(expert_id) * padded_scale_n_tensor * scale_k_tensor),
            make_layout(make_shape(Int<padded_scale_n_tensor>{}, scale_k_tensor, 1),
                        make_stride(_1{}, Int<padded_scale_n_tensor>{},
                                    int64_t(padded_scale_n_tensor) * scale_k_tensor)));

        // TensorGroupK = MMA_K = Shape_MNK[2] = BLK_K. Matches the collective.
        constexpr int MMA_K_T = get<2>(typename TiledMMA::Shape_MNK{});

        using GemmIterM_T = Int<decltype(size<1>(tCrA_even.shape()))::value>;
        using GemmIterN_T = Int<decltype(size<1>(tCrB_even.shape()))::value>;
        using GemmIterK_T = Int<decltype(size<2>(tCrB_even.shape()))::value>;

        // Tensor scale: one scale per row (A) / per expert (B) for all K tiles,
        // so k_tile_count=1 — loaded once, not per K tile.
        auto [tc_sA_te, ci_sA_te, fr_sA_te] =
            coll::make_scaled_copy<void, ElementScaleA, SG_M, SG_K, MMA_K_T>(
                sA_block_t, m_coord_even, l_coord, 1);
        auto [tc_sB_te, ci_sB_te, fr_sB_te] =
            coll::make_scaled_copy<void, ElementScaleB, SG_N, SG_K, MMA_K_T>(
                sB_block_t, 0, l_coord, 1);
        auto [sm_te, sn_te, sak_te, sbk_te] =
            coll::make_scaled_offsets<GemmIterM_T::value, GemmIterN_T::value, GemmIterK_T::value,
                                      MMA_K_T, MMA_K_T,
                                      typename decltype(tc_sA_te)::BlockShape,
                                      typename decltype(tc_sB_te)::BlockShape>();
        auto [tp_sA_te, pi_sA_te] =
            coll::make_scaled_prefetch<decltype(tc_sA_te), SG_M, SG_K, MMA_K_T>(
                tc_sA_te, m_coord_even, l_coord, 1);
        auto [tp_sB_te, pi_sB_te] =
            coll::make_scaled_prefetch<decltype(tc_sB_te), SG_N, SG_K, MMA_K_T>(
                tc_sB_te, 0, l_coord, 1);

        auto [tc_sA_to, ci_sA_to, fr_sA_to] =
            coll::make_scaled_copy<void, ElementScaleA, SG_M, SG_K, MMA_K_T>(
                sA_block_t, m_coord_odd, l_coord, 1);
        auto [tc_sB_to, ci_sB_to, fr_sB_to] =
            coll::make_scaled_copy<void, ElementScaleB, SG_N, SG_K, MMA_K_T>(
                sB_block_t, 0, l_coord, 1);
        auto [tp_sA_to, pi_sA_to] =
            coll::make_scaled_prefetch<decltype(tc_sA_to), SG_M, SG_K, MMA_K_T>(
                tc_sA_to, m_coord_odd, l_coord, 1);
        auto [tp_sB_to, pi_sB_to] =
            coll::make_scaled_prefetch<decltype(tc_sB_to), SG_N, SG_K, MMA_K_T>(
                tc_sB_to, 0, l_coord, 1);

        using scA_t_t = intel::vector_t<ElementScaleA, decltype(size(fr_sA_te))::value>;
        using scB_t_t = intel::vector_t<ElementScaleB, decltype(size(fr_sB_te))::value>;

        // Tensor scale: prefetch and load once before the k-loop (same as collective).
        prefetch(tp_sA_te, pi_sA_te(_, _, _, 0));
        prefetch(tp_sB_te, pi_sB_te(_, _, _, 0));
        copy(tc_sA_te, ci_sA_te(_, _, _, 0), fr_sA_te);
        copy(tc_sB_te, ci_sB_te(_, _, _, 0), fr_sB_te);

        Tensor scA_te_out = make_tensor(recast<scA_t_t>(fr_sA_te).data(),
            make_layout(Shape<_1, GemmIterM_T, _1>{}, Stride<_1, _0, _0>{}));
        Tensor scB_te_out = make_tensor(recast<scB_t_t>(fr_sB_te).data(),
            make_layout(Shape<_1, GemmIterN_T, _1>{}, Stride<_1, _0, _0>{}));

        // ── Even k-loop ────────────────────────────────────────────────────
        for (int k = 0; k < k_tile_count; ++k, ++prefetch_k) {
          if (!is_first_wave && sg_id / k_tile_subgroup_store_ratio == k) {
            auto D_tensor_prev     = make_moe_tensor<ElementD, LayoutKindD>(ptr_D_prev, uniform_M, N);
            auto cD_prev           = make_identity_tensor(D_tensor_prev.shape());
            auto tiled_copy_d_prev = get_block_2d_copy_D<GmemTiledCopyD>(mma, D_tensor_prev);
            auto thr_copy_d_prev   = tiled_copy_d_prev.get_slice(local_id);
            auto gD_odd_prev       = local_tile(cD_prev, wg_tile, make_coord(wg_m_prev, wg_n_odd_prev, 0), Step<_1, _1, X>{});
            auto tCrD_final_odd    = thr_copy_d_prev.partition_sg_fragment_S(gD_odd_prev);
            auto tCgD_odd_prev     = thr_copy_d_prev.partition_D(gD_odd_prev);
            reorder(tCrD_odd, tCrD_final_odd);
            copy(tiled_copy_d_prev, tCrD_final_odd, tCgD_odd_prev);
            clear(tCrD_odd);
          }
          copy(tiled_copy_a, tAgA_even(_, _, _, k), tArA_even);
          copy(tiled_copy_b, tBgB_even(_, _, _, k), tBrB_even);
          if (prefetch_k < k_tile_count) {
            prefetch(prefetch_a, pAgA_even(_, _, _, prefetch_k));
            prefetch(prefetch_b, pBgB_even(_, _, _, prefetch_k));
            prefetch(tp_sA_te, pi_sA_te(_, _, _, 0));
            prefetch(tp_sB_te, pi_sB_te(_, _, _, 0));
          } else {
            prefetch(prefetch_a, pAgA_odd(_, _, _, prefetch_k_odd));
            prefetch(prefetch_b, pBgB_odd(_, _, _, prefetch_k_odd));
            prefetch(tp_sA_to, pi_sA_to(_, _, _, 0));
            prefetch(tp_sB_to, pi_sB_to(_, _, _, 0));
            ++prefetch_k_odd;
          }
          reorder(tArA_even, tCrA_even);
          reorder(tBrB_even, tCrB_even);
          cute::gemm<false>(mma,
              make_zip_tensor(tCrA_even, scA_te_out, sm_te, sak_te),
              make_zip_tensor(tCrB_even, scB_te_out, sn_te, sbk_te),
              tCrD_even);
        }

        // Load odd tile scales once before the odd k-loop.
        prefetch(tp_sA_to, pi_sA_to(_, _, _, 0));
        prefetch(tp_sB_to, pi_sB_to(_, _, _, 0));
        copy(tc_sA_to, ci_sA_to(_, _, _, 0), fr_sA_to);
        copy(tc_sB_to, ci_sB_to(_, _, _, 0), fr_sB_to);
        Tensor scA_to_out = make_tensor(recast<scA_t_t>(fr_sA_to).data(),
            make_layout(Shape<_1, GemmIterM_T, _1>{}, Stride<_1, _0, _0>{}));
        Tensor scB_to_out = make_tensor(recast<scB_t_t>(fr_sB_to).data(),
            make_layout(Shape<_1, GemmIterN_T, _1>{}, Stride<_1, _0, _0>{}));

        // ── Odd k-loop ────────────────────────────────────────────────────
        for (int k = 0; k < k_tile_count; ++k, ++prefetch_k_odd) {
          if (sg_id / k_tile_subgroup_store_ratio == k) {
            auto tCrD_final_even = thr_copy_d.partition_sg_fragment_S(gD_even);
            reorder(tCrD_even, tCrD_final_even);
            copy(tiled_copy_d, tCrD_final_even, tCgD_even);
            clear(tCrD_even);
          }
          copy(tiled_copy_a, tAgA_odd(_, _, _, k), tArA_odd);
          copy(tiled_copy_b, tBgB_odd(_, _, _, k), tBrB_odd);
          if (prefetch_k_odd < k_tile_count) {
            prefetch(prefetch_a, pAgA_odd(_, _, _, prefetch_k_odd));
            prefetch(prefetch_b, pBgB_odd(_, _, _, prefetch_k_odd));
          }
          reorder(tArA_odd, tCrA_odd);
          reorder(tBrB_odd, tCrB_odd);
          cute::gemm<false>(mma,
              make_zip_tensor(tCrA_odd, scA_to_out, sm_te, sak_te),
              make_zip_tensor(tCrB_odd, scB_to_out, sn_te, sbk_te),
              tCrD_odd);
        }
        ptr_D_prev    = ptr_D;
        wg_m_prev     = wg_m_odd;
        wg_n_odd_prev = wg_n_odd;

      } else {
        // ── BLOCK (MX) scale path ─────────────────────────────────────────
        constexpr int MMA_K = get<2>(typename TiledMMA::Shape_MNK{});
        constexpr int k_reload_factor = cute::max(CfgGroupK / BLK_K, 1);

        using GemmIterM = Int<decltype(size<1>(tCrA_even.shape()))::value>;
        using GemmIterN = Int<decltype(size<1>(tCrB_even.shape()))::value>;
        using GemmIterK = Int<decltype(size<2>(tCrB_even.shape()))::value>;

        // Even tile scale iterators
        auto [tiled_copy_scaleA_e, copy_iter_scaleA_e, fragment_scaleA_e] =
            coll::make_scaled_copy<void, ElementScaleA, SG_M, SG_K, CfgGroupK>(
                sA_block, m_coord_even, l_coord, k_tile_count);
        auto [tiled_copy_scaleB_e, copy_iter_scaleB_e, fragment_scaleB_e] =
            coll::make_scaled_copy<void, ElementScaleB, SG_N, SG_K, CfgGroupK>(
                sB_block, n_coord_even, l_coord, k_tile_count);
        auto [scale_m_offsets_e, scale_n_offsets_e, scale_ak_offsets_e, scale_bk_offsets_e] =
            coll::make_scaled_offsets<
                GemmIterM::value, GemmIterN::value, GemmIterK::value, MMA_K, CfgGroupK,
                typename decltype(tiled_copy_scaleA_e)::BlockShape,
                typename decltype(tiled_copy_scaleB_e)::BlockShape>();
        auto [tiled_prefetch_scaleA_e, prefetch_iter_scaleA_e] =
            coll::make_scaled_prefetch<decltype(tiled_copy_scaleA_e), SG_M, SG_K, CfgGroupK>(
                tiled_copy_scaleA_e, m_coord_even, l_coord, k_tile_count);
        auto [tiled_prefetch_scaleB_e, prefetch_iter_scaleB_e] =
            coll::make_scaled_prefetch<decltype(tiled_copy_scaleB_e), SG_N, SG_K, CfgGroupK>(
                tiled_copy_scaleB_e, n_coord_even, l_coord, k_tile_count);

        using scaleA_vec_t = intel::vector_t<ElementScaleA, decltype(size(fragment_scaleA_e))::value>;
        using scaleB_vec_t = intel::vector_t<ElementScaleB, decltype(size(fragment_scaleB_e))::value>;

        auto [tiled_copy_scaleA_o, copy_iter_scaleA_o, fragment_scaleA_o] =
            coll::make_scaled_copy<void, ElementScaleA, SG_M, SG_K, CfgGroupK>(
                sA_block, m_coord_odd, l_coord, k_tile_count);
        auto [tiled_copy_scaleB_o, copy_iter_scaleB_o, fragment_scaleB_o] =
            coll::make_scaled_copy<void, ElementScaleB, SG_N, SG_K, CfgGroupK>(
                sB_block, n_coord_odd, l_coord, k_tile_count);
        auto [scale_m_offsets_o, scale_n_offsets_o, scale_ak_offsets_o, scale_bk_offsets_o] =
            coll::make_scaled_offsets<
                GemmIterM::value, GemmIterN::value, GemmIterK::value, MMA_K, CfgGroupK,
                typename decltype(tiled_copy_scaleA_o)::BlockShape,
                typename decltype(tiled_copy_scaleB_o)::BlockShape>();
        auto [tiled_prefetch_scaleA_o, prefetch_iter_scaleA_o] =
            coll::make_scaled_prefetch<decltype(tiled_copy_scaleA_o), SG_M, SG_K, CfgGroupK>(
                tiled_copy_scaleA_o, m_coord_odd, l_coord, k_tile_count);
        auto [tiled_prefetch_scaleB_o, prefetch_iter_scaleB_o] =
            coll::make_scaled_prefetch<decltype(tiled_copy_scaleB_o), SG_N, SG_K, CfgGroupK>(
                tiled_copy_scaleB_o, n_coord_odd, l_coord, k_tile_count);

        // Scale warm-up (A/B already prefetched above odd tile setup).
        CUTE_UNROLL
        for (int p = 0; p < prefetch_dist; ++p) {
          prefetch(tiled_prefetch_scaleA_e, prefetch_iter_scaleA_e(_, _, _, p / k_reload_factor));
          prefetch(tiled_prefetch_scaleB_e, prefetch_iter_scaleB_e(_, _, _, p / k_reload_factor));
        }

        // ── Even k-loop ────────────────────────────────────────────────────
        for (int k = 0; k < k_tile_count; ++k, ++prefetch_k) {
          if (!is_first_wave && sg_id / k_tile_subgroup_store_ratio == k) {
            auto D_tensor_prev  = make_moe_tensor<ElementD, LayoutKindD>(ptr_D_prev, uniform_M, N);
            auto cD_prev        = make_identity_tensor(D_tensor_prev.shape());
            auto tiled_copy_d_prev = get_block_2d_copy_D<GmemTiledCopyD>(mma, D_tensor_prev);
            auto thr_copy_d_prev   = tiled_copy_d_prev.get_slice(local_id);
            auto gD_odd_prev       = local_tile(cD_prev, wg_tile, make_coord(wg_m_prev, wg_n_odd_prev, 0), Step<_1, _1, X>{});
            auto tCrD_final_odd    = thr_copy_d_prev.partition_sg_fragment_S(gD_odd_prev);
            auto tCgD_odd_prev     = thr_copy_d_prev.partition_D(gD_odd_prev);
            reorder(tCrD_odd, tCrD_final_odd);
            copy(tiled_copy_d_prev, tCrD_final_odd, tCgD_odd_prev);
            clear(tCrD_odd);
          }
          copy(tiled_copy_a, tAgA_even(_, _, _, k), tArA_even);
          copy(tiled_copy_b, tBgB_even(_, _, _, k), tBrB_even);
          copy(tiled_copy_scaleA_e, copy_iter_scaleA_e(_, _, _, k / k_reload_factor), fragment_scaleA_e);
          copy(tiled_copy_scaleB_e, copy_iter_scaleB_e(_, _, _, k / k_reload_factor), fragment_scaleB_e);
          if (prefetch_k < k_tile_count) {
            prefetch(prefetch_a, pAgA_even(_, _, _, prefetch_k));
            prefetch(prefetch_b, pBgB_even(_, _, _, prefetch_k));
            prefetch(tiled_prefetch_scaleA_e, prefetch_iter_scaleA_e(_, _, _, prefetch_k / k_reload_factor));
            prefetch(tiled_prefetch_scaleB_e, prefetch_iter_scaleB_e(_, _, _, prefetch_k / k_reload_factor));
          } else {
            prefetch(prefetch_a, pAgA_odd(_, _, _, prefetch_k_odd));
            prefetch(prefetch_b, pBgB_odd(_, _, _, prefetch_k_odd));
            prefetch(tiled_prefetch_scaleA_o, prefetch_iter_scaleA_o(_, _, _, prefetch_k_odd / k_reload_factor));
            prefetch(tiled_prefetch_scaleB_o, prefetch_iter_scaleB_o(_, _, _, prefetch_k_odd / k_reload_factor));
            ++prefetch_k_odd;
          }
          reorder(tArA_even, tCrA_even);
          reorder(tBrB_even, tCrB_even);

          Tensor scaleA_e = make_tensor(recast<scaleA_vec_t>(fragment_scaleA_e).data(),
              make_layout(Shape<_1, GemmIterM, _1>{}, Stride<_1, _0, _0>{}));
          Tensor scaleB_e = make_tensor(recast<scaleB_vec_t>(fragment_scaleB_e).data(),
              make_layout(Shape<_1, GemmIterN, _1>{}, Stride<_1, _0, _0>{}));

  
            cute::gemm<false>(mma,
                make_zip_tensor(tCrA_even, scaleA_e, scale_m_offsets_e, scale_ak_offsets_e),
                make_zip_tensor(tCrB_even, scaleB_e, scale_n_offsets_e, scale_bk_offsets_e),
                tCrD_even);
        }

        // ── Odd k-loop ───────────────────────────────────────────────────
        for (int k = 0; k < k_tile_count; ++k, ++prefetch_k_odd) {
          if (sg_id / k_tile_subgroup_store_ratio == k) {
            auto tCrD_final_even = thr_copy_d.partition_sg_fragment_S(gD_even);
            reorder(tCrD_even, tCrD_final_even);
            copy(tiled_copy_d, tCrD_final_even, tCgD_even);
            clear(tCrD_even);
          }
          copy(tiled_copy_a, tAgA_odd(_, _, _, k), tArA_odd);
          copy(tiled_copy_b, tBgB_odd(_, _, _, k), tBrB_odd);
          copy(tiled_copy_scaleA_o, copy_iter_scaleA_o(_, _, _, k / k_reload_factor), fragment_scaleA_o);
          copy(tiled_copy_scaleB_o, copy_iter_scaleB_o(_, _, _, k / k_reload_factor), fragment_scaleB_o);
          if (prefetch_k_odd < k_tile_count) {
            prefetch(prefetch_a, pAgA_odd(_, _, _, prefetch_k_odd));
            prefetch(prefetch_b, pBgB_odd(_, _, _, prefetch_k_odd));
            prefetch(tiled_prefetch_scaleA_o, prefetch_iter_scaleA_o(_, _, _, prefetch_k_odd / k_reload_factor));
            prefetch(tiled_prefetch_scaleB_o, prefetch_iter_scaleB_o(_, _, _, prefetch_k_odd / k_reload_factor));
          }
          reorder(tArA_odd, tCrA_odd);
          reorder(tBrB_odd, tCrB_odd);

          Tensor scaleA_o = make_tensor(recast<scaleA_vec_t>(fragment_scaleA_o).data(),
              make_layout(Shape<_1, GemmIterM, _1>{}, Stride<_1, _0, _0>{}));
          Tensor scaleB_o = make_tensor(recast<scaleB_vec_t>(fragment_scaleB_o).data(),
              make_layout(Shape<_1, GemmIterN, _1>{}, Stride<_1, _0, _0>{}));

          cute::gemm<false>(mma,
              make_zip_tensor(tCrA_odd, scaleA_o, scale_m_offsets_o, scale_ak_offsets_o),
              make_zip_tensor(tCrB_odd, scaleB_o, scale_n_offsets_o, scale_bk_offsets_o),
              tCrD_odd);
        }
        ptr_D_prev    = ptr_D;
        wg_m_prev     = wg_m_odd;
        wg_n_odd_prev = wg_n_odd;
      } // end if constexpr CfgGroupK

      is_first_wave = false;
  } // strided pair loop

  // Flush the final odd result. Scale was fused into DPAS (both paths) — no
  // post-multiply needed.
  if (!is_first_wave) {
    auto D_tensor_last     = make_moe_tensor<ElementD, LayoutKindD>(ptr_D_prev, uniform_M, N);
    auto cD_last           = make_identity_tensor(D_tensor_last.shape());
    auto tiled_copy_d_last = get_block_2d_copy_D<GmemTiledCopyD>(mma, D_tensor_last);
    auto thr_copy_d_last   = tiled_copy_d_last.get_slice(local_id);
    auto gD_odd_last       = local_tile(cD_last, wg_tile, make_coord(wg_m_prev, wg_n_odd_prev, 0), Step<_1, _1, X>{});
    auto tCrD_final_odd    = thr_copy_d_last.partition_sg_fragment_S(gD_odd_last);
    auto tCgD_odd_last     = thr_copy_d_last.partition_D(gD_odd_last);
    reorder(tCrD_odd, tCrD_final_odd);
    copy(tiled_copy_d_last, tCrD_final_odd, tCgD_odd_last);
  }
}


} // namespace MoE
