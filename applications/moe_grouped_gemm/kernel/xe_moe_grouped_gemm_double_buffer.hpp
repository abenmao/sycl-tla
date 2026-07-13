/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel corporation. All rights reserved.
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
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/platform/platform.h"
#include "moe_grouped_gemm/kernel/xe_moe_grouped_gemm.hpp"
#include <cute/util/compat.hpp>

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

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
  // across the entire persistent loop. Initialised with dummy coordinates.

  auto D_tensor_init = make_moe_tensor<ElementD, LayoutKindD>(
      Outputs, uniform_M, N);

  Tensor cD_init = make_identity_tensor(D_tensor_init.shape());

  Tensor gD_even_init = local_tile(cD_init, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});
  Tensor gD_odd_init  = local_tile(cD_init, wg_tile, make_coord(0, 1, 0), Step<_1, _1, X>{});

  auto thr_mma = mma.get_slice(local_id);

  auto tCrD_even = thr_mma.partition_sg_fragment_C(gD_even_init);
  auto tCrD_odd  = thr_mma.partition_sg_fragment_C(gD_odd_init);

  bool is_first_wave  = true;
  ElementD *ptr_D_prev    = Outputs;
  int32_t   wg_m_prev     = 0;
  int32_t   wg_n_odd_prev = 1;

  constexpr int prefetch_dist = 2;
  int k_tile_count = ceil_div(K, get<2>(wg_tile));
  constexpr uint8_t NUM_SUBGROUPS = 32;
  uint8_t k_tile_subgroup_store_ratio = k_tile_count >= NUM_SUBGROUPS ? 1 : ceil_div(NUM_SUBGROUPS, k_tile_count);

  // ── Strided pair loop ────────────────────────────────────────────────────
  // Each WG owns pair slots wg_id, wg_id+grid_size, wg_id+2*grid_size, ...
  // Each slot covers two consecutive flat MN indices (even=idx, odd=idx+1).
  // Expert-level setup is rebuilt only when the expert changes.

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

} // namespace MoE
