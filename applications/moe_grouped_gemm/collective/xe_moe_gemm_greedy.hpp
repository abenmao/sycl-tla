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
#pragma once

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cute/tensor.hpp>
#include <cute/tensor_zip.hpp>

#include "cutlass/fp8_to_fp16.h"
#include "cutlass/gemm/collective/xe_mma_blockscaled_scale_traits.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/sycl_event_manager.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

template <class T> struct is_16_bit_fp : std::false_type {};

template <> struct is_16_bit_fp<cutlass::half_t> : std::true_type {};
template <> struct is_16_bit_fp<cutlass::bfloat16_t> : std::true_type {};

template <class T>
inline constexpr bool is_16_bit_fp_v =
    is_16_bit_fp<std::remove_cv_t<std::remove_reference_t<T>>>::value;

static_assert(is_16_bit_fp_v<cutlass::bfloat16_t>);
static_assert(is_16_bit_fp_v<cutlass::half_t>);

namespace MoE {

using namespace cute;

// ---------------------------------------------------------------------------
// GREEDY plain (bf16) mainloop. blk_coord get<0> is the M-ROW ORIGIN (m_off), NOT
// a tile index -- the greedy split emits mixed tile_m per expert, so a tile's row
// origin is not wg_m*BLK_M in general. A/D origins are domain_offset by m_off; B
// is untouched. Pre-cleared accumulator (clear + gemm<false> every K-tile).
template <
    class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
    class ATensor, class BTensor, class DTensor, class TiledMMA,
    class = std::enable_if_t<is_16_bit_fp_v<typename ATensor::element_type> &&
                             is_16_bit_fp_v<typename BTensor::element_type>>>
CUTE_DEVICE void
moe_gemm_greedy(ATensor const &A, // (M,K)
         BTensor const &B, // (N,K)
         DTensor &D,       // (M,N)
         cute::Coord<int, int, cute::Underscore, int> blk_coord,
         TiledMMA const &mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto local_id = item.get_local_linear_id();
  const int m_off = get<0>(blk_coord);   // M-row origin
  auto wg_n = get<1>(blk_coord);         // N tile index

  auto wg_tile = mma.tile_mnk();

  // Shift the M origin so logical M-tile 0 begins at row m_off. domain_offset
  // moves the identity coords (which drive both the 2D gmem address and M-bounds
  // predication) by (m_off, 0) for A/D; B untouched. Then always pick M-tile 0.
  Tensor cA = domain_offset(make_coord(m_off, 0), make_identity_tensor(A.shape())); // (M,K)
  Tensor cB = make_identity_tensor(B.shape());                                      // (N,K)
  Tensor cD = domain_offset(make_coord(m_off, 0), make_identity_tensor(D.shape())); // (M,N)

  auto wg_coord = make_coord(_0{}, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(_0{}, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gD = local_tile(cD, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto thr_mma = mma.get_slice(local_id);

  auto tiled_copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto tiled_copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto tiled_copy_d = get_block_2d_copy_D<GmemTiledCopyD>(mma, D);

  auto thr_copy_a = tiled_copy_a.get_slice(local_id);
  auto thr_copy_b = tiled_copy_b.get_slice(local_id);
  auto thr_copy_d = tiled_copy_d.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tCrD = thr_mma.partition_sg_fragment_C(gD);
  auto tCrD_final = thr_copy_d.partition_sg_fragment_S(gD);

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  auto tCgD = thr_copy_d.partition_D(gD);

  auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
  auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  // Pre-clear, then accumulate D += A*B every K-tile (single gemm<false>).
  clear(tCrD);

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
  int k_start_idx = 0;
  int prefetch_k = k_start_idx;
  const int prefetch_dist = 2;
  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

  CUTE_UNROLL
  for (; prefetch_k < prefetch_dist; prefetch_k++) {
    prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
    prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
  }

  for (int k_tile = k_start_idx; k_tile < k_tile_count;
       k_tile++, prefetch_k++) {
    barrier_arrive(barrier_scope);

    copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (prefetch_k < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    cute::gemm<false>(mma, tCrA, tCrB, tCrD);
    barrier_wait(barrier_scope);
  }
  reorder(tCrD, tCrD_final);
  copy(tiled_copy_d, tCrD_final, tCgD);
}

// GREEDY block-scaled mainloop. blk_coord get<0> is the M-ROW ORIGIN (m_off), not
// a tile index -- the greedy split emits mixed tile_m per expert. A/D origins are
// domain_offset by m_off; the A-scale row is m_off + sub-group row; B-scale is
// untouched. Pre-cleared accumulator (clear + gemm<false> every K-tile).
template <int CfgGroupN, int CfgGroupK, class GmemTiledCopyA,
          class GmemTiledCopyB, class GmemTiledCopyD, class ATensor,
          class BTensor, class SATensor, class SBTensor, class DTensor,
          class TiledMMA>
CUTE_DEVICE void
moe_gemm_scaled_greedy(ATensor const &A,        // (M,K)
                     BTensor const &B,        // (N,K)
                     SATensor const &mAscale, // MN-major, see moe_gemm_scaled
                     SBTensor const &mBscale, // MN-major, see moe_gemm_scaled
                     DTensor &D,              // (M,N)
                     cute::Coord<int, int, cute::Underscore, int> blk_coord,
                     TiledMMA const &mma, int GroupN, int GroupK) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto local_id = item.get_local_linear_id();
  const int m_off = get<0>(blk_coord);   // M-row origin
  auto wg_n = get<1>(blk_coord);         // N tile index

  auto wg_tile = mma.tile_mnk();

  // Shift A/D origins so logical M-tile 0 begins at row m_off (as in moe_gemm).
  Tensor cA = domain_offset(make_coord(m_off, 0), make_identity_tensor(A.shape())); // (M,K)
  Tensor cB = make_identity_tensor(B.shape());                                      // (N,K)
  Tensor cD = domain_offset(make_coord(m_off, 0), make_identity_tensor(D.shape())); // (M,N)

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(_0{}, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
  Tensor gD = local_tile(cD, wg_tile, make_coord(_0{}, wg_n, 0), Step<_1, _1, X>{});

  auto thr_mma = mma.get_slice(local_id);

  auto tiled_copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto tiled_copy_b = get_block_2d_copy_B<GmemTiledCopyB>(mma, B);
  auto tiled_copy_d = get_block_2d_copy_D<GmemTiledCopyD>(mma, D);

  auto thr_copy_a = tiled_copy_a.get_slice(local_id);
  auto thr_copy_b = tiled_copy_b.get_slice(local_id);
  auto thr_copy_d = tiled_copy_d.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tCrD = thr_mma.partition_sg_fragment_C(gD);
  auto tCrD_final = thr_copy_d.partition_sg_fragment_S(gD);

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  auto tCgD = thr_copy_d.partition_D(gD);

  auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
  auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  // Subgroup-tile origin within the workgroup tile.
  constexpr int BLK_M = decltype(get<0>(wg_tile))::value;
  constexpr int BLK_N = decltype(get<1>(wg_tile))::value;
  constexpr int BLK_K = decltype(get<2>(wg_tile))::value;
  constexpr int SG_NUMS_M = get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  constexpr int SG_NUMS_N = get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  constexpr int SG_NUMS_K = get<3>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  constexpr int SG_M = ceil_div(BLK_M, SG_NUMS_M);
  constexpr int SG_N = ceil_div(BLK_N, SG_NUMS_N);
  constexpr int SG_K = ceil_div(BLK_K, SG_NUMS_K);

  const int sg_id = cutlass::get_sub_group_id();
  // A-scale row = absolute M row: m_off + sub-group row (not wg_m*BLK_M).
  const int m_coord = m_off + (sg_id / SG_NUMS_N) * SG_M;
  const int n_coord = wg_n * BLK_N + (sg_id % SG_NUMS_N) * SG_N;

  // Pre-clear, then accumulate D += A*B every K-tile (single gemm<false>).
  clear(tCrD);

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;
  int k_start_idx = 0;
  int prefetch_k = k_start_idx;
  const int prefetch_dist = 2;
  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

  if constexpr (CfgGroupK == 0) {
    // TENSOR scale path (see moe_gemm_scaled for full commentary).
    namespace coll = cutlass::gemm::collective;
    using ElementScaleA = typename SATensor::element_type;
    using ElementScaleB = typename SBTensor::element_type;

    constexpr int MMA_K = get<2>(typename TiledMMA::Shape_MNK{});
    constexpr int TensorGroupK = MMA_K;

    using GemmIterM = Int<decltype(size<1>(tCrA.shape()))::value>;
    using GemmIterN = Int<decltype(size<1>(tCrB.shape()))::value>;
    using GemmIterK = Int<decltype(size<2>(tCrB.shape()))::value>;

    const int l_coord = 0;

    auto [tiled_copy_scaleA, copy_iter_scaleA, fragment_scaleA] =
        coll::make_scaled_copy<void, ElementScaleA, SG_M, SG_K, TensorGroupK>(
            mAscale, m_coord, l_coord, 1);
    auto [tiled_copy_scaleB, copy_iter_scaleB, fragment_scaleB] =
        coll::make_scaled_copy<void, ElementScaleB, SG_N, SG_K, TensorGroupK>(
            mBscale, 0, l_coord, 1);
    auto [scale_m_offsets, scale_n_offsets, scale_ak_offsets,
          scale_bk_offsets] =
        coll::make_scaled_offsets<
            GemmIterM::value, GemmIterN::value, GemmIterK::value, MMA_K,
            TensorGroupK,
            typename decltype(tiled_copy_scaleA)::BlockShape,
            typename decltype(tiled_copy_scaleB)::BlockShape>();
    auto [tiled_prefetch_scaleA, prefetch_iter_scaleA] =
        coll::make_scaled_prefetch<decltype(tiled_copy_scaleA), SG_M, SG_K,
                                   TensorGroupK>(tiled_copy_scaleA, m_coord,
                                                 l_coord, 1);
    auto [tiled_prefetch_scaleB, prefetch_iter_scaleB] =
        coll::make_scaled_prefetch<decltype(tiled_copy_scaleB), SG_N, SG_K,
                                   TensorGroupK>(tiled_copy_scaleB, 0, l_coord,
                                                 1);

    using scaleA_vec_t =
        intel::vector_t<ElementScaleA, decltype(size(fragment_scaleA))::value>;
    using scaleB_vec_t =
        intel::vector_t<ElementScaleB, decltype(size(fragment_scaleB))::value>;

    prefetch(tiled_prefetch_scaleA, prefetch_iter_scaleA(_, _, _, 0));
    prefetch(tiled_prefetch_scaleB, prefetch_iter_scaleB(_, _, _, 0));

    copy(tiled_copy_scaleA, copy_iter_scaleA(_, _, _, 0), fragment_scaleA);
    copy(tiled_copy_scaleB, copy_iter_scaleB(_, _, _, 0), fragment_scaleB);

    Tensor scaleA = make_tensor(
        recast<scaleA_vec_t>(fragment_scaleA).data(),
        make_layout(Shape<_1, GemmIterM, _1>{}, Stride<_1, _0, _0>{}));
    Tensor scaleB = make_tensor(
        recast<scaleB_vec_t>(fragment_scaleB).data(),
        make_layout(Shape<_1, GemmIterN, _1>{}, Stride<_1, _0, _0>{}));

    CUTE_UNROLL
    for (; prefetch_k < prefetch_dist; prefetch_k++) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    for (int k_tile = k_start_idx; k_tile < k_tile_count;
         k_tile++, prefetch_k++) {
      barrier_arrive(barrier_scope);
      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA);
      copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBrB);
      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }
      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);
      cute::gemm<false>(
          mma,
          make_zip_tensor(tCrA, scaleA, scale_m_offsets, scale_ak_offsets),
          make_zip_tensor(tCrB, scaleB, scale_n_offsets, scale_bk_offsets),
          tCrD);
      barrier_wait(barrier_scope);
    }
  } else {
    // MX BLOCK scale path (see moe_gemm_scaled for full commentary).
    namespace coll = cutlass::gemm::collective;
    using ElementScaleA = typename SATensor::element_type;
    using ElementScaleB = typename SBTensor::element_type;

    constexpr int MMA_K = get<2>(typename TiledMMA::Shape_MNK{});
    constexpr int GroupK = CfgGroupK;

    using GemmIterM = Int<decltype(size<1>(tCrA.shape()))::value>;
    using GemmIterN = Int<decltype(size<1>(tCrB.shape()))::value>;
    using GemmIterK = Int<decltype(size<2>(tCrB.shape()))::value>;

    const int l_coord = 0;
    constexpr int k_reload_factor = cute::max(GroupK / BLK_K, 1);

    auto [tiled_copy_scaleA, copy_iter_scaleA, fragment_scaleA] =
        coll::make_scaled_copy<void, ElementScaleA, SG_M, SG_K, GroupK>(
            mAscale, m_coord, l_coord, k_tile_count);
    auto [tiled_copy_scaleB, copy_iter_scaleB, fragment_scaleB] =
        coll::make_scaled_copy<void, ElementScaleB, SG_N, SG_K, GroupK>(
            mBscale, n_coord, l_coord, k_tile_count);
    auto [scale_m_offsets, scale_n_offsets, scale_ak_offsets,
          scale_bk_offsets] =
        coll::make_scaled_offsets<
            GemmIterM::value, GemmIterN::value, GemmIterK::value, MMA_K, GroupK,
            typename decltype(tiled_copy_scaleA)::BlockShape,
            typename decltype(tiled_copy_scaleB)::BlockShape>();
    auto [tiled_prefetch_scaleA, prefetch_iter_scaleA] =
        coll::make_scaled_prefetch<decltype(tiled_copy_scaleA), SG_M, SG_K,
                                   GroupK>(tiled_copy_scaleA, m_coord, l_coord,
                                           k_tile_count);
    auto [tiled_prefetch_scaleB, prefetch_iter_scaleB] =
        coll::make_scaled_prefetch<decltype(tiled_copy_scaleB), SG_N, SG_K,
                                   GroupK>(tiled_copy_scaleB, n_coord, l_coord,
                                           k_tile_count);

    using scaleA_vec_t =
        intel::vector_t<ElementScaleA, decltype(size(fragment_scaleA))::value>;
    using scaleB_vec_t =
        intel::vector_t<ElementScaleB, decltype(size(fragment_scaleB))::value>;

    CUTE_UNROLL
    for (; prefetch_k < prefetch_dist; prefetch_k++) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      prefetch(tiled_prefetch_scaleA,
               prefetch_iter_scaleA(_, _, _, prefetch_k / k_reload_factor));
      prefetch(tiled_prefetch_scaleB,
               prefetch_iter_scaleB(_, _, _, prefetch_k / k_reload_factor));
    }

    for (int k_tile = k_start_idx; k_tile < k_tile_count;
         k_tile++, prefetch_k++) {
      barrier_arrive(barrier_scope);
      copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA);
      copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBrB);
      copy(tiled_copy_scaleA,
           copy_iter_scaleA(_, _, _, k_tile / k_reload_factor), fragment_scaleA);
      copy(tiled_copy_scaleB,
           copy_iter_scaleB(_, _, _, k_tile / k_reload_factor), fragment_scaleB);
      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        prefetch(tiled_prefetch_scaleA,
                 prefetch_iter_scaleA(_, _, _, prefetch_k / k_reload_factor));
        prefetch(tiled_prefetch_scaleB,
                 prefetch_iter_scaleB(_, _, _, prefetch_k / k_reload_factor));
      }
      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);
      Tensor scaleA = make_tensor(
          recast<scaleA_vec_t>(fragment_scaleA).data(),
          make_layout(Shape<_1, GemmIterM, _1>{}, Stride<_1, _0, _0>{}));
      Tensor scaleB = make_tensor(
          recast<scaleB_vec_t>(fragment_scaleB).data(),
          make_layout(Shape<_1, GemmIterN, _1>{}, Stride<_1, _0, _0>{}));
      cute::gemm<false>(
          mma,
          make_zip_tensor(tCrA, scaleA, scale_m_offsets, scale_ak_offsets),
          make_zip_tensor(tCrB, scaleB, scale_n_offsets, scale_bk_offsets),
          tCrD);
      barrier_wait(barrier_scope);
    }
  }
  reorder(tCrD, tCrD_final);
  copy(tiled_copy_d, tCrD_final, tCgD);
}

} // namespace MoE
