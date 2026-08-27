/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
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
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*!
  \file xe35_chunk_gated_delta_rule_gemm.hpp
  \brief CuTe GEMM helpers for the GDN attention kernels (cutlass::gdn::detail).

  Each helper computes C += op(A) * op(B) on Xe 2D-block loads and DPAS,
  accumulating into the caller's register fragment so calls chain (C += A1*B1 +
  A2*B2 ...) with no round trip through memory.

  One name letter per operand, in A B C order -- T = tensor in gmem, loaded tile
  by tile; S = sub-group fragment already in registers:
    gemm_TTS   A(gmem) * B(gmem) -> C(regs)
    gemm_STS   A(regs) * B(gmem) -> C(regs)
    gemm_TSS   A(gmem) * B(regs) -> C(regs)

  Suffixes compose onto those:
    _k_multi     pre-scales each k-slice of A by a per-lane float from SLM
                 (diagonal scaling in compute_wu / fwd_o).
    _shareB      two GEMMs share one B operand in one k-loop, so B is loaded
                 and reordered once per k-tile.
    _multi_tile  sliced by a caller-supplied tile_local_id rather than the
                 WG-local id -- for WGs hosting several independent MMA tiles,
                 which may run different numbers of GEMMs.
    _sg          one DPAS tile per call, sub-group-scoped end to end.

  Only the unsuffixed helpers carry the 3-tiles-ahead prefetch pipeline and the
  k-loop WG barriers; _multi_tile and _sg have neither, by design -- a WG whose
  tiles are independent must not synchronize them.
*/

#pragma once

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace cutlass::gdn::detail {

using namespace cute;

template <
    class ATensor,
    class BTensor,
    class SGCTensor,
    class TiledMMA>
/* gemm_TTS: C += A(gmem, M×K) * B(gmem, N×K)^T  -- both operands from global memory. */
CUTE_DEVICE void gemm_TTS(
    ATensor const& A,  // (M,K)
    BTensor const& B,  // (N,K)
    SGCTensor& tCrC,   // (M,N)
    int wg_m,          // m tile start id
    int wg_n,          // n tile start id
    TiledMMA const& mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gA = local_tile(
      cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)

  auto copy_a = get_block_2d_copy_A<void>(mma, A);
  auto copy_b = get_block_2d_copy_B<void>(mma, B);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;

  constexpr auto barrier_scope = SPIRVScope::ScopeWorkgroup;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist && k_tile_prefetch < k_tile_count;
      k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }
}

template <
    class ASGCTensor,
    class BTensor,
    class CSGCTensor,
    class TiledMMA>
/* gemm_STS: C += A(regs, M×K) * B(gmem, N×K)^T  -- A is reused from a prior gemm_TTS call. */
CUTE_DEVICE void gemm_STS(
    ASGCTensor const& tCrA,  // (M,K)
    BTensor const& B,        // (N,K)
    CSGCTensor& tCrC,        // (M,N)
    int wg_m,                // m tile start id
    int wg_n,                // n tile start id
    TiledMMA const& mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();

  Tensor cB = make_identity_tensor(B.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)

  auto copy_b = get_block_2d_copy_B<void>(mma, B);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;

  constexpr auto barrier_scope = SPIRVScope::ScopeWorkgroup;

  int k_tile_count = ceil_div(shape<1>(B), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist && k_tile_prefetch < k_tile_count;
      k_tile_prefetch++) {
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tBrB, tCrB);

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }
}

template <
    class ATensor,
    class BSGCTensor,
    class CSGCTensor,
    class TiledMMA>
/* gemm_TSS: C += A(gmem, M×K) * B(regs, N×K)^T  -- B is reused from a prior call. */
CUTE_DEVICE void gemm_TSS(
    ATensor const& A,        // (M,K)
    BSGCTensor const& tCrB,  // (N,K)
    CSGCTensor& tCrC,        // (M,N)
    int wg_m,                // m tile start id
    int wg_n,                // n tile start id
    TiledMMA const& mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gA = local_tile(
      cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)

  auto copy_a = get_block_2d_copy_A<void>(mma, A);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);

  auto prefetch_a = make_block_2d_prefetch(copy_a);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);

  const int prefetch_dist = 3;

  constexpr auto barrier_scope = SPIRVScope::ScopeWorkgroup;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist && k_tile_prefetch < k_tile_count;
      k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }
}

template <
    class ATensor,
    class BTensor,
    class SGCTensor,
    class TiledMMA>
/* gemm_TTS_k_multi: C += diag(K_multi[k]) * A(gmem) * B(gmem)^T
 * Before each DPAS call the corresponding k-slice of A is element-wise
 * scaled by K_multi[k_tile * tile_k + k * 16 + lane_id], which lets the
 * caller fold a per-token diagonal weight (e.g. exp(a)*beta) into the
 * matrix multiply without a separate pass. */
CUTE_DEVICE void gemm_TTS_k_multi(
    ATensor const& A,  // (M,K)
    BTensor const& B,  // (N,K)
    SGCTensor& tCrC,   // (M,N)
    int wg_m,          // m tile start id
    int wg_n,          // n tile start id
    TiledMMA const& mma,
    float* K_multi) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();
  auto sg = item.get_sub_group();
  int sg_local_id = sg.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gA = local_tile(
      cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)

  auto copy_a = get_block_2d_copy_A<void>(mma, A);
  auto copy_b = get_block_2d_copy_B<void>(mma, B);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;

  constexpr auto barrier_scope = SPIRVScope::ScopeWorkgroup;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist && k_tile_prefetch < k_tile_count;
      k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    using TA = typename ATensor::element_type;
    Tensor A_frag = make_tensor<TA>(tCrA.layout());
    static constexpr auto I = decltype(size<0>(A_frag))::value;
    static constexpr auto J = decltype(size<1>(A_frag))::value;
    static constexpr auto K = decltype(size<2>(A_frag))::value;
    static constexpr int mma_K = 16;

    CUTE_UNROLL
    for (int k = 0; k < K; ++k) {
      float scale = K_multi[k_tile * get<2>(wg_tile) + k * mma_K + sg_local_id];
      CUTE_UNROLL
      for (int e = 0; e < I * J; ++e) {
        tCrA[k * I * J + e] =
            static_cast<TA>(static_cast<float>(tCrA[k * I * J + e]) * scale);
      }
    }

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }
}

/* Aliasing a 2D-copy fragment onto an MMA fragment's registers is only valid
 * while the two TV-layouts agree once mode nesting is flattened away. That is
 * true by construction while the copy comes from the same MMA, but a different
 * atom choice would break it silently and the DPAS would read garbage. */
template <class MMAFrag, class CopyFrag>
CUTE_DEVICE void assert_aliasable_frag(MMAFrag const& mma_frag,
                                       CopyFrag const& copy_frag) {
  static_assert(
      is_same_v<decltype(coalesce(project_strides(mma_frag.tv_layout()))),
                decltype(coalesce(project_strides(copy_frag.tv_layout())))>,
      "MMA and 2D-copy fragments disagree on their coalesced TV-layout: "
      "cannot alias their registers, use reorder() instead");
  static_assert(cosize_v<decltype(copy_frag.layout())> <=
                    cosize_v<decltype(mma_frag.layout())>,
                "2D-copy fragment is larger than the MMA fragment it aliases: "
                "the load would write past the MMA fragment's registers");
}

/* In-place C = -C on a register fragment: block forward substitution ends
 * every block with a sign flip. */
template <class SGCTensor>
CUTE_DEVICE void negate_frag(SGCTensor& tCrC) {
  CUTE_UNROLL
  for (int i = 0; i < tCrC.size(); ++i) {
    tCrC(i) *= -1.0f;
  }
}

/* A fragment is sub-viewable only while it is one compact run in row order; a
 * gap would make sub_frag() below a copy rather than a view. */
template <class Frag>
CUTE_DEVICE void assert_compact_frag(Frag const&) {
  using Layout_t = decltype(declval<Frag>().layout());
  static_assert(
      is_same_v<decltype(coalesce(declval<Layout_t>())),
                Layout<Int<cosize_v<Layout_t>>, _1>>,
      "fragment is not one compact run: a sub-block of it is not contiguous, "
      "so it cannot be viewed without a copy");
}

/* Sub-block (m,k) of a larger A fragment: a view aliasing `outer`'s registers
 * with `blk`'s TV-layout (shape only), so gemm()/reorder() see a fragment. */
template <class OuterFrag, class BlockFrag, class M, class K>
CUTE_DEVICE auto sub_frag(OuterFrag& outer, BlockFrag const& blk, M m, K k) {
  assert_compact_frag(outer);
  assert_compact_frag(blk);
  auto outer_layout = outer.layout();
  auto blk_layout = blk.layout();
  /* Both are (atom row, m iteration, k iteration), so the sub-block starts
   * where its first iteration does and runs contiguously from there. */
  auto offset =
      outer_layout(_0{}, m * size<1>(blk_layout), k * size<2>(blk_layout));
  return make_subgroup_tensor(make_tensor(outer.data() + offset, blk_layout),
                              blk.tv_layout());
}

/* Store C fragment `src` to tile (i,j) of `cL`'s coordinate space, staging the
 * reorder through `stg`. `mn_tile` is the (M,N) tile, i.e. select<0,1>(mnk). */
template <class Atom, class ThrCopy, class CTensor, class Tile, class SrcFrag,
          class StgFrag, class I, class J>
CUTE_DEVICE void store_sg_tile(Atom const& atom, ThrCopy const& thr_copy,
                               CTensor const& cL, Tile const& mn_tile,
                               SrcFrag const& src, StgFrag& stg, I i, J j) {
  Tensor gD = local_tile(cL, mn_tile, make_coord(i, j));
  reorder(src, stg);
  copy(atom, stg, thr_copy.partition_D(gD));
}

/* Load (M,K) tile (m,k) of `cL` into A fragment `frag`, aliasing the copy layout
 * onto its registers. Payload prepared, else CSE folds two loads into one. */
template <class Atom, class ThrCopy, class CTensor, class Tile, class Frag,
          class M, class K>
CUTE_DEVICE void load_sg_tile_A(Atom const& atom, ThrCopy const& thr_copy,
                                CTensor const& cL, Tile const& mk_tile,
                                Frag& frag, M m, K k) {
  Tensor gA = local_tile(cL, mk_tile, make_coord(m, k));
  auto cpr = thr_copy.partition_sg_fragment_D(gA);
  assert_aliasable_frag(frag, cpr);
  Tensor alias = make_tensor(frag.data(), cpr.layout());
  auto payload = prepare_payloads(atom, thr_copy.partition_S(gA));
  copy(atom, payload, alias);
}

/* ---- Sub-group-scoped GEMM: one DPAS tile per call ------------------------
 * One sub-group drives one (blk_m, blk_n, blk_k) tile of its own matrix, so
 * there is no k-loop to pipeline and nothing to prefetch. Two consequences:
 *   - MMA and copies slice by the sub-group-local lane id (0..15); the
 *     work-group-global id would index the 16-thread layout out of range.
 *   - No barrier of any kind: one tile is a single k-tile, so there is nothing
 *     to fence, and a work-group barrier can deadlock when the sub-groups walk
 *     grid-stride streams of different length.
 *
 * The copy fragment loads straight into the MMA fragment's registers. Both
 * fragments come from this MMA, so they agree on which (M,K)/(N,K) coordinate
 * each lane's value holds and differ only in mode nesting; the copy->MMA
 * reorder() is therefore an identity, and aliasing skips it along with a second
 * set of GRFs. assert_aliasable_frag enforces the precondition.
 *
 * Tile ids are generic integers: a cute::Int<> folds the block offset into the
 * 2D-copy descriptor, a plain int leaves it as runtime address arithmetic. */

/* C += A(regs) * B(gmem)^T on one tile. */
template <
    class ASGCTensor,
    class BTensor,
    class CSGCTensor,
    class BlkN,
    class BlkK,
    class TiledMMA>
CUTE_DEVICE void gemm_STS_sg(
    ASGCTensor const& tCrA,  // (M,K)
    BTensor const& B,        // (N,K)
    CSGCTensor& tCrC,        // (M,N)
    BlkN blk_n,              // n tile id
    BlkK blk_k,              // k tile id
    TiledMMA const& mma) {
  auto sg = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_sub_group();
  int sg_local_id = sg.get_local_linear_id();

  Tensor cB = make_identity_tensor(B.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(blk_n, blk_k));  // (BLK_N,BLK_K)

  auto copy_b = get_block_2d_copy_B<void>(mma, B);

  auto thr_mma = mma.get_slice(sg_local_id);
  auto thr_copy_b = copy_b.get_slice(sg_local_id);

  auto tCrB = thr_mma.partition_sg_fragment_B(gB);

  /* Alias the MMA fragment with the copy fragment's layout: same registers,
   * the nesting `copy()` dispatches on. */
  auto cprB = thr_copy_b.partition_sg_fragment_D(gB);
  assert_aliasable_frag(tCrB, cprB);

  Tensor tBrB = make_tensor(tCrB.data(), cprB.layout());

  copy(copy_b, thr_copy_b.partition_S(gB), tBrB);

  cute::gemm(mma, tCrA, tCrB, tCrC);
}

/* gemm_TTS_k_multi_tile: same as gemm_TTS_k_multi, but no prefetch pipeline
 * or k-loop barriers, and
 * sliced by a caller-supplied tile_local_id instead of the work-group's raw
 * local_id. Barrier-free, so it's safe to call from independent MMA tiles
 * sharing one work-group -- tiles may run different numbers of GEMMs or
 * finish at different times without deadlocking. */
template <
    class ATensor,
    class BTensor,
    class SGCTensor,
    class TiledMMA>
CUTE_DEVICE void gemm_TTS_k_multi_tile(
    ATensor const& A,  // (M,K)
    BTensor const& B,  // (N,K)
    SGCTensor& tCrC,   // (M,N)
    int wg_m,          // m tile start id
    int wg_n,          // n tile start id
    TiledMMA const& mma,
    float* K_multi,
    int tile_local_id) {  // local_id % size(mma): position within this MMA tile
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto sg = item.get_sub_group();
  int sg_local_id = sg.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gA = local_tile(
      cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)

  auto copy_a = get_block_2d_copy_A<void>(mma, A);
  auto copy_b = get_block_2d_copy_B<void>(mma, B);

  auto thr_mma = mma.get_slice(tile_local_id);
  auto thr_copy_a = copy_a.get_slice(tile_local_id);
  auto thr_copy_b = copy_b.get_slice(tile_local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

  using TA = typename ATensor::element_type;
  Tensor A_frag = make_tensor<TA>(tCrA.layout());
  static constexpr auto I = decltype(size<0>(A_frag))::value;
  static constexpr auto J = decltype(size<1>(A_frag))::value;
  static constexpr auto K = decltype(size<2>(A_frag))::value;
  static constexpr int mma_K = 16;

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++) {
    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    CUTE_UNROLL
    for (int k = 0; k < K; ++k) {
      float scale = K_multi[k_tile * get<2>(wg_tile) + k * mma_K + sg_local_id];
      CUTE_UNROLL
      for (int e = 0; e < I * J; ++e) {
        tCrA[k * I * J + e] =
            static_cast<TA>(static_cast<float>(tCrA[k * I * J + e]) * scale);
      }
    }

    cute::gemm(mma, tCrA, tCrB, tCrC);
  }
}

template <
    class A1Tensor,
    class A2Tensor,
    class BTensor,
    class C1SGCTensor,
    class C2SGCTensor,
    class TiledMMA>
/* gemm_TTS_shareB_multi_tile: two TTS GEMMs that share the B operand, fused
 * into one k-loop, with no prefetch/barriers -- same tiling contract as
 * gemm_TTS_k_multi_tile, see note above.
 *   C1 += A1(gmem, M×K) * B(gmem, N×K)^T
 *   C2 += A2(gmem, M×K) * B(gmem, N×K)^T
 * B is loaded, prefetched, and reordered once per k-tile and consumed by both
 * DPAS calls — replaces two back-to-back gemm_TTS calls that share B. Takes a
 * caller-supplied tile_local_id (0..size(mma)-1) instead of deriving it from
 * `this_work_item`, and has no internal WG barriers (all work is
 * register-private per lane), so WGs that hold several independent MMA tiles
 * can call it per tile and idle tiles can skip it entirely at the caller. */
CUTE_DEVICE void gemm_TTS_shareB_multi_tile(
    int tile_local_id,   // caller-supplied, 0..size(mma)-1
    A1Tensor const& A1,  // (M,K)
    A2Tensor const& A2,  // (M,K)
    BTensor const& B,    // (N,K)
    C1SGCTensor& tCrC1,  // (M,N)
    C2SGCTensor& tCrC2,  // (M,N)
    int wg_m,            // m tile start id
    int wg_n,            // n tile start id
    TiledMMA const& mma) {
  Tensor cA1 = make_identity_tensor(A1.shape());
  Tensor cA2 = make_identity_tensor(A2.shape());
  Tensor cB = make_identity_tensor(B.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gA1 = local_tile(
      cA1, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gA2 = local_tile(
      cA2, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)

  auto copy_a1 = get_block_2d_copy_A<void>(mma, A1);
  auto copy_a2 = get_block_2d_copy_A<void>(mma, A2);
  auto copy_b = get_block_2d_copy_B<void>(mma, B);

  auto thr_mma = mma.get_slice(tile_local_id);
  auto thr_copy_a1 = copy_a1.get_slice(tile_local_id);
  auto thr_copy_a2 = copy_a2.get_slice(tile_local_id);
  auto thr_copy_b = copy_b.get_slice(tile_local_id);

  auto tCrA1 = thr_mma.partition_sg_fragment_A(gA1(_, _, 0));
  auto tCrA2 = thr_mma.partition_sg_fragment_A(gA2(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tA1rA1 = thr_copy_a1.partition_sg_fragment_D(gA1(_, _, 0));
  auto tA2rA2 = thr_copy_a2.partition_sg_fragment_D(gA2(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tA1gA1 = thr_copy_a1.partition_S(gA1);
  Tensor tA2gA2 = thr_copy_a2.partition_S(gA2);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto prefetch_a1 = make_block_2d_prefetch(copy_a1);
  auto prefetch_a2 = make_block_2d_prefetch(copy_a2);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A1 = prefetch_a1.get_slice(tile_local_id);
  auto thr_prefetch_A2 = prefetch_a2.get_slice(tile_local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(tile_local_id);

  auto pA1gA1 = thr_prefetch_A1.partition_S(gA1);
  auto pA2gA2 = thr_prefetch_A2.partition_S(gA2);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;

  int k_tile_count = ceil_div(shape<1>(B), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist && k_tile_prefetch < k_tile_count;
      ++k_tile_prefetch) {
    prefetch(prefetch_a1, pA1gA1(_, _, _, k_tile_prefetch));
    prefetch(prefetch_a2, pA2gA2(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  /* No WG barriers here: the copy/reorder/DPAS are all register-private per
   * lane, so the former split barrier was pure over-sync. */
  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    copy(copy_a1, tA1gA1(_, _, _, k_tile), tA1rA1);
    copy(copy_a2, tA2gA2(_, _, _, k_tile), tA2rA2);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a1, pA1gA1(_, _, _, k_tile_prefetch));
      prefetch(prefetch_a2, pA2gA2(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tA1rA1, tCrA1);
    reorder(tA2rA2, tCrA2);
    reorder(tBrB, tCrB);

    cute::gemm(mma, tCrA1, tCrB, tCrC1);
    cute::gemm(mma, tCrA2, tCrB, tCrC2);
  }
}

}  // namespace cutlass::gdn::detail
