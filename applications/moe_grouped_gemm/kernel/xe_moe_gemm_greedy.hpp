/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/platform/platform.h"
#include "moe_grouped_gemm/collective/xe_moe_gemm_greedy.hpp"
#include <cute/util/compat.hpp>

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace MoE {
using namespace cute;

// Which M-bucket a tile uses; selects the mainloop MMA (LARGE/SMALL/TINY).
enum class TileId : int32_t { Large = 0, Small = 1, Tiny = 2 };

// ---------------------------------------------------------------------------
// Unified GREEDY MoE GEMM kernel. ONE implementation for both uniform-M and
// dynamic-M grouped GEMM; the ONLY difference is how the owning expert
// of a flat tile index is resolved, gated by `IsDynamicM`:
//   * UNIFORM M: constant tiles_per_expert, so (expert, within) is a closed-form
//     FastDivmod and all bases derive from `expert`.
//   * DYNAMIC M: M varies, so the owning expert is found by a FORWARD SCAN
//     (linear_idx only grows -> monotonic), fusing A/D/scale row prefix sums.
//
// GREEDY M split (both modes): peel LARGE tiles, then ONE leftover tile in the
// smallest bucket that fits (TINY / SMALL / LARGE). LARGE tile_m is a power of
// two so the peel is shifts/masks. LOOK-2 tail: if the last [LARGE,leftover]
// pair fits in two SMALL tiles at less padding, use those (SAME tile count,
// never TINY). Extents come from the MMA types (Config). TINY uses its own SG
// layout but the same WG thread count, so all variants share one nd_range.
// Dispatch on TileId (Large/Small/Tiny) to one of three mainloops.
// ---------------------------------------------------------------------------

// Operand-layout tensor builder. B inverts A/D because it is stored (N,K) with
// operand shape KxN. A/D: Row->(c,1), Col->(1,r); B: Row->(1,r), Col->(c,1).
template <class T, class Layout, bool IsBOperand = false>
CUTE_DEVICE auto make_greedy_tensor(T *ptr, int r, int c) {
  auto shape = make_shape(r, c);
  constexpr bool is_row = cute::is_same_v<Layout, cutlass::layout::RowMajor>;
  constexpr bool unit_first = IsBOperand ? is_row : !is_row;
  if constexpr (cute::is_subbyte_v<T>) {
    // Sub-byte types (mxfp4) need const pointers for 2D block loads.
    auto const_ptr = const_cast<T const *>(ptr);
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
// Plain (bf16) unified greedy kernel.
template <bool IsDynamicM,
          class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          class LayoutA, class LayoutB, class LayoutD,
          class MmaLarge, class MmaSmall, class MmaTiny,
          typename ElementA, typename ElementD>
CUTE_DEVICE void
MoEGEMMGreedy(const ElementA *Activations, const ElementA *Weights,
              ElementD *Outputs, const int32_t *M_per_expert,
              const int32_t num_experts, const int32_t N, const int32_t K,
              const int32_t M_uniform_param) {
  MmaLarge mma_large{};
  MmaSmall mma_small{};
  MmaTiny  mma_tiny{};

  // Tile geometry from the MMA types. LARGE tile_m must be a power of two.
  constexpr int32_t kLargeTileM = get<0>(decltype(mma_large.tile_mnk()){});
  constexpr int32_t kSmallTileM = get<0>(decltype(mma_small.tile_mnk()){});
  constexpr int32_t kTinyTileM  = get<0>(decltype(mma_tiny.tile_mnk()){});
  constexpr int32_t kTileN      = get<1>(decltype(mma_large.tile_mnk()){});
  static_assert((kLargeTileM & (kLargeTileM - 1)) == 0,
                "LARGE tile_m must be a power of two (greedy split uses shifts)");
  constexpr int32_t kLargeTileMLog2 = cute::log_2(uint32_t(kLargeTileM));
  // LOOK-2 tail threshold: last LARGE+leftover -> two SMALL when leftover <= this.
  constexpr int32_t kTailTwoSmallMax = 2 * kSmallTileM - kLargeTileM;
  const int32_t n_tiles = (N + kTileN - 1) / kTileN;
  const cutlass::FastDivmod n_divmod(n_tiles);

  // UNIFORM-M only: constant tiles_per_expert + total, decoded by divmod. M comes
  // in as a scalar param (host-computed) -- no device read of M_per_expert[0].
  const int32_t M_uniform = M_uniform_param;
  const int32_t uni_num_full = M_uniform >> kLargeTileMLog2;
  const int32_t uni_tail_rem = M_uniform - (uni_num_full << kLargeTileMLog2);
  const int32_t uni_m_tiles = uni_num_full + (uni_tail_rem > 0 ? 1 : 0);
  const int32_t tiles_per_expert = uni_m_tiles * n_tiles;
  const cutlass::FastDivmod tpe_divmod(tiles_per_expert > 0 ? tiles_per_expert : 1);
  const uint64_t total_tiles_uniform =
      uint64_t(num_experts) * uint64_t(tiles_per_expert);

  const uint64_t wg_id =
      uint64_t(BlockIdxX()) +
      uint64_t(BlockIdxY()) * uint64_t(GridDimX()) +
      uint64_t(BlockIdxZ()) * uint64_t(GridDimX()) * uint64_t(GridDimY());
  const uint64_t grid_size =
      uint64_t(GridDimX()) * uint64_t(GridDimY()) * uint64_t(GridDimZ());
  uint64_t linear_idx = wg_id;   // grid-stride cursor

  // UNIFORM-M MID-RUN CORE-ID FLIP invariants (computed once). full_waves =
  // waves where ALL cores have a tile; the flip covers only those (partial tail
  // unflipped) so the mirror always has a partner tile -- no per-tile bounds
  // check. All cores flip at the same wave (its midpoint). Unused in dynamic-M.
  const uint64_t flip_full_waves =
      grid_size > 0 ? total_tiles_uniform / grid_size : 0;
  const uint64_t flip_start_wave = flip_full_waves / 2;

  // DYNAMIC-M only: forward-scan cursor + fused A/D row prefix sum.
  int32_t scan_expert = 0;
  uint64_t scan_cum = 0;
  int64_t scan_cum_m_rows = 0;
  auto tiles_of = [&](int32_t e) -> uint64_t {
    int32_t mt = (M_per_expert[e] + kLargeTileM - 1) >> kLargeTileMLog2;
    return uint64_t(mt) * uint64_t(n_tiles);
  };

  auto make_A = [&](const ElementA *p, int32_t m) {
    return make_greedy_tensor<ElementA, LayoutA>(const_cast<ElementA *>(p), m, K);
  };
  auto make_B = [&](const ElementA *p) {
    return make_greedy_tensor<ElementA, LayoutB, /*IsBOperand=*/true>(
        const_cast<ElementA *>(p), N, K);
  };
  auto make_D = [&](ElementD *p, int32_t m) {
    return make_greedy_tensor<ElementD, LayoutD>(p, m, N);
  };
  int32_t prev_expert = -1;
  int32_t expert_m_rows = 0;
  int32_t num_full = 0;
  bool tail_two_small = false;
  auto activations_tensor = make_A(Activations, M_uniform);
  auto weights_tensor = make_B(Weights);
  auto outputs_tensor = make_D(Outputs, M_uniform);

  // Loop which goes over experts, m, n linearly (flat tile index, grid-strided).
  for (; ; linear_idx += grid_size) {
    // ---- Resolve owning expert + within-expert index (mode-dependent) ----
    int32_t expert, within;
    int64_t cumulative_m_rows;
    if constexpr (IsDynamicM) {
      while (scan_expert < num_experts &&
             linear_idx >= scan_cum + tiles_of(scan_expert)) {
        scan_cum += tiles_of(scan_expert);
        scan_cum_m_rows += M_per_expert[scan_expert];
        ++scan_expert;
      }
      if (scan_expert >= num_experts)
        break;
      expert = scan_expert;
      within = int32_t(linear_idx - scan_cum);
      cumulative_m_rows = scan_cum_m_rows;
    } else {
      if (linear_idx >= total_tiles_uniform)
        break;
      // MID-RUN CORE-ID FLIP: in the flipped region run the MIRROR core's tile
      // for this wave (src = grid-1-wg_id).
      const uint64_t wave = (linear_idx - wg_id) / grid_size;
      const bool flip = (wave >= flip_start_wave) & (wave < flip_full_waves);
      const uint64_t src = flip ? (grid_size - 1 - wg_id) : wg_id;
      tpe_divmod(expert, within, int32_t(src + wave * grid_size));
      cumulative_m_rows = int64_t(expert) * M_uniform;
    }

    // ---- EXPERT CHANGED: rebuild per-expert pointers + greedy split ----
    if (expert != prev_expert) {
      prev_expert = expert;
      expert_m_rows = IsDynamicM ? M_per_expert[expert] : M_uniform;
      num_full = expert_m_rows >> kLargeTileMLog2;
      const int32_t tail_rem = expert_m_rows - (num_full << kLargeTileMLog2);
      tail_two_small =
          (num_full >= 1) & (tail_rem > 0) & (tail_rem <= kTailTwoSmallMax);

      const int64_t byte_off_a =
          cumulative_m_rows * K * int64_t(cute::sizeof_bits_v<ElementA>) / 8;
      const int64_t byte_off_b = int64_t(expert) * K * N *
                                 int64_t(cute::sizeof_bits_v<ElementA>) / 8;
      activations_tensor = make_A(
          reinterpret_cast<const ElementA *>(
              reinterpret_cast<const uint8_t *>(Activations) + byte_off_a),
          expert_m_rows);
      weights_tensor = make_B(reinterpret_cast<const ElementA *>(
          reinterpret_cast<const uint8_t *>(Weights) + byte_off_b));
      outputs_tensor = make_D(Outputs + cumulative_m_rows * N, expert_m_rows);
    }

    // ---- Per-tile decode: split the within-expert index into (m_tile, n_tile)
    // (AlongN divmod), then pick this m_tile's bucket (TINY/SMALL/LARGE) and its
    // row offset m_off. Three cases:
    //   1. LOOK-2 tail active + this is one of the last two m_tiles -> SMALL
    //      (the [LARGE,leftover] pair was replaced by two SMALLs).
    //   2. a full LARGE tile (m_tile < num_full) -> LARGE at m_tile*kLargeTileM.
    //   3. the single leftover after the LARGEs -> smallest bucket that covers
    //      the remaining rows (TINY if <=kTinyTileM, else SMALL, else LARGE).
    int32_t m_tile, n_tile;
    n_divmod(m_tile, n_tile, within);

    int32_t m_off;
    TileId tile_id;
    if (tail_two_small && m_tile >= num_full - 1) {
      tile_id = TileId::Small;
      m_off = ((num_full - 1) << kLargeTileMLog2) +
              (m_tile == num_full ? kSmallTileM : 0);
    } else if (m_tile < num_full) {
      m_off = m_tile << kLargeTileMLog2;
      tile_id = TileId::Large;
    } else {
      m_off = num_full << kLargeTileMLog2;
      int32_t rem = expert_m_rows - m_off;
      tile_id = (rem <= kTinyTileM)  ? TileId::Tiny
                : (rem <= kSmallTileM) ? TileId::Small
                                       : TileId::Large;
    }

    auto tile_coord = make_coord(m_off, n_tile, _, 0);

    if (tile_id == TileId::Tiny)
      moe_gemm_greedy<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
          activations_tensor, weights_tensor, outputs_tensor, tile_coord, mma_tiny);
    else if (tile_id == TileId::Small)
      moe_gemm_greedy<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
          activations_tensor, weights_tensor, outputs_tensor, tile_coord, mma_small);
    else
      moe_gemm_greedy<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
          activations_tensor, weights_tensor, outputs_tensor, tile_coord, mma_large);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Block-scaled (fp8, mxfp8, mxfp4) unified greedy kernel. Scheduler identical to
// MoEGEMMGreedy; adds per-expert SCALE base pointers and dispatches to
// moe_gemm_scaled_greedy. ElementB/LayoutB explicit (mxfp4 = ColumnMajor B).
template <bool IsDynamicM,
          class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          class LayoutA, class LayoutB, class LayoutD,
          int CfgGroupN, int CfgGroupK,
          class MmaLarge, class MmaSmall, class MmaTiny,
          typename ElementA, typename ElementB, typename ElementS,
          typename ElementD>
CUTE_DEVICE void
MoEGEMMGreedyScaled(const ElementA *Activations, const ElementB *Weights,
                    const ElementS *ScalesA, const ElementS *ScalesB,
                    ElementD *Outputs, const int32_t *M_per_expert,
                    const int32_t num_experts, const int32_t N, const int32_t K,
                    const int32_t GroupN, const int32_t GroupK,
                    const int32_t M_uniform_param) {
  static_assert(!cute::is_void_v<ElementS>, "Use MoEGEMMGreedy for plain BF16");
  MmaLarge mma_large{};
  MmaSmall mma_small{};
  MmaTiny  mma_tiny{};

  constexpr int32_t kLargeTileM = get<0>(decltype(mma_large.tile_mnk()){});
  constexpr int32_t kSmallTileM = get<0>(decltype(mma_small.tile_mnk()){});
  constexpr int32_t kTinyTileM  = get<0>(decltype(mma_tiny.tile_mnk()){});
  constexpr int32_t kTileN      = get<1>(decltype(mma_large.tile_mnk()){});
  static_assert((kLargeTileM & (kLargeTileM - 1)) == 0,
                "LARGE tile_m must be a power of two (greedy split uses shifts)");
  constexpr int32_t kLargeTileMLog2 = cute::log_2(uint32_t(kLargeTileM));
  constexpr int32_t kTailTwoSmallMax = 2 * kSmallTileM - kLargeTileM;
  constexpr int kScaleAlign = 64;   // scale-surface padding (fixed HW alignment)
  const int32_t n_tiles = (N + kTileN - 1) / kTileN;
  const cutlass::FastDivmod n_divmod(n_tiles);

  // Scale surface geometry (see the non-greedy scaled kernel for full rationale).
  //   scale_k: BLOCK -> ceil(K/GroupK);  TENSOR -> BLK_K / MMA_K.
  //   scale_n / padded_scale_n: BLOCK -> ceil(N/GroupN) padded; TENSOR -> SG_N padded.
  constexpr int32_t kBlkK  = get<2>(decltype(mma_large.tile_mnk()){});
  constexpr int32_t kMmaK  = 256 / int32_t(cute::sizeof_bits_v<ElementA>);
  constexpr int32_t kScaleTensorK = (kBlkK + kMmaK - 1) / kMmaK;
  constexpr int32_t kBlkN     = get<1>(decltype(mma_large.tile_mnk()){});
  constexpr int32_t kSgNumsN  = get<2>(typename MmaLarge::ThrLayoutVMNK{}.shape());
  constexpr int32_t kSgN      = cute::ceil_div(kBlkN, kSgNumsN);
  constexpr int32_t kTensorPaddedN = (kSgN + kScaleAlign - 1) & ~(kScaleAlign - 1);

  const int32_t scale_k = (CfgGroupK > 0) ? ((K + GroupK - 1) / GroupK) : kScaleTensorK;
  const int32_t scale_n = (CfgGroupK > 0)
      ? ((N + (GroupN > 0 ? GroupN : 1) - 1) / (GroupN > 0 ? GroupN : 1))
      : kSgN;
  const int32_t padded_scale_n = (CfgGroupK > 0)
      ? ((scale_n + kScaleAlign - 1) & ~(kScaleAlign - 1))
      : kTensorPaddedN;

  // UNIFORM-M only.
  const int32_t M_uniform = M_uniform_param;  // host-computed scalar (no device read)
  const int32_t uni_num_full = M_uniform >> kLargeTileMLog2;
  const int32_t uni_tail_rem = M_uniform - (uni_num_full << kLargeTileMLog2);
  const int32_t uni_m_tiles = uni_num_full + (uni_tail_rem > 0 ? 1 : 0);
  const int32_t tiles_per_expert = uni_m_tiles * n_tiles;
  const cutlass::FastDivmod tpe_divmod(tiles_per_expert > 0 ? tiles_per_expert : 1);
  const uint64_t total_tiles_uniform =
      uint64_t(num_experts) * uint64_t(tiles_per_expert);
  const int32_t round_up_M_uniform = (M_uniform + kScaleAlign - 1) & ~(kScaleAlign - 1);

  const uint64_t wg_id =
      uint64_t(BlockIdxX()) +
      uint64_t(BlockIdxY()) * uint64_t(GridDimX()) +
      uint64_t(BlockIdxZ()) * uint64_t(GridDimX()) * uint64_t(GridDimY());
  const uint64_t grid_size =
      uint64_t(GridDimX()) * uint64_t(GridDimY()) * uint64_t(GridDimZ());
  uint64_t linear_idx = wg_id;   // grid-stride cursor

  // UNIFORM-M MID-RUN CORE-ID FLIP invariants (computed once). full_waves =
  // waves where ALL cores have a tile; the flip covers only those (partial tail
  // unflipped) so the mirror always has a partner tile -- no per-tile bounds
  // check. All cores flip at the same wave (its midpoint). Unused in dynamic-M.
  const uint64_t flip_full_waves =
      grid_size > 0 ? total_tiles_uniform / grid_size : 0;
  const uint64_t flip_start_wave = flip_full_waves / 2;

  // DYNAMIC-M only: forward scan + fused A/D + padded-scale row prefix sums.
  int32_t scan_expert = 0;
  uint64_t scan_cum = 0;
  int64_t scan_cum_m_rows = 0;
  int64_t scan_padded_scale_m_rows = 0;
  auto tiles_of = [&](int32_t e) -> uint64_t {
    int32_t mt = (M_per_expert[e] + kLargeTileM - 1) >> kLargeTileMLog2;
    return uint64_t(mt) * uint64_t(n_tiles);
  };

  auto make_A = [&](const ElementA *p, int32_t m) {
    return make_greedy_tensor<ElementA, LayoutA>(const_cast<ElementA *>(p), m, K);
  };
  auto make_B = [&](const ElementB *p) {
    return make_greedy_tensor<ElementB, LayoutB, /*IsBOperand=*/true>(
        const_cast<ElementB *>(p), N, K);
  };
  auto make_D = [&](ElementD *p, int32_t m) {
    return make_greedy_tensor<ElementD, LayoutD>(p, m, N);
  };
  int32_t prev_expert = -1;
  int32_t expert_m_rows = 0;
  int32_t num_full = 0;
  bool tail_two_small = false;
  auto activations_tensor = make_A(Activations, M_uniform);
  auto weights_tensor = make_B(Weights);
  auto outputs_tensor = make_D(Outputs, M_uniform);

  const ElementS *sA_base = ScalesA;
  const ElementS *sB_base = ScalesB;

  // Loop which goes over experts, m, n linearly (flat tile index, grid-strided).
  for (; ; linear_idx += grid_size) {
    int32_t expert, within;
    int64_t cumulative_m_rows;
    int64_t scaleA_row_base;    // ScaleA row prefix (padded) for this expert
    if constexpr (IsDynamicM) {
      while (scan_expert < num_experts &&
             linear_idx >= scan_cum + tiles_of(scan_expert)) {
        scan_cum += tiles_of(scan_expert);
        scan_cum_m_rows += M_per_expert[scan_expert];
        scan_padded_scale_m_rows +=
            (M_per_expert[scan_expert] + kScaleAlign - 1) & ~(kScaleAlign - 1);
        ++scan_expert;
      }
      if (scan_expert >= num_experts)
        break;
      expert = scan_expert;
      within = int32_t(linear_idx - scan_cum);
      cumulative_m_rows = scan_cum_m_rows;
      scaleA_row_base = scan_padded_scale_m_rows;
    } else {
      if (linear_idx >= total_tiles_uniform)
        break;
      // MID-RUN CORE-ID FLIP: in the flipped region run the MIRROR core's tile
      // for this wave (src = grid-1-wg_id).
      const uint64_t wave = (linear_idx - wg_id) / grid_size;
      const bool flip = (wave >= flip_start_wave) & (wave < flip_full_waves);
      const uint64_t src = flip ? (grid_size - 1 - wg_id) : wg_id;
      tpe_divmod(expert, within, int32_t(src + wave * grid_size));
      cumulative_m_rows = int64_t(expert) * M_uniform;
      scaleA_row_base = int64_t(expert) * round_up_M_uniform;
    }

    if (expert != prev_expert) {
      prev_expert = expert;
      expert_m_rows = IsDynamicM ? M_per_expert[expert] : M_uniform;
      num_full = expert_m_rows >> kLargeTileMLog2;
      const int32_t tail_rem = expert_m_rows - (num_full << kLargeTileMLog2);
      tail_two_small =
          (num_full >= 1) & (tail_rem > 0) & (tail_rem <= kTailTwoSmallMax);

      const int64_t byte_off_a =
          cumulative_m_rows * K * int64_t(cute::sizeof_bits_v<ElementA>) / 8;
      const int64_t byte_off_b = int64_t(expert) * K * N *
                                 int64_t(cute::sizeof_bits_v<ElementB>) / 8;
      activations_tensor = make_A(
          reinterpret_cast<const ElementA *>(
              reinterpret_cast<const uint8_t *>(Activations) + byte_off_a),
          expert_m_rows);
      weights_tensor = make_B(reinterpret_cast<const ElementB *>(
          reinterpret_cast<const uint8_t *>(Weights) + byte_off_b));
      outputs_tensor = make_D(Outputs + cumulative_m_rows * N, expert_m_rows);

      sA_base = ScalesA + scaleA_row_base * scale_k;
      sB_base = ScalesB + int64_t(expert) * padded_scale_n * scale_k;
    }

    // Per-tile decode: same greedy bucket pick as MoEGEMMGreedy (see there) --
    // (m_tile,n_tile) via AlongN divmod, then TINY/SMALL/LARGE + row offset m_off.
    int32_t m_tile, n_tile;
    n_divmod(m_tile, n_tile, within);

    int32_t m_off;
    TileId tile_id;
    if (tail_two_small && m_tile >= num_full - 1) {
      tile_id = TileId::Small;
      m_off = ((num_full - 1) << kLargeTileMLog2) +
              (m_tile == num_full ? kSmallTileM : 0);
    } else if (m_tile < num_full) {
      m_off = m_tile << kLargeTileMLog2;
      tile_id = TileId::Large;
    } else {
      m_off = num_full << kLargeTileMLog2;
      int32_t rem = expert_m_rows - m_off;
      tile_id = (rem <= kTinyTileM)  ? TileId::Tiny
                : (rem <= kSmallTileM) ? TileId::Small
                                       : TileId::Large;
    }

    auto tile_coord = make_coord(m_off, n_tile, _, 0);

    const int32_t round_up_M_e =
        (expert_m_rows + kScaleAlign - 1) & ~(kScaleAlign - 1);
    auto sA = make_tensor(
        make_gmem_ptr(const_cast<ElementS *>(sA_base)),
        make_layout(make_shape(expert_m_rows, int(scale_k), 1),
                    make_stride(_1{}, round_up_M_e,
                                int64_t(round_up_M_e) * scale_k)));
    auto sB = make_tensor(
        make_gmem_ptr(const_cast<ElementS *>(sB_base)),
        make_layout(make_shape(int(scale_n), int(scale_k), 1),
                    make_stride(_1{}, padded_scale_n,
                                int64_t(padded_scale_n) * scale_k)));

    if (tile_id == TileId::Tiny)
      moe_gemm_scaled_greedy<CfgGroupN, CfgGroupK, GmemTiledCopyA, GmemTiledCopyB,
                             GmemTiledCopyD>(
          activations_tensor, weights_tensor, sA, sB, outputs_tensor,
          tile_coord, mma_tiny, GroupN, GroupK);
    else if (tile_id == TileId::Small)
      moe_gemm_scaled_greedy<CfgGroupN, CfgGroupK, GmemTiledCopyA, GmemTiledCopyB,
                             GmemTiledCopyD>(
          activations_tensor, weights_tensor, sA, sB, outputs_tensor,
          tile_coord, mma_small, GroupN, GroupK);
    else
      moe_gemm_scaled_greedy<CfgGroupN, CfgGroupK, GmemTiledCopyA, GmemTiledCopyB,
                             GmemTiledCopyD>(
          activations_tensor, weights_tensor, sA, sB, outputs_tensor,
          tile_coord, mma_large, GroupN, GroupK);
  }
}

} // namespace MoE
