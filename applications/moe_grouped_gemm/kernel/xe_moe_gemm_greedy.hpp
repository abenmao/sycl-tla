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

// Which M-bucket a tile uses; selects the mainloop MMA. Six buckets:
//   large_bucket / small_bucket serve normal experts (LARGE peel + the
//   {small_bucket, large_bucket} end of the leftover ladder).
//   tiny_expert_{large,medium,small,tiny}_bucket serve the single-tile
//   tiny-expert schedule (smallest that covers M); tiny_expert_small_bucket
//   is also the low rung of the normal-expert leftover ladder.
// All six tile_m extents come from the MMA types (Config).
enum class TileId : int32_t {
  Large = 0,
  Small = 1,
  TinyExpertLarge = 2,
  TinyExpertMedium = 3,
  TinyExpertSmall = 4,
  TinyExpertTiny = 5
};

// ---------------------------------------------------------------------------
// Unified GREEDY MoE GEMM kernel. ONE implementation for both uniform-M and
// dynamic-M; the ONLY difference is how a flat tile index's owning expert is
// resolved, gated by `IsDynamicM`:
//   * UNIFORM M: constant tiles_per_expert -> (expert, within) via FastDivmod.
//   * DYNAMIC M: M varies -> owning expert found by a FORWARD SCAN (linear_idx
//     only grows -> monotonic), fusing A/D/scale row prefix sums.
//
// GREEDY M split (both modes), two scheduling methods by expert size:
//   * NORMAL expert (M > kTinyExpertMax): peel large_bucket tiles, then ONE
//     leftover tile from the RESTRICTED ladder
//     {tiny_expert_small_bucket, small_bucket, large_bucket} -- the
//     tiny_expert_{large,medium,tiny}_bucket are NOT used for normal-expert
//     leftovers. large_bucket tile_m is a power of two so the peel is
//     shifts/masks. LOOK-2 tail: if the last [large_bucket,leftover] pair fits
//     in two small_bucket tiles at less padding, use those (SAME tile count,
//     always small_bucket -- never a tiny_expert bucket).
//   * TINY expert (M <= kTinyExpertMax): scheduled as ONE tile -- the smallest
//     tiny_expert bucket that covers M
//     ({tiny_expert_tiny_bucket, tiny_expert_small_bucket,
//       tiny_expert_medium_bucket, tiny_expert_large_bucket}); the single tile
//     is partial/padded.
// Extents come from the MMA types (Config). The tiny_expert buckets use their
// own SG layout but the same WG thread count, so all variants share one
// nd_range; dispatch on TileId to one of six mainloops.
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

// Number of M-tiles a single expert with `M` rows contributes, under the
// two-method greedy split. Must stay in lockstep with the per-tile decode.
//   * tiny expert (M <= tiny_max): ONE tile (the smallest tiny_expert bucket
//     that covers M -- see the per-tile decode).
//   * normal expert: floor(M/large_bucket) large_bucket + one leftover (if any)
CUTE_DEVICE inline int32_t
greedy_m_tiles(int32_t M, int32_t large_m, int32_t tiny_max) {
  if (M <= 0) {
    return 0;  // dropped expert (0 tokens) contributes no tiles
  }
  if (M <= tiny_max) {
    return 1;  // one tile per tiny expert (partial/padded)
  }
  const int32_t num_full = M / large_m;
  const int32_t tail_rem = M - num_full * large_m;
  return num_full + (tail_rem > 0 ? 1 : 0);
}

///////////////////////////////////////////////////////////////////////////////
// Plain (bf16) unified greedy kernel.
template <bool IsDynamicM,
          class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          class LayoutA, class LayoutB, class LayoutD,
          class MmaLarge, class MmaSmall,
          class MmaTinyExpertLarge, class MmaTinyExpertMedium,
          class MmaTinyExpertSmall, class MmaTinyExpertTiny,
          typename ElementA, typename ElementD>
CUTE_DEVICE void
MoEGEMMGreedy(const ElementA *Activations, const ElementA *Weights,
              ElementD *Outputs, const int32_t *M_per_expert,
              const int32_t num_experts, const int32_t N, const int32_t K,
              const int32_t M_uniform_param) {
  MmaLarge mma_large{};
  MmaSmall mma_small{};
  MmaTinyExpertLarge  mma_tiny_expert_large{};
  MmaTinyExpertMedium mma_tiny_expert_medium{};
  MmaTinyExpertSmall  mma_tiny_expert_small{};
  MmaTinyExpertTiny   mma_tiny_expert_tiny{};

  // Tile geometry from the MMA types. large_bucket tile_m must be a power of two.
  constexpr int32_t kLargeBucketM = get<0>(decltype(mma_large.tile_mnk()){});
  constexpr int32_t kSmallBucketM = get<0>(decltype(mma_small.tile_mnk()){});
  constexpr int32_t kTinyExpertLargeBucketM =
      get<0>(decltype(mma_tiny_expert_large.tile_mnk()){});
  constexpr int32_t kTinyExpertMediumBucketM =
      get<0>(decltype(mma_tiny_expert_medium.tile_mnk()){});
  constexpr int32_t kTinyExpertSmallBucketM =
      get<0>(decltype(mma_tiny_expert_small.tile_mnk()){});
  constexpr int32_t kTinyExpertTinyBucketM =
      get<0>(decltype(mma_tiny_expert_tiny.tile_mnk()){});
  constexpr int32_t kTileN      = get<1>(decltype(mma_large.tile_mnk()){});
  // A tiny expert (M <= this) uses the single-tile tiny_expert schedule.
  constexpr int32_t kTinyExpertMax = 128;
  static_assert((kLargeBucketM & (kLargeBucketM - 1)) == 0,
                "large_bucket tile_m must be a power of two (greedy split uses shifts)");
  constexpr int32_t kLargeBucketMLog2 = cute::log_2(uint32_t(kLargeBucketM));
  // LOOK-2 tail threshold: last large_bucket+leftover -> two small_bucket when
  // leftover <= this.
  constexpr int32_t kTailTwoSmallMax = 2 * kSmallBucketM - kLargeBucketM;
  const int32_t n_tiles = (N + kTileN - 1) / kTileN;
  const cutlass::FastDivmod n_divmod(n_tiles);

  // UNIFORM-M only: constant tiles_per_expert + total, decoded by divmod. M comes
  // in as a scalar param (host-computed) -- no device read of M_per_expert[0].
  const int32_t M_uniform = M_uniform_param;
  const int32_t uni_m_tiles = greedy_m_tiles(
      M_uniform, kLargeBucketM, kTinyExpertMax);
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
  // waves where ALL cores have a tile; the flip covers only those, so the mirror
  // always has a partner tile (no per-tile bounds check). Unused in dynamic-M.
  const uint64_t flip_full_waves =
      grid_size > 0 ? total_tiles_uniform / grid_size : 0;
  const uint64_t flip_start_wave = flip_full_waves / 2;

  // DYNAMIC-M only: forward-scan cursor + fused A/D row prefix sum.
  int32_t scan_expert = 0;
  uint64_t scan_cum = 0;
  int64_t scan_cum_m_rows = 0;
  auto tiles_of = [&](int32_t e) -> uint64_t {
    int32_t mt = greedy_m_tiles(M_per_expert[e], kLargeBucketM, kTinyExpertMax);
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
  bool is_tiny_expert = false;
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
      is_tiny_expert = (expert_m_rows <= kTinyExpertMax);
      num_full = expert_m_rows >> kLargeBucketMLog2;
      const int32_t tail_rem = expert_m_rows - (num_full << kLargeBucketMLog2);
      // LOOK-2 tail is a NORMAL-expert-only optimization (never for tiny).
      tail_two_small = (!is_tiny_expert) & (num_full >= 1) & (tail_rem > 0) &
                       (tail_rem <= kTailTwoSmallMax);

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

    // ---- Per-tile decode: split within-expert index into (m_tile, n_tile),
    // then pick this m_tile's bucket + row offset m_off by scheduling method:
    //   TINY EXPERT (M<=kTinyExpertMax): the ONE tile -> smallest tiny_expert
    //     bucket that covers M, at m_off 0 (partial/padded).
    //   NORMAL EXPERT:
    //     1. LOOK-2 tail active + last two m_tiles -> small_bucket (never a
    //        tiny_expert bucket).
    //     2. full large_bucket tile (m_tile < num_full) at m_tile*large_bucket.
    //     3. single leftover -> RESTRICTED ladder
    //        (tiny_expert_small_bucket, small_bucket, else large_bucket).
    int32_t m_tile, n_tile;
    n_divmod(m_tile, n_tile, within);

    int32_t m_off;
    TileId tile_id;
    if (is_tiny_expert) {
      m_off = 0;
      tile_id = (expert_m_rows <= kTinyExpertTinyBucketM)   ? TileId::TinyExpertTiny
                : (expert_m_rows <= kTinyExpertSmallBucketM)  ? TileId::TinyExpertSmall
                : (expert_m_rows <= kTinyExpertMediumBucketM) ? TileId::TinyExpertMedium
                                                              : TileId::TinyExpertLarge;
    } else if (tail_two_small && m_tile >= num_full - 1) {
      tile_id = TileId::Small;
      m_off = ((num_full - 1) << kLargeBucketMLog2) +
              (m_tile == num_full ? kSmallBucketM : 0);
    } else if (m_tile < num_full) {
      m_off = m_tile << kLargeBucketMLog2;
      tile_id = TileId::Large;
    } else {
      m_off = num_full << kLargeBucketMLog2;
      int32_t rem = expert_m_rows - m_off;
      tile_id = (rem <= kTinyExpertSmallBucketM) ? TileId::TinyExpertSmall
                : (rem <= kSmallBucketM)          ? TileId::Small
                                                  : TileId::Large;
    }

    auto tile_coord = make_coord(m_off, n_tile, _, 0);

    switch (tile_id) {
    case TileId::TinyExpertTiny:
      moe_gemm_greedy<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
          activations_tensor, weights_tensor, outputs_tensor, tile_coord,
          mma_tiny_expert_tiny);
      break;
    case TileId::TinyExpertSmall:
      moe_gemm_greedy<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
          activations_tensor, weights_tensor, outputs_tensor, tile_coord,
          mma_tiny_expert_small);
      break;
    case TileId::TinyExpertMedium:
      moe_gemm_greedy<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
          activations_tensor, weights_tensor, outputs_tensor, tile_coord,
          mma_tiny_expert_medium);
      break;
    case TileId::TinyExpertLarge:
      moe_gemm_greedy<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
          activations_tensor, weights_tensor, outputs_tensor, tile_coord,
          mma_tiny_expert_large);
      break;
    case TileId::Small:
      moe_gemm_greedy<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
          activations_tensor, weights_tensor, outputs_tensor, tile_coord,
          mma_small);
      break;
    case TileId::Large:
    default:
      moe_gemm_greedy<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
          activations_tensor, weights_tensor, outputs_tensor, tile_coord,
          mma_large);
      break;
    }
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
          class MmaLarge, class MmaSmall,
          class MmaTinyExpertLarge, class MmaTinyExpertMedium,
          class MmaTinyExpertSmall, class MmaTinyExpertTiny,
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
  MmaTinyExpertLarge  mma_tiny_expert_large{};
  MmaTinyExpertMedium mma_tiny_expert_medium{};
  MmaTinyExpertSmall  mma_tiny_expert_small{};
  MmaTinyExpertTiny   mma_tiny_expert_tiny{};

  constexpr int32_t kLargeBucketM = get<0>(decltype(mma_large.tile_mnk()){});
  constexpr int32_t kSmallBucketM = get<0>(decltype(mma_small.tile_mnk()){});
  constexpr int32_t kTinyExpertLargeBucketM =
      get<0>(decltype(mma_tiny_expert_large.tile_mnk()){});
  constexpr int32_t kTinyExpertMediumBucketM =
      get<0>(decltype(mma_tiny_expert_medium.tile_mnk()){});
  constexpr int32_t kTinyExpertSmallBucketM =
      get<0>(decltype(mma_tiny_expert_small.tile_mnk()){});
  constexpr int32_t kTinyExpertTinyBucketM =
      get<0>(decltype(mma_tiny_expert_tiny.tile_mnk()){});
  constexpr int32_t kTileN      = get<1>(decltype(mma_large.tile_mnk()){});
  constexpr int32_t kTinyExpertMax = 128;
  static_assert((kLargeBucketM & (kLargeBucketM - 1)) == 0,
                "large_bucket tile_m must be a power of two (greedy split uses shifts)");
  constexpr int32_t kLargeBucketMLog2 = cute::log_2(uint32_t(kLargeBucketM));
  constexpr int32_t kTailTwoSmallMax = 2 * kSmallBucketM - kLargeBucketM;
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
  const int32_t uni_m_tiles = greedy_m_tiles(
      M_uniform, kLargeBucketM, kTinyExpertMax);
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
  // waves where ALL cores have a tile; the flip covers only those, so the mirror
  // always has a partner tile (no per-tile bounds check). Unused in dynamic-M.
  const uint64_t flip_full_waves =
      grid_size > 0 ? total_tiles_uniform / grid_size : 0;
  const uint64_t flip_start_wave = flip_full_waves / 2;

  // DYNAMIC-M only: forward scan + fused A/D + padded-scale row prefix sums.
  int32_t scan_expert = 0;
  uint64_t scan_cum = 0;
  int64_t scan_cum_m_rows = 0;
  int64_t scan_padded_scale_m_rows = 0;
  auto tiles_of = [&](int32_t e) -> uint64_t {
    int32_t mt = greedy_m_tiles(M_per_expert[e], kLargeBucketM, kTinyExpertMax);
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
  bool is_tiny_expert = false;
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
      is_tiny_expert = (expert_m_rows <= kTinyExpertMax);
      num_full = expert_m_rows >> kLargeBucketMLog2;
      const int32_t tail_rem = expert_m_rows - (num_full << kLargeBucketMLog2);
      // LOOK-2 tail is a NORMAL-expert-only optimization (never for tiny).
      tail_two_small = (!is_tiny_expert) & (num_full >= 1) & (tail_rem > 0) &
                       (tail_rem <= kTailTwoSmallMax);

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

    // Per-tile decode: same two-method greedy pick as MoEGEMMGreedy (see there).
    // (m_tile,n_tile) via AlongN divmod; m_off is the tile's absolute row origin
    // -- the scale-A row = m_off + sub-group row, so the stacked tiny tiles index
    // their scale rows correctly with no special-casing here.
    int32_t m_tile, n_tile;
    n_divmod(m_tile, n_tile, within);

    int32_t m_off;
    TileId tile_id;
    if (is_tiny_expert) {
      m_off = 0;
      tile_id = (expert_m_rows <= kTinyExpertTinyBucketM)   ? TileId::TinyExpertTiny
                : (expert_m_rows <= kTinyExpertSmallBucketM)  ? TileId::TinyExpertSmall
                : (expert_m_rows <= kTinyExpertMediumBucketM) ? TileId::TinyExpertMedium
                                                              : TileId::TinyExpertLarge;
    } else if (tail_two_small && m_tile >= num_full - 1) {
      tile_id = TileId::Small;
      m_off = ((num_full - 1) << kLargeBucketMLog2) +
              (m_tile == num_full ? kSmallBucketM : 0);
    } else if (m_tile < num_full) {
      m_off = m_tile << kLargeBucketMLog2;
      tile_id = TileId::Large;
    } else {
      m_off = num_full << kLargeBucketMLog2;
      int32_t rem = expert_m_rows - m_off;
      tile_id = (rem <= kTinyExpertSmallBucketM) ? TileId::TinyExpertSmall
                : (rem <= kSmallBucketM)          ? TileId::Small
                                                  : TileId::Large;
    }

    auto tile_coord = make_coord(m_off, n_tile, _, 0);

    const int32_t round_up_M_e =
        (expert_m_rows + kScaleAlign - 1) & ~(kScaleAlign - 1);
    // Padded M extent (round_up_M_e) for DWord-aligned MXFP scale load.
    auto sA = make_tensor(
        make_gmem_ptr(const_cast<ElementS *>(sA_base)),
        make_layout(make_shape(round_up_M_e, int(scale_k), 1),
                    make_stride(_1{}, round_up_M_e,
                                int64_t(round_up_M_e) * scale_k)));
    auto sB = make_tensor(
        make_gmem_ptr(const_cast<ElementS *>(sB_base)),
        make_layout(make_shape(int(scale_n), int(scale_k), 1),
                    make_stride(_1{}, padded_scale_n,
                                int64_t(padded_scale_n) * scale_k)));

    switch (tile_id) {
    case TileId::TinyExpertTiny:
      moe_gemm_scaled_greedy<CfgGroupN, CfgGroupK, GmemTiledCopyA, GmemTiledCopyB,
                             GmemTiledCopyD>(
          activations_tensor, weights_tensor, sA, sB, outputs_tensor,
          tile_coord, mma_tiny_expert_tiny, GroupN, GroupK);
      break;
    case TileId::TinyExpertSmall:
      moe_gemm_scaled_greedy<CfgGroupN, CfgGroupK, GmemTiledCopyA, GmemTiledCopyB,
                             GmemTiledCopyD>(
          activations_tensor, weights_tensor, sA, sB, outputs_tensor,
          tile_coord, mma_tiny_expert_small, GroupN, GroupK);
      break;
    case TileId::TinyExpertMedium:
      moe_gemm_scaled_greedy<CfgGroupN, CfgGroupK, GmemTiledCopyA, GmemTiledCopyB,
                             GmemTiledCopyD>(
          activations_tensor, weights_tensor, sA, sB, outputs_tensor,
          tile_coord, mma_tiny_expert_medium, GroupN, GroupK);
      break;
    case TileId::TinyExpertLarge:
      moe_gemm_scaled_greedy<CfgGroupN, CfgGroupK, GmemTiledCopyA, GmemTiledCopyB,
                             GmemTiledCopyD>(
          activations_tensor, weights_tensor, sA, sB, outputs_tensor,
          tile_coord, mma_tiny_expert_large, GroupN, GroupK);
      break;
    case TileId::Small:
      moe_gemm_scaled_greedy<CfgGroupN, CfgGroupK, GmemTiledCopyA, GmemTiledCopyB,
                             GmemTiledCopyD>(
          activations_tensor, weights_tensor, sA, sB, outputs_tensor,
          tile_coord, mma_small, GroupN, GroupK);
      break;
    case TileId::Large:
    default:
      moe_gemm_scaled_greedy<CfgGroupN, CfgGroupK, GmemTiledCopyA, GmemTiledCopyB,
                             GmemTiledCopyD>(
          activations_tensor, weights_tensor, sA, sB, outputs_tensor,
          tile_coord, mma_large, GroupN, GroupK);
      break;
    }
  }
}

} // namespace MoE
