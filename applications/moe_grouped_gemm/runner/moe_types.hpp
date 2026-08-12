/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Kernel-free shared declarations for the MoE grouped-GEMM benchmark:
           the per-dtype Config structs, ScaleKind, the host scale generator
           (fill_flat_scales), the scale-surface packer (scale_surface_geom /
           pack_moe_scales), and the client-facing VendorTensorMapping.
*/

#pragma once

#include <cute/tensor.hpp>
#include "cutlass/numeric_types.h"   // bfloat16_t, float_e4m3_t, float_ue8m0_t, ...
#include "cutlass/layout/matrix.h"   // RowMajor / ColumnMajor
#include "cutlass/platform/platform.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/device_memory.h"
// kBlockScaleAlign / kTensorScaleK / kTensorPaddedScaleN — shared with the
// device kernels so host padding and kernel strides cannot drift.
#include "moe_grouped_gemm/moe_scale_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cutlass::moe {

using namespace cute;

// Round up to kBlockScaleAlign (or a custom alignment).
inline int round_up_align(int v, int align = kBlockScaleAlign) {
  return ((v + align - 1) / align) * align;
}

// Scale layout kind for a config.
//   Plain  : no scales — plain BF16 (the kernel takes the non-scaled path).
//   Block  : per-row, K-blocked (and N-blocked) scales (MX style).
//   Tensor : a single global scale per operand tensor (broadcast).
enum class ScaleKind { Plain, Block, Tensor };

// Client-facing tensor mapping. The client builds it on the host (owning the
// host scale grids + problem description) and also owns the A/B/D device
// buffers, filling them and pointing these fields at them before the launch —
// for the benchmark that happens in MoEBenchmarkRunner::run's build_inputs
// (benchmarks/applications/01_grouped_gemm/). Pointers only — no cute / kernel
// types — so both TUs can name it.
template <class ElementA, class ElementScaleIn, class ElementD>
struct VendorTensorMapping {
  // Dtype-independent problem description first, so these fields sit at the
  // same offsets in every instantiation
  const int            *experts_token_count = nullptr; // host, length E
  // The same counts on the device — the variable-M kernel's M_per_group, which it
  // reads per workgroup to find its expert and row offset. Allocated and uploaded
  // by the client alongside A/B/D; unused by the uniform-M (double-buffer) path,
  // which takes a scalar M instead.
  const int32_t        *experts_token_count_device = nullptr; // device, length E
  int                   num_experts         = 0;
  int                   N                   = 0;
  int                   K                   = 0;
  const ElementA       *scatter_tokens      = nullptr; // [T, K]    (device; provider-filled)
  const ElementA       *experts_weight      = nullptr; // [E, N, K]  (device; provider-filled)
  // Logical (unpadded) host scale grids. Only read by verify, which compares
  // against these rather than the packed surface below.
  const ElementScaleIn *per_token_scale     = nullptr; // host, rank per format
  const ElementScaleIn *experts_scale       = nullptr; // host, rank per format
  // Packed, padded DEVICE scale surfaces in exactly the layout the kernel
  // reads — allocated, packed and uploaded by the client, which owns the Config
  // (hence ScaleKind and the group sizes) and derives the padding from
  // moe_scale_layout.hpp. Null for ScaleKind::Plain. See pack_moe_scales() and
  // MoEBenchmarkRunner::run's build_inputs.
  const ElementScaleIn *packed_scale_a      = nullptr; // device, padded per expert
  const ElementScaleIn *packed_scale_b      = nullptr; // device, padded per expert
  ElementD             *y                   = nullptr; // device [T, N]
};

template <class Config, class = void>
struct ConfigScale { using type = float; };

template <class Config>
struct ConfigScale<Config,
                   cutlass::platform::enable_if_t<
                       !cutlass::platform::is_void<typename Config::ElementScale>::value>> {
  using type = typename Config::ElementScale;
};

template <class Config>
using ScaleStoreFor =
    cutlass::platform::conditional_t<Config::scale_kind == ScaleKind::Plain, float,
                                     typename ConfigScale<Config>::type>;

// Fill a flat host scale buffer with random dequant-scale values.
template <class ElementScaleStore>
void fill_flat_scales(std::vector<ElementScaleStore> &buf, uint64_t seed, bool constant) {
  // Span 10 octaves to exercise varied per-block exponents without E8M0 underflow.
  const float scale_max = 0.25f;                    // == 2^-2
  const float scale_min = constant ? scale_max      // constant fill: single value
                                    : scale_max / 1024.0f;  // 2^-12 → 10 octaves
  cutlass::reference::host::BlockFillRandomUniform(
      buf.data(), buf.size(), seed, double(scale_max), double(scale_min));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Scale packing (host) + upload.
//
// Repacks the logical (unpadded) host scale grids into the padded, MN-major
// surface the kernel's 2D block-scale load reads, then uploads it. Owned by the
// client, which has the Config — and so the ScaleKind, the group sizes, and the
// padding constants from moe_scale_layout.hpp that the kernels also build their
// strides from. Nothing re-derives this geometry on the far side.
//
//   scale_k_store  : surface K height (rows per MN entry)
//   padded_scale_n : per-expert N stripe width in the scale-B surface
///////////////////////////////////////////////////////////////////////////////////////////////////

// Padded device scale-surface geometry for one Config and problem shape.
struct ScaleSurfaceGeom {
  int scale_k_store;
  int padded_scale_n;
};

template <class Config>
ScaleSurfaceGeom scale_surface_geom(int N, int K) {
  constexpr bool kIsTensor = (Config::scale_kind == ScaleKind::Tensor);
  // Tensor geometry is fixed by the BDPAS scale layout: height kTensorScaleK
  // follows from MMA_K = 256 / sizeof_bits(Element) = 32 against the tensor
  // tiles' BLK_K = 64 — an 8-bit-only result. A 4-bit tensor config would need
  // height 4, so fail here rather than silently mis-stride.
  static_assert(!kIsTensor || cute::sizeof_bits_v<typename Config::Element> == 8,
                "ScaleKind::Tensor scale surface geometry is fp8-only.");
  if constexpr (kIsTensor)
    return {kTensorScaleK, kTensorPaddedScaleN};
  else
    return {(K + Config::group_k - 1) / Config::group_k,
            round_up_align((N + Config::group_n - 1) / Config::group_n)};
}

// Packs the host grids into the padded surface and uploads both halves into
// out_sA / out_sB (resized to fit). GroupK/GroupN are the logical block sizes
// (K/N for the tensor path, Config::group_* for the block path).
template <bool IsTensor, class ElementScaleStore, class ElementScaleIn>
void pack_moe_scales(const ElementScaleIn *per_token_scale,
                     const ElementScaleIn *experts_scale, const int *count_host,
                     int num_experts, int N, int K, int GroupK, int GroupN,
                     int scale_k_store, int padded_scale_n,
                     cutlass::DeviceAllocation<ElementScaleStore> &out_sA,
                     cutlass::DeviceAllocation<ElementScaleStore> &out_sB) {
  const int scale_k_logical = IsTensor ? 1 : (K + GroupK - 1) / GroupK;
  const int scale_n         = IsTensor ? 1 : (N + GroupN - 1) / GroupN;
  const auto cast = [](float v) { return ElementScaleStore(v); };

  std::vector<int64_t> input_row_base(num_experts);
  std::vector<int64_t> output_row_base(num_experts);
  int64_t input_rows = 0, output_rows = 0;
  for (int g = 0; g < num_experts; g++) {
    input_row_base[g]  = input_rows;
    output_row_base[g] = output_rows;
    input_rows  += count_host[g];
    output_rows += round_up_align(count_host[g]);
  }

  std::vector<ElementScaleStore> h_sA(output_rows * scale_k_store, cast(0.f));
  for (int g = 0; g < num_experts; g++) {
    const int Mg          = count_host[g];
    const int padded_Mg   = round_up_align(Mg);
    const int64_t src_base = input_row_base[g] * scale_k_logical;
    const int64_t dst_base = output_row_base[g] * scale_k_store;
    for (int row = 0; row < Mg; row++)
      for (int ks = 0; ks < scale_k_store; ks++) {
        const int src_k = IsTensor ? 0 : ks;
        h_sA[dst_base + row + int64_t(ks) * padded_Mg] =
            cast(float(per_token_scale[src_base + int64_t(row) * scale_k_logical + src_k]));
      }
  }

  const int64_t sB_expert_stride = int64_t(padded_scale_n) * scale_k_store;
  std::vector<ElementScaleStore> h_sB(int64_t(num_experts) * sB_expert_stride, cast(0.f));
  for (int g = 0; g < num_experts; g++) {
    const int64_t dst_base = int64_t(g) * sB_expert_stride;
    for (int col = 0; col < padded_scale_n; col++)
      for (int ks = 0; ks < scale_k_store; ks++) {
        if (!IsTensor && col >= scale_n) continue;
        const int64_t src = IsTensor
            ? int64_t(g)
            : (int64_t(g) * N + col) * scale_k_logical + ks;
        h_sB[dst_base + col + int64_t(ks) * padded_scale_n] =
            cast(float(experts_scale[src]));
      }
  }

  out_sA.reset(h_sA.size());
  out_sB.reset(h_sB.size());
  out_sA.copy_from_host(h_sA.data());
  out_sB.copy_from_host(h_sB.data());
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// GREEDY configs. Each carries the LARGE/SMALL/TINY WG-tile variants the on-device
// greedy split dispatches to. TinyTile has its own SG layout (its few rows can't
// tile under the main one) but the same WG thread count, so all three share one
// nd_range. `is_dynamic_m` selects the MoEGEMMGreedy<IsDynamicM> instantiation.
///////////////////////////////////////////////////////////////////////////////////////////////////

using SG_4x8 = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;
using SG_1x32 = Layout<Shape<_1, cute::Int<32>, _1>, Stride<cute::Int<32>, _1, _0>>;

// 3 buckets: LARGE 256x512, SMALL 192x512, TINY 8x512. tile_k per dtype.
using MoeTile_256_512_32  = Shape<_256, _512, _32>;
using MoeTile_192_512_32  = Shape<cute::Int<192>, _512, _32>;
using MoeTile_8_512_32    = Shape<_8, _512, _32>;
using MoeTile_256_512_64  = Shape<_256, _512, _64>;
using MoeTile_192_512_64  = Shape<cute::Int<192>, _512, _64>;
using MoeTile_8_512_64    = Shape<_8, _512, _64>;
using MoeTile_256_512_128 = Shape<_256, _512, _128>;
using MoeTile_192_512_128 = Shape<cute::Int<192>, _512, _128>;
using MoeTile_8_512_128   = Shape<_8, _512, _128>;

template <bool IsDynamicM, class TLarge, class TSmall, class TSG,
          class TTiny = MoeTile_8_512_32, class TTinySG = SG_1x32>
struct Bf16GreedyConfig {
  using Element = cutlass::bfloat16_t;
  using ElementScale = void;
  using ElementOutput = cutlass::bfloat16_t;
  using LargeTile = TLarge;
  using SmallTile = TSmall;
  using TinyTile  = TTiny;
  using SGLayout = TSG;
  using TinySGLayout = TTinySG;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int group_k = 0;
  static constexpr int group_n = 0;
  static constexpr ScaleKind scale_kind = ScaleKind::Plain;
  static constexpr bool is_dynamic_m = IsDynamicM;
};

template <bool IsDynamicM, class TElement, class TScale, int GroupK, int GroupN,
          ScaleKind Kind, class TLarge, class TSmall, class TSG,
          class TTiny, class TTinySG = SG_1x32, class TOut = cutlass::bfloat16_t>
struct LowpGreedyConfig {
  using Element = TElement;
  using ElementScale = TScale;
  using ElementOutput = TOut;
  using LargeTile = TLarge;
  using SmallTile = TSmall;
  using TinyTile  = TTiny;
  using SGLayout = TSG;
  using TinySGLayout = TTinySG;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;   // mxfp4 overrides to ColumnMajor
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int group_k = GroupK;
  static constexpr int group_n = GroupN;
  static constexpr ScaleKind scale_kind = Kind;
  static constexpr bool is_dynamic_m = IsDynamicM;
};

template <bool IsDynamicM, class TElement, class TScale, int GroupK, int GroupN,
          ScaleKind Kind, class TLarge, class TSmall, class TSG,
          class TTiny, class TTinySG = SG_1x32, class TOut = cutlass::bfloat16_t>
struct MxFp4GreedyConfig
    : LowpGreedyConfig<IsDynamicM, TElement, TScale, GroupK, GroupN, Kind, TLarge,
                       TSmall, TSG, TTiny, TTinySG, TOut> {
  using LayoutB = cutlass::layout::ColumnMajor;   // mxfp4 weights are K-contiguous
};

// Uniform-M and dynamic-M greedy configs per dtype (same tiles; differ only by
// is_dynamic_m). The benchmark picks the dynamic-M variant when M varies across
// experts, else the uniform-M variant.
using Bf16Greedy      = Bf16GreedyConfig<false, MoeTile_256_512_32, MoeTile_192_512_32, SG_4x8, MoeTile_8_512_32>;
using Bf16GreedyDynM  = Bf16GreedyConfig<true,  MoeTile_256_512_32, MoeTile_192_512_32, SG_4x8, MoeTile_8_512_32>;
using MxFp8Greedy     = LowpGreedyConfig<false, cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_512_64, MoeTile_192_512_64, SG_4x8, MoeTile_8_512_64>;
using MxFp8GreedyDynM = LowpGreedyConfig<true,  cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_512_64, MoeTile_192_512_64, SG_4x8, MoeTile_8_512_64>;
using MxFp4Greedy     = MxFp4GreedyConfig<false, cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_512_128, MoeTile_192_512_128, SG_4x8, MoeTile_8_512_128>;
using MxFp4GreedyDynM = MxFp4GreedyConfig<true,  cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_512_128, MoeTile_192_512_128, SG_4x8, MoeTile_8_512_128>;
using Fp8TensorGreedy     = LowpGreedyConfig<false, cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 0, 0, ScaleKind::Tensor, MoeTile_256_512_64, MoeTile_192_512_64, SG_4x8, MoeTile_8_512_64>;
using Fp8TensorGreedyDynM = LowpGreedyConfig<true,  cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 0, 0, ScaleKind::Tensor, MoeTile_256_512_64, MoeTile_192_512_64, SG_4x8, MoeTile_8_512_64>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Per-dtype configs. Each is a pure bag of type/constant members (no kernel types):
//   Element      : packed input data type (A and B).
//   ElementScale : scale storage type (E8M0 for MX, FP32 for tensor scale).
//   group_k      : K block size for block scaling (ignored for tensor scale).
//   group_n      : N block size for block scaling (ignored for tensor scale).
//   scale_kind   : Plain (BF16), Block (MX), or Tensor (single global scale).
///////////////////////////////////////////////////////////////////////////////////////////////////

// Workgroup tile (M, N, K) per config and hardware target — the #1 perf knob for
// the scaled paths. K must be a multiple of the DPAS atom K (32 for 8-bit, 64 for
// 4-bit).
using Bf16TileCri = Shape<_256, _128, _32>;
using MxFp8TileCri = Shape<_256, _256, _64>;
using MxFp4TileCri = Shape<_512, _256, _128>;
using Fp8TensorTileCri = Shape<_256, _256, _64>;

using Bf16TileBmg = Shape<_256, _128, _32>;

// Output (D) element type is declared per Config (Config::ElementOutput). All
// current paths emit bf16 — fp16 triggered incorrect D writes on some wide-M tiles.

// BF16: no scaling. ElementScale = void so the kernel takes the plain path.
struct Bf16Config {
  using Element = cutlass::bfloat16_t;
  using ElementScale = void;
  using ElementOutput = cutlass::bfloat16_t;
  using TileShapeCri = Bf16TileCri;
  using TileShapeBmg = Bf16TileBmg;
  // Operand layouts (see make_moe_tensor for the operand convention).
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int group_k = 0;
  static constexpr int group_n = 0;
  static constexpr ScaleKind scale_kind = ScaleKind::Plain;
};

template <class TElement, class TScale, int GroupK, int GroupN, ScaleKind Kind,
          class TTileCri, class TOut = cutlass::bfloat16_t>
struct LowpConfig {
  using Element = TElement;
  using ElementScale = TScale;
  using ElementOutput = TOut;
  using TileShapeCri = TTileCri;
  // Operand layouts (see make_moe_tensor for the operand convention).
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int group_k = GroupK;
  static constexpr int group_n = GroupN;
  static constexpr ScaleKind scale_kind = Kind;
};

// MXFP8 — 8-bit data, E8M0 (MX) block scale, K block = 32 (per the MX spec).
// group_n = 1: MX-exact per-N-row B scaling, loaded via the 2D block-scale path.
using MxFp8E4m3Config =
    LowpConfig<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1,
               ScaleKind::Block, MxFp8TileCri>;

// MXFP4 — 4-bit data, E8M0 scale, MX-spec block size 32 (group_k = 32). The
// 4-bit DPAS atom-K is 64, so each MMA K-step spans two K-scale blocks; the
// scale K-offset (MMA_K / GroupK = 2) is handled in make_scaled_offsets_k.
// group_n = 1 (per-N-row, MX-exact).
using MxFp4E2m1Config =
    LowpConfig<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1,
               ScaleKind::Block, MxFp4TileCri>;

// FP8 with E8M0 per-row A / per-tensor B scale via the BDPAS path.
// Scale-A: one E8M0 per M row, replicated to scale_k=K/32.
// Scale-B: single E8M0 per expert, broadcast-filled into a padded surface.
using Fp8TensorE4m3Config =
    LowpConfig<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 0, 0,
               ScaleKind::Tensor, Fp8TensorTileCri>;

} // namespace cutlass::moe
