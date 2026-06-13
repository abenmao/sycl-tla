/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include "cute/layout.hpp"     // cute::Shape
#include "cute/numeric/numeric_types.hpp" // cute::sizeof_bits_v
#include "cutlass/arch/mma.h"  // cutlass::arch::OpClassTensorOp, cutlass::OpClassSparseTensorOp
#include "cute/util/type_traits.hpp" // cute::is_same_v

#include "cutlass/arch/arch.h"
#include "cutlass/detail/dependent_false.hpp" // cutlass::detail::dependent_false
#include "cutlass/detail/layout.hpp"
#include "cutlass/numeric_size.h" // cutlass::bytes_to_bits
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/xe4_epilogue_adma_warpspecialized.hpp"

#include "cute/arch/copy_xe4_dma_legacy.hpp"
#include "cute/arch/copy_xe4_adma.hpp"
#include "cute/atom/copy_traits_xe4_ldsm.hpp"


///////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective {

namespace detail {

// Selects the largest vectorized smem store atom available
template <int EpilogueWarpTileN, class ElementD>
constexpr auto
xe4_get_smem_store_op() {
  constexpr int CoreMatrixRowSize = 32;  // 32B
  constexpr int VS = cute::min(CoreMatrixRowSize/sizeof(ElementD), EpilogueWarpTileN);
  return cute::xe4::XE4_STSM<VS, ElementD, ElementD>{};
}

// Selects the largest vectorized smem load atom available
template <int EpilogueWarpTileN, class ElementD>
constexpr auto
xe4_get_smem_load_op() {
  constexpr int CoreMatrixRowSize = 32;  // 32B
  constexpr int VS = cute::min(CoreMatrixRowSize/sizeof(ElementD), EpilogueWarpTileN);
  return cute::xe4::XE4_LDSM<VS, ElementD, ElementD>{};
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// LDSM descriptor-based op selectors (Module 3)
/// Builder SLM is always Type1 (LayoutRight, stride<1>=1) and epilogue thread count
/// is always % 128 == 0, so we always select UnorderedVector — the highest priority mode.
/// SmemLayout is derived from MMA tile shape and core matrix constants (kCoreMatrixM=32).
/// See MODULE_DESIGN.md - Module 3 for design rationale.
////////////////////////////////////////////////////////////////////////////////////////////////////

// LDSM descriptor-based load op (XE4_LOAD_MATRIX with UnorderedVector)
// SmemLayout derived from core matrix M (32) and EpilogueWarpTileN (from MMA tile shape).
template <int EpilogueWarpTileN, class ElementD>
constexpr auto
xe4_get_ldsm_load_op() {
  // Representative Type1 row-major layout for ldsm_selector.
  // M = kCoreMatrixM (32 rows per warp), N = EpilogueWarpTileN (from MMA tile / warp count).
  constexpr int kCoreMatrixM = 32;  // hw::kCoreMatrixM
  using SmemLayout = decltype(cute::make_layout(
      cute::make_shape(cute::Int<kCoreMatrixM>{}, cute::Int<EpilogueWarpTileN>{}),
      cute::LayoutRight{}));
  return cute::ldsm_selector<ElementD, /*IsStore=*/false>(SmemLayout{});
}

// LDSM descriptor-based store op (XE4_STORE_MATRIX with UnorderedVector)
template <int EpilogueWarpTileN, class ElementD>
constexpr auto
xe4_get_ldsm_store_op() {
  constexpr int kCoreMatrixM = 32;  // hw::kCoreMatrixM
  using SmemLayout = decltype(cute::make_layout(
      cute::make_shape(cute::Int<kCoreMatrixM>{}, cute::Int<EpilogueWarpTileN>{}),
      cute::LayoutRight{}));
  return cute::ldsm_selector<ElementD, /*IsStore=*/true>(SmemLayout{});
}

// Helper for building TMA warp-specialized collective epilogues, specialized by
// the fusion operation performed and the dispatch policy to use.
template <
  class OpClass,
  class MmaTileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class FusionOpOrCallbacks
>
struct Xe4TmaBuilderImpl {
private:
  static constexpr int StagesC = 2;
  static constexpr int StagesD = 1;
  static constexpr bool ReuseSmemC = false;
  static constexpr bool DelayTmaStore = false;
  static constexpr int NumControlWarps = 4;
  static constexpr int NumEpilogueWarps = 16;
  static constexpr int EpilogueWarpTileN = 32;
  static constexpr int FragmentSize = 32 / sizeof(ElementD);

  static constexpr bool DisableSource = cute::is_void_v<ElementC_>;
  using ElementC = cute::conditional_t<DisableSource, ElementD, ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = cute::conditional_t<DisableSource, GmemLayoutTagD, GmemLayoutTagC_>;
  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

  constexpr static bool is_fp_postop = is_floating_t<ElementD>::value && (sizeof_bits_v<ElementD> < 16);
  constexpr static bool is_int8_postop = is_integral<ElementD>::value && (sizeof_bits_v<ElementD> == 8);
  using ElementImm = cute::conditional_t<is_fp_postop, bf16, cute::conditional_t<is_int8_postop, int32_t, ElementD>>;

  using CtaTileShape_MNK = MmaTileShape_MNK;
  using TileShape_MN = decltype(select<0,1>(MmaTileShape_MNK{}));

  static constexpr auto
  epilogue_tile() {
    using namespace cute;
    if constexpr (not is_same_v<EpilogueTileType, EpilogueTileAuto>) {
      static_assert(is_tuple_v<EpilogueTileType>, "Shape or Tile");
      return EpilogueTileType{};
    }
    else {
      constexpr int WarpSizeM = cutlass::NumThreadsPerWarp;
      constexpr int ElementsPerWarpN = 32;
      constexpr int TileShapeM = size<0>(CtaTileShape_MNK{});
      constexpr int TileShapeN = size<1>(CtaTileShape_MNK{});
      static_assert(TileShapeM % WarpSizeM == 0, "CTA tile must be divisible by warp size in M dimension");
      static_assert(TileShapeN % ElementsPerWarpN == 0, "CTA tile must be divisible by elements per warp in N dimension");

      constexpr int WarpsAlongM = TileShapeM / WarpSizeM;
      constexpr int WarpsAlongN = TileShapeN / ElementsPerWarpN;
      static_assert((WarpsAlongM * WarpsAlongN) % (NumEpilogueWarps) == 0, "Total warp count must be divisible by NumEpilogueWarps");

      constexpr int NumWarpsAlongN = min(WarpsAlongN, NumEpilogueWarps);
      constexpr int NumWarpsAlongM = min(WarpsAlongM, NumEpilogueWarps / NumWarpsAlongN);
      constexpr int EpilogueTileM = NumWarpsAlongM * WarpSizeM;
      constexpr int EpilogueTileN = NumWarpsAlongN * ElementsPerWarpN;

      return make_tile(Int<EpilogueTileM>{}, Int<EpilogueTileN>{});
    }
  }
  using EpilogueTile = decltype(epilogue_tile());

  using FusionCallbacks = fusion::FusionCallbacks<
    Xe4TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    FusionOpOrCallbacks, CtaTileShape_MNK, EpilogueTile
  >;

  using SmemLayoutAtomC = decltype(make_ordered_layout(EpilogueTile{}, Step<_1, _0>{}));
  using SmemLayoutAtomD = decltype(make_ordered_layout(TileShape_MN{}, Step<_1, _0>{}));
  using CopyOpS2G = cute::conditional_t<detail::is_im2col_mode<GmemLayoutTagD>,
      xe4::ASYNC_ROW_STORE_IM2COL<slm_matrix_type::type1, size<1>(CtaTileShape_MNK{})>,
      xe4::ASYNC_TENSOR_STORE<slm_matrix_type::type1, size<1>(CtaTileShape_MNK{})>
    >;
  using CopyOpG2S = cute::conditional_t<detail::is_im2col_mode<GmemLayoutTagC>,
      xe4::ASYNC_ROW_LOAD_IM2COL<slm_matrix_type::type1, size<1>(CtaTileShape_MNK{})>,
      xe4::ASYNC_TENSOR_LOAD<slm_matrix_type::type1, size<1>(CtaTileShape_MNK{})>
    >;

public:
  using CollectiveOp =
    cutlass::epilogue::collective::CollectiveEpilogue<
      Xe4AdmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, NumControlWarps, NumEpilogueWarps>,
      CtaTileShape_MNK,
      EpilogueTile,
      ElementC_, // Need to pass void through to expose via GemmUniversal
      GmemStrideTypeC,
      ElementD,
      GmemStrideTypeD,
      FusionCallbacks,
      CopyOpG2S,
      SmemLayoutAtomC,
      decltype(xe4_get_smem_load_op<EpilogueWarpTileN, ElementD>()),
      decltype(xe4_get_smem_load_op<EpilogueWarpTileN, ElementImm>()),
      CopyOpS2G,
      SmemLayoutAtomD,
      decltype(xe4_get_smem_store_op<EpilogueWarpTileN, ElementD>()),
      void
    >;
};

// Helper for building ADMA warp-specialized collective epilogues, specialized by
// the fusion operation performed and the dispatch policy to use.
template <
  class OpClass,
  class MmaTileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class FusionOpOrCallbacks
>
struct Xe4AdmaBuilderImpl {
private:
  static constexpr int StagesC = 2;
  static constexpr int StagesD = 1;
  static constexpr bool ReuseSmemC = false;
  static constexpr bool DelayTmaStore = false;
  static constexpr int NumControlWarps = 4;
  static constexpr int NumEpilogueWarps = 16;
  static constexpr int EpilogueWarpTileN = 32;
  static constexpr int FragmentSize = 32 / sizeof(ElementD);

  static constexpr bool DisableSource = cute::is_void_v<ElementC_>;
  using ElementC = cute::conditional_t<DisableSource, ElementD, ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = cute::conditional_t<DisableSource, GmemLayoutTagD, GmemLayoutTagC_>;
  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

  constexpr static bool is_fp_postop = is_floating_t<ElementD>::value && (sizeof_bits_v<ElementD> < 16);
  constexpr static bool is_int8_postop = is_integral<ElementD>::value && (sizeof_bits_v<ElementD> == 8);
  using ElementImm = cute::conditional_t<is_fp_postop, bf16, cute::conditional_t<is_int8_postop, int32_t, ElementD>>;

  using CtaTileShape_MNK = MmaTileShape_MNK;
  using TileShape_MN = decltype(select<0,1>(MmaTileShape_MNK{}));

  static constexpr auto
  epilogue_tile() {
    using namespace cute;
    if constexpr (not is_same_v<EpilogueTileType, EpilogueTileAuto>) {
      static_assert(is_tuple_v<EpilogueTileType>, "Shape or Tile");
      return EpilogueTileType{};
    }
    else {
      constexpr int WarpSizeM = cutlass::NumThreadsPerWarp;
      constexpr int ElementsPerWarpN = 32;
      constexpr int TileShapeM = size<0>(CtaTileShape_MNK{});
      constexpr int TileShapeN = size<1>(CtaTileShape_MNK{});
      static_assert(TileShapeM % WarpSizeM == 0, "CTA tile must be divisible by warp size in M dimension");
      static_assert(TileShapeN % ElementsPerWarpN == 0, "CTA tile must be divisible by elements per warp in N dimension");

      constexpr int WarpsAlongM = TileShapeM / WarpSizeM;
      constexpr int WarpsAlongN = TileShapeN / ElementsPerWarpN;

      constexpr int NumWarpsAlongN = min(WarpsAlongN, NumEpilogueWarps);
      constexpr int NumWarpsAlongM = min(WarpsAlongM, NumEpilogueWarps / NumWarpsAlongN);
      constexpr int EpilogueTileM = NumWarpsAlongM * WarpSizeM;
      constexpr int EpilogueTileN = NumWarpsAlongN * ElementsPerWarpN;

      return make_tile(Int<EpilogueTileM>{}, Int<EpilogueTileN>{});
    }
  }
  using EpilogueTile = decltype(epilogue_tile());

  using FusionCallbacks = fusion::FusionCallbacks<
    Xe4TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    FusionOpOrCallbacks, CtaTileShape_MNK, EpilogueTile
  >;

  using SmemLayoutAtomC = decltype(make_ordered_layout(EpilogueTile{}, Step<_1, _0>{}));
  using SmemLayoutAtomD = decltype(make_ordered_layout(TileShape_MN{}, Step<_1, _0>{}));
  using CopyOpS2G = XE4_ADMA_STORE;
  using CopyOpG2S = XE4_ADMA_LOAD;

public:
  using CollectiveOp =
    cutlass::epilogue::collective::CollectiveEpilogue<
      Xe4AdmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, NumControlWarps, NumEpilogueWarps>,
      CtaTileShape_MNK,
      EpilogueTile,
      ElementC_, // Need to pass void through to expose via GemmUniversal
      GmemStrideTypeC,
      ElementD,
      GmemStrideTypeD,
      FusionCallbacks,
      CopyOpG2S,
      SmemLayoutAtomC,
      decltype(xe4_get_smem_load_op<EpilogueWarpTileN, ElementD>()),
      decltype(xe4_get_smem_load_op<EpilogueWarpTileN, ElementImm>()),
      CopyOpS2G,
      SmemLayoutAtomD,
      decltype(xe4_get_smem_store_op<EpilogueWarpTileN, ElementD>()),
      void
    >;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Xe4AdmaSkReduceBuilderImpl — same as Xe4AdmaBuilderImpl but with CopyOpS2GReduce wired in.
/// Used for StreamK GEMMs: non-final splits fire async_tensor_fred (smem_Imm -> gmem D
/// reduction workspace); final split polls the per-tile counter, does G2S load-back, then
/// runs the full epilogue normally. Rop and BType are forwarded to XE4_ADMA_STORE_REDUCE.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <
  class OpClass,
  class MmaTileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class FusionOpOrCallbacks,
  cute::RedOp Rop   = cute::RedOp::Add,
  cute::BarrierType BType = cute::BarrierType::Abarrier
>
struct Xe4AdmaSkReduceBuilderImpl {
private:
  static constexpr int StagesC = 2;
  static constexpr int StagesD = 1;
  static constexpr bool ReuseSmemC = false;
  static constexpr bool DelayTmaStore = false;
  static constexpr int NumControlWarps = 4;
  static constexpr int NumEpilogueWarps = 16;
  static constexpr int EpilogueWarpTileN = 32;
  static constexpr int FragmentSize = 32 / sizeof(ElementD);

  static constexpr bool DisableSource = cute::is_void_v<ElementC_>;
  using ElementC = cute::conditional_t<DisableSource, ElementD, ElementC_>;
  using GmemLayoutTagC = cute::conditional_t<DisableSource, GmemLayoutTagD, GmemLayoutTagC_>;
  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

  constexpr static bool is_fp_postop = is_floating_t<ElementD>::value && (sizeof_bits_v<ElementD> < 16);
  constexpr static bool is_int8_postop = is_integral<ElementD>::value && (sizeof_bits_v<ElementD> == 8);
  using ElementImm = cute::conditional_t<is_fp_postop, bf16, cute::conditional_t<is_int8_postop, int32_t, ElementD>>;

  using CtaTileShape_MNK = MmaTileShape_MNK;
  using TileShape_MN = decltype(select<0,1>(MmaTileShape_MNK{}));

  static constexpr auto
  epilogue_tile() {
    using namespace cute;
    if constexpr (not is_same_v<EpilogueTileType, EpilogueTileAuto>) {
      static_assert(is_tuple_v<EpilogueTileType>, "Shape or Tile");
      return EpilogueTileType{};
    }
    else {
      constexpr int WarpSizeM = cutlass::NumThreadsPerWarp;
      constexpr int ElementsPerWarpN = 32;
      constexpr int TileShapeM = size<0>(CtaTileShape_MNK{});
      constexpr int TileShapeN = size<1>(CtaTileShape_MNK{});
      static_assert(TileShapeM % WarpSizeM == 0);
      static_assert(TileShapeN % ElementsPerWarpN == 0);
      constexpr int WarpsAlongM = TileShapeM / WarpSizeM;
      constexpr int WarpsAlongN = TileShapeN / ElementsPerWarpN;
      constexpr int NumWarpsAlongN = min(WarpsAlongN, NumEpilogueWarps);
      constexpr int NumWarpsAlongM = min(WarpsAlongM, NumEpilogueWarps / NumWarpsAlongN);
      return make_tile(Int<NumWarpsAlongM * WarpSizeM>{}, Int<NumWarpsAlongN * ElementsPerWarpN>{});
    }
  }
  using EpilogueTile = decltype(epilogue_tile());

  using FusionCallbacks = fusion::FusionCallbacks<
    Xe4TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    FusionOpOrCallbacks, CtaTileShape_MNK, EpilogueTile
  >;

  using SmemLayoutAtomC = decltype(make_ordered_layout(EpilogueTile{}, Step<_1, _0>{}));
  using SmemLayoutAtomD = decltype(make_ordered_layout(TileShape_MN{}, Step<_1, _0>{}));
  using CopyOpS2G        = XE4_ADMA_STORE;
  using CopyOpG2S        = XE4_ADMA_LOAD;
  using CopyOpS2GReduce  = cute::XE4_ADMA_STORE_REDUCE<ElementImm, Rop, BType>;

public:
  using CollectiveOp =
    cutlass::epilogue::collective::CollectiveEpilogue<
      Xe4AdmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, NumControlWarps, NumEpilogueWarps>,
      CtaTileShape_MNK,
      EpilogueTile,
      ElementC_,
      GmemStrideTypeC,
      ElementD,
      GmemStrideTypeD,
      FusionCallbacks,
      CopyOpG2S,
      SmemLayoutAtomC,
      decltype(xe4_get_smem_load_op<EpilogueWarpTileN, ElementD>()),
      decltype(xe4_get_smem_load_op<EpilogueWarpTileN, ElementImm>()),
      CopyOpS2G,
      SmemLayoutAtomD,
      decltype(xe4_get_smem_store_op<EpilogueWarpTileN, ElementD>()),
      void,
      CopyOpS2GReduce
    >;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// LDSM ADMA builder impl (Module 3)
/// Same as Xe4AdmaBuilderImpl but uses descriptor-based XE4_LOAD_MATRIX / XE4_STORE_MATRIX
/// for S2R and R2S operations instead of legacy XE4_LDSM / XE4_STSM.
/// See MODULE_DESIGN.md - Module 3 for design rationale.
////////////////////////////////////////////////////////////////////////////////////////////////////
template <
  class OpClass,
  class MmaTileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class FusionOpOrCallbacks
>
struct Xe4LdsmAdmaBuilderImpl {
private:
  static constexpr int StagesC = 2;
  static constexpr int StagesD = 1;
  static constexpr bool ReuseSmemC = false;
  static constexpr bool DelayTmaStore = false;
  static constexpr int NumControlWarps = 4;
  static constexpr int NumEpilogueWarps = 16;

  // Core matrix constants for MMA-derived layout computation
  static constexpr int kBytesPerCmRow = 32;  // hw::kBytesPerCmRow from xe4_async_gmma_slm_layout.hpp
  static constexpr int FragmentSize = kBytesPerCmRow / sizeof(ElementD);

  static constexpr bool DisableSource = cute::is_void_v<ElementC_>;
  using ElementC = cute::conditional_t<DisableSource, ElementD, ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = cute::conditional_t<DisableSource, GmemLayoutTagD, GmemLayoutTagC_>;
  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

  constexpr static bool is_fp_postop = is_floating_t<ElementD>::value && (sizeof_bits_v<ElementD> < 16);
  constexpr static bool is_int8_postop = is_integral<ElementD>::value && (sizeof_bits_v<ElementD> == 8);
  using ElementImm = cute::conditional_t<is_fp_postop, bf16, cute::conditional_t<is_int8_postop, int32_t, ElementD>>;

  using CtaTileShape_MNK = MmaTileShape_MNK;
  using TileShape_MN = decltype(select<0,1>(MmaTileShape_MNK{}));

  static constexpr auto
  epilogue_tile() {
    using namespace cute;
    if constexpr (not is_same_v<EpilogueTileType, EpilogueTileAuto>) {
      static_assert(is_tuple_v<EpilogueTileType>, "Shape or Tile");
      return EpilogueTileType{};
    }
    else {
      constexpr int WarpSizeM = cutlass::NumThreadsPerWarp;
      // ElementsPerWarpN: N-elements per warp, derived from MMA tile and warp count.
      // For the EpilogueTile computation, this determines how N is partitioned among warps.
      constexpr int ElementsPerWarpN = 32;
      constexpr int TileShapeM = size<0>(CtaTileShape_MNK{});
      constexpr int TileShapeN = size<1>(CtaTileShape_MNK{});
      static_assert(TileShapeM % WarpSizeM == 0, "CTA tile must be divisible by warp size in M dimension");
      static_assert(TileShapeN % ElementsPerWarpN == 0, "CTA tile must be divisible by elements per warp in N dimension");

      constexpr int WarpsAlongM = TileShapeM / WarpSizeM;
      constexpr int WarpsAlongN = TileShapeN / ElementsPerWarpN;

      constexpr int NumWarpsAlongN = min(WarpsAlongN, NumEpilogueWarps);
      constexpr int NumWarpsAlongM = min(WarpsAlongM, NumEpilogueWarps / NumWarpsAlongN);
      constexpr int EpilogueTileM = NumWarpsAlongM * WarpSizeM;
      constexpr int EpilogueTileN = NumWarpsAlongN * ElementsPerWarpN;

      return make_tile(Int<EpilogueTileM>{}, Int<EpilogueTileN>{});
    }
  }
  using EpilogueTile = decltype(epilogue_tile());

  // EpilogueWarpTileN: N-elements per warp, derived from EpilogueTile and warp count.
  // This replaces the previous hardcoded value of 32, making it MMA-tile-aware.
  static constexpr int EpilogueWarpTileN = []() {
    constexpr int EpiTileM = cute::size<0>(EpilogueTile{});
    constexpr int EpiTileN = cute::size<1>(EpilogueTile{});
    constexpr int WarpSizeM = cutlass::NumThreadsPerWarp;  // 32
    constexpr int NumWarpsAlongM = EpiTileM / WarpSizeM;
    constexpr int NumWarpsAlongN = NumEpilogueWarps / NumWarpsAlongM;
    return EpiTileN / NumWarpsAlongN;
  }();

  using FusionCallbacks = fusion::FusionCallbacks<
    Xe4TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    FusionOpOrCallbacks, CtaTileShape_MNK, EpilogueTile
  >;

  using SmemLayoutAtomC = decltype(make_ordered_layout(EpilogueTile{}, Step<_1, _0>{}));
  using SmemLayoutAtomD = decltype(make_ordered_layout(TileShape_MN{}, Step<_1, _0>{}));
  using CopyOpS2G = XE4_ADMA_STORE;
  using CopyOpG2S = XE4_ADMA_LOAD;

public:
  using CollectiveOp =
    cutlass::epilogue::collective::CollectiveEpilogue<
      Xe4AdmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, NumControlWarps, NumEpilogueWarps>,
      CtaTileShape_MNK,
      EpilogueTile,
      ElementC_, // Need to pass void through to expose via GemmUniversal
      GmemStrideTypeC,
      ElementD,
      GmemStrideTypeD,
      FusionCallbacks,
      CopyOpG2S,
      SmemLayoutAtomC,
      decltype(xe4_get_ldsm_load_op<EpilogueWarpTileN, ElementD>()),
      decltype(xe4_get_ldsm_load_op<EpilogueWarpTileN, ElementImm>()),
      CopyOpS2G,
      SmemLayoutAtomD,
      decltype(xe4_get_ldsm_store_op<EpilogueWarpTileN, ElementD>()),
      void
    >;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// MMA-aware LDSM epilogue builder
///
/// Like Xe4LdsmAdmaBuilderImpl but enables the MMA-aware make_ldsm_copy_C/D path
/// via UseMmaAwareLdsm=true in the dispatch policy. The MMA-aware algorithm derives
/// LDSM warp distribution from TiledMMA::AtomShape_MNK, gracefully handling cases
/// where the MMA atom is larger than the EpilogueTile by clamping effective atom
/// dimensions to the tile (see make_ldsm_copy_CD_mma_impl in copy_traits_xe4_ldsm.hpp).
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class OpClass,
  class MmaTileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class FusionOpOrCallbacks
>
struct Xe4LdsmMmaAdmaBuilderImpl {
private:
  static constexpr int StagesC = 2;
  static constexpr int StagesD = 1;
  static constexpr bool ReuseSmemC = false;
  static constexpr bool DelayTmaStore = false;
  static constexpr int NumControlWarps = 4;
  static constexpr int NumEpilogueWarps = 16;

  static constexpr int kBytesPerCmRow = 32;
  static constexpr int FragmentSize = kBytesPerCmRow / sizeof(ElementD);

  static constexpr bool DisableSource = cute::is_void_v<ElementC_>;
  using ElementC = cute::conditional_t<DisableSource, ElementD, ElementC_>;
  using GmemLayoutTagC = cute::conditional_t<DisableSource, GmemLayoutTagD, GmemLayoutTagC_>;
  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

  constexpr static bool is_fp_postop = is_floating_t<ElementD>::value && (sizeof_bits_v<ElementD> < 16);
  constexpr static bool is_int8_postop = is_integral<ElementD>::value && (sizeof_bits_v<ElementD> == 8);
  using ElementImm = cute::conditional_t<is_fp_postop, bf16, cute::conditional_t<is_int8_postop, int32_t, ElementD>>;

  using CtaTileShape_MNK = MmaTileShape_MNK;
  using TileShape_MN = decltype(select<0,1>(MmaTileShape_MNK{}));

  // EpilogueTile: same auto-resolution logic as Xe4LdsmAdmaBuilderImpl
  static constexpr auto
  epilogue_tile() {
    using namespace cute;
    if constexpr (not is_same_v<EpilogueTileType, EpilogueTileAuto>) {
      static_assert(is_tuple_v<EpilogueTileType>, "Shape or Tile");
      return EpilogueTileType{};
    }
    else {
      constexpr int WarpSizeM = cutlass::NumThreadsPerWarp;
      constexpr int ElementsPerWarpN = 32;
      constexpr int TileShapeM = size<0>(CtaTileShape_MNK{});
      constexpr int TileShapeN = size<1>(CtaTileShape_MNK{});
      static_assert(TileShapeM % WarpSizeM == 0, "CTA tile must be divisible by warp size in M dimension");
      static_assert(TileShapeN % ElementsPerWarpN == 0, "CTA tile must be divisible by elements per warp in N dimension");

      constexpr int WarpsAlongM = TileShapeM / WarpSizeM;
      constexpr int WarpsAlongN = TileShapeN / ElementsPerWarpN;

      constexpr int NumWarpsAlongN = min(WarpsAlongN, NumEpilogueWarps);
      constexpr int NumWarpsAlongM = min(WarpsAlongM, NumEpilogueWarps / NumWarpsAlongN);
      constexpr int EpilogueTileM = NumWarpsAlongM * WarpSizeM;
      constexpr int EpilogueTileN = NumWarpsAlongN * ElementsPerWarpN;

      return make_tile(Int<EpilogueTileM>{}, Int<EpilogueTileN>{});
    }
  }
  using EpilogueTile = decltype(epilogue_tile());

  static constexpr int EpilogueWarpTileN = []() {
    constexpr int EpiTileM = cute::size<0>(EpilogueTile{});
    constexpr int EpiTileN = cute::size<1>(EpilogueTile{});
    constexpr int WarpSizeM = cutlass::NumThreadsPerWarp;
    constexpr int NumWarpsAlongM = EpiTileM / WarpSizeM;
    constexpr int NumWarpsAlongN = NumEpilogueWarps / NumWarpsAlongM;
    return EpiTileN / NumWarpsAlongN;
  }();

  using FusionCallbacks = fusion::FusionCallbacks<
    Xe4TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    FusionOpOrCallbacks, CtaTileShape_MNK, EpilogueTile
  >;

  using SmemLayoutAtomC = decltype(make_ordered_layout(EpilogueTile{}, Step<_1, _0>{}));
  using SmemLayoutAtomD = decltype(make_ordered_layout(TileShape_MN{}, Step<_1, _0>{}));
  using CopyOpS2G = XE4_ADMA_STORE;
  using CopyOpG2S = XE4_ADMA_LOAD;

public:
  using CollectiveOp =
    cutlass::epilogue::collective::CollectiveEpilogue<
      Xe4AdmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore,
                              NumControlWarps, NumEpilogueWarps, /*UseMmaAwareLdsm=*/true>,
      CtaTileShape_MNK,
      EpilogueTile,
      ElementC_,
      GmemStrideTypeC,
      ElementD,
      GmemStrideTypeD,
      FusionCallbacks,
      CopyOpG2S,
      SmemLayoutAtomC,
      decltype(xe4_get_ldsm_load_op<EpilogueWarpTileN, ElementD>()),
      decltype(xe4_get_ldsm_load_op<EpilogueWarpTileN, ElementImm>()),
      CopyOpS2G,
      SmemLayoutAtomD,
      decltype(xe4_get_ldsm_store_op<EpilogueWarpTileN, ElementD>()),
      void
    >;
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////

// Auto epilogue builder for TensorOp kernels
template <
  class OpClass,
  class MmaTileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOp
>
struct CollectiveBuilder<
    arch::Xe4,
    OpClass,
    MmaTileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueScheduleAuto,
    FusionOp,
    // only for TensorOp kernels
    cute::enable_if_t<not cute::is_same_v<OpClass, arch::OpClassSimt>>
>
 {
private:
  using EpilogueSchedule = TmaWarpSpecialized;

public:
  using CollectiveOp =
    typename detail::Xe4AdmaBuilderImpl<
      OpClass,
      MmaTileShape_MNK,
      ClusterShape_MNK,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      GmemLayoutTagC,
      AlignmentC,
      ElementD,
      GmemLayoutTagD,
      AlignmentD,
      EpilogueSchedule,
      FusionOp
    >::CollectiveOp;
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::collective
