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

#include "cutlass/conv/collective/builders/sm90_common.inl"
#include <cutlass/arch/arch.h>

namespace cutlass::conv::collective {
using namespace cute;

template<
  class GmemTiledCopy_,
  class SmemLayout_,
  class SmemCopyAtom_ = void
>
struct Xe4ImplicitGemmTileTraits {
  using GmemTiledCopy = GmemTiledCopy_;
  using SmemLayout = SmemLayout_;
  using SmemCopyAtom = SmemCopyAtom_;
};

// Intel_DMA_WS_SS_FPROP
template <
  conv::Operator ConvOp,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Xe4,
    arch::OpClassTensorOp,
    ConvOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelImplicitTmaWarpSpecializedXe4>>
> {
  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  // For fprop, majorA = K,  major B = K;
  // For wgrad, majorA = MN, major B = MN;
  // For dgrad, majorA = K,  major B = MN;
  static constexpr auto GmmaMajorA =
    (ConvOp == conv::Operator::kWgrad) ? cute::AMMA::Major::MN : cute::AMMA::Major::K;
  static constexpr auto GmmaMajorB =
    (ConvOp == conv::Operator::kFprop) ? cute::AMMA::Major::K : cute::AMMA::Major::MN;

  using TiledMma = decltype(cute::make_tiled_mma(AMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, decltype(cute::product_each(TileShape_MNK{})), ClusterShape_MNK, GmmaMajorA, GmmaMajorB>()));

  // For wgrad kernel, tensor A uses tma tiled mode and tensor B uses tma im2col mode.
  static constexpr slm_matrix_type cmTypeA = GmmaMajorA == cute::AMMA::Major::K ? slm_matrix_type::type1 : slm_matrix_type::type2;
  static constexpr uint32_t cmStrideA = GmmaMajorA == cute::AMMA::Major::MN ? size<0>(TileShape_MNK{}) : size<2>(TileShape_MNK{});
  static constexpr uint32_t cmStrideB = GmmaMajorB == cute::AMMA::Major::MN ? size<1>(TileShape_MNK{}) : size<2>(TileShape_MNK{});

  using GmemTiledCopyA = cute::conditional_t<
    ConvOp == conv::Operator::kWgrad,
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::xe4::ASYNC_TENSOR_LOAD<cmTypeA, cmStrideA>,
      cute::xe4::ASYNC_TENSOR_LOAD_MULTICAST<cmTypeA, cmStrideA>
    >,
    cute::xe4::ASYNC_ROW_LOAD_IM2COL<cmTypeA, cmStrideA>
  >;

  using GmemTiledCopyB = cute::conditional_t<
    ConvOp == conv::Operator::kWgrad,
    cute::xe4::ASYNC_ROW_LOAD_IM2COL<slm_matrix_type::type1, cmStrideB>,
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::xe4::ASYNC_TENSOR_LOAD<slm_matrix_type::type1, cmStrideB>,
      cute::xe4::ASYNC_TENSOR_LOAD<slm_matrix_type::type1, cmStrideB>
    >
  >;

  using SmemLayoutAtomA = cute::conditional_t<
    GmmaMajorA == cute::AMMA::Major::K,
    decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenRowMajor{})),
    decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenColMajor{}))
  >;

  using SmemLayoutAtomB = cute::conditional_t<
    GmmaMajorB == cute::AMMA::Major::K,
    decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenRowMajor{})),
    decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenColMajor{}))
  >;

  static constexpr int PipelineStages = StageCountType::value;

  using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<PipelineStages>{}),
    cute::conditional_t<GmmaMajorA == cute::AMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{},
    make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<PipelineStages>{}),
    cute::conditional_t<GmmaMajorB == cute::AMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  constexpr static int NumSpatialDimensions = cutlass::conv::collective::detail::gmem_layout_tags_to_spatial_dims<GmemLayoutA, GmemLayoutB>();

  using DispatchPolicy = MainloopXe4DmaGmmaWarpSpecializedImplicitGemm<
    ConvOp, PipelineStages, NumSpatialDimensions, ClusterShape_MNK, KernelScheduleType, 1>;

  using CollectiveOp = CollectiveConv<
    DispatchPolicy,
    TileShape_MNK,
    ElementA,
    ElementB,
    TiledMma,
    detail::Xe4ImplicitGemmTileTraits<GmemTiledCopyA, SmemLayoutA>,
    detail::Xe4ImplicitGemmTileTraits<GmemTiledCopyB, SmemLayoutB>
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA auto kernel schedule
template <
  conv::Operator ConvOp,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Xe4,
    arch::OpClassTensorOp,
    ConvOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelScheduleAuto>>
> {
    using KernelWarpSpecializedSchedule = KernelImplicitTmaWarpSpecializedXe4;

    using CollectiveOp = typename CollectiveBuilder<
      arch::Xe4,
      arch::OpClassTensorOp,
      ConvOp,
      ElementA,
      GmemLayoutA,
      AlignmentA,
      ElementB,
      GmemLayoutB,
      AlignmentB,
      ElementAccumulator,
      TileShape_MNK,
      ClusterShape_MNK,
      StageCountType,
      KernelWarpSpecializedSchedule
    >::CollectiveOp;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
