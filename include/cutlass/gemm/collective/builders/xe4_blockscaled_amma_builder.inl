/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation. All rights reserved.
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

#include <cutlass/arch/arch.h>
#include <cute/arch/mma_xe4.hpp>

#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/detail/xe4_blockscaled_layout.hpp>
#include <cutlass/float_subbyte.h>

#include "cutlass/gemm/collective/collective_mma.hpp"

namespace cutlass::gemm::collective {

namespace detail {

template <class T>
struct block_scale_traits;

template <class T, class SF, int VS>
struct block_scale_traits<cute::tuple<T, SF, cute::Int<VS>>> {
  using data_type = T;
  using sf_type   = SF;
  static constexpr int SfVectorSize = VS;
};

} // namespace detail

template <
  class ElementTupleA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementTupleB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
  >
struct CollectiveBuilder<
    arch::Xe4,
    arch::OpClassBlockScaledTensorOp,
    ElementTupleA,
    GmemLayoutATag,
    AlignmentA,
    ElementTupleB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cutlass::detail::is_kernel_tag_of_v<KernelScheduleType, KernelTmaWarpSpecializedXe4>>>
{
  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");

  // Extract element types, SF types, and SFVecSize from ElementTupleA/B
  using TraitsA = detail::block_scale_traits<ElementTupleA>;
  using TraitsB = detail::block_scale_traits<ElementTupleB>;

  using ElementA   = typename TraitsA::data_type;
  using ElementSFA = typename TraitsA::sf_type;
  using ElementB   = typename TraitsB::data_type;
  using ElementSFB = typename TraitsB::sf_type;

  static constexpr int SFVecSizeA = TraitsA::SfVectorSize;
  static constexpr int SFVecSizeB = TraitsB::SfVectorSize;

  static_assert(cute::is_same_v<ElementSFA, ElementSFB>,
    "Block-scaled GEMM requires the same scale factor type for A and B.");
  static_assert(SFVecSizeA == SFVecSizeB,
    "Block-scaled GEMM requires the same SF vector size for A and B.");
  static_assert(cute::is_same_v<ElementA, ElementB>,
    "Mixed data-type support for block-scaled GEMM is not available yet.");

  using ElementSF = ElementSFA;
  static constexpr int SFVecSize = SFVecSizeA;

  static_assert(cute::size<2>(TileShape_MNK{}) % SFVecSize == 0,
    "TileK must be a multiple of SFVecSize for block-scaled GEMM.");

  // FP8 data (e4m3) only supports ue8m0+32; all 5 SF/VS combos are valid for FP4 (e2m1).
  static_assert(!cute::is_same_v<ElementA, cutlass::float_e4m3_t> ||
                (cute::is_same_v<ElementSF, cutlass::float_ue8m0_t> && SFVecSize == 32),
    "FP8 (e4m3) data only supports ue8m0+VS=32 scale factors.");

  // Determine major mode for A and B
  static constexpr auto majorA = cutlass::gemm::detail::is_mn_major_A<GmemLayoutATag>()
      ? cute::AMMA::Major::MN : cute::AMMA::Major::K;
  static constexpr auto majorB = cutlass::gemm::detail::is_mn_major_B<GmemLayoutBTag>()
      ? cute::AMMA::Major::MN : cute::AMMA::Major::K;

  // Select the block-scaled MMA atom via bs_op_selector and wrap in TiledMma
  // ElementAccumulator is cute::tuple<AccumType, OutputType> by convention.
  using TiledMma = decltype(cute::make_tiled_mma(
    cute::AMMA::bs_op_selector<
      typename cute::tuple_element<1, ElementAccumulator>::type,   // ElementD
      ElementA, ElementB,
      typename cute::tuple_element<0, ElementAccumulator>::type,   // ElementC
      ElementSF,
      SFVecSize,
      decltype(cute::product_each(TileShape_MNK{})),
      ClusterShape_MNK, majorA, majorB>()
  ));

  // Configure dispatch policy
  static constexpr int PipelineStages = StageCountType::value;
  // Hardcoded to same values as xe4_amma_builder.inl for now
  static constexpr int SchedulerPipelineStageCount = 3;
  static constexpr int AccumulatorPipelineStageCount = 1;

  using DispatchPolicy =
    cutlass::gemm::MainloopXe4DmaGmmaWarpSpecializedBlockScaled<
        PipelineStages,
        SchedulerPipelineStageCount,
        AccumulatorPipelineStageCount,
        ClusterShape_MNK
    >;

  // ADMA copy operation types (unicast vs multicast based on cluster size).
  // Actual copy atoms are built at runtime in the mainloop's to_underlying_arguments.
  using GmemTiledCopyA =
    cute::conditional_t<
      cute::size(ClusterShape_MNK{}) == 1,
      cute::XE4_ADMA_LOAD,
      cute::XE4_ADMA_LOAD_MULTICAST
    >;

  using GmemTiledCopyB =
    cute::conditional_t<
      cute::size(ClusterShape_MNK{}) == 1,
      cute::XE4_ADMA_LOAD,
      cute::XE4_ADMA_LOAD_MULTICAST
    >;

  // SF copy ops mirror their data counterparts
  using GmemTiledCopySFA = GmemTiledCopyA;
  using GmemTiledCopySFB = GmemTiledCopyB;

  // Bundle data + SF copy ops as Pair types for the mainloop
  using GmemTiledCopyPairA = decltype(cute::make_tuple(GmemTiledCopyA{}, GmemTiledCopySFA{}));
  using GmemTiledCopyPairB = decltype(cute::make_tuple(GmemTiledCopyB{}, GmemTiledCopySFB{}));

  // SMEM layouts for data tensors (A and B)
  using SmemLayoutAtomA =
    cute::conditional_t<
      majorA == cute::AMMA::Major::K,
      decltype(cute::make_layout(cute::select<0, 2>(TileShape_MNK{}), cute::GenRowMajor{})),
      decltype(cute::make_layout(cute::select<0, 2>(TileShape_MNK{}), cute::GenColMajor{}))
    >;

  using SmemLayoutAtomB =
    cute::conditional_t<
      majorB == cute::AMMA::Major::K,
      decltype(cute::make_layout(cute::select<1, 2>(TileShape_MNK{}), cute::GenRowMajor{})),
      decltype(cute::make_layout(cute::select<1, 2>(TileShape_MNK{}), cute::GenColMajor{}))
    >;

  // SMEM layouts for SF tensors via Xe4BlockScaledConfig from xe4_blockscaled_layout.hpp
  using Xe4BlkScaledConfig = cutlass::detail::Xe4BlockScaledConfig<SFVecSize>;

  using SmemLayoutAtomSFA = decltype(Xe4BlkScaledConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape_MNK{}));
  using SmemLayoutAtomSFB = decltype(Xe4BlkScaledConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape_MNK{}));

  // Bundle data + SF SMEM layouts as Pair types for the mainloop
  using SmemLayoutAtomPairA = decltype(cute::make_tuple(SmemLayoutAtomA{}, SmemLayoutAtomSFA{}));
  using SmemLayoutAtomPairB = decltype(cute::make_tuple(SmemLayoutAtomB{}, SmemLayoutAtomSFB{}));

  // Stride types for data and SF global memory
  using StrideA = cutlass::gemm::TagToStrideA_t<GmemLayoutATag>;
  using StrideB = cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>;

  // SF global memory layouts deduced from Xe4BlockScaledConfig
  using LayoutSFA = decltype(Xe4BlkScaledConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(Xe4BlkScaledConfig::deduce_layoutSFB());

  // Bundle data stride + SF layout as Pair types
  using StridePairA = decltype(cute::make_tuple(StrideA{}, LayoutSFA{}));
  using StridePairB = decltype(cute::make_tuple(StrideB{}, LayoutSFB{}));

  // Assemble CollectiveOp
  // Forward the original 3-element ElementTupleA/B so the mainloop can extract SFVecSize.
  // ElementTupleA = cute::tuple<ElementA, ElementSF, cute::Int<SFVecSize>>
  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy,
    TileShape_MNK,
    ElementTupleA,
    StridePairA,
    ElementTupleB,
    StridePairB,
    TiledMma,
    GmemTiledCopyPairA,
    SmemLayoutAtomPairA,
    void,
    cute::identity,
    GmemTiledCopyPairB,
    SmemLayoutAtomPairB,
    void,
    cute::identity
  >;
};

} // namespace cutlass::gemm::collective
