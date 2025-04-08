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


///////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective {

namespace detail {

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
  static constexpr bool DisableSource = cute::is_void_v<ElementC_>;
  using ElementC = cute::conditional_t<DisableSource, ElementD, ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = cute::conditional_t<DisableSource, GmemLayoutTagD, GmemLayoutTagC_>;
  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

  using TileShape_MN = decltype(select<0,1>(MmaTileShape_MNK{}));

  static constexpr int StagesC = 1;
  static constexpr int StagesD = 1;
  static constexpr int FragmentSize = 2;
  static constexpr int NumControlWarps = 4;
  static constexpr int NumEpilogueWarps = 16;

  using FusionOp = fusion::LinCombEltAct<cutlass::epilogue::thread::ReLu, ElementC, ElementC, void>;
  using FusionCallbacks = fusion::FusionCallbacks<
    Sm90TmaWarpSpecialized<1, 1, FragmentSize, false, false>,
    FusionOp, decltype(append(TileShape_MN{}, _1{})), Shape<_2,_1>
  >;

  using SmemLayoutAtomC = decltype(make_ordered_layout(TileShape_MN{}, Step<_1, _0>{}));

public:
  static_assert(cute::is_same_v<EpilogueTileType, EpilogueTileAuto>, "Don't specify epilogue tile with auto schedule");

  using CollectiveOp =
    cutlass::epilogue::collective::CollectiveEpilogue<
    Xe4DmaWarpSpecialized<StagesC, StagesD, FragmentSize, NumControlWarps, NumEpilogueWarps>,
    ElementC, GmemStrideTypeD, SmemLayoutAtomC, TileShape_MN, FusionCallbacks
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
    typename detail::Xe4TmaBuilderImpl<
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
