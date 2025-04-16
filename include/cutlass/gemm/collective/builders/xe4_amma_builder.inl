/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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

#include "cutlass/gemm/collective/collective_mma.hpp"

namespace cutlass::gemm::collective {

template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
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
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,    // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    ClusterShape_MNK, // Static cluster shape or dynamic (int, int, _1)
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cutlass::detail::is_kernel_tag_of_v<KernelScheduleType, KernelTmaWarpSpecializedXe4>>>
{
  #ifdef SYCL_NVIDIA_TARGET
  static_assert(cutlass::detail::dependent_false<arch::Xe4>,"Trying to use Intel pipeline on Non Intel hardware");
  #endif

  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");

  static constexpr auto majorA = cutlass::gemm::detail::is_mn_major_A<GmemLayoutATag>() ? cute::xe4::GMMA::Major::MN : cute::xe4::GMMA::Major::K;
  static constexpr auto majorB = cutlass::gemm::detail::is_mn_major_B<GmemLayoutBTag>() ? cute::xe4::GMMA::Major::MN : cute::xe4::GMMA::Major::K;

  using TiledMma = decltype(cute::make_tiled_mma(
    cute::xe4::GMMA::ss_op_selector<ElementA, ElementB, ElementAccumulator,
    decltype(cute::product_each(TileShape_MNK{})), ClusterShape_MNK, majorA, majorB>()));

  static constexpr int PipelineStages = StageCountType::value;
  static constexpr int SchedulerPipelineStageCount = 3;
  static constexpr int AccumulatorPipelineStageCount = 1;

  using DispatchPolicy =
    cutlass::gemm::MainloopXe4DmaGmmaWarpSpecialized<
        PipelineStages,
        SchedulerPipelineStageCount,
        AccumulatorPipelineStageCount,
        ClusterShape_MNK
    >;
    
  static constexpr slm_matrix_type cmTypeA =
    cutlass::gemm::detail::is_mn_major_A<GmemLayoutATag>() ? slm_matrix_type::type2 : slm_matrix_type::type1;

  using GmemTiledCopyA =
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::xe4::ASYNC_TENSOR_LOAD<cmTypeA>,
      cute::xe4::ASYNC_TENSOR_LOAD_MULTICAST<cmTypeA>
    >;

  using GmemTiledCopyB =
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::xe4::ASYNC_TENSOR_LOAD<slm_matrix_type::type1>,
      cute::xe4::ASYNC_TENSOR_LOAD_MULTICAST<slm_matrix_type::type1>
    >;

  using SmemLayoutAtomA =
    cute::conditional_t<
      majorA == cute::xe4::GMMA::Major::K,
      decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenRowMajor{})),
      decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenColMajor{}))
    >;

  using SmemLayoutAtomB =
    cute::conditional_t<
      majorB == cute::xe4::GMMA::Major::K,
      decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenRowMajor{})),
      decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenColMajor{}))
    >;

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy,
    TileShape_MNK,
    ElementA,
    cutlass::gemm::TagToStrideA_t<GmemLayoutATag>,
    ElementB,
    cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>,
    TiledMma,
    GmemTiledCopyA,
    SmemLayoutAtomA,
    void,
    cute::identity,
    GmemTiledCopyB,
    SmemLayoutAtomB,
    void,
    cute::identity
  >;
};

} // namespace cutlass::gemm::collective
