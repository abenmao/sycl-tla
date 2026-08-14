/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

#include "default_gemm_configuration.hpp"

namespace cutlass {
namespace gemm {
namespace device {
using namespace cute;

// This type is only intended to demonstrate porting 2.x kernels to 3.0
template<
  class OperatorClass, class ArchTag,
  class ElementA, class LayoutA,
  class ElementB, class LayoutB,
  class ElementC, class LayoutC,
  class ElementAccumulator,
  class ElementOutput>
struct XeDefaultGemmConfigurationToCutlass3Types {
  static_assert(sizeof(ElementA) == 0, "No valid XeDefaultGemmConfigurationToCutlass3Types configuration exists.");
};

///////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpXe_OperandA;

template <typename Element, typename Layout, int Alignment, int SizeK>
struct DefaultGemm_TensorOpXe_OperandB;

//
// Bfloat16
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpXe_OperandA<bfloat16_t, layout::RowMajor, 32, 32>
{
  using GmemTiledCopy = XE_2D_U16x32x32_LD_N;
};

/// Operand A - Column-major (M-major)
template <int SizeK>
struct DefaultGemm_TensorOpXe_OperandA<bfloat16_t, layout::ColumnMajor, 32, SizeK>
{
  // Gmem
  using GmemTiledCopy = XE_2D_U16x16x16_LD_T;
};

/// Operand B - Row-major (N-Major)
template <>
struct DefaultGemm_TensorOpXe_OperandB<bfloat16_t, layout::RowMajor, 32, 32>
{
  using GmemTiledCopy = XE_2D_U16x32x32_LD_V;
};

/// Operand B - Column-major (K-major)
template <int SizeK>
struct DefaultGemm_TensorOpXe_OperandB<bfloat16_t, layout::ColumnMajor, 32, SizeK>
{
  // Gmem
  using GmemTiledCopy = XE_2D_U16x16x16_LD_T;
};

}

///////////////////////////////////////////////////////////////////////////////

// Intel XE MMA F32BF16
template <typename LayoutA, typename LayoutB, typename LayoutC, typename ElementOutput>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    bfloat16_t, LayoutA,
    bfloat16_t, LayoutB,
    float, LayoutC,
    ElementOutput>
{
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
               Layout<TileShape>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = detail::DefaultGemm_TensorOpXe_OperandA<
    bfloat16_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = detail::DefaultGemm_TensorOpXe_OperandB<
    bfloat16_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
      cute::bfloat16_t, LayoutA, 1,
      cute::bfloat16_t, LayoutB, 1,
      float,
      TileShape, Shape<_1, _1, _1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = epilogue::fusion::LinearCombination<float, float>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelXeXMX16,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
      TileShape, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 1,
      ElementOutput, LayoutC, 1,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp
    >::CollectiveOp;
};

///////////////////////////////////////////////////////////////////////////////

// Intel XE MMA F32BF16
// ElementC - > void
// ElementCompute and ElementOutput different in LinearCombination
template <typename LayoutA, typename LayoutB, typename LayoutC, typename ElementOutput>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    bfloat16_t, LayoutA,
    bfloat16_t, LayoutB,
    void, LayoutC,
    ElementOutput>
{
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
               Layout<TileShape>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = detail::DefaultGemm_TensorOpXe_OperandA<
    bfloat16_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = detail::DefaultGemm_TensorOpXe_OperandB<
    bfloat16_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
      cute::bfloat16_t, LayoutA, 1,
      cute::bfloat16_t, LayoutB, 1,
      float,
      TileShape, Shape<_1, _1, _1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  //using EpilogueOp = epilogue::fusion::LinearCombination<ElementOutput, float>;
  using EpilogueOp = epilogue::fusion::LinearCombination<cute::bfloat16_t, float>;


  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelXeXMX16,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
      TileShape, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      void, LayoutC, 1,
      cute::bfloat16_t, LayoutC, 1,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp
    >::CollectiveOp;
};

///////////////////////////////////////////////////////////////////////////////

// Intel XE MMA F32BF16
// D=Ax B + C;  => BF16=BF16xBF16+BF16 <=>BF16=FP32+BF16
// ElementAccumulator and ElementC are different types.
template <
  typename LayoutA,
  typename LayoutB,
  typename LayoutC,
  typename ElementAccumulator,
  typename ElementOutput>
struct XeDefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    bfloat16_t, LayoutA,
    bfloat16_t, LayoutB,
    bfloat16_t, LayoutC,
    ElementAccumulator,
    ElementOutput>
{
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
               Layout<TileShape>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = detail::DefaultGemm_TensorOpXe_OperandA<
    bfloat16_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = detail::DefaultGemm_TensorOpXe_OperandB<
    bfloat16_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
      cute::bfloat16_t, LayoutA, 1,
      cute::bfloat16_t, LayoutB, 1,
      ElementAccumulator,
      TileShape, Shape<_1, _1, _1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using EpilogueOp = epilogue::fusion::LinearCombination<ElementOutput, float>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelXeXMX16,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
      TileShape, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, float,
      bfloat16_t, LayoutC, 1,
      ElementOutput, LayoutC, 1,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp
    >::CollectiveOp;
};


///////////////////////////////////////////////////////////////////////////////

namespace detail {

//
// half
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpXe_OperandA<half_t, layout::RowMajor, 32, 32>
{
  using GmemTiledCopy = XE_2D_U16x32x32_LD_N;
};

/// Operand A - Column-major (M-major)
template <int SizeK>
struct DefaultGemm_TensorOpXe_OperandA<half_t, layout::ColumnMajor, 32, SizeK>
{
  // Gmem
  using GmemTiledCopy = XE_2D_U16x16x16_LD_T;
};

/// Operand B - Row-major (N-Major)
template <>
struct DefaultGemm_TensorOpXe_OperandB<half_t, layout::RowMajor, 32, 32>
{
  using GmemTiledCopy = XE_2D_U16x32x32_LD_V;
};

/// Operand B - Column-major (K-major)
template <int SizeK>
struct DefaultGemm_TensorOpXe_OperandB<half_t, layout::ColumnMajor, 32, SizeK>
{
  // Gmem
  using GmemTiledCopy = XE_2D_U16x16x16_LD_T;
};

}

///////////////////////////////////////////////////////////////////////////////

// Intel XE MMA F32F16
template <typename LayoutA, typename LayoutB, typename LayoutC, typename ElementOutput>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    half_t, LayoutA,
    half_t, LayoutB,
    float, LayoutC,
    ElementOutput>
{
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
               Layout<TileShape>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = detail::DefaultGemm_TensorOpXe_OperandA<
    half_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = detail::DefaultGemm_TensorOpXe_OperandB<
    half_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
    cute::bfloat16_t, LayoutA, 1,
    cute::bfloat16_t, LayoutB, 1,
    float,
    TileShape, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using EpilogueOp = epilogue::fusion::LinearCombination<float, float>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelXeXMX16,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
      TileShape, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      float, LayoutC, 1,
      ElementOutput, LayoutC, 1,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp
    >::CollectiveOp;
};

///////////////////////////////////////////////////////////////////////////////

namespace detail {

//
// int8
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpXe_OperandA<int8_t, layout::RowMajor, 32, 32>
{
  using GmemTiledCopy = XE_2D_Packed_U8x32x32_LD_N;
};

/// Operand A - Column-major (M-major)
template <int SizeK>
struct DefaultGemm_TensorOpXe_OperandA<int8_t, layout::ColumnMajor, 32, SizeK>
{
  // Gmem
  using GmemTiledCopy = XE_2D_U8x32x8_LD_T;
};

/// Operand B - Row-major (N-Major)
template <>
struct DefaultGemm_TensorOpXe_OperandB<int8_t, layout::RowMajor, 32, 32>
{
  using GmemTiledCopy = XE_2D_U8x32x32_LD_V;
};

/// Operand B - Column-major (K-major)
template <int SizeK>
struct DefaultGemm_TensorOpXe_OperandB<int8_t, layout::ColumnMajor, 32, SizeK>
{
  // Gmem
  using GmemTiledCopy = XE_2D_U8x16x32_LD_T;
};

}

///////////////////////////////////////////////////////////////////////////////

// Intel XE MMA F32F16
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    bfloat16_t, LayoutA,
    bfloat16_t, LayoutB,
    bfloat16_t, LayoutC,
    bfloat16_t>
{
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_BF16BF16BF16BF16_TT>,
               Layout<TileShape>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = detail::DefaultGemm_TensorOpXe_OperandA<
    bfloat16_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = detail::DefaultGemm_TensorOpXe_OperandB<
    bfloat16_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
    bfloat16_t, LayoutA, 1,
    bfloat16_t, LayoutB, 1,
    bfloat16_t,
    TileShape, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using EpilogueOp = epilogue::fusion::LinearCombination<bfloat16_t, bfloat16_t>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelXeXMX16,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
      TileShape, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      bfloat16_t, bfloat16_t,
      bfloat16_t, LayoutC, 1,
      bfloat16_t, LayoutC, 1,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp
    >::CollectiveOp;
};

///////////////////////////////////////////////////////////////////////////////

// Intel XE MMA F32F16
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    half_t, LayoutA,
    half_t, LayoutB,
    half_t, LayoutC,
    half_t>
{
  using TileShape = Shape<_256, _256, _32>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F16F16F16F16_TT>,
               Layout<TileShape>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = detail::DefaultGemm_TensorOpXe_OperandA<
    half_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = detail::DefaultGemm_TensorOpXe_OperandB<
    half_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
    half_t, LayoutA, 1,
    half_t, LayoutB, 1,
    half_t,
    TileShape, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using EpilogueOp = epilogue::fusion::LinearCombination<half_t, half_t>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelXeXMX16,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
      TileShape, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      half_t, half_t,
      half_t, LayoutC, 1,
      half_t, LayoutC, 1,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp
    >::CollectiveOp;
};

///////////////////////////////////////////////////////////////////////////////

// Intel XE MMA S32S8
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    int8_t, LayoutA,
    int8_t, LayoutB,
    int32_t, LayoutC,
    int32_t>
{
  using TileShape = Shape<_256, _256, _32>;

  using DispatchPolicy = MainloopIntelXeXMX16<3>;
  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x32_S32S8S8S32_TT>,
               Layout<TileShape>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = detail::DefaultGemm_TensorOpXe_OperandA<
    int8_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = detail::DefaultGemm_TensorOpXe_OperandB<
    int8_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    int8_t, TagToStrideA_t<LayoutA>,
    int8_t, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,  // A
    GmemTiledCopyB, void, void, cute::identity   // B
  >;

  using EpilogueOp = epilogue::fusion::LinearCombination<int32_t, int32_t>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelXeXMX16,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    epilogue::IntelXeXMX16,
    TileShape,
    int32_t, TagToStrideC_t<LayoutC>,
    int32_t, TagToStrideC_t<LayoutC>,
    FusionCallBacks,
    XE_2D_U32x8x16_LD_N, void, void,
    XE_2D_U32x8x16_ST_N, void, void>;
};

///////////////////////////////////////////////////////////////////////////////

namespace detail {

//
// tfloat32
//

/// Operand A - Row-major (K-Major)
template <>
struct DefaultGemm_TensorOpXe_OperandA<tfloat32_t, layout::RowMajor, 32, 32>
{
  using GmemTiledCopy = XE_2D_TF32x32x16_LD_N;
};

/// Operand A - Column-major (M-major)
template <int SizeK>
struct DefaultGemm_TensorOpXe_OperandA<tfloat32_t, layout::ColumnMajor, 32, SizeK>
{
  // Gmem
  // TODO(Codeplay): transposed version is not implemented.
  using GmemTiledCopy = XE_2D_TF32x32x16_LD_N;
};

/// Operand B - Row-major (N-Major)
template <>
struct DefaultGemm_TensorOpXe_OperandB<tfloat32_t, layout::RowMajor, 32, 32>
{
  using GmemTiledCopy = XE_2D_U32x32x16_LD_N;
};

/// Operand B - Column-major (K-major)
template <int SizeK>
struct DefaultGemm_TensorOpXe_OperandB<tfloat32_t, layout::ColumnMajor, 32, SizeK>
{
  // Gmem
  using GmemTiledCopy = XE_2D_U32x16x8_LD_T;
};

}

///////////////////////////////////////////////////////////////////////////////

// Intel XE MMA S32S8
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    tfloat32_t, LayoutA,
    tfloat32_t, LayoutB,
    float, LayoutC,
    float>
{
  using TileShape = Shape<_256, _256, _32>;

  using DispatchPolicy = MainloopIntelXeXMX16<3>;
  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x8_F32TF32TF32F32_TT>,
               Layout<TileShape>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = detail::DefaultGemm_TensorOpXe_OperandA<
    tfloat32_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = detail::DefaultGemm_TensorOpXe_OperandB<
    tfloat32_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    tfloat32_t, TagToStrideA_t<LayoutA>,
    tfloat32_t, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,  // A
    GmemTiledCopyB, void, void, cute::identity   // B
  >;

  using EpilogueOp = epilogue::fusion::LinearCombination<float, float>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelXeXMX16,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    epilogue::IntelXeXMX16,
    TileShape,
    float, TagToStrideC_t<LayoutC>,
    float, TagToStrideC_t<LayoutC>,
    FusionCallBacks,
    XE_2D_U32x8x16_LD_N, void, void,
    XE_2D_U32x8x16_ST_N, void, void>;
};

///////////////////////////////////////////////////////////////////////////////

namespace detail {

  //
  // float_e5m2_t
  //
  
  /// Operand A - Row-major (K-Major)
  template <>
  struct DefaultGemm_TensorOpXe_OperandA<float_e5m2_t, layout::RowMajor, 32, 32>
  {
    using GmemTiledCopy = XE_2D_U8x32x32_LD_N;
  };
  
  /// Operand A - Column-major (M-major)
  template <int SizeK>
  struct DefaultGemm_TensorOpXe_OperandA<float_e5m2_t, layout::ColumnMajor, 32, SizeK>
  {
    // Gmem
    using GmemTiledCopy = XE_2D_U8x16x8_LD_T;
  };
  
  /// Operand B - Row-major (N-Major)
  template <>
  struct DefaultGemm_TensorOpXe_OperandB<float_e5m2_t, layout::RowMajor, 32, 32>
  {
    using GmemTiledCopy = XE_2D_U8x32x32_LD_V;
  };
  
  /// Operand B - Column-major (K-major)
  template <int SizeK>
  struct DefaultGemm_TensorOpXe_OperandB<float_e5m2_t, layout::ColumnMajor, 32, SizeK>
  {
    // Gmem
    using GmemTiledCopy = XE_2D_U8x16x16_LD_T;
  };
  
  }

// Intel XE MMA FP32FP8
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    float_e5m2_t, LayoutA,
    float_e5m2_t, LayoutB,
    float, LayoutC,
    float>
{
  using TileShape = Shape<_256, _256, _32>;

  using DispatchPolicy = MainloopIntelW8A8<3>;
  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
               Layout<TileShape>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = detail::DefaultGemm_TensorOpXe_OperandA<
    float_e5m2_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = detail::DefaultGemm_TensorOpXe_OperandB<
    float_e5m2_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    float_e5m2_t, TagToStrideA_t<LayoutA>,
    float_e5m2_t, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,  // A
    GmemTiledCopyB, void, void, cute::identity   // B
  >;

  using EpilogueOp = epilogue::fusion::LinearCombination<float, float>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelXeXMX16,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    epilogue::IntelXeXMX16,
    TileShape,
    float, TagToStrideC_t<LayoutC>,
    float, TagToStrideC_t<LayoutC>,
    FusionCallBacks,
    XE_2D_U32x8x16_LD_N, void, void,
    XE_2D_U32x8x16_ST_N, void, void>;
};

template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    float_e4m3_t, LayoutA,
    float_e4m3_t, LayoutB,
    float, LayoutC,
    float>
{
  using TileShape = Shape<_256, _256, _32>;

  using DispatchPolicy = MainloopIntelW8A8<3>;
  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
               Layout<TileShape>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = detail::DefaultGemm_TensorOpXe_OperandA<
    float_e5m2_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = detail::DefaultGemm_TensorOpXe_OperandB<
    float_e5m2_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    float_e4m3_t, TagToStrideA_t<LayoutA>,
    float_e4m3_t, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,  // A
    GmemTiledCopyB, void, void, cute::identity   // B
  >;

  using EpilogueOp = epilogue::fusion::LinearCombination<float, float>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelXeXMX16,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    epilogue::IntelXeXMX16,
    TileShape,
    float, TagToStrideC_t<LayoutC>,
    float, TagToStrideC_t<LayoutC>,
    FusionCallBacks,
    XE_2D_U32x8x16_LD_N, void, void,
    XE_2D_U32x8x16_ST_N, void, void>;
};

///////////////////////////////////////////////////////////////////////////////

// Intel XE MMA f16f8f32
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    cute::half_t, LayoutA,
    float_e5m2_t, LayoutB,
    float, LayoutC,
    float>
{
  using TileShape = Shape<_256, _256, _32>;

  using DispatchPolicy = MainloopIntelXeXMX16MixedPrecision<3>;
  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
               Layout<TileShape>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = detail::DefaultGemm_TensorOpXe_OperandA<
    cute::half_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = detail::DefaultGemm_TensorOpXe_OperandB<
    float_e5m2_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    cute::half_t, TagToStrideA_t<LayoutA>,
    cute::tuple<float_e5m2_t>, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,  // A
    GmemTiledCopyB, void, void, cute::identity   // B
  >;

  using EpilogueOp = epilogue::fusion::LinearCombination<float, float>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelXeXMX16,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    epilogue::IntelXeXMX16,
    TileShape,
    float, TagToStrideC_t<LayoutC>,
    float, TagToStrideC_t<LayoutC>,
    FusionCallBacks,
    XE_2D_U32x8x16_LD_N, void, void,
    XE_2D_U32x8x16_ST_N, void, void>;
};

///////////////////////////////////////////////////////////////////////////////

// Intel XE MMA f16s8f32
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    cute::half_t, LayoutA,
    int8_t, LayoutB,
    float, LayoutC,
    float>
{
  using TileShape = Shape<_256, _256, _32>;

  using DispatchPolicy = MainloopIntelXeXMX16MixedPrecision<3>;
  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
               Layout<TileShape>,
               Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  // A
  static constexpr int kAlignmentA = 32;
  using DefaultOperandA = detail::DefaultGemm_TensorOpXe_OperandA<
    half_t, LayoutA, kAlignmentA, 32>;
  using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;
  
  // B
  static constexpr int kAlignmentB = 32;
  using DefaultOperandB = detail::DefaultGemm_TensorOpXe_OperandB<
    int8_t, LayoutB, kAlignmentB, 32>;
  using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

  // Mainloop
  using CollectiveMainloop = collective::CollectiveMma<
    DispatchPolicy, TileShape,
    cute::half_t, TagToStrideA_t<LayoutA>,
    cute::tuple<int8_t>, TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,  // A
    GmemTiledCopyB, void, void, cute::identity   // B
  >;

  using EpilogueOp = epilogue::fusion::LinearCombination<float, float>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
    epilogue::IntelXeXMX16,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))
  >;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    epilogue::IntelXeXMX16,
    TileShape,
    float, TagToStrideC_t<LayoutC>,
    float, TagToStrideC_t<LayoutC>,
    FusionCallBacks,
    XE_2D_U32x8x16_LD_N, void, void,
    XE_2D_U32x8x16_ST_N, void, void>;
};

///////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass
