/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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
/*! \file
    \brief Block-wise quantization for SubgroupTensor on Intel Xe GPUs.
    
    Provides in-register quantization from high-precision types (half_t, BF16, FP32) to
    low-precision types (E4M3, E5M2, E2M1) with block-wise scale factors.
    
    This API is designed to enable fusion with GEMM prologue/epilogue operations,
    avoiding expensive register-to-memory conversions.
*/

#pragma once

#include <cute/config.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_sg.hpp>
#include <cute/algorithm/reorder.hpp>
#include <cute/util/sycl_vec.hpp>

#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/float8.h>

#include <sycl/sycl.hpp>

namespace cute {

//////////////////////////////////////////////////////////////////////////////
/// quantize_block_wise
///
/// Quantizes a high-precision source tensor to a low-precision destination
/// tensor with per-block scale factors following the MX format algorithm:
///   Step 0: Extract compile-time constants (M, N, NumBlocks) from TV layout
///   Step 1: Per-thread vertical abs-max into local_max[M][NumBlocks]
///   Step 2: Cross-lane reduce_over_group + compute per-block scale
///   Step 3: Write scale factors to output tensor
///   Step 4: Quantize: dst[i] = round_to_nearest(src[i] * scale)
///   Step 5: Reorder elements to destination layout
///
/// Coordinates (m, block_id) are derived at compile time via
/// tv_layout(0, C<0>{} + v) inside CUTE_UNROLL for loops.
///
/// Preconditions:
///   - BlockSize >= sg_size (16).  Each quantization block must span at
///     least one full subgroup width for compile-time block index derivation.
///   - The source TV layout's thread stride must map ONLY to dimension 1 (N).
///     The row index (M) must be thread-independent so that tv_layout(0, v)
///     yields the correct row for all threads.
///
/// @tparam BlockSize  Number of elements per quantization block
//////////////////////////////////////////////////////////////////////////////
template <int BlockSize,
          class SrcEngine, class SrcLayout, class SrcTVLayout,
          class DstEngine, class DstLayout, class DstTVLayout,
          class ScaleEngine, class ScaleLayout, class ScaleTVLayout>
CUTE_HOST_DEVICE
void quantize_block_wise(
  SubgroupTensor<SrcEngine, SrcLayout, SrcTVLayout> const& src,
  SubgroupTensor<DstEngine, DstLayout, DstTVLayout>& dst,
  SubgroupTensor<ScaleEngine, ScaleLayout, ScaleTVLayout>& scale)
{
  using SrcType   = typename SrcEngine::element_type;
  using DstType   = typename DstEngine::element_type;
  using ScaleType = typename ScaleEngine::element_type;

  // ---------- Step 0: Compile-time constants ----------
  constexpr auto tv_layout     = SrcTVLayout{};
  constexpr auto logical_shape = atuple_coshape(tv_layout);
  constexpr int  M             = get<0>(logical_shape);
  constexpr int  N             = get<1>(logical_shape);
  constexpr int  NumBlocks     = N / BlockSize;
  constexpr int  NumValues     = cosize_v<SrcLayout>;

  static_assert(N % BlockSize == 0,
    "BlockSize must evenly divide the N dimension");
  static_assert(BlockSize >= intel::sg_size,
    "BlockSize must be >= subgroup size for compile-time block index");
  static_assert(BlockSize % intel::sg_size == 0,
    "BlockSize must be a multiple of the subgroup size for lane-independent block indices");

  // Verify dst logical shape matches src
  constexpr auto dst_shape = atuple_coshape(DstTVLayout{});
  static_assert(get<0>(dst_shape) == M && get<1>(dst_shape) == N,
    "Source and destination must have the same logical shape");

  // Verify scale shape is (M, NumBlocks)
  // When NumBlocks == 1, atuple_coshape may collapse to a 1D scalar
  constexpr auto scale_shape = atuple_coshape(ScaleTVLayout{});
  if constexpr (NumBlocks > 1) {
    static_assert(get<0>(scale_shape) == M,
      "Scale M dimension must match source M");
    static_assert(get<1>(scale_shape) == NumBlocks,
      "Scale N dimension must equal N/BlockSize");
  } else {
    // For the single-block case, allow rank-collapsed layouts but enforce:
    //   - Leading dimension equals M
    //   - Total number of elements equals M * NumBlocks
    static_assert(get<0>(scale_shape) == M,
      "Scale M dimension must match source M");
    constexpr int scale_num_elems = cosize_v<ScaleLayout>;
    static_assert(scale_num_elems == M * NumBlocks,
      "Scale tensor must have M * (N/BlockSize) elements");
  }

  // ---------- Step 1: Per-thread vertical abs-max ----------
  float local_max[M][NumBlocks];
  CUTE_UNROLL
  for (int m = 0; m < M; m++) {
    CUTE_UNROLL
    for (int b = 0; b < NumBlocks; b++) {
      local_max[m][b] = 0.0f;
    }
  }

  CUTE_UNROLL
  for (int v = 0; v < NumValues; v++) {
    auto coord    = tv_layout(0, C<0>{} + v);
    int  m        = get<0>(coord);
    int  block_id = get<1>(coord) / BlockSize;
    float abs_val = sycl::fabs(static_cast<float>(src.tensor()(v)));
    local_max[m][block_id] = sycl::fmax(local_max[m][block_id], abs_val);
  }

  // ---------- Step 2: Cross-lane horizontal reduction ----------
  const float target_max = static_cast<float>(cutlass::platform::numeric_limits<DstType>::max());
  using FloatToScale = cutlass::NumericConverter<ScaleType, float,
                                                 cutlass::FloatRoundStyle::round_toward_zero>;
  ScaleType block_scale[M][NumBlocks];
  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  CUTE_UNROLL
  for (int m = 0; m < M; m++) {
    CUTE_UNROLL
    for (int b = 0; b < NumBlocks; b++) {
      local_max[m][b] = reduce_over_group(sg, local_max[m][b],
                                          sycl::maximum<float>{});
      float amax = local_max[m][b];
      float s = (amax > 0.0f) ? (target_max / amax) : 0.0f;
      block_scale[m][b] = FloatToScale{}(s);
    }
  }

  // ---------- Step 3: Write scales to output ----------
  constexpr int ScaleNumValues = cosize_v<ScaleLayout>;
  CUTE_UNROLL
  for (int sv = 0; sv < ScaleNumValues; sv++) {
    auto sc = ScaleTVLayout{}(0, C<0>{} + sv);
    scale.tensor()(sv) = block_scale[int(get<0>(sc))][int(get<1>(sc))];
  }

  // ---------- Step 4: Quantize each element ----------
  using FloatToDst = cutlass::NumericConverter<DstType, float,
                                                cutlass::FloatRoundStyle::round_to_nearest>;

  auto tmp_dst_frag = make_fragment_like<DstType>(src.tensor());

  CUTE_UNROLL
  for (int v = 0; v < NumValues; v++) {
    auto coord    = tv_layout(0, C<0>{} + v);
    int  m        = get<0>(coord);
    int  block_id = get<1>(coord) / BlockSize;
    float val    = static_cast<float>(src.tensor()(v));
    float scaled = val * static_cast<float>(block_scale[m][block_id]);
    tmp_dst_frag(v) = FloatToDst{}(scaled);
  }

  // ---------- Step 5: Reorder to destination layout ----------
  auto tmp_dst_sgt = make_subgroup_tensor(tmp_dst_frag, tv_layout);
  reorder(tmp_dst_sgt, dst);
}

//////////////////////////////////////////////////////////////////////////////
/// quantize — Public API dispatcher
///
/// Delegates to quantize_block_wise.  See that function for the full
/// algorithm description and preconditions.
///
/// @tparam BlockSize  Number of elements per quantization block
/// @param  src        High-precision input  (half_t, BF16, FP32)
/// @param  dst        Low-precision output  (E4M3, E5M2, E2M1)
/// @param  scale      Block-wise scale factors output
//////////////////////////////////////////////////////////////////////////////
template <int BlockSize,
          class SrcEngine, class SrcLayout, class SrcTVLayout,
          class DstEngine, class DstLayout, class DstTVLayout,
          class ScaleEngine, class ScaleLayout, class ScaleTVLayout>
CUTE_HOST_DEVICE
void quantize(
  SubgroupTensor<SrcEngine, SrcLayout, SrcTVLayout> const& src,
  SubgroupTensor<DstEngine, DstLayout, DstTVLayout>& dst,
  SubgroupTensor<ScaleEngine, ScaleLayout, ScaleTVLayout>& scale)
{
  quantize_block_wise<BlockSize>(src, dst, scale);
}

// tensor-wise quantize: single global scale factor for the entire tensor (no blocking)
template <class SrcEngine, class SrcLayout, class SrcTVLayout,
          class DstEngine, class DstLayout, class DstTVLayout,
          class ScaleT>
CUTE_HOST_DEVICE
void quantize(
  SubgroupTensor<SrcEngine, SrcLayout, SrcTVLayout> const& src,
  SubgroupTensor<DstEngine, DstLayout, DstTVLayout>& dst,
  ScaleT& scale)
{
  static_assert(cute::dependent_false<ScaleT>, "Not implemented: tensor-wise quantization with a single global scale factor.");
}

} // namespace cute
