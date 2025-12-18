/***************************************************************************************************
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

#include "cutlass/cutlass.h"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"

namespace cutlass::gemm::collective {

using namespace cute;

// -----------------------------------------------------------------------------
// Traits for scale loading
// -----------------------------------------------------------------------------

template <class datatype, size_t height, size_t width, class Stride = cute::Stride<_1, int64_t, int64_t>, class = void>
struct scale_copy_traits {
  static_assert(cute::dependent_false<cute::tuple<datatype, Int<height>, Int<width>, Stride>>, "scale_copy_traits not defined");
};

// 8 bits specialization for height <= 1 (covers 1 and 0 if applicable)
template<class datatype, size_t height, size_t width, class stride>
struct scale_copy_traits<datatype, height, width, stride,
          std::enable_if_t<sizeof_bits_v<datatype> == 8 && height <= 1>> {
  using type = XE_2D_U8x1x16_LD_N;
};

// 8 bits specialization for height == 2
template<class datatype, size_t width, class stride>
struct scale_copy_traits<datatype, 2, width, stride,
          std::enable_if_t<sizeof_bits_v<datatype> == 8>> {
  using type = XE_2D_U8x2x16_LD_N;
};

template<class datatype, size_t width, class stride>
struct scale_copy_traits<datatype, 4, width, stride,
          std::enable_if_t<sizeof_bits_v<datatype> == 8>> {
  using type = XE_2D_U8x4x16_LD_N;
};

// -----------------------------------------------------------------------------
// Tiled Copy for Scale
// -----------------------------------------------------------------------------

template<
  class SelectedGmemTiledCopyScale,
  class StrideScale,
  class ElementScale
>
struct TiledCopyScaleTraits {
  using CopyThreadShape = Shape<_1, Int<intel::sg_size>>;
  using CopyThreadShapeRev = decltype(cute::reverse(CopyThreadShape{}));

  using traits_load_scale = Copy_Traits<SelectedGmemTiledCopyScale, StrideScale>;
  using atom_load_scale = Copy_Atom<traits_load_scale, ElementScale>;
  using val_layout_load_scale = decltype(make_layout(shape_div(typename traits_load_scale::BlockShape{}, CopyThreadShapeRev{})));
  using Copy_Scale = decltype(make_tiled_copy(atom_load_scale{}, Layout<CopyThreadShapeRev>{}, val_layout_load_scale{}));
};

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------


// Helper to create scale copy iterator
template <
    int scale_traits_size,
    int scale_traits_num,
    int sg_k,
    class TiledCopyScale
  >
CUTLASS_DEVICE static auto
make_scale_copy_iterator(int coord, int l_coord, int k_tile_count) {
    constexpr int GROUP_K = 32;
    return make_tensor(make_inttuple_iter(make_coord(coord, 0, l_coord)),
                       make_layout(make_shape(Int<scale_traits_size>{}, Int<scale_traits_num>{}, _1{}, k_tile_count),
                                   make_stride(E<0>{} * _16{}, E<0>{} * size<1>(typename TiledCopyScale::BlockShape{}),
                                               E<1>{} * size<0>(typename TiledCopyScale::BlockShape{}), E<1>{} * (sg_k / GROUP_K))));
}

// Helper to generate index data for GEMM offsets (M/N/Q dimension)
template <int GemmIterM, typename BlockShape>
CUTLASS_DEVICE static auto
make_scaled_offsets_m() {
  auto offsets = make_tensor<uint16_t>(Layout<Shape<_1, Int<GemmIterM>, _1>, Stride<_0, _1, _0>>{});
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < GemmIterM; ++m) {
    offsets(m) = (m / 2) * decltype(size(BlockShape{}))::value + (m % 2) * 8;
  }
  return offsets;
}

template <int GemmIterN, typename BlockShape>
CUTLASS_DEVICE static auto
make_scaled_offsets_n() {
  auto offsets = make_tensor<uint16_t>(Layout<Shape<_1, Int<GemmIterN>, _1>, Stride<_0, _1, _0>>{});
  CUTLASS_PRAGMA_UNROLL
  for (int n = 0; n < GemmIterN; ++n) {
    offsets(n) = n * decltype(size(BlockShape{}))::value;
  }
  return offsets;
}

// Helper to generate K offsets
template <int GemmIterK, int MMA_K, int GROUP_K, typename BlockShape>
CUTLASS_DEVICE static auto
make_scaled_offsets_k() {
  auto offsets = make_tensor<uint16_t>(Layout<Shape<_1, _1, Int<GemmIterK>>, Stride<_0, _0, _1>>{});
  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < GemmIterK; ++k) {
    offsets(k) = (MMA_K / GROUP_K) * decltype(size<1>(BlockShape{}))::value * k;
  }
  return offsets;
}

template <typename ScaleCopy, typename Element, int SG_MN, int SG_K, int GROUP_K, typename Tensor>
CUTLASS_DEVICE static auto
make_scaled_copy(Tensor const& tensor, int mn_coord, int l_coord, int k_tile_count) {
  using Stride = cute::remove_cvref_t<decltype(tensor.stride())>;
  using NonVoidScaleCopy = typename scale_copy_traits<Element, SG_K / GROUP_K, SG_MN>::type;
  using SelectedCopyScale = cute::conditional_t<cute::is_void_v<ScaleCopy>, NonVoidScaleCopy, ScaleCopy>;
  using Copy_Scale = typename TiledCopyScaleTraits<SelectedCopyScale, Stride, Element>::Copy_Scale;

  constexpr auto SubgroupSize = 16;
  static constexpr auto scale_traits_size = decltype(size(typename SelectedCopyScale::BlockShape{}))::value / SubgroupSize;
  static constexpr auto scale_traits_num = SG_MN / size<1>(typename SelectedCopyScale::BlockShape{});

  auto tiled_copy = Copy_Scale{}.with(tensor);

  auto copy_iter = make_scale_copy_iterator<scale_traits_size, scale_traits_num, SG_K, SelectedCopyScale>(mn_coord, l_coord, k_tile_count);

  auto fragment = make_tensor<Element>(Layout<Shape<Int<scale_traits_size>, Int<scale_traits_num>, _1>>{});

  return cute::make_tuple(tiled_copy, copy_iter, fragment);
}

template <int GemmIterM, int GemmIterN, int GemmIterK, int MMA_K, int GROUP_K, typename BlockShapeA, typename BlockShapeB>
CUTLASS_DEVICE static auto
make_scaled_offsets() {
  return cute::make_tuple(make_scaled_offsets_m<GemmIterM, BlockShapeA>(),
                          make_scaled_offsets_n<GemmIterN, BlockShapeB>(),
                          make_scaled_offsets_k<GemmIterK, MMA_K, GROUP_K, BlockShapeA>(),
                          make_scaled_offsets_k<GemmIterK, MMA_K, GROUP_K, BlockShapeB>());
}

} // namespace cutlass::gemm::collective


