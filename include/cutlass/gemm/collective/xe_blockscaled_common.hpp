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
    class SelectedGmemTiledCopyScale
  >
CUTLASS_DEVICE static auto
make_scale_copy_iterator(int coord, int l_coord, int k_tile_count, int sg_d, int mma_d) {
    constexpr int GROUP_K = 32;
    return make_tensor(make_inttuple_iter(make_coord(coord, 0, l_coord)),
                        make_layout(make_shape(Int<scale_traits_size>{}, Int<scale_traits_num>{}, sg_d / mma_d, k_tile_count),
                                    make_stride(E<0>{} * _16{}, E<0>{} * size<1>(typename SelectedGmemTiledCopyScale::BlockShape{}), E<1>{} * size<0>(typename SelectedGmemTiledCopyScale::BlockShape{}), E<1>{} * (sg_d / GROUP_K))));
}

// Helper to generate index data for GEMM offsets (M/N/Q dimension)
template <typename Element, int Count>
CUTLASS_DEVICE static auto
make_gemm_idx_by_m() {
    auto indices = make_tensor<uint8_t>(make_shape(Int<Count>{}));
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < Count; ++m) {
      indices(m) = sizeof_bits_v<Element> < 8 ? (m / 2) * 32 + (m % 2) * 8 : m * 8;
    }
    return indices;
}

template <typename Element, int Count>
CUTLASS_DEVICE static auto
make_gemm_idx_by_n() {
    auto indices = make_tensor<uint8_t>(make_shape(Int<Count>{}));
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Count; ++n) {
      indices(n) = sizeof_bits_v<Element> < 8 ? n * 32 : n * 16;
    }
    return indices;
}

// Helper to generate K offsets
template <int Count, class BlockShape, int SG_Dim>
CUTLASS_DEVICE static auto
make_gemm_k_offsets() {
    auto offsets = make_tensor<uint16_t>(Layout<Shape<_1, _1, Int<Count>>, Stride<_0, _0, _1>>{});
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Count; ++k) {
      offsets(k) = static_cast<uint16_t>(k * size(BlockShape{}) * (SG_Dim / 16));
    }
    return offsets;
}

} // namespace cutlass::gemm::collective


