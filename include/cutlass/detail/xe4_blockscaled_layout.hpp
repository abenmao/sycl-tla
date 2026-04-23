/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation. All rights reserved. All rights reserved.
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
    \brief Blocked Scale configs specific for Xe4 (AMMA Xe4) BlockScaled MMA
*/

#pragma once

#include "cutlass/layout/matrix.h"

#include "cute/int_tuple.hpp"
#include "cute/atom/mma_traits_xe4.hpp"
#include "cute/arch/xe4_util.hpp"

namespace cutlass::detail {

/////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute;

template<int SFVecSize, AMMA::Major major = AMMA::Major::MN>
struct Xe4BlockScaledBasicChunk {

  using Blk_MN    = _1;
  using Blk_SF    = _1;

  static_assert(major == AMMA::Major::MN);

  using SfMNMajorAtom = Layout< Shape< _1, Int<SFVecSize>>,
                                Stride<_1,             _0>>;

  using SfAtom    = SfMNMajorAtom;
};

template<int SFVecSize_>
struct Xe4BlockScaledConfig {
  // SF tensor layouts for block-scaled GEMM (MN-major, per Xe4BlockScaledBasicChunk).
  static constexpr int SFVecSize = SFVecSize_;

  // Type3 SF descriptor window height, derived from the hardware core-matrix geometry.
  static constexpr auto SfCoreMatrixSize = get_core_matrix_size<slm_matrix_type::type3, uint8_t>();
  static constexpr int SfDescMinRows = get_height<SfCoreMatrixSize>();
  using Xe4BlkScaledChunk = Xe4BlockScaledBasicChunk<SFVecSize>;
  using Blk_MN = typename Xe4BlkScaledChunk::Blk_MN;
  using Blk_SF = typename Xe4BlkScaledChunk::Blk_SF;
  using SfAtom = typename Xe4BlkScaledChunk::SfAtom;

  using LayoutSF = decltype(blocked_product(SfAtom{}, make_layout( make_shape(int32_t(0), int32_t(0), int32_t(0)),
                                                                  make_stride(_1{},       int32_t(0), int32_t(0)))));

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFA() {
    return LayoutSF{};
  }

  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_layoutSFB() {
    return LayoutSF{};
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFA.
  template < class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFA(ProblemShape problem_shape) {
    if constexpr (rank(ProblemShape{}) == 3) {
      auto [M, N, K] = problem_shape;
      return tile_to_shape(SfAtom{}, make_shape(M,K), Step<_1,_2>{});
    }
    else {
      auto [M, N, K, L] = problem_shape;
      return tile_to_shape(SfAtom{}, make_shape(M,K,L), Step<_1,_2,_3>{});
    }
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFB.
  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFB(ProblemShape problem_shape) {
    if constexpr (rank(ProblemShape{}) == 3) {
      auto [M, N, K] = problem_shape;
      return tile_to_shape(SfAtom{}, make_shape(N,K), Step<_1,_2>{});
    }
    else {
      auto [M, N, K, L] = problem_shape;
      return tile_to_shape(SfAtom{}, make_shape(N,K,L), Step<_1,_2,_3>{});
    }
  }

  template<class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_smem_layoutSFA(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {

    constexpr int MMA_NSF = size<2>(typename TiledMma::Shape_MNK{}) / SFVecSize;
    constexpr int M = size<0>(typename TiledMma::Shape_MNK{});
    // SF block granularity (both 1 for Xe4: one SF element per MN position).
    using Blk_MN    = typename Xe4BlkScaledChunk::Blk_MN;
    using Blk_SF    = typename Xe4BlkScaledChunk::Blk_SF;
    using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});

    using TL_VMNK = typename TiledMma::ThrLayoutVMNK;
    constexpr TL_VMNK tl_vmnk{};
    constexpr int MMA_M = cute::size<0>(TileShape_MNK{}) / cute::size<0>(tl_vmnk);
    constexpr int MMA_MBlk = MMA_M / Blk_MN{};

    // Basic storage block for new Scaling Factor Layouts
    using mnBasicBlockShape  =  Shape<_1, Int<MMA_MBlk>>;
    using mnBasicBlockStride = Stride<_1,       _1>;
    using kBasicBlockShape  = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
    using kBasicBlockStride = Stride<_0,            Int<MMA_MBlk>>;

    using mma_SFA_shape  = decltype( make_shape( mnBasicBlockShape{},  kBasicBlockShape{}));
    using mma_SFA_stride = decltype(make_stride( mnBasicBlockStride{}, kBasicBlockStride{}));

    using blk_Shape1 = Int<size<0>(TileShape_MNK{})/M>;
    using blk_Shape2 = Int<size<2>(TileShape_MNK{})/size<2>(typename TiledMma::Shape_MNK{})>;
    using blk_Stride1 =  Int<MMA_M /Blk_MN{} * Blk_Elems{} * MMA_NSF>;
    // Pad k-block stride so each descriptor's 8-row read window stays within its own k-block.
    using blk_Stride2_natural = Int<blk_Stride1{} / blk_Shape1{}>;
    using blk_Stride2 = Int<cute::max(int(blk_Stride2_natural{}), SfDescMinRows * MMA_MBlk)>;

    constexpr auto blk_Stride1_or_0 = [=] {
      if constexpr (blk_Shape1{} == 1) return _0{};
      else return blk_Stride1{};
    };

    using sSFA_shape     = decltype( make_shape( mma_SFA_shape{}, _1{},  make_shape( blk_Shape1{}, blk_Shape2{} )));
    using sSFA_stride    = decltype(make_stride(mma_SFA_stride{}, _0{},  make_stride( blk_Stride1_or_0(), blk_Stride2{})));
    using SmemLayoutAtomSFA = decltype(make_layout(sSFA_shape{}, sSFA_stride{}));
    return SmemLayoutAtomSFA{};
  }

  template<class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_smem_layoutSFB(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {

    constexpr int MMA_NSF = size<2>(typename TiledMma::Shape_MNK{}) / SFVecSize;
    constexpr int N = size<1>(typename TiledMma::Shape_MNK{});
    // SF block granularity (both 1 for Xe4: one SF element per MN position).
    using Blk_MN    = typename Xe4BlkScaledChunk::Blk_MN;
    using Blk_SF    = typename Xe4BlkScaledChunk::Blk_SF;
    using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});

    using TL_VMNK = typename TiledMma::ThrLayoutVMNK;
    constexpr TL_VMNK tl_vmnk{};
    constexpr int MMA_N = cute::size<1>(TileShape_MNK{});
    constexpr int MMA_NBlk = MMA_N / Blk_MN{};

    // Basic storage block for new Scaling Factor Layouts
    using mnBasicBlockShape  =  Shape<_1, Int<MMA_NBlk>>;
    using mnBasicBlockStride = Stride<_1,       _1>;
    using kBasicBlockShape  = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
    using kBasicBlockStride = Stride<_0,            Int<MMA_NBlk>>;

    using mma_SFB_shape  = decltype(make_shape( mnBasicBlockShape{},  kBasicBlockShape{}));
    using mma_SFB_stride = decltype(make_stride(mnBasicBlockStride{}, kBasicBlockStride{}));

    using blk_Shape1 = Int<size<1>(TileShape_MNK{})/N>;
    using blk_Shape2 = Int<size<2>(TileShape_MNK{})/size<2>(typename TiledMma::Shape_MNK{})>;
    using blk_Stride1 =  Int<MMA_N / Blk_MN{} * Blk_Elems{} * MMA_NSF>;
    // Pad k-block stride so each descriptor's 8-row read window stays within its own k-block.
    using blk_Stride2_natural = Int<blk_Stride1{} / blk_Shape1{}>;
    using blk_Stride2 = Int<cute::max(int(blk_Stride2_natural{}), SfDescMinRows * MMA_NBlk)>;

    constexpr auto blk_Stride1_or_0 = [=] {
      if constexpr (blk_Shape1{} == 1) return _0{};
      else return blk_Stride1{};
    };

    using sSFB_shape     = decltype( make_shape( mma_SFB_shape{}, _1{},  make_shape( blk_Shape1{}, blk_Shape2{} )));
    using sSFB_stride    = decltype(make_stride(mma_SFB_stride{}, _0{}, make_stride( blk_Stride1_or_0(),blk_Stride2{})));
    using SmemLayoutAtomSFB = decltype(make_layout(sSFB_shape{}, sSFB_stride{}));
    return SmemLayoutAtomSFB{};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::detail
