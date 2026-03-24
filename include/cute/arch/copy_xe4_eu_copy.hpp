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


///////////////////////////////////////////////////////////////////////////////
///
/// XE4 Type I core-matrix swizzled SLM copy operations (arch layer).
/// Contains the raw copy logic for G2S (swizzle) and S2G (unswizzle).
///
/// These operations work on CuTe tensors with shape:
///   ((kRowsPerCmTile, CM_K), RestM, RestK)
///
/// For even K-partitions: straight element copy
/// For odd  K-partitions: swap even/odd rows (m XOR 1)
///
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/arch/xe4_async_gmma_slm_layout.hpp>

namespace cute {

template <typename T, int M, int K, int SrcStride>
struct XE4_EU_COPY_G2S;

template <typename T, int M, int K, int DstStride>
struct XE4_EU_COPY_S2G;

template <typename T, int M, int K>
struct XE4_EU_COPY_S2R;

template <typename T, int M, int K>
struct XE4_EU_COPY_G2R;

template <typename T, int M, int K>
struct XE4_EU_COPY_R2G;

namespace detail {

template <int CM_K, class DstTensor, class CoreTileEven, class CoreTileOdd>
CUTE_HOST_DEVICE
void
store_k_tile_even_odd(DstTensor& dst_tensor,
                      int dst_row0, int dst_row1,
                      int elem_k_idx,
                      CoreTileEven const& core_tile_even,
                      CoreTileOdd const& core_tile_odd)
{
    CUTE_UNROLL
    for (int t = 0; t < CM_K; t++) {
      dst_tensor(dst_row0, elem_k_idx+t) = core_tile_even(t);
      dst_tensor(dst_row1, elem_k_idx+t) = core_tile_odd(t);
    }
}

template <int CM_K, class DstTensor, class CoreTileEven, class CoreTileOdd>
CUTE_HOST_DEVICE
void
store_k_tile_odd_even(DstTensor& dst_tensor,
                      int dst_row0, int dst_row1,
                      int elem_k_idx,
                      CoreTileEven const& core_tile_even,
                      CoreTileOdd const& core_tile_odd)
{
    CUTE_UNROLL
    for (int t = 0; t < CM_K; t++) {
      dst_tensor(dst_row0, elem_k_idx+t) = core_tile_odd(t);
      dst_tensor(dst_row1, elem_k_idx+t) = core_tile_even(t);
    }
}

} // namespace detail

template <typename T, int M, int K, int SrcStride,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<Copy_Traits<
      XE4_EU_COPY_G2S<T, M, K, SrcStride>>, T> const& atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout> & dst)
{
    CUTE_STATIC_ASSERT_V(rank(SrcLayout{}) == rank(DstLayout{}),
        "Copy Invariant Failed");
    CUTE_STATIC_ASSERT_V(rank(SrcLayout{}) == _3{}, "Should have 3 modes");
    CUTE_STATIC_ASSERT_V(rank<0>(SrcLayout{}) == rank<0>(DstLayout{}),
          "M0 Copy Invariant Failed");
    CUTE_STATIC_ASSERT_V(rank<0>(SrcLayout{}) == _2{}, "Copy Invariant Failed");


	 // shapes: ( (x,y), kNumRowsPerCmTile), rest_x, rest_y ) //

    using Traits = Copy_Traits<XE4_EU_COPY_G2S<T, M, K, SrcStride>>;
    using FullSrcLayout = typename Traits::FullSrcLayout;
    using FullDstLayout = typename Traits::FullDstLayout;

    auto src_layout = FullSrcLayout{};
    auto dst_layout = FullDstLayout{};
    auto inv_src = left_inverse(src_layout);
    auto inv_dst = left_inverse(dst_layout);

    auto dst_tensor = make_tensor((T *)atom.dst_base_, dst_layout);
    auto partition_layout = select<1,2>(layout(src));

    CUTE_UNROLL
    for (int m = 0; m < size<0>(partition_layout); m++) {
      // source row tile offsets //
      auto core_tile0 = make_coord(make_coord(_0{}, _0{}), m, _0{});
      auto core_tile1 = make_coord(make_coord(_1{}, _0{}), m, _0{});
      auto core_tile0_off = (&src(core_tile0) - atom.src_base_);
      auto core_tile1_off = (&src(core_tile1) - atom.src_base_);
      int src_r0 = get<0>(idx2crd(inv_src(core_tile0_off), //index
              shape(src_layout)));
      int src_r1 = get<0>(idx2crd(inv_src(core_tile1_off), //index
              shape(src_layout)));

#if HOST_DEBUG
      auto dest_coord_r0 = idx2crd(dst_layout(src_r0, 0), //offset//
            shape(dst_layout), stride(dst_layout));
      auto dest_coord_r1 = idx2crd((dst_layout(src_r1, 0)), //offset//
            shape(dst_layout), stride(dst_layout));

      std::cout << "===================================" << std::endl;
      std::cout << "src_offsets: " << core_tile0_off << " " << core_tile1_off
                << std::endl;
      std::cout << "src_row(" << src_r0 << " " << src_r1;
      std::cout << ") dst_coords: " ; print(dest_coord_r0);
      std::cout <<  " " ; print(dest_coord_r1);
      std::cout << std::endl;
      std::cout << " dst_offsest: " << (int) dst_layout(src_r0, 0) << " , ";
      std::cout << (int) dst_layout(src_r1, 0) << " " << std::endl;
      std::cout << "CM_K=" << Traits::CM_K << std::endl;;
      std::cout << "===================================" << std::endl;
#endif

      CUTE_UNROLL
      for (int k = 0; k < size<1>(partition_layout); k+=2) {
        auto even_coord = make_coord(make_coord(make_coord(_0{},_),_), m, k);
        auto odd_coord = make_coord(make_coord(make_coord(_1{},_),_), m, k);

        auto core_tile_even = src(even_coord);
        auto core_tile_odd = src(odd_coord);
        int elem_k_idx = k*Traits::CM_K;

        detail::store_k_tile_even_odd<Traits::CM_K>(dst_tensor,
                                                    src_r0, src_r1,
                                                    elem_k_idx,
                                                    core_tile_even,
                                                    core_tile_odd);
      }

      CUTE_UNROLL
      for (int k = 1; k < size<1>(partition_layout); k+=2) {
        auto even_coord = make_coord(make_coord(make_coord(_0{},_),_), m, k);
        auto odd_coord = make_coord(make_coord(make_coord(_1{},_),_), m, k);

        auto core_tile_even = src(even_coord);
        auto core_tile_odd = src(odd_coord);
        int elem_k_idx = k*Traits::CM_K;
        detail::store_k_tile_odd_even<Traits::CM_K>(dst_tensor,
                                                    src_r0, src_r1,
                                                    elem_k_idx,
                                                    core_tile_even,
                                                    core_tile_odd);
      }
    }
}


//TODO(vamsikku): combine both these copy methods into one by making FullSrcLayout
// and FullDstLayout as template parameters //
template <typename T, int M, int K, int DstStride,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<Copy_Traits<
      XE4_EU_COPY_S2G<T, M, K, DstStride>>, T> const& atom,
     Tensor<SrcEngine, SrcLayout> const& src,Tensor<DstEngine, DstLayout> & dst)
{
    CUTE_STATIC_ASSERT_V(rank(SrcLayout{}) == rank(DstLayout{}),
        "Copy Invariant Failed");
    CUTE_STATIC_ASSERT_V(rank(SrcLayout{}) == _3{}, "Should have 3 modes");
    CUTE_STATIC_ASSERT_V(rank<0>(SrcLayout{}) == rank<0>(DstLayout{}),
          "M0 Copy Invariant Failed");
    CUTE_STATIC_ASSERT_V(rank<0>(SrcLayout{}) == _2{}, "Copy Invariant Failed");

    auto partition_layout = select<1,2>(layout(src));

    using Traits = Copy_Traits<XE4_EU_COPY_S2G<T, M, K, DstStride>>;
    using FullSrcLayout = typename Traits::FullSrcLayout;
    using FullDstLayout = typename Traits::FullDstLayout;


    auto src_layout = FullSrcLayout{};
    auto dst_layout = FullDstLayout{};
    auto inv_src = left_inverse(src_layout);
    auto inv_dst = left_inverse(dst_layout);

    auto dst_tensor = make_tensor((T *)atom.dst_base_, dst_layout);

    CUTE_UNROLL
    for (int m = 0; m < size<0>(partition_layout); m++) {
      // source row tile offsets //
      auto core_tile0 = make_coord(make_coord(_0{}, _0{}), m, _0{});
      auto core_tile1 = make_coord(make_coord(_1{}, _0{}), m, _0{});
      auto core_tile0_off = (&src(core_tile0) - atom.src_base_);
      auto core_tile1_off = (&src(core_tile1) - atom.src_base_);
      int src_r0 = get<0>(idx2crd(inv_src(core_tile0_off), //index
              shape(dst_layout)));
      int src_r1 = get<0>(idx2crd(inv_src(core_tile1_off), //index
              shape(dst_layout)));

#if HOST_ONLY_DEBUG
      auto dest_coord_r0 = idx2crd(dst_layout(src_r0, 0), //offset//
            shape(dst_layout), stride(dst_layout));
      auto dest_coord_r1 = idx2crd((dst_layout(src_r1, 0)), //offset//
            shape(dst_layout), stride(dst_layout));

      std::cout << "===================================" << std::endl;
      std::cout << "src_offsets: " << core_tile0_off << " " << core_tile1_off
                << std::endl;
      std::cout << "src_row(" << src_r0 << " " << src_r1;
      std::cout << ") dst_coords: " ; print(dest_coord_r0);
      std::cout <<  " " ; print(dest_coord_r1);
      std::cout << std::endl;
      std::cout << " dst_offsest: " << (int) dst_layout(src_r0, 0) << " , ";
      std::cout << (int) dst_layout(src_r1, 0) << " " << std::endl;
      std::cout << "CM_K=" << Traits::CM_K << std::endl;;
      std::cout << "===================================" << std::endl;
#endif

      CUTE_UNROLL
      for (int k = 0; k < size<1>(partition_layout); k+=2) {
        auto even_coord = make_coord(make_coord(make_coord(_0{},_),_), m, k);
        auto odd_coord = make_coord(make_coord(make_coord(_1{},_),_), m, k);

        auto core_tile_even = src(even_coord);
        auto core_tile_odd = src(odd_coord);
        int elem_k_idx = k*Traits::CM_K;

        detail::store_k_tile_even_odd<Traits::CM_K>(dst_tensor,
                                                    src_r0, src_r1,
                                                    elem_k_idx,
                                                    core_tile_even,
                                                    core_tile_odd);
      }

      CUTE_UNROLL
      for (int k = 1; k < size<1>(partition_layout); k+=2) {
        auto even_coord = make_coord(make_coord(make_coord(_0{},_),_), m, k);
        auto odd_coord = make_coord(make_coord(make_coord(_1{},_),_), m, k);

        auto core_tile_even = src(even_coord);
        auto core_tile_odd = src(odd_coord);
        int elem_k_idx = k*Traits::CM_K;
        detail::store_k_tile_odd_even<Traits::CM_K>(dst_tensor,
                                                    src_r0, src_r1,
                                                    elem_k_idx,
                                                    core_tile_even,
                                                    core_tile_odd);
      }
    }
}

template <typename T, int M, int K,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<Copy_Traits<XE4_EU_COPY_S2R<T, M, K> >, T> const& atom,
     Tensor<SrcEngine, SrcLayout> const& src,Tensor<DstEngine, DstLayout> & dst)
{
    namespace hw = xe4::slm::type1::kmajor;
    CUTE_STATIC_ASSERT_V(rank<0>(SrcLayout{}) == _2{}, "Copy Invariant Failed");
    CUTE_STATIC_ASSERT_V(
        size<0>(dst) == (size<1>(src)*(Int<hw::kRowsPerCmTile>{})),
          "Destination size mismatch");
    CUTE_STATIC_ASSERT_V(size<1>(dst) == Int<K>{}, "Destination size mismatch");

    auto partition_layout = select<1,2>(layout(src));
    using Traits = Copy_Traits<XE4_EU_COPY_S2R<T, M, K> >;

    CUTE_UNROLL
    for (int m = 0; m < size<0>(partition_layout); m++) {
      const int dst_odd_row = 2*m + 1;
      const int dst_even_row = 2*m;

      CUTE_UNROLL
      for (int k = 0; k < size<1>(partition_layout); k+=2) {
        auto even_coord = make_coord(make_coord(make_coord(_0{},_),_), m, k);
        auto odd_coord = make_coord(make_coord(make_coord(_1{},_),_), m, k);

        auto core_tile_even = src(even_coord);
        auto core_tile_odd = src(odd_coord);
        int elem_k_idx = k*Traits::CM_K;

        detail::store_k_tile_even_odd<Traits::CM_K>(dst,
                                                    dst_even_row, dst_odd_row,
                                                    elem_k_idx,
                                                    core_tile_even,
                                                    core_tile_odd);
      }

      CUTE_UNROLL
      for (int k = 1; k < size<1>(partition_layout); k+=2) {
        auto even_coord = make_coord(make_coord(make_coord(_0{},_),_), m, k);
        auto odd_coord = make_coord(make_coord(make_coord(_1{},_),_), m, k);

        auto core_tile_even = src(even_coord);
        auto core_tile_odd = src(odd_coord);
        int elem_k_idx = k*Traits::CM_K;
        detail::store_k_tile_odd_even<Traits::CM_K>(dst,
                                                    dst_even_row, dst_odd_row,
                                                    elem_k_idx,
                                                    core_tile_even,
                                                    core_tile_odd);
      }
    }
}

template<typename T>
struct show_type_t;

template <typename T, int M, int K,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<Copy_Traits<XE4_EU_COPY_G2R<T, M, K>>, T> const& atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>& dst)
{
    CUTE_STATIC_ASSERT_V(size(SrcLayout{}) == size(DstLayout{}),
        "Copy Size Invariant Failed");
    CUTE_STATIC_ASSERT_V(rank(src) == _3{}, "Copy Rank Invariant Failed");

    auto partition_layout = select<1,2>(layout(src));

    CUTE_UNROLL
    for (int m = 0; m < size<0>(partition_layout); m++) {
      const int dst_odd_row  = 2 * m + 1; const int dst_even_row = 2 * m;
      CUTE_UNROLL
      for (int k=0; k < size<1>(partition_layout); k++) {
        auto even_coord = make_coord(make_coord(make_coord(_0{},_),_), m, k);
        auto odd_coord = make_coord(make_coord(make_coord(_1{},_),_), m, k);
        copy(src(even_coord), dst(dst_even_row, _));
        copy(src(odd_coord), dst(dst_odd_row, _));
      }
    }
}

template <typename T, int M, int K,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
copy(Copy_Atom<Copy_Traits<XE4_EU_COPY_R2G<T, M, K>>, T> const& atom,
     Tensor<SrcEngine, SrcLayout> const& src,
     Tensor<DstEngine, DstLayout>& dst)
{
    CUTE_STATIC_ASSERT_V(size(SrcLayout{}) == size(DstLayout{}),
        "Copy Size Invariant Failed");
    CUTE_STATIC_ASSERT_V(rank(dst) == _3{}, "Copy Rank Invariant Failed");

    auto partition_layout = select<1,2>(layout(dst));

    CUTE_UNROLL
    for (int m = 0; m < size<0>(partition_layout); m++) {
        const int src_odd_row  = 2 * m + 1; const int src_even_row = 2 * m;
        for (int k=0; k < size<1>(partition_layout); k++) {
          // note: we cant to the 1d rank copy because of the hier. coords
          auto even_coord = make_coord(make_coord(make_coord(_0{},_),_), m, k);
          auto odd_coord = make_coord(make_coord(make_coord(_1{},_),_), m, k);
          copy(src(src_even_row, _), dst(even_coord));
          copy(src(src_odd_row, _), dst(odd_coord));
        }
    }
}

} // namespace cute