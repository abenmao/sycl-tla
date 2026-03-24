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
/// CuTe atom layer for XE4 EU copy.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/arch/xe4_async_gmma_slm_layout.hpp>
#include <cute/arch/copy_xe4_eu_copy.hpp>

namespace cute {

// ============================================================================
// CopyOp tags
// ============================================================================
template <typename T, int M, int K, int SrcStride>
struct XE4_EU_COPY_G2S{};

template <typename T, int M, int K, int DstStride>
struct XE4_EU_COPY_S2G{};

template <typename T, int M, int K>
struct XE4_EU_COPY_S2R{};

template <typename T, int M, int K>
struct XE4_EU_COPY_G2R{};

template <typename T, int M, int K>
struct XE4_EU_COPY_R2G{};


namespace hw = xe4::slm::type1::kmajor;

// ============================================================================
//  Shared constants and layout types
// ============================================================================
namespace detail {

template <typename T, int M, int K>
struct XE4_SLM_COPY_Constants
{

    static constexpr int ELEM_SZ        = int(sizeof(T));
    static constexpr int CM_K           = hw::kBytesPerCmRow / ELEM_SZ;
    static constexpr int SG_SIZE        = hw::kSubGroupSize;
    static constexpr int CM_M           = hw::kCoreMatrixM;
    static constexpr int kRowsPerCmTile = hw::kRowsPerCmTile;

    static_assert(M % CM_M == 0, "M must be a multiple of core matrix M (32)");
    static_assert(K % CM_K == 0, "K must be a multiple of CM_K");

    // Row-major layout for the linear side (GMEM or registers)
    // (32, CM_K) //
    using LinearLayout =
      //decltype(hw:: template make_linear_layout_of_single_core_matrix<T>());
      decltype(hw::make_linear_layout_of_single_core_matrix_bit());

    // SLM tiled layout from xe4_async_gmma_slm_layout.hpp
    using SlmLayout =
        //decltype(hw:: template make_single_core_matrix_slm_layout_bit<T>());
        decltype(hw::make_single_core_matrix_slm_layout_bit());
};

} // namespace detail

// ============================================================================
//  Copy_Traits: Source will be swizzled
// ============================================================================
template <typename T, int M, int K, int SrcStride>
struct Copy_Traits<XE4_EU_COPY_G2S<T, M, K, SrcStride>>
{
    using C = detail::XE4_SLM_COPY_Constants<T, M, K>;
    using ThrID     = Layout<Int<C::SG_SIZE>>;
    using SrcLayout = typename C::LinearLayout;
    using DstLayout = typename C::SlmLayout;
    using RefLayout = SrcLayout;
    using FullSrcLayout = Layout<Shape<Int<M>, Int<K>>,
            Stride<Int<SrcStride>, _1>>;
    using FullDstLayout =
        decltype(hw::template make_slm_layout_elem<T, M, K>());

    static constexpr int CM_K = C::CM_K;

    template <class SrcPtr, class DstPtr>
    CUTE_HOST_DEVICE
    Copy_Traits with(SrcPtr src_base, DstPtr dst_base) const {
        Copy_Traits t = *this;
        t.src_base_ = raw_pointer_cast(src_base);
        t.dst_base_ = raw_pointer_cast(dst_base);
        return t;
    }

    const T* src_base_ = nullptr;
    T* dst_base_ = nullptr;
};


// ============================================================================
//  Copy_Traits: Dest will be swizzled
// ============================================================================
template <typename T, int M, int K, int DstStride>
struct Copy_Traits<XE4_EU_COPY_S2G<T, M, K, DstStride>>
{
    using C = detail::XE4_SLM_COPY_Constants<T, M, K>;
    using ThrID     = Layout<Int<C::SG_SIZE>>;
    using DstLayout = typename C::LinearLayout;
    using SrcLayout = typename C::SlmLayout;
    using RefLayout = SrcLayout;
    using FullDstLayout = Layout<Shape<Int<M>, Int<K>>,
            Stride<Int<DstStride>, _1>>;
    using FullSrcLayout =
        decltype(hw::template make_slm_layout_elem<T, M, K>());

    static constexpr int CM_K = C::CM_K;

    template <class SrcPtr, class DstPtr>
    CUTE_HOST_DEVICE
    Copy_Traits with(SrcPtr src_base, DstPtr dst_base) const {
        Copy_Traits t = *this;
        t.src_base_ = raw_pointer_cast(src_base);
        t.dst_base_ = raw_pointer_cast(dst_base);
        return t;
    }

    const T* src_base_ = nullptr;
    T* dst_base_ = nullptr;
};



template <typename T, int M, int K>
struct Copy_Traits<XE4_EU_COPY_S2R<T, M, K>>
{
    using C = detail::XE4_SLM_COPY_Constants<T, M, K>;
    using ThrID     = Layout<Int<C::SG_SIZE>>;
    using DstLayout = typename C::LinearLayout;
    using SrcLayout = typename C::SlmLayout;
    using RefLayout = SrcLayout;

    static constexpr int CM_K = C::CM_K;

    template <class SrcPtr, class DstPtr>
    CUTE_HOST_DEVICE
    Copy_Traits with(SrcPtr src_base, DstPtr dst_base) const {
        Copy_Traits t = *this;
        t.src_base_ = raw_pointer_cast(src_base);
        t.dst_base_ = raw_pointer_cast(dst_base);
        return t;
    }

    const T* src_base_ = nullptr;
    T* dst_base_ = nullptr;
};

template <typename T, int M, int K>
struct Copy_Traits<XE4_EU_COPY_G2R<T, M, K>>
{
    using C = detail::XE4_SLM_COPY_Constants<T, M, K>;
    using ThrID     = Layout<Int<C::SG_SIZE>>;
    using DstLayout = decltype(hw:: template 
          make_linear_layout_of_single_core_matrix_row_partition_bit<T, K>());
    using SrcLayout = DstLayout;
    using RefLayout = SrcLayout;
};

template <typename T, int M, int K>
struct Copy_Traits<XE4_EU_COPY_R2G<T, M, K>>
{
    using C = detail::XE4_SLM_COPY_Constants<T, M, K>;
    using ThrID     = Layout<Int<C::SG_SIZE>>;
    using DstLayout = decltype(hw:: template 
          make_linear_layout_of_single_core_matrix_row_partition_bit<T, K>());
    using SrcLayout = DstLayout;
    using RefLayout = SrcLayout;
};


// ============================================================================
//  Factories -- enforce correct thr_layout and val_layout
//
//  thr_layout = (32, 1)  -- 32 threads along M
//  val_layout = (2, CM_K) -- 2 rows x CM_K elements (even/odd pair)
// ============================================================================

// GMEM -> SLM (swizzled store)
template <typename T, int M, int K,
          typename SrcEngine, typename SrcLayout, 
          typename DstEngine, typename DstLayout>
CUTE_HOST_DEVICE
auto make_xe4_g2s_tiled_copy(const Tensor<SrcEngine, SrcLayout>& src,
    Tensor<DstEngine, DstLayout>& dst)
{

    constexpr int CM_K           = hw::kBytesPerCmRow / int(sizeof(T));
    constexpr int SG_SIZE        = hw::kSubGroupSize;
    constexpr int kRowsPerCmTile = hw::kRowsPerCmTile;
    CUTE_STATIC_ASSERT_V(rank(SrcLayout{}) == _2{}, "Invalid source rank");
    CUTE_STATIC_ASSERT_V(get<1>(stride(SrcLayout{})) == _1{},
            "Invalid K-Major source");
    static_assert(M  >= 64, "TODO: M=32 needs special implementation");

    constexpr int SrcStride = get<0>(stride(SrcLayout{}));
        
    using CopyOp = XE4_EU_COPY_G2S<T, M, K, SrcStride>;
    using Traits = Copy_Traits<CopyOp>;
    using Atom   = Copy_Atom<Traits, T>;

    auto thr_layout = make_layout(make_shape(Int<SG_SIZE>{}, _1{}));
    auto val_layout = make_layout(make_shape(Int<kRowsPerCmTile>{}, Int<CM_K>{}));

    return make_tiled_copy(Atom{}.with(src.data(), dst.data()),
          thr_layout, val_layout);
}


template<typename T>
struct show_type_t;

// SLM -> GMEM (unswizzled load)
template <typename T, int M, int K,
          typename SrcEngine, typename SrcLayout, 
          typename DstEngine, typename DstLayout>
CUTE_HOST_DEVICE
auto make_xe4_s2g_tiled_copy(const Tensor<SrcEngine, SrcLayout>& src,
    Tensor<DstEngine, DstLayout>& dst)
{

    constexpr int CM_K           = hw::kBytesPerCmRow / int(sizeof(T));
    constexpr int SG_SIZE        = hw::kSubGroupSize;
    constexpr int kRowsPerCmTile = hw::kRowsPerCmTile;

    static_assert(M  >= 64, "TODO: M=32 needs special implementation");
    CUTE_STATIC_ASSERT_V(rank(DstLayout{}) == _2{}, "Invalid source rank");
    CUTE_STATIC_ASSERT_V(get<1>(stride(DstLayout{})) == _1{},
            "Invalid K-Major source");
    constexpr int DstStride = get<0>(stride(DstLayout{}));
        

    using CopyOp = XE4_EU_COPY_S2G<T, M, K, DstStride>;
    using Traits = Copy_Traits<CopyOp>;
    using Atom   = Copy_Atom<Traits, T>;

    auto thr_layout = make_layout(make_shape(Int<SG_SIZE>{}, _1{}));
    auto val_layout = make_layout(make_shape(Int<kRowsPerCmTile>{}, Int<CM_K>{}));

    return make_tiled_copy(Atom{}.with(src.data(), dst.data()),
          thr_layout, val_layout);
}

template<typename T>
struct list_type_t;

template <typename T, int M, int K>
CUTE_HOST_DEVICE
auto make_xe4_reg_layout_for_tiled_copy()
{
  constexpr int SG_SIZE = hw::kSubGroupSize;
  static_assert(M >= 2*SG_SIZE,
        "Number of rows should twice the subgroup size");
  static_assert(M%SG_SIZE == 0,
        "Number of rows should be a multiple of subgroup size");
  constexpr int m = M/SG_SIZE;

  return make_layout(make_shape(Int<m>{}, Int<K>{}),
        make_stride(Int<K>{}, _1{}));
}


// SLM -> REGS (unswizzled load)
template <typename T, int M, int K,
          typename SrcEngine, typename SrcLayout, 
          typename DstEngine, typename DstLayout>
CUTE_HOST_DEVICE
auto make_xe4_s2r_tiled_copy(const Tensor<SrcEngine, SrcLayout>& src,
    Tensor<DstEngine, DstLayout>& dst)
{

    constexpr int CM_K           = hw::kBytesPerCmRow / int(sizeof(T));
    constexpr int SG_SIZE        = hw::kSubGroupSize;
    constexpr int kRowsPerCmTile = hw::kRowsPerCmTile;

    static_assert(M  >= 64, "TODO: M=32 needs special implementation");

    using CopyOp = XE4_EU_COPY_S2R<T, M, K>;
    using Traits = Copy_Traits<CopyOp>;
    using Atom   = Copy_Atom<Traits, T>;

    auto thr_layout = make_layout(make_shape(Int<SG_SIZE>{}, _1{}));
    auto val_layout = make_layout(make_shape(Int<kRowsPerCmTile>{}, Int<CM_K>{}));

    return make_tiled_copy(Atom{}.with(src.data(), dst.data()),
          thr_layout, val_layout);
}


// GMEM -> REGS (unswizzled load)
template <typename T, int M, int K>
CUTE_HOST_DEVICE
auto make_xe4_g2r_tiled_copy()
{
    constexpr int SG_SIZE        = hw::kSubGroupSize;
    constexpr int kRowsPerCmTile = hw::kRowsPerCmTile;

    static_assert(M  >= 64, "TODO: M=32 needs special implementation");

    using CopyOp = XE4_EU_COPY_G2R<T, M, K>;
    using Traits = Copy_Traits<CopyOp>;
    using Atom   = Copy_Atom<Traits, T>;

    auto thr_layout = make_layout(make_shape(Int<SG_SIZE>{}, _1{}));
    auto val_layout = make_layout(make_shape(Int<kRowsPerCmTile>{}, Int<K>{}));

    return make_tiled_copy(Atom{}, thr_layout, val_layout);
}


template <typename T, int M, int K>
CUTE_HOST_DEVICE
auto make_xe4_r2g_tiled_copy()
{
    constexpr int SG_SIZE        = hw::kSubGroupSize;
    constexpr int kRowsPerCmTile = hw::kRowsPerCmTile;

    static_assert(M  >= 64, "TODO: M=32 needs special implementation");

    using CopyOp = XE4_EU_COPY_R2G<T, M, K>;
    using Traits = Copy_Traits<CopyOp>;
    using Atom   = Copy_Atom<Traits, T>;

    auto thr_layout = make_layout(make_shape(Int<SG_SIZE>{}, _1{}));
    auto val_layout = make_layout(make_shape(Int<kRowsPerCmTile>{}, Int<K>{}));

    return make_tiled_copy(Atom{}, thr_layout, val_layout);
}

} // namespace cute
