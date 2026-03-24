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
/// XE4 Type I core-matrix SLM layout 
///
/// Core-matrix layout (byte space):
///
///   Shape:   ((2, 4, 4), 32)
///              |  |  |    +-- 32 bytes per row (k-dimension)
///              |  |  +------- 4 MMA banks
///              |  +---------- 4 ESub banks per MMA bank
///              +------------- 2 rows per core tile
///
///   Stride:  ((32, 256, 64), 1)
///              |    |    |    +-- byte stride along k = 1
///              |    |    +------- MMA bank stride    = 64
///              |    +------------ ESub bank stride   = 256
///              +----------------- row-in-tile stride = 32
///
///   Total: 2*4*4*32 = 1024 bytes
///
///   Element-space variants (byte strides preserved):
///     bf16  -> ((2,4,4), 16) : ((32, 256, 64), 2)
///     uint8 -> ((2,4,4), 32) : ((32, 256, 64), 1)
///     float -> ((2,4,4),  8) : ((32, 256, 64), 4)
///
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <cute/tensor.hpp>
#include <cute/swizzle.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>

namespace cute {
namespace xe4 {
namespace slm {
namespace type1 {
namespace kmajor {

using namespace cute;
// -------------------------------------------------------------------------
//  Hardware constants
// -------------------------------------------------------------------------

inline constexpr unsigned int kCoreMatrixBytes  = 1024;
inline constexpr unsigned int kRowsPerCmTile    = 2;
inline constexpr unsigned int kEsubBanks        = 8; // per MMA Bank //
inline constexpr unsigned int kHalfEsubBanks    = kEsubBanks/2;
inline constexpr unsigned int kMmaBanks         = 4;
inline constexpr unsigned int kBytesPerCmRow    = 32;
inline constexpr unsigned int kCoreMatrixM      = 32;
inline constexpr unsigned int kEsubBankBytes    = 64;
inline constexpr unsigned int kSubGroupSize     = 32;
inline constexpr unsigned int kMaxThreadsToAvoidBankConflicts = 
    kRowsPerCmTile*kMmaBanks;


template <typename T>
CUTE_HOST_DEVICE constexpr auto make_linear_layout_of_single_core_matrix()
{
    constexpr int elem_sz = int(sizeof(T));
    constexpr int cm_k = kBytesPerCmRow / elem_sz; 
    using CmPartitionLayout = Layout< Shape<Int<kCoreMatrixM>, Int<cm_k>>,
                                      Stride<Int<cm_k>, _1> >;
    return CmPartitionLayout{};
}

template <class T, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto
to_bit_layout(Layout<Shape, Stride> const& elem_layout)
{
    constexpr int N = sizeof_bits<T>::value;
    return make_layout(elem_layout.shape(),
                       transform_leaf(elem_layout.stride(), 
                                      [](auto s) { return s * Int<N>{}; }));
}

CUTE_HOST_DEVICE constexpr auto make_linear_layout_of_single_core_matrix_bit()
{
    using CmPartitionLayout = Layout< Shape<Int<kCoreMatrixM>, Int<kBytesPerCmRow*8>>,
                                      Stride<Int<kBytesPerCmRow*8>, _1> >;
    return CmPartitionLayout{};
}

template<typename T, int K>
CUTE_HOST_DEVICE constexpr auto 
  make_linear_layout_of_single_core_matrix_row_partition_bit()
{
    constexpr int row_bits = (sizeof_bits<T>::value)*K;
    using CmPartitionLayout = Layout< Shape<Int<kCoreMatrixM>, Int<row_bits>>,
                                      Stride<Int<row_bits>, _1> >;
    return CmPartitionLayout{};
}



// -------------------------------------------------------------------------
//  (5).1 Single core matrix layout 4x8 (QSubBank) layout  (in element space)
// -------------------------------------------------------------------------
template <typename T>
CUTE_HOST_DEVICE constexpr auto make_single_core_matrix_slm_layout()
{
    static_assert(kBytesPerCmRow >= int(sizeof(T)),
        "Element splitting not yet supported across core matrices");
    constexpr int elem_sz = int(sizeof(T));
    constexpr int cm_k = kBytesPerCmRow / elem_sz; 

    // Destination: XE4 tiled within the (32, CM_K) atom, element space
    //   row_in_tile stride = 32 / elem_sz
    //   esub_bank   stride = 256 / elem_sz
    //   mma_bank    stride = 64 / elem_sz
    //   k_elem      stride = 1

    using SingleCmLayout = Layout<
        Shape < Shape<Int<kRowsPerCmTile>, Int<kHalfEsubBanks>, Int<kMmaBanks>>, 
                Int<cm_k>>,
        Stride<Stride<Int<32/elem_sz>, Int<256/elem_sz>, Int<64/elem_sz>>, _1>
    >;

    return SingleCmLayout{};

}

CUTE_HOST_DEVICE constexpr auto make_single_core_matrix_slm_layout_bit()
{
    using SingleCmLayout = Layout<
        Shape < Shape<Int<kRowsPerCmTile>, Int<kHalfEsubBanks>, Int<kMmaBanks>>, 
                Int<kBytesPerCmRow*8>>,
        Stride<Stride<Int<32*8>, Int<256*8>, Int<64*8>>, _1>
    >;

    return SingleCmLayout{};
}

// -------------------------------------------------------------------------
//  (5).1  Full MxK tiled SLM layout (byte offsets)
//
//  Tiles the full matrix into core matrices:
//    Along M: num_cm_rows = M / 32
//    Along K: num_cm_cols = K / cm_k
//
//  Each core matrix is 1024 bytes.
//  Core matrix (cr, cc) starts at byte: (cr * num_cm_cols + cc) * 1024
//
//  Shape:   ((2, 4, 4, num_cm_rows), (cm_k, num_cm_cols))
//  Stride:  ((32, 256, 64, num_cm_cols*1024), (sizeof(T), 1024))
//
//  layout(m, k) returns a BYTE OFFSET.
// -------------------------------------------------------------------------

template <typename T, int M, int K>
CUTE_HOST_DEVICE constexpr auto make_slm_layout()
{
    static_assert(M % kCoreMatrixM == 0, "M must be a multiple of 32");
    static_assert(kBytesPerCmRow >= int(sizeof(T)), 
          "Element splitting not yet supported across core matrices");
    constexpr int cm_k = kBytesPerCmRow / int(sizeof(T));
    static_assert(K % cm_k == 0, "K must be a multiple of cm_k");

    constexpr int num_cm_rows = M / kCoreMatrixM;
    constexpr int num_cm_cols = K / cm_k;

    // M-dimension strides (byte0)
    constexpr int stride_row    = 32;                          // row_in_tile
    constexpr int stride_esub   = 256;                         // ESub bank
    constexpr int stride_mma    = 64;                          // MMA bank
    constexpr int stride_cm_row = num_cm_cols * kCoreMatrixBytes; // next CM row

    // K-dimension strides (bytes)
    constexpr int stride_k      = int(sizeof(T));              // element within row
    constexpr int stride_cm_col = kCoreMatrixBytes;            // next CM column

    auto base = make_layout(
        make_shape(
            make_shape(Int<kRowsPerCmTile>{},
                       Int<kHalfEsubBanks>{},
                       Int<kMmaBanks>{},
                       Int<num_cm_rows>{}),
            make_shape(Int<cm_k>{},
                       Int<num_cm_cols>{})
        ),
        make_stride(
            make_stride(Int<stride_row>{},
                        Int<stride_esub>{},
                        Int<stride_mma>{},
                        Int<stride_cm_row>{}),
            make_stride(Int<stride_k>{},
                        Int<stride_cm_col>{})
        )
    );
    return base;
}


// -------------------------------------------------------------------------
//  (5).1  Full MxK tiled SLM layout (elem offsets)
//
//  Tiles the full matrix into core matrices:
//
//--------------------------------------------------------------------------
template <typename T, int M, int K>
CUTE_HOST_DEVICE constexpr auto make_slm_layout_elem()
{
    static_assert(M % kCoreMatrixM == 0, "M must be a multiple of 32");
    static_assert(kBytesPerCmRow >= int(sizeof(T)), 
          "Element splitting not yet supported across core matrices");
    constexpr int cm_k = kBytesPerCmRow / int(sizeof(T));
    static_assert(K % cm_k == 0, "K must be a multiple of cm_k");

    constexpr int num_cm_rows = M / kCoreMatrixM;
    constexpr int num_cm_cols = K / cm_k;

    // M-dimension strides (elem)
    constexpr int elem_sz       = int(sizeof(T));
    constexpr int stride_row    = 32/elem_sz;                  // row_in_tile
    constexpr int stride_esub   = 256/elem_sz;                 // ESub bank
    constexpr int stride_mma    = 64/elem_sz;                 // MMA bank
    constexpr int stride_cm_row = 
        (num_cm_cols * kCoreMatrixBytes)/elem_sz; // next CM row

    // K-dimension strides (elem)
    constexpr int stride_k      = int(sizeof(T))/elem_sz;  // element within row
    constexpr int stride_cm_col = kCoreMatrixBytes/elem_sz; // next CM column

    auto base = make_layout(
        make_shape(
            make_shape(Int<kRowsPerCmTile>{},
                       Int<kHalfEsubBanks>{},
                       Int<kMmaBanks>{},
                       Int<num_cm_rows>{}),
            make_shape(Int<cm_k>{},
                       Int<num_cm_cols>{})
        ),
        make_stride(
            make_stride(Int<stride_row>{},
                        Int<stride_esub>{},
                        Int<stride_mma>{},
                        Int<stride_cm_row>{}),
            make_stride(Int<stride_k>{},
                        Int<stride_cm_col>{})
        )
    );
    return base;
}

template <typename T, int M, int K>
CUTE_HOST_DEVICE constexpr auto make_full_slm_layout_elem() {
  return make_slm_layout_elem<T, M, K>();
}

}  // namespace kmajor 
}  // namespace type1
}  // namespace slm
}  // namespace xe4
}  // namespace cute
