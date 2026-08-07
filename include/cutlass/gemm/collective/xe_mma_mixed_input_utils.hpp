/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cute/numeric/integral_constant.hpp"

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class T, class = void>
struct general_same_bits {
  using type = T;
};

template <class T>
struct general_same_bits<T, std::enable_if_t<sizeof_bits_v<T> == 8>> {
  using type = int8_t;
};

template <class T>
struct general_same_bits<T, std::enable_if_t<sizeof_bits_v<T> == 16>> {
  using type = int16_t;
};

template <class T>
struct general_same_bits<T, std::enable_if_t<sizeof_bits_v<T> == 32>> {
  using type = int32_t;
};

template <class datatype, size_t N, class Stride = cute::Stride<_1, int64_t, int64_t>, class = void>
struct scale_zero_copy_traits {
  static_assert(cute::dependent_false<cute::tuple<datatype, Int<N>, Stride>>, "scale_zero_copy_traits not defined");
};

// 4 bits
template<class datatype, size_t N, class stride>
struct scale_zero_copy_traits<datatype, N, stride,
          std::enable_if_t<sizeof_bits_v<datatype> == 4 && decltype(get<0>(stride{}))::value == 8>> {
  using type = XE_2D_Packed_U4x1x128_LD_N;  // 8 elements along K packed into one int32 and then N-major
};

// 8 bits
template<class datatype, class stride>
struct scale_zero_copy_traits<datatype, 16, stride,
          std::enable_if_t<sizeof_bits_v<datatype> == 8>> {
  using type = XE_2D_U8x1x16_LD_N;
};
template<class datatype, size_t N, class stride>
struct scale_zero_copy_traits<datatype, N, stride,
          std::enable_if_t<sizeof_bits_v<datatype> == 8 && N >= 32>> {
  using type = XE_2D_U8x1x32_LD_N;
};

// 16 bits
template<class datatype, class stride>
struct scale_zero_copy_traits<datatype, 16, stride,
          std::enable_if_t<sizeof_bits_v<datatype> == 16>> {
  using type = XE_2D_U16x1x16_LD_N;
};
template<class datatype, size_t N, class stride>
struct scale_zero_copy_traits<datatype, N, stride,
          std::enable_if_t<sizeof_bits_v<datatype> == 16 && N >= 32>> {
  using type = XE_2D_U16x1x32_LD_N;
};

// 32 bits
template<class datatype, size_t N, class stride>
struct scale_zero_copy_traits<datatype, N, stride,
          std::enable_if_t<sizeof_bits_v<datatype> == 32>> {
  using type = XE_2D_U32x1x16_LD_N;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective
