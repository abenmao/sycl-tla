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
#pragma once

#include <cute/arch/asm_helper.hpp>
#include <cute/arch/async_tensor_copy.hpp>

namespace cute {
namespace detail {

// Specializations emit async_linear_fred (FloatType) or async_linear_ired (IntType).
template <RedType RDType>
struct AsyncLinearReduce;

template <>
struct AsyncLinearReduce<RedType::FloatType> {
  template <CacheCtrl CacheType, RedOp Rop, BarrierType BarType, typename T>
  static inline void
  Reduce(void* gmem_ptr, void* slm_ptr, uint32_t size, uint64_t const* abar_ptr)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    static_assert(CacheType == L2wb_L3wb || CacheType == L2uc_L3wb);

    if constexpr (BarType == BarrierType::Abarrier) {
      asm volatile(
        ("async_linear_fred.global.shared_workgroup"
         + _red_algo<Rop> + _mdtype<T> + _cc<CacheType>
         + ".abarrier [%0], [%1], [%2], %3;\n")
        ::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(size));
    } else {
      asm volatile(
        ("async_linear_fred.global.shared_workgroup"
         + _red_algo<Rop> + _mdtype<T> + _cc<CacheType>
         + ".groupsync [%0], [%1], %2;\n")
        ::"r"(gmem_ptr), "r"(slm_ptr), "r"(size));
    }
#endif
  }
};

template <>
struct AsyncLinearReduce<RedType::IntType> {
  template <CacheCtrl CacheType, RedOp Rop, BarrierType BarType, typename T>
  static inline void
  Reduce(void* gmem_ptr, void* slm_ptr, uint32_t size, uint64_t const* abar_ptr)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    static_assert(CacheType == L2wb_L3wb || CacheType == L2uc_L3wb);

    if constexpr (BarType == BarrierType::Abarrier) {
      asm volatile(
        ("async_linear_ired.global.shared_workgroup"
         + _red_algo<Rop> + _mdtype<T> + _cc<CacheType>
         + ".abarrier [%0], [%1], [%2], %3;\n")
        ::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(size));
    } else {
      asm volatile(
        ("async_linear_ired.global.shared_workgroup"
         + _red_algo<Rop> + _mdtype<T> + _cc<CacheType>
         + ".groupsync [%0], [%1], %2;\n")
        ::"r"(gmem_ptr), "r"(slm_ptr), "r"(size));
    }
#endif
  }
};

} // namespace detail
} // namespace cute
