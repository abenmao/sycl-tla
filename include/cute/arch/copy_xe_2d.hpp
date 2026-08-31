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

#include <type_traits>

#include "cute/numeric/int.hpp"

#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
#define CUTE_ARCH_COPY_XE_ENABLED
#endif

namespace cute {

// Xe 2D copy atoms.
//       Bits: bits per element in underlying memory operation.
//     Height: number of elements in the gmem-strided matrix dimension
//      Width: number of elements in the gmem-contiguous matrix dimension
// BlockWidth: blocking factor (in registers) for the width dimension.
//
// For a row-major-in-memory matrix:
//    height = #rows, width = #columns.
//
// For a column-major-in-memory matrix:
//    height = #columns, width = #rows.

/* Base class for 2D copy ops, to support common queries */
template <int Bits, int Height, int Width, int Count = 1, bool Transpose = false>
struct XE_Copy_Op_2D_Base
{
  static_assert(Height <= 32, "Height exceeds hardware limits");
  static_assert(Bits * Width <= 8 * 64, "Total width exceeds hardware limits");
  static_assert(Bits * Count <= 64 && Count <= 4 && Count != 3, "Unsupported block count");
  static_assert(Bits == 8 || Bits == 16 || Bits == 32 || Bits == 64, "Unsupported data size");

  static constexpr int CopyBits = Bits;
  static constexpr int AtomWidth = Width;
  static constexpr int AtomHeight = Height;
  static constexpr int BlockCount = Count;
  static constexpr bool Transposing = Transpose;
};

template <int Bits, int Height, int Width, int BlockWidth = 0 /* unused */>
struct XE_PREFETCH_2D;


template <int Bits, int Height, int Width, int BlockWidth = Width>
struct XE_LOAD_2D : XE_Copy_Op_2D_Base<Bits, Height, Width, Width/BlockWidth>
{
  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int *payload, T *dst) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    using namespace intel;
    // TODO: to workaround the GRF aligned visa issue, may have better way in the future
    constexpr auto bits_per_grf = 64 * 8;
    constexpr auto grf_aligned_bits = cute::ceil_div(Bits * Width * Height, bits_per_grf) * bits_per_grf;
    auto &dv = *reinterpret_cast<storage_vector_t<T, grf_aligned_bits / sg_size> *>(dst);
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
    asm (
      "lsc_load_block2d.ugm.ca.ca.uc (M1, 1)  %0:d%2.%3x%4x%5nn flat[%1+(0,0)]"
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width/BlockWidth), "P"(BlockWidth), "P"(Height)
    );
#else
    asm (
      "lsc_load_block2d.ugm (M1, 1)  %0:d%2.%3x%4x%5nn flat[%1+(0,0)]"
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width/BlockWidth), "P"(BlockWidth), "P"(Height)
    );
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }

  using PREFETCH = XE_PREFETCH_2D<Bits, Height, Width>;
};

template <int Bits, int Height, int Width, int BlockWidth = Width>
struct XE_LOAD_2D_VNNI : XE_Copy_Op_2D_Base<Bits, Height, Width, Width/BlockWidth>
{
  static_assert(Bits == 8 || Bits == 16, "Unsupported data size");

  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int *payload, T *dst) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    using namespace intel;
    auto &dv = *reinterpret_cast<storage_vector_t<T, Width * Height * Bits / sg_size>*>(dst);
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
    asm (
      "lsc_load_block2d.ugm.ca.ca.uc (M1, 1)  %0:d%2.%3x%4x%5nt flat[%1+(0,0)]"
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width/BlockWidth), "P"(BlockWidth), "P"(Height)
    );
#else
    asm (
      "lsc_load_block2d.ugm (M1, 1)  %0:d%2.%3x%4x%5nt flat[%1+(0,0)]"
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width/BlockWidth), "P"(BlockWidth), "P"(Height)
    );
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }

  using PREFETCH = XE_PREFETCH_2D<Bits, Height, Width>;
};

template <int Bits, int Height, int Width>
struct XE_LOAD_2D_TRANSPOSE : XE_Copy_Op_2D_Base<Bits, Height, Width, 1, true>
{
  static_assert(Bits == 32 || Bits == 64, "Unsupported data size");
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
  static_assert((Bits == 32 && Width <= 16) || (Bits == 64 && Width <= 8),
                "Width exceeds hardware limits");
  static_assert(Bits != 64 || (Height == 8 && Width <= 8), "Unsupported D64 transpose block size");
#else
  static_assert(Width <= 8, "Width exceeds hardware limits");
  static_assert(Bits != 64 || (Height == 8 && Width <= 4), "Unsupported D64 transpose block size");
#endif

  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int *payload, T *dst) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    using namespace intel;
    auto &dv = *reinterpret_cast<storage_vector_t<T, Width * Height * Bits / sg_size>*>(dst);
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
    asm (
      "lsc_load_block2d.ugm.ca.ca.uc (M1, 1)  %0:d%2.%3x%4tn flat[%1+(0,0)]"
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height)
    );
#else
    asm (
      "lsc_load_block2d.ugm (M1, 1)  %0:d%2.%3x%4tn flat[%1+(0,0)]"
        : "=rw"(dv)
        : "rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height)
    );
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }

  using PREFETCH = XE_PREFETCH_2D<Bits, Height, Width>;
};

template <int Bits, int Height, int Width, int>
struct XE_PREFETCH_2D : XE_Copy_Op_2D_Base<Bits, Height, Width>
{
  CUTE_HOST_DEVICE static void copy(const int *payload) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
    asm (
      "lsc_load_block2d.ugm.ca.ca.uc (M1, 1)  %%null:d%1.%2x%3nn flat[%0+(0,0)]"
        :: "rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height)
    );
#else
    asm (
      "lsc_load_block2d.ugm.ca.ca (M1, 1)  %%null:d%1.%2x%3nn flat[%0+(0,0)]"
        :: "rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height)
    );
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }

  using PREFETCH = XE_PREFETCH_2D;
};

enum class XeStoreCachePolicy {
  kDefault,
  kUC_UC_UC,
  kUC_UC_WB,
  kUC_WB_UC,
  kUC_WB_WB,
  kWT_UC_UC,
  kWT_UC_WB,
  kWT_WB_UC,
  kWT_WB_WB,
  kWB_UC_UC,
  kWB_UC_WB,
  kWB_WB_UC
};

CUTE_HOST_DEVICE
constexpr char const*
to_string(XeStoreCachePolicy policy) {
  switch (policy) {
    case XeStoreCachePolicy::kDefault:  return "Default";
    case XeStoreCachePolicy::kUC_UC_UC: return "UC_UC_UC";
    case XeStoreCachePolicy::kUC_UC_WB: return "UC_UC_WB";
    case XeStoreCachePolicy::kUC_WB_UC: return "UC_WB_UC";
    case XeStoreCachePolicy::kUC_WB_WB: return "UC_WB_WB";
    case XeStoreCachePolicy::kWT_UC_UC: return "WT_UC_UC";
    case XeStoreCachePolicy::kWT_UC_WB: return "WT_UC_WB";
    case XeStoreCachePolicy::kWT_WB_UC: return "WT_WB_UC";
    case XeStoreCachePolicy::kWT_WB_WB: return "WT_WB_WB";
    case XeStoreCachePolicy::kWB_UC_UC: return "WB_UC_UC";
    case XeStoreCachePolicy::kWB_UC_WB: return "WB_UC_WB";
    case XeStoreCachePolicy::kWB_WB_UC: return "WB_WB_UC";
  }
  return "Unknown";
}

template <typename T>
struct is_xe_store_cache : std::false_type {};

template <XeStoreCachePolicy P>
struct is_xe_store_cache<C<P>> : std::true_type {};

template <typename T>
constexpr bool is_xe_store_cache_v = is_xe_store_cache<T>::value;

template <auto>
constexpr bool dependent_false_v = false;

#define CUTE_XE_STORE_2D_ASM(CACHE_SUFFIX)                                      \
  asm (                                                                         \
    "lsc_store_block2d.ugm" CACHE_SUFFIX                                        \
    " (M1, 1) flat[%1+(0,0)] %0:d%2.%3x%4nn"                                    \
      :: "rw"(sv), "rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height)         \
  )

template <int Bits, int Height, int Width, XeStoreCachePolicy CachePolicy = XeStoreCachePolicy::kDefault>
struct XE_STORE_2D : XE_Copy_Op_2D_Base<Bits, Height, Width>
{
  static_assert(Height <= 8, "Height exceeds hardware limits");
  static constexpr XeStoreCachePolicy StoreCachePolicy = CachePolicy;

  template <typename T>
  CUTE_HOST_DEVICE static void copy(const int *payload, const T *src) {
#ifdef CUTE_ARCH_COPY_XE_ENABLED
    using namespace intel;
    auto &sv = *reinterpret_cast<const storage_vector_t<T, Width * Height * Bits / sg_size>*>(src);
    if constexpr (CachePolicy == XeStoreCachePolicy::kDefault) {
      CUTE_XE_STORE_2D_ASM("");
    } else if constexpr (CachePolicy == XeStoreCachePolicy::kUC_UC_UC) {
      CUTE_XE_STORE_2D_ASM(".uc.uc.uc");
    } else if constexpr (CachePolicy == XeStoreCachePolicy::kUC_UC_WB) {
      CUTE_XE_STORE_2D_ASM(".uc.uc.wb");
    } else if constexpr (CachePolicy == XeStoreCachePolicy::kUC_WB_UC) {
      CUTE_XE_STORE_2D_ASM(".uc.wb.uc");
    } else if constexpr (CachePolicy == XeStoreCachePolicy::kUC_WB_WB) {
      CUTE_XE_STORE_2D_ASM(".uc.wb.wb");
    } else if constexpr (CachePolicy == XeStoreCachePolicy::kWT_UC_UC) {
      CUTE_XE_STORE_2D_ASM(".wt.uc.uc");
    } else if constexpr (CachePolicy == XeStoreCachePolicy::kWT_UC_WB) {
      CUTE_XE_STORE_2D_ASM(".wt.uc.wb");
    } else if constexpr (CachePolicy == XeStoreCachePolicy::kWT_WB_UC) {
      CUTE_XE_STORE_2D_ASM(".wt.wb.uc");
    } else if constexpr (CachePolicy == XeStoreCachePolicy::kWT_WB_WB) {
      CUTE_XE_STORE_2D_ASM(".wt.wb.wb");
    } else if constexpr (CachePolicy == XeStoreCachePolicy::kWB_UC_UC) {
      CUTE_XE_STORE_2D_ASM(".wb.uc.uc");
    } else if constexpr (CachePolicy == XeStoreCachePolicy::kWB_UC_WB) {
      CUTE_XE_STORE_2D_ASM(".wb.uc.wb");
    } else if constexpr (CachePolicy == XeStoreCachePolicy::kWB_WB_UC) {
      CUTE_XE_STORE_2D_ASM(".wb.wb.uc");
    } else {
      static_assert(dependent_false_v<CachePolicy>,
                    "Unsupported Xe store cache policy");
    }
#else
    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
#endif
  }
};

#undef CUTE_XE_STORE_2D_ASM

} // end namespace cute
