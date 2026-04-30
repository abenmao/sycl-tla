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

/*******************************************************************************
 * XE4 TMM (Tensor Matrix Multiply) — synchronous, register-based MMA operation.
 *
 * The MMA_Atom wraps a single TMM instruction:  D[32,N] = A[32,K] * B[K,N] + C[32,N]
 *
 *   M = 32  (always, fixed by hardware)
 *   K       (determined by element types — see tmm_k_for)
 *   N       (atom parameter: 1-4, 8, 16, or 32)
 *
 * CuTe's TiledMMA tiles this atom across the full workgroup tile.
 * For a 64×64×16 tile with N_atom=8:  cute::gemm() generates 2×8=16 TMM calls.
 *
 * Type -> K mapping (from TMM spec Table 2):
 *   tf32:          K = 8
 *   bf16, f16:     K = 16
 *   fp8, fp6:      K = 32
 *   fp4:           K = 32
 *   s8, u8:        K = 32
 *   s4, u4:        K = 32
 ******************************************************************************/
#pragma once

#include <cute/arch/xe4_base_datatype.hpp>
#include <cute/arch/asm_helper.hpp>       // _s<N>, fixstr::fixed_string
#include <cute/numeric/numeric_types.hpp> // sizeof_bits_v
#include <cute/util/sycl_vec.hpp>

#include <sycl/sycl.hpp>


namespace cute {

// Device-side ext_vector_type for inline asm register operands.
// On host, falls back to sycl::vec (never executed, just needs to compile).
template<class T, int N>
using tmm_vec_t = cute::intel::vector_t<T, N>;

// ============================================================================
// TMM PISA type-name mapping (lowercase, no dot prefix)
// Used to build instruction strings at compile time:
//   "tmm." + _tmm<d> + "_" + _tmm<a> + "_" + _tmm<b> + "_" + _tmm<c> + ...
// ============================================================================
template <typename> struct tmm_pisa_type;
template <> struct tmm_pisa_type<float>                        { static constexpr fixstr::fixed_string value{"f32"};  };
template <> struct tmm_pisa_type<fp16>                         { static constexpr fixstr::fixed_string value{"f16"};  };
template <> struct tmm_pisa_type<sycl::ext::oneapi::bfloat16>  { static constexpr fixstr::fixed_string value{"bf16"}; };
template <> struct tmm_pisa_type<cutlass::tfloat32_t>          { static constexpr fixstr::fixed_string value{"tf32"}; };
template <> struct tmm_pisa_type<cutlass::float_e5m2_t>        { static constexpr fixstr::fixed_string value{"e5m2"}; };
template <> struct tmm_pisa_type<cutlass::float_e4m3_t>        { static constexpr fixstr::fixed_string value{"e4m3"}; };
template <> struct tmm_pisa_type<cutlass::float_e2m1_t>        { static constexpr fixstr::fixed_string value{"e2m1"}; };
template <> struct tmm_pisa_type<int8_t>                       { static constexpr fixstr::fixed_string value{"s8"};   };
template <> struct tmm_pisa_type<uint8_t>                      { static constexpr fixstr::fixed_string value{"u8"};   };
template <typename T> constexpr auto _tmm = tmm_pisa_type<T>::value;

// ============================================================================
// Type → K mapping (hardware-determined, from TMM spec Table 2)
// ============================================================================
template <typename> struct tmm_k;
template <> struct tmm_k<cutlass::tfloat32_t>          { static constexpr int value = 8;  };
template <> struct tmm_k<fp16>                         { static constexpr int value = 16; };
template <> struct tmm_k<sycl::ext::oneapi::bfloat16>  { static constexpr int value = 16; };
template <> struct tmm_k<cutlass::float_e5m2_t>        { static constexpr int value = 32; };
template <> struct tmm_k<cutlass::float_e4m3_t>        { static constexpr int value = 32; };
template <> struct tmm_k<cutlass::float_e2m1_t>        { static constexpr int value = 32; };
template <> struct tmm_k<int8_t>                       { static constexpr int value = 32; };
template <> struct tmm_k<uint8_t>                      { static constexpr int value = 32; };

// K is determined by Major(A,B) — the type with the larger element size.
// If equal size, A is major (per TMM spec).
template <typename A, typename B>
struct tmm_k_for {
  using Major = std::conditional_t<(cute::sizeof_bits_v<A> >= cute::sizeof_bits_v<B>), A, B>;
  static constexpr int value = tmm_k<Major>::value;
};

// ============================================================================
// Fragment size computation (in uint32 units, per lane)
//
// From the TMM spec fragment tables:
//   A[32, K]: each of 32 lanes holds one row of K elements
//              ASize = ceil(K * element_bits / 32)
//   B[K, N]:  N*K elements distributed across 32 lanes
//              BSize = ceil((N * K / 32) * element_bits / 32)
//   D/C[32,N]: each lane holds one row of N elements
//              DSize = ceil(N * element_bits / 32)
// ============================================================================
template <typename T, int Count>
constexpr int tmm_frag_size() {
  constexpr int bits = cute::sizeof_bits_v<T>;
  return (Count * bits + 31) / 32;  // ceil(Count * element_bits / 32)
}


// ============================================================================
// XE4_TMM<d_type, a_type, b_type, c_type, N>
//
// Wraps ONE tmm instruction:  D[32,N] = A[32,K] * B[K,N] + C[32,N]
//   M = 32  (always, hardware-fixed)
//   K       (determined by types via tmm_k_for<a_type, b_type>)
//   N       (template parameter: 1-4, 8, 16, or 32)
//
// CuTe's TiledMMA tiles this atom to cover larger workgroup tiles.
// ============================================================================
template <class d_type, class a_type, class b_type, class c_type,
          int N=32, int kSubGroupSize=32>
struct XE4_TMM {
  static constexpr int M = 32;
  static constexpr int K = tmm_k_for<a_type, b_type>::value;
  static constexpr int N_val = N;  // expose atom-N for sg_layout computation

  using ValTypeD = d_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  using ThrID = Layout<Int<M>>;

  // Per-lane fragment sizes (uint32 units)
  static constexpr int ASize = tmm_frag_size<a_type, K>();                    // A: lane owns one row of K elems
  // B: ceil(N * K * bits(b_type) / 1024) — matches PISA spec formula
  // Uses ceiling division to avoid truncation to 0 when N*K < kSubGroupSize.
  static constexpr int BSize = (N * K * cute::sizeof_bits_v<b_type> + kSubGroupSize * 32 - 1) / (kSubGroupSize * 32);
  static constexpr int CSize = tmm_frag_size<c_type, N>();                    // C: lane owns one row of N elems
  static constexpr int DSize = tmm_frag_size<d_type, N>();                    // D: lane owns one row of N elems

  // Register array types — used by MMA_Traits for fragment allocation
  using ARegisters = sycl::vec<uint32_t, ASize>;
  using BRegisters = sycl::vec<uint32_t, BSize>;
  using CRegisters = sycl::vec<uint32_t, CSize>;
  using DRegisters = sycl::vec<uint32_t, DSize>;

  // ---- 4-operand: D = A * B + C ----
  CUTE_HOST_DEVICE static void
  fma(DRegisters& dst, ARegisters const& a, BRegisters const& b, CRegisters const& c)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    tmm_vec_t<uint32_t, DSize> sdst;
    asm volatile(
      ("tmm." + _tmm<d_type> + "_" + _tmm<a_type> + "_" + _tmm<b_type> + "_" + _tmm<c_type> +
       ".m" + _s<M> + "n" + _s<N> + "k" + _s<K> + " %0, %1, %2, %3;")
      : "=r"(sdst)
      : "r"(sycl::bit_cast<tmm_vec_t<uint32_t, ASize>>(a)),
        "r"(sycl::bit_cast<tmm_vec_t<uint32_t, BSize>>(b)),
        "r"(sycl::bit_cast<tmm_vec_t<uint32_t, CSize>>(c))
    );
    dst = sycl::bit_cast<DRegisters>(sdst);
#else
    (void)dst; (void)a; (void)b; (void)c;
#endif
  }

  // ---- 3-operand: D = A * B ----
  CUTE_HOST_DEVICE static void
  fma(DRegisters& dst, ARegisters const& a, BRegisters const& b)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    tmm_vec_t<uint32_t, DSize> sdst;
    asm volatile(
      ("tmm." + _tmm<d_type> + "_" + _tmm<a_type> + "_" + _tmm<b_type> +
       ".m" + _s<M> + "n" + _s<N> + "k" + _s<K> + " %0, %1, %2;")
      : "=r"(sdst)
      : "r"(sycl::bit_cast<tmm_vec_t<uint32_t, ASize>>(a)),
        "r"(sycl::bit_cast<tmm_vec_t<uint32_t, BSize>>(b))
    );
    dst = sycl::bit_cast<DRegisters>(sdst);
#else
    (void)dst; (void)a; (void)b;
#endif
  }
};

} // namespace cute
