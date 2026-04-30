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
 * Device-side unit test for XE4_TMM atom fma() functions.
 *
 * Exercises the real inline-asm TMM instruction path (__SYCL_DEVICE_ONLY__)
 * to confirm that the atom produces correct matrix multiply results.
 *
 * Memory model: USM shared allocators (same pattern as
 *   examples/cute/tutorial/xe4/gemm_eu_copy_matrix_atoms.cpp).
 * Kernel launch: sycl::queue::submit + handler::parallel_for + sycl::nd_range.
 *
 * Fragment layout used in the device kernel
 * ------------------------------------------
 * All three matrix fragment layouts are taken from MMA_Traits<XE4_TMM<...>>.
 *
 *   A[M=32, K]:  Row-major in memory (a_ptr[m*K+k]).
 *                Lane `i` owns row i — all K elements.
 *                ALayout(i, vid) = i*K + vid  (vid = 0..K-1)
 *                Packed: marray<uint32_t, ASize>, bit_cast from marray<TA, K>.
 *
 *   B[K, N]:     Column-major in memory (b_ptr[k + n*K]).
 *                Distributed across 32 lanes using the BLayout formula:
 *                  global_packed_idx g = lane + reg * 32
 *                  n = g / (K / packF_B),  p = g % (K / packF_B)
 *                  b_ptr index = n*K + p*packF_B + sub
 *                where packF_B = 32 / bits(TB).
 *                Packed: marray<uint32_t, BSize>, bit_cast from marray<TB, BSize*packF_B>.
 *
 *   C/D[M=32, N]: Row-major in memory (c_ptr[m*N+n]).
 *                 Lane `i` owns row i — all N elements.
 *                 CLayout(i, vid) = i*N + vid  (vid = 0..N-1)
 *                 Packed: marray<uint32_t, CSize>, bit_cast from marray<TC, N>.
 *
 * Test coverage
 * -------------
 *  Test 1 — 4-operand FMA: D = A*B + C
 *    XE4_TMM<float, sycl::half, sycl::half, float, 8>  (fp16, N=8, K=16)
 *
 *  Test 2 — 3-operand FMA: D = A*B
 *    Same atom type, C omitted; verifies the zero-accumulate path.
 *
 *  Test 3 — Different data types
 *    XE4_TMM<float, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, float, 16>
 *    (bf16, N=16, K=16)
 ******************************************************************************/

#include "cutlass_unit_test.h"

#include "tiled_mma_utils.hpp"

#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include <cstdint>

#include <cute/atom/mma_atom.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cute/arch/mma_xe4_tmm.hpp>
#include <cute/atom/mma_traits_xe4_tmm.hpp>

using namespace cute;

// ============================================================================
// Reference computation: D[m,n] = sum_k A[m,k] * B[k,n] + C[m,n]
// All matrices stored row-major.
// ============================================================================
template <typename TA, typename TB, typename TC, typename TD>
void gemm_ref(const TA* a_ptr, const TB* b_ptr, const TC* c_ptr, TD* d_ptr,
              int M, int K, int N, bool with_c)
{
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += float(a_ptr[m * K + k]) * float(b_ptr[k + n*K]);
      }
      if (with_c) {
        acc += float(c_ptr[m * N + n]);
      }
      d_ptr[m * N + n] = static_cast<TD>(acc);
    }
  }
}

template <typename TA, typename TB, typename TC, typename TD>
void gemm_ref_mkn(const TA* a_ptr, const TB* b_ptr, const TC* c_ptr, TD* d_ptr,
              int M, int K, int N, bool with_c)
{
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += float(a_ptr[m * K + k]) * float(b_ptr[k * N + n]);
      }
      if (with_c) {
        acc += float(c_ptr[m * N + n]);
      }
      d_ptr[m * N + n] = static_cast<TD>(acc);
    }
  }
}

// ============================================================================
// Tolerance helper for near-equal float comparison.
// ============================================================================
static bool float_near(float got, float ref, float abs_tol = 0.5f, float rel_tol = 2e-3f) {
  float tol = abs_tol + rel_tol * std::abs(ref);
  return std::abs(got - ref) <= tol;
}


template<typename TA, typename TB, typename TC, typename TD>
bool gemm_ref_host_check(const TA *a_ptr, const TB*b_ptr, const TC *c_ptr,
    const TD* dataD, int M, int K, int N) {
  std::vector<TD> d_ref(M*N);

  gemm_ref_mkn(a_ptr, b_ptr, c_ptr, d_ref.data(), M, K, N, true);
  int bad = 0;
  for (int m = 0; m < M && bad < 10; ++m) {
    for (int n = 0; n < N && bad < 10; ++n) {
      float got = float(dataD[m * N + n]);
      float ref = float(d_ref[m * N + n]);
      if (!float_near(got, ref)) {
        std::cout << "mismatch at (m=" << m << ", n=" << n << "): " 
          << "got=" << got << " ref=" << ref << std::endl;;
        ++bad;
      }
    }
  }
  return (bad == 0);
}

// ============================================================================
// run_tmm_kernel — shared device kernel launcher for both FMA4 and FMA3.
//
// Template parameters:
//   MMA_Op  — XE4_TMM instantiation (encodes ValTypeA/B/C/D, M, K, N)
//   UseC    — true  → 4-operand fma(D, A, B, C)
//             false → 3-operand fma(D, A, B)
//
// Arguments:
//   q       — SYCL queue to submit on
//   a_ptr   — input A[M,K], row-major (USM or device-accessible)
//   b_ptr   — input B[K,N], column-major: b_ptr[k + n*K] (USM or device-accessible)
//   c_ptr   — input C[M,N], row-major (ignored when UseC == false)
//   d_ptr   — output D[M,N], row-major
// ============================================================================
template <typename MMA_Op, bool UseC>
void run_tmm_kernel(sycl::queue& q,
                    typename MMA_Op::ValTypeA* a_ptr,
                    typename MMA_Op::ValTypeB* b_ptr,
                    typename MMA_Op::ValTypeC* c_ptr,
                    typename MMA_Op::ValTypeD* d_ptr)
{
  constexpr int workgroup_size = 32;
  sycl::nd_range<3> nd_range(sycl::range<3>(1, 1, workgroup_size),
                              sycl::range<3>(1, 1, workgroup_size));

  q.submit([&](sycl::handler& h) {
    h.parallel_for(nd_range,
      [=](sycl::nd_item<3> it)
        [[sycl::reqd_work_group_size(1, 1, workgroup_size)]]
        [[sycl::reqd_sub_group_size(32)]]
      {
        using MMA = MMA_Op;

        constexpr int K_atom = MMA::K;
        constexpr int N_atom = MMA::N_val;

        constexpr int ASize = MMA::ASize;
        constexpr int BSize = MMA::BSize;
        constexpr int CSize = MMA::CSize;
        constexpr int DSize = MMA::DSize;

        using ValA = typename MMA::ValTypeA;
        using ValB = typename MMA::ValTypeB;
        using ValC = typename MMA::ValTypeC;
        using ValD = typename MMA::ValTypeD;

        constexpr int packF_A = 32 / cute::sizeof_bits_v<ValA>;
        constexpr int packF_B = 32 / cute::sizeof_bits_v<ValB>;
        constexpr int packF_C = 32 / cute::sizeof_bits_v<ValC>;
        constexpr int packF_D = 32 / cute::sizeof_bits_v<ValD>;

        constexpr int aVals = ASize * packF_A;  // = K_atom
        constexpr int bVals = BSize * packF_B;  // = N_atom * K_atom / 32
        constexpr int cVals = CSize * packF_C;  // = N_atom
        constexpr int dVals = DSize * packF_D;  // = N_atom

        int lane_id = (int)it.get_sub_group().get_local_id();

        // Load A: lane `lane_id` owns row `lane_id` of A[M,K]
        // a_ptr is row-major: a_ptr[m*K+k]. Lane i → a_ptr[i*K .. i*K+K-1].
        sycl::marray<ValA, aVals> a_vals;
        for (int k = 0; k < K_atom; ++k)
          a_vals[k] = a_ptr[lane_id * K_atom + k];
        auto a_regs = sycl::bit_cast<typename MMA::ARegisters>(a_vals);
        //auto& a_regs = reinterpret_cast<typename MMA::ARegisters&>(a_vals);

        // Load B: B[K,N] is column-major — b_ptr[k + n*K].
        // Lane l holds global packed entries at positions g = l, l+32, l+64, ..., l+(BSize-1)*32.
        // Each g maps to column n = g / (K/packF_B) and packed-K index p = g % (K/packF_B),
        // so b_ptr index = n*K + p*packF_B + sub  ==  (lane + reg*32)*packF_B + sub.
        sycl::marray<ValB, bVals> b_vals;
        for (int r = 0; r < BSize; ++r)
          for (int s = 0; s < packF_B; ++s)
            b_vals[r * packF_B + s] = b_ptr[lane_id * packF_B + s + r * 32 * packF_B];
        auto b_regs = sycl::bit_cast<typename MMA::BRegisters>(b_vals);
        //auto& b_regs = reinterpret_cast<typename MMA::BRegisters&>(b_vals);

        typename MMA::DRegisters d_regs;

        if constexpr (UseC) {
          // Load C: lane `lane_id` owns row `lane_id` of C[M,N]
          sycl::marray<ValC, cVals> c_vals;
          for (int n = 0; n < N_atom; ++n)
            c_vals[n] = c_ptr[lane_id * N_atom + n];
          auto c_regs = sycl::bit_cast<typename MMA::CRegisters>(c_vals);
          //auto& c_regs = reinterpret_cast<typename MMA::CRegisters&>(c_vals);

          // 4-operand: D = A*B + C
          MMA::fma(d_regs, a_regs, b_regs, c_regs);
        } else {
          // 3-operand: D = A*B
          MMA::fma(d_regs, a_regs, b_regs);
        }

        // Store D: lane `lane_id` stores row `lane_id` of D[M,N]
        auto d_vals = sycl::bit_cast<sycl::marray<ValD, dVals>>(d_regs);
        for (int n = 0; n < N_atom; ++n)
          d_ptr[lane_id * N_atom + n] = d_vals[n];
      });
  }).wait();
}

template<int bM, int bN, int bK, int wgSize, typename MMA_Op>
CUTE_HOST_DEVICE
void cute_gemm_kernel(
    typename MMA_Op::ValTypeA const *A, typename MMA_Op::ValTypeB const *B,
    typename MMA_Op::ValTypeC const *C, typename MMA_Op::ValTypeD *D,
    sycl::nd_item<1> it, sycl::stream out) {

  auto my_sg_id = it.get_sub_group().get_group_id();
  auto my_lane_id = it.get_sub_group().get_local_id(); // within the subgroup //
  auto my_wg_level_lane_id = it.get_local_id(0); // within the workgroup //


  auto thr_sg_layout = typename default_warp_layout_for_tiled_mma<MMA_Op,
       bM, bN, bK, wgSize>::type{}; 
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Op>{}, thr_sg_layout);

  // Create tensors //
  auto gA = make_tensor(make_gmem_ptr(A), make_shape(Int<bM>{}, Int<bK>{}),
        make_stride(Int<bK>{}, _1{}));
  auto gB = make_tensor(make_gmem_ptr(B), make_shape(Int<bN>{}, Int<bK>{}),
        make_stride(_1{}, Int<bN>{})); // transposed //
  auto gC = make_tensor(make_gmem_ptr(C), make_shape(Int<bM>{}, Int<bN>{}),
        make_stride(Int<bN>{}, _1{}));
  auto gD = make_tensor(make_gmem_ptr(D), make_shape(Int<bM>{}, Int<bN>{}),
        make_stride(Int<bN>{}, _1{}));

  auto thr_mma = tiled_mma.get_slice((int)my_wg_level_lane_id);
  // Partition the tensors into atoms tiles//
  auto tAgA = thr_mma.partition_A(gA);
  auto tBgB = thr_mma.partition_B(gB);
  auto tCgC = thr_mma.partition_C(gC);
  auto tDgD = thr_mma.partition_C(gD);

  // Allocate registers for each of the lanes //
  auto tArA = thr_mma.make_fragment_A(tAgA);
  auto tBrB = thr_mma.make_fragment_B(tBgB);
  auto tDrD = thr_mma.make_fragment_C(tDgD);

  //XeSim-Bug(vamsikku): if we use copy directly it will create a ld.global.v8.32 
  // instruction which XeSim fails to simulate with message:
  //
  // "partial simd16 processing failed! Remove this fatal once partial simd16 is stable..."
  //
  //
  
  //copy(tAgA, tArA);
  //copy(tBgB, tBrB);
  copy(tCgC, tDrD);

  for (int i=0; i<size(tAgA); i++) {
    tArA(i) = tAgA(i);
  }

  for (int i=0; i<size(tBgB); i++) {
    tBrB(i) = tBgB(i);
  }

  // Call cute::gemm with tiled_mma directly //
  // TODO(vamsikku): currently direct call gemm will do a scalar explosion of
  // the fma also extent does not directly work on marray it needs to be replaced
  // with carrays and Variadic templates which can unpack arguments //
  //gemm(tiled_mma, tArA, tBrB, tDrD);

  // Extract the sizes of the REST dimensions
  // tCrC is ((Vals), REST_M, REST_N), so size<1> is M, size<2> is N
  int const M_iters = cute::size<1>(tDrD);
  int const N_iters = cute::size<2>(tDrD);
  int const K_iters = cute::size<2>(tArA);

  if (!my_lane_id) {
    out << "my_wg_lane_id=" << my_wg_level_lane_id << sycl::endl; 
    out << "sg=" << my_sg_id << " " << "(rest: m x n x k)="
        << M_iters << " x " << N_iters << " x " << K_iters << " Addr="
        << &tAgA(0) << sycl::endl;;
  }

  CUTE_UNROLL
  for (int k = 0; k < K_iters; ++k) {
    CUTE_UNROLL
    for (int m = 0; m < M_iters; ++m) {
      CUTE_UNROLL
      for (int n = 0; n < N_iters; ++n) {
        MMA_Op::fma(
					*reinterpret_cast<typename MMA_Op::DRegisters *>(&tDrD(0, m, n)),
          *reinterpret_cast<typename MMA_Op::ARegisters const *>(&tArA(0, m, k)),
          *reinterpret_cast<typename MMA_Op::BRegisters const *>(&tBrB(0, n, k)),
          *reinterpret_cast<typename MMA_Op::DRegisters *>(&tDrD(0, m, n))
        );
      }
    }
  }
  copy(tDrD, tDgD);
}


template<typename T>
using usm_allocator_t = sycl::usm_allocator<T, sycl::usm::alloc::shared>;
template<typename T>
using usm_vector_t = std::vector<T, usm_allocator_t<T>>;

enum class GemmInputInitMode {
  kBaseline,
  kRandom
};

constexpr uint32_t kDefaultRandomSeed = 0xC0FFEEu;

template <typename mma_op_t, int bM, int bN, int bK, int subgroup_count>
bool run_cute_gemm_kernel(GemmInputInitMode init_mode = GemmInputInitMode::kBaseline,
                          uint32_t seed = kDefaultRandomSeed) {
  using TA = typename mma_op_t::ValTypeA;
  using TB = typename mma_op_t::ValTypeB;
  using TC = typename mma_op_t::ValTypeC;
  using TD = typename mma_op_t::ValTypeD;

  static_assert(bM > 0 && bN > 0 && bK > 0,
                "Tile dimensions bM, bN, and bK must be positive");
  static_assert(subgroup_count > 0, "subgroup_count must be positive");

  sycl::queue q;
  usm_allocator_t<TA> alloc_TA(q);
  usm_allocator_t<TB> alloc_TB(q);
  usm_allocator_t<TC> alloc_TC(q);
  usm_allocator_t<TD> alloc_TD(q);


  usm_vector_t<TA> dataA(bM * bK, alloc_TA);
  usm_vector_t<TB> dataB(bK * bN, alloc_TB);
  usm_vector_t<TC> dataC(bM * bN, alloc_TC);
  usm_vector_t<TD> dataD(bM * bN, alloc_TD);

  constexpr uint32_t kLcgMul = 1664525u;
  constexpr uint32_t kLcgAdd = 1013904223u;
  constexpr int kRandomBias = 16;
  constexpr float kRandomScale = 0.125f;
  auto next_random = [&seed]() {
    seed = seed * kLcgMul + kLcgAdd;
    int value = static_cast<int>((seed >> 27) & 0x1f) - kRandomBias;
    return static_cast<float>(value) * kRandomScale;
  };

  for (size_t i = 0; i < bM; i++) {
    for (size_t j = 0; j < bK; j++) {
      if (init_mode == GemmInputInitMode::kRandom) {
        dataA[i * bK + j] = static_cast<TA>(next_random());
      } else {
        dataA[i * bK + j] = (i % bK == j) ? static_cast<TA>(1) : static_cast<TA>(0);
      }
    }
  }

  for (size_t i = 0; i < bK; i++) {
    for (size_t j = 0; j < bN; j++) {
      if (init_mode == GemmInputInitMode::kRandom) {
        dataB[i * bN + j] = static_cast<TB>(next_random());
      } else {
        dataB[i * bN + j] = static_cast<TB>((int)i);
      }
    }
  }
  for (size_t i = 0; i < bM * bN; i++) {
    if (init_mode == GemmInputInitMode::kRandom) {
      dataC[i] = static_cast<TC>(next_random());
    } else {
      constexpr int kCBaselineMod = 7;
      constexpr int kCBaselineBias = 3;
      dataC[i] = static_cast<TC>((static_cast<int>(i) % kCBaselineMod) - kCBaselineBias);
    }
    dataD[i] = 0;
  }

  constexpr int total_size = 32 * subgroup_count;
  constexpr int workgroup_size = 32 * subgroup_count;
  constexpr int wgSize = workgroup_size;

  std::cout << "Tiled MMA SG Layout: " << std::endl;
  using SgLayoutMeta = default_warp_layout_for_tiled_mma<mma_op_t, bM, bN, bK, wgSize>;
  static_assert(SgLayoutMeta::num_consumer_sgs == subgroup_count,
                "subgroup_count parameter must match default_warp_layout_for_tiled_mma "
                "consumer subgroup count");
  auto thr_sg_layout = typename SgLayoutMeta::type{};
  std::cout << std::endl;

  std::cout << thr_sg_layout << std::endl;

  sycl::range<1> global_range{total_size};
  sycl::range<1> local_range{workgroup_size};
  sycl::nd_range<1> execution_range{global_range, local_range};

  TA *A_ptr = dataA.data();
  TB *B_ptr = dataB.data();
  TC *C_ptr = dataC.data();
  TD *D_ptr = dataD.data();

  q.submit(
      [&](sycl::handler& h) {
        auto out = sycl::stream(1024*1024, 1024, h);
        h.parallel_for(execution_range,
          [=](sycl::nd_item<1> item) 
          [[sycl::reqd_work_group_size(workgroup_size)]]
          [[sycl::reqd_sub_group_size(32)]]
          {
            cute_gemm_kernel<bM, bN, bK, wgSize, mma_op_t>(
                A_ptr, B_ptr, C_ptr, D_ptr, item, out);
          }
        );
      } 
  ).wait();


#if DEBUG_TEST 
  auto gA = make_tensor(make_gmem_ptr(A_ptr), make_shape(Int<bM>{}, Int<bK>{}),
        make_stride(Int<bK>{}, _1{}));
  auto gB = make_tensor(make_gmem_ptr(B_ptr), make_shape(Int<bK>{}, Int<bN>{}),
        make_stride(Int<bN>{}, _1{})); // transposed //
  auto gC = make_tensor(make_gmem_ptr(C_ptr), make_shape(Int<bM>{}, Int<bN>{}),
        make_stride(_1{}, Int<bN>{}));
  auto gD = make_tensor(make_gmem_ptr(D_ptr), make_shape(Int<bM>{}, Int<bN>{}),
        make_stride(Int<bN>{}, _1{}));

  std::cout << "=====A=====" << std::endl;
  std::cout << gA << std::endl;
  std::cout << "=====B=====" << std::endl;
  std::cout << gB << std::endl;
  std::cout << "=====C=====" << std::endl;
  std::cout << gC << std::endl;
  std::cout << "result" << std::endl;
  std::cout << gD << std::endl;
  std::cout << "============" << std::endl;
#endif

  bool result =
    gemm_ref_host_check<TA, TB, TC, TD>(A_ptr, B_ptr, C_ptr, D_ptr, bM, bK, bN);
  return result;
}



TEST(Test_Cute_Gemm_MMA, tmm_cute_gemm_fp16_n1_bM128_bN1_bK16_sg4) {
  EXPECT_TRUE((run_cute_gemm_kernel<XE4_TMM<float, sycl::half, sycl::half, float, 1>,
                                     128, 1, 16, 4>()));
}

TEST(Test_Cute_Gemm_MMA, tmm_cute_gemm_fp16_n1_bM128_bN1_bK16_sg4_random) {
  EXPECT_TRUE((run_cute_gemm_kernel<XE4_TMM<float, sycl::half, sycl::half, float, 1>,
                                     128, 1, 16, 4>(GemmInputInitMode::kRandom, 0x1A2B3C4Du)));
}

TEST(Test_Cute_Gemm_MMA, tmm_cute_gemm_fp16_n8_bM32_bN8_bK16_sg1) {
  EXPECT_TRUE((run_cute_gemm_kernel<XE4_TMM<float, sycl::half, sycl::half, float, 8>,
                                     32, 8, 16, 1>()));
}

TEST(Test_Cute_Gemm_MMA, tmm_cute_gemm_bf16_n16_bM64_bN16_bK16_sg2) {
  EXPECT_TRUE((run_cute_gemm_kernel<XE4_TMM<float,
                                            sycl::ext::oneapi::bfloat16,
                                            sycl::ext::oneapi::bfloat16,
                                            float, 16>, 64, 16, 16, 2>()));
}

TEST(Test_Cute_Gemm_MMA, tmm_cute_gemm_fp8_e5m2_n16_bM32_bN16_bK32_sg1) {
  EXPECT_TRUE((run_cute_gemm_kernel<XE4_TMM<float,
                                            cutlass::float_e5m2_t,
                                            cutlass::float_e5m2_t,
                                            float, 16>, 32, 16, 32, 1>()));
}

TEST(Test_Cute_Gemm_MMA, tmm_cute_gemm_fp8_e5m2_n16_bM32_bN16_bK32_sg1_random) {
  EXPECT_TRUE((run_cute_gemm_kernel<XE4_TMM<float,
                                            cutlass::float_e5m2_t,
                                            cutlass::float_e5m2_t,
                                            float, 16>, 32, 16, 32, 1>(
      GemmInputInitMode::kRandom, 0x5EEDC0DEu)));
}



// ============================================================================
// test_tmm_fma4 — launches the 4-operand kernel (D = A*B + C) and verifies.
//
// Accepts raw USM pointers for A, B, C (inputs) and D (output).
// ============================================================================
template <typename MMA_Op>
void test_tmm_fma4(sycl::queue& q,
                   typename MMA_Op::ValTypeA* dataA,
                   typename MMA_Op::ValTypeB* dataB,
                   typename MMA_Op::ValTypeC* dataC,
                   typename MMA_Op::ValTypeD* dataD)
{
  using TD = typename MMA_Op::ValTypeD;
  constexpr int M = MMA_Op::M;
  constexpr int K = MMA_Op::K;
  constexpr int N = MMA_Op::N_val;

  run_tmm_kernel<MMA_Op, /*UseC=*/true>(q, dataA, dataB, dataC, dataD);

  // Host reference: D_ref = A * B + C
  std::vector<TD> d_ref(M * N);
  gemm_ref(dataA, dataB, dataC, d_ref.data(), M, K, N, /*with_c=*/true);

  int bad = 0;
  for (int m = 0; m < M && bad < 10; ++m) {
    for (int n = 0; n < N && bad < 10; ++n) {
      float got = float(dataD[m * N + n]);
      float ref = float(d_ref[m * N + n]);
      EXPECT_TRUE(float_near(got, ref))
          << "FMA4 mismatch at (m=" << m << ", n=" << n << "): "
          << "got=" << got << " ref=" << ref;
      if (!float_near(got, ref)) ++bad;
    }
  }
}

// ============================================================================
// test_tmm_fma3 — launches the 3-operand kernel (D = A*B) and verifies.
//
// Accepts raw USM pointers for A, B, C (inputs) and D (output).
// C is accepted for interface symmetry with test_tmm_fma4 but is not used
// in the TMM call itself.
// ============================================================================
template <typename MMA_Op>
void test_tmm_fma3(sycl::queue& q,
                   typename MMA_Op::ValTypeA* dataA,
                   typename MMA_Op::ValTypeB* dataB,
                   typename MMA_Op::ValTypeC* dataC,
                   typename MMA_Op::ValTypeD* dataD)
{
  using TC = typename MMA_Op::ValTypeC;
  using TD = typename MMA_Op::ValTypeD;
  constexpr int M = MMA_Op::M;
  constexpr int K = MMA_Op::K;
  constexpr int N = MMA_Op::N_val;

  // Pass c_ptr but UseC=false — the kernel ignores it
  run_tmm_kernel<MMA_Op, /*UseC=*/false>(q, dataA, dataB, dataC, dataD);

  // Host reference: D_ref = A * B  (C not accumulated)
  std::vector<TC> c_zero(M * N, TC(0));
  std::vector<TD> d_ref(M * N);
  gemm_ref(dataA, dataB, c_zero.data(), d_ref.data(), M, K, N, /*with_c=*/false);

  int bad = 0;
  for (int m = 0; m < M && bad < 10; ++m) {
    for (int n = 0; n < N && bad < 10; ++n) {
      float got = float(dataD[m * N + n]);
      float ref = float(d_ref[m * N + n]);
      EXPECT_TRUE(float_near(got, ref))
          << "FMA3 mismatch at (m=" << m << ", n=" << n << "): "
          << "got=" << got << " ref=" << ref;
      if (!float_near(got, ref)) ++bad;
    }
  }
}

// ============================================================================
// XE4TMMTest<MMA_Op> — Google Test fixture for XE4_TMM fma() tests.
//
// Provides a shared sycl::queue and USM input/output buffers (dataA, dataB,
// dataC, dataD) with standard fill patterns, eliminating boilerplate from
// each TEST_F body.
//
// SetUp()    — allocates USM shared memory and fills with test data.
// TearDown() — frees USM memory.
// ============================================================================
template <typename MMA_Op>
class XE4TMMTest : public ::testing::Test {
protected:
  using TA = typename MMA_Op::ValTypeA;
  using TB = typename MMA_Op::ValTypeB;
  using TC = typename MMA_Op::ValTypeC;
  using TD = typename MMA_Op::ValTypeD;

  static constexpr int M = MMA_Op::M;
  static constexpr int K = MMA_Op::K;
  static constexpr int N = MMA_Op::N_val;

  sycl::queue q;

  TA* dataA = nullptr;
  TB* dataB = nullptr;
  TC* dataC = nullptr;
  TD* dataD = nullptr;

  void SetUp() override {
    dataA = sycl::malloc_shared<TA>(M * K, q);
    dataB = sycl::malloc_shared<TB>(K * N, q);
    dataC = sycl::malloc_shared<TC>(M * N, q);
    dataD = sycl::malloc_shared<TD>(M * N, q);

    // Standard fill: A[m,k] = m%4+1, B[k,n] = n%4+1, C[m,n] = m+n, D = 0
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < K; ++k)
        dataA[m * K + k] = static_cast<TA>(m % 4 + 1);
    for (int k = 0; k < K; ++k)
      for (int n = 0; n < N; ++n)
        dataB[k + n*K] = static_cast<TB>(n % 4 + 1);
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        dataC[m * N + n] = static_cast<TC>(m + n);
        dataD[m * N + n] = static_cast<TD>(0);
      }
  }

  void TearDown() override {
    sycl::free(dataA, q);
    sycl::free(dataB, q);
    sycl::free(dataC, q);
    sycl::free(dataD, q);
  }
};

// ============================================================================
// Concrete fixture aliases — one per atom configuration under test.
// ============================================================================
using FP16_N8_MMA  = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
using FP16_N1_MMA  = XE4_TMM<float, sycl::half, sycl::half, float, 1>;
using FP16_N2_MMA  = XE4_TMM<float, sycl::half, sycl::half, float, 2>;
using BF16_N16_MMA = XE4_TMM<float,
                               sycl::ext::oneapi::bfloat16,
                               sycl::ext::oneapi::bfloat16,
                               float, 16>;

using XE4TMM_FP16_N8_Test  = XE4TMMTest<FP16_N8_MMA>;
using XE4TMM_FP16_N1_Test  = XE4TMMTest<FP16_N1_MMA>;
using XE4TMM_BF16_N16_Test = XE4TMMTest<BF16_N16_MMA>;


TEST(Test_XE4_CUTE_TMM_Atom, test_core_tv_layouts_in_cute_layout_algebra) {
  using namespace cute;
  MMA_Atom<FP16_N1_MMA> mma_atom;
  typedef MMA_Traits<FP16_N1_MMA> mma_traits_t; 
  typedef typename mma_traits_t::ValTypeA a_type_t; 
  typedef typename mma_traits_t::ValTypeB b_type_t; 
  typedef typename mma_traits_t::ValTypeC c_type_t; 

  auto tiled_mma = make_tiled_mma(mma_atom,
        make_layout(make_shape(_1{}, _1{}, _1{}) ));

  //print(tiled_mma);

  // create A //
  constexpr int M = get<0>( mma_traits_t::Shape_MNK{} );
  constexpr int N = get<1>( mma_traits_t::Shape_MNK{} );
  constexpr int K = get<2>( mma_traits_t::Shape_MNK{} );

  //std::cout << "M=" << M << " N=" << N << " K=" << K << std::endl;

  auto gA = make_tensor<a_type_t>(
      make_layout(Shape<Int<M>, Int<K>>{}, Stride<Int<K>, _1>{}));
  for (int i=0; i<size(gA); i++) { gA(i) = i; }

  auto gB = make_tensor<b_type_t>(
      make_layout(Shape<Int<K>, Int<N>>{}, Stride<Int<N>, _1>{}));
  for (int i=0; i<size(gB); i++) { gB(i) = i; }

  auto gAfound = make_tensor<a_type_t>(layout(gA));
  auto gBfound = make_tensor<b_type_t>(layout(gB));

  for (int i=0; i<size(gAfound); i++) { gAfound(i) = -1; }
  for (int i=0; i<size(gBfound); i++) { gBfound(i) = -1; }

  const int thread_count = size<0>(mma_traits_t::ThrID{});
  for (size_t i=0; i<thread_count; i++) {
    //std::cout << "=========ThrID=" << i << "=================" << std::endl;
    auto thr_mma = tiled_mma.get_slice( (int) i );
    auto tAgA = thr_mma.partition_A(gA);
    auto tAgAfound = thr_mma.partition_A(gAfound);
    ASSERT_EQ(size(tAgA), size(tAgAfound));

    for (size_t t=0; t<size(tAgA); t++) {
      tAgAfound(t) = tAgA(t);
    }

    auto tBgB = thr_mma.partition_B(gB);
    auto tBgBfound = thr_mma.partition_B(gBfound);
    ASSERT_EQ(size(tBgB), size(tBgBfound));

    for (size_t t=0; t<size(tBgB); t++) {
      tBgBfound(t) = tBgB(t);
    }
  }


  ASSERT_EQ(size(gA), size(gAfound));
  ASSERT_EQ(size(gB), size(gBfound));
  for (int i=0; i<size(gA); i++) {
    ASSERT_EQ(gA(i), gAfound(i));
  }
  for (int i=0; i<size(gB); i++) {
    ASSERT_EQ(gB(i), gBfound(i));
  }
}
// Unit tests to verify the TV layouts //
TEST(Test_XE4_TMM_TV_layouts, latex_tv_layouts) {
  using namespace cute;
  {
    printf("FP16_N16_MMA\n");
    using FP16_N16_MMA = XE4_TMM<float, sycl::half, sycl::half, float, 16>;
		MMA_Atom<FP16_N16_MMA> atom;
		print(atom);
		// Print TV layouts in detail
		printf("=== TV Layout for B: (tid, vid) -> flat (m,k) coord ===\n");
		print_latex(typename decltype(atom)::LayoutB_TV{}); printf("\n"); 
  }

  { 
    printf("FP16_N32_MMA\n");
    using FP16_N32_MMA = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
		MMA_Atom<FP16_N32_MMA> atom;
		print(atom);
		// Print TV layouts in detail
		printf("=== TV Layout for B: (tid, vid) -> flat (m,k) coord ===\n");
		print_latex(typename decltype(atom)::LayoutB_TV{}); printf("\n"); 
  }

}


// ============================================================================
// TEST_F cases
// ============================================================================

// fp16 N=1 3-operand
TEST_F(XE4TMM_FP16_N1_Test, FMA3) {
  test_tmm_fma3<FP16_N1_MMA>(q, dataA, dataB, dataC, dataD);
}

// fp16 N=8 — 4-operand FMA: D = A*B + C
TEST_F(XE4TMM_FP16_N8_Test, FMA4) {
  test_tmm_fma4<FP16_N8_MMA>(q, dataA, dataB, dataC, dataD);
}

// fp16 N=8 — 3-operand FMA: D = A*B
TEST_F(XE4TMM_FP16_N8_Test, FMA3) {
  test_tmm_fma3<FP16_N8_MMA>(q, dataA, dataB, dataC, dataD);
}

// bf16 N=16 — 4-operand FMA: D = A*B + C
TEST_F(XE4TMM_BF16_N16_Test, FMA4) {
  test_tmm_fma4<BF16_N16_MMA>(q, dataA, dataB, dataC, dataD);
}

// ============================================================================
// Compile-time fragment size validation against PISA spec tables.
//
// All three formulas (ASize, BSize, DSize/CSize) are tested via static_assert
// inside TEST() bodies, so failures show up as both compile errors and runtime
// test failures.
//
// Fragment size formulas:
//   ASize         = ceil(K * bits(A) / 32)
//   BSize         = ceil(N * K * bits(B) / 1024)
//   CSize = DSize = ceil(N * bits(D/C) / 32)
//
// Type aliases used in tests:
//   tf32  = cutlass::tfloat32_t  (32-bit, K=8)
//   fp16  = sycl::half           (16-bit, K=16)
//   bf16  = sycl::ext::oneapi::bfloat16 (16-bit, K=16)
//   e5m2  = cute::float_e5m2_t   (8-bit,  K=32)
//   e2m1  = cute::float_e2m1_t   (4-bit,  K=32)
// ============================================================================

// -- Matrix A fragment sizes (ASize = ceil(K * bits(A) / 32)) --

TEST(XE4TMMFragmentSizes, ASize) {
  // tf32 → K=8,  32-bit A: ceil(8*32/32)  = 8
  static_assert(XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 1>::ASize == 8);
  // f16  → K=16, 16-bit A: ceil(16*16/32) = 8
  static_assert(XE4_TMM<float, sycl::half, sycl::half, float, 1>::ASize == 8);
  // bf16 → K=16, 16-bit A: ceil(16*16/32) = 8
  static_assert(XE4_TMM<float, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, float, 1>::ASize == 8);
  // e5m2 → K=32,  8-bit A: ceil(32*8/32)  = 8
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e5m2_t, float, 1>::ASize == 8);
  // e2m1 → K=32,  4-bit A: ceil(32*4/32)  = 4
  static_assert(XE4_TMM<float, cute::float_e2m1_t, cute::float_e2m1_t, float, 1>::ASize == 4);
  // Mixed f16/e5m2: major=f16, K=16, A is f16 (16-bit): ceil(16*16/32) = 8
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 1>::ASize == 8);
  SUCCEED();
}

// -- Matrix B fragment sizes (BSize = ceil(N * K * bits(B) / 1024)) --
//
// Full PISA spec B-fragment table — register count per lane:
//
// BSize = ceil(N * K * bits(B) / 1024)
//
// N  | k8,32 | k16,16 | k32,8 | k16,8 | k32,4 | k16,4
// ---+-------+--------+-------+-------+-------+------
//  1 |  1    |   1    |   1   |   1   |   1   |   1
//  2 |  1    |   1    |   1   |   1   |   1   |   1
//  3 |  1    |   1    |   1   |   1   |   1   |   1
//  4 |  1    |   1    |   1   |   1   |   1   |   1
//  5 |  2    |   2    |   2   |   1   |   1   |   1
//  6 |  2    |   2    |   2   |   1   |   1   |   1
//  7 |  2    |   2    |   2   |   1   |   1   |   1
//  8 |  2    |   2    |   2   |   1   |   1   |   1
//  9 |  3    |   3    |   3   |   2   |   2   |   1
// 10 |  3    |   3    |   3   |   2   |   2   |   1
// 11 |  3    |   3    |   3   |   2   |   2   |   1
// 12 |  3    |   3    |   3   |   2   |   2   |   1
// 13 |  4    |   4    |   4   |   2   |   2   |   1
// 14 |  4    |   4    |   4   |   2   |   2   |   1
// 15 |  4    |   4    |   4   |   2   |   2   |   1
// 16 |  4    |   4    |   4   |   2   |   2   |   1
// 17 |  5    |   5    |   5   |   3   |   3   |   2
// 18 |  5    |   5    |   5   |   3   |   3   |   2
// 19 |  5    |   5    |   5   |   3   |   3   |   2
// 20 |  5    |   5    |   5   |   3   |   3   |   2
// 21 |  6    |   6    |   6   |   3   |   3   |   2
// 22 |  6    |   6    |   6   |   3   |   3   |   2
// 23 |  6    |   6    |   6   |   3   |   3   |   2
// 24 |  6    |   6    |   6   |   3   |   3   |   2
// 25 |  7    |   7    |   7   |   4   |   4   |   2
// 26 |  7    |   7    |   7   |   4   |   4   |   2
// 27 |  7    |   7    |   7   |   4   |   4   |   2
// 28 |  7    |   7    |   7   |   4   |   4   |   2
// 29 |  8    |   8    |   8   |   4   |   4   |   2
// 30 |  8    |   8    |   8   |   4   |   4   |   2
// 31 |  8    |   8    |   8   |   4   |   4   |   2
// 32 |  8    |   8    |   8   |   4   |   4   |   2

// k8,32:  tf32/tf32 → K=8,  B 32-bit  (BSize = ceil(N/4))
TEST(XE4TMMFragmentSizes, BSize_k8_32bit) {
  using T = cutlass::tfloat32_t;
  // N=1..4 → 1
  static_assert(XE4_TMM<float, T, T, float,  1>::BSize == 1);
  static_assert(XE4_TMM<float, T, T, float,  2>::BSize == 1);
  static_assert(XE4_TMM<float, T, T, float,  3>::BSize == 1);
  static_assert(XE4_TMM<float, T, T, float,  4>::BSize == 1);
  // N=5..8 → 2
  static_assert(XE4_TMM<float, T, T, float,  5>::BSize == 2);
  static_assert(XE4_TMM<float, T, T, float,  6>::BSize == 2);
  static_assert(XE4_TMM<float, T, T, float,  7>::BSize == 2);
  static_assert(XE4_TMM<float, T, T, float,  8>::BSize == 2);
  // N=9..12 → 3
  static_assert(XE4_TMM<float, T, T, float,  9>::BSize == 3);
  static_assert(XE4_TMM<float, T, T, float, 10>::BSize == 3);
  static_assert(XE4_TMM<float, T, T, float, 11>::BSize == 3);
  static_assert(XE4_TMM<float, T, T, float, 12>::BSize == 3);
  // N=13..16 → 4
  static_assert(XE4_TMM<float, T, T, float, 13>::BSize == 4);
  static_assert(XE4_TMM<float, T, T, float, 14>::BSize == 4);
  static_assert(XE4_TMM<float, T, T, float, 15>::BSize == 4);
  static_assert(XE4_TMM<float, T, T, float, 16>::BSize == 4);
  // N=17..20 → 5
  static_assert(XE4_TMM<float, T, T, float, 17>::BSize == 5);
  static_assert(XE4_TMM<float, T, T, float, 18>::BSize == 5);
  static_assert(XE4_TMM<float, T, T, float, 19>::BSize == 5);
  static_assert(XE4_TMM<float, T, T, float, 20>::BSize == 5);
  // N=21..24 → 6
  static_assert(XE4_TMM<float, T, T, float, 21>::BSize == 6);
  static_assert(XE4_TMM<float, T, T, float, 22>::BSize == 6);
  static_assert(XE4_TMM<float, T, T, float, 23>::BSize == 6);
  static_assert(XE4_TMM<float, T, T, float, 24>::BSize == 6);
  // N=25..28 → 7
  static_assert(XE4_TMM<float, T, T, float, 25>::BSize == 7);
  static_assert(XE4_TMM<float, T, T, float, 26>::BSize == 7);
  static_assert(XE4_TMM<float, T, T, float, 27>::BSize == 7);
  static_assert(XE4_TMM<float, T, T, float, 28>::BSize == 7);
  // N=29..32 → 8
  static_assert(XE4_TMM<float, T, T, float, 29>::BSize == 8);
  static_assert(XE4_TMM<float, T, T, float, 30>::BSize == 8);
  static_assert(XE4_TMM<float, T, T, float, 31>::BSize == 8);
  static_assert(XE4_TMM<float, T, T, float, 32>::BSize == 8);
  SUCCEED();
}

// k16,16: f16/f16 → K=16, B 16-bit  (BSize = ceil(N/4), same formula as k8,32)
TEST(XE4TMMFragmentSizes, BSize_k16_16bit) {
  using T = sycl::half;
  // N=1..4 → 1
  static_assert(XE4_TMM<float, T, T, float,  1>::BSize == 1);
  static_assert(XE4_TMM<float, T, T, float,  2>::BSize == 1);
  static_assert(XE4_TMM<float, T, T, float,  3>::BSize == 1);
  static_assert(XE4_TMM<float, T, T, float,  4>::BSize == 1);
  // N=5..8 → 2
  static_assert(XE4_TMM<float, T, T, float,  5>::BSize == 2);
  static_assert(XE4_TMM<float, T, T, float,  6>::BSize == 2);
  static_assert(XE4_TMM<float, T, T, float,  7>::BSize == 2);
  static_assert(XE4_TMM<float, T, T, float,  8>::BSize == 2);
  // N=9..12 → 3
  static_assert(XE4_TMM<float, T, T, float,  9>::BSize == 3);
  static_assert(XE4_TMM<float, T, T, float, 10>::BSize == 3);
  static_assert(XE4_TMM<float, T, T, float, 11>::BSize == 3);
  static_assert(XE4_TMM<float, T, T, float, 12>::BSize == 3);
  // N=13..16 → 4
  static_assert(XE4_TMM<float, T, T, float, 13>::BSize == 4);
  static_assert(XE4_TMM<float, T, T, float, 14>::BSize == 4);
  static_assert(XE4_TMM<float, T, T, float, 15>::BSize == 4);
  static_assert(XE4_TMM<float, T, T, float, 16>::BSize == 4);
  // N=17..20 → 5
  static_assert(XE4_TMM<float, T, T, float, 17>::BSize == 5);
  static_assert(XE4_TMM<float, T, T, float, 18>::BSize == 5);
  static_assert(XE4_TMM<float, T, T, float, 19>::BSize == 5);
  static_assert(XE4_TMM<float, T, T, float, 20>::BSize == 5);
  // N=21..24 → 6
  static_assert(XE4_TMM<float, T, T, float, 21>::BSize == 6);
  static_assert(XE4_TMM<float, T, T, float, 22>::BSize == 6);
  static_assert(XE4_TMM<float, T, T, float, 23>::BSize == 6);
  static_assert(XE4_TMM<float, T, T, float, 24>::BSize == 6);
  // N=25..28 → 7
  static_assert(XE4_TMM<float, T, T, float, 25>::BSize == 7);
  static_assert(XE4_TMM<float, T, T, float, 26>::BSize == 7);
  static_assert(XE4_TMM<float, T, T, float, 27>::BSize == 7);
  static_assert(XE4_TMM<float, T, T, float, 28>::BSize == 7);
  // N=29..32 → 8
  static_assert(XE4_TMM<float, T, T, float, 29>::BSize == 8);
  static_assert(XE4_TMM<float, T, T, float, 30>::BSize == 8);
  static_assert(XE4_TMM<float, T, T, float, 31>::BSize == 8);
  static_assert(XE4_TMM<float, T, T, float, 32>::BSize == 8);
  SUCCEED();
}

// k32,8:  e5m2/e5m2 → K=32, B 8-bit  (BSize = ceil(N/4), same formula as k8,32)
TEST(XE4TMMFragmentSizes, BSize_k32_8bit) {
  using T = cute::float_e5m2_t;
  // N=1..4 → 1
  static_assert(XE4_TMM<float, T, T, float,  1>::BSize == 1);
  static_assert(XE4_TMM<float, T, T, float,  2>::BSize == 1);
  static_assert(XE4_TMM<float, T, T, float,  3>::BSize == 1);
  static_assert(XE4_TMM<float, T, T, float,  4>::BSize == 1);
  // N=5..8 → 2
  static_assert(XE4_TMM<float, T, T, float,  5>::BSize == 2);
  static_assert(XE4_TMM<float, T, T, float,  6>::BSize == 2);
  static_assert(XE4_TMM<float, T, T, float,  7>::BSize == 2);
  static_assert(XE4_TMM<float, T, T, float,  8>::BSize == 2);
  // N=9..12 → 3
  static_assert(XE4_TMM<float, T, T, float,  9>::BSize == 3);
  static_assert(XE4_TMM<float, T, T, float, 10>::BSize == 3);
  static_assert(XE4_TMM<float, T, T, float, 11>::BSize == 3);
  static_assert(XE4_TMM<float, T, T, float, 12>::BSize == 3);
  // N=13..16 → 4
  static_assert(XE4_TMM<float, T, T, float, 13>::BSize == 4);
  static_assert(XE4_TMM<float, T, T, float, 14>::BSize == 4);
  static_assert(XE4_TMM<float, T, T, float, 15>::BSize == 4);
  static_assert(XE4_TMM<float, T, T, float, 16>::BSize == 4);
  // N=17..20 → 5
  static_assert(XE4_TMM<float, T, T, float, 17>::BSize == 5);
  static_assert(XE4_TMM<float, T, T, float, 18>::BSize == 5);
  static_assert(XE4_TMM<float, T, T, float, 19>::BSize == 5);
  static_assert(XE4_TMM<float, T, T, float, 20>::BSize == 5);
  // N=21..24 → 6
  static_assert(XE4_TMM<float, T, T, float, 21>::BSize == 6);
  static_assert(XE4_TMM<float, T, T, float, 22>::BSize == 6);
  static_assert(XE4_TMM<float, T, T, float, 23>::BSize == 6);
  static_assert(XE4_TMM<float, T, T, float, 24>::BSize == 6);
  // N=25..28 → 7
  static_assert(XE4_TMM<float, T, T, float, 25>::BSize == 7);
  static_assert(XE4_TMM<float, T, T, float, 26>::BSize == 7);
  static_assert(XE4_TMM<float, T, T, float, 27>::BSize == 7);
  static_assert(XE4_TMM<float, T, T, float, 28>::BSize == 7);
  // N=29..32 → 8
  static_assert(XE4_TMM<float, T, T, float, 29>::BSize == 8);
  static_assert(XE4_TMM<float, T, T, float, 30>::BSize == 8);
  static_assert(XE4_TMM<float, T, T, float, 31>::BSize == 8);
  static_assert(XE4_TMM<float, T, T, float, 32>::BSize == 8);
  SUCCEED();
}

// k16,8:  f16 major / e5m2 minor → K=16, B 8-bit  (BSize = ceil(N/8))
TEST(XE4TMMFragmentSizes, BSize_k16_8bit) {
  // N=1..8 → 1
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float,  1>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float,  2>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float,  3>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float,  4>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float,  5>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float,  6>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float,  7>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float,  8>::BSize == 1);
  // N=9..16 → 2
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float,  9>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 10>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 11>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 12>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 13>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 14>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 15>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 16>::BSize == 2);
  // N=17..24 → 3
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 17>::BSize == 3);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 18>::BSize == 3);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 19>::BSize == 3);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 20>::BSize == 3);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 21>::BSize == 3);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 22>::BSize == 3);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 23>::BSize == 3);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 24>::BSize == 3);
  // N=25..32 → 4
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 25>::BSize == 4);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 26>::BSize == 4);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 27>::BSize == 4);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 28>::BSize == 4);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 29>::BSize == 4);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 30>::BSize == 4);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 31>::BSize == 4);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e5m2_t, float, 32>::BSize == 4);
  SUCCEED();
}

// k32,4:  e5m2 major / e2m1 minor → K=32, B 4-bit  (BSize = ceil(N/8), same as k16,8)
TEST(XE4TMMFragmentSizes, BSize_k32_4bit) {
  // N=1..8 → 1
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float,  1>::BSize == 1);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float,  2>::BSize == 1);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float,  3>::BSize == 1);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float,  4>::BSize == 1);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float,  5>::BSize == 1);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float,  6>::BSize == 1);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float,  7>::BSize == 1);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float,  8>::BSize == 1);
  // N=9..16 → 2
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float,  9>::BSize == 2);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 10>::BSize == 2);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 11>::BSize == 2);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 12>::BSize == 2);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 13>::BSize == 2);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 14>::BSize == 2);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 15>::BSize == 2);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 16>::BSize == 2);
  // N=17..24 → 3
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 17>::BSize == 3);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 18>::BSize == 3);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 19>::BSize == 3);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 20>::BSize == 3);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 21>::BSize == 3);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 22>::BSize == 3);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 23>::BSize == 3);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 24>::BSize == 3);
  // N=25..32 → 4
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 25>::BSize == 4);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 26>::BSize == 4);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 27>::BSize == 4);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 28>::BSize == 4);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 29>::BSize == 4);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 30>::BSize == 4);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 31>::BSize == 4);
  static_assert(XE4_TMM<float, cute::float_e5m2_t, cute::float_e2m1_t, float, 32>::BSize == 4);
  SUCCEED();
}

// k16,4:  f16 major / e2m1 minor → K=16, B 4-bit  (BSize = ceil(N/16))
TEST(XE4TMMFragmentSizes, BSize_k16_4bit) {
  // N=1..16 → 1
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float,  1>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float,  2>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float,  3>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float,  4>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float,  5>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float,  6>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float,  7>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float,  8>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float,  9>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 10>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 11>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 12>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 13>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 14>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 15>::BSize == 1);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 16>::BSize == 1);
  // N=17..32 → 2
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 17>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 18>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 19>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 20>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 21>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 22>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 23>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 24>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 25>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 26>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 27>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 28>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 29>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 30>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 31>::BSize == 2);
  static_assert(XE4_TMM<float, sycl::half, cute::float_e2m1_t, float, 32>::BSize == 2);
  SUCCEED();
}

// Regression: BSize must be >= 1 for all N=1 cases (previously BSize could be 0
// when N*K < kSubGroupSize due to integer truncation in the old formula).
TEST(XE4TMMFragmentSizes, BSize_never_zero) {
  static_assert(XE4_TMM<float, cutlass::tfloat32_t,                cutlass::tfloat32_t,                float, 1>::BSize >= 1);
  static_assert(XE4_TMM<float, sycl::half,                         sycl::half,                         float, 1>::BSize >= 1);
  static_assert(XE4_TMM<float, sycl::ext::oneapi::bfloat16,        sycl::ext::oneapi::bfloat16,        float, 1>::BSize >= 1);
  static_assert(XE4_TMM<float, cute::float_e5m2_t,                 cute::float_e5m2_t,                 float, 1>::BSize >= 1);
  static_assert(XE4_TMM<float, cute::float_e2m1_t,                 cute::float_e2m1_t,                 float, 1>::BSize >= 1);
  static_assert(XE4_TMM<float, sycl::half,                         cute::float_e5m2_t,                 float, 1>::BSize >= 1);
  static_assert(XE4_TMM<float, sycl::half,                         cute::float_e2m1_t,                 float, 1>::BSize >= 1);
  static_assert(XE4_TMM<float, cute::float_e5m2_t,                 cute::float_e2m1_t,                 float, 1>::BSize >= 1);
  SUCCEED();
}

// -- Matrix D/C fragment sizes (DSize = CSize = ceil(N * bits(D/C) / 32)) --
// PISA spec D/C-fragment table:
//
// N  | 32-bit (float) | 16-bit (f16/bf16)
// ---+----------------+------------------
//  1 |  1             |  1
//  2 |  2             |  1
//  3 |  3             |  2
//  4 |  4             |  2
//  8 |  8             |  4
// 16 | 16             |  8
// 32 | 32             | 16

TEST(XE4TMMFragmentSizes, DSize_32bit) {
  static_assert(XE4_TMM<float, sycl::half, sycl::half, float,  1>::DSize ==  1);
  static_assert(XE4_TMM<float, sycl::half, sycl::half, float,  2>::DSize ==  2);
  static_assert(XE4_TMM<float, sycl::half, sycl::half, float,  3>::DSize ==  3);
  static_assert(XE4_TMM<float, sycl::half, sycl::half, float,  4>::DSize ==  4);
  static_assert(XE4_TMM<float, sycl::half, sycl::half, float,  8>::DSize ==  8);
  static_assert(XE4_TMM<float, sycl::half, sycl::half, float, 16>::DSize == 16);
  static_assert(XE4_TMM<float, sycl::half, sycl::half, float, 32>::DSize == 32);
  SUCCEED();
}

TEST(XE4TMMFragmentSizes, DSize_16bit_f16) {
  static_assert(XE4_TMM<sycl::half, sycl::half, sycl::half, sycl::half,  1>::DSize ==  1);
  static_assert(XE4_TMM<sycl::half, sycl::half, sycl::half, sycl::half,  2>::DSize ==  1);
  static_assert(XE4_TMM<sycl::half, sycl::half, sycl::half, sycl::half,  3>::DSize ==  2);
  static_assert(XE4_TMM<sycl::half, sycl::half, sycl::half, sycl::half,  4>::DSize ==  2);
  static_assert(XE4_TMM<sycl::half, sycl::half, sycl::half, sycl::half,  8>::DSize ==  4);
  static_assert(XE4_TMM<sycl::half, sycl::half, sycl::half, sycl::half, 16>::DSize ==  8);
  static_assert(XE4_TMM<sycl::half, sycl::half, sycl::half, sycl::half, 32>::DSize == 16);
  SUCCEED();
}

TEST(XE4TMMFragmentSizes, DSize_16bit_bf16) {
  using bf16 = sycl::ext::oneapi::bfloat16;
  static_assert(XE4_TMM<bf16, bf16, bf16, bf16,  1>::DSize ==  1);
  static_assert(XE4_TMM<bf16, bf16, bf16, bf16,  2>::DSize ==  1);
  static_assert(XE4_TMM<bf16, bf16, bf16, bf16,  3>::DSize ==  2);
  static_assert(XE4_TMM<bf16, bf16, bf16, bf16,  4>::DSize ==  2);
  static_assert(XE4_TMM<bf16, bf16, bf16, bf16,  8>::DSize ==  4);
  static_assert(XE4_TMM<bf16, bf16, bf16, bf16, 16>::DSize ==  8);
  static_assert(XE4_TMM<bf16, bf16, bf16, bf16, 32>::DSize == 16);
  SUCCEED();
}
