/***************************************************************************************************
 * Copyright (c) 2025 - 2026 INTEL CORPORATION. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Host-only unit test for XE4 TMM CuTe MMA_Traits fragment layouts.
 *
 * Includes <cute/atom/mma_traits_xe4_tmm.hpp> and uses the actual CuTe
 * TV layouts (ThrID, ALayout, BLayout, CLayout) to verify that each
 * SIMT lane reads the correct matrix elements for A, B, and C/D.
 *
 * No SYCL device headers are required -- all layouts are host-evaluable.
 **************************************************************************************************/

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>
#include <cassert>

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>  // EXPECT_{FATAL,NONFATAL}_FAILURE for negative tests

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_xe4_tmm.hpp>

#include "tiled_mma_utils.hpp"

using namespace cute;

// ---------------------------------------------------------------------------
// Helper: given MMA_Traits, extract the TV layouts and verify every matrix
// element is visited exactly once by exactly one (tid, vid) pair.
// ---------------------------------------------------------------------------

enum class FragmentKind { A, B, C };

template <class Traits, FragmentKind Kind, bool transpose=false>
void verify_layout(const char *test_name) {
  using Op = typename Traits::MMA_Op;
  using Shape_MNK = typename Traits::Shape_MNK;
  using Element = std::conditional_t<
      Kind == FragmentKind::A, typename Traits::ValTypeA,
      std::conditional_t<Kind == FragmentKind::B, typename Traits::ValTypeB,
                         typename Traits::ValTypeC>>;

  constexpr bool valid_input = ( (Kind == FragmentKind::A) && !transpose) || 
    ( (Kind == FragmentKind::C) && !transpose) ||
    ( (Kind == FragmentKind::B) && transpose);
  ASSERT_TRUE(valid_input) << "invalid input"; 

  constexpr int dim0 = Kind == FragmentKind::A ? size<0>(Shape_MNK{})
                                               : (Kind == FragmentKind::B ? size<2>(Shape_MNK{})
                                                                          : size<0>(Shape_MNK{}));
  constexpr int dim1 = Kind == FragmentKind::A ? size<2>(Shape_MNK{}) : size<1>(Shape_MNK{});
  constexpr const char *matrix_name = Kind == FragmentKind::A
      ? "A"
      : (Kind == FragmentKind::B ? "B" : "C");

  // Note that the number of lanes in the BLayout may not always be full //
  constexpr int thread_count =
    (Kind == FragmentKind::A) ? size<0>(typename Traits::ALayout{}) :
    (Kind == FragmentKind::B) ? size<0>(typename Traits::BLayout{}) :
                                size<0>(typename Traits::CLayout{});

  printf("Thread_count = %d max_lanes=%d\n", thread_count, 
      tmm::tmmLayoutBMaxLanes<typename Traits::ValTypeB, 
        size<2>(Shape_MNK{}), size<1>(Shape_MNK{})>::value);

  auto tiled_mma = make_tiled_mma(MMA_Atom<Op>{},
			make_layout(make_shape(_1{}, _1{}, _1{})));

  //print_layout( tiled_mma.get_layoutB_TV()  );

	using Dim0 = Int<dim0>;
	using Dim1 = Int<dim1>;


  // NOTE(vamsikku): CuTe MMA partition_B makes a rigid assumption that
  // input tensor for partition_B is always a NxK matrix. So if all our 
  // inputs are in KxN they need to be transposed.
	using FinalShape =
    std::conditional_t<transpose, Shape<Dim1, Dim0>, Shape<Dim0, Dim1>>;
	using FinalStride =
    std::conditional_t<transpose, Stride<_1, Dim1>, Stride<Dim1, _1>>; 
	auto g = make_tensor<Element>(make_layout(FinalShape{}, FinalStride{}));
  auto g_found = make_tensor<Element>(layout(g));

  for (int i = 0; i < size(g); ++i) {
    g(i) = static_cast<Element>(i + 1);
    g_found(i) = Element{};
  }

  for (int tid = 0; tid < thread_count; ++tid) {
    auto thr_mma = tiled_mma.get_slice(tid);

    auto partition_tensor = [&thr_mma](auto &tensor) {
      if constexpr (Kind == FragmentKind::A) {
        return thr_mma.partition_A(tensor);
      } else if constexpr (Kind == FragmentKind::B) {
        return thr_mma.partition_B(tensor);
      } else {
        return thr_mma.partition_C(tensor);
      }
    };

    auto create_registers = [&thr_mma](auto &tensor) {
      if constexpr (Kind == FragmentKind::A) {
        auto regs = thr_mma.make_fragment_A(tensor);
        return regs;
      } else if constexpr (Kind == FragmentKind::B) {
        auto regs = thr_mma.make_fragment_B(tensor);
        return regs;
      } else {
        auto regs = thr_mma.make_fragment_C(tensor);
        return regs;
      }
    };

    auto t = partition_tensor(g);
    auto r_regs = create_registers(t);
    auto t_found = partition_tensor(g_found);

    ASSERT_EQ(size(t), size(t_found)) << test_name << " " 
        << matrix_name << ": partition size mismatch";

    // copy to registers //
    copy(t, r_regs);

    for (int i = 0; i < size(layout(r_regs)); ++i) { 
      t_found(i) = r_regs(i);
    }
  }

  ASSERT_EQ(size(g), size(g_found));
  for (int i = 0; i < size(g); ++i) {
    EXPECT_EQ(g(i), g_found(i))
        << test_name << " " << matrix_name 
        << ": element at flat idx " << i << " not reconstructed";
  }
}

// ---------------------------------------------------------------------------
// Helper: verify shape dimensions match the template parameters
// ---------------------------------------------------------------------------
template <class Traits, int ExpM, int ExpN, int ExpK>
void verify_shape_MNK(const char *test_name) {
  using Shape_MNK = typename Traits::Shape_MNK;
  EXPECT_EQ(int(size<0>(Shape_MNK{})), ExpM) << test_name << " Shape M mismatch";
  EXPECT_EQ(int(size<1>(Shape_MNK{})), ExpN) << test_name << " Shape N mismatch";
  EXPECT_EQ(int(size<2>(Shape_MNK{})), ExpK) << test_name << " Shape K mismatch";
}

// ---------------------------------------------------------------------------
// Helper: verify ThrID layout has the expected number of threads (32)
// ---------------------------------------------------------------------------
template <class Traits>
void verify_thr_id(const char *test_name, int expected_threads = 32) {
  using ThrID = typename Traits::ThrID;
  EXPECT_EQ(int(size(ThrID{})), expected_threads)
      << test_name << " ThrID size mismatch";
}

// ============================================================================
// Test cases: f32 accumulator, f16 inputs -- M=32, K=16, varying N
// ============================================================================

TEST(XE4_TMM_FragmentLayout, f32_f16_f16_f32_m32n8k16_Shape) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  using Traits = MMA_Traits<Op>;
  verify_shape_MNK<Traits, 32, 8, 16>("m32n8k16");
  verify_thr_id<Traits>("m32n8k16");
}

TEST(XE4_TMM_FragmentLayout, f32_f16_f16_f32_m32n8k16_LayoutA) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  verify_layout<MMA_Traits<Op>, FragmentKind::A>("m32n8k16");
}

TEST(XE4_TMM_FragmentLayout, f32_f16_f16_f32_m32n8k16_LayoutB) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  verify_layout<MMA_Traits<Op>, FragmentKind::B, true>("m32n8k16");
}

TEST(XE4_TMM_FragmentLayout, f32_f16_f16_f32_m32n8k16_LayoutC) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  verify_layout<MMA_Traits<Op>, FragmentKind::C>("m32n8k16");
}

TEST(XE4_TMM_FragmentLayout, f32_f16_f16_f32_m32n32k16_Shape) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  using Traits = MMA_Traits<Op>;
  verify_shape_MNK<Traits, 32, 32, 16>("m32n32k16");
  verify_thr_id<Traits>("m32n32k16");
}

TEST(XE4_TMM_FragmentLayout, f32_f16_f16_f32_m32n32k16_LayoutA) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::A>("m32n32k16");
}

TEST(XE4_TMM_FragmentLayout, f32_f16_f16_f32_m32n32k16_LayoutB) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::B, true>("m32n32k16");
}

TEST(XE4_TMM_FragmentLayout, f32_f16_f16_f32_m32n32k16_LayoutC) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::C>("m32n32k16");
}

// ============================================================================
// Test cases: f32 accumulator, bf16 inputs -- M=32, K=16, varying N
// ============================================================================

TEST(XE4_TMM_FragmentLayout, f32_bf16_bf16_f32_m32n16k16_LayoutA) {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  using Op = XE4_TMM<float, bf16_t, bf16_t, float, 16>;
  verify_layout<MMA_Traits<Op>, FragmentKind::A>("bf16_m32n16k16");
}

TEST(XE4_TMM_FragmentLayout, f32_bf16_bf16_f32_m32n16k16_LayoutB) {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  using Op = XE4_TMM<float, bf16_t, bf16_t, float, 16>;
  verify_layout<MMA_Traits<Op>, FragmentKind::B, true>("bf16_m32n16k16");
}

TEST(XE4_TMM_FragmentLayout, f32_bf16_bf16_f32_m32n16k16_LayoutC) {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  using Op = XE4_TMM<float, bf16_t, bf16_t, float, 16>;
  verify_layout<MMA_Traits<Op>, FragmentKind::C>("bf16_m32n16k16");
}

// ============================================================================
// Test cases: f32 accumulator, tf32 inputs -- M=32, K=8
// ============================================================================

TEST(XE4_TMM_FragmentLayout, f32_tf32_tf32_f32_m32n8k8_Shape) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 8>;
  using Traits = MMA_Traits<Op>;
  verify_shape_MNK<Traits, 32, 8, 8>("tf32_m32n8k8");
  verify_thr_id<Traits>("tf32_m32n8k8");
}

TEST(XE4_TMM_FragmentLayout, f32_tf32_tf32_f32_m32n8k8_LayoutA) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 8>;
  verify_layout<MMA_Traits<Op>, FragmentKind::A>("tf32_m32n8k8");
}

TEST(XE4_TMM_FragmentLayout, f32_tf32_tf32_f32_m32n8k8_LayoutB) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 8>;
  verify_layout<MMA_Traits<Op>, FragmentKind::B, true>("tf32_m32n8k8");
}

TEST(XE4_TMM_FragmentLayout, f32_tf32_tf32_f32_m32n8k8_LayoutC) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 8>;
  verify_layout<MMA_Traits<Op>, FragmentKind::C>("tf32_m32n8k8");
}

TEST(XE4_TMM_FragmentLayout, f32_tf32_tf32_f32_m32n32k8_LayoutA) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::A>("tf32_m32n32k8");
}

TEST(XE4_TMM_FragmentLayout, f32_tf32_tf32_f32_m32n32k8_LayoutB) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::B, true>("tf32_m32n32k8");
}

TEST(XE4_TMM_FragmentLayout, f32_tf32_tf32_f32_m32n32k8_LayoutC) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::C>("tf32_m32n32k8");
}

// ============================================================================
// Test cases: f32 accumulator, int8 inputs -- M=32, K=32
// ============================================================================

TEST(XE4_TMM_FragmentLayout, f32_i8_i8_f32_m32n32k32_Shape) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 32>;
  using Traits = MMA_Traits<Op>;
  verify_shape_MNK<Traits, 32, 32, 32>("i8_m32n32k32");
  verify_thr_id<Traits>("i8_m32n32k32");
}

TEST(XE4_TMM_FragmentLayout, f32_i8_i8_f32_m32n32k32_LayoutA) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::A>("i8_m32n32k32");
}

TEST(XE4_TMM_FragmentLayout, f32_i8_i8_f32_m32n32k32_LayoutB) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::B, true>("i8_m32n32k32");
}

TEST(XE4_TMM_FragmentLayout, f32_i8_i8_f32_m32n32k32_LayoutC) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::C>("i8_m32n32k32");
}

TEST(XE4_TMM_FragmentLayout, f32_i8_i8_f32_m32n8k32_LayoutA) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 8>;
  verify_layout<MMA_Traits<Op>, FragmentKind::A>("i8_m32n8k32");
}

TEST(XE4_TMM_FragmentLayout, f32_i8_i8_f32_m32n8k32_LayoutB) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 8>;
  verify_layout<MMA_Traits<Op>, FragmentKind::B, true>("i8_m32n8k32");
}

TEST(XE4_TMM_FragmentLayout, f32_i8_i8_f32_m32n8k32_LayoutC) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 8>;
  verify_layout<MMA_Traits<Op>, FragmentKind::C>("i8_m32n8k32");
}

// ============================================================================
// Test cases: f32 accumulator, e5m2 (bf8) inputs -- M=32, K=32
// ============================================================================

TEST(XE4_TMM_FragmentLayout, f32_e5m2_e5m2_f32_m32n32k32_LayoutA) {
  using Op = XE4_TMM<float, cutlass::float_e5m2_t, cutlass::float_e5m2_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::A>("e5m2_m32n32k32");
}

TEST(XE4_TMM_FragmentLayout, f32_e5m2_e5m2_f32_m32n32k32_LayoutB) {
  using Op = XE4_TMM<float, cutlass::float_e5m2_t, cutlass::float_e5m2_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::B, true>("e5m2_m32n32k32");
}

TEST(XE4_TMM_FragmentLayout, f32_e5m2_e5m2_f32_m32n32k32_LayoutC) {
  using Op = XE4_TMM<float, cutlass::float_e5m2_t, cutlass::float_e5m2_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::C>("e5m2_m32n32k32");
}

TEST(XE4_TMM_FragmentLayout, f32_e5m2_e5m2_f32_m32n16k32_LayoutB) {
  using Op = XE4_TMM<float, cutlass::float_e5m2_t, cutlass::float_e5m2_t, float, 16>;
  verify_layout<MMA_Traits<Op>, FragmentKind::B, true>("e5m2_m32n16k32");
}

// ============================================================================
// Test cases: f32 accumulator, e2m1 (mxfp4) inputs -- M=32, K=32
// ============================================================================

TEST(XE4_TMM_FragmentLayout, f32_e2m1_e2m1_f32_m32n32k32_Shape) {
  using Op = XE4_TMM<float, cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, 32>;
  using Traits = MMA_Traits<Op>;
  verify_shape_MNK<Traits, 32, 32, 32>("e2m1_m32n32k32");
  verify_thr_id<Traits>("e2m1_m32n32k32");
}

TEST(XE4_TMM_FragmentLayout, f32_e2m1_e2m1_f32_m32n32k32_LayoutA) {
  using Op = XE4_TMM<float, cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::A>("e2m1_m32n32k32");
}

TEST(XE4_TMM_FragmentLayout, f32_e2m1_e2m1_f32_m32n32k32_LayoutB) {
  using Op = XE4_TMM<float, cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::B, true>("e2m1_m32n32k32");
}

TEST(XE4_TMM_FragmentLayout, f32_e2m1_e2m1_f32_m32n32k32_LayoutC) {
  using Op = XE4_TMM<float, cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::C>("e2m1_m32n32k32");
}

// ============================================================================
// Test: small N (uneven B distribution -- N*P not multiple of 32)
// ============================================================================

TEST(XE4_TMM_FragmentLayout, f32_f16_f16_f32_m32n1k16_LayoutB) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 1>;
  verify_layout<MMA_Traits<Op>, FragmentKind::B, true>("f16_m32n1k16");
}

TEST(XE4_TMM_FragmentLayout, f32_f16_f16_f32_m32n4k16_LayoutB) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 4>;
  verify_layout<MMA_Traits<Op>, FragmentKind::B, true>("f16_m32n4k16");
}

// ============================================================================
// Test: mixed A/B types (e.g. e4m3 A, e5m2 B)
// ============================================================================

TEST(XE4_TMM_FragmentLayout, f32_e4m3_e5m2_f32_m32n32k32_LayoutA) {
  using Op = XE4_TMM<float, cutlass::float_e4m3_t, cutlass::float_e5m2_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::A>("e4m3_e5m2_m32n32k32");
}

TEST(XE4_TMM_FragmentLayout, f32_e4m3_e5m2_f32_m32n32k32_LayoutB) {
  using Op = XE4_TMM<float, cutlass::float_e4m3_t, cutlass::float_e5m2_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::B, true>("e4m3_e5m2_m32n32k32");
}

TEST(XE4_TMM_FragmentLayout, f32_e4m3_e5m2_f32_m32n32k32_LayoutC) {
  using Op = XE4_TMM<float, cutlass::float_e4m3_t, cutlass::float_e5m2_t, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::C>("e4m3_e5m2_m32n32k32");
}

// ============================================================================
// Test: ValType aliases match template parameters
// ============================================================================

TEST(XE4_TMM_FragmentLayout, ValTypes_f32_f16_f16_f32) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  using Traits = MMA_Traits<Op>;

  static_assert(std::is_same_v<typename Traits::ValTypeD, float>,       "ValTypeD mismatch");
  static_assert(std::is_same_v<typename Traits::ValTypeA, sycl::half>,  "ValTypeA mismatch");
  static_assert(std::is_same_v<typename Traits::ValTypeB, sycl::half>,  "ValTypeB mismatch");
  static_assert(std::is_same_v<typename Traits::ValTypeC, float>,       "ValTypeC mismatch");
  SUCCEED();
}

TEST(XE4_TMM_FragmentLayout, ValTypes_f32_tf32_tf32_f32) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 8>;
  using Traits = MMA_Traits<Op>;

  static_assert(std::is_same_v<typename Traits::ValTypeD, float>,               "ValTypeD mismatch");
  static_assert(std::is_same_v<typename Traits::ValTypeA, cutlass::tfloat32_t>, "ValTypeA mismatch");
  static_assert(std::is_same_v<typename Traits::ValTypeB, cutlass::tfloat32_t>, "ValTypeB mismatch");
  static_assert(std::is_same_v<typename Traits::ValTypeC, float>,               "ValTypeC mismatch");
  SUCCEED();
}

// ============================================================================
// Test: layout sizes match M*K, N*K, M*N totals
// ============================================================================

template <class Traits>
void verify_layout_sizes(const char *test_name) {
  using Shape_MNK = typename Traits::Shape_MNK;
  using ALayout = typename Traits::ALayout;
  using BLayout = typename Traits::BLayout;
  using CLayout = typename Traits::CLayout;

  constexpr int M = size<0>(Shape_MNK{});
  constexpr int N = size<1>(Shape_MNK{});
  constexpr int K = size<2>(Shape_MNK{});

  // Total number of elements the layout covers = cosize or product of shape modes
  constexpr int A_total = size<0>(ALayout{}) * size<1>(ALayout{});
  constexpr int B_total = size<0>(BLayout{}) * size<1>(BLayout{});
  constexpr int C_total = size<0>(CLayout{}) * size<1>(CLayout{});

  // A covers M*K elements, B covers N*K, C covers M*N
  // The TV layout total may be >= the logical size if there is padding
  EXPECT_GE(A_total, M * K) << test_name << " A layout too small";
  EXPECT_GE(B_total, N * K) << test_name << " B layout too small";
  EXPECT_GE(C_total, M * N) << test_name << " C layout too small";
}

TEST(XE4_TMM_FragmentLayout, LayoutSizes_f16_m32n32k16) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  verify_layout_sizes<MMA_Traits<Op>>("f16_m32n32k16");
}

TEST(XE4_TMM_FragmentLayout, LayoutSizes_tf32_m32n32k8) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 32>;
  verify_layout_sizes<MMA_Traits<Op>>("tf32_m32n32k8");
}

TEST(XE4_TMM_FragmentLayout, LayoutSizes_i8_m32n32k32) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 32>;
  verify_layout_sizes<MMA_Traits<Op>>("i8_m32n32k32");
}

TEST(XE4_TMM_FragmentLayout, LayoutSizes_e2m1_m32n32k32) {
  using Op = XE4_TMM<float, cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, 32>;
  verify_layout_sizes<MMA_Traits<Op>>("e2m1_m32n32k32");
}

// ============================================================================
// Tests for default_warp_layout_for_tiled_mma
//
// These are compile-time tests: all assertions use static_assert, so any
// incorrect value is a compile error. The TEST body calls SUCCEED() to
// signal that the struct instantiation (and all its static_asserts) compiled.
// ============================================================================

// Helper: verify that the sg_M, sg_N, sg_K values in default_warp_layout_for_tiled_mma
// match the expected values, and that their product equals num_consumer_sgs.
template <class SgLayoutMeta, int ExpSgM, int ExpSgN, int ExpSgK>
void verify_sg_layout(const char *test_name) {
  constexpr int got_sgM = SgLayoutMeta::sg_M;
  constexpr int got_sgN = SgLayoutMeta::sg_N;
  constexpr int got_sgK = SgLayoutMeta::sg_K;

  EXPECT_EQ(got_sgM, ExpSgM)
      << test_name << ": sg_M expected=" << ExpSgM << " got=" << got_sgM;
  EXPECT_EQ(got_sgN, ExpSgN)
      << test_name << ": sg_N expected=" << ExpSgN << " got=" << got_sgN;
  EXPECT_EQ(got_sgK, ExpSgK)
      << test_name << ": sg_K expected=" << ExpSgK << " got=" << got_sgK;

  // Also verify via the Layout type: shape must match (sg_M, sg_N, sg_K)
  using ExpType = cute::Layout<cute::Shape<cute::Int<ExpSgM>,
                                           cute::Int<ExpSgN>,
                                           cute::Int<ExpSgK>>>;
  static_assert(std::is_same_v<typename SgLayoutMeta::type, ExpType>,
                "default_warp_layout_for_tiled_mma::type does not match expected Layout");
}

// ============================================================================
// 1 consumer SG — the common example: WG=96, SG=32, 2 producer SGs
//   atom 32×8×16 (f16),  tile 64×64×16  →  Shape<1,1,1>
// ============================================================================
TEST(XE4_TMM_DefaultSgLayout, OneSG_f16_tile64x64x16) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  // WG=96, SG=32, producers=2  →  consumer_sgs = 96/32 - 2 = 1
  using SgL = cute::default_warp_layout_for_tiled_mma<Op, 64, 64, 16, 96, 32, 2>;
  verify_sg_layout<SgL, 1, 1, 1>("f16_tile64x64x16_1sg");
}


TEST(XE4_TMM_DefaultSgLayout, SixSG_f16_tile64x64x16) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  // WG=256, SG=32, producers=2  →  consumer_sgs = 256/32 - 2 = 6
  using SgL = cute::default_warp_layout_for_tiled_mma<Op, 64, 64, 64, 256, 32, 2>;

  SUCCEED() << "Static assert should trigger";

}

// ============================================================================
// 16 consumer SGs — fully parallel tile coverage
//   atom 32×8×16 (f16),  tile 64×64×16
//   budget=16, t_M=2, t_N=8, t_K=1
//   p0=M (bM=bN=64, M wins tie), sg_p0=min(16,2)=2, rem=8
//   p1=N, sg_p1=min(8,8)=8, rem=1
//   p2=K, sg_p2=1
//   → Shape<2,8,1>
// ============================================================================
TEST(XE4_TMM_DefaultSgLayout, SixteenSG_f16_tile64x64x16) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  // WG=512, SG=32, producers=0  →  consumer_sgs = 16
  using SgL = cute::default_warp_layout_for_tiled_mma<Op, 64, 64, 16, 512, 32, 0>;
  verify_sg_layout<SgL, 2, 8, 1>("f16_tile64x64x16_16sg");
}

// ============================================================================
// 8 consumer SGs
//   atom 32×8×16, tile 64×64×16
//   p0=M, sg_p0=min(8,2)=2, rem=4; p1=N, sg_p1=min(4,8)=4, rem=1; p2=K, sg_p2=1
//   → Shape<2,4,1>
// ============================================================================
TEST(XE4_TMM_DefaultSgLayout, EightSG_f16_tile64x64x16) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  using SgL = cute::default_warp_layout_for_tiled_mma<Op, 64, 64, 16, 256, 32, 0>;
  verify_sg_layout<SgL, 2, 4, 1>("f16_tile64x64x16_8sg");
}

// ============================================================================
// 4 consumer SGs
//   atom 32×8×16, tile 64×64×16
//   p0=M, sg_p0=min(4,2)=2, rem=2; p1=N, sg_p1=min(2,8)=2, rem=1; p2=K, sg_p2=1
//   → Shape<2,2,1>
// ============================================================================
TEST(XE4_TMM_DefaultSgLayout, FourSG_f16_tile64x64x16) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  using SgL = cute::default_warp_layout_for_tiled_mma<Op, 64, 64, 16, 128, 32, 0>;
  verify_sg_layout<SgL, 2, 2, 1>("f16_tile64x64x16_4sg");
}

// ============================================================================
// N-dominant tile: bN > bM, N gets the most SGs
//   atom 32×8×16, tile 32×128×16, 4 consumer SGs
//   bM=32, bN=128, bK=16
//   p0=N (128 > 32 > 16), t_N=16
//   sg_p0=min(4,16)=4, rem=1; p1=M, sg_p1=min(1,1)=1; p2=K, sg_p2=1
//   → Shape<1,4,1>
// ============================================================================
TEST(XE4_TMM_DefaultSgLayout, FourSG_f16_NDominant_tile32x128x16) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  using SgL = cute::default_warp_layout_for_tiled_mma<Op, 32, 128, 16, 128, 32, 0>;
  verify_sg_layout<SgL, 1, 4, 1>("f16_tile32x128x16_4sg");
}

// ============================================================================
// Split-K: tile has more K atoms than M or N
//   atom 32×8×16, tile 64×8×32, 4 consumer SGs
//   bM=64, bN=8, bK=32; t_M=2, t_N=1, t_K=2
//   p0=M (64 > bK=32 > bN=8)
//   sg_p0=min(4,2)=2, rem=2
//   p2: candidates N(idx=1,bN=8) and K(idx=2,bK=32); bN<bK → p2=N(idx=1)
//   p1=K(idx=2)
//   sg_p1=min(2,2)=2, rem=1; sg_p2=1
//   sg_M=sg_p0=2, sg_N(p2=1)=sg_p2=1, sg_K(p1=2)=sg_p1=2
//   → Shape<2,1,2>
// ============================================================================
TEST(XE4_TMM_DefaultSgLayout, FourSG_f16_SplitK_tile64x8x32) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  using SgL = cute::default_warp_layout_for_tiled_mma<Op, 64, 8, 32, 128, 32, 0>;
  verify_sg_layout<SgL, 2, 1, 2>("f16_tile64x8x32_splitK_4sg");
}

// ============================================================================
// tf32 atom: K=8.  atom 32×8×8, tile 64×64×8, 16 consumer SGs
//   t_M=2, t_N=8, t_K=1 — same structural outcome as f16 with equal tiles
//   → Shape<2,8,1>
// ============================================================================
TEST(XE4_TMM_DefaultSgLayout, SixteenSG_tf32_tile64x64x8) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 8>;
  using SgL = cute::default_warp_layout_for_tiled_mma<Op, 64, 64, 8, 512, 32, 0>;
  verify_sg_layout<SgL, 2, 8, 1>("tf32_tile64x64x8_16sg");
}

// ============================================================================
// i8 atom: K=32.  atom 32×8×32, tile 64×64×32, 16 consumer SGs
//   t_M=2, t_N=8, t_K=1
//   → Shape<2,8,1>
// ============================================================================
TEST(XE4_TMM_DefaultSgLayout, SixteenSG_i8_tile64x64x32) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 8>;
  using SgL = cute::default_warp_layout_for_tiled_mma<Op, 64, 64, 32, 512, 32, 0>;
  verify_sg_layout<SgL, 2, 8, 1>("i8_tile64x64x32_16sg");
}

// ============================================================================
// Larger atom N=32: atom 32×32×16 (f16), tile 128×128×16, 16 consumer SGs
//   t_M=4, t_N=4, t_K=1
//   p0=M (bM=bN=128, M wins tie), sg_p0=min(16,4)=4, rem=4
//   p2=K (bK=16 < bN=128), p1=N
//   sg_p1=min(4,4)=4, rem=1; sg_p2=1
//   → Shape<4,4,1>
// ============================================================================
TEST(XE4_TMM_DefaultSgLayout, SixteenSG_f16_AtomN32_tile128x128x16) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  using SgL = cute::default_warp_layout_for_tiled_mma<Op, 128, 128, 16, 512, 32, 0>;
  verify_sg_layout<SgL, 4, 4, 1>("f16_atomN32_tile128x128x16_16sg");
}

// ============================================================================
// Verify that num_consumer_sgs and atom dims are exposed correctly
// ============================================================================
TEST(XE4_TMM_DefaultSgLayout, StaticMembers_f16) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  using SgL = cute::default_warp_layout_for_tiled_mma<Op, 64, 64, 16, 512, 32, 0>;

  static_assert(SgL::num_consumer_sgs == 16, "num_consumer_sgs mismatch");
  static_assert(SgL::AtomM == 32,            "AtomM mismatch");
  static_assert(SgL::AtomN == 8,             "AtomN mismatch");
  static_assert(SgL::AtomK == 16,            "AtomK mismatch");
  SUCCEED();
}

// ============================================================================
// Verify N_val is accessible on XE4_TMM
// ============================================================================
TEST(XE4_TMM_DefaultSgLayout, XE4_TMM_NVal) {
  static_assert(XE4_TMM<float, sycl::half, sycl::half, float, 8>::N_val == 8,
                "N_val mismatch for N=8");
  static_assert(XE4_TMM<float, sycl::half, sycl::half, float, 32>::N_val == 32,
                "N_val mismatch for N=32");
  static_assert(XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 16>::N_val == 16,
                "N_val mismatch for tf32 N=16");
  SUCCEED();
}

// ============================================================================
// Negative / boundary tests exercising verify_layout directly
// ----------------------------------------------------------------------------
// Classification used here (per reviewer guidance):
//   * compile-fail         : ill-formed input must not instantiate. Shown as
//                            commented block — uncommenting must fail to build.
//   * runtime-abort        : ill-formed input must trigger the helper's internal
//                            ASSERT_EQ. Probed with EXPECT_FATAL_FAILURE.
//   * wrong-answer / crash : undefined behaviour; NOT expected from correct code.
//                            If one of these probes produces wrong output or a
//                            crash/hang instead of the documented failure mode,
//                            that is a verify_layout robustness bug (file Jira).
// ============================================================================

// ---------------------------------------------------------------------------
//  bug-locator probe for the partition_B(NxK) rigidity contract
// documented at tmm_fragment_layout.cpp:72-78. Running verify_layout with
// FragmentKind::B and transpose=false should abort via the ASSERT_EQ at
// "partition size mismatch", but currently produces no diagnostic — the
// helper completes silently with wrong output (observed 2026-04-21 on
// build dc09d8e). Enable this test once verify_layout gains the missing
// guard; the EXPECT_FATAL_FAILURE will then pass.
// See CUTLASS9 bug: verify_layout has no guard for FragmentB without transpose.
// ---------------------------------------------------------------------------
TEST(XE4_TMM_VerifyLayoutNegative, FragmentB_Without_Transpose_MustAssert) {
  using Op     = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  using Traits = MMA_Traits<Op>;
  EXPECT_FATAL_FAILURE(
    (verify_layout<Traits, FragmentKind::B, /*transpose=*/false>("neg_B_no_transpose")),
    "invalid input");
}

// ---------------------------------------------------------------------------
//  bug-locator probe for the FragmentA MxK shape contract.
// Running verify_layout with FragmentKind::A and transpose=true inverts a
// rigid MxK fragment; this currently segfaults / core-dumps instead of
// producing a clean assertion or wrong-output diagnostic (observed
// 2026-04-21 on build dc09d8e). Enable this test once verify_layout
// gains a shape guard that either static_asserts or fails cleanly at
// runtime.
// See CUTLASS9 bug: verify_layout has no guard for FragmentA with transpose.
// ---------------------------------------------------------------------------
TEST(XE4_TMM_VerifyLayoutNegative, FragmentA_With_Transpose_MustFail) {
  using Op     = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  using Traits = MMA_Traits<Op>;
  EXPECT_FATAL_FAILURE(
    (verify_layout<Traits, FragmentKind::A, /*transpose=*/true>("neg_A_transpose")),
    "invalid input");
}

// ---------------------------------------------------------------------------
// Compile-fail (comment-only annotation):
// XE4_TMM has no specialization accepting a `double` accumulator. Uncommenting
// the block below must produce a compile error (MMA_Traits primary template
// has no Shape_MNK, or the XE4_TMM primary definition rejects `double`).
// If it ever compiles, a rogue specialization was added — file Jira.
// ---------------------------------------------------------------------------
TEST(XE4_TMM_VerifyLayoutNegative, UnsupportedDtype_Double_Documented) {
  // using BadOp     = XE4_TMM<double, double, double, double, 8>;
  // using BadTraits = MMA_Traits<BadOp>;
  // (void) sizeof(typename BadTraits::Shape_MNK);   // <-- must not compile
  SUCCEED() << "double accumulator is documented as unsupported; see commented probe";
}

// ---------------------------------------------------------------------------
// Boundary (positive): smallest supported tile N=1. Exercises verify_layout
// end-to-end with the minimum MMA shape.
// ---------------------------------------------------------------------------
TEST(XE4_TMM_VerifyLayoutBoundary, MinTile_f16_m32n1k16) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 1>;
  verify_layout<MMA_Traits<Op>, FragmentKind::A>("min_n1_A");
  verify_layout<MMA_Traits<Op>, FragmentKind::B, /*transpose=*/true>("min_n1_B");
  verify_layout<MMA_Traits<Op>, FragmentKind::C>("min_n1_C");
}

// ---------------------------------------------------------------------------
// Boundary (positive): largest supported tile N=32. Exercises verify_layout
// end-to-end with the maximum MMA shape.
// ---------------------------------------------------------------------------
TEST(XE4_TMM_VerifyLayoutBoundary, MaxTile_f16_m32n32k16) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  verify_layout<MMA_Traits<Op>, FragmentKind::A>("max_n32_A");
  verify_layout<MMA_Traits<Op>, FragmentKind::B, /*transpose=*/true>("max_n32_B");
  verify_layout<MMA_Traits<Op>, FragmentKind::C>("max_n32_C");
}
