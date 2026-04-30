/***************************************************************************************************
 * Copyright (c) 2025 - 2026 INTEL CORPORATION. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Host-only unit test for XE4 TMM tiled copy using MMA_Traits TV layouts.
 *
 * For each (M,N,K) shape and data-type combination the test:
 *   1. Extracts the TV layout directly from MMA_Traits<XE4_TMM<...>>.
 *   2. Constructs a TiledCopy via make_tiled_copy_impl with a UniversalCopy
 *      atom -- no TiledMMA is created.
 *   3. For each of the threads defined by the TV layout:
 *        - partition_S partitions the source memory tensor
 *        - partition_fragment_D allocates a register fragment (destination)
 *        - cute::copy scatters source elements into the fragment
 *   4. Gathers fragments back into a destination tensor:
 *        - partition_fragment_S shapes the register fragment as source
 *        - partition_D partitions the destination memory tensor
 *        - cute::copy stores fragment elements into destination
 *   5. Verifies that the round-tripped destination matches the source
 *      exactly -- proving the TV layout is a correct bijection.
 *
 * No SYCL device headers are required -- all layouts are host-evaluable.
 **************************************************************************************************/

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>

#include "cutlass_unit_test.h"

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_traits_xe4_tmm.hpp>

using namespace cute;

// ---------------------------------------------------------------------------
// Matrix A: scatter/gather round-trip via MMA_Traits::ALayout
// ---------------------------------------------------------------------------

template <class MMAOp>
void tiled_copy_A_round_trip(const char* test_name) {
  using Traits    = MMA_Traits<MMAOp>;
  using Shape_MNK = typename Traits::Shape_MNK;

  constexpr int M = size<0>(Shape_MNK{});
  constexpr int K = size<2>(Shape_MNK{});
  constexpr int num_elems = M * K;

  // Source: col-major (M, K)
  std::vector<int> src_data(num_elems);
  std::iota(src_data.begin(), src_data.end(), 1);

  auto layout = make_layout(make_shape(Int<M>{}, Int<K>{}),
                             make_stride(Int<1>{}, Int<M>{}));
  auto src_tensor = make_tensor(src_data.data(), layout);

  // Destination: same layout, zeroed
  std::vector<int> dst_data(num_elems, 0);
  auto dst_tensor = make_tensor(dst_data.data(), layout);

  // TiledCopy directly from the trait's ALayout -- no TiledMMA needed
  auto copy_atom  = Copy_Atom<UniversalCopy<int>, int>{};
  auto tiled_copy = make_tiled_copy_impl(
      copy_atom,
      typename Traits::ALayout{},
      make_shape(Int<M>{}, Int<K>{}));

  constexpr int num_threads = decltype(size(tiled_copy))::value;
  std::vector<std::vector<int>> fragments(num_threads);

  // SCATTER: src_tensor -> fragments
  for (int tid = 0; tid < num_threads; ++tid) {
    auto thr_copy = tiled_copy.get_thread_slice(tid);
    auto tSrc     = thr_copy.partition_S(src_tensor);
    auto tFrg     = thr_copy.partition_fragment_D(src_tensor);
    copy(tiled_copy, tSrc, tFrg);
    fragments[tid].resize(size(tFrg));
    for (int i = 0; i < size(tFrg); ++i) fragments[tid][i] = tFrg(i);
  }

  // GATHER: fragments -> dst_tensor
  for (int tid = 0; tid < num_threads; ++tid) {
    auto thr_copy = tiled_copy.get_thread_slice(tid);
    auto tFrg     = thr_copy.partition_fragment_S(dst_tensor);
    auto tDst     = thr_copy.partition_D(dst_tensor);
    for (int i = 0; i < size(tFrg); ++i) tFrg(i) = fragments[tid][i];
    copy(tiled_copy, tFrg, tDst);
  }

  // VERIFY
  for (int i = 0; i < num_elems; ++i) {
    EXPECT_EQ(dst_data[i], src_data[i])
        << test_name << ": mismatch at flat index " << i
        << " (m=" << i % M << ", k=" << i / M << ")";
  }
}

// ---------------------------------------------------------------------------
// Matrix B: scatter/gather round-trip via MMA_Traits::BLayout
// ---------------------------------------------------------------------------

template <class MMAOp>
void tiled_copy_B_round_trip(const char* test_name) {
  using Traits    = MMA_Traits<MMAOp>;
  using Shape_MNK = typename Traits::Shape_MNK;

  constexpr int N = size<1>(Shape_MNK{});
  constexpr int K = size<2>(Shape_MNK{});
  constexpr int num_elems = N * K;

  std::vector<int> src_data(num_elems);
  std::iota(src_data.begin(), src_data.end(), 1);

  auto layout = make_layout(make_shape(Int<K>{}, Int<N>{}),
                             make_stride(Int<N>{}, Int<1>{}));
  auto src_tensor = make_tensor(src_data.data(), layout);

  std::vector<int> dst_data(num_elems, 0);
  auto dst_tensor = make_tensor(dst_data.data(), layout);

  auto copy_atom  = Copy_Atom<UniversalCopy<int>, int>{};
  auto tiled_copy = make_tiled_copy_impl(
      copy_atom,
      typename Traits::BLayout{},
      make_shape(Int<K>{}, Int<N>{}));

  constexpr int num_threads = decltype(size(tiled_copy))::value;
	printf("num_threads = %d \n", num_threads);
  std::vector<std::vector<int>> fragments(num_threads);

  for (int tid = 0; tid < num_threads; ++tid) {
    auto thr_copy = tiled_copy.get_thread_slice(tid);
    auto tSrc     = thr_copy.partition_S(src_tensor);
    auto tFrg     = thr_copy.partition_fragment_D(src_tensor);
    copy(tiled_copy, tSrc, tFrg);
    fragments[tid].resize(size(tFrg));
    for (int i = 0; i < size(tFrg); ++i) fragments[tid][i] = tFrg(i);
  }

  for (int tid = 0; tid < num_threads; ++tid) {
    auto thr_copy = tiled_copy.get_thread_slice(tid);
    auto tFrg     = thr_copy.partition_fragment_S(dst_tensor);
    auto tDst     = thr_copy.partition_D(dst_tensor);
    for (int i = 0; i < size(tFrg); ++i) tFrg(i) = fragments[tid][i];
    copy(tiled_copy, tFrg, tDst);
  }

  for (int i = 0; i < num_elems; ++i) {
    EXPECT_EQ(dst_data[i], src_data[i])
        << test_name << ": mismatch at flat index " << i
        << " (n=" << i % N << ", k=" << i / N << ")";
  }
}

// ---------------------------------------------------------------------------
// Matrix C/D: scatter/gather round-trip via MMA_Traits::CLayout
// ---------------------------------------------------------------------------

template <class MMAOp>
void tiled_copy_C_round_trip(const char* test_name) {
  using Traits    = MMA_Traits<MMAOp>;
  using Shape_MNK = typename Traits::Shape_MNK;

  constexpr int M = size<0>(Shape_MNK{});
  constexpr int N = size<1>(Shape_MNK{});
  constexpr int num_elems = M * N;

  std::vector<int> src_data(num_elems);
  std::iota(src_data.begin(), src_data.end(), 1);

  auto layout = make_layout(make_shape(Int<M>{}, Int<N>{}),
                             make_stride(Int<1>{}, Int<M>{}));
  auto src_tensor = make_tensor(src_data.data(), layout);

  std::vector<int> dst_data(num_elems, 0);
  auto dst_tensor = make_tensor(dst_data.data(), layout);

  auto copy_atom  = Copy_Atom<UniversalCopy<int>, int>{};
  auto tiled_copy = make_tiled_copy_impl(
      copy_atom,
      typename Traits::CLayout{},
      make_shape(Int<M>{}, Int<N>{}));

  constexpr int num_threads = decltype(size(tiled_copy))::value;
  std::vector<std::vector<int>> fragments(num_threads);

  for (int tid = 0; tid < num_threads; ++tid) {
    auto thr_copy = tiled_copy.get_thread_slice(tid);
    auto tSrc     = thr_copy.partition_S(src_tensor);
    auto tFrg     = thr_copy.partition_fragment_D(src_tensor);
    copy(tiled_copy, tSrc, tFrg);
    fragments[tid].resize(size(tFrg));
    for (int i = 0; i < size(tFrg); ++i) fragments[tid][i] = tFrg(i);
  }

  for (int tid = 0; tid < num_threads; ++tid) {
    auto thr_copy = tiled_copy.get_thread_slice(tid);
    auto tFrg     = thr_copy.partition_fragment_S(dst_tensor);
    auto tDst     = thr_copy.partition_D(dst_tensor);
    for (int i = 0; i < size(tFrg); ++i) tFrg(i) = fragments[tid][i];
    copy(tiled_copy, tFrg, tDst);
  }

  for (int i = 0; i < num_elems; ++i) {
    EXPECT_EQ(dst_data[i], src_data[i])
        << test_name << ": mismatch at flat index " << i
        << " (m=" << i % M << ", n=" << i / M << ")";
  }
}

// ---------------------------------------------------------------------------
// Thread-coverage verification: every element owned by exactly one thread.
// Uses the same scatter/gather pattern to confirm the union of all thread
// partitions covers the full matrix with no gaps or overlaps.
// ---------------------------------------------------------------------------

template <class MMAOp, char Matrix>
void verify_thread_coverage(const char* test_name) {
  using Traits    = MMA_Traits<MMAOp>;
  using Shape_MNK = typename Traits::Shape_MNK;

  constexpr int Dim0 = (Matrix == 'A') ? int(size<0>(Shape_MNK{})) :
                       (Matrix == 'B') ? int(size<2>(Shape_MNK{})) :
                                         int(size<1>(Shape_MNK{}));
  constexpr int Dim1 = (Matrix == 'A') ? int(size<2>(Shape_MNK{})) :
                       (Matrix == 'B') ? int(size<1>(Shape_MNK{})) :
                                         int(size<0>(Shape_MNK{}));
  constexpr int num_elems = Dim0 * Dim1;

  // Unique-value source so round-trip detects any mixup
  std::vector<int> src_data(num_elems);
  std::iota(src_data.begin(), src_data.end(), 1);

	// both row-major //
  auto layout = make_layout(make_shape(Int<Dim0>{}, Int<Dim1>{}),
                             make_stride(Int<Dim1>{}, Int<1>{}));
  auto src_tensor = make_tensor(src_data.data(), layout);

  // Destination filled with sentinel
  std::vector<int> dst_data(num_elems, -1);
  auto dst_tensor = make_tensor(dst_data.data(), layout);

  auto copy_atom = Copy_Atom<UniversalCopy<int>, int>{};

  auto tiled_copy = [&]() {
    if constexpr (Matrix == 'A') {
      return make_tiled_copy_impl(
          copy_atom, typename Traits::ALayout{},
          make_shape(Int<Dim0>{}, Int<Dim1>{}));
    } else if constexpr (Matrix == 'B') {
      return make_tiled_copy_impl(
          copy_atom, typename Traits::BLayout{},
          make_shape(Int<Dim0>{}, Int<Dim1>{}));
    } else {
      return make_tiled_copy_impl(
          copy_atom, typename Traits::CLayout{},
          make_shape(Int<Dim0>{}, Int<Dim1>{}));
    }
  }();

  constexpr int num_threads = decltype(size(tiled_copy))::value;

  // Single pass: scatter then gather per thread
  for (int tid = 0; tid < num_threads; ++tid) {
    auto thr_copy = tiled_copy.get_thread_slice(tid);

    // Scatter: src -> fragment
    auto tSrc = thr_copy.partition_S(src_tensor);
    auto tFrg = thr_copy.partition_fragment_D(src_tensor);
    copy(tiled_copy, tSrc, tFrg);

    // Gather: fragment -> dst
    auto tFrgS = thr_copy.partition_fragment_S(dst_tensor);
    auto tDst  = thr_copy.partition_D(dst_tensor);
    for (int i = 0; i < size(tFrgS); ++i) tFrgS(i) = tFrg(i);
    copy(tiled_copy, tFrgS, tDst);
  }

  // Every element must be written exactly once with the correct value
  for (int i = 0; i < num_elems; ++i) {
    EXPECT_NE(dst_data[i], -1)
        << test_name << ": element " << i << " not written by any thread";
    EXPECT_EQ(dst_data[i], src_data[i])
        << test_name << ": element " << i << " has wrong value "
        << dst_data[i] << ", expected " << src_data[i];
  }
}

// ============================================================================
// f32 accumulator, f16 inputs
// ============================================================================

TEST(XE4_TMM_TiledCopy, f16_m32n8k16_CopyA) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  tiled_copy_A_round_trip<Op>("f16_m32n8k16_A");
}

TEST(XE4_TMM_TiledCopy, f16_m32n8k16_CopyB) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  tiled_copy_B_round_trip<Op>("f16_m32n8k16_B");
}

TEST(XE4_TMM_TiledCopy, f16_m32n8k16_CopyC) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 8>;
  tiled_copy_C_round_trip<Op>("f16_m32n8k16_C");
}

TEST(XE4_TMM_TiledCopy, f16_m32n32k16_CopyA) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  tiled_copy_A_round_trip<Op>("f16_m32n32k16_A");
}

TEST(XE4_TMM_TiledCopy, f16_m32n32k16_CopyB) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  tiled_copy_B_round_trip<Op>("f16_m32n32k16_B");
}

TEST(XE4_TMM_TiledCopy, f16_m32n32k16_CopyC) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  tiled_copy_C_round_trip<Op>("f16_m32n32k16_C");
}

TEST(XE4_TMM_TiledCopy, f16_m32n32k16_CoverageA) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  verify_thread_coverage<Op, 'A'>("f16_m32n32k16_cov_A");
}

TEST(XE4_TMM_TiledCopy, f16_m32n32k16_CoverageB) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  verify_thread_coverage<Op, 'B'>("f16_m32n32k16_cov_B");
}

TEST(XE4_TMM_TiledCopy, f16_m32n32k16_CoverageC) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 32>;
  verify_thread_coverage<Op, 'C'>("f16_m32n32k16_cov_C");
}

// ============================================================================
// f32 accumulator, bf16 inputs
// ============================================================================

TEST(XE4_TMM_TiledCopy, bf16_m32n16k16_CopyA) {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  using Op = XE4_TMM<float, bf16_t, bf16_t, float, 16>;
  tiled_copy_A_round_trip<Op>("bf16_m32n16k16_A");
}

TEST(XE4_TMM_TiledCopy, bf16_m32n16k16_CopyB) {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  using Op = XE4_TMM<float, bf16_t, bf16_t, float, 16>;
  tiled_copy_B_round_trip<Op>("bf16_m32n16k16_B");
}

TEST(XE4_TMM_TiledCopy, bf16_m32n16k16_CopyC) {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  using Op = XE4_TMM<float, bf16_t, bf16_t, float, 16>;
  tiled_copy_C_round_trip<Op>("bf16_m32n16k16_C");
}

// ============================================================================
// f32 accumulator, tf32 inputs
// ============================================================================

TEST(XE4_TMM_TiledCopy, tf32_m32n8k8_CopyA) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 8>;
  tiled_copy_A_round_trip<Op>("tf32_m32n8k8_A");
}

TEST(XE4_TMM_TiledCopy, tf32_m32n8k8_CopyB) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 8>;
  tiled_copy_B_round_trip<Op>("tf32_m32n8k8_B");
}

TEST(XE4_TMM_TiledCopy, tf32_m32n8k8_CopyC) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 8>;
  tiled_copy_C_round_trip<Op>("tf32_m32n8k8_C");
}

TEST(XE4_TMM_TiledCopy, tf32_m32n32k8_CopyA) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 32>;
  tiled_copy_A_round_trip<Op>("tf32_m32n32k8_A");
}

TEST(XE4_TMM_TiledCopy, tf32_m32n32k8_CopyB) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 32>;
  tiled_copy_B_round_trip<Op>("tf32_m32n32k8_B");
}

TEST(XE4_TMM_TiledCopy, tf32_m32n32k8_CopyC) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 32>;
  tiled_copy_C_round_trip<Op>("tf32_m32n32k8_C");
}

TEST(XE4_TMM_TiledCopy, tf32_m32n32k8_CoverageA) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 32>;
  verify_thread_coverage<Op, 'A'>("tf32_m32n32k8_cov_A");
}

TEST(XE4_TMM_TiledCopy, tf32_m32n32k8_CoverageB) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 32>;
  verify_thread_coverage<Op, 'B'>("tf32_m32n32k8_cov_B");
}

TEST(XE4_TMM_TiledCopy, tf32_m32n32k8_CoverageC) {
  using Op = XE4_TMM<float, cutlass::tfloat32_t, cutlass::tfloat32_t, float, 32>;
  verify_thread_coverage<Op, 'C'>("tf32_m32n32k8_cov_C");
}

// ============================================================================
// f32 accumulator, int8 inputs
// ============================================================================

TEST(XE4_TMM_TiledCopy, i8_m32n32k32_CopyA) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 32>;
  tiled_copy_A_round_trip<Op>("i8_m32n32k32_A");
}

TEST(XE4_TMM_TiledCopy, i8_m32n32k32_CopyB) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 32>;
  tiled_copy_B_round_trip<Op>("i8_m32n32k32_B");
}

TEST(XE4_TMM_TiledCopy, i8_m32n32k32_CopyC) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 32>;
  tiled_copy_C_round_trip<Op>("i8_m32n32k32_C");
}

TEST(XE4_TMM_TiledCopy, i8_m32n8k32_CopyA) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 8>;
  tiled_copy_A_round_trip<Op>("i8_m32n8k32_A");
}

TEST(XE4_TMM_TiledCopy, i8_m32n8k32_CopyB) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 8>;
  tiled_copy_B_round_trip<Op>("i8_m32n8k32_B");
}

TEST(XE4_TMM_TiledCopy, i8_m32n8k32_CopyC) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 8>;
  tiled_copy_C_round_trip<Op>("i8_m32n8k32_C");
}

TEST(XE4_TMM_TiledCopy, i8_m32n32k32_CoverageA) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 32>;
  verify_thread_coverage<Op, 'A'>("i8_m32n32k32_cov_A");
}

TEST(XE4_TMM_TiledCopy, i8_m32n32k32_CoverageB) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 32>;
  verify_thread_coverage<Op, 'B'>("i8_m32n32k32_cov_B");
}

TEST(XE4_TMM_TiledCopy, i8_m32n32k32_CoverageC) {
  using Op = XE4_TMM<float, int8_t, int8_t, float, 32>;
  verify_thread_coverage<Op, 'C'>("i8_m32n32k32_cov_C");
}

// ============================================================================
// f32 accumulator, e5m2 (bf8) inputs
// ============================================================================

TEST(XE4_TMM_TiledCopy, e5m2_m32n32k32_CopyA) {
  using Op = XE4_TMM<float, cutlass::float_e5m2_t, cutlass::float_e5m2_t, float, 32>;
  tiled_copy_A_round_trip<Op>("e5m2_m32n32k32_A");
}

TEST(XE4_TMM_TiledCopy, e5m2_m32n32k32_CopyB) {
  using Op = XE4_TMM<float, cutlass::float_e5m2_t, cutlass::float_e5m2_t, float, 32>;
  tiled_copy_B_round_trip<Op>("e5m2_m32n32k32_B");
}

TEST(XE4_TMM_TiledCopy, e5m2_m32n32k32_CopyC) {
  using Op = XE4_TMM<float, cutlass::float_e5m2_t, cutlass::float_e5m2_t, float, 32>;
  tiled_copy_C_round_trip<Op>("e5m2_m32n32k32_C");
}

TEST(XE4_TMM_TiledCopy, e5m2_m32n16k32_CopyB) {
  using Op = XE4_TMM<float, cutlass::float_e5m2_t, cutlass::float_e5m2_t, float, 16>;
  tiled_copy_B_round_trip<Op>("e5m2_m32n16k32_B");
}

// ============================================================================
// f32 accumulator, e2m1 (mxfp4) inputs
// ============================================================================

TEST(XE4_TMM_TiledCopy, e2m1_m32n32k32_CopyA) {
  using Op = XE4_TMM<float, cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, 32>;
  tiled_copy_A_round_trip<Op>("e2m1_m32n32k32_A");
}

TEST(XE4_TMM_TiledCopy, e2m1_m32n32k32_CopyB) {
  using Op = XE4_TMM<float, cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, 32>;
  tiled_copy_B_round_trip<Op>("e2m1_m32n32k32_B");
}

TEST(XE4_TMM_TiledCopy, e2m1_m32n32k32_CopyC) {
  using Op = XE4_TMM<float, cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, 32>;
  tiled_copy_C_round_trip<Op>("e2m1_m32n32k32_C");
}

TEST(XE4_TMM_TiledCopy, e2m1_m32n32k32_CoverageA) {
  using Op = XE4_TMM<float, cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, 32>;
  verify_thread_coverage<Op, 'A'>("e2m1_m32n32k32_cov_A");
}

TEST(XE4_TMM_TiledCopy, e2m1_m32n32k32_CoverageB) {
  using Op = XE4_TMM<float, cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, 32>;
  verify_thread_coverage<Op, 'B'>("e2m1_m32n32k32_cov_B");
}

TEST(XE4_TMM_TiledCopy, e2m1_m32n32k32_CoverageC) {
  using Op = XE4_TMM<float, cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, 32>;
  verify_thread_coverage<Op, 'C'>("e2m1_m32n32k32_cov_C");
}

// ============================================================================
// Mixed A/B types: e4m3 A, e5m2 B
// ============================================================================

TEST(XE4_TMM_TiledCopy, e4m3_e5m2_m32n32k32_CopyA) {
  using Op = XE4_TMM<float, cutlass::float_e4m3_t, cutlass::float_e5m2_t, float, 32>;
  tiled_copy_A_round_trip<Op>("e4m3_e5m2_m32n32k32_A");
}

TEST(XE4_TMM_TiledCopy, e4m3_e5m2_m32n32k32_CopyB) {
  using Op = XE4_TMM<float, cutlass::float_e4m3_t, cutlass::float_e5m2_t, float, 32>;
  tiled_copy_B_round_trip<Op>("e4m3_e5m2_m32n32k32_B");
}

TEST(XE4_TMM_TiledCopy, e4m3_e5m2_m32n32k32_CopyC) {
  using Op = XE4_TMM<float, cutlass::float_e4m3_t, cutlass::float_e5m2_t, float, 32>;
  tiled_copy_C_round_trip<Op>("e4m3_e5m2_m32n32k32_C");
}

// ============================================================================
// Small N edge cases
// ============================================================================

TEST(XE4_TMM_TiledCopy, f16_m32n1k16_CopyB) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 1>;
  tiled_copy_B_round_trip<Op>("f16_m32n1k16_B");
}

TEST(XE4_TMM_TiledCopy, f16_m32n4k16_CopyB) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 4>;
  tiled_copy_B_round_trip<Op>("f16_m32n4k16_B");
}

TEST(XE4_TMM_TiledCopy, f16_m32n4k16_CopyC) {
  using Op = XE4_TMM<float, sycl::half, sycl::half, float, 4>;
  tiled_copy_C_round_trip<Op>("f16_m32n4k16_C");
}
