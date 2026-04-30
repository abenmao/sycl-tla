/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gemm_prefetch_reduce.hpp"
#include <gtest/gtest.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Test configurations for XE4 GEMM with ADMA Prefetch and ADMA Store-Reduce
///
/// GEMM_PREFETCH_CONFIG: Base configuration shared by both test variants.
///   - fp16 A/B/C/D, float accumulator
///   - Row-major A and B
///   - 256x256x128 CTA tile, 2 mainloop stages
///   - Problem shape {512, 768, 384, 1}
///
/// GEMM_PREFETCH: Tests prefetch-ahead in mainloop with standard epilogue (TBX-safe).
///   Validates that L2 prefetch does not corrupt data and GEMM produces correct results.
///
/// GEMM_PREFETCH_REDUCE: Tests prefetch + STORE_REDUCE epilogue (requires real XE4 HW).
///   Pre-initializes D=0 so reduce-Add(0, result) == result for validation equivalence.
///   The TBX simulator does not support send.dma.tensor.reduce_l2g.
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GEMM_PREFETCH_CONFIG {
  using ElementA = fp16;
  using ElementB = fp16;
  using ElementC = fp16;
  using ElementD = fp16;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using CtaTileShape_MNK = Shape<_256,_256,_128>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  static constexpr int StagesA = 2;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 768, 384, 1};
};

// Prefetch-only test: standard epilogue, TBX-safe
struct GEMM_PREFETCH : public GEMM_PREFETCH_CONFIG {};

// Prefetch + Store-Reduce test: reduce epilogue, requires real HW
struct GEMM_PREFETCH_REDUCE : public GEMM_PREFETCH_CONFIG {};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Prefetch test suite (TBX-safe)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class GemmPrefetchTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(GemmPrefetchTest);

TYPED_TEST_P(GemmPrefetchTest, prefetch_run) {
  run_gemm_prefetch<TypeParam>();
}

REGISTER_TYPED_TEST_SUITE_P(GemmPrefetchTest, prefetch_run);
using PrefetchTests = ::testing::Types<GEMM_PREFETCH>;
INSTANTIATE_TYPED_TEST_SUITE_P(GemmPrefetch, GemmPrefetchTest, PrefetchTests);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Prefetch + Reduce test suite (requires real XE4 hardware)
///
/// IMPORTANT: These tests compile and generate correct asm on TBX, but the
/// send.dma.tensor.reduce_l2g instruction is not simulated — the store-reduce
/// returns base_val (no reduction applied). Expected to pass on actual hardware.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class GemmPrefetchReduceTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(GemmPrefetchReduceTest);

TYPED_TEST_P(GemmPrefetchReduceTest, prefetch_reduce_run) {
  run_gemm_prefetch_reduce<TypeParam>();
}

REGISTER_TYPED_TEST_SUITE_P(GemmPrefetchReduceTest, prefetch_reduce_run);
using PrefetchReduceTests = ::testing::Types<GEMM_PREFETCH_REDUCE>;
INSTANTIATE_TYPED_TEST_SUITE_P(GemmPrefetchReduce, GemmPrefetchReduceTest, PrefetchReduceTests);

////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int run_tests = RUN_ALL_TESTS();
  if (!::testing::UnitTest::GetInstance()->test_to_run_count()) {
    std::cout << "No tests were run.\n";
    return 1;
  }
  return run_tests;
}
