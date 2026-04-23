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

// GEMM example using MMA-aware LDSM epilogue (Xe4LdsmMmaAdmaBuilderImpl)
// Validates GEMM correctness with MMA atom-boundary-aware warp distribution
// in descriptor-based S2R/R2S copy operations

#include "gemm_ldsm_mma_epilogue.hpp"
#include <gtest/gtest.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
// Base config
//
// Default GEMM configuration for MMA-aware LDSM epilogue tests.
// All test configs inherit from this and override only the parameters they differ on.
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GEMM_LDSM_MMA_BASE_CONFIG {
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
  static constexpr auto activation_type = ActivationType::None;
  static constexpr auto operationC_type = OperationCType::None;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {256, 256, 128, 1};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bf16: D = A*B (identity epilogue, no source C)
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GEMM_LDSM_MMA_BF16 : public GEMM_LDSM_MMA_BASE_CONFIG {
  using ElementA = bf16;
  using ElementB = bf16;
  using ElementC = void;
  using ElementD = bf16;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// fp32: D = A*B (identity, no source C, fp32 output)
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GEMM_LDSM_MMA_FP32 : public GEMM_LDSM_MMA_BASE_CONFIG {
  using ElementC = void;
  using ElementD = float;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {256, 256, 128, 1};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bf16: D = A*B + C (residual add)
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GEMM_LDSM_MMA_BF16_ADD : public GEMM_LDSM_MMA_BASE_CONFIG {
  using ElementA = bf16;
  using ElementB = bf16;
  using ElementC = bf16;
  using ElementD = bf16;
  static constexpr auto operationC_type = OperationCType::Add;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bf16 with larger problem size (multi-tile)
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GEMM_LDSM_MMA_BF16_LARGE : public GEMM_LDSM_MMA_BF16 {
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 768, 384, 1};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test suite
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class GemmLdsmMmaEpilogueTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(GemmLdsmMmaEpilogueTest);

TYPED_TEST_P(GemmLdsmMmaEpilogueTest, simple_run) {
  run_gemm_ldsm_mma_epilogue<TypeParam>();
}

REGISTER_TYPED_TEST_SUITE_P(GemmLdsmMmaEpilogueTest, simple_run);
using GemmLdsmMmaEpilogueTests = ::testing::Types<
    GEMM_LDSM_MMA_BF16,
    GEMM_LDSM_MMA_FP32,
    GEMM_LDSM_MMA_BF16_ADD,
    GEMM_LDSM_MMA_BF16_LARGE
  >;
INSTANTIATE_TYPED_TEST_SUITE_P(GemmLdsmMmaEpilogue, GemmLdsmMmaEpilogueTest, GemmLdsmMmaEpilogueTests);

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int run_tests = RUN_ALL_TESTS();
  if (!::testing::UnitTest::GetInstance()->test_to_run_count()) {
    std::cout << "No tests were run.\n";
    return 1;
  }
  return run_tests;
}
