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

// Standalone GEMM example using LDSM-based epilogue (matrix descriptor path)
// Validates GEMM correctness with descriptor-based S2R/R2S copy operations
// See MODULE_DESIGN.md - Module 5

#include "gemm_ldsm_epilogue.hpp"
#include <gtest/gtest.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
// Base config
//
// Default GEMM configuration for LDSM epilogue tests.
// All test configs inherit from this and override only the parameters they differ on.
// The base uses fp16 for all tensors, 256x256x128 tile with identity epilogue (D = A*B).
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GEMM_LDSM_BASE_CONFIG {
  using ElementA = fp16;
  using ElementB = fp16;
  using ElementC = fp16;                  // Source C (for residual add, multiply, etc.)
  using ElementD = fp16;                  // Output D
  using ElementAccumulator = float;       // Accumulator precision (always fp32)
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using CtaTileShape_MNK = Shape<_256,_256,_128>;  // CTA tile (must match SLM capacity)
  using ClusterShape_MNK = Shape<_1, _1, _1>;      // Single-CTA cluster

  static constexpr int StagesA = 2;                           // Pipeline stages for A/B loads
  static constexpr auto activation_type = ActivationType::None; // No activation function
  static constexpr auto operationC_type = OperationCType::None; // No source C operation
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {256, 256, 128, 1};  // Single-tile problem
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bf16: D = A*B (identity epilogue, no source C)
// Tests the simplest LDSM epilogue path — no fusion, just accumulator → output conversion.
// ElementC = void means no source C tensor is loaded.
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GEMM_LDSM_BF16 : public GEMM_LDSM_BASE_CONFIG {
  using ElementA = bf16;
  using ElementB = bf16;
  using ElementC = void;
  using ElementD = bf16;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// fp16: D = SiLu(A*B) * C (activation + source multiply)
// Tests fused epilogue with SiLu activation AND element-wise multiply with source C.
// Exercises both the S2R (load C from SLM) and R2S (store D to SLM) LDSM paths.
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GEMM_LDSM_FP16_SILU_MUL : public GEMM_LDSM_BASE_CONFIG {
  static constexpr auto activation_type = ActivationType::SiLu;
  static constexpr auto operationC_type = OperationCType::Mul;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// fp32: D = A*B (identity, no source C, fp32 output)
// Tests LDSM epilogue with wider output type (32-bit float).
// This exercises a different Vlen in the LDSM op (fewer elements per vector due to wider type).
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GEMM_LDSM_FP32 : public GEMM_LDSM_BASE_CONFIG {
  using ElementC = void;
  using ElementD = float;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {256, 256, 128, 1};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bf16: D = A*B + C (residual add)
// Tests LDSM epilogue with element-wise addition of source C (residual connection pattern).
// Validates that the S2R load of C and the fused add operate correctly through the
// descriptor-based path.
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GEMM_LDSM_BF16_ADD : public GEMM_LDSM_BASE_CONFIG {
  using ElementA = bf16;
  using ElementB = bf16;
  using ElementC = bf16;
  using ElementD = bf16;
  static constexpr auto operationC_type = OperationCType::Add;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// bf16 with larger problem size (multi-tile)
// Tests that LDSM epilogue works correctly when the problem spans multiple CTA tiles.
// With CtaTileShape 256x256x128 and problem 512x768x384, this exercises the epilogue
// loop over multiple (epi_m, epi_n) iterations and multiple K-loop accumulator stages.
////////////////////////////////////////////////////////////////////////////////////////////////////

struct GEMM_LDSM_BF16_LARGE : public GEMM_LDSM_BF16 {
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 768, 384, 1};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test suite
//
// Uses GTest TYPED_TEST_SUITE_P to run each config struct through run_gemm_ldsm_epilogue<>().
// Each config exercises a different combination of data types, epilogue operations,
// and problem sizes to validate the LDSM descriptor-based epilogue path.
//
// Test coverage:
//   GEMM_LDSM_BF16           — basic bf16, no fusion (identity epilogue)
//   GEMM_LDSM_FP16_SILU_MUL  — fp16 with SiLu activation + multiply with source C
//   GEMM_LDSM_FP32           — fp32 output (wider element type)
//   GEMM_LDSM_BF16_ADD       — bf16 with residual add
//   GEMM_LDSM_BF16_LARGE     — multi-tile problem to test iteration logic
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class GemmLdsmEpilogueTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(GemmLdsmEpilogueTest);

TYPED_TEST_P(GemmLdsmEpilogueTest, simple_run) {
  run_gemm_ldsm_epilogue<TypeParam>();
}

REGISTER_TYPED_TEST_SUITE_P(GemmLdsmEpilogueTest, simple_run);
using GemmLdsmEpilogueTests = ::testing::Types<
    GEMM_LDSM_BF16,
    GEMM_LDSM_FP32,
    GEMM_LDSM_BF16_ADD,
    GEMM_LDSM_BF16_LARGE
  >;
INSTANTIATE_TYPED_TEST_SUITE_P(GemmLdsmEpilogue, GemmLdsmEpilogueTest, GemmLdsmEpilogueTests);

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int run_tests = RUN_ALL_TESTS();
  if (!::testing::UnitTest::GetInstance()->test_to_run_count()) {
    std::cout << "No tests were run.\n";
    return 1;
  }
  return run_tests;
}
