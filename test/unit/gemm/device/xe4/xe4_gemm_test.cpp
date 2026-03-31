/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation, All rights reserved.
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
/*! \file
    \brief Tests for device-wide GEMM interface using the Xe4 testbed
*/

#include <iostream>

#include "cutlass_unit_test.h"
#include "xe4_gemm_testbed.hpp"

using namespace cute;

#if defined(CUTLASS_ENABLE_SYCL) && defined(SYCL_INTEL_TARGET)

namespace cutlass {
namespace test {
namespace gemm {
namespace xe4 {

template <
  typename LayoutB_,
  typename CtaNumMN_,
  ActivationType activation_type_,
  OperationCType operationC_type_,
  int M_,
  int N_,
  int K_,
  int L_>
struct GEMM_CONFIG {
  using ElementA = fp16;
  using ElementB = fp16;
  using ElementC = fp16;
  using ElementD = fp16;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = LayoutB_;
  using LayoutC = cutlass::layout::RowMajor;
  using CtaTileShape_MNK = Shape<_256, _256, _128>;
  using CtaNum_MN = CtaNumMN_;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  static constexpr int StagesA = 2;
  static constexpr bool is_persistent = false;
  static constexpr auto activation_type = activation_type_;
  static constexpr auto operationC_type = operationC_type_;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {M_, N_, K_, L_};
};

template <
  typename ElementAB_,
  typename ElementSF_,
  typename ElementD_,
  int SFVecSize_,
  typename CtaTileShape_MNK_,
  int PipelineStages_,
  int M_,
  int N_,
  int K_,
  int L_,
  typename ClusterShape_MNK_ = Shape<_1, _1, _1>>
struct BS_GEMM_CONFIG_BASE {
  using ElementA  = ElementAB_;
  using ElementB  = ElementAB_;
  using ElementSF = ElementSF_;
  static constexpr int SFVecSize = SFVecSize_;

  using ElementD = ElementD_;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  static constexpr cute::array<int, 4> ProblemShape_MNKL = {M_, N_, K_, L_};
  using CtaTileShape_MNK = CtaTileShape_MNK_;
  using ClusterShape_MNK = ClusterShape_MNK_;
  static constexpr int PipelineStages = PipelineStages_;
};

TEST(XE4_GEMM, config_a_row_row) {
  using ConfigA = GEMM_CONFIG<
    cutlass::layout::RowMajor,
    Shape<_2, _1>,
    ActivationType::SiLu,
    OperationCType::Mul,
    512,
    768,
    384,
    1>;
  run_gemm<ConfigA>();
}

TEST(XE4_GEMM, config_a_row_row_small) {
  using ConfigA = GEMM_CONFIG<
    cutlass::layout::RowMajor,
    Shape<_1, _1>,
    ActivationType::SiLu,
    OperationCType::Mul,
    32,
    32,
    128,
    1>;
  run_gemm<ConfigA>();
}

TEST(XE4_GEMM, config_b_row_col) {
  using ConfigB = GEMM_CONFIG<
    cutlass::layout::ColumnMajor,
    Shape<_1, _1>,
    ActivationType::None,
    OperationCType::Add,
    256,
    512,
    384,
    1>;
  run_gemm<ConfigB>();
}

using BS_GEMM_NVFP4 = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue4m3_t,
  float,
  16,
  Shape<_256, _256, _128>,
  4,
  256,
  512,
  384,
  1>;

using BS_GEMM_NVFP4_small = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue4m3_t,
  float,
  16,
  Shape<_64, _64, _128>,
  2,
  64,
  64,
  128,
  1>;

  using BS_GEMM_NVFP4_2k = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue4m3_t,
  float,
  16,
  Shape<_256, _512, _128>,
  2,
  2048,
  2048,
  2048,
  1>;

using BS_GEMM_NVFP4_OP_FP16 = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue4m3_t,
  sycl::half,
  16,
  Shape<_256, _256, _128>,
  4,
  256,
  512,
  384,
  1>;

using BS_GEMM_NVFP4_OP_BF16 = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue4m3_t,
  sycl::ext::oneapi::bfloat16,
  16,
  Shape<_256, _256, _128>,
  4,
  256,
  512,
  384,
  1>;

using BS_GEMM_NVFP4_CLUSTER_211 = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue4m3_t,
  float,
  16,
  Shape<_256, _256, _128>,
  4,
  256,
  512,
  384,
  1,
  Shape<_2, _1, _1>>;

  using BS_GEMM_NVFP4_CLUSTER_121 = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue4m3_t,
  float,
  16,
  Shape<_256, _256, _128>,
  4,
  256,
  512,
  384,
  1,
  Shape<_1, _2, _1>>;

  using BS_GEMM_NVFP4_CLUSTER_112 = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue4m3_t,
  float,
  16,
  Shape<_256, _256, _128>,
  4,
  256,
  512,
  384,
  1,
  Shape<_1, _1, _2>>;

template <typename ElementD_>
using BS_GEMM_NVFP4P_VS16_T = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue5m3_t,
  ElementD_,
  16,
  Shape<_256, _256, _128>,
  4,
  256,
  512,
  384,
  1>;

using BS_GEMM_NVFP4P_VS16 = BS_GEMM_NVFP4P_VS16_T<float>;
using BS_GEMM_NVFP4P_VS16_OP_FP16 = BS_GEMM_NVFP4P_VS16_T<sycl::half>;
using BS_GEMM_NVFP4P_VS16_OP_BF16 = BS_GEMM_NVFP4P_VS16_T<sycl::ext::oneapi::bfloat16>;

using BS_GEMM_NVFP4P_VS16_small = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue5m3_t,
  float,
  16,
  Shape<_64, _64, _128>,
  2,
  64,
  64,
  128,
  1>;

template <typename ElementD_>
using BS_GEMM_NVFP4P_VS32_T = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue5m3_t,
  ElementD_,
  32,
  Shape<_256, _256, _256>,
  4,
  256,
  512,
  512,
  1>;

using BS_GEMM_NVFP4P_VS32 = BS_GEMM_NVFP4P_VS32_T<float>;
using BS_GEMM_NVFP4P_VS32_OP_FP16 = BS_GEMM_NVFP4P_VS32_T<sycl::half>;
using BS_GEMM_NVFP4P_VS32_OP_BF16 = BS_GEMM_NVFP4P_VS32_T<sycl::ext::oneapi::bfloat16>;

template <typename ElementD_>
using BS_GEMM_MXFP4_VS16_T = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue8m0_t,
  ElementD_,
  16,
  Shape<_256, _256, _128>,
  4,
  256,
  512,
  384,
  1>;

using BS_GEMM_MXFP4_VS16 = BS_GEMM_MXFP4_VS16_T<float>;
using BS_GEMM_MXFP4_VS16_OP_FP16 = BS_GEMM_MXFP4_VS16_T<sycl::half>;
using BS_GEMM_MXFP4_VS16_OP_BF16 = BS_GEMM_MXFP4_VS16_T<sycl::ext::oneapi::bfloat16>;

using BS_GEMM_MXFP4_VS16_small = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue8m0_t,
  float,
  16,
  Shape<_64, _64, _128>,
  2,
  64,
  64,
  384,
  1>;

template <typename ElementD_>
using BS_GEMM_MXFP4_VS32_T = BS_GEMM_CONFIG_BASE<
  cutlass::float_e2m1_t,
  cutlass::float_ue8m0_t,
  ElementD_,
  32,
  Shape<_128, _256, _256>,
  4,
  256,
  512,
  512,
  1>;

using BS_GEMM_MXFP4_VS32 = BS_GEMM_MXFP4_VS32_T<float>;
using BS_GEMM_MXFP4_VS32_OP_FP16 = BS_GEMM_MXFP4_VS32_T<sycl::half>;
using BS_GEMM_MXFP4_VS32_OP_BF16 = BS_GEMM_MXFP4_VS32_T<sycl::ext::oneapi::bfloat16>;

template <typename ElementD_>
using BS_GEMM_FP8_T = BS_GEMM_CONFIG_BASE<
  cutlass::float_e4m3_t,
  cutlass::float_ue8m0_t,
  ElementD_,
  32,
  Shape<_256, _256, _256>,
  4,
  256,
  512,
  512,
  1>;

using BS_GEMM_FP8 = BS_GEMM_FP8_T<float>;
using BS_GEMM_FP8_OP_FP16 = BS_GEMM_FP8_T<sycl::half>;
using BS_GEMM_FP8_OP_BF16 = BS_GEMM_FP8_T<sycl::ext::oneapi::bfloat16>;

using BS_GEMM_FP8_small = BS_GEMM_CONFIG_BASE<
  cutlass::float_e4m3_t,
  cutlass::float_ue8m0_t,
  float,
  32,
  Shape<_64, _64, _256>,
  2,
  64,
  64,
  256,
  1>;


TEST(XE4_GEMM, blockscaled_fp4_nvfp4) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4>());
}

TEST(XE4_GEMM, blockscaled_fp4_nvfp4_small) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4_small>());
}

TEST(XE4_GEMM, DISABLED_blockscaled_fp4_nvfp4_2k) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4_2k>());
}

TEST(XE4_GEMM, blockscaled_fp4_nvfp4_op_fp16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4_OP_FP16>());
}


TEST(XE4_GEMM, blockscaled_fp4_nvfp4_op_bf16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4_OP_BF16>());
}

// Not supported in current HW generation due to lack of cluster-level support
// TEST(XE4_GEMM, DISABLED_blockscaled_fp4_nvfp4_cluster211) {
//    EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4_CLUSTER_211>());
// }
// TEST(XE4_GEMM, DISABLED_blockscaled_fp4_nvfp4_cluster112) {
//    EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4_CLUSTER_112>());
// }
// TEST(XE4_GEMM, DISABLED_blockscaled_fp4_nvfp4_cluster121) {
//    EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4_CLUSTER_121>());
// }

TEST(XE4_GEMM, blockscaled_fp4_nvfp4p_vs16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4P_VS16>());
}

TEST(XE4_GEMM, blockscaled_fp4_nvfp4p_vs16_op_fp16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4P_VS16_OP_FP16>());
}

TEST(XE4_GEMM, blockscaled_fp4_nvfp4p_vs16_op_bf16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4P_VS16_OP_BF16>());
}

TEST(XE4_GEMM, blockscaled_fp4_nvfp4p_vs16_small) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4P_VS16_small>());
}

TEST(XE4_GEMM, blockscaled_fp4_nvfp4p_vs32) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4P_VS32>());
}

TEST(XE4_GEMM, blockscaled_fp4_nvfp4p_vs32_op_fp16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4P_VS32_OP_FP16>());
}

TEST(XE4_GEMM, blockscaled_fp4_nvfp4p_vs32_op_bf16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_NVFP4P_VS32_OP_BF16>());
}

TEST(XE4_GEMM, blockscaled_fp4_mxfp4_vs16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_MXFP4_VS16>());
}

TEST(XE4_GEMM, blockscaled_fp4_mxfp4_vs16_op_fp16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_MXFP4_VS16_OP_FP16>());
}

TEST(XE4_GEMM, blockscaled_fp4_mxfp4_vs16_op_bf16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_MXFP4_VS16_OP_BF16>());
}

TEST(XE4_GEMM, blockscaled_fp4_mxfp4_vs16_small) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_MXFP4_VS16_small>());
}

TEST(XE4_GEMM, blockscaled_fp4_mxfp4_vs32) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_MXFP4_VS32>());
}

TEST(XE4_GEMM, blockscaled_fp4_mxfp4_vs32_op_fp16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_MXFP4_VS32_OP_FP16>());
}

TEST(XE4_GEMM, blockscaled_fp4_mxfp4_vs32_op_bf16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_MXFP4_VS32_OP_BF16>());
}

TEST(XE4_GEMM, blockscaled_fp8) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_FP8>());
}

TEST(XE4_GEMM, blockscaled_fp8_op_fp16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_FP8_OP_FP16>());
}

TEST(XE4_GEMM, blockscaled_fp8_op_bf16) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_FP8_OP_BF16>());
}

TEST(XE4_GEMM, blockscaled_fp8_small) {
  EXPECT_TRUE(run_blockscaled_gemm<BS_GEMM_FP8_small>());
}

} // namespace xe4
} // namespace gemm
} // namespace test
} // namespace cutlass

#endif
