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

#include "gemm_blockscaled.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Config template for FP8 block-scaled GEMM.
// Input: ElementA/B = float_e4m3_t, ElementSF = float_ue8m0_t, SFVecSize = 32,
//        ElementAccumulator = float, TileShape = 128x256x64 (padding will be enabled) / 128x256x256 (padding will be disabled)
// Output: ElementD (fp32, fp16, bf16).
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class ElementSF_, int SFVecSize_, class ElementD_, const char* Name_>
struct BS_FP8_Config {
  using ElementA  = cutlass::float_e4m3_t;
  using ElementB  = cutlass::float_e4m3_t;
  using ElementSF = ElementSF_;
  static constexpr int SFVecSize = SFVecSize_;

  using ElementD = ElementD_;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  // K=1024 = 16*TileK (TileK=64): K/TileK=16 = 16x PipelineStages, exercises pipeline steady-state.
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 512, 1024, 1};

  using CtaTileShape_MNK  = Shape<_128, _256, _64>;
  using ClusterShape_MNK  = Shape<_1, _1, _1>;
  static constexpr int PipelineStages = 4;
  static constexpr bool EnableCooperativeSF = false;

  static constexpr const char* Name = Name_;
};

// Cluster variant: parameterized cluster shape.
// Cooperative SF loading is NOT supported for MXFP8 (max atom K=256, VS=32
// cannot satisfy cm_8x32B after cluster truncation). run_if_selected guards
// against --coop_sf=1 at compile time; the mainloop static_assert is a backstop.
template <class ElementSF_, int SFVecSize_, class ElementD_, int ClusterM_, int ClusterN_, const char* Name_>
struct BS_FP8_Cluster_Config {
  using ElementA  = cutlass::float_e4m3_t;
  using ElementB  = cutlass::float_e4m3_t;
  using ElementSF = ElementSF_;
  static constexpr int SFVecSize = SFVecSize_;

  using ElementD = ElementD_;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 512, 1024, 1};

  using CtaTileShape_MNK  = Shape<_128, _256, _64>;
  using ClusterShape_MNK  = Shape<cute::Int<ClusterM_>, cute::Int<ClusterN_>, _1>;
  static constexpr int PipelineStages = 4;
  static constexpr bool EnableCooperativeSF = false;

  static constexpr const char* Name = Name_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Config
///////////////////////////////////////////////////////////////////////////////////////////////////

#define DECL_FP8_CONFIG(SF, VS, D, name_)      \
  inline constexpr char name_[] = #name_;       \
  using cfg_##name_ = BS_FP8_Config<SF, VS, D, name_>

DECL_FP8_CONFIG(cutlass::float_ue8m0_t, 32, float,                           ue8m0_k32_fp32);
DECL_FP8_CONFIG(cutlass::float_ue8m0_t, 32, sycl::half,                      ue8m0_k32_fp16);
DECL_FP8_CONFIG(cutlass::float_ue8m0_t, 32, sycl::ext::oneapi::bfloat16,     ue8m0_k32_bf16);

#undef DECL_FP8_CONFIG

// Cluster configs for three cluster shapes: <1,2,1>, <2,1,1>, <2,2,1>
// Naming: {sf_type}_k{vecsize}_{output_dtype}_cluster_{MxN}

#define DECL_FP8_CLUSTER_CONFIG(SF, VS, D, CM, CN, name_)      \
  inline constexpr char name_[] = #name_;                       \
  using cfg_##name_ = BS_FP8_Cluster_Config<SF, VS, D, CM, CN, name_>

// ClusterShape<1,2,1>
DECL_FP8_CLUSTER_CONFIG(cutlass::float_ue8m0_t, 32, float,                           1, 2, ue8m0_k32_fp32_cluster_1x2);
DECL_FP8_CLUSTER_CONFIG(cutlass::float_ue8m0_t, 32, sycl::half,                      1, 2, ue8m0_k32_fp16_cluster_1x2);
DECL_FP8_CLUSTER_CONFIG(cutlass::float_ue8m0_t, 32, sycl::ext::oneapi::bfloat16,     1, 2, ue8m0_k32_bf16_cluster_1x2);

// ClusterShape<2,1,1>
DECL_FP8_CLUSTER_CONFIG(cutlass::float_ue8m0_t, 32, float,                           2, 1, ue8m0_k32_fp32_cluster_2x1);
DECL_FP8_CLUSTER_CONFIG(cutlass::float_ue8m0_t, 32, sycl::half,                      2, 1, ue8m0_k32_fp16_cluster_2x1);
DECL_FP8_CLUSTER_CONFIG(cutlass::float_ue8m0_t, 32, sycl::ext::oneapi::bfloat16,     2, 1, ue8m0_k32_bf16_cluster_2x1);

// ClusterShape<2,2,1>
DECL_FP8_CLUSTER_CONFIG(cutlass::float_ue8m0_t, 32, float,                           2, 2, ue8m0_k32_fp32_cluster_2x2);
DECL_FP8_CLUSTER_CONFIG(cutlass::float_ue8m0_t, 32, sycl::half,                      2, 2, ue8m0_k32_fp16_cluster_2x2);
DECL_FP8_CLUSTER_CONFIG(cutlass::float_ue8m0_t, 32, sycl::ext::oneapi::bfloat16,     2, 2, ue8m0_k32_bf16_cluster_2x2);

#undef DECL_FP8_CLUSTER_CONFIG

///////////////////////////////////////////////////////////////////////////////////////////////////
// Test runner
///////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream& print_usage(std::ostream& out) {
  out << "Block-Scaled FP8 GEMM Example\n\n"
      << "Options:\n\n"
      << "  --help                      Displays this usage statement\n\n"
      << "  --m=<int>                   Override M dimension (per-config default if not set)\n"
      << "  --n=<int>                   Override N dimension\n"
      << "  --k=<int>                   Override K dimension\n"
      << "  --coop_sf=<0>               Cooperative SF loading NOT supported for MXFP8 (max atom K=256, VS=32 cannot satisfy cm_8x32B after cluster truncation)\n"
      << "  --config=<name>[,<name>]    Run specific configs (comma-separated); omit to run all 12\n\n"
      << "Available configs ({sf_type}_k{vecsize}_{output_dtype}[_cluster_{MxN}]):\n"
      << "  ue8m0_k32_{fp32,fp16,bf16}\n"
      << "  ue8m0_k32_{fp32,fp16,bf16}_cluster_1x2   (ClusterShape<1,2,1>)\n"
      << "  ue8m0_k32_{fp32,fp16,bf16}_cluster_2x1   (ClusterShape<2,1,1>)\n"
      << "  ue8m0_k32_{fp32,fp16,bf16}_cluster_2x2   (ClusterShape<2,2,1>)\n\n"
      << "Examples:\n"
      << "  ./xe4_gemm_blockscaled_fp8                                         # all 3 configs\n"
      << "  ./xe4_gemm_blockscaled_fp8 --m=256 --n=512 --k=1024                # all configs, custom shape\n"
      << "  ./xe4_gemm_blockscaled_fp8 --config=ue8m0_k32_fp32                 # one config\n"
      << "  ./xe4_gemm_blockscaled_fp8 --config=ue8m0_k32_fp32,ue8m0_k32_bf16  # two configs\n"
      << "  ./xe4_gemm_blockscaled_fp8 --config=ue8m0_k32_fp32 --m=256         # config + custom shape\n";
  return out;
}

int main(int argc, char **argv) {
  Options::parse(argc, argv);
  if (Options::help) {
    print_usage(std::cout);
    return 0;
  }

  sycl::queue q;
  return run_configs<
      cfg_ue8m0_k32_fp32, cfg_ue8m0_k32_fp16, cfg_ue8m0_k32_bf16,
      // ClusterShape<1,2,1>
      cfg_ue8m0_k32_fp32_cluster_1x2, cfg_ue8m0_k32_fp16_cluster_1x2, cfg_ue8m0_k32_bf16_cluster_1x2,
      // ClusterShape<2,1,1>
      cfg_ue8m0_k32_fp32_cluster_2x1, cfg_ue8m0_k32_fp16_cluster_2x1, cfg_ue8m0_k32_bf16_cluster_2x1,
      // ClusterShape<2,2,1>
      cfg_ue8m0_k32_fp32_cluster_2x2, cfg_ue8m0_k32_fp16_cluster_2x2, cfg_ue8m0_k32_bf16_cluster_2x2
  >(Options::configs, q) ? 0 : 1;
}