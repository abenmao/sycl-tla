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
// Config template for FP4 block-scaled GEMM.
// Fixed: ElementA/B = float_e2m1_t, ElementAccumulator = float
// Varying: TileShape = 128x256x64 (padding will be enabled) / 128x256x768 (padding will be disabled), ElementSF, SFVecSize, ElementD.
//
// Name format: {sf_type}_k{vecsize}_{output_dtype}
//   e.g. ue4m3_k16_fp32, ue5m3_k32_bf16
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class ElementSF_, int SFVecSize_, class ElementD_, int TileK_, const char* Name_>
struct BS_FP4_Config {
  using ElementA  = cutlass::float_e2m1_t;
  using ElementB  = cutlass::float_e2m1_t;
  using ElementSF = ElementSF_;
  static constexpr int SFVecSize = SFVecSize_;

  using ElementD = ElementD_;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  // K=1024 = 16*TileK (TileK=64): K/TileK=16 = 16x PipelineStages, exercises pipeline steady-state.
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 512, 1024, 1};

  using CtaTileShape_MNK  = Shape<_128, _256, cute::Int<TileK_>>;
  using ClusterShape_MNK  = Shape<_1, _1, _1>;
  static constexpr int PipelineStages = 4;
  static constexpr bool EnableCooperativeSF = false;

  static constexpr const char* Name = Name_;
};

// Cluster variant: parameterized cluster shape.
// Non-cooperative (default): TileK=64 — padding handles the cm_8x32B constraint.
// Cooperative (--coop_sf=1): ConfigWithCoopSF<Config, true> overrides TileK to
//   256 (VS=16) or 512 (VS=32), ensuring each CTA's ADMA box satisfies cm_8x32B.
template <class ElementSF_, int SFVecSize_, class ElementD_, int TileK_, int ClusterM_, int ClusterN_, const char* Name_>
struct BS_FP4_Cluster_Config {
  using ElementA  = cutlass::float_e2m1_t;
  using ElementB  = cutlass::float_e2m1_t;
  using ElementSF = ElementSF_;
  static constexpr int SFVecSize = SFVecSize_;

  using ElementD = ElementD_;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 512, 1024, 1};

  using CtaTileShape_MNK  = Shape<_128, _256, cute::Int<TileK_>>;
  using ClusterShape_MNK  = Shape<cute::Int<ClusterM_>, cute::Int<ClusterN_>, _1>;
  static constexpr int PipelineStages = 4;
  static constexpr bool EnableCooperativeSF = false;

  // Cooperative SF constraint: (TileK / SFVecSize / max(ClusterM, ClusterN)) must be
  // a multiple of 8. The ADMA 2D-block-copy uses cm_8x32B core-matrix units (8 rows
  // per write). With cooperative loading, each CTA's ADMA box covers only
  // (TileK / SFVecSize / max(ClusterM, ClusterN)) SF rows. If this is not a multiple
  // of 8 (e.g. 12), the last 8-row core-matrix write overflows into a peer CTA's SF
  // region within the same pipeline stage, causing intra-stage data corruption that
  // padding cannot fix. This assert guards the config struct directly; ConfigWithCoopSF
  // in gemm_blockscaled.hpp re-validates when cooperative SF is actually enabled.
  static constexpr int MaxClusterDim_ = (ClusterM_ > ClusterN_) ? ClusterM_ : ClusterN_;
  static_assert(!EnableCooperativeSF ||
      ((TileK_ / SFVecSize_ / MaxClusterDim_ >= 8) &&
       (TileK_ / SFVecSize_ / MaxClusterDim_) % 8 == 0),
      "Cooperative SF loading requires (TileK / SFVecSize / max(ClusterM, ClusterN)) to be "
      "a multiple of 8 (>= 8). The ADMA 2D-block-copy writes in cm_8x32B core-matrix units "
      "(8 rows). If the per-CTA SF row count is not a multiple of 8 (e.g. 12), the last "
      "core-matrix write overflows into a peer CTA's SF region within the same pipeline "
      "stage, causing intra-stage collision that padding cannot fix. "
      "VS=16 needs TileK>=256, VS=32 needs TileK>=512.");

  static constexpr const char* Name = Name_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Config aliases — 5 SF/VS combos x 3 output dtypes = 15 configs
// Each macro invocation declares both the name string and the type alias.
///////////////////////////////////////////////////////////////////////////////////////////////////

#define DECL_FP4_CONFIG(SF, VS, D, name_)                    \
  inline constexpr char name_[] = #name_;                     \
  using cfg_##name_ = BS_FP4_Config<SF, VS, D, 64, name_>

DECL_FP4_CONFIG(cutlass::float_ue4m3_t, 16, float,                           ue4m3_k16_fp32);
DECL_FP4_CONFIG(cutlass::float_ue4m3_t, 16, sycl::half,                      ue4m3_k16_fp16);
DECL_FP4_CONFIG(cutlass::float_ue4m3_t, 16, sycl::ext::oneapi::bfloat16,     ue4m3_k16_bf16);

DECL_FP4_CONFIG(cutlass::float_ue5m3_t, 16, float,                           ue5m3_k16_fp32);
DECL_FP4_CONFIG(cutlass::float_ue5m3_t, 16, sycl::half,                      ue5m3_k16_fp16);
DECL_FP4_CONFIG(cutlass::float_ue5m3_t, 16, sycl::ext::oneapi::bfloat16,     ue5m3_k16_bf16);

DECL_FP4_CONFIG(cutlass::float_ue5m3_t, 32, float,                           ue5m3_k32_fp32);
DECL_FP4_CONFIG(cutlass::float_ue5m3_t, 32, sycl::half,                      ue5m3_k32_fp16);
DECL_FP4_CONFIG(cutlass::float_ue5m3_t, 32, sycl::ext::oneapi::bfloat16,     ue5m3_k32_bf16);

DECL_FP4_CONFIG(cutlass::float_ue8m0_t, 16, float,                           ue8m0_k16_fp32);
DECL_FP4_CONFIG(cutlass::float_ue8m0_t, 16, sycl::half,                      ue8m0_k16_fp16);
DECL_FP4_CONFIG(cutlass::float_ue8m0_t, 16, sycl::ext::oneapi::bfloat16,     ue8m0_k16_bf16);

DECL_FP4_CONFIG(cutlass::float_ue8m0_t, 32, float,                           ue8m0_k32_fp32);
DECL_FP4_CONFIG(cutlass::float_ue8m0_t, 32, sycl::half,                      ue8m0_k32_fp16);
DECL_FP4_CONFIG(cutlass::float_ue8m0_t, 32, sycl::ext::oneapi::bfloat16,     ue8m0_k32_bf16);

#undef DECL_FP4_CONFIG

// Cluster configs for three cluster shapes: <1,2,1>, <2,1,1>, <2,2,1>
// Naming: {sf_type}_k{vecsize}_{output_dtype}_cluster_{MxN}
//
// Non-cooperative (default): TileK=64 — padding handles the cm_8x32B constraint.
// Cooperative (--coop_sf=1): CoopConfigType in the struct provides TileK=256 for
//   VS=16, TileK=512 for VS=32 — ensures each CTA's ADMA box satisfies cm_8x32B.

#define DECL_FP4_CLUSTER_CONFIG_K16(SF, D, CM, CN, name_)            \
  inline constexpr char name_[] = #name_;                             \
  using cfg_##name_ = BS_FP4_Cluster_Config<SF, 16, D, 64, CM, CN, name_>

#define DECL_FP4_CLUSTER_CONFIG_K32(SF, D, CM, CN, name_)            \
  inline constexpr char name_[] = #name_;                             \
  using cfg_##name_ = BS_FP4_Cluster_Config<SF, 32, D, 64, CM, CN, name_>

// ClusterShape<1,2,1>
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue4m3_t, float,                           1, 2, ue4m3_k16_fp32_cluster_1x2);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue4m3_t, sycl::half,                      1, 2, ue4m3_k16_fp16_cluster_1x2);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue4m3_t, sycl::ext::oneapi::bfloat16,     1, 2, ue4m3_k16_bf16_cluster_1x2);

DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue5m3_t, float,                           1, 2, ue5m3_k16_fp32_cluster_1x2);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue5m3_t, sycl::half,                      1, 2, ue5m3_k16_fp16_cluster_1x2);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue5m3_t, sycl::ext::oneapi::bfloat16,     1, 2, ue5m3_k16_bf16_cluster_1x2);

DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue5m3_t, float,                           1, 2, ue5m3_k32_fp32_cluster_1x2);
DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue5m3_t, sycl::half,                      1, 2, ue5m3_k32_fp16_cluster_1x2);
DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue5m3_t, sycl::ext::oneapi::bfloat16,     1, 2, ue5m3_k32_bf16_cluster_1x2);

DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue8m0_t, float,                           1, 2, ue8m0_k16_fp32_cluster_1x2);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue8m0_t, sycl::half,                      1, 2, ue8m0_k16_fp16_cluster_1x2);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue8m0_t, sycl::ext::oneapi::bfloat16,     1, 2, ue8m0_k16_bf16_cluster_1x2);

DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue8m0_t, float,                           1, 2, ue8m0_k32_fp32_cluster_1x2);
DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue8m0_t, sycl::half,                      1, 2, ue8m0_k32_fp16_cluster_1x2);
DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue8m0_t, sycl::ext::oneapi::bfloat16,     1, 2, ue8m0_k32_bf16_cluster_1x2);

// ClusterShape<2,1,1>
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue4m3_t, float,                           2, 1, ue4m3_k16_fp32_cluster_2x1);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue4m3_t, sycl::half,                      2, 1, ue4m3_k16_fp16_cluster_2x1);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue4m3_t, sycl::ext::oneapi::bfloat16,     2, 1, ue4m3_k16_bf16_cluster_2x1);

DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue5m3_t, float,                           2, 1, ue5m3_k16_fp32_cluster_2x1);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue5m3_t, sycl::half,                      2, 1, ue5m3_k16_fp16_cluster_2x1);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue5m3_t, sycl::ext::oneapi::bfloat16,     2, 1, ue5m3_k16_bf16_cluster_2x1);

DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue5m3_t, float,                           2, 1, ue5m3_k32_fp32_cluster_2x1);
DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue5m3_t, sycl::half,                      2, 1, ue5m3_k32_fp16_cluster_2x1);
DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue5m3_t, sycl::ext::oneapi::bfloat16,     2, 1, ue5m3_k32_bf16_cluster_2x1);

DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue8m0_t, float,                           2, 1, ue8m0_k16_fp32_cluster_2x1);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue8m0_t, sycl::half,                      2, 1, ue8m0_k16_fp16_cluster_2x1);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue8m0_t, sycl::ext::oneapi::bfloat16,     2, 1, ue8m0_k16_bf16_cluster_2x1);

DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue8m0_t, float,                           2, 1, ue8m0_k32_fp32_cluster_2x1);
DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue8m0_t, sycl::half,                      2, 1, ue8m0_k32_fp16_cluster_2x1);
DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue8m0_t, sycl::ext::oneapi::bfloat16,     2, 1, ue8m0_k32_bf16_cluster_2x1);

// ClusterShape<2,2,1>
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue4m3_t, float,                           2, 2, ue4m3_k16_fp32_cluster_2x2);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue4m3_t, sycl::half,                      2, 2, ue4m3_k16_fp16_cluster_2x2);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue4m3_t, sycl::ext::oneapi::bfloat16,     2, 2, ue4m3_k16_bf16_cluster_2x2);

DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue5m3_t, float,                           2, 2, ue5m3_k16_fp32_cluster_2x2);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue5m3_t, sycl::half,                      2, 2, ue5m3_k16_fp16_cluster_2x2);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue5m3_t, sycl::ext::oneapi::bfloat16,     2, 2, ue5m3_k16_bf16_cluster_2x2);

DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue5m3_t, float,                           2, 2, ue5m3_k32_fp32_cluster_2x2);
DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue5m3_t, sycl::half,                      2, 2, ue5m3_k32_fp16_cluster_2x2);
DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue5m3_t, sycl::ext::oneapi::bfloat16,     2, 2, ue5m3_k32_bf16_cluster_2x2);

DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue8m0_t, float,                           2, 2, ue8m0_k16_fp32_cluster_2x2);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue8m0_t, sycl::half,                      2, 2, ue8m0_k16_fp16_cluster_2x2);
DECL_FP4_CLUSTER_CONFIG_K16(cutlass::float_ue8m0_t, sycl::ext::oneapi::bfloat16,     2, 2, ue8m0_k16_bf16_cluster_2x2);

DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue8m0_t, float,                           2, 2, ue8m0_k32_fp32_cluster_2x2);
DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue8m0_t, sycl::half,                      2, 2, ue8m0_k32_fp16_cluster_2x2);
DECL_FP4_CLUSTER_CONFIG_K32(cutlass::float_ue8m0_t, sycl::ext::oneapi::bfloat16,     2, 2, ue8m0_k32_bf16_cluster_2x2);

#undef DECL_FP4_CLUSTER_CONFIG_K16
#undef DECL_FP4_CLUSTER_CONFIG_K32

///////////////////////////////////////////////////////////////////////////////////////////////////
// Usage and entry point
///////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream& print_usage(std::ostream& out) {
  out << "Block-Scaled FP4 GEMM Example\n\n"
      << "Options:\n\n"
      << "  --help                      Displays this usage statement\n\n"
      << "  --m=<int>                   Override M dimension (per-config default if not set)\n"
      << "  --n=<int>                   Override N dimension\n"
      << "  --k=<int>                   Override K dimension\n"
      << "  --coop_sf=<0|1>             Override cooperative SF loading (0=disabled, 1=enabled), only applicable if cluster size > 1\n"
      << "  --config=<name>[,<name>]    Run specific configs (comma-separated); omit to run all 60\n\n"
      << "Available configs ({sf_type}_k{vecsize}_{output_dtype}[_cluster_{MxN}]):\n"
      << "  ue4m3_k16_{fp32,fp16,bf16}\n"
      << "  ue5m3_k{16,32}_{fp32,fp16,bf16}\n"
      << "  ue8m0_k{16,32}_{fp32,fp16,bf16}\n"
      << "  *_cluster_1x2   (ClusterShape<1,2,1>)\n"
      << "  *_cluster_2x1   (ClusterShape<2,1,1>)\n"
      << "  *_cluster_2x2   (ClusterShape<2,2,1>)\n\n"
      << "Examples:\n"
      << "  ./xe4_gemm_blockscaled_fp4                                         # all 15 configs\n"
      << "  ./xe4_gemm_blockscaled_fp4 --m=256 --n=512 --k=3072                # all configs, custom shape\n"
      << "  ./xe4_gemm_blockscaled_fp4 --config=ue4m3_k16_fp32                 # one config\n"
      << "  ./xe4_gemm_blockscaled_fp4 --config=ue4m3_k16_fp32,ue5m3_k32_bf16  # two configs\n"
      << "  ./xe4_gemm_blockscaled_fp4 --config=ue4m3_k16_fp32 --m=256         # one config, custom shape\n";
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
      cfg_ue4m3_k16_fp32, cfg_ue4m3_k16_fp16, cfg_ue4m3_k16_bf16,
      cfg_ue5m3_k16_fp32, cfg_ue5m3_k16_fp16, cfg_ue5m3_k16_bf16,
      cfg_ue5m3_k32_fp32, cfg_ue5m3_k32_fp16, cfg_ue5m3_k32_bf16,
      cfg_ue8m0_k16_fp32, cfg_ue8m0_k16_fp16, cfg_ue8m0_k16_bf16,
      cfg_ue8m0_k32_fp32, cfg_ue8m0_k32_fp16, cfg_ue8m0_k32_bf16,
      // ClusterShape<1,2,1>
      cfg_ue4m3_k16_fp32_cluster_1x2, cfg_ue4m3_k16_fp16_cluster_1x2, cfg_ue4m3_k16_bf16_cluster_1x2,
      cfg_ue5m3_k16_fp32_cluster_1x2, cfg_ue5m3_k16_fp16_cluster_1x2, cfg_ue5m3_k16_bf16_cluster_1x2,
      cfg_ue5m3_k32_fp32_cluster_1x2, cfg_ue5m3_k32_fp16_cluster_1x2, cfg_ue5m3_k32_bf16_cluster_1x2,
      cfg_ue8m0_k16_fp32_cluster_1x2, cfg_ue8m0_k16_fp16_cluster_1x2, cfg_ue8m0_k16_bf16_cluster_1x2,
      cfg_ue8m0_k32_fp32_cluster_1x2, cfg_ue8m0_k32_fp16_cluster_1x2, cfg_ue8m0_k32_bf16_cluster_1x2,
      // ClusterShape<2,1,1>
      cfg_ue4m3_k16_fp32_cluster_2x1, cfg_ue4m3_k16_fp16_cluster_2x1, cfg_ue4m3_k16_bf16_cluster_2x1,
      cfg_ue5m3_k16_fp32_cluster_2x1, cfg_ue5m3_k16_fp16_cluster_2x1, cfg_ue5m3_k16_bf16_cluster_2x1,
      cfg_ue5m3_k32_fp32_cluster_2x1, cfg_ue5m3_k32_fp16_cluster_2x1, cfg_ue5m3_k32_bf16_cluster_2x1,
      cfg_ue8m0_k16_fp32_cluster_2x1, cfg_ue8m0_k16_fp16_cluster_2x1, cfg_ue8m0_k16_bf16_cluster_2x1,
      cfg_ue8m0_k32_fp32_cluster_2x1, cfg_ue8m0_k32_fp16_cluster_2x1, cfg_ue8m0_k32_bf16_cluster_2x1,
      // ClusterShape<2,2,1>
      cfg_ue4m3_k16_fp32_cluster_2x2, cfg_ue4m3_k16_fp16_cluster_2x2, cfg_ue4m3_k16_bf16_cluster_2x2,
      cfg_ue5m3_k16_fp32_cluster_2x2, cfg_ue5m3_k16_fp16_cluster_2x2, cfg_ue5m3_k16_bf16_cluster_2x2,
      cfg_ue5m3_k32_fp32_cluster_2x2, cfg_ue5m3_k32_fp16_cluster_2x2, cfg_ue5m3_k32_bf16_cluster_2x2,
      cfg_ue8m0_k16_fp32_cluster_2x2, cfg_ue8m0_k16_fp16_cluster_2x2, cfg_ue8m0_k16_bf16_cluster_2x2,
      cfg_ue8m0_k32_fp32_cluster_2x2, cfg_ue8m0_k32_fp16_cluster_2x2, cfg_ue8m0_k32_bf16_cluster_2x2
  >(Options::configs, q) ? 0 : 1;
}