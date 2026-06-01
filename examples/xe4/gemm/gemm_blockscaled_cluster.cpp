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

namespace {

// Operand bundle: compile-time operand specification for non-mixed cluster configs.
template <class ElementData_, class ElementSF_, int VS_>
struct Operand {
  using ElementData = ElementData_;
  using ElementSF   = ElementSF_;
  static constexpr int VS = VS_;
};

// Representative default selection: covers all cluster shapes, both data types,
// multiple SF types, both VS sizes, and fp32/bf16 output (10 of 54 configs).
template <class Spec, class D, int CM, int CN>
constexpr bool is_default_cluster_config() {
  using fp4   = cutlass::float_e2m1_t;
  using fp8   = cutlass::float_e4m3_t;
  using ue4m3 = cutlass::float_ue4m3_t;
  using ue5m3 = cutlass::float_ue5m3_t;
  using ue8m0 = cutlass::float_ue8m0_t;

  constexpr bool is_fp4   = std::is_same_v<typename Spec::ElementData, fp4>;
  constexpr bool is_fp8   = std::is_same_v<typename Spec::ElementData, fp8>;
  constexpr bool sf_ue4m3 = std::is_same_v<typename Spec::ElementSF, ue4m3>;
  constexpr bool sf_ue5m3 = std::is_same_v<typename Spec::ElementSF, ue5m3>;
  constexpr bool sf_ue8m0 = std::is_same_v<typename Spec::ElementSF, ue8m0>;
  constexpr int  vs       = Spec::VS;
  constexpr bool out_fp32 = std::is_same_v<D, float>;
  constexpr bool out_bf16 = std::is_same_v<D, sycl::ext::oneapi::bfloat16>;

  // FP4 ue4m3 k16: fp32 on all 3 cluster shapes + bf16 on 1×2
  if constexpr (is_fp4 && sf_ue4m3 && vs == 16 && out_fp32) return true;
  if constexpr (is_fp4 && sf_ue4m3 && vs == 16 && out_bf16 && CM == 1 && CN == 2) return true;
  // FP4 ue5m3 k32: fp32 on 1×2 (VS=32 coverage)
  if constexpr (is_fp4 && sf_ue5m3 && vs == 32 && out_fp32 && CM == 1 && CN == 2) return true;
  // FP4 ue8m0 k16: fp32 on 2×2
  if constexpr (is_fp4 && sf_ue8m0 && vs == 16 && out_fp32 && CM == 2 && CN == 2) return true;
  // FP8: fp32 on all 3 shapes + bf16 on 2×2
  if constexpr (is_fp8) return out_fp32 || (out_bf16 && CM == 2 && CN == 2);

  return false;
}

// Cluster config: non-mixed (same type for A and B), with cluster shape.
template <class Spec, class D, int ClusterM_, int ClusterN_>
struct ClusterConfig {
  using ElementA           = typename Spec::ElementData;
  using ElementB           = typename Spec::ElementData;
  using ElementSFA         = typename Spec::ElementSF;
  using ElementSFB         = typename Spec::ElementSF;
  static constexpr int SFVecSizeA = Spec::VS;
  static constexpr int SFVecSizeB = Spec::VS;
  using ElementD           = D;
  using ElementAccumulator = float;
  using LayoutA            = cutlass::layout::RowMajor;
  using LayoutB            = cutlass::layout::RowMajor;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 512, 1024, 1};
  using CtaTileShape_MNK   = Shape<_128, _256, _256>;
  using ClusterShape_MNK   = Shape<cute::Int<ClusterM_>, cute::Int<ClusterN_>, _1>;
  static constexpr int  PipelineStages       = 4;
  static constexpr bool EnableCooperativeSF  = false;
  static constexpr bool run_by_default       = is_default_cluster_config<Spec, D, ClusterM_, ClusterN_>();
  static const char*    Name;
};

template <class Spec, class D, int CM, int CN>
const char* ClusterConfig<Spec, D, CM, CN>::Name = [] {
  static const std::string s = []{
    std::string name = std::string(type_name<typename Spec::ElementData>())
      + "_" + sf_tag<typename Spec::ElementSF>() + "k" + std::to_string(Spec::VS)
      + "_" + type_name<D>()
      + "_cluster_" + std::to_string(CM) + "x" + std::to_string(CN);
    return name;
  }();
  return s.c_str();
}();

// FP4 operand variants
using FP4_ue4m3_k16 = Operand<cutlass::float_e2m1_t, cutlass::float_ue4m3_t, 16>;
using FP4_ue5m3_k16 = Operand<cutlass::float_e2m1_t, cutlass::float_ue5m3_t, 16>;
using FP4_ue5m3_k32 = Operand<cutlass::float_e2m1_t, cutlass::float_ue5m3_t, 32>;
using FP4_ue8m0_k16 = Operand<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 16>;
using FP4_ue8m0_k32 = Operand<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32>;

// FP8 operand variant
using FP8_ue8m0_k32 = Operand<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32>;

// Output types
using bf16_t = sycl::ext::oneapi::bfloat16;
using fp16_t = sycl::half;

// -- FP4 cluster configs: 5 SF variants × 3 output types × 3 cluster shapes = 45 configs --

// Cluster 1×2
using fp4_ue4m3k16_fp32_c1x2  = ClusterConfig<FP4_ue4m3_k16, float,  1, 2>;
using fp4_ue4m3k16_fp16_c1x2  = ClusterConfig<FP4_ue4m3_k16, fp16_t, 1, 2>;
using fp4_ue4m3k16_bf16_c1x2  = ClusterConfig<FP4_ue4m3_k16, bf16_t, 1, 2>;
using fp4_ue5m3k16_fp32_c1x2  = ClusterConfig<FP4_ue5m3_k16, float,  1, 2>;
using fp4_ue5m3k16_fp16_c1x2  = ClusterConfig<FP4_ue5m3_k16, fp16_t, 1, 2>;
using fp4_ue5m3k16_bf16_c1x2  = ClusterConfig<FP4_ue5m3_k16, bf16_t, 1, 2>;
using fp4_ue5m3k32_fp32_c1x2  = ClusterConfig<FP4_ue5m3_k32, float,  1, 2>;
using fp4_ue5m3k32_fp16_c1x2  = ClusterConfig<FP4_ue5m3_k32, fp16_t, 1, 2>;
using fp4_ue5m3k32_bf16_c1x2  = ClusterConfig<FP4_ue5m3_k32, bf16_t, 1, 2>;
using fp4_ue8m0k16_fp32_c1x2  = ClusterConfig<FP4_ue8m0_k16, float,  1, 2>;
using fp4_ue8m0k16_fp16_c1x2  = ClusterConfig<FP4_ue8m0_k16, fp16_t, 1, 2>;
using fp4_ue8m0k16_bf16_c1x2  = ClusterConfig<FP4_ue8m0_k16, bf16_t, 1, 2>;
using fp4_ue8m0k32_fp32_c1x2  = ClusterConfig<FP4_ue8m0_k32, float,  1, 2>;
using fp4_ue8m0k32_fp16_c1x2  = ClusterConfig<FP4_ue8m0_k32, fp16_t, 1, 2>;
using fp4_ue8m0k32_bf16_c1x2  = ClusterConfig<FP4_ue8m0_k32, bf16_t, 1, 2>;

// Cluster 2×1
using fp4_ue4m3k16_fp32_c2x1  = ClusterConfig<FP4_ue4m3_k16, float,  2, 1>;
using fp4_ue4m3k16_fp16_c2x1  = ClusterConfig<FP4_ue4m3_k16, fp16_t, 2, 1>;
using fp4_ue4m3k16_bf16_c2x1  = ClusterConfig<FP4_ue4m3_k16, bf16_t, 2, 1>;
using fp4_ue5m3k16_fp32_c2x1  = ClusterConfig<FP4_ue5m3_k16, float,  2, 1>;
using fp4_ue5m3k16_fp16_c2x1  = ClusterConfig<FP4_ue5m3_k16, fp16_t, 2, 1>;
using fp4_ue5m3k16_bf16_c2x1  = ClusterConfig<FP4_ue5m3_k16, bf16_t, 2, 1>;
using fp4_ue5m3k32_fp32_c2x1  = ClusterConfig<FP4_ue5m3_k32, float,  2, 1>;
using fp4_ue5m3k32_fp16_c2x1  = ClusterConfig<FP4_ue5m3_k32, fp16_t, 2, 1>;
using fp4_ue5m3k32_bf16_c2x1  = ClusterConfig<FP4_ue5m3_k32, bf16_t, 2, 1>;
using fp4_ue8m0k16_fp32_c2x1  = ClusterConfig<FP4_ue8m0_k16, float,  2, 1>;
using fp4_ue8m0k16_fp16_c2x1  = ClusterConfig<FP4_ue8m0_k16, fp16_t, 2, 1>;
using fp4_ue8m0k16_bf16_c2x1  = ClusterConfig<FP4_ue8m0_k16, bf16_t, 2, 1>;
using fp4_ue8m0k32_fp32_c2x1  = ClusterConfig<FP4_ue8m0_k32, float,  2, 1>;
using fp4_ue8m0k32_fp16_c2x1  = ClusterConfig<FP4_ue8m0_k32, fp16_t, 2, 1>;
using fp4_ue8m0k32_bf16_c2x1  = ClusterConfig<FP4_ue8m0_k32, bf16_t, 2, 1>;

// Cluster 2×2
using fp4_ue4m3k16_fp32_c2x2  = ClusterConfig<FP4_ue4m3_k16, float,  2, 2>;
using fp4_ue4m3k16_fp16_c2x2  = ClusterConfig<FP4_ue4m3_k16, fp16_t, 2, 2>;
using fp4_ue4m3k16_bf16_c2x2  = ClusterConfig<FP4_ue4m3_k16, bf16_t, 2, 2>;
using fp4_ue5m3k16_fp32_c2x2  = ClusterConfig<FP4_ue5m3_k16, float,  2, 2>;
using fp4_ue5m3k16_fp16_c2x2  = ClusterConfig<FP4_ue5m3_k16, fp16_t, 2, 2>;
using fp4_ue5m3k16_bf16_c2x2  = ClusterConfig<FP4_ue5m3_k16, bf16_t, 2, 2>;
using fp4_ue5m3k32_fp32_c2x2  = ClusterConfig<FP4_ue5m3_k32, float,  2, 2>;
using fp4_ue5m3k32_fp16_c2x2  = ClusterConfig<FP4_ue5m3_k32, fp16_t, 2, 2>;
using fp4_ue5m3k32_bf16_c2x2  = ClusterConfig<FP4_ue5m3_k32, bf16_t, 2, 2>;
using fp4_ue8m0k16_fp32_c2x2  = ClusterConfig<FP4_ue8m0_k16, float,  2, 2>;
using fp4_ue8m0k16_fp16_c2x2  = ClusterConfig<FP4_ue8m0_k16, fp16_t, 2, 2>;
using fp4_ue8m0k16_bf16_c2x2  = ClusterConfig<FP4_ue8m0_k16, bf16_t, 2, 2>;
using fp4_ue8m0k32_fp32_c2x2  = ClusterConfig<FP4_ue8m0_k32, float,  2, 2>;
using fp4_ue8m0k32_fp16_c2x2  = ClusterConfig<FP4_ue8m0_k32, fp16_t, 2, 2>;
using fp4_ue8m0k32_bf16_c2x2  = ClusterConfig<FP4_ue8m0_k32, bf16_t, 2, 2>;

// -- FP8 cluster configs: 1 SF variant × 3 output types × 3 cluster shapes = 9 configs --

// Cluster 1×2
using fp8_ue8m0k32_fp32_c1x2  = ClusterConfig<FP8_ue8m0_k32, float,  1, 2>;
using fp8_ue8m0k32_fp16_c1x2  = ClusterConfig<FP8_ue8m0_k32, fp16_t, 1, 2>;
using fp8_ue8m0k32_bf16_c1x2  = ClusterConfig<FP8_ue8m0_k32, bf16_t, 1, 2>;

// Cluster 2×1
using fp8_ue8m0k32_fp32_c2x1  = ClusterConfig<FP8_ue8m0_k32, float,  2, 1>;
using fp8_ue8m0k32_fp16_c2x1  = ClusterConfig<FP8_ue8m0_k32, fp16_t, 2, 1>;
using fp8_ue8m0k32_bf16_c2x1  = ClusterConfig<FP8_ue8m0_k32, bf16_t, 2, 1>;

// Cluster 2×2
using fp8_ue8m0k32_fp32_c2x2  = ClusterConfig<FP8_ue8m0_k32, float,  2, 2>;
using fp8_ue8m0k32_fp16_c2x2  = ClusterConfig<FP8_ue8m0_k32, fp16_t, 2, 2>;
using fp8_ue8m0k32_bf16_c2x2  = ClusterConfig<FP8_ue8m0_k32, bf16_t, 2, 2>;

} // anonymous namespace

void Options::parse(int argc, char** argv) {
  cutlass::CommandLine cmd(argc, const_cast<char const**>(argv));
  help = cmd.check_cmd_line_flag("help");
  run_all = cmd.check_cmd_line_flag("run-all");

  int val;
  if (cmd.check_cmd_line_flag("m")) { cmd.get_cmd_line_argument("m", val); m = val; }
  if (cmd.check_cmd_line_flag("n")) { cmd.get_cmd_line_argument("n", val); n = val; }
  if (cmd.check_cmd_line_flag("k")) { cmd.get_cmd_line_argument("k", val); k = val; }
  if (cmd.check_cmd_line_flag("l")) { cmd.get_cmd_line_argument("l", val); l = val; }
  if (cmd.check_cmd_line_flag("coop_sf")) {
    int csf = 0;
    cmd.get_cmd_line_argument("coop_sf", csf);
    coop_sf = (csf != 0);
  }
  std::string config_str;
  cmd.get_cmd_line_argument("config", config_str);
  if (!config_str.empty()) {
    std::istringstream iss(config_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
      if (!token.empty()) configs.push_back(token);
    }
  }
}

int main(int argc, char** argv) {
  Options::parse(argc, argv);

  if (Options::help) {
    std::cout << "Block-Scaled Cluster GEMM\n\n"
              << "  --help / --m --n --k --l=<int>\n"
              << "  --config=<name>[,<name>...]  Run specific configs (comma-separated)\n"
              << "  --run-all                    Run all 54 configs (default: representative subset)\n"
              << "  --coop_sf=0|1                Override cooperative SF loading for cluster configs\n\n"
              << "Total: 54 configs (45 FP4 + 9 FP8), default runs representative subset\n";
    return 0;
  }

  sycl::queue q;

  return run_configs<
    // FP4 cluster 1×2
    fp4_ue4m3k16_fp32_c1x2, fp4_ue4m3k16_fp16_c1x2, fp4_ue4m3k16_bf16_c1x2,
    fp4_ue5m3k16_fp32_c1x2, fp4_ue5m3k16_fp16_c1x2, fp4_ue5m3k16_bf16_c1x2,
    fp4_ue5m3k32_fp32_c1x2, fp4_ue5m3k32_fp16_c1x2, fp4_ue5m3k32_bf16_c1x2,
    fp4_ue8m0k16_fp32_c1x2, fp4_ue8m0k16_fp16_c1x2, fp4_ue8m0k16_bf16_c1x2,
    fp4_ue8m0k32_fp32_c1x2, fp4_ue8m0k32_fp16_c1x2, fp4_ue8m0k32_bf16_c1x2,
    // FP4 cluster 2×1
    fp4_ue4m3k16_fp32_c2x1, fp4_ue4m3k16_fp16_c2x1, fp4_ue4m3k16_bf16_c2x1,
    fp4_ue5m3k16_fp32_c2x1, fp4_ue5m3k16_fp16_c2x1, fp4_ue5m3k16_bf16_c2x1,
    fp4_ue5m3k32_fp32_c2x1, fp4_ue5m3k32_fp16_c2x1, fp4_ue5m3k32_bf16_c2x1,
    fp4_ue8m0k16_fp32_c2x1, fp4_ue8m0k16_fp16_c2x1, fp4_ue8m0k16_bf16_c2x1,
    fp4_ue8m0k32_fp32_c2x1, fp4_ue8m0k32_fp16_c2x1, fp4_ue8m0k32_bf16_c2x1,
    // FP4 cluster 2×2
    fp4_ue4m3k16_fp32_c2x2, fp4_ue4m3k16_fp16_c2x2, fp4_ue4m3k16_bf16_c2x2,
    fp4_ue5m3k16_fp32_c2x2, fp4_ue5m3k16_fp16_c2x2, fp4_ue5m3k16_bf16_c2x2,
    fp4_ue5m3k32_fp32_c2x2, fp4_ue5m3k32_fp16_c2x2, fp4_ue5m3k32_bf16_c2x2,
    fp4_ue8m0k16_fp32_c2x2, fp4_ue8m0k16_fp16_c2x2, fp4_ue8m0k16_bf16_c2x2,
    fp4_ue8m0k32_fp32_c2x2, fp4_ue8m0k32_fp16_c2x2, fp4_ue8m0k32_bf16_c2x2,
    // FP8 cluster 1×2
    fp8_ue8m0k32_fp32_c1x2, fp8_ue8m0k32_fp16_c1x2, fp8_ue8m0k32_bf16_c1x2,
    // FP8 cluster 2×1
    fp8_ue8m0k32_fp32_c2x1, fp8_ue8m0k32_fp16_c2x1, fp8_ue8m0k32_bf16_c2x1,
    // FP8 cluster 2×2
    fp8_ue8m0k32_fp32_c2x2, fp8_ue8m0k32_fp16_c2x2, fp8_ue8m0k32_bf16_c2x2
  >(Options::configs, q) ? 0 : 1;
}
