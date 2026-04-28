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

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "testFixture.hpp"
using namespace cute;

// Concrete adapter class to access testFixture::default_run_gemm from main()
class GemmClusterPerfAdapter : public testFixture {
 public:
  template <typename Config>
  void run_gemm() {
    this->template default_run_gemm<Config>();
  }

 private:
  // Implement pure virtual methods from testFixture as no-ops
  void runTest() override {}
  
  // Satisfy TestWithParam pure virtual method
  void TestBody() override {}
};

struct ClusterPerfConfig_1 {
  using ElementA = fp16;
  using ElementB = fp16;
  using ElementC = void;
  using ElementD = fp16;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using CtaTileShape_MNK = Shape<_256, _512, _128>;
  using CtaNum_MN = Shape<_2, _2>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  static constexpr int StagesA = 2;
  static constexpr bool is_persistent = true;
  static constexpr auto activation_type = ActivationType::None;
  static constexpr auto operationC_type = OperationCType::None;
  inline static cute::array<int, 4> ProblemShape_MNKL = {2048, 2048, 2048, 1};
};

struct ClusterPerfConfig_2 {
  using ElementA = fp16;
  using ElementB = fp16;
  using ElementC = void;
  using ElementD = fp16;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using CtaTileShape_MNK = Shape<_256, _512, _128>;
  using CtaNum_MN = Shape<_1, _1>;
  using ClusterShape_MNK = Shape<_2, _2, _1>;

  static constexpr int StagesA = 2;
  static constexpr bool is_persistent = true;
  static constexpr auto activation_type = ActivationType::None;
  static constexpr auto operationC_type = OperationCType::None;
  inline static cute::array<int, 4> ProblemShape_MNKL = {2048, 2048, 2048, 1};
};

struct ClusterPerfConfig_3 {
  using ElementA = fp16;
  using ElementB = fp16;
  using ElementC = void;
  using ElementD = fp16;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using CtaTileShape_MNK = Shape<_256, _512, _128>;
  using CtaNum_MN = Shape<_1, _1>;
  using ClusterShape_MNK = Shape<_2, _2, _1>;

  static constexpr int StagesA = 2;
  static constexpr bool is_persistent = true;
  static constexpr auto activation_type = ActivationType::None;
  static constexpr auto operationC_type = OperationCType::None;
  inline static cute::array<int, 4> ProblemShape_MNKL = {2048, 2112, 4163, 1};
};

struct CliOptions {
  int m12 = 2048;
  int n12 = 2048;
  int k12 = 2048;

  int m3 = 2048;
  int n3 = 2112;
  int k3 = 4163;

  int iterations = 2;
};

bool parse_positive_int(char const* text, int& value) {
  if (!text) {
    return false;
  }

  char* end = nullptr;
  long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed <= 0 ||
      parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    return false;
  }

  value = static_cast<int>(parsed);
  return true;
}

bool parse_cli_options(int argc, char** argv, CliOptions& options) {
  if (argc == 1) {
    return true;
  }

  // Usage:
  //   gemm_cluster_simple_perf [iterations [M12 N12 K12 [M3 N3 K3]]]
  // Defaults:
  //   config1/config2: 2048 2048 2048
  //   config3:         2048 2112 4163
  //   iterations:      2
  if (argc != 2 && argc != 5 && argc != 8) {
    return false;
  }

  if (!parse_positive_int(argv[1], options.iterations)) {
    return false;
  }

  if (argc >= 5) {
    if (!parse_positive_int(argv[2], options.m12) || !parse_positive_int(argv[3], options.n12) ||
        !parse_positive_int(argv[4], options.k12)) {
      return false;
    }
  }

  if (argc == 8) {
    if (!parse_positive_int(argv[5], options.m3) || !parse_positive_int(argv[6], options.n3) ||
        !parse_positive_int(argv[7], options.k3)) {
      return false;
    }
  }

  return true;
}

template <typename Config>
void set_problem_shape(int m, int n, int k) {
  Config::ProblemShape_MNKL = {m, n, k, 1};
}

template <typename Config>
double compute_gflops(double ms_per_iter) {
  double m = static_cast<double>(Config::ProblemShape_MNKL[0]);
  double n = static_cast<double>(Config::ProblemShape_MNKL[1]);
  double k = static_cast<double>(Config::ProblemShape_MNKL[2]);
  return (2.0 * m * n * k) / (ms_per_iter * 1.0e6);
}

template <typename Config>
void run_config_perf(int iterations, char const* config_name) {
  std::cout << "Running " << config_name << " for " << iterations << " iterations" << std::endl;

  double total_ms = 0.0;
  for (int iter = 0; iter < iterations; ++iter) {
    auto start = std::chrono::high_resolution_clock::now();
    GemmClusterPerfAdapter adapter;
    adapter.run_gemm<Config>();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    total_ms += elapsed_ms;
    std::cout << "config=" << config_name << " iter=" << (iter + 1) << " elapsed_ms=" << elapsed_ms
              << std::endl;
  }

  double avg_ms = total_ms / static_cast<double>(iterations);
  double avg_gflops = compute_gflops<Config>(avg_ms);

  std::cout << "config=" << config_name << " avg_elapsed_ms=" << avg_ms << std::endl;
  std::cout << "config=" << config_name << " avg_gflops=" << avg_gflops << std::endl;
}

int main(int argc, char** argv) {
  CliOptions options;
  if (!parse_cli_options(argc, argv, options)) {
    std::cerr << "Usage: " << argv[0] << " [iterations [M12 N12 K12 [M3 N3 K3]]]" << std::endl;
    std::cerr << "Defaults: config1/config2=(2048,2048,2048), config3=(2048,2112,4163), iterations=2"
              << std::endl;
    return 1;
  }

  set_problem_shape<ClusterPerfConfig_1>(options.m12, options.n12, options.k12);
  set_problem_shape<ClusterPerfConfig_2>(options.m12, options.n12, options.k12);
  set_problem_shape<ClusterPerfConfig_3>(options.m3, options.n3, options.k3);

  run_config_perf<ClusterPerfConfig_1>(options.iterations, "ClusterPerfConfig_1");
  run_config_perf<ClusterPerfConfig_2>(options.iterations, "ClusterPerfConfig_2");
  run_config_perf<ClusterPerfConfig_3>(options.iterations, "ClusterPerfConfig_3");

  return 0;
}
