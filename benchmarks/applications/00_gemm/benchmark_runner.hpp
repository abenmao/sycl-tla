/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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

#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cute/tensor.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/initialize_block.hpp"

#include "../common.hpp"
#include "benchmark_cache_flush.hpp"
#include "benchmark_verify.hpp"
#include <benchmark/benchmark.h>
#include <chrono>

using namespace cute;

// ---------------------------------------------------------------------------
// GEMM benchmark naming convention
// ---------------------------------------------------------------------------
// Benchmark names registered via CUTLASS_CREATE_GEMM_BENCHMARK (see the
// `benchmarks_sycl_types_*.hpp` files) follow a fixed pattern, e.g.:
//
//   Gemm_BF16BF16FP32BF16FP32_RRR_WG128x192x64_SG16x48x64
//   BLockScalingGemm_E4M3E4M3FP32BF16FP32_RRR_WG16x256x256_SG16x16x256_GS32
//   BLockScalingGemmNonNative_E2M1E2M1FP32FP32FP32_RCR_WG512x256x128_SG64x64x128_GS32
//
// Reading left to right:
//   - "Gemm" / "BLockScalingGemm" prefix (followed by "_")
//       Plain GEMM, or an MX/OCP microscaling block-scaled GEMM
//       (see the `GroupSize`/`GS` note below).
//   - "NonNative" (block-scaled only, optional)
//       Uses software dequantization of the scale factors before the MMA,
//       as opposed to the default "native" path where the DPAS instruction
//       consumes the scale factors directly.
//   - Element/type sequence, e.g. "E4M3E4M3FP32BF16FP32"
//       Five back-to-back type tags with no separator:
//         ElementA, ElementB, ElementC, ElementD, AccType (accumulator /
//         epilogue compute type). E.g. E4M3(A) E4M3(B) FP32(C) BF16(D) FP32(Acc).
//   - "_RRR" (or "_RCR", etc.) layout tag
//       One letter per operand, in order A / B / C: 'R' = RowMajor,
//       'C' = ColumnMajor. E.g. "_RCR_" means A=RowMajor, B=ColumnMajor,
//       C=RowMajor. A and C are RowMajor in every benchmark here; only the
//       B letter actually varies.
//   - "_WGmxnxk" workgroup tile shape
//       WG_M x WG_N x WG_K: the tile computed by one workgroup/threadgroup.
//   - "_SGmxnxk" subgroup tile shape
//       SG_M x SG_N x SG_K: the tile computed by one subgroup within the
//       workgroup (WG_M/SG_M and WG_N/SG_N give the subgroup grid).
//   - "_GSn" group size (block-scaled only)
//       The MX/OCP microscaling group size: the number of contiguous
//       K-elements that share one scale factor (defaults to 32; some
//       benchmarks use 128). See MainloopIntelXeXMX16BlockScaled's
//       `GroupSize` template parameter in dispatch_policy.hpp.

// The default benchmark name is the SYCL target passed to CMake via
// -DDPCPP_SYCL_TARGET. Falls back to "GEMM" when unset/empty.
#ifndef CUTLASS_BENCHMARK_SYCL_TARGET
#define CUTLASS_BENCHMARK_SYCL_TARGET ""
#endif

namespace cutlass::benchmark {

static inline std::string default_bm_name() {
  std::string const target = CUTLASS_BENCHMARK_SYCL_TARGET;
  return target.empty() ? std::string("GEMM") : target;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class T, class = void>
static constexpr auto is_blocked_scaled = false;
template <class T>
static constexpr auto is_blocked_scaled<T, cute::void_t<typename T::ElementScaleA, typename T::ElementScaleB>> = true;

// Block-scaled mainloops (MainloopIntelXeXMX16BlockScaled<Stages, GroupSize>) expose
// the scale group size as CollectiveMainloop::GroupK. Default to 32 otherwise.
template <class T, class = void>
struct GroupKType {
  static constexpr int value = 32;
};
template <class T>
struct GroupKType<T, cute::void_t<decltype(T::GroupK)>> {
  static constexpr int value = static_cast<int>(T::GroupK);
};

template <class T, class = void>
struct ElementScaleAType {
  using type = int;
};
template <class T>
struct ElementScaleAType<T, cute::void_t<typename T::ElementScaleA>> {
  using type = typename T::ElementScaleA;
};

template <class T, class = void>
struct ElementScaleBType {
  using type = int;
};
template <class T>
struct ElementScaleBType<T, cute::void_t<typename T::ElementScaleB>> {
  using type = typename T::ElementScaleB;
};

template <class T, class = void>
struct StrideScaleAType {
  using type = int;
};
template <class T>
struct StrideScaleAType<T, cute::void_t<typename T::StrideScaleA>> {
  using type = typename T::StrideScaleA;
};

template <class T, class = void>
struct StrideScaleBType {
  using type = int;
};
template <class T>
struct StrideScaleBType<T, cute::void_t<typename T::StrideScaleB>> {
  using type = typename T::StrideScaleB;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct GEMMOptions {

  bool error;

  int m, n, k, l;
  float alpha, beta;
  std::string bm_name;
  // Selects how correctness is checked: on the device with
  // reference::device::GemmComplex, on the host with reference::host::GemmComplexMkl,
  // or skipped entirely.
  VerifyMode verify_mode;
  CacheFlushMode cache_flush_mode;
  int iterations;
  int warmup;

  GEMMOptions():
          error(false),
          m(5120), n(4096), k(4096), l(1),
          alpha(1.f), beta(0.f),
          iterations(CUTLASS_BENCHMARK_DEFAULT_ITERATIONS),
          warmup(CUTLASS_WARMUP_DEFAULT_ITERATIONS),
          bm_name(default_bm_name()),
          verify_mode(VerifyMode::None),
          cache_flush_mode(CacheFlushMode::FlushKernel)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    CommandLine cmd(argc, args);

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("bm_name", bm_name, default_bm_name());

    // Parse verification mode. Default to device verification on real hardware,
    // but skip verification on the simulator where device verification +
    // warmup are too time-consuming.
#ifdef SYCLTLA_TARGET_XESIM
    std::string default_verify = "none";
#else
    std::string default_verify = "device";
#endif
    std::string verify_str = default_verify;
    cmd.get_cmd_line_argument("verify", verify_str, default_verify);
    if (verify_str == "none") {
      verify_mode = VerifyMode::None;
    } else if (verify_str == "device") {
      verify_mode = VerifyMode::Device;
    } else if (verify_str == "host") {
      verify_mode = VerifyMode::Host;
    } else {
      std::cerr << "Invalid verify mode or mode wasn't defined, using default None" << std::endl;
      verify_mode = VerifyMode::None;
    }

    cmd.get_cmd_line_argument("iterations", iterations, CUTLASS_BENCHMARK_DEFAULT_ITERATIONS);
    cmd.get_cmd_line_argument("warmup", warmup, CUTLASS_WARMUP_DEFAULT_ITERATIONS);

    std::string cache_flush_str = "kernel";
    cmd.get_cmd_line_argument("cache_flush", cache_flush_str, std::string("kernel"));
    if (cache_flush_str == "rotate") {
      cache_flush_mode = CacheFlushMode::RotateBuffers;
    } else {
      cache_flush_mode = CacheFlushMode::FlushKernel;
    }
  }

  std::string benchmark_name() const {
    std::stringstream full_name;
    full_name << bm_name << "/";
    std::string const test_name_suffix = "MNKL_" + std::to_string(m) + "x" +
                                   std::to_string(n) + "x" +
                                   std::to_string(k) + "x" +
                                   std::to_string(l);
    full_name << test_name_suffix;

    return full_name.str();
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class GemmConfiguration>
struct BenchmarkRunnerGemm {

  using Gemm = typename GemmConfiguration::Gemm;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementD = typename Gemm::ElementD;

  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementMMAVerify = float;

  using CollectiveMainloop = typename Gemm::GemmKernel::CollectiveMainloop;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementMma = typename CollectiveMainloop::TiledMma::ValTypeA;

  using ElementScaleA = typename ElementScaleAType<CollectiveMainloop>::type;
  using ElementScaleB = typename ElementScaleBType<CollectiveMainloop>::type;
  using StrideScaleA = typename StrideScaleAType<CollectiveMainloop>::type;
  using StrideScaleB = typename StrideScaleBType<CollectiveMainloop>::type;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  static constexpr int GROUP_SIZE = GroupKType<CollectiveMainloop>::value;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  StrideScaleA stride_SA;
  StrideScaleB stride_SB;


  uint64_t seed = 0;

  DeviceAllocation<ElementA> block_A;
  DeviceAllocation<ElementB> block_B;
  DeviceAllocation<ElementC> block_C;
  DeviceAllocation<ElementOutput> block_D;
  DeviceAllocation<ElementOutput> block_ref_D;

  cutlass::DeviceAllocation<ElementScaleA> block_scaleA;
  cutlass::DeviceAllocation<ElementScaleB> block_scaleB;
  cutlass::DeviceAllocation<ElementMMAVerify> block_A_dq; // Dequantized copy of A for validation
  cutlass::DeviceAllocation<ElementMMAVerify> block_B_dq; // Dequantized copy of B for validation

  CacheFlushHelper<ElementA, ElementB, ElementC, ElementScaleA, ElementScaleB,
                   is_blocked_scaled<CollectiveMainloop>> cache_flush_;

  BenchmarkRunnerGemm() : seed(0) {};

  //
  // Methods
  //

  template <class Element>
  bool initialize_scale(
    cutlass::DeviceAllocation<Element>& block) {
    const float elt_max_f = float(cutlass::platform::numeric_limits<Element>::max());
    // Need to fix max_dequant_val and min_dequant_val?
    const float max_dequant_val = elt_max_f * 0.25f;
    const float min_dequant_val = 0.5f;
    const float scale_max = max_dequant_val / elt_max_f;
    const float scale_min = min_dequant_val / elt_max_f;
    cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, Element(scale_max), Element(scale_min));
    return true;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(::benchmark::State& state, const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    
    const int scale_k = cute::ceil_div(K, GROUP_SIZE);
    auto shape_A = cute::make_shape(M, K, L);
    auto shape_B = cute::make_shape(N, K, L);
    auto shape_CD = cute::make_shape(M, N, L);
    auto shape_scale_A = cute::make_shape(M, scale_k, L);
    auto shape_scale_B = cute::make_shape(N, scale_k, L);    

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    if constexpr (is_blocked_scaled<CollectiveMainloop>) {
      stride_SA = cutlass::make_cute_packed_stride(StrideScaleA{}, shape_scale_A);
      stride_SB = cutlass::make_cute_packed_stride(StrideScaleB{}, shape_scale_B);
    }

    block_A.reset(static_cast<std::size_t>(M) * K * L);
    block_A_dq.reset(static_cast<std::size_t>(M) * K * L);
    block_B.reset(static_cast<std::size_t>(K) * N * L);
    block_B_dq.reset(static_cast<std::size_t>(K) * N * L);
    block_C.reset(static_cast<std::size_t>(M) * N * L);
    block_D.reset(static_cast<std::size_t>(M) * N * L);
    block_ref_D.reset(static_cast<std::size_t>(M) * N * L);

    if constexpr (is_blocked_scaled<CollectiveMainloop>) {
      block_scaleA.reset(static_cast<std::size_t>(scale_k) * L * M);
      block_scaleB.reset(static_cast<std::size_t>(scale_k) * L * N);
    }

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);

    cutlass::benchmark::convert_dtype<ElementA, ElementMMAVerify, BenchmarkRunnerGemm>(
        block_A,
        block_A_dq
    );
    cutlass::benchmark::convert_dtype<ElementB, ElementMMAVerify, BenchmarkRunnerGemm>(
        block_B,
        block_B_dq
    );

    if constexpr (is_blocked_scaled<CollectiveMainloop>) {
      initialize_scale(block_scaleA);
      initialize_scale(block_scaleB);
      auto layout_A = make_layout(shape_A, stride_A);
      auto layout_B = make_layout(shape_B, stride_B);
      auto layout_scale_A = make_layout(shape_scale_A, stride_SA);
      auto layout_scale_B = make_layout(shape_scale_B, stride_SB);
      apply_scale(block_A_dq.get(), block_A.get(), layout_A, block_scaleA.get(),  layout_scale_A);
      apply_scale(block_B_dq.get(), block_B.get(), layout_B, block_scaleB.get(),  layout_scale_B);
    }
  }

  void run(::benchmark::State& state, const GEMMOptions& options, const KernelHardwareInfo& hw_info) {
    auto wall_start = std::chrono::steady_clock::now();
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(state, problem_size);

    {
      auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
      auto [M_cf, N_cf, K_cf, L_cf] = problem_shape_MNKL;
      const int scale_k_cf = cute::ceil_div(K_cf, GROUP_SIZE);
      cache_flush_.initialize(
          options.cache_flush_mode, options.iterations,
          M_cf, N_cf, K_cf, L_cf, scale_k_cf, seed,
          block_A.get(), block_B.get(), block_C.get(),
          block_scaleA.get(), block_scaleB.get(),
          [this](auto& blk) { initialize_scale(blk); });
    }

    typename Gemm::GemmKernel::Arguments arguments = GemmConfiguration::defaultArguments();
    arguments.mode = gemm::GemmUniversalMode::kGemm;
    arguments.problem_shape = problem_size;

    if constexpr (!is_blocked_scaled<CollectiveMainloop>) {
      arguments.mainloop = {cache_flush_.ptr_A(), stride_A, cache_flush_.ptr_B(), stride_B};
    } else {
      arguments.mainloop = {cache_flush_.ptr_A(), stride_A, cache_flush_.ptr_B(), stride_B,
        cache_flush_.ptr_SA(), stride_SA, cache_flush_.ptr_SB(), stride_SB};
    }


    arguments.epilogue = {{ElementAccumulator(options.alpha), ElementAccumulator(options.beta)}, cache_flush_.ptr_C(), stride_C, block_D.get(), stride_D};
    
    arguments.hw_info = hw_info;

    Gemm gemm_op;

    device_memory::allocation<uint8_t> workspace;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    try {
      workspace.reset(workspace_size);
    } catch (std::exception const &e) {
      state.SkipWithError(e.what());
    }

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess)
      state.SkipWithError("GEMM unable to implement given args.");

    if (gemm_op.initialize(arguments, workspace.get()) != cutlass::Status::kSuccess)
      state.SkipWithError("GEMM failed to initialize.");

    if (state.error_occurred()) return;

    auto warmups = options.warmup;
    if (options.verify_mode != VerifyMode::None && options.warmup == 0) {
      warmups = 1;
    }
    // Warmup runs
    for (int i = 0; i < warmups; ++i) {
      gemm_op.run();
    }
    compat::wait();

    std::stringstream extra_label;
    if (options.verify_mode != VerifyMode::None) {
      bool passed = cutlass::benchmark::run_verify<ProblemShapeType, ElementCompute,
          ElementC, ElementOutput, ElementAccumulator, ElementMMAVerify,
          LayoutA, LayoutB, LayoutC, LayoutD>(
          options.verify_mode, problem_size,
          ElementCompute(options.alpha), ElementCompute(options.beta),
          block_A_dq, block_B_dq, block_C, block_D, block_ref_D, extra_label);
      if (!passed) {
        state.SkipWithError("Disposition Failed.");
      }
    }

    state.counters["m"] = options.m;
    state.counters["n"] = options.n;
    state.counters["k"] = options.k;
    state.counters["l"] = options.l;
    state.counters["alpha"] = options.alpha;
    state.counters["beta"] = options.beta;

    if constexpr (cute::size<0>(StrideA{}) == 1) {
      extra_label << "layoutA=ColumnMajor ";
    } else if constexpr (cute::size<1>(StrideA{}) == 1) {
      extra_label << "layoutA=RowMajor ";
    }
    if constexpr (cute::size<0>(StrideB{}) == 1) {
      extra_label << "layoutB=RowMajor ";
    } else if constexpr (cute::size<1>(StrideB{}) == 1) {
      extra_label << "layoutB=ColumnMajor ";
    }
    if constexpr (cute::size<0>(StrideC{}) == 1) {
      extra_label << "layoutC=ColumnMajor ";
    } else if constexpr (cute::size<1>(StrideC{}) == 1) {
      extra_label << "layoutC=RowMajor ";
    }
    state.SetLabel(extra_label.str());

    auto gflop = 2.0 * options.m * options.n * options.k * options.l * 1e-9;

    // Compatible with data types smaller than 8 bits here
    constexpr double bits_per_byte = static_cast<double>(sizeof_bits_v<char>);
    constexpr double sizeof_a = sizeof_bits_v<ElementA> / bits_per_byte;
    constexpr double sizeof_b = sizeof_bits_v<ElementB> / bits_per_byte;
    constexpr double sizeof_o = sizeof_bits_v<ElementOutput> / bits_per_byte;
    double scale_bytes_transferred = 0.0;
    if constexpr (is_blocked_scaled<CollectiveMainloop>) {
      constexpr double sizeof_scale_a = sizeof_bits_v<ElementScaleA> / bits_per_byte;
      constexpr double sizeof_scale_b = sizeof_bits_v<ElementScaleB> / bits_per_byte;
      auto const scale_k = cute::ceil_div(options.k, GROUP_SIZE);
      scale_bytes_transferred = static_cast<double>(
          options.m * scale_k * sizeof_scale_a +
          options.n * scale_k * sizeof_scale_b) * options.l;
    }
    auto mega_bytes_transferred = static_cast<double>(
        options.m * options.k * sizeof_a +
        options.k * options.n * sizeof_b +
        (options.beta != 0 ? 2 : 1) * options.m * options.n * sizeof_o
      ) * 1e-6 * options.l + scale_bytes_transferred * 1e-6;

    initialize_counters(state);
    for(auto _ : state) {
      state.PauseTiming();
      cache_flush_.prepare();

      typename Gemm::GemmKernel::Arguments arguments = [&]() {
        if constexpr (!is_blocked_scaled<CollectiveMainloop>) {
          return typename Gemm::GemmKernel::Arguments{
            gemm::GemmUniversalMode::kGemm,
            problem_size,
            {cache_flush_.ptr_A(), stride_A, cache_flush_.ptr_B(), stride_B},
            {{ElementAccumulator(options.alpha), ElementAccumulator(options.beta)}, cache_flush_.ptr_C(), stride_C, block_D.get(), stride_D},
            hw_info
          };
        } else {
          return typename Gemm::GemmKernel::Arguments{
            gemm::GemmUniversalMode::kGemm,
            problem_size,
            {cache_flush_.ptr_A(), stride_A, cache_flush_.ptr_B(), stride_B,
              cache_flush_.ptr_SA(), stride_SA, cache_flush_.ptr_SB(), stride_SB},
            {{ElementAccumulator(options.alpha), ElementAccumulator(options.beta)}, cache_flush_.ptr_C(), stride_C, block_D.get(), stride_D},
            hw_info
          };
        }
      }();

      gemm_op.initialize(arguments, workspace.get());
      state.ResumeTiming();

      GPU_Clock timer;
      timer.start();
      gemm_op.run();
      auto ms_elapsed = timer.milliseconds();
      update_counters(state, ms_elapsed);
      state.SetIterationTime(ms_elapsed / 1000);
    }
    auto wall_end = std::chrono::steady_clock::now();
    finalize_counters(state, gflop, mega_bytes_transferred);
    state.counters["execution_time_s"] =
        (std::chrono::duration<double, std::milli>(wall_end - wall_start).count())/1000;
  }

private:
  static void initialize_counters(::benchmark::State& state) {
    state.counters["avg_runtime_ms"] = 0;
    state.counters["best_runtime_ms"] = std::numeric_limits<double>::max();
    state.counters["worst_runtime_ms"] = std::numeric_limits<double>::lowest();
  }

  static void update_counters(::benchmark::State& state, double ms_elapsed) {
    state.PauseTiming();
    state.counters["total_runtime_ms"] += ms_elapsed;
    state.counters["best_runtime_ms"] = std::min<double>(state.counters["best_runtime_ms"], ms_elapsed);
    state.counters["worst_runtime_ms"] = std::max<double>(state.counters["worst_runtime_ms"], ms_elapsed);
    state.ResumeTiming();
  }

  static void finalize_counters(::benchmark::State& state,  double gflop, double mega_bytes_transferred) {
    state.counters["avg_runtime_ms"] =
      (state.counters["total_runtime_ms"] -state.counters["best_runtime_ms"] - state.counters["worst_runtime_ms"] ) / static_cast<double>(state.iterations() - 2);
    state.counters["avg_tflops"] = gflop / state.counters["avg_runtime_ms"];
    state.counters["avg_bandwidth_gbs"] = mega_bytes_transferred / state.counters["avg_runtime_ms"];
    state.counters["best_tflops"] = gflop / state.counters["best_runtime_ms"];
    state.counters["best_bandwidth_gbs"] = mega_bytes_transferred / state.counters["best_runtime_ms"];
  }
};

// Template-based benchmark runner
template <class GemmConfig>
void gemm_bench_runner(
    ::benchmark::State& state,
    GEMMOptions const& options,
    KernelHardwareInfo const& hw_info) {
  auto bench = BenchmarkRunnerGemm<GemmConfig>();
  bench.run(state, options, hw_info);
}

}

#define CUTLASS_BENCHMARK(F) cutlass::benchmark::BenchmarkRegistry<cutlass::benchmark::GEMMOptions>::Register(#F, &F##_func)

// Template-based benchmark registration: takes a name string and a template type.
// Usage: CUTLASS_BENCHMARK_T("MyBenchmarkName", MyGemmType<float, float, float, 256, 256, 64, 32, 64>);
#define CUTLASS_BENCHMARK_T(name, ...)                                              \
  cutlass::benchmark::BenchmarkRegistry<cutlass::benchmark::GEMMOptions>::Register( \
      name,                                                                         \
      &cutlass::benchmark::gemm_bench_runner<__VA_ARGS__>)

#define CUTLASS_CREATE_GEMM_BENCHMARK(F)                          \
  static void F##_func(                                           \
      ::benchmark::State& state,                                  \
      cutlass::benchmark::GEMMOptions const& options,                 \
      cutlass::KernelHardwareInfo const& hw_info) {               \
    auto bench = cutlass::benchmark::BenchmarkRunnerGemm<F>();    \
    bench.run(state, options, hw_info);                           \
  }
