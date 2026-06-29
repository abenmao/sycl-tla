/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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
    \brief MoE GEMM benchmark runner.

    Aligned with example 12_xe20_moe_gemm_cute_interface. The MoE GEMM uses the
    custom MoE::MoEGEMM kernel together with the PersistentTileSchedulerXeMoE
    tile scheduler (TileShape <256, 128, 32>), which is launched manually rather
    than through GemmUniversalAdapter. This runner replicates the example's
    MoEGEMMLauncher inside the google-benchmark timing loop, and reuses the
    grouped-gemm BenchmarkRegistry<GroupedGEMMOptions> so the same config-file
    dispatch mechanism applies.

    The benchmark interprets the grouped-gemm options as a MoE problem:
      groups -> number of experts
      m      -> tokens routed to each expert (uniform distribution)
      n / k  -> MoE GEMM N / K extents
*/

#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <vector>

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/initialize_block.hpp"
#include "cutlass/util/sycl_event_manager.hpp"

// MoE example headers (added to the include path via benchmarks/grouped_gemm/CMakeLists.txt).
#include "moe_grouped_gemm.hpp"
#include "moe_tile_scheduler.hpp"

// benchmark_runner.hpp (already included by main.cpp before this header) provides
// cutlass::benchmark::GroupedGEMMOptions and the BenchmarkRegistry infrastructure.
#include "benchmark_runner.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace cutlass::benchmark {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

// Configuration describing a MoE GEMM benchmark instance, aligned with example 12.
template <
    typename ElementA_, typename ElementB_, typename ElementD_,
    typename TileShape_, typename TiledMMA_,
    typename GmemTiledCopyA_ = XE_LOAD_2D<16, 32, 32, 16>,
    typename GmemTiledCopyB_ = XE_LOAD_2D_VNNI<16, 32, 16, 16>,
    typename GmemTiledCopyD_ = XE_STORE_2D<16, 8, 32>>
struct MoEGemmConfiguration {
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementD = ElementD_;
  using TileShape = TileShape_;
  using TiledMMA = TiledMMA_;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyD = GmemTiledCopyD_;

  static constexpr char LayoutA = 'R';
  static constexpr char LayoutB = 'R';
  static constexpr char LayoutD = 'R';
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Unique SYCL kernel name tag per configuration.
template <typename Config> class MoEGemmBenchKernelName;

template <class Config>
struct BenchmarkRunnerMoEGemm {

  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementD = typename Config::ElementD;
  using TileShape = typename Config::TileShape;
  using TiledMMA = typename Config::TiledMMA;
  using GmemTiledCopyA = typename Config::GmemTiledCopyA;
  using GmemTiledCopyB = typename Config::GmemTiledCopyB;
  using GmemTiledCopyD = typename Config::GmemTiledCopyD;

  using ProblemShape = MoE::ProblemShape;
  using TileScheduler = MoE::PersistentTileSchedulerXeMoE<ProblemShape>;
  using RasterOrderOptions = typename TileScheduler::RasterOrderOptions;
  using ClusterShape = Shape<_1, _1, _1>;

  //
  // Data members
  //
  uint64_t seed = 0;

  int num_experts = 0;
  int gemm_n = 0;
  int gemm_k = 0;
  int64_t total_tokens = 0;

  std::vector<int32_t> tokens_per_expert_host;

  cutlass::DeviceAllocation<int32_t> num_rows_per_expert;
  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementD> block_D;

  // Dummy problem shape kept as a member so it outlives the asynchronous kernel
  // launch in launch(). scheduler_params (captured by value into the kernel)
  // stores a pointer to it, while the caller waits only after launch() returns;
  // stack-local storage here would dangle (use-after-scope).
  cute::Shape<int, int, int> dummy_problem_shape_{};
  ProblemShape dummy_group_problem_shape_{};

  //
  // Methods
  //

  void allocate(const GroupedGEMMOptions& options) {
    num_experts = options.groups;
    gemm_n = options.n;
    gemm_k = options.k;

    // Uniform token distribution: each expert receives options.m tokens.
    tokens_per_expert_host.assign(num_experts, options.m);
    total_tokens = static_cast<int64_t>(options.m) * num_experts;

    int64_t a_size = total_tokens * gemm_k;
    int64_t b_size = static_cast<int64_t>(num_experts) * gemm_n * gemm_k;
    int64_t d_size = total_tokens * gemm_n;

    num_rows_per_expert.reset(num_experts);
    block_A.reset(a_size);
    block_B.reset(b_size);
    block_D.reset(d_size);
  }

  void initialize(::benchmark::State& state) {
    try {
      num_rows_per_expert.copy_from_host(tokens_per_expert_host.data());
      initialize_block(block_A, seed + 2023);
      initialize_block(block_B, seed + 2022);
      initialize_block(block_D, seed + 2021);
    } catch (std::exception const& e) {
      state.SkipWithError(e.what());
    }
  }

  // Replicates example 12's MoEGEMMLauncher: build scheduler params, derive grid
  // shape, and launch the custom MoE::MoEGEMM kernel.
  sycl::event launch(const KernelHardwareInfo& hw_info) {
    dummy_problem_shape_ = cute::Shape<int, int, int>{1, gemm_k, gemm_n};
    dummy_group_problem_shape_ = ProblemShape{1, &dummy_problem_shape_, nullptr};

    auto scheduler_params = TileScheduler::to_underlying_arguments(
        dummy_group_problem_shape_, TileShape{}, ClusterShape{}, hw_info,
        typename TileScheduler::Arguments{1, RasterOrderOptions::AlongN});
    auto group_distribution = TileScheduler::get_grid_shape(
        scheduler_params, dummy_group_problem_shape_, TileShape{}, ClusterShape{},
        hw_info, typename TileScheduler::Arguments{1, RasterOrderOptions::AlongN});

    TiledMMA mma{};
    auto MaxThreadsPerWorkgroup = size(mma);

    sycl::range<3> local = {1, 1, static_cast<size_t>(MaxThreadsPerWorkgroup)};
    sycl::range<3> groups = {group_distribution.z, group_distribution.y,
                             group_distribution.x};
    sycl::range<3> global = {local[0] * groups[0], local[1] * groups[1],
                             local[2] * groups[2]};

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    syclex::properties kernel_props{syclex::sub_group_size<16>,
#if (defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
                                    intelex::grf_size<512>
#else
                                    intelex::grf_size<256>
#endif
    };

    const ElementA* activations = block_A.get();
    const ElementB* weights = block_B.get();
    ElementD* outputs = block_D.get();
    const int32_t* num_rows = num_rows_per_expert.get();
    const float* scales = nullptr;
    const int n = gemm_n;
    const int k = gemm_k;
    const int experts = num_experts;

    sycl::queue Q = compat::get_default_queue();
    auto event = Q.parallel_for<MoEGemmBenchKernelName<Config>>(
        sycl::nd_range<3>(global, local), kernel_props, [=](auto) {
          MoE::MoEGEMM<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD,
                       Config::LayoutA, Config::LayoutB, Config::LayoutD>(
              activations, weights, scales, outputs, mma, num_rows, experts, n,
              k, scheduler_params);
        });
    EventManager::getInstance().addEvent(event);
    return event;
  }

  void run(::benchmark::State& state, const GroupedGEMMOptions& options,
           const KernelHardwareInfo& hw_info) {
    allocate(options);
    initialize(state);
    if (state.error_occurred()) return;

#ifdef CUTLASS_TEST_FOR_CRI
    // Disable warmup run for the CRI simulator as it's time-consuming.
#else
    launch(hw_info);
    compat::wait();
#endif

    state.counters["m"] = options.m;
    state.counters["n"] = options.n;
    state.counters["k"] = options.k;
    state.counters["groups"] = options.groups;

    // Number of real-valued multiply-adds across all experts.
    uint64_t fmas = uint64_t();
    for (auto rows : tokens_per_expert_host) {
      fmas += static_cast<uint64_t>(rows) * static_cast<uint64_t>(gemm_n) *
              static_cast<uint64_t>(gemm_k);
    }
    uint64_t flop = static_cast<uint64_t>(2) * fmas;
    double gflop = double(flop) / double(1.0e9);

    constexpr double bits_per_byte = static_cast<double>(sizeof_bits_v<char>);
    constexpr double sizeof_a = sizeof_bits_v<ElementA> / bits_per_byte;
    constexpr double sizeof_b = sizeof_bits_v<ElementB> / bits_per_byte;
    constexpr double sizeof_d = sizeof_bits_v<ElementD> / bits_per_byte;
    auto mega_bytes_transferred = static_cast<double>(
        total_tokens * gemm_k * sizeof_a +
        static_cast<int64_t>(num_experts) * gemm_n * gemm_k * sizeof_b +
        total_tokens * gemm_n * sizeof_d) * 1e-6;

    initialize_counters(state);
    for (auto _ : state) {
      GPU_Clock timer;
      timer.start();
      launch(hw_info);
      compat::wait();
      auto ms_elapsed = timer.milliseconds();
      update_counters(state, ms_elapsed);
      state.SetIterationTime(ms_elapsed / 1000);
    }
    finalize_counters(state, gflop, mega_bytes_transferred);
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

  static void finalize_counters(::benchmark::State& state, double gflop, double mega_bytes_transferred) {
    auto denom = static_cast<double>(state.iterations());
    if (state.iterations() > 2) {
      state.counters["avg_runtime_ms"] =
          (state.counters["total_runtime_ms"] - state.counters["best_runtime_ms"] - state.counters["worst_runtime_ms"]) /
          static_cast<double>(state.iterations() - 2);
    } else {
      state.counters["avg_runtime_ms"] = state.counters["total_runtime_ms"] / denom;
    }
    state.counters["avg_tflops"] = gflop / state.counters["avg_runtime_ms"];
    state.counters["avg_throughput"] = mega_bytes_transferred / state.counters["avg_runtime_ms"];
    state.counters["best_tflop"] = gflop / state.counters["best_runtime_ms"];
    state.counters["best_bandwidth"] = mega_bytes_transferred / state.counters["best_runtime_ms"];
  }
};

} // namespace cutlass::benchmark

#define CUTLASS_CREATE_MOE_GEMM_BENCHMARK(F)                              \
  static void F##_func(                                                   \
      ::benchmark::State& state,                                         \
      cutlass::benchmark::GroupedGEMMOptions const& options,             \
      cutlass::KernelHardwareInfo const& hw_info) {                      \
    auto bench = cutlass::benchmark::BenchmarkRunnerMoEGemm<F>();         \
    bench.run(state, options, hw_info);                                  \
  }
