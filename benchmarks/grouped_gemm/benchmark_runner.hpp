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
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_silu.h"
#include "cutlass/util/initialize_block.hpp"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/relatively_equal.h"

#include "../common.hpp"
#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <random>

using namespace cute;

namespace cutlass::benchmark {

///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cutlass::gemm;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group

template <class T, class = void>
struct ScaleType {
  using type = int;
};
template <class T>
struct ScaleType<T, cute::void_t<typename T::ElementScale>> {
  using type = typename T::ElementScale;
};

template <class T, class = void>
struct ZeroType {
  using type = int;
};
template <class T>
struct ZeroType<T, cute::void_t<typename T::ElementZero>> {
  using type = typename T::ElementZero;
};

template <class T, class = void>
struct ScaleStride {
  using type = int;
};
template <class T>
struct ScaleStride<T, cute::void_t<typename T::StrideScale>> {
  using type = typename T::StrideScale;
};

template <class T, class = void>
struct ZeroStride {
  using type = int;
};
template <class T>
struct ZeroStride<T, cute::void_t<typename T::StrideZero>> {
  using type = typename T::StrideZero;
};

template <class T, class = void>
static constexpr auto is_blocked_scaled = false;
template <class T>
static constexpr auto is_blocked_scaled<T, cute::void_t<typename T::ElementScaleA, typename T::ElementScaleB>> = true;

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
struct StrideScaleAType<T, cute::void_t<typename T::InternalStrideScaleA>> {
  using type = typename T::InternalStrideScaleA;
};

template <class T, class = void>
struct StrideScaleBType {
  using type = int;
};
template <class T>
struct StrideScaleBType<T, cute::void_t<typename T::InternalStrideScaleB>> {
  using type = typename T::InternalStrideScaleB;
};

template <class T, class = void>
struct GroupSizeType {
  static constexpr int value = 32;  // Default for non-block-scaled
};
template <class T>
struct GroupSizeType<T, cute::void_t<decltype(T::GROUP_K)>> {
  static constexpr int value = T::GROUP_K;
};

// SFINAE helper: reads DispatchPolicy::GroupSize only when it exists (block-scaled collectives).
// Falls back to 32 for non-block-scaled collectives (BF16, plain FP8) that have no GroupSize.
template <class T, class = void>
struct DispatchGroupSizeType {
  static constexpr int value = 32;
};
template <class T>
struct DispatchGroupSizeType<T, cute::void_t<typename T::DispatchPolicy::GroupSize>> {
  static constexpr int value = int(typename T::DispatchPolicy::GroupSize{});
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Verification mode enum
enum class VerifyMode {
  None = 0,           // No verification (skip both device and host)
  Device = 1,         // Device-only verification (default) - uses reference::device::GemmComplex
  Host = 2            // Host verification - takes a long time, use only for small problem sizes
};

// Command line options parsing
struct GroupedGEMMOptions {

  bool error;
  VerifyMode verify_mode;

  int m, n, k, l, groups;  // groups is computed from num_experts/ep_size
  int topk, num_experts, ep_size;
  bool random_mode;
  int seed;
  float alpha, beta;
  std::string bm_name;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

  GroupedGEMMOptions():
          error(false),
          verify_mode(VerifyMode::Device),
          m(5120), n(4096), k(4096), l(1),
          groups(128),  // Will be recomputed in parse()
          topk(8), num_experts(128), ep_size(1),
          random_mode(false), seed(42),
          alpha(1.f), beta(0.f),
          bm_name("GroupedGEMM")
  {
    problem_sizes_host.reserve(groups);
    for(int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({m, n, k});
    }

  }

  // Parses the command line
  void parse(int argc, char const **args) {
    CommandLine cmd(argc, args);

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("topk", topk, 8);
    cmd.get_cmd_line_argument("num_experts", num_experts, 128);
    cmd.get_cmd_line_argument("ep_size", ep_size, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("bm_name", bm_name, std::string("GEMM"));
    cmd.get_cmd_line_argument("seed", seed, 42);
    random_mode = cmd.check_cmd_line_flag("random");

    // Parse verification mode
    std::string verify_str = "device";
    cmd.get_cmd_line_argument("verify", verify_str, std::string("device"));
    if (verify_str == "none") {
      verify_mode = VerifyMode::None;
    } else if (verify_str == "device") {
      verify_mode = VerifyMode::Device;
    } else if (verify_str == "host") {
      verify_mode = VerifyMode::Host;
    } else {
      std::cerr << "Invalid verify mode or mode wasn't defined, using default Verify Mode as None" << std::endl;
      verify_mode = VerifyMode::None;
    }

    // Validate basic dimensions
    assert(m > 0 && "m must be positive");
    assert(n > 0 && "n must be positive");
    assert(k > 0 && "k must be positive");

    // Validate MOE parameters
    assert(topk > 0 && "topk must be positive");
    assert(num_experts > 0 && "num_experts must be positive");
    assert(ep_size > 0 && "ep_size must be positive");

    problem_sizes_host.clear();

    // Compute groups and generate problem sizes using distribution logic
    int experts_per_gpu = num_experts / ep_size;
    int tokens_per_gpu = (m * topk) / ep_size;

    groups = experts_per_gpu;
    problem_sizes_host.reserve(groups);

    std::vector<int> m_per_expert(experts_per_gpu);

    if (random_mode) {
      // Deterministic random distribution using the seed value.
      // Some experts may get 0 tokens.
      std::mt19937 rng(seed);
      int remaining = tokens_per_gpu;
      for (int i = 0; i < experts_per_gpu - 1; i++) {
        std::uniform_int_distribution<int> dist(0, remaining);
        m_per_expert[i] = dist(rng);
        remaining -= m_per_expert[i];
      }
      m_per_expert[experts_per_gpu - 1] = remaining;

      // Shuffle to avoid bias toward later experts getting fewer tokens
      std::shuffle(m_per_expert.begin(), m_per_expert.end(), rng);
    } else {
      // Default: uniform distribution — divide tokens evenly among experts.
      // Some experts may get 0 if tokens_per_gpu < experts_per_gpu.
      int base = tokens_per_gpu / experts_per_gpu;
      int remainder = tokens_per_gpu % experts_per_gpu;
      std::fill_n(m_per_expert.begin(), remainder, base + 1);
      std::fill(m_per_expert.begin() + remainder, m_per_expert.end(), base);
    }

    for (int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({m_per_expert[i], n, k});
    }
  }

  /// Compute performance in TFLOP/s
  double tflops(double runtime_s, std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host) const
  {
    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();

    for (auto const & problem : problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = static_cast<uint64_t>(2) * static_cast<uint64_t>(fmas);
    double tflop = double(flop) / double(1.0e12);
    return tflop / runtime_s;
  }

  std::string benchmark_name() const {
    std::stringstream full_name;
    full_name << bm_name << "/";
    int tokens_per_gpu = (m * topk) / ep_size;
    int experts_per_gpu = num_experts / ep_size;
    full_name << "t" << m << "_tpg" << tokens_per_gpu
              << "_e" << experts_per_gpu << "x" << n << "x" << k;
    if (random_mode) {
      full_name << "_rng" << seed;
    } else {
      full_name << "_uniform";
    }

    return full_name.str();
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Non-owning view into a single pooled DeviceAllocation. Drop-in for the subset of
// DeviceAllocation<T> API the grouped GEMM benchmark uses (.get()/.size()). Allocating one
// big pool per tensor type and slicing it into per-group views avoids the many small device
// allocations (~9 per group) that fragment / exhaust the simulated GPU address space at high
// group counts, which made big-MOE block-scaled runs crawl and crash during allocation.
template <class T>
struct PooledView {
  T* ptr_ = nullptr;
  size_t size_ = 0;
  PooledView() = default;
  PooledView(T* p, size_t n) : ptr_(p), size_(n) {}
  T* get() const { return ptr_; }
  size_t size() const { return size_; }
};

// Helper functions for pooled allocation: convert element offsets to byte-aligned pointers.
// Extracted as free functions to enable cross-call optimization and reuse.
namespace detail {
  template<typename T>
  static inline T* slice_aligned(T* pool_ptr, int64_t elem_off, int bits_per_elem) {
    int64_t byte_off = (elem_off * bits_per_elem + 7) / 8;  // logical elems -> bytes (round up)
    byte_off = ((byte_off + 63) / 64) * 64;                 // 64-byte align the base
    auto* base = reinterpret_cast<uint8_t*>(pool_ptr) + byte_off;
    return reinterpret_cast<T*>(base);
  }

  static inline int64_t bump_aligned(int64_t elem_off, int64_t added, int bits_per_elem) {
    int64_t byte_off = ((((elem_off * bits_per_elem + 7) / 8) + 63) / 64) * 64;
    int64_t end_bytes = byte_off + (added * bits_per_elem + 7) / 8;
    return (end_bytes * 8) / bits_per_elem;
  }
}

template <class GemmConfiguration>
struct BenchmarkRunnerGemm {

  using Gemm = typename GemmConfiguration::Gemm;

  using CollectiveMainloop = typename Gemm::GemmKernel::CollectiveMainloop;
  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;  

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementMMAVerify = float;

  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementMma = typename CollectiveMainloop::TiledMma::ValTypeA;

  using ElementScale = typename ScaleType<CollectiveMainloop>::type;
  using ElementZero = typename ZeroType<CollectiveMainloop>::type;
  using StrideS = typename ScaleStride<CollectiveMainloop>::type;
  using StrideZ = typename ZeroStride<CollectiveMainloop>::type;

  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;

  using ElementScaleA = typename ElementScaleAType<CollectiveMainloop>::type;
  using ElementScaleB = typename ElementScaleBType<CollectiveMainloop>::type;
  using StrideScaleA = typename StrideScaleAType<CollectiveMainloop>::type;
  using StrideScaleB = typename StrideScaleBType<CollectiveMainloop>::type;

  static constexpr int GROUP_SIZE = GroupSizeType<CollectiveMainloop>::value;

  // 2D block load requires 4-byte aligned pitch
  // Compile-time check: GROUP_SIZE must match the collective's actual GroupK
  // Uses DispatchGroupSizeType which safely falls back to 32 for non-block-scaled collectives
  // that don't have DispatchPolicy::GroupSize (e.g. BF16, plain FP8 grouped GEMM kernels).
  static_assert(
      GROUP_SIZE == DispatchGroupSizeType<CollectiveMainloop>::value,
      "GROUP_SIZE does not match the kernel's actual GroupK. "
      "Add 'static constexpr int GROUP_K = GroupSize;' to the collective.");

  static constexpr int ScaleAlignA =
      is_blocked_scaled<CollectiveMainloop> ? cute::ceil_div(4, (int)sizeof(ElementScaleA)) : 1;
  static constexpr int ScaleAlignB =
      is_blocked_scaled<CollectiveMainloop> ? cute::ceil_div(4, (int)sizeof(ElementScaleB)) : 1;

  //int32_t count;

  //
  // Data members
  //

  /// Initialization
  // // Host-side allocation
  std::vector<ElementAccumulator> alpha_host;
  std::vector<ElementAccumulator> beta_host;

  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideScaleA> stride_SFA_host;
  std::vector<StrideScaleB> stride_SFB_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;

  StrideS stride_S;
  StrideZ stride_Z;

  // // Device-side allocations
  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

  cutlass::DeviceAllocation<StrideA> stride_A;
  cutlass::DeviceAllocation<StrideB> stride_B;
  cutlass::DeviceAllocation<StrideC> stride_C;
  cutlass::DeviceAllocation<StrideD> stride_D;
  cutlass::DeviceAllocation<StrideScaleA> stride_SFA;
  cutlass::DeviceAllocation<StrideScaleB> stride_SFB;

  uint64_t seed = 0;

  // Pooled allocation: one big DeviceAllocation per tensor type, sliced into per-group views.
  std::vector<PooledView<ElementA>> block_A;
  std::vector<PooledView<ElementB>> block_B;
  std::vector<PooledView<ElementC>> block_C;
  std::vector<PooledView<ElementOutput>> block_D;
  std::vector<PooledView<ElementScaleA>> block_scaleA;
  std::vector<PooledView<ElementScaleB>> block_scaleB;
  std::vector<PooledView<ElementOutput>> block_ref_D;

  cutlass::DeviceAllocation<ElementA> pool_A;
  cutlass::DeviceAllocation<ElementB> pool_B;
  cutlass::DeviceAllocation<ElementC> pool_C;
  cutlass::DeviceAllocation<ElementOutput> pool_D;
  cutlass::DeviceAllocation<ElementOutput> pool_ref_D;
  cutlass::DeviceAllocation<ElementScaleA> pool_scaleA;
  cutlass::DeviceAllocation<ElementScaleB> pool_scaleB;

  cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
  cutlass::DeviceAllocation<ElementAccumulator> block_beta;
  
  cutlass::DeviceAllocation<ElementScale> block_scale;
  cutlass::DeviceAllocation<ElementZero> block_zero;

  std::vector<DeviceAllocation<ElementMma>> block_A_verify;
  std::vector<DeviceAllocation<ElementMma>> block_B_verify;

  cutlass::DeviceAllocation<const ElementA *> ptr_A;
  cutlass::DeviceAllocation<const ElementB *> ptr_B;
  cutlass::DeviceAllocation<const ElementC *> ptr_C;
  cutlass::DeviceAllocation<ElementOutput *> ptr_D;
  cutlass::DeviceAllocation<const ElementScaleA *> ptr_SFA;
  cutlass::DeviceAllocation<const ElementScaleB *> ptr_SFB;
  cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
  cutlass::DeviceAllocation<ElementAccumulator*> beta_device;

  std::vector<PooledView<ElementMMAVerify>> block_A_dq; // Dequantized copy of A for validation
  std::vector<PooledView<ElementMMAVerify>> block_B_dq; // Dequantized copy of B for validation
  cutlass::DeviceAllocation<ElementMMAVerify> pool_A_dq;
  cutlass::DeviceAllocation<ElementMMAVerify> pool_B_dq;

  BenchmarkRunnerGemm() : seed(0) {};

  //
  // Methods
  //

    /// Populates a Gemm::Arguments structure from the given commandline options
  auto args_from_options(const GroupedGEMMOptions &options, const cutlass::KernelHardwareInfo& hw_info)
  {
    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;

    if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {
      // If both alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
      fusion_args.alpha = options.alpha;
      fusion_args.beta = options.beta;
      fusion_args.alpha_ptr = nullptr;
      fusion_args.beta_ptr = nullptr;
      fusion_args.alpha_ptr_array = nullptr;
      fusion_args.beta_ptr_array = nullptr;
      // Single alpha and beta for all groups
      fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
      fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
    }
    else {
      // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups.
      fusion_args.alpha = 0;
      fusion_args.beta = 0;
      fusion_args.alpha_ptr = nullptr;
      fusion_args.beta_ptr = nullptr;
      fusion_args.alpha_ptr_array = alpha_device.get();
      fusion_args.beta_ptr_array = beta_device.get();
      // One alpha and beta per each group
      fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
      fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
    }
    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;

    // Per-GEMM problem shape info may only exist on the device.
    return cute::make_tuple(cutlass::gemm::GemmUniversalMode::kGrouped,
                            typename Gemm::GemmKernel::ProblemShape{options.groups, problem_sizes.get(), options.problem_sizes_host.data()},
                            fusion_args, hw_info,
                            typename Gemm::GemmKernel::TileSchedulerArguments{1, RasterOrderOptions::AlongN});

  }

  bool verify(const GroupedGEMMOptions &options) {
    bool passed = true;
    ElementOutput const epsilon(1e-2f);
    ElementOutput const non_zero_floor(1e-4f);
    for (int i = 0; i < options.groups; i++){
      Shape<int, int, int, int> problem_size = append<4>(options.problem_sizes_host[i], 1);
      auto M = get<0>(problem_size);
      auto N = get<1>(problem_size);
      auto K = get<2>(problem_size);
      TensorRef ref_A(block_A_dq.at(i).get(), LayoutA::packed({M, K}));
      TensorRef ref_B(block_B_dq.at(i).get(), LayoutB::packed({K, N}));
      TensorRef ref_C(block_C.at(i).get(), LayoutC::packed({M, N}));
      TensorRef ref_D(block_ref_D.at(i).get(), LayoutD::packed({M, N}));
      reference::device::GemmComplex(
              {M, N, K},
              alpha_host.at(i),
              ref_A,
              ComplexTransform::kNone,
              ref_B,
              ComplexTransform::kNone,
              beta_host.at(i),
              ref_C,
              ref_D,
              ElementAccumulator(0),
              1,     // batch_count
              M * K, // batch_stride_A
              K * N, // batch_stride_B
              M * N, // batch_stride_C
              M * N  // batch_stride_D
      );

      compat::wait();

      passed &= cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref_D.at(i).get(), block_D.at(i).get(), block_D.at(i).size(), epsilon, non_zero_floor);

      if (!passed) break;
    }

    return passed;
  }
  static float compute_rtol(int K) {
    float base_tol;

    // Check input element types (A or B) for quantization error, not output
    // Use the looser tolerance if inputs differ
    constexpr bool is_mxfp4 = std::is_same_v<ElementA, cutlass::float_e2m1_t> ||
                              std::is_same_v<ElementB, cutlass::float_e2m1_t>;
    constexpr bool is_mxfp8 = std::is_same_v<ElementA, cutlass::float_ue4m3_t> ||
                              std::is_same_v<ElementB, cutlass::float_ue4m3_t>;
    constexpr bool is_fp8 = std::is_same_v<ElementA, cutlass::float_e4m3_t> ||
                            std::is_same_v<ElementB, cutlass::float_e4m3_t> ||
                            std::is_same_v<ElementA, cutlass::float_e5m2_t> ||
                            std::is_same_v<ElementB, cutlass::float_e5m2_t>;
    constexpr bool is_bf16 = std::is_same_v<ElementA, cutlass::bfloat16_t> ||
                             std::is_same_v<ElementB, cutlass::bfloat16_t>;

    if constexpr (is_mxfp4) {
      base_tol = 2e-2f;  // MXFP4: block-scaled 4-bit, very coarse
    } else if constexpr (is_mxfp8) {
      base_tol = 1e-2f;  // MXFP8: block-scaled FP8
    } else if constexpr (is_fp8) {
      base_tol = 8e-3f;  // FP8: 3-bit or 2-bit mantissa
    } else if constexpr (is_bf16) {
      base_tol = 4e-3f;  // BF16: ~7 bits mantissa
    } else {
      base_tol = 5e-4f;  // FP32, FP16, TF32
    }

    return base_tol * (1.0f + 0.1f * std::log2f(float(K)));
  }

  bool verify_host(::benchmark::State& state, const GroupedGEMMOptions& options) {
    bool all_passed = true;

    // Reuse buffers across groups to avoid repeated allocations
    std::vector<float> host_A, host_B, host_C, host_D, host_ref_D;
    std::vector<ElementC> host_C_raw;
    std::vector<ElementOutput> host_D_raw;

    for (int i = 0; i < options.groups; i++) {
      auto problem = options.problem_sizes_host[i];
      int M = get<0>(problem);
      int N = get<1>(problem);
      int K = get<2>(problem);

      uint64_t flops = uint64_t(M) * uint64_t(N) * uint64_t(K) * 2;
      if (flops > 20'000'000'000ULL) {  // 20 GFLOP threshold - may take significant CPU time
        std::cerr << "[WARNING] Group " << i << " (" << M << "x" << N << "x" << K
                  << "): Large problem size (" << flops/1e9 << " GFLOP), host verification may be slow" << std::endl;
      }

      size_t size_A = size_t(M) * K;
      size_t size_B = size_t(N) * K;
      size_t size_CD = size_t(M) * N;

      host_A.resize(size_A);
      host_B.resize(size_B);
      host_C.resize(size_CD);
      host_D.resize(size_CD);
      host_ref_D.assign(size_CD, 0.0f);
      host_C_raw.resize(size_CD);
      host_D_raw.resize(size_CD);

      // block_* are non-owning PooledView slices, so copy via the free function.
      cutlass::device_memory::copy_to_host(host_A.data(), block_A_dq.at(i).get(), block_A_dq.at(i).size());
      cutlass::device_memory::copy_to_host(host_B.data(), block_B_dq.at(i).get(), block_B_dq.at(i).size());
      compat::wait();

      cutlass::device_memory::copy_to_host(host_C_raw.data(), block_C.at(i).get(), block_C.at(i).size());
      compat::wait();
      for (size_t idx = 0; idx < size_CD; idx++) {
        host_C[idx] = float(host_C_raw[idx]);
      }

      cutlass::device_memory::copy_to_host(host_D_raw.data(), block_D.at(i).get(), block_D.at(i).size());
      compat::wait();
      for (size_t idx = 0; idx < size_CD; idx++) {
        host_D[idx] = float(host_D_raw[idx]);
      }

      cutlass::TensorRef<float, LayoutA> ref_A(host_A.data(), LayoutA::packed({M, K}));
      cutlass::TensorRef<float, LayoutB> ref_B(host_B.data(), LayoutB::packed({K, N}));
      cutlass::TensorRef<float, LayoutD> ref_C(host_C.data(), LayoutD::packed({M, N}));
      cutlass::TensorRef<float, LayoutD> ref_D(host_ref_D.data(), LayoutD::packed({M, N}));

      float alpha_f = float(alpha_host.at(i));
      float beta_f = float(beta_host.at(i));

      cutlass::reference::host::compute_gemm<
        float, LayoutA,
        float, LayoutB,
        float, LayoutD,
        float, float>(
        {M, N, K},
        alpha_f,
        ref_A,
        ref_B,
        beta_f,
        ref_C,
        ref_D,
        0.0f);

      float rtol = compute_rtol(K);
      float nonzero_floor = std::numeric_limits<float>::min();
      int fail_count = 0;
      float max_rel_error = 0.0f;

      for (size_t idx = 0; idx < size_CD; idx++) {
        if (!cutlass::relatively_equal(host_ref_D[idx], host_D[idx], rtol, nonzero_floor)) {
          fail_count++;
          float diff = std::abs(host_ref_D[idx] - host_D[idx]);
          float denom = std::abs(host_ref_D[idx]) + std::abs(host_D[idx]);
          if (denom > 0) {
            max_rel_error = std::max(max_rel_error, diff / denom);
          } else {
            max_rel_error = std::max(max_rel_error, diff);
          }
        }
      }

      if (fail_count == 0) {
        std::cerr << "[VERIFY] Group " << i << " (" << M << "x" << N << "x" << K
                  << "): PASSED (rtol=" << rtol << ")" << std::endl;
      } else {
        std::cerr << "[VERIFY] Group " << i << " (" << M << "x" << N << "x" << K
                  << "): FAILED (" << fail_count << "/" << size_CD
                  << " mismatches, max_rel=" << max_rel_error << ", rtol=" << rtol << ")" << std::endl;
        all_passed = false;
      }
    }

    state.counters["verify_passed"] = all_passed ? 1.0 : 0.0;
    return all_passed;
  }

  template <class Element>
  bool initialize_scale(
    Element* block_ptr, size_t block_size,
    GroupedGEMMOptions const& options) {
    const float elt_max_f = float(cutlass::platform::numeric_limits<Element>::max());
    // Need to fix max_dequant_val and min_dequant_val?
    const float max_dequant_val = elt_max_f * 0.25f;
    const float min_dequant_val = 0.5f;
    const float scale_max = max_dequant_val / elt_max_f;
    const float scale_min = min_dequant_val / elt_max_f;
    cutlass::reference::device::BlockFillRandomUniform(
        block_ptr, block_size, seed, Element(scale_max), Element(scale_min));
    return true;
  }

  template <
  class DstElement,
  class SrcElement,
  class Layout,
  class ElementScale,
  class ScaleLayout>
  static void apply_scale(DstElement* dq_buffer,
                       SrcElement const* q_buffer,
                       Layout const operand_layout,
                       ElementScale const* scale_buffer,
                       ScaleLayout const scale_layout) {
    if constexpr (std::is_same_v<DstElement, SrcElement>) {
      return;
    }

    std::vector<uint8_t> dst(size(operand_layout) * sizeof_bits_v<DstElement> / 8, 0);
    cutlass::device_memory::copy_to_host(dst.data(), (uint8_t*)dq_buffer, dst.size());

    std::vector<uint8_t> src(size(operand_layout) * sizeof_bits_v<SrcElement> / 8, 0);
    cutlass::device_memory::copy_to_host(src.data(), (uint8_t*)q_buffer, src.size());

    std::vector<uint8_t> scale(size(scale_layout) * sizeof_bits_v<ElementScale> / 8, 0);
    cutlass::device_memory::copy_to_host(scale.data(), (uint8_t*)scale_buffer, scale.size());

    compat::wait();

    static_assert(sizeof_bits_v<DstElement> >= 8);

    auto dst_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<DstElement*>(dst.data())), operand_layout);

    auto src_tensor = [&]() {
      if constexpr (sizeof_bits_v<SrcElement> < 8) {
        return make_tensor(cute::subbyte_iterator<const SrcElement>(src.data()), operand_layout);
      } else {
        return make_tensor(make_gmem_ptr(reinterpret_cast<SrcElement const *>(src.data())), operand_layout);
      }
    }();

    auto scale_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<ElementScale const *>(scale.data())), scale_layout);

    auto MN = size<0>(src_tensor);
    auto K = size<1>(src_tensor);
    auto L = size<2>(src_tensor);

    using ret_type = float;

    for (int l = 0; l < L; l++) {
      for (int k= 0; k < K; k++) {
        for (int mn = 0; mn < MN; mn++) {
          auto src_data = [&]() {
            if constexpr (sizeof_bits_v<SrcElement> >= 8) {
              return  (ret_type)(src_tensor(mn, k, l));
            } else {
              return (ret_type)(src_tensor(mn, k, l).get());
            }
          }();

          auto scale_data = (ret_type)(scale_tensor(mn, k / GROUP_SIZE, l));

          dst_tensor(mn, k, l) = (src_data) * scale_data;
        }
      }
    }

    cutlass::device_memory::copy_to_device(dq_buffer, (DstElement*)(raw_pointer_cast(dst_tensor.data())), dst_tensor.size());
    compat::wait();
  }
  
  void allocate(const GroupedGEMMOptions &options) {
    constexpr bool is_bs = is_blocked_scaled<CollectiveMainloop>;

    // Compute and cache sizes once per group (avoid repeated calculations in pass 2)
    struct GroupSizes { int64_t A, B, CD, SFA, SFB; };
    std::vector<GroupSizes> cached_sizes;
    cached_sizes.reserve(options.groups);

    // Pass 1: compute sizes and sum totals in a single pass
    int64_t tot_A = 0, tot_B = 0, tot_CD = 0, tot_SFA = 0, tot_SFB = 0;
    for (int32_t i = 0; i < options.groups; ++i) {
      auto problem = options.problem_sizes_host.at(i);
      int64_t M = get<0>(problem), N = get<1>(problem), K = get<2>(problem);
      int64_t scale_k = cute::ceil_div(K, GROUP_SIZE);
      int64_t padded_M = cute::round_up(M, (int64_t)ScaleAlignA);
      int64_t padded_N = cute::round_up(N, (int64_t)ScaleAlignB);

      cached_sizes.push_back({M * K, N * K, M * N, scale_k * padded_M, scale_k * padded_N});
      tot_A += cached_sizes.back().A;
      tot_B += cached_sizes.back().B;
      tot_CD += cached_sizes.back().CD;
      tot_SFA += cached_sizes.back().SFA;
      tot_SFB += cached_sizes.back().SFB;
    }

    // Pad each pool by one 64-byte block (in elements) per group, so the per-group 64-byte
    // base alignment in Pass 2 can never overrun the pool.
    auto pad = [&](int bits_per_elem) -> int64_t {
      return int64_t(options.groups) * ((64 * 8 + bits_per_elem - 1) / bits_per_elem);
    };
    // One big reset() per pool (A_dq/B_dq are float verify copies; C and ref_D share D's shape).
    pool_A.reset(tot_A + pad(cutlass::sizeof_bits<ElementA>::value));
    pool_A_dq.reset(tot_A + pad(cutlass::sizeof_bits<ElementMMAVerify>::value));
    pool_B.reset(tot_B + pad(cutlass::sizeof_bits<ElementB>::value));
    pool_B_dq.reset(tot_B + pad(cutlass::sizeof_bits<ElementMMAVerify>::value));
    pool_C.reset(tot_CD + pad(cutlass::sizeof_bits<ElementC>::value));
    pool_D.reset(tot_CD + pad(cutlass::sizeof_bits<ElementOutput>::value));
    // ref_D holds the device reference output, only needed for device verification.
    if (options.verify_mode == VerifyMode::Device) {
      pool_ref_D.reset(tot_CD + pad(cutlass::sizeof_bits<ElementOutput>::value));
    }
    if constexpr (is_bs) {
      pool_scaleA.reset(tot_SFA + pad(cutlass::sizeof_bits<ElementScaleA>::value));
      pool_scaleB.reset(tot_SFB + pad(cutlass::sizeof_bits<ElementScaleB>::value));
    }

    // Pass 2: carve non-owning per-group views out of each pool via running offsets.
    //
    // CRITICAL for sub-byte element types (4-bit E2M1 / mxfp4): pool.get() returns a typed
    // pointer (e.g. float_e2m1_t*), and float_e2m1_t has sizeof()==1 byte but sizeof_bits==4.
    // So `pool.get() + elem_off` advances elem_off WHOLE BYTES = 2x the intended 4-bit
    // elements, sending every group i>0 to the wrong (out-of-range) address — the kernel then
    // reads unmapped pages and the simulator hangs (thousands of uninitialized-PTE reads).
    // Fix: compute every per-group base in BYTES (elem_off * sizeof_bits / 8), additionally
    // rounded up to a 64-byte boundary (Xe block-2D loads need a 64-byte-aligned base, which
    // per-group DeviceAllocation used to provide for free), then cast back to the element ptr.
    const int bA  = cutlass::sizeof_bits<ElementA>::value;
    const int bB  = cutlass::sizeof_bits<ElementB>::value;
    const int bV  = cutlass::sizeof_bits<ElementMMAVerify>::value;
    const int bC  = cutlass::sizeof_bits<ElementC>::value;
    const int bO  = cutlass::sizeof_bits<ElementOutput>::value;
    const int bSA = cutlass::sizeof_bits<ElementScaleA>::value;
    const int bSB = cutlass::sizeof_bits<ElementScaleB>::value;
    block_A.clear(); block_B.clear(); block_C.clear(); block_D.clear();
    block_ref_D.clear(); block_A_dq.clear(); block_B_dq.clear();
    block_scaleA.clear(); block_scaleB.clear();
    int64_t oA = 0, oB = 0, oCD = 0, oSFA = 0, oSFB = 0;
    for (int32_t i = 0; i < options.groups; ++i) {
      const auto& s = cached_sizes[i];
      block_A.emplace_back(detail::slice_aligned(pool_A.get(), oA, bA), s.A);
      block_A_dq.emplace_back(detail::slice_aligned(pool_A_dq.get(), oA, bV), s.A);
      block_B.emplace_back(detail::slice_aligned(pool_B.get(), oB, bB), s.B);
      block_B_dq.emplace_back(detail::slice_aligned(pool_B_dq.get(), oB, bV), s.B);
      block_C.emplace_back(detail::slice_aligned(pool_C.get(), oCD, bC), s.CD);
      block_D.emplace_back(detail::slice_aligned(pool_D.get(), oCD, bO), s.CD);
      if (options.verify_mode == VerifyMode::Device) {
        block_ref_D.emplace_back(detail::slice_aligned(pool_ref_D.get(), oCD, bO), s.CD);
      }
      oA = detail::bump_aligned(oA, s.A, bA); oB = detail::bump_aligned(oB, s.B, bB); oCD = detail::bump_aligned(oCD, s.CD, bO);
      if constexpr (is_bs) {
        block_scaleA.emplace_back(detail::slice_aligned(pool_scaleA.get(), oSFA, bSA), s.SFA);
        block_scaleB.emplace_back(detail::slice_aligned(pool_scaleB.get(), oSFB, bSB), s.SFB);
        oSFA = detail::bump_aligned(oSFA, s.SFA, bSA); oSFB = detail::bump_aligned(oSFB, s.SFB, bSB);
      }
    }

    block_alpha.reset(options.groups);
    block_beta.reset(options.groups);
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(::benchmark::State& state, GroupedGEMMOptions const &options) {
    problem_sizes.reset(options.groups);
    problem_sizes.copy_from_host(options.problem_sizes_host.data());

    std::vector<ElementA *> ptr_A_host(options.groups);
    std::vector<ElementB *> ptr_B_host(options.groups);
    std::vector<ElementC *> ptr_C_host(options.groups);
    std::vector<ElementOutput *> ptr_D_host(options.groups);
    std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
    std::vector<ElementAccumulator *> ptr_beta_host(options.groups);
    
    std::vector<ElementScaleA *> ptr_SFA_host;
    std::vector<ElementScaleB *> ptr_SFB_host;
    if constexpr (is_blocked_scaled<CollectiveMainloop>) {
      ptr_SFA_host.resize(options.groups);
      ptr_SFB_host.resize(options.groups);
    }

    for(int i = 0; i < options.groups; i++) {
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);
      auto L = 1;

      auto shape_A = cute::make_shape(M, K, L);
      auto shape_B = cute::make_shape(N, K, L);
      auto shape_CD = cute::make_shape(M, N, L);

      auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
      auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
      auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
      auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);

      stride_A_host.push_back(stride_a);
      stride_B_host.push_back(stride_b);
      stride_C_host.push_back(stride_c);
      stride_D_host.push_back(stride_d);

      initialize_block(block_A.at(i).get(), block_A.at(i).size(), seed + 2023 + i);
      initialize_block(block_B.at(i).get(), block_B.at(i).size(), seed + 2022 + i);
      initialize_block(block_C.at(i).get(), block_C.at(i).size(), seed + 2021 + i);

      cutlass::benchmark::convert_dtype<ElementA, ElementMMAVerify, BenchmarkRunnerGemm>(
          block_A.at(i).get(), block_A_dq.at(i).get(), block_A.at(i).size()
      );
      cutlass::benchmark::convert_dtype<ElementB, ElementMMAVerify, BenchmarkRunnerGemm>(
          block_B.at(i).get(), block_B_dq.at(i).get(), block_B.at(i).size()
      );

      if constexpr (is_blocked_scaled<CollectiveMainloop>) {
        const int scale_k = cute::ceil_div(K, GROUP_SIZE);
        const int padded_M = cute::round_up(M, ScaleAlignA);
        const int padded_N = cute::round_up(N, ScaleAlignB);
        auto shape_scale_A = cute::make_shape(padded_M, scale_k, L);
        auto shape_scale_B = cute::make_shape(padded_N, scale_k, L);
        auto stride_sfa = cutlass::make_cute_packed_stride(StrideScaleA{}, shape_scale_A);
        auto stride_sfb = cutlass::make_cute_packed_stride(StrideScaleB{}, shape_scale_B);
        stride_SFA_host.push_back(stride_sfa);
        stride_SFB_host.push_back(stride_sfb);

        initialize_scale(block_scaleA.at(i).get(), block_scaleA.at(i).size(), options);
        initialize_scale(block_scaleB.at(i).get(), block_scaleB.at(i).size(), options);

        auto layout_A = make_layout(shape_A, stride_a);
        auto layout_B = make_layout(shape_B, stride_b);
        auto layout_scale_A = make_layout(shape_scale_A, stride_sfa);
        auto layout_scale_B = make_layout(shape_scale_B, stride_sfb);

        apply_scale(block_A_dq.at(i).get(), block_A.at(i).get(), layout_A, block_scaleA.at(i).get(),  layout_scale_A);
        apply_scale(block_B_dq.at(i).get(), block_B.at(i).get(), layout_B, block_scaleB.at(i).get(),  layout_scale_B);
      }

      ptr_A_host.at(i) = block_A.at(i).get();
      ptr_B_host.at(i) = block_B.at(i).get();
      ptr_C_host.at(i) = block_C.at(i).get();
      ptr_D_host.at(i) = block_D.at(i).get();
      
      if constexpr (is_blocked_scaled<CollectiveMainloop>) {
        ptr_SFA_host.at(i) = block_scaleA.at(i).get();
        ptr_SFB_host.at(i) = block_scaleB.at(i).get();
      }

      alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
      beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
      // Fill host ptr vectors with offset addresses into device alpha/beta blocks
      ptr_alpha_host.at(i) = block_alpha.get() + i;
      ptr_beta_host.at(i) = block_beta.get() + i;
    }
    // Allocate device memory & copy from host
    try {
    ptr_A.reset(options.groups);
    ptr_A.copy_from_host(ptr_A_host.data());

    ptr_B.reset(options.groups);
    ptr_B.copy_from_host(ptr_B_host.data());
    
    ptr_C.reset(options.groups);
    ptr_C.copy_from_host(ptr_C_host.data());

    ptr_D.reset(options.groups);
    ptr_D.copy_from_host(ptr_D_host.data());

    if constexpr (is_blocked_scaled<CollectiveMainloop>) {
      ptr_SFA.reset(options.groups);
      ptr_SFA.copy_from_host(ptr_SFA_host.data());

      ptr_SFB.reset(options.groups);
      ptr_SFB.copy_from_host(ptr_SFB_host.data());
    }

    stride_A.reset(options.groups);
    stride_A.copy_from_host(stride_A_host.data());

    stride_B.reset(options.groups);
    stride_B.copy_from_host(stride_B_host.data());

    stride_C.reset(options.groups);
    stride_C.copy_from_host(stride_C_host.data());

    stride_D.reset(options.groups);
    stride_D.copy_from_host(stride_D_host.data());

    if constexpr (is_blocked_scaled<CollectiveMainloop>) {
      stride_SFA.reset(options.groups);
      stride_SFA.copy_from_host(stride_SFA_host.data());

      stride_SFB.reset(options.groups);
      stride_SFB.copy_from_host(stride_SFB_host.data());
    }

    // Per-group alpha and beta ptrs
    alpha_device.reset(options.groups);
    alpha_device.copy_from_host(ptr_alpha_host.data());
    beta_device.reset(options.groups);
    beta_device.copy_from_host(ptr_beta_host.data());

    block_alpha.copy_from_host(alpha_host.data());
    block_beta.copy_from_host(beta_host.data());

    } catch (std::exception const &e) {
      state.SkipWithError(e.what());
    }
  }

  void run(::benchmark::State& state, const GroupedGEMMOptions& options, const KernelHardwareInfo& hw_info) {
    allocate(options);
    initialize(state, options);

    auto args_tuple = args_from_options(options, hw_info);
    
    auto mainloop_args = [&]() {
      if constexpr (!is_blocked_scaled<CollectiveMainloop>) {
        return typename Gemm::GemmKernel::MainloopArguments{
          ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()
        };
      } else {
        return typename Gemm::GemmKernel::MainloopArguments{
          ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get(),
          ptr_SFA.get(), stride_SFA.get(), ptr_SFB.get(), stride_SFB.get(),
        };
      }
    }();
    
    typename Gemm::GemmKernel::Arguments arguments {
      get<0>(args_tuple), get<1>(args_tuple),
      mainloop_args,
      typename Gemm::GemmKernel::EpilogueArguments{get<2>(args_tuple), ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      get<3>(args_tuple), get<4>(args_tuple)
    };

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

#ifndef CUTLASS_TEST_FOR_CRI
    // Run warmup on real hardware (skip on CRI simulator as it's time-consuming)
#else
    gemm_op.run();
    compat::wait();
#endif

    state.counters["m"] = options.m;
    state.counters["n"] = options.n;
    state.counters["k"] = options.k;
    state.counters["l"] = options.l;
    state.counters["alpha"] = options.alpha;
    state.counters["beta"] = options.beta;

    // Report distribution parameters
    state.counters["groups"] = options.groups;
    state.counters["tokens_per_gpu"] = (options.m * options.topk) / options.ep_size;
    state.counters["ep_size"] = options.ep_size;
    state.counters["topk"] = options.topk;
    state.counters["num_experts"] = options.num_experts;

    for (int i = 0; i < options.groups; i++) {
      state.counters["M_" + std::to_string(i)] = get<0>(options.problem_sizes_host[i]);
    }

    std::cerr << "[DIST] " << options.benchmark_name() << " M_per_expert=[";
    for (int i = 0; i < options.groups; i++) {
      std::cerr << get<0>(options.problem_sizes_host[i]);
      if (i < options.groups - 1) std::cerr << ",";
    }
    std::cerr << "]" << std::endl;

    std::stringstream extra_label;
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


    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();

    for (auto const & problem : options.problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = static_cast<uint64_t>(2) * static_cast<uint64_t>(fmas);
    double gflop = double(flop) / double(1.0e9);

    // Compatible with data types smaller than 8 bits here
    constexpr double bits_per_byte = static_cast<double>(sizeof_bits_v<char>);
    constexpr double sizeof_a = sizeof_bits_v<ElementA> / bits_per_byte;
    constexpr double sizeof_b = sizeof_bits_v<ElementB> / bits_per_byte;
    constexpr double sizeof_c = sizeof_bits_v<ElementC> / bits_per_byte;
    auto mega_bytes_transferred = static_cast<double>(
        options.m * options.k * sizeof_a +
        options.k * options.n * sizeof_b +
        (options.beta != 0 ? 2 : 1) * options.m * options.n * sizeof_c
      ) * 1e-6 * options.l;

    initialize_counters(state);
    for(auto _ : state) {
      state.PauseTiming();
      auto args_tuple = args_from_options(options, hw_info);
      
      auto mainloop_args = [&]() {
        if constexpr (!is_blocked_scaled<CollectiveMainloop>) {
          return typename Gemm::GemmKernel::MainloopArguments{
            ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()
          };
        } else {
          return typename Gemm::GemmKernel::MainloopArguments{
            ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get(),
            ptr_SFA.get(), stride_SFA.get(), ptr_SFB.get(), stride_SFB.get(),
          };
        }
      }();
      
      typename Gemm::GemmKernel::Arguments arguments {
        get<0>(args_tuple), get<1>(args_tuple),
        mainloop_args,
        typename Gemm::GemmKernel::EpilogueArguments{get<2>(args_tuple), ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
        get<3>(args_tuple), get<4>(args_tuple)
      };
      gemm_op.initialize(arguments, workspace.get());
      state.ResumeTiming();

      GPU_Clock timer;
      timer.start();
      gemm_op.run();
      auto ms_elapsed = timer.milliseconds();
      update_counters(state, ms_elapsed);
      state.SetIterationTime(ms_elapsed / 1000);
    }
    finalize_counters(state, gflop, mega_bytes_transferred);
    if (options.verify_mode == VerifyMode::Host) {
      bool passed = verify_host(state, options);
      if (!passed) {
        state.SkipWithError("Host reference verification FAILED.");
      }
    }
    else if  (options.verify_mode == VerifyMode::Device) {
      bool passed = verify(options);
      if(not passed) {
        state.SkipWithError("Disposition Failed.");
      }
    }
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
    state.counters["avg_throughput"] = mega_bytes_transferred / state.counters["avg_runtime_ms"];
    state.counters["best_tflop"] = gflop / state.counters["best_runtime_ms"];
    state.counters["best_bandwidth"] = mega_bytes_transferred / state.counters["best_runtime_ms"];
  }
};

}

#define CUTLASS_BENCHMARK(F) cutlass::benchmark::BenchmarkRegistry<cutlass::benchmark::GroupedGEMMOptions>::Register(#F, &F##_func)

#define CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(F)                          \
  static void F##_func(                                           \
      ::benchmark::State& state,                                  \
      cutlass::benchmark::GroupedGEMMOptions const& options,                 \
      cutlass::KernelHardwareInfo const& hw_info) {               \
    auto bench = cutlass::benchmark::BenchmarkRunnerGemm<F>();    \
    bench.run(state, options, hw_info);                           \
  }
