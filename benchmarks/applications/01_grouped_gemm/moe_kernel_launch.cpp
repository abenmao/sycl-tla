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
    \brief Device-codegen translation unit for the MoE grouped-GEMM benchmark:
           the ONLY TU that instantiates the MoE::MoEGEMM kernel (and so the only
           emitter of device SPIR-V). It includes the runner machinery + this
           file's thin interface, but NOT benchmark/oneMKL/common.hpp. See
           moe_kernel_launch.hpp for the TU-split rationale (avoids the IGC ICE).

    Allocation, scale setup, and the timed submission live here; the
    timed core (moe_launch_timed) and config/scale/verify machinery come from
    moe_grouped_gemm/runner/moe_gemm_runner.hpp. Everything is wrapped behind an
    opaque handle so the benchmark TU never sees cute / MoE / SYCL types.
*/

#include "moe_kernel_launch.hpp"

// The shared runner pulls in everything: cute, MoE::MoEGEMM, the persistent
// tile scheduler, choose_tiled_mma, fill_scale, ScaleKind, and the Config
// structs (Bf16Config / MxFp8E4m3Config / ...). It also includes GPU_Clock and
// the SYCL event manager, and consumes the kernel API under
// applications/moe_grouped_gemm/ directly (no dependency on examples/). NO
// benchmark / oneMKL headers here. Reached via ${CUTLASS_DIR}/applications on
// the include path.
#include "moe_grouped_gemm/runner/moe_gemm_runner.hpp"

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <stdexcept>
// The shared runner lives in namespace cutlass::moe; pull it in unqualified so
// the existing references (Config structs, moe_launch_timed, VerificationHelper,
// fill_scale, ScaleKind, ...) resolve without per-symbol qualification.
using namespace cutlass::moe;

// ONE config compiled per binary, selected by -DMOE_BENCH_CONFIG=<name> at
// build time (matches the runner header default). Bf16Config / MxFp8E4m3Config /
// ... are defined in the shared runner included above.
#ifndef MOE_BENCH_CONFIG
#define MOE_BENCH_CONFIG Bf16Config
#endif

namespace moe_bench {

// NOTE: the SYCL kernel-name tag is owned by the shared core (GemmCuteName)
// now that this TU delegates the launch to moe_launch_timed. Each benchmark
// binary compiles exactly one Config (-DMOE_BENCH_CONFIG, the IGC-ICE
// one-config-per-binary split), so there is a single kernel per binary and no
// cross-config name aliasing to guard against.

// Pull a config's scale-storage element type (only meaningful for scaled
// configs; Plain stays float and the pointers stay null).
template <class C, class = void> struct ConfigScale {
  using type = float;
};
template <class C>
struct ConfigScale<C, cute::void_t<typename C::ElementScale>> {
  using type = std::conditional_t<std::is_void_v<typename C::ElementScale>,
                                  float, typename C::ElementScale>;
};

// The opaque handle, fully defined here where the kernel types are known. It
// owns all device allocations (so they stay alive across the whole timed loop)
// and a type-erased `launch` closure that does ONE GPU_Clock-timed submission +
// wait and returns elapsed milliseconds. The benchmark TU only ever holds a
// MoeRunHandle* and calls the three free functions.
struct MoeRunHandle {
  // Type-erased owners for the device allocations. Each config has different
  // element types, so we hide them behind a vector of generic deleters captured
  // by the setup closure; simplest is to keep them alive inside the launch
  // closure's capture. To make teardown explicit we also keep a `release`
  // closure.
  std::function<double()> launch; // timed single launch -> elapsed ms
  std::function<void()> release;  // frees device allocations
};

template <class DeviceState, class LaunchFn>
MoeRunHandle *make_handle(DeviceState *st, LaunchFn timed_launch,
                          double verify_time_ms, int verify) {
  auto *handle = new MoeRunHandle();
  handle->launch  = (verify == kVerifyHost)
      ? std::function<double()>([verify_time_ms]() { return verify_time_ms; })
      : std::function<double()>([timed_launch]()   { return timed_launch(); });
  handle->release = [st]() { delete st; };
  return handle;
}

} // namespace moe_bench

namespace moe_bench {

// Config-templated implementation. The public non-template moe_setup (below)
// dispatches to this for the single MOE_BENCH_CONFIG this binary compiles, so
// the cute/MoE kernel is instantiated exactly once, only in this TU.
template <class Config>
MoeRunHandle *moe_setup_impl(int N, int K, int num_experts,
                             std::vector<int> const &M_per_expert,
                             std::string *error, int verify) {
  using ElementInput = typename Config::Element;
  // Output (D) element type is declared per Config (Config::ElementOutput);
  // all current paths emit bf16. The 16-bit XE_STORE_2D store atom is element-agnostic.
  using ElementOutput = typename Config::ElementOutput;

  const uint64_t seed = 2023;

  if (static_cast<int>(M_per_expert.size()) != num_experts) {
    if (error)
      *error = "num_experts does not match per-expert M list size.";
    return nullptr;
  }

  int num_tokens = 0;
  for (int m : M_per_expert)
    num_tokens += m;

  // Shared device state held alive by the handle. Allocated on the heap so it
  // outlives this function and survives until moe_teardown.
  struct DeviceState {
    cutlass::DeviceAllocation<int32_t> num_rows_per_expert_device;
    cutlass::DeviceAllocation<ElementInput> activations_data;
    cutlass::DeviceAllocation<ElementInput> weights_data;
    cutlass::DeviceAllocation<ElementOutput> output_data;
    using ElementScaleStore =
        std::conditional_t<Config::scale_kind == ScaleKind::Plain, float,
                           typename ConfigScale<Config>::type>;
    cutlass::DeviceAllocation<ElementScaleStore> scaleA_data;
    cutlass::DeviceAllocation<ElementScaleStore> scaleB_data;
  };
  using ElementScaleStore = typename DeviceState::ElementScaleStore;

  auto *st = new DeviceState();

  try {
    // ---- Per-expert M table (host -> device) ----
    st->num_rows_per_expert_device.reset(num_experts);
    st->num_rows_per_expert_device.copy_from_host(M_per_expert.data());

    // ---- A / B / D allocation (row-major packed, matches launcher<Config>) ----
    int64_t A_size = int64_t(num_tokens) * K;
    int64_t B_size = int64_t(num_experts) * N * K;
    int64_t D_size = int64_t(num_tokens) * N;
    st->activations_data.reset(A_size);
    st->weights_data.reset(B_size);
    st->output_data.reset(D_size);
    initialize_block(st->activations_data, seed + 2023);
    initialize_block(st->weights_data, seed + 2022);
    initialize_block(st->output_data, seed + 2021);
  } catch (std::exception const &e) {
    if (error)
      *error = e.what();
    delete st;
    return nullptr;
  }

  // ---- Scale allocation (padded MX layout) — identical to launcher<Config> ----
  // For Plain (BF16) the scale pointers stay null and group_n/group_k = 0.
  int group_n = 0;
  int group_k = 0;
  const ElementScaleStore *ptr_sA = nullptr;
  const ElementScaleStore *ptr_sB = nullptr;

  if constexpr (Config::scale_kind != ScaleKind::Plain) {
    constexpr bool is_tensor = (Config::scale_kind == ScaleKind::Tensor);
    constexpr int kScaleAlign = 64;
    // TENSOR path (BDPAS): scale_k = BLK_K / MMA_K (= 2 for fp8 64-deep tile).
    // BLOCK path: scale_k=ceil(K/GroupK), scale_n=ceil(N/GroupN).
    using TileShape__ = typename Config::TileShapeCri;
    constexpr int kScaleTensor = cute::get<2>(TileShape__{}) / 32; // BLK_K / MMA_K
    int scale_k, scale_n;
    if constexpr (is_tensor) {
      scale_k = kScaleTensor;
      scale_n = 1;
    } else {
      scale_k = (K + Config::group_k - 1) / Config::group_k;
      scale_n = (N + Config::group_n - 1) / Config::group_n;
    }
    const int padded_scale_n = ((scale_n + kScaleAlign - 1) / kScaleAlign) * kScaleAlign;
    // Kernel passes group_n=0, group_k=0 for tensor; keep that for the launch.
    group_k = Config::group_k;
    group_n = Config::group_n;

    int64_t scaleA_size, scaleB_size;
    if constexpr (is_tensor) {
      // scaleA: per-expert padded (round_up_M, 2) layout.
      int64_t padded_M_total = 0;
      for (int i = 0; i < num_experts; i++)
        padded_M_total += ((M_per_expert[i] + kScaleAlign - 1) / kScaleAlign) * int64_t(kScaleAlign);
      scaleA_size = padded_M_total * scale_k;
      scaleB_size = int64_t(num_experts) * padded_scale_n * scale_k;
    } else {
      // Block (MX): per-expert rows padded to 64.
      int64_t padded_M_total = 0;
      for (int i = 0; i < num_experts; i++)
        padded_M_total += ((M_per_expert[i] + kScaleAlign - 1) / kScaleAlign) * int64_t(kScaleAlign);
      scaleA_size = padded_M_total * scale_k;
      scaleB_size = int64_t(num_experts) * padded_scale_n * scale_k;
    }
    try {
      st->scaleA_data.reset(scaleA_size);
      st->scaleB_data.reset(scaleB_size);
    } catch (std::exception const &e) {
      if (error) *error = e.what();
      delete st;
      return nullptr;
    }
    fill_scale(st->scaleA_data, seed + 2020, is_tensor);
    fill_scale(st->scaleB_data, seed + 2019, /*constant=*/is_tensor);
    ptr_sA = st->scaleA_data.get();
    ptr_sB = st->scaleB_data.get();
  }

  // Raw device pointers — the DeviceAllocation wrappers are not device-copyable,
  // so only plain pointers cross into the timed launch.
  const ElementInput *ptr_A = st->activations_data.get();
  const ElementInput *ptr_B = st->weights_data.get();
  ElementOutput *ptr_D = st->output_data.get();
  const int32_t *ptr_rows = st->num_rows_per_expert_device.get();
  const ElementScaleStore *cap_sA = ptr_sA;
  const ElementScaleStore *cap_sB = ptr_sB;
  const int cap_group_n = group_n;
  const int cap_group_k = group_k;

  // The single timed launch closure — the one source of truth for the
  // submission geometry + parallel_for body, delegating to the shared core
  // moe_launch_timed. Warmup, the optional verify pre-run, and the benchmark
  // loop all reuse this exact closure, so the launch can never drift and setup
  // never pays for a duplicate launch. This TU stays the only one that
  // instantiates the device kernel (the IGC-ICE lean-TU isolation).
  //
  // The Plain (BF16) path passes typed-null void* scale pointers so ElementS
  // deduces to void and the kernel takes the non-scaled path; the scaled path
  // passes the real typed scale pointers.
  auto timed_launch = [=]() -> double {
    if constexpr (Config::scale_kind == ScaleKind::Plain) {
      return moe_launch_timed<Config>(
          ptr_A, ptr_B, static_cast<const void *>(nullptr),
          static_cast<const void *>(nullptr), ptr_D, N, K, ptr_rows,
          num_experts, cap_group_n, cap_group_k);
    } else {
      return moe_launch_timed<Config>(
          ptr_A, ptr_B, cap_sA, cap_sB, ptr_D, N, K, ptr_rows, num_experts,
          cap_group_n, cap_group_k);
    }
  };

  double verify_time_ms = 0.0;

  if (verify == kVerifyHost || verify == kVerifyDevice) {
    verify_time_ms = timed_launch();
    VerificationHelper helper;
    helper.parse(num_experts, M_per_expert.data(), N, K);
    bool ok = true;
    if constexpr (Config::scale_kind == ScaleKind::Plain) {
      ok = helper.verify(ptr_A, ptr_B, ptr_D);
    } else {
      constexpr bool kIsTensor = (Config::scale_kind == ScaleKind::Tensor);
      // Match the kernel's weight (B) layout so the host reference reads the
      // flat weight buffer the same way the kernel does.
      constexpr bool kBColMajor =
          cute::is_same_v<typename Config::LayoutB, cutlass::layout::ColumnMajor>;
      // Tensor configs store group_k=0/group_n=0; verify_scaled divides by them,
      // so substitute K/N (one scale per K/N tile) as the double-buffer path does.
      const int verify_group_n = kIsTensor ? N : group_n;
      const int verify_group_k = kIsTensor ? K : group_k;
      sycl::queue Q = compat::get_default_queue();
      ok = helper.template verify_scaled<kIsTensor, kBColMajor>(
          Q, ptr_A, ptr_B, ptr_sA, ptr_sB, ptr_D, verify_group_n, verify_group_k);
    }
    std::cerr << "[MoE bench verify] "
              << (ok ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m")
              << std::endl;
  }

  return make_handle(st, timed_launch, verify_time_ms, verify);
}

// (legacy single-config moe_setup removed: the tile-sweep registry
// moe_setup_by_name supersedes it; keeping it would duplicate a kernel
// instantiation and collide on the GemmCuteName mangled name.)

// Double-buffer uniform-M setup: all experts share the same M. Calls moe_launch_timed_double_buffer
// instead of moe_launch_timed; no scale pointers, no per-expert M table.
template <class Config>
MoeRunHandle *moe_setup_impl_double_buffer(int N, int K, int num_experts,
                                     std::vector<int> const &M_per_expert,
                                     std::string *error, int verify) {
  using ElementInput  = typename Config::Element;
  using ElementOutput = typename Config::ElementOutput;
  const uint64_t seed = 2023;

  if (static_cast<int>(M_per_expert.size()) != num_experts) {
    if (error) *error = "num_experts does not match per-expert M list size.";
    return nullptr;
  }
  // Verify uniform M assumption.
  int uniform_m = M_per_expert[0];
  for (int m : M_per_expert) {
    if (m != uniform_m) {
      if (error) *error = "Uniform-M kernel requires all experts to have equal M.";
      return nullptr;
    }
  }
  int num_tokens = uniform_m * num_experts;

  struct DeviceState {
    cutlass::DeviceAllocation<ElementInput>  activations_data;
    cutlass::DeviceAllocation<ElementInput>  weights_data;
    cutlass::DeviceAllocation<ElementOutput> output_data;
  };
  auto *st = new DeviceState();
  try {
    st->activations_data.reset(int64_t(num_tokens) * K);
    st->weights_data.reset(int64_t(num_experts) * N * K);
    st->output_data.reset(int64_t(num_tokens) * N);
    initialize_block(st->activations_data, seed + 2023);
    initialize_block(st->weights_data,     seed + 2022);
    initialize_block(st->output_data,      seed + 2021);
  } catch (std::exception const &e) {
    if (error) *error = e.what();
    delete st;
    return nullptr;
  }

  const ElementInput  *ptr_A = st->activations_data.get();
  const ElementInput  *ptr_B = st->weights_data.get();
  ElementOutput       *ptr_D = st->output_data.get();
  const int cap_uniform_m    = uniform_m;

  auto timed_launch = [=]() -> double {
    return moe_launch_timed_double_buffer<Config>(
        ptr_A, ptr_B, ptr_D, cap_uniform_m, num_experts, N, K);
  };

  double verify_time_ms = 0.0;

  if (verify == kVerifyHost || verify == kVerifyDevice) {
    verify_time_ms = timed_launch();
    VerificationHelper helper;
    std::vector<int> rows(num_experts, uniform_m);
    helper.parse(num_experts, rows.data(), N, K);
    bool ok = helper.verify(ptr_A, ptr_B, ptr_D);
    std::cerr << "[MoE bench double buffer verify] "
              << (ok ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m")
              << std::endl;
  }

  return make_handle(st, timed_launch, verify_time_ms, verify);
}


// Scaled double-buffer uniform-M setup: uniform M, with scale pointers (MX/FP8 paths).
template <class Config>
MoeRunHandle *moe_setup_impl_double_buffer_scaled(int N, int K, int num_experts,
                                            std::vector<int> const &M_per_expert,
                                            std::string *error, int verify) {
  using ElementInput  = typename Config::Element;
  using ElementOutput = typename Config::ElementOutput;
  using ElementScaleStore = typename ConfigScale<Config>::type;
  const uint64_t seed = 2023;

  if (static_cast<int>(M_per_expert.size()) != num_experts) {
    if (error) *error = "num_experts does not match per-expert M list size.";
    return nullptr;
  }
  int uniform_m = M_per_expert[0];
  for (int m : M_per_expert) {
    if (m != uniform_m) {
      if (error) *error = "Uniform-M scaled kernel requires all experts to have equal M.";
      return nullptr;
    }
  }
  int num_tokens = uniform_m * num_experts;

  constexpr bool is_tensor = (Config::scale_kind == ScaleKind::Tensor);
  constexpr int kScaleAlign = 64;
  // TENSOR (BDPAS): scale_k = BLK_K / MMA_K where MMA_K=32 for fp8 (XE_BDPAS_TT<8>).
  // This equals 2 for the current 64-deep fp8 tile (BLK_K=64, MMA_K=32).
  using TileShape_ = typename Config::TileShapeCri;
  constexpr int BLK_K_  = cute::get<2>(TileShape_{});
  constexpr int MMA_K_  = 32; // XE_BDPAS_TT<8> MMA atom K = SG_K * DPAS_K = 4*8 = 32
  constexpr int kScaleTensor = BLK_K_ / MMA_K_;
  int scale_k, scale_n;
  if constexpr (is_tensor) {
    scale_k = kScaleTensor;
    scale_n = 1;
  } else {
    scale_k = (K + Config::group_k - 1) / Config::group_k;
    scale_n = (N + Config::group_n - 1) / Config::group_n;
  }
  const int padded_scale_n = ((scale_n + kScaleAlign - 1) / kScaleAlign) * kScaleAlign;

  struct DeviceState {
    cutlass::DeviceAllocation<ElementInput>      activations_data;
    cutlass::DeviceAllocation<ElementInput>      weights_data;
    cutlass::DeviceAllocation<ElementOutput>     output_data;
    cutlass::DeviceAllocation<ElementScaleStore> scaleA_data;
    cutlass::DeviceAllocation<ElementScaleStore> scaleB_data;
  };
  auto *st = new DeviceState();

  try {
    st->activations_data.reset(int64_t(num_tokens) * K);
    st->weights_data.reset(int64_t(num_experts) * N * K);
    st->output_data.reset(int64_t(num_tokens) * N);
    initialize_block(st->activations_data, seed + 2023);
    initialize_block(st->weights_data,     seed + 2022);
    initialize_block(st->output_data,      seed + 2021);

    const int round_up_M = (uniform_m + kScaleAlign - 1) & ~(kScaleAlign - 1);
    // TENSOR: scaleA is (padded_M, 2) per expert; scaleB is (padded_scale_n, 2)
    //         where padded_scale_n = ceil(SG_N / 64)*64 (always 64 for current tiles).
    // BLOCK:  same padded layout, scale_n derived from GroupN.
    const int64_t scaleA_size = int64_t(num_experts) * round_up_M * scale_k;
    const int64_t scaleB_size = int64_t(num_experts) * padded_scale_n * scale_k;
    st->scaleA_data.reset(scaleA_size);
    st->scaleB_data.reset(scaleB_size);
    fill_scale(st->scaleA_data, seed + 2020, is_tensor);
    fill_scale(st->scaleB_data, seed + 2019, is_tensor);
  } catch (std::exception const &e) {
    if (error) *error = e.what();
    delete st;
    return nullptr;
  }

  const ElementInput      *ptr_A  = st->activations_data.get();
  const ElementInput      *ptr_B  = st->weights_data.get();
  ElementOutput           *ptr_D  = st->output_data.get();
  const ElementScaleStore *ptr_sA = st->scaleA_data.get();
  const ElementScaleStore *ptr_sB = st->scaleB_data.get();
  const int cap_uniform_m = uniform_m;
  // Mirror moe_setup_impl: tensor path passes group_n=N, group_k=K to verify_scaled.
  const int verify_group_n = is_tensor ? N : Config::group_n;
  const int verify_group_k = is_tensor ? K : Config::group_k;

  auto timed_launch = [=]() -> double {
    return moe_launch_timed_double_buffer_scaled<Config>(
        ptr_A, ptr_B, ptr_sA, ptr_sB, ptr_D,
        cap_uniform_m, num_experts, N, K, Config::group_n, Config::group_k);
  };

  double verify_time_ms = 0.0;

  if (verify == kVerifyHost || verify == kVerifyDevice) {
    verify_time_ms = timed_launch();
    VerificationHelper helper;
    std::vector<int> rows(num_experts, uniform_m);
    helper.parse(num_experts, rows.data(), N, K);
    constexpr bool kIsTensor   = is_tensor;
    constexpr bool kBColMajor =
        cute::is_same_v<typename Config::LayoutB, cutlass::layout::ColumnMajor>;
    sycl::queue Q = compat::get_default_queue();
    bool ok = helper.template verify_scaled<kIsTensor, kBColMajor>(
        Q, ptr_A, ptr_B, ptr_sA, ptr_sB, ptr_D, verify_group_n, verify_group_k);
    std::cerr << "[MoE bench double buffer scaled verify] "
              << (ok ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m")
              << std::endl;
  }

  return make_handle(st, timed_launch, verify_time_ms, verify);
}

// ---- TILE SWEEP: name -> setup-fn registry (all cute types stay in this TU) ----
// Each (dtype,tile) config registers moe_setup_impl<Config> under its string
// name. moe_setup_by_name() looks it up so the benchmark TU can pick a tile at
// runtime from the .in line's first token, with NO cute type crossing the TU
// boundary. Multiple configs in ONE binary is now safe because GemmCuteName
// encodes the tile (unique SYCL kernel name per tile).
namespace {
using SetupFn = MoeRunHandle *(*)(int, int, int, std::vector<int> const &,
                                  std::string *, int);
std::map<std::string, SetupFn> &moe_setup_registry() {
  static std::map<std::string, SetupFn> r;
  return r;
}
struct MoeSetupRegistrar {
  MoeSetupRegistrar(const char *name, SetupFn fn) {
    moe_setup_registry().emplace(name, fn);
  }
};
} // namespace

// Register one (name, Config). Used by the per-binary tile list below.
#define MOE_REGISTER_TILE(NAME, CONFIG)    static MoeRunHandle *moe_setup_##NAME(  int N, int K, int ne, std::vector<int> const &M, std::string *e, int v) {  return moe_setup_impl<CONFIG>(N, K, ne, M, e, v);   }  static MoeSetupRegistrar moe_reg_##NAME(#NAME, &moe_setup_##NAME);

// Register a double-buffer uniform-M config (calls moe_setup_impl_double_buffer instead).
#define MOE_REGISTER_TILE_DOUBLE_BUFFER(NAME, CONFIG)                            static MoeRunHandle *moe_setup_##NAME(                                             int N, int K, int ne, std::vector<int> const &M, std::string *e,               int v) {                                                                     return moe_setup_impl_double_buffer<CONFIG>(N, K, ne, M, e, v);              }                                                                              static MoeSetupRegistrar moe_reg_##NAME(#NAME, &moe_setup_##NAME);

// Register a scaled double-buffer uniform-M config (calls moe_setup_impl_double_buffer_scaled).
#define MOE_REGISTER_TILE_DOUBLE_BUFFER_SCALED(NAME, CONFIG)                     static MoeRunHandle *moe_setup_##NAME(                                             int N, int K, int ne, std::vector<int> const &M, std::string *e,               int v) {                                                                     return moe_setup_impl_double_buffer_scaled<CONFIG>(N, K, ne, M, e, v);       }                                                                              static MoeSetupRegistrar moe_reg_##NAME(#NAME, &moe_setup_##NAME);

// The per-binary tile list comes from moe_tile_list.hpp, selected by the
// -DMOE_DTYPE_<TAG> this binary is built with. Expand each X(NAME,CONFIG) into a
// MOE_REGISTER_TILE registration so every tile this binary holds is in the
// name->setup table.
#include "moe_tile_list.hpp"
#ifdef MOE_TILE_X_LIST
#define X(NAME, CONFIG) MOE_REGISTER_TILE(NAME, CONFIG)
#define X_DOUBLE_BUFFER(NAME, CONFIG) MOE_REGISTER_TILE_DOUBLE_BUFFER(NAME, CONFIG)
#define X_DOUBLE_BUFFER_SCALED(NAME, CONFIG) MOE_REGISTER_TILE_DOUBLE_BUFFER_SCALED(NAME, CONFIG)
MOE_TILE_X_LIST
#undef X
#undef X_DOUBLE_BUFFER
#undef X_DOUBLE_BUFFER_SCALED
#endif

MoeRunHandle *moe_setup_by_name(const char *name, int N, int K, int num_experts,
                                std::vector<int> const &M_per_expert,
                                std::string *error, int verify) {
  auto &r = moe_setup_registry();
  auto it = r.find(name);
  if (it == r.end()) {
    if (error)
      *error = std::string("no registered tile config named '") + name + "'";
    return nullptr;
  }
  return it->second(N, K, num_experts, M_per_expert, error, verify);
}

double moe_launch_once(MoeRunHandle *handle) { return handle->launch(); }

void moe_teardown(MoeRunHandle *handle) {
  if (!handle)
    return;
  handle->release();
  delete handle;
}

} // namespace moe_bench
