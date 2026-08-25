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
    \brief Device-codegen TU for the MoE grouped-GEMM benchmark: the only TU that
           instantiates MoE::MoEGEMM (and so emits device SPIR-V). See moe_api.hpp
           for the TU-split rationale. Entry point: launch_moe().
*/

#include "moe_api.hpp"

// The shared runner pulls in cute, MoE::MoEGEMM, the tile scheduler, and the
// Config structs. No benchmark / oneMKL headers here.
#include "moe_grouped_gemm/runner/moe_gemm_runner.hpp"

#include "moe_grouped_gemm/runner/moe_device_state.hpp"
#include "cutlass/util/device_memory.h"

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>
using namespace cutlass::moe;

// Compiles the device kernels for whichever dtypes the binary enables
// (-DMOE_DTYPE_<TAG>, or all under -DMOE_BENCH_ALL). Each compiled tile registers
// a per-Config run thunk via the X-macro list below; launch_moe() picks one.

namespace moe_bench {

template <class Config>
using VendorTM = cutlass::moe::VendorTensorMapping<
    typename Config::Element, cutlass::moe::ScaleStoreFor<Config>,
    typename Config::ElementOutput>;

// The VerifyKind int from the harness -> the runner's MoeVerify. Anything
// unrecognized is treated as None rather than silently verifying.
inline cutlass::moe::MoeVerify to_moe_verify(int verify) {
  switch (verify) {
  case kVerifyDevice: return cutlass::moe::MoeVerify::Device;
  case kVerifyHost:   return cutlass::moe::MoeVerify::Host;
  default:            return cutlass::moe::MoeVerify::None;
  }
}

// Run the reference for one finished launch and print PASSED/FAILED. Shared by
// both kernel impls so the two can't drift apart; `label` names the kernel in the
// print. Diagnostic only: a mismatch is reported but does not fail the benchmark
// line -- the launch already produced a valid timing, matching the pre-existing
// behavior.
template <class Config>
void run_verification(const VendorTM<Config> &host_tm, const int *counts,
                      int num_experts, int N, int K, int verify,
                      const char *label) {
#ifdef FULL_RUN_TIMING_AND_VERIFY
  const auto mode = to_moe_verify(verify);
  if (mode == cutlass::moe::MoeVerify::None)
    return;
  const char *ref = mode == cutlass::moe::MoeVerify::Device ? "device" : "host";
  bool ok = false;
  try {
    ok = cutlass::moe::moe_verify_output<Config>(host_tm, counts, num_experts, N,
                                                 K, mode);
  } catch (std::exception const &e) {
    // Device verify allocates FP32 reference buffers and launches a reference
    // GEMM; contain an OOM/SYCL error here so it can't escape launch_moe.
    std::cerr << "[MoE bench " << label << " verify, " << ref
              << " reference] \033[31mFAILED\033[0m (" << e.what() << ")\n";
    return;
  }
  std::cerr << "[MoE bench " << label << " verify, " << ref << " reference] "
            << (ok ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m")
            << std::endl;
#else
  // Verify compiled out with the timing/verify build. The .in config selects the
  // mode, so a build without it just runs the kernel -- no print, no overhead.
  (void)host_tm; (void)counts; (void)num_experts; (void)N; (void)K; (void)verify;
  (void)label;
#endif
}

// Double-buffer uniform-M run (plain BF16 and all scaled paths).
template <class Config>
double moe_run_impl_double_buffer(const void *vendor_tm, int verify,
                                  std::string *error) {
  auto const &host_tm = *static_cast<VendorTM<Config> const *>(vendor_tm);
  const int N = host_tm.N, K = host_tm.K, num_experts = host_tm.num_experts;
  const int *M_per_expert = host_tm.experts_token_count;

  int uniform_m = num_experts > 0 ? M_per_expert[0] : 0;
  for (int g = 0; g < num_experts; ++g) {
    if (M_per_expert[g] != uniform_m) {
      if (error) *error = "Double-buffer kernel requires all experts to have equal M.";
      return -1.0;
    }
  }

  // One timed launch; the reference below reads host_tm.y.
  double ms;
  try {
    ms = cutlass::moe::moe_run_double_buffer<Config>(host_tm);
  } catch (std::exception const &e) {
    if (error) *error = e.what();
    return -1.0;
  }

  // This kernel took M as a scalar, so hand the reference the same M repeated.
  std::vector<int> rows(num_experts, uniform_m);
  run_verification<Config>(host_tm, rows.data(), num_experts, N, K, verify,
                           "double buffer");
  return ms;
}

// GREEDY run (plain BF16 + all scaled paths). Handles both uniform and dynamic M
// (Config::is_dynamic_m selects the kernel instantiation); if constexpr on
// scale_kind selects the launch.
template <class Config>
double moe_run_impl_greedy(const void *vendor_tm, int verify,
                           std::string *error) {
  using ElementInput  = typename Config::Element;
  using ElementOutput = typename Config::ElementOutput;
  using ElementScaleStore = cutlass::moe::ScaleStoreFor<Config>;

  auto const &host_tm = *static_cast<VendorTM<Config> const *>(vendor_tm);
  const int N = host_tm.N, K = host_tm.K, num_experts = host_tm.num_experts;
  const int *M_per_expert = host_tm.experts_token_count;
  // Uniform per-expert M (host-side), passed as a scalar so the uniform-M kernel
  // path needn't read it back from the device counts array.
  const int32_t uniform_m = num_experts > 0 ? M_per_expert[0] : 0;

  double ms;
  try {
    if constexpr (Config::scale_kind == ScaleKind::Plain) {
      ms = cutlass::moe::moe_launch_timed_greedy<Config>(host_tm);
    } else {
      ms = cutlass::moe::moe_launch_timed_greedy_scaled<
          Config, ElementInput, ElementInput, ElementScaleStore, ElementOutput>(
          host_tm.scatter_tokens, host_tm.experts_weight,
          host_tm.packed_scale_a, host_tm.packed_scale_b, host_tm.y, N, K,
          host_tm.experts_token_count_device, num_experts, uniform_m);
    }
  } catch (std::exception const &e) {
    if (error) *error = e.what();
    return -1.0;
  }
  run_verification<Config>(host_tm, M_per_expert, num_experts, N, K, verify,
                           "greedy");
  return ms;
}

// Tile registration. Each X() entry generates a per-tile run function bound to
// its exact Config, registered with its geometry + KernelKind; launch_moe() calls
// it via chosen.run() (the Config can't be named from the dtype string alone).
// GREEDY tile — geometry from the large_bucket tile (the greedy peel tile).
#define MOE_REGISTER_TILE_GREEDY(DTYPE, NAME, CONFIG)                          \
  static double moe_run_##NAME(const void *vendor_tm, int v,                   \
                               std::string *e) {                               \
    return moe_run_impl_greedy<CONFIG>(vendor_tm, v, e);                       \
  }                                                                            \
  static const bool moe_reg_##NAME = (cutlass::moe::moe_register_tile(          \
      #DTYPE,                                                                  \
      cutlass::moe::TileGeom{                                                   \
          static_cast<int>(cute::get<0>(CONFIG::LargeBucketTile{})),           \
          static_cast<int>(cute::get<1>(CONFIG::LargeBucketTile{})),           \
          static_cast<int>(cute::get<2>(CONFIG::LargeBucketTile{})), #NAME,    \
          /*is_db=*/false, /*is_dynamic_m=*/CONFIG::is_dynamic_m},             \
      cutlass::moe::KernelKind::Greedy,                                         \
      &moe_run_##NAME), true);

// mxfp4 BigK greedy tile: same as GREEDY but flags is_bigk=true so the host
// selects it for mxfp4 at large K (see pick_best_solution).
#define MOE_REGISTER_TILE_GREEDY_BIGK(DTYPE, NAME, CONFIG)                     \
  static double moe_run_##NAME(const void *vendor_tm, int v,                   \
                               std::string *e) {                               \
    return moe_run_impl_greedy<CONFIG>(vendor_tm, v, e);                       \
  }                                                                            \
  static const bool moe_reg_##NAME = (cutlass::moe::moe_register_tile(          \
      #DTYPE,                                                                  \
      cutlass::moe::TileGeom{                                                   \
          static_cast<int>(cute::get<0>(CONFIG::LargeBucketTile{})),           \
          static_cast<int>(cute::get<1>(CONFIG::LargeBucketTile{})),           \
          static_cast<int>(cute::get<2>(CONFIG::LargeBucketTile{})), #NAME,    \
          /*is_db=*/false, /*is_dynamic_m=*/CONFIG::is_dynamic_m,              \
          /*is_bigk=*/true},                                                   \
      cutlass::moe::KernelKind::Greedy,                                         \
      &moe_run_##NAME), true);

// Double-buffer tile — geometry flagged is_db so tile-select applies the DB policy.
#define MOE_REGISTER_TILE_DOUBLE_BUFFER(DTYPE, NAME, CONFIG)                   \
  static double moe_run_##NAME(const void *vendor_tm, int v,                   \
                               std::string *e) {                               \
    return moe_run_impl_double_buffer<CONFIG>(vendor_tm, v, e);                \
  }                                                                            \
  static const bool moe_reg_##NAME = (cutlass::moe::moe_register_tile(          \
      #DTYPE,                                                                  \
      cutlass::moe::TileGeom{                                                   \
          static_cast<int>(cute::get<0>(CONFIG::TileShapeCri{})),              \
          static_cast<int>(cute::get<1>(CONFIG::TileShapeCri{})),              \
          static_cast<int>(cute::get<2>(CONFIG::TileShapeCri{})), #NAME,        \
          /*is_db=*/true, /*is_dynamic_m=*/false},                             \
      cutlass::moe::KernelKind::DoubleBuffer,                                   \
      &moe_run_##NAME), true);

// Scaled double-buffer: same impl as plain; if constexpr inside
// moe_run_impl_double_buffer selects the scaled verify path.
#define MOE_REGISTER_TILE_DOUBLE_BUFFER_SCALED(DTYPE, NAME, CONFIG)            \
  MOE_REGISTER_TILE_DOUBLE_BUFFER(DTYPE, NAME, CONFIG)

// The per-binary tile list (selected by -DMOE_DTYPE_<TAG>) expands each entry
// into a registration, joining every compiled tile to its dtype's candidate list.
#include "moe_grouped_gemm/runner/moe_tile_list.hpp"
#ifdef MOE_TILE_X_LIST
#define X(DTYPE, NAME, CONFIG) MOE_REGISTER_TILE_GREEDY(DTYPE, NAME, CONFIG)
#define X_GREEDY_BIGK(DTYPE, NAME, CONFIG) MOE_REGISTER_TILE_GREEDY_BIGK(DTYPE, NAME, CONFIG)
#define X_DOUBLE_BUFFER(DTYPE, NAME, CONFIG) MOE_REGISTER_TILE_DOUBLE_BUFFER(DTYPE, NAME, CONFIG)
#define X_DOUBLE_BUFFER_SCALED(DTYPE, NAME, CONFIG) MOE_REGISTER_TILE_DOUBLE_BUFFER_SCALED(DTYPE, NAME, CONFIG)
MOE_TILE_X_LIST
#undef X
#undef X_GREEDY_BIGK
#undef X_DOUBLE_BUFFER
#undef X_DOUBLE_BUFFER_SCALED
#endif

// Single entry point (see moe_api.hpp): read the problem shape from vendor_tm,
// pick the best tile for this dtype, then run it. Returns elapsed ms (-1.0 on
// failure, *error set).
double launch_moe(const void *vendor_tm, const char *dtype, int verify,
                  std::string *error, bool force_greedy) {
  // Leading fields are at identical offsets in every instantiation.
  auto const &dims =
      *static_cast<cutlass::moe::VendorTensorMapping<char, char, char> const *>(
          vendor_tm);
  const int N = dims.N, K = dims.K, num_experts = dims.num_experts;

  // Decide uniform-vs-dynamic M ONCE here (the counts array first becomes flat
  // here), then pass it down -- no re-scanning in the selector or kernel. M is the
  // uniform per-expert count (only meaningful when !dynamic_m).
  const int *counts = dims.experts_token_count;
  const int M = num_experts > 0 ? counts[0] : 0;
  bool dynamic_m = false;
  for (int e = 1; e < num_experts; ++e)
    if (counts[e] != counts[0]) { dynamic_m = true; break; }

  const cutlass::moe::TileEntry *chosen_ptr =
      cutlass::moe::select_tile(dtype, N, K, M, dynamic_m, error, force_greedy);
  if (!chosen_ptr)
    return -1.0;
  auto const &chosen = *chosen_ptr;
  #ifdef MOE_DEBUG_PRINT
  // Greedy picks its M-tile per expert ON-DEVICE, so there is no host-side tile
  // to report -- only double buffer has a fixed WG tile.
  std::cerr << "[MoE tile-select] dtype='" << dtype << "' N=" << N << " K=" << K
            << " experts=" << num_experts << " -> ";
  if (chosen.kind == cutlass::moe::KernelKind::DoubleBuffer)
    std::cerr << "DOUBLE-BUFFER tile '" << chosen.geom.name << "' ("
              << chosen.geom.blk_m << "x" << chosen.geom.blk_n << "x"
              << chosen.geom.blk_k << ")";
  else
    std::cerr << "GREEDY (per-expert split chosen in kernel)";
  std::cerr << std::endl;
  #endif
  return chosen.run(vendor_tm, verify, error);
}

} // namespace moe_bench
