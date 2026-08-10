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
           instantiates the MoE::MoEGEMM kernel (and so emits device SPIR-V). It
           includes the runner machinery but not benchmark/oneMKL/common.hpp. See
           moe_api.hpp for the TU-split rationale. Entry point: launch_moe().
*/

#include "moe_api.hpp"

// The shared runner pulls in cute, MoE::MoEGEMM, the tile scheduler, fill_scale,
// GPU_Clock, and the Config structs. No benchmark / oneMKL headers here.
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
// a per-Config run thunk via the X-macro list below; launch_moe() selects the tile.

namespace moe_bench {

template <class Config>
using VendorTM = cutlass::moe::VendorTensorMapping<
    typename Config::Element, cutlass::moe::ScaleStoreFor<Config>,
    typename Config::ElementOutput>;

template <class Config>
double moe_run_impl(const void *vendor_tm, int verify, std::string *error) {
  using ElementInput = typename Config::Element;
  using ElementOutput = typename Config::ElementOutput;

  auto const &host_tm = *static_cast<VendorTM<Config> const *>(vendor_tm);
  const int N = host_tm.N, K = host_tm.K, num_experts = host_tm.num_experts;
  const int *M_per_expert = host_tm.experts_token_count;

  sycl::queue Q = compat::get_default_queue();

  const ElementInput *ptr_A = host_tm.scatter_tokens;
  const ElementInput *ptr_B = host_tm.experts_weight;
  ElementOutput *ptr_D = host_tm.y;

  // One timed launch; verify (below) reads ptr_D.
  double ms;
  try {
    ms = cutlass::moe::moe_run<Config>(host_tm);
  } catch (std::exception const &e) {
    if (error) *error = e.what();
    return -1.0;
  }
  #ifdef FULL_RUN_TIMING_AND_VERIFY
  if (verify == kVerifyHost || verify == kVerifyDevice) {
    VerificationHelper helper;
    helper.parse(num_experts, M_per_expert, N, K);
    bool ok = true;
    if constexpr (Config::scale_kind == ScaleKind::Plain) {
      ok = helper.verify(ptr_A, ptr_B, ptr_D);
    } else {
      constexpr bool kIsTensor = (Config::scale_kind == ScaleKind::Tensor);
      constexpr bool kBColMajor =
          cute::is_same_v<typename Config::LayoutB, cutlass::layout::ColumnMajor>;
      const int verify_group_n = kIsTensor ? N : Config::group_n;
      const int verify_group_k = kIsTensor ? K : Config::group_k;
      ok = helper.template verify_scaled<kIsTensor, kBColMajor,
                                         typename Config::ElementScale>(
          Q, ptr_A, ptr_B, ptr_D,
          host_tm.per_token_scale, host_tm.experts_scale,
          verify_group_n, verify_group_k);
    }
    std::cerr << "[MoE bench verify] "
              << (ok ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m")
              << std::endl;
  }
  #endif
  return ms;
}

// Double-buffer uniform-M run (plain BF16 and all scaled paths). if constexpr on
// scale_kind selects the verify branch at compile time.
template <class Config>
double moe_run_impl_double_buffer(const void *vendor_tm, int verify,
                                  std::string *error) {
  using ElementInput  = typename Config::Element;
  using ElementOutput = typename Config::ElementOutput;

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

  sycl::queue Q = compat::get_default_queue();

  const ElementInput *ptr_A = host_tm.scatter_tokens;
  const ElementInput *ptr_B = host_tm.experts_weight;
  ElementOutput      *ptr_D = host_tm.y;

  // One timed launch; verify (below) reads ptr_D.
  double ms;
  try {
    ms = cutlass::moe::moe_run_double_buffer<Config>(host_tm);
  } catch (std::exception const &e) {
    if (error) *error = e.what();
    return -1.0;
  }
#ifdef FULL_RUN_TIMING_AND_VERIFY
  if (verify == kVerifyHost || verify == kVerifyDevice) {
    VerificationHelper helper;
    std::vector<int> rows(num_experts, uniform_m);
    helper.parse(num_experts, rows.data(), N, K);
    bool ok = false;
    if constexpr (Config::scale_kind == ScaleKind::Plain) {
      ok = helper.verify(ptr_A, ptr_B, ptr_D);
    } else {
      constexpr bool kIsTensor  = (Config::scale_kind == ScaleKind::Tensor);
      constexpr bool kBColMajor =
          cute::is_same_v<typename Config::LayoutB, cutlass::layout::ColumnMajor>;
      const int verify_group_n = kIsTensor ? N : Config::group_n;
      const int verify_group_k = kIsTensor ? K : Config::group_k;
      // Verify against the unpadded host scale grids, not the kernel's packed
      // surface — keeps verify independent of pack_moe_scales.
      ok = helper.template verify_scaled<kIsTensor, kBColMajor,
                                         typename Config::ElementScale>(
          Q, ptr_A, ptr_B, ptr_D,
          host_tm.per_token_scale, host_tm.experts_scale,
          verify_group_n, verify_group_k);
    }
    std::cerr << "[MoE bench double buffer verify] "
              << (ok ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m")
              << std::endl;
  }
#endif

  return ms;
}

// Tile registration. Each X() entry generates a per-tile run function bound to
// its exact Config, registered with its geometry + KernelKind; launch_moe() calls
// it via chosen.run() (the Config can't be named from the dtype string alone).
#define MOE_REGISTER_TILE(DTYPE, NAME, CONFIG)                                 \
  static double moe_run_##NAME(const void *vendor_tm, int v,                   \
                               std::string *e) {                               \
    return moe_run_impl<CONFIG>(vendor_tm, v, e);                              \
  }                                                                            \
  static const bool moe_reg_##NAME = (cutlass::moe::moe_register_tile(          \
      #DTYPE,                                                                  \
      cutlass::moe::TileGeom{                                                   \
          static_cast<int>(cute::get<0>(CONFIG::TileShapeCri{})),              \
          static_cast<int>(cute::get<1>(CONFIG::TileShapeCri{})),              \
          static_cast<int>(cute::get<2>(CONFIG::TileShapeCri{})), #NAME,        \
          /*is_db=*/false},                                                     \
      cutlass::moe::KernelKind::Regular,                                        \
      &moe_run_##NAME), true);

// Double-buffer tile — geometry flagged is_db so pick_tile applies the DB policy.
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
          /*is_db=*/true},                                                      \
      cutlass::moe::KernelKind::DoubleBuffer,                                   \
      &moe_run_##NAME), true);

// Scaled double-buffer: same impl as plain double-buffer; if constexpr inside
// moe_run_impl_double_buffer selects the scaled verify path automatically.
#define MOE_REGISTER_TILE_DOUBLE_BUFFER_SCALED(DTYPE, NAME, CONFIG)            \
  MOE_REGISTER_TILE_DOUBLE_BUFFER(DTYPE, NAME, CONFIG)

// The per-binary tile list (selected by -DMOE_DTYPE_<TAG>) expands each entry
// into a registration so every compiled tile joins its dtype's candidate list.
#include "moe_grouped_gemm/runner/moe_tile_list.hpp"
#ifdef MOE_TILE_X_LIST
#define X(DTYPE, NAME, CONFIG) MOE_REGISTER_TILE(DTYPE, NAME, CONFIG)
#define X_DOUBLE_BUFFER(DTYPE, NAME, CONFIG) MOE_REGISTER_TILE_DOUBLE_BUFFER(DTYPE, NAME, CONFIG)
#define X_DOUBLE_BUFFER_SCALED(DTYPE, NAME, CONFIG) MOE_REGISTER_TILE_DOUBLE_BUFFER_SCALED(DTYPE, NAME, CONFIG)
MOE_TILE_X_LIST
#undef X
#undef X_DOUBLE_BUFFER
#undef X_DOUBLE_BUFFER_SCALED
#endif

// Single entry point (see moe_api.hpp): read the problem shape from vendor_tm,
// pick the best tile for this dtype, then run it. Returns elapsed ms (-1.0 on
// failure, *error set).
double launch_moe(const void *vendor_tm, const char *dtype, int verify,
                  std::string *error) {
  // The leading fields are at identical offsets in every instantiation.
  auto const &dims =
      *static_cast<cutlass::moe::VendorTensorMapping<char, char, char> const *>(
          vendor_tm);
  const int N = dims.N, K = dims.K, num_experts = dims.num_experts;
  std::vector<int> M_per_expert(dims.experts_token_count,
                                dims.experts_token_count + num_experts);

  // Registry lookup + pick_tile() for this dtype and problem shape.
  const cutlass::moe::TileEntry *chosen_ptr =
      cutlass::moe::select_tile(dtype, N, K, M_per_expert, num_experts, error);
  if (!chosen_ptr)
    return -1.0;
  auto const &chosen = *chosen_ptr;
  #ifdef FULL_RUN_TIMING_AND_VERIFY
  const char *kind_str =
      chosen.kind == cutlass::moe::KernelKind::DoubleBuffer ? "double-buffer"
                                                            : "regular";

  std::cerr << "[MoE tile-select] dtype='" << dtype << "' -> "
            << kind_str << " tile '" << chosen.geom.name << "' ("
            << chosen.geom.blk_m << "x" << chosen.geom.blk_n << "x"
            << chosen.geom.blk_k << ") for N=" << N
            << " K=" << K << " experts=" << num_experts << std::endl;
  #endif
  return chosen.run(vendor_tm, verify, error);
}

} // namespace moe_bench
