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
    \brief Google-Benchmark harness for the hand-written MoE grouped-GEMM kernel
           (applications/moe_grouped_gemm). Benchmarks one GEMM of the exact
           N/K/experts/M described in the config line.
*/

#pragma once

// Lean benchmark TU: does not include the cute / MoE kernel, which lives only in
// moe_api.cpp. See moe_api.hpp for the TU-split.
#include "../common.hpp"
#include <benchmark/benchmark.h>

#include "moe_grouped_gemm/runner/moe_api.hpp"
// Kernel-free declarations (Config structs, ScaleKind, fill_flat_scales,
// VendorTensorMapping) — not moe_gemm_runner.hpp, which would pull the device
// kernel into the benchmark TU.
#include "moe_grouped_gemm/runner/moe_types.hpp"
// Provides MOE_DTYPE_TAG_LIST, which drives per-dtype registration below.
#include "moe_grouped_gemm/runner/moe_tile_list.hpp"

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cutlass::benchmark {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Every input the kernel reads for one benchmarked config: device A/B/D plus the
// host scale grids. Allocated and filled once per config by build_inputs. The
// vendor_tm only holds pointers into these, so they must outlive the timed loop —
// released when the owning shared_ptr goes out of scope at the end of run().
// Scale grids are empty for the unscaled (Plain) path.
template <class Config, class ElementInput, class ElementOutput>
struct MoeDeviceBuffers {
  using ElementScaleStore = cutlass::moe::ScaleStoreFor<Config>;

  cutlass::DeviceAllocation<ElementInput>  A;   // [num_tokens, K]
  cutlass::DeviceAllocation<ElementInput>  B;   // [num_experts, N, K]
  cutlass::DeviceAllocation<ElementOutput> D;   // [num_tokens, N]
  // Per-expert token counts on the device — the variable-M kernel's M_per_group,
  // read per workgroup to find its expert and row offset.
  cutlass::DeviceAllocation<int32_t> experts_token_count; // [num_experts]

  // Host-side, quantized to the type the kernel reads (see fill_flat_scales).
  // Block  : [num_tokens, scale_k] and [num_experts, N, scale_k].
  // Tensor : one scale per token and one per expert.
  // Kept for verify, which compares against these unpadded grids.
  std::vector<ElementScaleStore> per_token_scale;
  std::vector<ElementScaleStore> experts_scale;

  // The packed, padded DEVICE scale surfaces the kernel reads, built from the
  // grids above by pack_moe_scales. Empty for the unscaled (Plain) path.
  cutlass::DeviceAllocation<ElementScaleStore> packed_scale_a;
  cutlass::DeviceAllocation<ElementScaleStore> packed_scale_b;
};
// Carries a Config type as a value so a generic lambda can recover it via
// decltype(tag)::type (templated lambdas are C++20).
template <class T> struct TypeTag { using type = T; };

// Rejects a mapping missing any device buffer this Config's kernel dereferences —
// cheaper to fail here, with a name, than to fault in device code. Runs on the
// finished vendor_tm (client-owned, since the client fills it) before launch_moe()
// crosses into the device TU.
template <class Config, class ElementA, class ElementD, class ElementScaleIn>
void moe_validate_mapping(
    const cutlass::moe::VendorTensorMapping<ElementA, ElementScaleIn, ElementD> &tm) {
  if (!tm.scatter_tokens || !tm.experts_weight || !tm.y || !tm.experts_token_count_device)
    throw std::runtime_error(
        "VendorTensorMapping::{scatter_tokens,experts_weight,y} must point at "
        "the client's device buffers, and experts_token_count_device must point at the device copy of the per-expert counts");
  if constexpr (Config::scale_kind != cutlass::moe::ScaleKind::Plain) {
    // Tensor surface geometry is 8-bit-only (see moe_scale_layout.hpp); a 4-bit
    // config would need height 4, so fail here rather than mis-stride.
    static_assert(Config::scale_kind != cutlass::moe::ScaleKind::Tensor ||
                      cute::sizeof_bits_v<typename Config::Element> == 8,
                  "ScaleKind::Tensor scale surface geometry is fp8-only.");
    // Allocated/packed/uploaded above in build_inputs. Null only on the Plain path.
    if (!tm.packed_scale_a || !tm.packed_scale_b)
      throw std::runtime_error(
          "VendorTensorMapping::{packed_scale_a,packed_scale_b} must point at "
          "the client's packed device scale surfaces for a scaled config");
  }
}

// The element-type families a .in dtype token can name; each maps to one Config
// in moe_types.hpp. The tile geometry is picked later by the tile-select table.
enum class DtypeFamily { Bf16, MxFp8E4m3, MxFp4E2m1, Fp8TensorE4m3 };

// Resolve the .in dtype token to its element-type family. Returns false for an
// unknown/uncompiled token.
inline bool resolve_dtype_family(std::string const &token, DtypeFamily &out) {
  static const std::map<std::string, DtypeFamily> registry = {
      {"bf16", DtypeFamily::Bf16},
      {"mxfp8_e4m3", DtypeFamily::MxFp8E4m3},
      {"mxfp4", DtypeFamily::MxFp4E2m1},
      {"fp8_tensor", DtypeFamily::Fp8TensorE4m3},
      // _moe-suffixed tags used by the greedy configs (.in files) map to the
      // same element-type families as the plain tags.
      {"bf16_moe", DtypeFamily::Bf16},
      {"mxfp8_e4m3_moe", DtypeFamily::MxFp8E4m3},
      {"mxfp4_moe", DtypeFamily::MxFp4E2m1},
      {"fp8_tensor_moe", DtypeFamily::Fp8TensorE4m3},
  };
  auto it = registry.find(token);
  if (it == registry.end()) return false;
  out = it->second;
  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options for one MoE grouped-GEMM benchmark line. Builds a
// per-expert M vector from one of two shape sources (highest precedence first):
//   1. --m_per_expert=<csv>   explicit per-expert M list
//   2. Default: M_i = (m * topk / ep_size) / (num_experts / ep_size)  (uniform)
struct MoEBenchmarkOptions {

  bool error;

  int n, k, num_experts;
  // MoE routing parameters (PR #687 semantics).
  int m, topk, ep_size;
  // Benchmark iteration count (per .in line: --iterations=N). Defaults to the
  // ITERATIONS macro (1 for CRI, 100 otherwise).
  int iterations;
  // Which reference to check the first iteration's output against, as a
  // moe_bench::VerifyKind. Set from --verify=none|host|device (see parse()).
  int verify_kind;
  std::string m_per_expert; // comma-separated per-expert M list
  std::string bm_name;
/*
  Reference-API (canonical) vocabulary vs. legacy aliases:
    num_tokens          <- total routed tokens (== legacy --m)
    hidden_size         <- model hidden dim
    new_hidden_size     <- intermediate / expert dim
    proj = up|down      <- selects N/K assignment:
                             up   : K=hidden_size,     N=new_hidden_size
                             down : K=new_hidden_size, N=hidden_size
    experts_token_count <- per-expert token counts (== legacy --m_per_expert)
    experts_token_offset<- per-expert prefix-sum; if given, validated to be
                           exactly the running sum of experts_token_count
    num_experts_per_rank<- experts on this rank (== num_experts/ep_size); if
                           given, overrides num_experts for the per-rank GEMM
   The legacy --m/--n/--k/--m_per_expert remain as silent aliases so committed .in
   files keep working. The canonical name wins when both are given.
*/
  int hidden_size, new_hidden_size, num_experts_per_rank;
  // When >0, pins the benched expert-group count, overriding the routing-derived
  // one (some configs carry a large model --num_experts but bench fewer groups).
  int override_number_experts;
  // When true (.in --override_use_greedy_always=1), always picks the greedy
  // kernel, ignoring the hardcoded double-buffer shapes.
  bool override_use_greedy_always;
  std::string proj; // "up" | "down" | "" (raw n/k)
  std::string experts_token_offset; // comma-separated prefix sum (optional)

  // Per-expert M list (the "groups").
  std::vector<int> rows_per_expert;

  MoEBenchmarkOptions()
      : error(false), n(2880), k(2880), num_experts(8),
        m(4096), topk(1), ep_size(1), iterations(ITERATIONS),
        verify_kind(moe_bench::kVerifyNone), m_per_expert(""),
        bm_name("MoEGEMM"),
        hidden_size(0), new_hidden_size(0), num_experts_per_rank(0),
        override_number_experts(0), override_use_greedy_always(false), proj(""),
        experts_token_offset("") {
    build_rows();
  }

  void build_rows() {
    rows_per_expert.clear();
    if (!m_per_expert.empty()) {
      std::stringstream ss(m_per_expert);
      std::string token;
      while (std::getline(ss, token, ',')) {
        if (token.empty())
          continue;
        rows_per_expert.push_back(std::stoi(token));
      }
      // num_experts follows the csv length.
      num_experts = static_cast<int>(rows_per_expert.size());
    } else {
      // Default uniform routing. ep_size shards experts across ranks, so both
      // token budget and expert count are per-rank -- the divisions cancel,
      // leaving M_per_expert independent of ep_size. parse() ensures both
      // divisors are >= 1. num_experts_per_rank is already a per-rank count (it
      // replaced num_experts in parse), so it must not be divided by ep_size again.
      int experts_per_gpu =
          num_experts_per_rank > 0 ? num_experts : num_experts / ep_size;
      const int tokens_per_gpu = (m * topk) / ep_size;
      int m_per_expert_val = tokens_per_gpu / experts_per_gpu;
      if (override_number_experts > 0)
        experts_per_gpu = override_number_experts;
      // A token budget below one row per expert still has to bench something.
      if (m_per_expert_val < 1)
        m_per_expert_val = 1;
      rows_per_expert.assign(experts_per_gpu, m_per_expert_val);
      num_experts = experts_per_gpu;
    }
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    cmd.get_cmd_line_argument("n", n, 2880);
    cmd.get_cmd_line_argument("k", k, 2880);
    cmd.get_cmd_line_argument("num_experts", num_experts, 8);
    cmd.get_cmd_line_argument("m_per_expert", m_per_expert, std::string(""));
    // Consume legacy --uniform_m silently (backward compat with old .in files).
    { int dummy = 0; cmd.get_cmd_line_argument("uniform_m", dummy, 0); }
    cmd.get_cmd_line_argument("bm_name", bm_name, std::string("MoEGEMM"));

    // PR #687 MoE routing parameters.
    cmd.get_cmd_line_argument("m", m, 4096);
    cmd.get_cmd_line_argument("topk", topk, 1);
    cmd.get_cmd_line_argument("ep_size", ep_size, 1);

    // Per-line benchmark iteration count; defaults to the ITERATIONS macro.
    cmd.get_cmd_line_argument("iterations", iterations, int(ITERATIONS));

    // Reference-API names, read after the legacy flags so a canonical name, when
    // present, overrides its alias.
    int num_tokens = 0;
    cmd.get_cmd_line_argument("num_tokens", num_tokens, 0);
    int num_of_tokens = 0;
    cmd.get_cmd_line_argument("num_of_tokens", num_of_tokens, 0);
    if (num_of_tokens > 0)
      m = num_of_tokens; // alias for num_tokens
    else if (num_tokens > 0)
      m = num_tokens; // total routed tokens == legacy --m

    cmd.get_cmd_line_argument("hidden_size", hidden_size, 0);
    cmd.get_cmd_line_argument("new_hidden_size", new_hidden_size, 0);
    cmd.get_cmd_line_argument("proj", proj, std::string(""));
    // Map (hidden_size, new_hidden_size, proj) -> raw (n, k). Up: K=hidden,
    // N=new_hidden; down swaps.
    if (hidden_size > 0 && new_hidden_size > 0) {
      if (proj == "down") {
        k = new_hidden_size;
        n = hidden_size;
      } else {
        // default + "up": K=hidden, N=new_hidden.
        if (!proj.empty() && proj != "up") {
          std::cerr << "Error: --proj must be 'up' or 'down' (got '" << proj
                    << "').\n";
          error = true;
        }
        k = hidden_size;
        n = new_hidden_size;
      }
    }

    // experts_token_count: canonical alias for --m_per_expert (wins if both given).
    std::string experts_token_count;
    cmd.get_cmd_line_argument("experts_token_count", experts_token_count,
                              std::string(""));
    if (!experts_token_count.empty())
      m_per_expert = experts_token_count;

    // num_experts_per_rank (canonical): sets num_experts directly for the
    // per-rank GEMM.
    cmd.get_cmd_line_argument("num_experts_per_rank", num_experts_per_rank, 0);
    if (num_experts_per_rank > 0)
      num_experts = num_experts_per_rank;

    // Pins the benched expert-group count, overriding build_rows()'s routing-derived one.
    cmd.get_cmd_line_argument("override_number_experts", override_number_experts,
                              0);

    // Force greedy for every shape (skip the hardcoded double-buffer set).
    { int g = 0; cmd.get_cmd_line_argument("override_use_greedy_always", g, 0);
      override_use_greedy_always = (g != 0); }

    cmd.get_cmd_line_argument("experts_token_offset", experts_token_offset,
                              std::string(""));

    // --verify=none|host|device selects the reference. The legacy boolean/numeric
    // spellings keep the reference they meant before this tri-state existed
    // (0/false = none, 1/true = host); `device` is opt-in by name, because on the
    // AubLoad simulator a naive reference GEMM is far slower than the host loop.
    // An unrecognized value warns and falls back to none rather than guessing.
    std::string verify_str;
    cmd.get_cmd_line_argument("verify", verify_str, std::string("none"));
    if (verify_str == "none" || verify_str == "false" || verify_str == "0") {
      verify_kind = moe_bench::kVerifyNone;
    } else if (verify_str == "device") {
      verify_kind = moe_bench::kVerifyDevice;
    } else if (verify_str == "host" || verify_str == "true" ||
               verify_str == "1") {
      verify_kind = moe_bench::kVerifyHost;
    } else {
      std::cerr << "Warning: unrecognized --verify=" << verify_str
                << " (expected none|host|device); verification disabled.\n";
      verify_kind = moe_bench::kVerifyNone;
    }

    // Routing counts must be >= 1 so build_rows()'s divisions are defined.
    // Checked after num_experts_per_rank may have replaced num_experts. Note
    // CommandLine cannot report a bad value: `istringstream >> int` silently
    // yields 0 ("abc") or truncates ("1.5" -> 1).
    if (num_experts < 1 || topk < 1 || ep_size < 1) {
      std::cerr << "Error: --num_experts/--topk/--ep_size must each be >= 1"
                << " (got " << num_experts << "/" << topk << "/" << ep_size
                << ").\n";
      error = true;
      return;
    }
    // A rank must own at least one expert, else num_experts / ep_size floors to
    // zero and build_rows() divides by it. Skipped when num_experts_per_rank
    // gave a per-rank count (that path does not divide by ep_size).
    if (num_experts_per_rank <= 0 && ep_size > num_experts) {
      std::cerr << "Error: --ep_size=" << ep_size
                << " exceeds --num_experts=" << num_experts << ".\n";
      error = true;
      return;
    }

    build_rows();

    // If experts_token_offset was supplied, validate it is exactly the running
    // prefix-sum of the per-expert counts. We derive the same offset internally,
    // so this only guards a mismatched hand-written .in line.
    if (!experts_token_offset.empty()) {
      std::vector<int> off;
      std::stringstream ss(experts_token_offset);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        if (tok.empty())
          continue;
        off.push_back(std::stoi(tok));
      }
      if (off.size() != rows_per_expert.size()) {
        std::cerr << "Error: experts_token_offset has " << off.size()
                  << " entries but experts_token_count has "
                  << rows_per_expert.size() << ".\n";
        error = true;
      } else {
        int running = 0;
        for (size_t i = 0; i < off.size(); ++i) {
          if (off[i] != running) {
            std::cerr << "Error: experts_token_offset[" << i << "]=" << off[i]
                      << " != prefix-sum " << running
                      << " of experts_token_count.\n";
            error = true;
            break;
          }
          running += rows_per_expert[i];
        }
      }
    }
  }

  int total_m() const {
    int t = 0;
    for (int m : rows_per_expert)
      t += m;
    return t;
  }

  /// Compute performance in TFLOP/s over all per-expert problems.
  double tflops(double runtime_s) const {
    uint64_t fmas = 0;
    for (int m : rows_per_expert) {
      fmas += static_cast<uint64_t>(m) * static_cast<uint64_t>(n) *
              static_cast<uint64_t>(k);
    }
    uint64_t flop = static_cast<uint64_t>(2) * fmas;
    double tflop = double(flop) / double(1.0e12);
    return tflop / runtime_s;
  }

  std::string benchmark_name() const {
    std::stringstream full_name;
    int tpg = total_m();
    // Reflect the actual M distribution: "uniform" iff every expert's M is equal,
    // else "dynamic_m" (M varies across experts).
    bool dynamic_m = false;
    for (int mm : rows_per_expert)
      if (!rows_per_expert.empty() && mm != rows_per_expert[0]) dynamic_m = true;
    full_name << bm_name << "/t" << m << "_tpg" << tpg
              << "_e" << num_experts << "x" << n << "x" << k
              << (dynamic_m ? "_dynamic_m" : "_uniform");
    return full_name.str();
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// All device work lives in moe_api.cpp, reached via launch_moe(). This runner
// only drives the Google-Benchmark loop + counters; it instantiates no cute /
// MoE / SYCL device code.
struct MoEBenchmarkRunner {

  void run(::benchmark::State &state, MoEBenchmarkOptions const &options,
           cutlass::KernelHardwareInfo const & /*hw_info*/,
           const char *dtype_tag) {
    const int num_experts = options.num_experts;
    const int N = options.n;
    const int K = options.k;

    std::vector<int> M_per_expert = options.rows_per_expert;
    if (static_cast<int>(M_per_expert.size()) != num_experts) {
      state.SkipWithError("num_experts does not match per-expert M list size.");
      return;
    }

    // Dynamic (per-expert varying) M is supported: such shapes route to the
    // dynamic-M greedy kernel (MoEGEMMGreedy<true>), so no uniform-M check here.
    // The double-buffer path requires uniform M but is only chosen for the
    // hardcoded (all-uniform) DB shapes and re-validates uniformity itself
    // (moe_run_impl_double_buffer), so an errant dynamic-M->DB pick fails loudly
    // at launch rather than silently here.

    int num_tokens = options.total_m();

    // Resolve the .in dtype tag to its element-type family so the data dtype
    // follows the .in line.
    DtypeFamily family;
    if (!resolve_dtype_family(dtype_tag, family)) {
      state.SkipWithError(
          (std::string("unknown/uncompiled dtype token '") + dtype_tag + "'")
              .c_str());
      return;
    }

    // The MoeDeviceBuffers for this config, type-erased because the struct is
    // Config-templated while this scope is dtype-agnostic; the shared_ptr deleter
    // frees the right type (and releases the device memory) on return.
    std::shared_ptr<void> inputs;

    // Global-memory traffic for this grouped GEMM, in MB. Filled by build_inputs
    // (where the Config's element sizes are known) and consumed by
    // finalize_counters to report GB/s (MB/ms == GB/s), mirroring 00_gemm's model.
    double mega_bytes_transferred = 0.0;

    // Allocates + fills every input the kernel reads and returns a vendor_tm
    // wired to it. Config passed as a type tag to stay valid C++17.
    auto build_inputs = [&](auto config_tag) {
      using Config = typename decltype(config_tag)::type;
      using ElementInput  = typename Config::Element;
      using ElementOutput = typename Config::ElementOutput;
      using Buffers = MoeDeviceBuffers<Config, ElementInput, ElementOutput>;
      using ElementScaleStore = typename Buffers::ElementScaleStore;
      constexpr cutlass::moe::ScaleKind kScaleKind = Config::scale_kind;
      const uint64_t seed = 2023;

      // Kept alive for the whole benchmark loop — vendor_tm only holds pointers into it.
      auto buf = std::make_shared<Buffers>();
      inputs = buf;

      cutlass::moe::VendorTensorMapping<ElementInput, ElementScaleStore, ElementOutput> vendor_tm;
      vendor_tm.experts_token_count = M_per_expert.data();
      vendor_tm.num_experts         = num_experts;
      vendor_tm.N                   = N;
      vendor_tm.K                   = K;

      if constexpr (kScaleKind != cutlass::moe::ScaleKind::Plain) {
        constexpr bool kIsTensor = (kScaleKind == cutlass::moe::ScaleKind::Tensor);
        // Tensor scale is fp8-only: the fixed device surface geometry
        // (moe_scale_layout.hpp) derives from an 8-bit MMA_K. Caught here, at the
        // dtype that selects the config, and also in the kernels and packer.
        static_assert(!kIsTensor || cute::sizeof_bits_v<ElementInput> == 8,
                      "ScaleKind::Tensor scale surface geometry is fp8-only.");
        // Logical (unpadded) host grids: one scale per token / per expert.
        const int scale_k =
            kIsTensor ? 1 : (K + Config::group_k - 1) / Config::group_k;
        buf->per_token_scale.resize(std::size_t(int64_t(num_tokens) * scale_k));
        buf->experts_scale.resize(
            kIsTensor ? std::size_t(num_experts)
                      : std::size_t(int64_t(num_experts) * int64_t(N) * scale_k));
        cutlass::moe::fill_flat_scales(buf->per_token_scale, seed + 2020, false);
        cutlass::moe::fill_flat_scales(buf->experts_scale, seed + 2019, false);
        vendor_tm.per_token_scale = buf->per_token_scale.data();
        vendor_tm.experts_scale   = buf->experts_scale.data();

        // Pack the grids into the padded device surface and upload, here where
        // the Config (ScaleKind, group sizes) is known. The kernel reads this
        // surface verbatim — nothing is re-derived downstream.
        const auto geom = cutlass::moe::scale_surface_geom<Config>(N, K);
        cutlass::moe::pack_moe_scales<kIsTensor, ElementScaleStore>(
            buf->per_token_scale.data(), buf->experts_scale.data(),
            M_per_expert.data(), num_experts, N, K,
            kIsTensor ? K : Config::group_k, kIsTensor ? N : Config::group_n,
            geom.scale_k_store, geom.padded_scale_n, buf->packed_scale_a,
            buf->packed_scale_b);
        vendor_tm.packed_scale_a = buf->packed_scale_a.get();
        vendor_tm.packed_scale_b = buf->packed_scale_b.get();
      }

      // Fill device A/B/D with initialize_block, which owns the per-element value
      // range for the compile-time type (and the sub-byte pack for e2m1), so no
      // host staging buffer is needed. D is fully overwritten by the kernel
      // (beta=0) but still seeded, so a partial store shows up in verify.
      buf->A.reset(std::size_t(int64_t(num_tokens) * K));
      buf->B.reset(std::size_t(int64_t(num_experts) * N * K));
      buf->D.reset(std::size_t(int64_t(num_tokens) * N));
      cutlass::initialize_block(buf->A, seed + 2023);
      cutlass::initialize_block(buf->B, seed + 2022);
      cutlass::initialize_block(buf->D, seed + 2021);

      // Per-expert counts -> device int32 (the kernel's M_per_group). Also kept on
      // the host in the mapping, for verify and the uniform-M check.
      std::vector<int32_t> counts32(M_per_expert.begin(), M_per_expert.end());
      buf->experts_token_count.reset(num_experts);
      buf->experts_token_count.copy_from_host(counts32.data());

      vendor_tm.scatter_tokens = buf->A.get();
      vendor_tm.experts_weight = buf->B.get();
      vendor_tm.y              = buf->D.get();
      vendor_tm.experts_token_count_device = buf->experts_token_count.get();

      // Global-memory traffic for this grouped GEMM, mirroring 00_gemm's model:
      // read A once over all routed tokens, read every expert's weights B, and
      // write D once. Scaled configs
      // add the logical per-token and per-expert scale grids. Sub-byte types
      // (e.g. e2m1) are handled by sizeof_bits_v / bits_per_byte.
      {
        constexpr double bits_per_byte = static_cast<double>(cute::sizeof_bits_v<char>);
        constexpr double sizeof_inputs = cute::sizeof_bits_v<ElementInput>  / bits_per_byte;
        constexpr double sizeof_output = cute::sizeof_bits_v<ElementOutput> / bits_per_byte;
        double scale_bytes = 0.0;
        if constexpr (kScaleKind != cutlass::moe::ScaleKind::Plain) {
          constexpr double sizeof_scale =
              cute::sizeof_bits_v<ElementScaleStore> / bits_per_byte;
          scale_bytes = static_cast<double>(buf->per_token_scale.size() +
                                            buf->experts_scale.size()) *
                        sizeof_scale;
        }
        mega_bytes_transferred =
            (static_cast<double>(int64_t(num_tokens) * K) * sizeof_inputs +
             static_cast<double>(int64_t(num_experts) * N * K) * sizeof_inputs +
             static_cast<double>(int64_t(num_tokens) * N) * sizeof_output) *
                1e-6 +
            scale_bytes * 1e-6;
      }

      // Check the completed mapping here, on the side that filled it. Throws;
      // run() reports it as a skip.
      moe_validate_mapping<Config>(vendor_tm);
      return vendor_tm;
    };

    // FLOP count over all per-expert problems.
    uint64_t fmas = 0;
    for (int m : M_per_expert) {
      fmas += static_cast<uint64_t>(m) * static_cast<uint64_t>(N) *
              static_cast<uint64_t>(K);
    }
    double gflop = double(static_cast<uint64_t>(2) * fmas) / double(1.0e9);

    state.counters["m"] = options.m;
    state.counters["n"] = N;
    state.counters["k"] = K;
    state.counters["num_experts"] = num_experts;
    state.counters["groups"] = num_experts;
    state.counters["topk"] = options.topk;
    state.counters["ep_size"] = options.ep_size;
    state.counters["tokens_per_gpu"] = num_tokens;
    state.counters["total_m"] = num_tokens;
    for (int i = 0; i < num_experts && i < 32; ++i) {
      state.counters["M_" + std::to_string(i)] = M_per_expert[i];
    }

    // Build the inputs once (outside the timed loop). Store the vendor_tm as raw
    // bytes so the loop body stays dtype-agnostic.
    std::vector<char> vtm_storage;
    auto store_vtm = [&](auto config_tag) {
      auto vtm = build_inputs(config_tag);
      vtm_storage.resize(sizeof(vtm));
      std::memcpy(vtm_storage.data(), &vtm, sizeof(vtm));
    };
    try {
      switch (family) {
      case DtypeFamily::Bf16:          store_vtm(TypeTag<cutlass::moe::Bf16Config>{});          break;
      case DtypeFamily::MxFp8E4m3:     store_vtm(TypeTag<cutlass::moe::MxFp8E4m3Config>{});     break;
      case DtypeFamily::MxFp4E2m1:     store_vtm(TypeTag<cutlass::moe::MxFp4E2m1Config>{});     break;
      case DtypeFamily::Fp8TensorE4m3: store_vtm(TypeTag<cutlass::moe::Fp8TensorE4m3Config>{}); break;
      }
    } catch (std::exception const &e) {
      // Device allocation / fill failure (usually OOM on a large shape) — report
      // as a skip instead of terminating the whole suite.
      state.SkipWithError(e.what());
      return;
    }
    const void *vtm_ptr = vtm_storage.data();

    // verify only on the first iteration; all subsequent runs are timing-only,
    // so the reference never enters the reported timings.
    int current_verify = options.verify_kind;

    initialize_counters(state);
    for (auto _ : state) {
      std::string error;
      double ms_elapsed = moe_bench::launch_moe(vtm_ptr, dtype_tag,
                                                current_verify, &error,
                                                options.override_use_greedy_always);
      if (ms_elapsed < 0.0) {
        state.SkipWithError(error.empty() ? "launch_moe failed" : error.c_str());
        return;
      }
      current_verify = moe_bench::kVerifyNone; // verify once, then timing-only
      update_counters(state, ms_elapsed);
      state.SetIterationTime(ms_elapsed / 1000);
    }
    finalize_counters(state, gflop, mega_bytes_transferred);
  }

private:

  static void initialize_counters(::benchmark::State &state) {
    state.counters["avg_runtime_ms"] = 0;
    state.counters["best_runtime_ms"] = std::numeric_limits<double>::max();
    state.counters["worst_runtime_ms"] = std::numeric_limits<double>::lowest();
    state.counters["total_runtime_ms"] = 0;
  }

  static void update_counters(::benchmark::State &state, double ms_elapsed) {
    state.PauseTiming();
    state.counters["total_runtime_ms"] += ms_elapsed;
    state.counters["best_runtime_ms"] =
        std::min<double>(state.counters["best_runtime_ms"], ms_elapsed);
    state.counters["worst_runtime_ms"] =
        std::max<double>(state.counters["worst_runtime_ms"], ms_elapsed);
    state.ResumeTiming();
  }

  static void finalize_counters(::benchmark::State &state, double gflop,
                                double mega_bytes_transferred) {
    auto safe_div = [](double num, double den) {
      return den > 0.0 ? num / den : 0.0;
    };

    auto iters = static_cast<double>(state.iterations());
    if (iters > 2) {
      state.counters["avg_runtime_ms"] =
          safe_div(state.counters["total_runtime_ms"] -
                       state.counters["best_runtime_ms"] -
                       state.counters["worst_runtime_ms"],
                   iters - 2);
    } else {
      state.counters["avg_runtime_ms"] =
          safe_div(state.counters["total_runtime_ms"], iters);
    }
    state.counters["avg_tflops"] =
        safe_div(gflop, state.counters["avg_runtime_ms"]);
    state.counters["best_tflop"] =
        safe_div(gflop, state.counters["best_runtime_ms"]);
    // MB / ms == GB/s. Drives the MBU (memory-bandwidth-utilization) calc.
    state.counters["avg_bandwidth_gbs"] =
        safe_div(mega_bytes_transferred, state.counters["avg_runtime_ms"]);
    state.counters["best_bandwidth_gbs"] =
        safe_div(mega_bytes_transferred, state.counters["best_runtime_ms"]);
  }
};

} // namespace cutlass::benchmark

///////////////////////////////////////////////////////////////////////////////////////////////////

// Two-level indirection so a macro argument is expanded BEFORE # / ## act on it.
// Applying # / ## to the bare parameter would stringize/paste the literal token
// instead of its value, silently registering the benchmark under the wrong name
// ("Benchmark not found" at run time).
#define CUTLASS_MOE_STR_IMPL(s) #s
#define CUTLASS_MOE_STR(s) CUTLASS_MOE_STR_IMPL(s)
#define CUTLASS_MOE_CAT_IMPL(a, b) a##b
#define CUTLASS_MOE_CAT(a, b) CUTLASS_MOE_CAT_IMPL(a, b)

#define CUTLASS_GROUPED_GEMM_BENCHMARK(Name)                                   \
  cutlass::benchmark::BenchmarkRegistry<                                       \
      cutlass::benchmark::MoEBenchmarkOptions>::Register(                      \
      CUTLASS_MOE_STR(Name), &CUTLASS_MOE_CAT(Name, _func))

// The runner is not templated on Config, so this macro just builds the
// registration thunk under the requested name.
#define CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(Name)                            \
  static void CUTLASS_MOE_CAT(Name, _func)(                                    \
      ::benchmark::State &state,                                               \
      cutlass::benchmark::MoEBenchmarkOptions const &options,                  \
      cutlass::KernelHardwareInfo const &hw_info) {                            \
    auto bench = cutlass::benchmark::MoEBenchmarkRunner();                     \
    bench.run(state, options, hw_info, CUTLASS_MOE_STR(Name));                 \
  }

///////////////////////////////////////////////////////////////////////////////////////////////////

// MOE_DTYPE_TAG_LIST expands to one F(DTYPE) per active dtype; the tile-select
// table chooses the tile geometry from the problem size at runtime.
#include "moe_grouped_gemm/runner/moe_tile_list.hpp"

// Regular families (auto-selected tile).
#ifdef MOE_DTYPE_TAG_LIST
#define F(DTYPE) CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(DTYPE)
MOE_DTYPE_TAG_LIST
#undef F
#endif

static void register_grouped_gemm_benchmarks() {
#ifdef MOE_DTYPE_TAG_LIST
#define F(DTYPE) CUTLASS_GROUPED_GEMM_BENCHMARK(DTYPE);
  MOE_DTYPE_TAG_LIST
#undef F
#endif
}
