/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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

/*!
  \file gdn_runner.hpp
  \brief Single shared host runner for the chunkwise Gated DeltaNet (GDN)
         attention kernel: shape derivation, device allocation, input
         initialization (including sigmoid(b)), GDNArguments construction,
         launch, and the recurrent + chunkwise host oracles. Also hosts the
         shared CLI shape helpers (parse_gdn_shape / validate_gdn_shape) and the
         ExampleOptions / BenchmarkOptions structs.

  Consumed by all three usages so they never re-implement setup:
    - examples/14_xe35_gdn_attention  : adds perf timing on top
    - benchmarks/gdn                  : adds FLOP/byte model + Google Benchmark
    - test/unit/gdn_attention         : adds the pass gate

  Lives next to the header-only launcher under applications/gdn_attention/,
  already on every consumer's include path, so all three include it as
  "gdn_attention/gdn_runner.hpp" with no extra include dir.
*/

#pragma once

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "cutlass/cutlass.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/initialize_block.hpp"

#include "gdn_attention/xe35_chunk_gated_delta_rule_launch.hpp"
// apply_sigmoid_b + chunkwise (5-stage) host oracle.
#include "cutlass/util/reference/host/xe35_gdn_attention_stage_references.hpp"
// recurrent (token-by-token fp32) host oracle + kTolE2E.
#include "cutlass/util/reference/host/xe35_gdn_attention_recurrent_reference.hpp"
// compare_with_stats / print_compare_stats (cutlass::gdn::perf).
#include "cutlass/util/reference/host/xe35_gdn_attention_compare.hpp"

/* Default head dims for the example Options below. The example build sets
 * these via -D in its CMakeLists; the benchmark and unit-test builds include
 * this header without defining them, so fall back to the validated 128/128. */
#ifndef HEAD_K_DIM
#define HEAD_K_DIM 128
#endif
#ifndef HEAD_V_DIM
#define HEAD_V_DIM 128
#endif

namespace cutlass::gdn {

/* Forward declaration: defined further below alongside the other shared CLI
 * shape helpers, but GdnRunner::initialize() (just below) calls it to guard
 * against invalid shape fields set directly (not via an Options::parse()). */
inline bool validate_gdn_shape(int batch, int num_v_heads, int num_k_heads,
                               int head_k_dim, int head_v_dim, int seq_len,
                               char const* prefix = "");

/* ---------------------------------------------------------------------------
 * GdnRunner<T, StateT>
 *
 *   T      - activation dtype (Q/K/V/O, dt_bias); bfloat16_t in current builds
 *   StateT - SSM recurrent-state dtype; always float to avoid chunk-to-chunk
 *            accumulation drift
 *
 * Usage:
 *   GdnRunner<bfloat16_t, float> r;
 *   r.num_v_heads = 16; r.seq_len = 256;   // set shape fields
 *   r.initialize();                        // allocate + fill device + snapshot
 *   if (r.launch() != Status::kSuccess) ...
 *   r.queue.wait_and_throw();
 *   bool ok = r.verify_recurrent();        // and/or verify_chunkwise()
 * --------------------------------------------------------------------------- */

template <typename T, typename StateT>
struct GdnRunner {

  // ---- shape (set before initialize(); defaults match the unit testbed) ----
  int batch       = 1;
  int num_v_heads  = 64;
  int num_k_heads  = 16;   // GQA: num_v_heads must be a multiple of num_k_heads
  int head_k_dim   = 128;  // must be a positive multiple of kChunkSize (=64)
  int head_v_dim   = 128;  // must be a positive multiple of kChunkSize (=64)
  int seq_len      = 64;   // kernel pads per-sequence to kChunkSize internally
  unsigned seed    = 0xC0FFEEu;

  // ---- derived shape (filled by initialize()) ----
  int total_seqlen         = 0;
  int total_virtual_seqlen = 0;  // per sequence padded up to kChunkSize multiples

  // ---- in-order queue (required by the GDN pipeline) ----
  /* The launcher submits prepare -> compute_A -> inverse -> compute_wu ->
   * fwd_o back-to-back with no explicit waits; an out-of-order queue lets them
   * race on the shared A/w/u workspaces. The profiling property is added only
   * in the profiling build (GPU_Clock/SYCLTimer reads device-side stage
   * timestamps via get_profiling_info(), which requires enable_profiling());
   * it is harmless when present and unread, so all three usages share this one
   * construction. */
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  sycl::queue queue{sycl::property_list{
      sycl::property::queue::in_order(),
      sycl::property::queue::enable_profiling()}};
#else
  sycl::queue queue{sycl::property::queue::in_order()};
#endif

  // ---- device allocations ----
  cutlass::DeviceAllocation<T>       d_q, d_k, d_v;
  cutlass::DeviceAllocation<T>       d_dt_bias;
  cutlass::DeviceAllocation<float>   d_b, d_a, d_A_log;
  cutlass::DeviceAllocation<int>     d_query_start_loc, d_cache_indices;
  /* bool stored as uint8_t: DeviceAllocation<bool> would request floor(N/8)
   * bytes (= 0 for small N) and trip the alloc-failed path. reinterpret to
   * bool* at the kernel boundary. */
  cutlass::DeviceAllocation<uint8_t> d_has_initial_state;
  cutlass::DeviceAllocation<T>       d_core_attn_out;
  cutlass::DeviceAllocation<StateT>  d_ssm_state;
  cutlass::DeviceAllocation<T>       d_A_ws, d_o2_ws, d_w_ws, d_u_ws;

  /* ---- pre-kernel host snapshots (the kernel mutates q,k,a,ssm_state in
   *      place; the oracles need the raw pre-launch values) ---- */
  std::vector<float>  h_b_raw;            // pre-sigmoid b
  std::vector<T>      h_q_raw, h_k_raw, h_v_raw;
  std::vector<float>  h_a_raw;
  std::vector<StateT> h_ssm_initial_raw;
  // read-only inputs, kept host-side for the oracles.
  std::vector<float>   h_A_log;
  std::vector<T>       h_dt_bias;
  std::vector<int>     h_qsl, h_cache;
  std::vector<uint8_t> h_has_init;

  // ---- setup ----

  void initialize() {
    /* Guard against garbage shape fields (e.g. a unit test or future caller
     * that sets a field to 0/negative without going through Options::parse,
     * which already runs this check): the size_t casts below would otherwise
     * turn a negative dimension into a huge allocation size, and a zero
     * num_k_heads would divide-by-zero in the GQA modulo just below. */
    if (!validate_gdn_shape(batch, num_v_heads, num_k_heads,
                            head_k_dim, head_v_dim, seq_len, "[GDN runner] ")) {
      throw std::invalid_argument("GdnRunner::initialize: invalid shape");
    }

    constexpr int C = cutlass::gdn::kChunkSize;
    total_seqlen         = batch * seq_len;
    const int per_seq_padded = ((seq_len + C - 1) / C) * C;
    total_virtual_seqlen = batch * per_seq_padded;

    auto alloc = [](auto& da, size_t n, char const* tag) {
      try {
        da.reset(n);
      } catch (std::exception const& e) {
        std::cerr << "[GDN] alloc FAILED for " << tag
                  << " (n=" << n << "): " << e.what() << "\n";
        throw;
      }
    };

    // TO DO: Measure time impact of multi. allocation vs. a single allocation of a contiguous buffer 
    alloc(d_q, size_t(total_virtual_seqlen) * num_k_heads * head_k_dim, "d_q");
    alloc(d_k, size_t(total_virtual_seqlen) * num_k_heads * head_k_dim, "d_k");
    alloc(d_v, size_t(total_virtual_seqlen) * num_v_heads * head_v_dim, "d_v");
    alloc(d_b, size_t(num_v_heads) * total_virtual_seqlen, "d_b");
    alloc(d_a, size_t(num_v_heads) * total_virtual_seqlen, "d_a");
    alloc(d_A_log,   num_v_heads, "d_A_log");
    alloc(d_dt_bias, num_v_heads, "d_dt_bias");
    alloc(d_query_start_loc, batch + 1, "d_query_start_loc");
    alloc(d_cache_indices,   batch,     "d_cache_indices");
    alloc(d_has_initial_state, batch,   "d_has_initial_state");
    alloc(d_core_attn_out, size_t(total_seqlen) * num_v_heads * head_v_dim, "d_core_attn_out");
    alloc(d_ssm_state, size_t(batch) * num_v_heads * head_v_dim * head_k_dim, "d_ssm_state");

    auto ws = cutlass::gdn::get_workspace_sizes(make_arguments_shape_only());
    alloc(d_A_ws, ws.A_elems, "d_A_ws");
    alloc(d_o2_ws, ws.o2_elems, "d_o2_ws");
    alloc(d_w_ws, ws.w_elems, "d_w_ws");
    alloc(d_u_ws, ws.u_elems, "d_u_ws");

    /* Zero-init workspaces (kernel reads tiles after partial writes across
     * stages, so leftover USM contents could leak into the math). */
    std::vector<T> zeros_T(
        std::max({ws.A_elems, ws.o2_elems, ws.w_elems, ws.u_elems}), T{0});
    d_A_ws.copy_from_host(zeros_T.data(), ws.A_elems);
    d_o2_ws.copy_from_host(zeros_T.data(), ws.o2_elems);
    d_w_ws.copy_from_host(zeros_T.data(), ws.w_elems);
    d_u_ws.copy_from_host(zeros_T.data(), ws.u_elems);

    /* Random fill in production-like ranges. The chunk_prepare stage computes
     *   g = softplus(a + dt_bias) * (-exp(A_log))
     * then cumsums g over a chunk. initialize_block's default float range is
     * [-64, 64]; A_log near +64 makes exp(A_log) ~ 6e27, overflowing compute_A
     * to +-inf and propagating NaN. Use ranges matching GDN model statistics:
     *   A_log in [-4, 0] (= log(A), A in (0,1]); a, dt_bias in [-2, 2];
     *   b in [-2, 2] then sigmoided to (0,1) below; q/k/v use the bf16 default. */
    cutlass::initialize_block(d_q,       static_cast<uint64_t>(seed) + 1);
    cutlass::initialize_block(d_k,       static_cast<uint64_t>(seed) + 2);
    cutlass::initialize_block(d_v,       static_cast<uint64_t>(seed) + 3);
    cutlass::initialize_block(d_b,       static_cast<uint64_t>(seed) + 4, -2.0f, 2.0f);
    cutlass::initialize_block(d_a,       static_cast<uint64_t>(seed) + 5, -2.0f, 2.0f);
    cutlass::initialize_block(d_A_log,   static_cast<uint64_t>(seed) + 6, -4.0f, 0.0f);
    cutlass::initialize_block(d_dt_bias, static_cast<uint64_t>(seed) + 7, T(-2.0f), T(2.0f));

    /* Sigmoid b in place: the kernel expects b in (0,1), normally produced by a
     * conv1d front-end that this runner bypasses. Snapshot raw b first; the
     * oracles take an already-sigmoided b, which verify_*() re-derives from the
     * raw snapshot. */
    h_b_raw.assign(d_b.size(), 0.0f);
    d_b.copy_to_host(h_b_raw.data(), h_b_raw.size());
    {
      std::vector<float> h_b_sigmoid =
          cutlass::gdn::reference::stages::apply_sigmoid_b(h_b_raw);
      d_b.copy_from_host(h_b_sigmoid.data(), h_b_sigmoid.size());
    }

    /* ssm_state starts at zero; has_initial_state is all-false, so the kernel
     * must ignore the buffer's prior contents. Seeding zero keeps device and
     * oracle inputs identical. */
    {
      std::vector<StateT> zeros_state(d_ssm_state.size(), StateT{0});
      d_ssm_state.copy_from_host(zeros_state.data(), zeros_state.size());
    }

    // query_start_loc / cache_indices: sequential layout; no initial state.
    h_qsl.assign(batch + 1, 0);
    h_cache.assign(batch, 0);
    h_has_init.assign(batch, 0);
    for (int i = 0; i < batch; ++i) {
      h_qsl[i + 1] = h_qsl[i] + seq_len;
      h_cache[i]   = i;
    }
    d_query_start_loc.copy_from_host(h_qsl.data(),  h_qsl.size());
    d_cache_indices.copy_from_host(h_cache.data(),  h_cache.size());
    d_has_initial_state.copy_from_host(h_has_init.data(), h_has_init.size());

    // Snapshot raw inputs for the oracles (kernel mutates q,k,a,ssm in place).
    h_q_raw.assign(d_q.size(), T{0});
    h_k_raw.assign(d_k.size(), T{0});
    h_v_raw.assign(d_v.size(), T{0});
    h_a_raw.assign(d_a.size(), 0.0f);
    h_ssm_initial_raw.assign(d_ssm_state.size(), StateT{0});
    d_q.copy_to_host(h_q_raw.data(), h_q_raw.size());
    d_k.copy_to_host(h_k_raw.data(), h_k_raw.size());
    d_v.copy_to_host(h_v_raw.data(), h_v_raw.size());
    d_a.copy_to_host(h_a_raw.data(), h_a_raw.size());
    d_ssm_state.copy_to_host(h_ssm_initial_raw.data(), h_ssm_initial_raw.size());

    // Read-only inputs needed by the oracles.
    h_A_log.assign(num_v_heads, 0.0f);
    h_dt_bias.assign(num_v_heads, T{0});
    d_A_log.copy_to_host(h_A_log.data(), h_A_log.size());
    d_dt_bias.copy_to_host(h_dt_bias.data(), h_dt_bias.size());
  }

  cutlass::gdn::GDNArguments make_arguments_shape_only() const {
    cutlass::gdn::GDNArguments a{};
    a.batch_size           = batch;
    a.total_seqlen         = total_seqlen;
    a.total_virtual_seqlen = total_virtual_seqlen;
    a.num_k_heads          = num_k_heads;
    a.num_v_heads          = num_v_heads;
    a.head_k_dim           = head_k_dim;
    a.head_v_dim           = head_v_dim;
    a.ssm_state_stride_0   = num_v_heads * head_v_dim * head_k_dim;
    return a;
  }

  cutlass::gdn::GDNArguments make_arguments() const {
    auto a = make_arguments_shape_only();
    a.q                 = d_q.get();
    a.k                 = d_k.get();
    a.v                 = d_v.get();
    a.b                 = d_b.get();
    a.a                 = d_a.get();
    a.A_log             = d_A_log.get();
    a.dt_bias           = d_dt_bias.get();
    a.query_start_loc   = d_query_start_loc.get();
    a.cache_indices     = d_cache_indices.get();
    a.has_initial_state = reinterpret_cast<bool const*>(d_has_initial_state.get());
    a.core_attn_out     = d_core_attn_out.get();
    a.ssm_state         = d_ssm_state.get();
    a.A_workspace       = d_A_ws.get();
    a.o2_workspace      = d_o2_ws.get();
    a.w_workspace       = d_w_ws.get();
    a.u_workspace       = d_u_ws.get();
    return a;
  }

  // Launch the 5-stage chunkwise pipeline on `queue`. Caller waits.
  cutlass::Status launch() {
    return cutlass::gdn::chunk_gated_delta_rule_launch<T, StateT>(queue, make_arguments());
  }

  // ---- verification (call after launch() + queue.wait_and_throw()) ----

  // Compare device core_attn_out + ssm_state against the recurrent fp32 oracle.
  bool verify_recurrent(float atol = cutlass::gdn::reference::recurrent::kTolE2E,
                        float rtol = cutlass::gdn::reference::recurrent::kTolE2E,
                        bool print = false) {
    std::vector<T>      dev_out(d_core_attn_out.size());
    std::vector<StateT> dev_ssm(d_ssm_state.size());
    d_core_attn_out.copy_to_host(dev_out.data(), dev_out.size());
    d_ssm_state.copy_to_host(dev_ssm.data(), dev_ssm.size());

    std::vector<float> h_b_sigmoid =
        cutlass::gdn::reference::stages::apply_sigmoid_b(h_b_raw);
    // The oracle normalizes q,k in place; work on copies of the snapshots.
    std::vector<T>      h_q = h_q_raw, h_k = h_k_raw;
    std::vector<float>  h_a = h_a_raw;
    std::vector<T>      ref_out(dev_out.size(), T{0});
    std::vector<StateT> ref_ssm = h_ssm_initial_raw;

    cutlass::gdn::reference::recurrent::run_recurrent_gdn_attn_reference<T, StateT>(
        ref_out, ref_ssm, h_q, h_k, h_v_raw, h_b_sigmoid, h_a,
        h_A_log, h_dt_bias, h_qsl, h_cache, h_has_init,
        batch, num_k_heads, num_v_heads, head_k_dim, head_v_dim, total_virtual_seqlen);

    return compare_pair("[recurrent] ", ref_out, dev_out, ref_ssm, dev_ssm, atol, rtol, print);
  }

  /* Compare device output against the 5-stage chunkwise oracle (mirrors the
   * kernel's own decomposition; a tighter check than the recurrent oracle). */
  bool verify_chunkwise(float atol = cutlass::gdn::reference::recurrent::kTolE2E,
                        float rtol = cutlass::gdn::reference::recurrent::kTolE2E,
                        bool print = false) {
    std::vector<T>      dev_out(d_core_attn_out.size());
    std::vector<StateT> dev_ssm(d_ssm_state.size());
    d_core_attn_out.copy_to_host(dev_out.data(), dev_out.size());
    d_ssm_state.copy_to_host(dev_ssm.data(), dev_ssm.size());

    std::vector<float> h_b_sigmoid =
        cutlass::gdn::reference::stages::apply_sigmoid_b(h_b_raw);
    // The chunkwise oracle mutates q,k,a in place; work on copies.
    std::vector<T>      h_q = h_q_raw, h_k = h_k_raw;
    std::vector<float>  h_a = h_a_raw;
    std::vector<T>      ref_out(dev_out.size(), T{0});
    std::vector<StateT> ref_ssm = h_ssm_initial_raw;

    cutlass::gdn::reference::stages::run_chunkwise_gdn_attn_reference<T, StateT>(
        ref_out, ref_ssm, h_q, h_k, h_v_raw, h_b_sigmoid, h_a,
        h_A_log, h_dt_bias, h_qsl, h_cache, h_has_init,
        batch, num_k_heads, num_v_heads, head_k_dim, head_v_dim, total_virtual_seqlen,
        make_arguments_shape_only().ssm_state_stride_0);

    return compare_pair("[chunkwise] ", ref_out, dev_out, ref_ssm, dev_ssm, atol, rtol, print);
  }

 private:
  template <typename Out, typename Ssm>
  static bool compare_pair(char const* tag,
                           Out const& ref_out, Out const& dev_out,
                           Ssm const& ref_ssm, Ssm const& dev_ssm,
                           float atol, float rtol, bool print) {
    auto so = cutlass::gdn::perf::compare_with_stats(ref_out, dev_out, atol, rtol);
    auto ss = cutlass::gdn::perf::compare_with_stats(ref_ssm, dev_ssm, atol, rtol);
    if (print) {
      cutlass::gdn::perf::print_compare_stats((std::string(tag) + "core_attn_out").c_str(), so);
      cutlass::gdn::perf::print_compare_stats((std::string(tag) + "ssm_state    ").c_str(), ss);
    }
    return so.passed() && ss.passed();
  }
};

/* ---------------------------------------------------------------------------
 * Shared CLI shape helpers (free functions -- NOT members of GdnRunner).
 *
 * The example driver and the benchmark driver both parse the same six shape
 * flags from a command line / config line and run the same three validity
 * checks. They live here so the two drivers share one implementation, but as
 * free functions they cost nothing to a consumer that doesn't call them: the
 * unit testbed constructs a GdnRunner and sets shape fields directly, never
 * touching a command line, and is wholly unaffected.
 * --------------------------------------------------------------------------- */

/* Parse the six GDN shape flags (--batch / --num_v_heads / --num_k_heads /
 * --head_k_dim / --head_v_dim / --seq_len) from `cmd`, leaving any field whose
 * flag is absent at its incoming default. Other flags (iterations, verify,
 * bm_name, ...) are the caller's concern. */
inline void parse_gdn_shape(cutlass::CommandLine& cmd,
                            int& batch, int& num_v_heads, int& num_k_heads,
                            int& head_k_dim, int& head_v_dim, int& seq_len) {
  cmd.get_cmd_line_argument("batch",       batch,       batch);
  cmd.get_cmd_line_argument("num_v_heads", num_v_heads, num_v_heads);
  cmd.get_cmd_line_argument("num_k_heads", num_k_heads, num_k_heads);
  cmd.get_cmd_line_argument("head_k_dim",  head_k_dim,  head_k_dim);
  cmd.get_cmd_line_argument("head_v_dim",  head_v_dim,  head_v_dim);
  cmd.get_cmd_line_argument("seq_len",     seq_len,     seq_len);
}

/* Validate a parsed GDN shape. Returns true iff the shape is launchable.
 * `prefix` tags the diagnostics (e.g. "" for the example, "[GDN benchmark] "
 * for the benchmark). The check order matters: positivity is verified before
 * the modulo so a zero num_k_heads cannot divide-by-zero -- mirrors the order
 * in xe35_chunk_gated_delta_rule_launch.hpp. */
inline bool validate_gdn_shape(int batch, int num_v_heads, int num_k_heads,
                               int head_k_dim, int head_v_dim, int seq_len,
                               char const* prefix) {
  if (batch <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
      head_k_dim <= 0 || head_v_dim <= 0 || seq_len <= 0) {
    std::cerr << prefix << "Error: shape parameters must be positive\n";
    return false;
  }
  if (num_v_heads % num_k_heads != 0) {
    std::cerr << prefix << "Error: num_v_heads (" << num_v_heads
              << ") must be a multiple of num_k_heads (" << num_k_heads << ")\n";
    return false;
  }
  /* The kernels tile the head dims into kChunkSize-wide 2D blocks
   * (`for (dv = 0; dv < head_v_dim / chunk_size; ++dv)` etc.), so a head dim
   * that is not a multiple of kChunkSize would silently drop its remainder. */
  constexpr int C = cutlass::gdn::kChunkSize;
  if (head_k_dim % C != 0 || head_v_dim % C != 0) {
    std::cerr << prefix << "Error: head_k_dim (" << head_k_dim
              << ") and head_v_dim (" << head_v_dim
              << ") must each be a multiple of the chunk size (" << C << ")\n";
    return false;
  }
  return true;
}

/* ---------------------------------------------------------------------------
 * CLI Options structs.
 *
 * The example driver and the benchmark driver each parse a command line into
 * their own Options before handing plain shape fields to a GdnRunner. Both
 * structs live here (next to parse_gdn_shape / validate_gdn_shape, which they
 * call) so all CLI parsing has one home. They are plain structs, NOT members
 * of GdnRunner: the unit testbed constructs a GdnRunner and sets shape fields
 * directly, never an Options, so it pays nothing for them. The defaults below
 * keep each struct fully usable with no flags supplied.
 * --------------------------------------------------------------------------- */

/* Example driver options. Defaults are sized for the CRI simulator (single
 * sub-chunk problem) and stay passive until parse() is called. */
struct ExampleOptions {
  bool help = false;
  bool error = false;
  /* Set by parse() to true iff the user explicitly supplied any of
   * --batch / --num_v_heads / --num_k_heads / --head_k_dim / --head_v_dim /
   * --seq_len. The example driver uses this to decide between iterating its
   * hard-coded case table and running a single CLI-driven case. */
  bool shape_overridden = false;

  /* Defaults sized for the CRI simulator: a single sub-chunk problem
   * (seq_len < kChunkSize=64) with minimal head count, one timed iteration.
   * On BMG (real hardware) the ctest passes no shape flags, so the driver
   * iterates its built-in case table instead of this single case; override on
   * the command line to run an arbitrary shape on either target. */
  int batch       = 1;
  int num_v_heads = 4;
  int num_k_heads = 1;        // GQA: num_v_heads must be a multiple of num_k_heads
  /* Head dims default to the compile-time HEAD_K_DIM / HEAD_V_DIM (128 / 128)
   * but are overridable via --head_k_dim / --head_v_dim. They must be positive
   * multiples of the kernel chunk size (kChunkSize=64): the kernels iterate
   * `head_*_dim / chunk_size` 2D blocks, so a non-multiple silently drops the
   * remainder. Only 128/128 is validated end-to-end by the unit test. */
  int head_k_dim  = HEAD_K_DIM;
  int head_v_dim  = HEAD_V_DIM;
  int seq_len     = 16;       // chunked kernel pads internally to kChunkSize=64
  int iterations  = 1;
  int verify      = 1;
  unsigned seed   = 0xC0FFEEu;

  std::ostream& print_usage(std::ostream& os) const {
    os << "14_xe35_gdn_attention -- Xe35 chunkwise Gated DeltaNet attention\n\n"
       << "Defaults are sized for the CRI simulator (single sub-chunk);\n"
       << "with no shape flags the driver iterates its built-in case table.\n\n"
       << "Options:\n"
       << "  --help                   Show this help\n"
       << "  --batch=<int>            Number of sequences in the batch (default 1)\n"
       << "  --num_v_heads=<int>      Number of value heads (default 4)\n"
       << "  --num_k_heads=<int>      Number of key heads (default 1, GQA: must divide num_v_heads)\n"
       << "  --head_k_dim=<int>       Key/query head dim (default " << HEAD_K_DIM << "; must be a multiple of chunk size 64)\n"
       << "  --head_v_dim=<int>       Value head dim (default " << HEAD_V_DIM << "; must be a multiple of chunk size 64)\n"
       << "  --seq_len=<int>          Tokens per sequence (default 16; kernel chunk size is 64)\n"
       << "  --iterations=<int>       Timed iterations (default 1)\n"
       << "  --verify=<0|1>           Run host reference and compare (default 1)\n"
       << "  --seed=<int>             RNG seed (default 0xC0FFEE)\n"
       << "\nDefault head dims (compile-time HEAD_K_DIM / HEAD_V_DIM): head_k_dim="
       << HEAD_K_DIM << " head_v_dim=" << HEAD_V_DIM << "\n";
    return os;
  }

  void parse(int argc, const char** argv) {
    cutlass::CommandLine cmd(argc, argv);
    if (cmd.check_cmd_line_flag("help")) { help = true; return; }
    /* Snapshot whether any shape flag was supplied BEFORE calling
     * get_cmd_line_argument; the cutlass CommandLine helper has no
     * "was value defaulted" return, so we probe the keys directly.
     * check_cmd_line_flag matches both bare `--name` and `--name=value`. */
    shape_overridden = cmd.check_cmd_line_flag("batch")       ||
                       cmd.check_cmd_line_flag("num_v_heads") ||
                       cmd.check_cmd_line_flag("num_k_heads") ||
                       cmd.check_cmd_line_flag("head_k_dim")  ||
                       cmd.check_cmd_line_flag("head_v_dim")  ||
                       cmd.check_cmd_line_flag("seq_len");
    parse_gdn_shape(cmd, batch, num_v_heads, num_k_heads,
                    head_k_dim, head_v_dim, seq_len);
    cmd.get_cmd_line_argument("iterations",  iterations,  iterations);
    cmd.get_cmd_line_argument("verify",      verify,      verify);
    int seed_int = static_cast<int>(seed);
    cmd.get_cmd_line_argument("seed", seed_int, seed_int);
    seed = static_cast<unsigned>(seed_int);

    error = !validate_gdn_shape(batch, num_v_heads, num_k_heads,
                                head_k_dim, head_v_dim, seq_len);
    /* Negative is nonsensical and rejected outright. Zero is an established
     * convention across the CRI examples (CRI_DEFAULT_ITERATIONS in
     * examples/CMakeLists.txt) meaning "single untimed run" -- clamp it to 1
     * rather than error so existing CRI ctest invocations keep working. */
    if (iterations < 0) {
      std::cerr << "Error: --iterations (" << iterations
                << ") must not be negative\n";
      error = true;
    }
  }
};

/* Benchmark driver options. Parsed from one line of the benchmark
 * configuration .in file, e.g.:
 *   GdnConfig_BF16_FP32 --bm_name=gdn_bf16 --batch=1 --num_v_heads=64
 *                       --num_k_heads=16 --head_k_dim=128 --head_v_dim=128
 *                       --seq_len=4096 */
struct BenchmarkOptions {
  bool error = false;

  int batch       = 1;
  int num_v_heads = 8;
  int num_k_heads = 1;    // GQA: num_v_heads must be a multiple of num_k_heads
  int head_k_dim  = 128;
  int head_v_dim  = 128;
  int seq_len     = 1024;

  std::string bm_name = "GDN";

  void parse(int argc, const char** args) {
    cutlass::CommandLine cmd(argc, args);
    parse_gdn_shape(cmd, batch, num_v_heads, num_k_heads,
                    head_k_dim, head_v_dim, seq_len);
    cmd.get_cmd_line_argument("bm_name", bm_name, bm_name);

    error = !validate_gdn_shape(batch, num_v_heads, num_k_heads,
                                head_k_dim, head_v_dim, seq_len,
                                "[GDN benchmark] ");
  }

  std::string benchmark_name() const {
    std::ostringstream ss;
    ss << bm_name << "/"
       << batch       << "x"
       << num_v_heads << "x"
       << num_k_heads << "x"
       << seq_len     << "x"
       << head_k_dim  << "x"
       << head_v_dim;
    return ss.str();
  }
};

}  // namespace cutlass::gdn
