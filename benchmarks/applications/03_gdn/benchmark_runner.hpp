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
  \file benchmarks/gdn/benchmark_runner.hpp
  \brief Google Benchmark layer for the Xe35 chunkwise GDN attention kernel.

  Adds the benchmark-only concerns ON TOP of the shared cutlass::gdn::GdnRunner
  (applications/gdn_attention/gdn_runner.hpp): CLI option parsing from the
  config .in file, the analytic FLOP/byte performance model, and the Google
  Benchmark timed loop + counters. The device allocation / initialization /
  GDNArguments construction is owned by the shared runner, so this file no
  longer re-implements setup -- it holds one GdnRunner and times its launches.

  Performance metrics emitted:
    avg_tflops      : rough arithmetic throughput estimate (GFLOPs / ms = TFLOPS/s)
    avg_throughput  : memory bandwidth estimate (MB/ms = GB/s)
    avg_runtime_ms  : trimmed mean latency (best + worst iterations removed)
    best_runtime_ms : minimum latency over all iterations
    worst_runtime_ms: maximum latency over all iterations
*/

#pragma once

#include <benchmark/benchmark.h>

#include "cutlass/cutlass.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/GPU_Clock.hpp"

#include "../common.hpp"
#include "gdn_attention/gdn_runner.hpp"


namespace cutlass::benchmark::gdn {

// The benchmark options struct (CLI parsing from the config .in line) lives in
// the shared runner alongside the example options; alias it here so the
// registration macros and main.cpp keep their existing name.
using GDNBenchmarkOptions = cutlass::gdn::BenchmarkOptions;

// ---------------------------------------------------------------------------
// BenchmarkRunnerGDN<T, StateT>
//
// Owns one shared GdnRunner (allocations + init + arguments) and adds the
// analytic FLOP/byte model + the Google Benchmark timed loop. The run()
// signature matches the BenchmarkRegistry callback:
//   void run(::benchmark::State&, GDNBenchmarkOptions const&,
//            cutlass::KernelHardwareInfo const&)
//
//   T      - activation dtype (bfloat16_t)
//   StateT - SSM state dtype  (always float in current builds)
// ---------------------------------------------------------------------------

template <typename T, typename StateT>
struct BenchmarkRunnerGDN {

  cutlass::gdn::GdnRunner<T, StateT> core;

  // ---- performance model ----

  /* FLOP estimate for the chunkwise GDN forward pass.
   * Counts the algebraically required mul-add flops (1 MAC = 2 flops) of each
   * of the five stages; numeric constants in comments are dominant-term
   * derivations, sub-dominant scalar work (softplus, exp, eps adds) is
   * absorbed into small constants.
   *
   * Notation:
   *   C   = kChunkSize (= 64)
   *   D_k = head_k_dim,  D_v = head_v_dim
   *   H_k = num_k_heads, H_v = num_v_heads
   *   T   = total tokens after per-batch padding (== total_virtual_seqlen)
   *   N   = per-batch chunks summed across batches = ceil_sum(seq_len_b / C)
   *         (matches the stride the kernels use; for fixed seq_len:
   *          N = batch * ceil(seq_len / C))
   */
  double flop_estimate() const {
    constexpr int C    = cutlass::gdn::kChunkSize;
    const double T_tot = double(core.total_virtual_seqlen);
    const double N     = T_tot / double(C);
    const double Hk    = core.num_k_heads;
    const double Hv    = core.num_v_heads;
    const double Dk    = core.head_k_dim;
    const double Dv    = core.head_v_dim;

    /* Stage 1 -- chunk_prepare:
     *   per k-head per token: L2-norm of q and k each = 2*D_k mul-adds + 1 rsqrt + D_k scales
     *     ~= 5 * D_k flops (q and k together)
     *   per v-head per token: softplus(a+dt_bias) * -exp(A_log) + cumsum
     *     ~= 8 flops */
    const double prepare = 5.0 * T_tot * Hk * Dk + 8.0 * T_tot * Hv;

    /* Stage 2 -- chunk_compute_A:
     *   L[m,n] = (K_m . K_n) * exp(a[m]-a[n]) * b[m]  for m,n in [0,C)
     *   Counted as a full C x C x D_k GEMM (2 flops per MAC); the lower-tri
     *   masking is done post-hoc on the same flops the hardware computes.
     *   Plus the per-element exp-and-multiply scaling (~3 flops). */
    const double compute_A = N * Hv * (2.0 * C * C * Dk + 3.0 * C * C);

    /* Stage 3 -- chunk_inverse:
     *   Block forward substitution to invert a CxC lower-triangular matrix.
     *   Classical complexity is C^3/3 mul-adds = 2*C^3/3 flops. The
     *   DPAS-tiled path performs additional rearrange ops but the asymptotic
     *   flop count is the same. */
    const double inverse = N * Hv * (2.0 * C * C * C / 3.0);

    /* Stage 4 -- chunk_compute_wu:
     *   U = L^-1 * V * diag(b)         -> (CxC) * (CxD_v) GEMM = 2*C^2*D_v
     *   W = L^-1 * K * diag(exp(a)*b)  -> (CxC) * (CxD_k) GEMM = 2*C^2*D_k
     *   Per-element diag scaling adds ~C*(D_v + D_k) flops; negligible. */
    const double compute_wu = N * Hv * 2.0 * C * C * (Dk + Dv);

    /* Stage 5 -- chunk_fwd_o:
     *   O2  = Q * K^T                       -> (CxD_k) * (D_k x C) = 2*C^2*D_k
     *   O_intra = O2 * U                    -> (CxC) * (CxD_v)     = 2*C^2*D_v
     *   O_inter = Q * S^T * exp(g)          -> (CxD_k) * (D_k xD_v) = 2*C*D_k*D_v
     *                                          (S_prev contribution; counted
     *                                           every chunk; the leading chunk
     *                                           with no prev state is a small
     *                                           constant overcharge)
     *   S_out  = exp(g_last)*S_prev + U^T * K_scaled
     *                                       -> (D_v xC) * (CxD_k)   = 2*C*D_k*D_v
     *   Subdominant: per-element exp(g[m]-g[n]) masking & exp(g) scales,
     *   ~3*C^2 flops; absorbed. */
    const double fwd_o = N * Hv * (2.0 * C * C * Dk      // O2 = Q*K^T
                                 + 2.0 * C * C * Dv      // O2 * U
                                 + 2.0 * C * Dk * Dv     // Q * S^T * exp(g)
                                 + 2.0 * C * Dk * Dv);   // S update

    return (prepare + compute_A + inverse + compute_wu + fwd_o) * 1e-9; // GFLOPs
  }

  /* Memory-traffic estimate.
   *
   * Charges each tensor for every kernel-stage in which it is read or
   * written, rather than once over the whole pipeline. Workspaces dominate
   * and are NOT scratched in cache between stages on this hardware, so they
   * are paid for again in every consuming stage. Pre-launch host copies and
   * post-launch readbacks are NOT counted; the benchmark times only kernel
   * execution. */
  double bytes_estimate() const {
    constexpr double sT = sizeof(T);
    constexpr double sS = sizeof(StateT);
    constexpr int C = cutlass::gdn::kChunkSize;

    const double tvs = double(core.total_virtual_seqlen);
    const double ts  = double(core.total_seqlen);
    const double Hk  = core.num_k_heads;
    const double Hv  = core.num_v_heads;
    const double Dk  = core.head_k_dim;
    const double Dv  = core.head_v_dim;
    const double Bn  = core.batch;

    // ---- per-tensor sizes (in bytes) ----
    const double sz_q       = tvs * Hk * Dk * sT;
    const double sz_k       = tvs * Hk * Dk * sT;
    const double sz_v       = tvs * Hv * Dv * sT;
    const double sz_a       = Hv * tvs * sizeof(float);
    const double sz_b       = Hv * tvs * sizeof(float);
    const double sz_A_log   = Hv * sizeof(float);
    const double sz_dt_bias = Hv * sT;
    const double sz_O       = ts  * Hv * Dv * sT;
    const double sz_ssm     = Bn  * Hv * Dv * Dk * sS;
    const double sz_A_ws    = Hv  * tvs * C   * sT;
    const double sz_w_ws    = Hv  * tvs * Dk  * sT;
    const double sz_u_ws    = Hv  * tvs * Dv  * sT;

    // ---- per-stage traffic ----
    // Stage 1: reads q,k,a,A_log,dt_bias; writes q,k,a (in place).
    const double bytes_prepare    = 2.0*sz_q + 2.0*sz_k + 2.0*sz_a + sz_A_log + sz_dt_bias;
    // Stage 2: reads k,b,a; writes A_workspace.
    const double bytes_compute_A  = sz_k + sz_b + sz_a + sz_A_ws;
    // Stage 3: reads A_workspace; writes A_workspace (in place).
    const double bytes_inverse    = 2.0 * sz_A_ws;
    // Stage 4: reads A_workspace, q, k, v, b, a, A_log, dt_bias; writes w, u.
    const double bytes_compute_wu = sz_A_ws + sz_q + sz_k + sz_v + sz_b + sz_a
                                  + sz_A_log + sz_dt_bias + sz_w_ws + sz_u_ws;
    // Stage 5: reads q, k, a, w, u, A_workspace (as O2 scratch), ssm_state;
    //          writes core_attn_out, A_workspace (O2), ssm_state.
    const double bytes_fwd_o      = sz_q + sz_k + sz_a + sz_w_ws + sz_u_ws
                                  + 2.0 * sz_A_ws + sz_O + 2.0 * sz_ssm;

    const double bytes = bytes_prepare + bytes_compute_A + bytes_inverse
                       + bytes_compute_wu + bytes_fwd_o;
    return bytes * 1e-6;  // MB
  }

  // ---- Google Benchmark integration ----

  void run(::benchmark::State& state,
           const GDNBenchmarkOptions& opts,
           const cutlass::KernelHardwareInfo& /* hw_info */) {

    core.batch       = opts.batch;
    core.num_v_heads = opts.num_v_heads;
    core.num_k_heads = opts.num_k_heads;
    core.head_k_dim  = opts.head_k_dim;
    core.head_v_dim  = opts.head_v_dim;
    core.seq_len     = opts.seq_len;
    core.initialize();

    // Drain any default-queue work (tensor initialisation submitted by
    // initialize_block / copy_from_host) before the timed loop.
    // SYCLTimer::start() calls compat::get_default_queue().wait() internally;
    // if that drains initialisation work on the first iteration it inflates the
    // wait time to thousands of ms and leaves ms_elapsed ≈ 0.
    compat::get_default_queue().wait();

    state.counters["batch"]       = opts.batch;
    state.counters["num_v_heads"] = opts.num_v_heads;
    state.counters["num_k_heads"] = opts.num_k_heads;
    state.counters["head_k_dim"]  = opts.head_k_dim;
    state.counters["head_v_dim"]  = opts.head_v_dim;
    state.counters["seq_len"]     = opts.seq_len;

    initialize_timing_counters(state);

    const double gflop    = flop_estimate();
    const double mega_bytes = bytes_estimate();

    /*
    * Every iteration re-runs core.launch(); the kernel mutates q/k/a/ssm_state
    * in place, so only iteration 0 runs on fresh inputs. Harmless for latency
    * (flop/byte count is data-independent), but the ssm_state keeps
    * accumulating across launches -- re-init under PauseTiming() here if a
    * data-dependent timing path is ever added.
    */
    for (auto _ : state) {
      /* Time with GPU_Clock (the repo-wide timing utility). Correct on both
      * builds: with CUTLASS_SYCL_PROFILING_ENABLED it sums the 5 stage events
      * that the launcher registers with EventManager (the GdnRunner queue
      * enables profiling); without it, SYCLTimer falls back to a host
      * wall-clock span bracketed by `compat::get_default_queue().wait()`. The
      * GDN kernel runs on the GdnRunner's private in-order queue, not the
      * default queue -- the default queue was drained above (and stays idle)
      * so that internal wait is a no-op, and `queue.wait_and_throw()` on the
      * private queue before `timer.milliseconds()` makes the wall-clock span
      * cover exactly the private-queue launch.
      */
      GPU_Clock timer;
      timer.start();
      auto status = core.launch();
      core.queue.wait_and_throw();
      double ms_elapsed = timer.milliseconds();

      if (status != cutlass::Status::kSuccess) {
        state.SkipWithError("GDN kernel launch failed");
        return;
      }

      update_timing_counters(state, ms_elapsed);
      state.SetIterationTime(ms_elapsed / 1000.0);
    }

    finalize_timing_counters(state, gflop, mega_bytes);
  }

 private:
  static void initialize_timing_counters(::benchmark::State& state) {
    state.counters["total_runtime_ms"]  = 0.0;
    state.counters["avg_runtime_ms"]    = 0.0;
    state.counters["best_runtime_ms"]   = std::numeric_limits<double>::max();
    state.counters["worst_runtime_ms"]  = std::numeric_limits<double>::lowest();
  }

  static void update_timing_counters(::benchmark::State& state, double ms) {
    state.PauseTiming();
    state.counters["total_runtime_ms"] += ms;
    state.counters["best_runtime_ms"]   = std::min<double>(state.counters["best_runtime_ms"],  ms);
    state.counters["worst_runtime_ms"]  = std::max<double>(state.counters["worst_runtime_ms"], ms);
    state.ResumeTiming();
  }

  static void finalize_timing_counters(::benchmark::State& state,
                                       double gflop, double mega_bytes) {
    const auto iters = static_cast<double>(state.iterations());
    // Trimmed mean: remove best + worst if we have enough iterations.
    double denom = (iters > 2) ? (iters - 2) : iters;
    double trimmed_total = state.counters["total_runtime_ms"]
                         - state.counters["best_runtime_ms"]
                         - state.counters["worst_runtime_ms"];
    if (iters <= 2) trimmed_total = state.counters["total_runtime_ms"];
    state.counters["avg_runtime_ms"]   = trimmed_total / denom;
    state.counters["avg_tflops"]       = gflop      / state.counters["avg_runtime_ms"];
    state.counters["avg_throughput"]   = mega_bytes  / state.counters["avg_runtime_ms"];
    state.counters["best_tflop"]       = gflop      / state.counters["best_runtime_ms"];
    state.counters["best_bandwidth"]   = mega_bytes  / state.counters["best_runtime_ms"];
  }
};

}  // namespace cutlass::benchmark::gdn

// ---------------------------------------------------------------------------
// Registration macros (mirrors CUTLASS_CREATE_FMHA_BENCHMARK pattern)
//
// CUTLASS_CREATE_GDN_BENCHMARK(F) — define a static trampoline function for F.
// CUTLASS_GDN_BENCHMARK(F)        — register F with the BenchmarkRegistry.
// ---------------------------------------------------------------------------

#define CUTLASS_GDN_BENCHMARK(F) \
  cutlass::benchmark::BenchmarkRegistry<cutlass::benchmark::gdn::GDNBenchmarkOptions>::Register( \
      #F, &F##_func)

#define CUTLASS_CREATE_GDN_BENCHMARK(F)                                   \
  static void F##_func(                                                   \
      ::benchmark::State& state,                                          \
      cutlass::benchmark::gdn::GDNBenchmarkOptions const& options,        \
      cutlass::KernelHardwareInfo const& hw_info) {                       \
    auto bench = cutlass::benchmark::gdn::BenchmarkRunnerGDN<             \
        typename F::ElementT, typename F::StateT>();                      \
    bench.run(state, options, hw_info);                                   \
  }
