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
 **************************************************************************************************/

/*!
  \file xe35_gdn_attention_runner.hpp
  \brief Example driver layer for the Xe35 chunkwise GDN attention example.

  Adds the example-only concerns ON TOP of the shared cutlass::gdn::GdnRunner
  (applications/gdn_attention/gdn_runner.hpp): CLI Options parsing/validation,
  and timed-launch reporting.
  The allocation / initialization / launch / verification core lives in the
  shared runner, so this header carries no setup code -- only the Options struct
  and a thin GdnExampleRunner that wires Options -> GdnRunner -> verify + time.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/GPU_Clock.hpp"

#include "sycl_common.hpp"

#include "gdn_attention/gdn_runner.hpp"

/* ---------------------------------------------------------------------------
 * Options (CLI). The struct + all parsing now live in the shared runner
 * (cutlass::gdn::ExampleOptions); alias it here so this driver and
 * xe35_gdn_attention.cpp keep their existing unqualified `Options` name.
 * --------------------------------------------------------------------------- */

using Options = cutlass::gdn::ExampleOptions;

/* ---------------------------------------------------------------------------
 * Example driver: Options -> shared GdnRunner -> verify + timed report.
 * --------------------------------------------------------------------------- */

template <typename T, typename StateT>
struct GdnExampleRunner {

  Options opt;
  cutlass::gdn::GdnRunner<T, StateT> core;

  explicit GdnExampleRunner(Options o) : opt(std::move(o)) {
    core.batch       = opt.batch;
    core.num_v_heads = opt.num_v_heads;
    core.num_k_heads = opt.num_k_heads;
    core.head_k_dim  = opt.head_k_dim;
    core.head_v_dim  = opt.head_v_dim;
    core.seq_len     = opt.seq_len;
    core.seed        = opt.seed;
  }

  int run() {
    core.initialize();

    /* First launch serve double duty -- it is both the verification launch AND timed
     * iteration 0 -- so verify costs no separate launch. The remaining
     * iterations-1 launches are timed together and folded in; `ms` is the
     * mean over all opt.iterations. */
    GPU_Clock timer;
    compat::get_default_queue().wait();  // match time_launches: keep the span clean
    timer.start();
    auto status = core.launch();
    core.queue.wait_and_throw();
    const double first_ms = timer.milliseconds();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "[error] kernel launch failed: status=" << int(status) << "\n";
      return -1;
    }

    if (opt.verify) {
      // E2E verify against the recurrent oracle, reading the first launch's output.
      if (!core.verify_recurrent(cutlass::gdn::reference::recurrent::kTolE2E,
                                 cutlass::gdn::reference::recurrent::kTolE2E,
                                 /*print=*/false)) {
        std::cout << "[verify] FAIL  device output does not match host reference\n";
        return -2;
      }
      std::cout << "[verify] PASS\n";
    }

    /* Time the remaining launches in one span and average with the first.
     * These re-run after the first launch already mutated q/k/a/ssm_state in
     * place, so iters>1 time the kernel on overwritten inputs. Harmless for
     * latency (flop/byte count is data-independent), but the numbers are not
     * a fresh-input measurement. */
    const int rest = (opt.iterations > 1) ? opt.iterations - 1 : 0;
    double rest_total = 0.0;
    if (rest > 0) {
      // Match SYCLTimer: drain the default queue so its internal waits are
      // no-ops and the span covers only the private-queue launches.
      compat::get_default_queue().wait();
      GPU_Clock rest_timer;
      rest_timer.start();
      for (int i = 0; i < rest; ++i) {
        status = core.launch();
        if (status != cutlass::Status::kSuccess) {
          std::cerr << "[error] kernel launch failed: status=" << int(status) << "\n";
          return -1;
        }
      }
      core.queue.wait_and_throw();
      rest_total = rest_timer.milliseconds();
    }
    const double ms = (first_ms + rest_total) / std::max(1, opt.iterations);

    std::cout << "Avg kernel time: " << ms << " ms over "
              << opt.iterations << " iterations\n"
              << "  batch=" << core.batch
              << " seq_len=" << core.seq_len
              << " num_v_heads=" << core.num_v_heads
              << " num_k_heads=" << core.num_k_heads
              << " head_k_dim=" << core.head_k_dim
              << " head_v_dim=" << core.head_v_dim << "\n";
    return 0;
  }
};
