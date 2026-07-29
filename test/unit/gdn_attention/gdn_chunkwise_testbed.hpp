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

/*! \file
 *  \brief Unit testbed for the Xe35 chunkwise GDN attention kernel.
 *
 *  A thin gtest-facing wrapper over the shared cutlass::gdn::GdnRunner (in
 *  applications/gdn_attention/gdn_runner.hpp): set the shape fields,
 *  initialize + launch via the runner, then require the recurrent host oracle
 *  to pass per-element at kTolE2E. The allocation / initialization / launch /
 *  verification code is owned by the runner so all three GDN usages (example,
 *  benchmark, unit test) stay in lockstep. */

#pragma once

#include "cutlass/bfloat16.h"

#include "gdn_attention/gdn_runner.hpp"

namespace test::gdn_attention {

template <typename T, typename StateT>
struct ChunkwiseTestbed {
  // ----- fixed shape (caller-supplied; defaults match the shared runner) -----
  int batch       = 1;
  int num_v_heads = 64;
  int num_k_heads = 16;
  int head_k_dim  = 128;
  int head_v_dim  = 128;
  int seq_len     = 64;
  unsigned seed   = 0xC0FFEEu;

  // Per-element pass tolerance (like FMHA), default kTolE2E.
  float atol = cutlass::gdn::reference::recurrent::kTolE2E;
  float rtol = cutlass::gdn::reference::recurrent::kTolE2E;

  bool run() {
    cutlass::gdn::GdnRunner<T, StateT> r;
    r.batch       = batch;
    r.num_v_heads = num_v_heads;
    r.num_k_heads = num_k_heads;
    r.head_k_dim  = head_k_dim;
    r.head_v_dim  = head_v_dim;
    r.seq_len     = seq_len;
    r.seed        = seed;

    r.initialize();
    if (r.launch() != cutlass::Status::kSuccess) return false;
    r.queue.wait_and_throw();

    return r.verify_recurrent(atol, rtol);
  }
};

}  // namespace test::gdn_attention
