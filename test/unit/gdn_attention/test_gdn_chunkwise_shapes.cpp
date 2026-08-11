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
 *  \brief Unit tests for the Xe35 chunkwise Gated DeltaNet attention kernel:
 *  primary shape / batching control-flow coverage.
 *
 *  Covers the "base" shapes that establish control-flow coverage across
 *  chunk counts and batch sizes:
 *
 *    - seq_len=64  : one chunk per batch (single trip through the per-chunk
 *                    state-update loop in `chunk_fwd_o_kernel`).
 *    - seq_len=256 : four chunks per batch (state hand-off between chunks is
 *                    exercised, including the `has_prev = (c != 0)` branch).
 *    - seq_len=65/127 : non-multiple-of-64 lengths, so the final chunk is
 *                    partial (current_chunk_size < chunk_size). Guards the
 *                    tail-sizing / zero-pad path that the aligned shapes miss.
 *    - multi-batch (batch>1) : multiple equal-length sequences, exercising
 *                    query_start_loc / cache_indices folding and per-batch
 *                    state slots (the example/benchmark batching).
 */

#include <gtest/gtest.h>

#include "cutlass/bfloat16.h"

#include "gdn_chunkwise_testbed.hpp"

namespace cutlass {

TEST(XE35_GDN_Chunkwise_bf16, seq_len_64) {
  test::gdn_attention::ChunkwiseTestbed<cutlass::bfloat16_t, float> tb;
  tb.num_v_heads = 16;
  tb.num_k_heads = 4;
  tb.seq_len = 64;
  EXPECT_TRUE(tb.run());
}

TEST(XE35_GDN_Chunkwise_bf16, seq_len_256) {
  test::gdn_attention::ChunkwiseTestbed<cutlass::bfloat16_t, float> tb;
  tb.seq_len = 256;
  EXPECT_TRUE(tb.run());
}

/* Partial tail, minimal remainder: seq_len=65 => 2 chunks, the second with a
 * single valid row (current_chunk_size = 1). Sole guard on the partial-tail
 * path the multiple-of-64 shapes never reach; row=1 is the extreme off-by-one. */
TEST(XE35_GDN_Chunkwise_bf16, seq_len_65_partial_tail) {
  test::gdn_attention::ChunkwiseTestbed<cutlass::bfloat16_t, float> tb;
  tb.num_v_heads = 16;
  tb.num_k_heads = 4;
  tb.seq_len     = 65;
  EXPECT_TRUE(tb.run());
}

/* Partial-tail chunk, near-full remainder: seq_len=127 => 2 chunks, the second
 * holding 63 valid rows (current_chunk_size = 63). Complements seq_len=65 by
 * exercising the tail at the opposite end of its range, so both a nearly-empty
 * and a nearly-full final chunk cover the tail-sizing arithmetic. */
TEST(XE35_GDN_Chunkwise_bf16, seq_len_127_partial_tail) {
  test::gdn_attention::ChunkwiseTestbed<cutlass::bfloat16_t, float> tb;
  tb.num_v_heads = 16;
  tb.num_k_heads = 4;
  tb.seq_len     = 127;
  EXPECT_TRUE(tb.run());
}

/* Uniform multi-batch: batch>1 with equal per-sequence lengths. Exercises the
 * query_start_loc / cache_indices folding and per-batch state slots with more
 * than one sequence -- the batch=1 shapes above never reach this. Equal,
 * chunk-aligned lengths, matching how the example/benchmark runners batch. */
TEST(XE35_GDN_Chunkwise_bf16, multi_batch_uniform) {
  test::gdn_attention::ChunkwiseTestbed<cutlass::bfloat16_t, float> tb;
  tb.batch       = 4;
  tb.num_v_heads = 16;
  tb.num_k_heads = 4;
  tb.seq_len     = 128;  // 2 chunks per sequence
  EXPECT_TRUE(tb.run());
}

}  // namespace cutlass
