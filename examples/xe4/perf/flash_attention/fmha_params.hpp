/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation, All rights reserved.
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

#include <string>

/*
 * Structure to hold parameters for Flash Attention performance tests
 * 
 * Note on data types:
 * - K and V tensors always share the same data type (dtype_kv) in the kernel
 * - P (attention probabilities) type is constrained to match KV type
 * - These constraints are enforced by the underlying FMHA kernel implementation
 */
struct FMHATestParams {
    std::string test_level;     // Test level: "L0", "L1", "L2", etc.
    std::string state;          // Test state: "enabled" or "disabled"
    
    // Problem shape parameters
    int batch;                  // Batch size (B)
    int num_heads;              // Number of attention heads (H)
    int seq_len_qo;             // Sequence length for Query/Output (Tq)
    int seq_len_kv;             // Sequence length for Key/Value (Tkv)
    int head_size_qk;           // Head dimension for Q*K (D_qk)
    int head_size_vo;           // Head dimension for V*O (D_vo)
    
    // Data types (simplified - K/V share type, P is constrained to match KV)
    std::string dtype_q;        // Data type for Q (fp16, bf16, etc.)
    std::string dtype_kv;       // Data type for K and V (they must be the same)
    std::string dtype_s;        // Data type for S (attention scores)
    std::string dtype_o;        // Data type for O (output)
    std::string dtype_acc;      // Accumulator data type
    
    // Tile shape configuration
    int tile_q_blk;             // Q block size in tile
    int tile_v_head_dim;        // V head dimension in tile
    int tile_kv_blk;            // KV block size in tile
    int tile_qk_head_dim;       // QK head dimension in tile
    
    // Kernel configuration
    int num_softmax_warps;      // Number of warps for softmax
    int num_thread_per_row;     // Number of threads per row
    int softmax_unroll;         // Softmax unroll factor
    int softmax_num_stage;      // Number of softmax stages
    bool is_persistent;         // Persistent kernel mode
    
    // Optional features
    bool causal_mask;           // Enable causal masking
    float softmax_scale;        // Softmax scaling factor (0.0 = use default 1/sqrt(head_size_qk))
};

// Array of FMHA test parameters - configured for various scenarios
static FMHATestParams fmha_test_params[] = {
    // ==================== L0: Smoke Tests (Basic Functionality) ====================

    // Match xe4_fmha_fwd.cpp default config: FP16 all, non-persistent, default Options
    {/* test_level */ "L0",
     /* state */ "enabled",
     /* batch, num_heads, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo */
     1, 1, 128, 8192, 128, 128,
     /* dtype_q, dtype_kv, dtype_s, dtype_o, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* tile_q_blk, tile_v_head_dim, tile_kv_blk, tile_qk_head_dim */
     128, 128, 512, 128,
     /* num_softmax_warps, num_thread_per_row, softmax_unroll, softmax_num_stage */
     16, 16, 2, 2,
     /* is_persistent */ false,
     /* causal_mask */ false,
     /* softmax_scale */ 0.0f},  // 0

     //decode
     {/* test_level */ "L0",
     /* state */ "enabled",
     /* batch, num_heads, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo */
     1, 1, 1, 8192, 128, 128,
     /* dtype_q, dtype_kv, dtype_s, dtype_o, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* tile_q_blk, tile_v_head_dim, tile_kv_blk, tile_qk_head_dim */
     128, 128, 512, 128,
     /* num_softmax_warps, num_thread_per_row, softmax_unroll, softmax_num_stage */
     16, 16, 2, 2,
     /* is_persistent */ false,
     /* causal_mask */ false,
     /* softmax_scale */ 0.0f},  // 1

    // Flash Attention V3: FP16, 2 query heads, non-persistent (from FlashAttenFwdFP16 config)
    {/* test_level */ "L0",
     /* state */ "enabled",
     /* batch, num_heads, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo */
     1, 2, 128, 8192, 128, 128,
     /* dtype_q, dtype_kv, dtype_s, dtype_o, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* tile_q_blk, tile_v_head_dim, tile_kv_blk, tile_qk_head_dim */
     128, 128, 512, 128,
     /* num_softmax_warps, num_thread_per_row, softmax_unroll, softmax_num_stage */
     16, 16, 2, 2,
     /* is_persistent */ false,
     /* causal_mask */ false,
     /* softmax_scale */ 0.0f},  // 2

     

};
