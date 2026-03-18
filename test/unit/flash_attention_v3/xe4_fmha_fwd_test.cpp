/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation, All rights reserved.
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

#include <cmath>

#include <cute/tensor.hpp>

#include "../common/cutlass_unit_test.h"
#include "flash_attention_v3_testbed.hpp"


namespace cutlass {
namespace test {
namespace flash_attention_v3 {

struct ProblemConfig_FP16FP16FP16FP16 {
  using ElementInputQ = fp16;
  using ElementInputKV = fp16;
  using ElementS = fp16;
  using ElementP = ElementInputKV;
  using ElementAccumulator = float;
  using ElementOutput = fp16;
};

struct ProblemConfig_FP16FP16FP32FP16 {
  using ElementInputQ = fp16;
  using ElementInputKV = fp16;
  using ElementS = float;
  using ElementP = ElementInputKV;
  using ElementAccumulator = float;
  using ElementOutput = fp16;
};

struct ProblemConfig_BF16BF16BF16BF16 {
  using ElementInputQ = bf16;
  using ElementInputKV = bf16;
  using ElementS = bf16;
  using ElementP = ElementInputKV;
  using ElementAccumulator = float;
  using ElementOutput = bf16;
};

struct ProblemConfig_BF16BF16FP32BF16 {
  using ElementInputQ = bf16;
  using ElementInputKV = bf16;
  using ElementS = float;
  using ElementP = ElementInputKV;
  using ElementAccumulator = float;
  using ElementOutput = bf16;
};

TEST(XE4_FMHA_FWD, DISABLED_smoke_fp16) {
  // Xe4 fwd kernel tile is fixed to 128x128x512x128 (M,N,K,cluster) for this smoke test.
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  // Problem dims: batch=1, num_heads=1, seq_len_qo=512, seq_len_kv=8192, head_size_qk=128, head_size_vo=128
  constexpr int batch = 1;
  constexpr int heads = 1;
  constexpr int seq_q = 512;
  constexpr int seq_kv = 8192;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// Case 1: batch=2, num_heads=1, seq_len_qo=1024, seq_len_kv=1024
TEST(XE4_FMHA_FWD, case1_batch2_seq1024) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 2;
  constexpr int heads = 1;
  constexpr int seq_q = 1024;
  constexpr int seq_kv = 1024;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// Case 2: batch=4, num_heads=16, seq_len_qo=128, seq_len_kv=512
// Note: K tile=512 requires seq_kv >= 512 for FP16
TEST(XE4_FMHA_FWD, DISABLED_case2_batch4_heads16) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 4;
  constexpr int heads = 16;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 512;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// Case 3: Long sequence - batch=1, num_heads=8, seq_len_qo=2048, seq_len_kv=2048
// DISABLED: Heavy test (~457s) - workload: 1x8x2048x2048 = 33.6M
TEST(XE4_FMHA_FWD, DISABLED_case3_heads8_seq2048) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 8;
  constexpr int seq_q = 2048;
  constexpr int seq_kv = 2048;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// Case 4: Asymmetric KV cache - batch=1, num_heads=4, seq_len_qo=256, seq_len_kv=4096
TEST(XE4_FMHA_FWD, DISABLED_case4_asymmetric_kv_cache) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 4096;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// Case 5: Small batch, many heads - batch=1, num_heads=32, seq_len_qo=512, seq_len_kv=512
TEST(XE4_FMHA_FWD, DISABLED_case5_heads32_seq512) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 32;
  constexpr int seq_q = 512;
  constexpr int seq_kv = 512;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// Case 6: Multi-batch multi-head test
TEST(XE4_FMHA_FWD, case6_batch2_heads4_seq512) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 2;
  constexpr int heads = 4;
  constexpr int seq_q = 512;
  constexpr int seq_kv = 512;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

/*
 EXHAUSTIVE COVERAGE TESTS
 TileShape: <M=128, N=128, K=512, HeadDim=128>
 Constraints for FP16:
   - seq_q must be divisible by M (128)
   - seq_kv must be divisible by K (512)
   - head_dim_qk = 128 (fixed)
   - head_dim_vo = 128 (fixed)
   - batch >= 1 (any)
   - heads >= 1 (any)

 BATCH SIZE COVERAGE: Testing batch = 1, 2, 4, 8
*/

TEST(XE4_FMHA_FWD, batch1_heads1_seq128_kv512) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 1;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 512;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, batch2_heads1_seq128_kv512) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 2;
  constexpr int heads = 1;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 512;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, batch4_heads1_seq128_kv512) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 4;
  constexpr int heads = 1;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 512;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, batch8_heads1_seq128_kv512) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 8;
  constexpr int heads = 1;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 512;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

/*
=============================================================================
 NUM_HEADS COVERAGE: Testing heads = 1, 2, 4, 8, 16, 32
=============================================================================
*/

TEST(XE4_FMHA_FWD, DISABLED_batch1_heads2_seq256_kv512) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 2;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 512;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, batch1_heads4_seq256_kv512) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 512;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, batch1_heads8_seq256_kv512) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 8;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 512;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, batch1_heads16_seq256_kv512) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 16;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 512;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, batch1_heads32_seq256_kv512) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 32;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 512;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

/*
=============================================================================
 SEQ_Q COVERAGE: Testing seq_q = 128, 256, 512, 1024, 2048 (multiples of 128)
=============================================================================
*/

TEST(XE4_FMHA_FWD, batch1_heads4_seq128_kv1024) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 1024;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, batch1_heads4_seq256_kv1024) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 1024;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, batch1_heads4_seq512_kv1024) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 512;
  constexpr int seq_kv = 1024;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, DISABLED_batch1_heads4_seq1024_kv1024) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 1024;
  constexpr int seq_kv = 1024;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// DISABLED: Heavy test - workload: 1x4x2048x2048 = 16.8M
TEST(XE4_FMHA_FWD, DISABLED_batch1_heads4_seq2048_kv2048) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 2048;
  constexpr int seq_kv = 2048;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

/*
=============================================================================
 SEQ_KV COVERAGE: Testing seq_kv = 512, 1024, 1536, 2048, 4096, 8192 (multiples of 512)
=============================================================================
*/

TEST(XE4_FMHA_FWD, DISABLED_batch1_heads4_seq256_kv1536) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 1536;  // 512 * 3
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, DISABLED_batch1_heads4_seq256_kv2048) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 2048;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, DISABLED_batch1_heads4_seq256_kv4096) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 4096;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// DISABLED: Heavy test - workload: 1x4x256x8192 = 8.4M (long KV)
TEST(XE4_FMHA_FWD, DISABLED_batch1_heads4_seq256_kv8192) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 8192;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

/*
=============================================================================
 ASYMMETRIC SEQUENCE LENGTH COVERAGE (KV cache scenarios)
=============================================================================
*/

TEST(XE4_FMHA_FWD, DISABLED_asymmetric_seq128_kv2048) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 8;
  constexpr int seq_q = 128;   // Small query (e.g., single token decode)
  constexpr int seq_kv = 2048; // Large KV cache
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, DISABLED_asymmetric_seq256_kv4096) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 2;
  constexpr int heads = 8;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 4096;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// DISABLED: Heaviest test - workload: 1x16x512x8192 = 67.1M
TEST(XE4_FMHA_FWD, DISABLED_asymmetric_seq512_kv8192) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 16;
  constexpr int seq_q = 512;
  constexpr int seq_kv = 8192;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

/*
=============================================================================
 COMBINED BATCH + HEADS COVERAGE (stress tests)
=============================================================================
*/

TEST(XE4_FMHA_FWD, DISABLED_DISABLED_batch2_heads8_seq512_kv1024) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 2;
  constexpr int heads = 8;
  constexpr int seq_q = 512;
  constexpr int seq_kv = 1024;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, DISABLED_batch4_heads8_seq256_kv1024) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 4;
  constexpr int heads = 8;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 1024;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// DISABLED: Heavy test - workload: 2x16x512x2048 = 33.6M
TEST(XE4_FMHA_FWD, DISABLED_batch2_heads16_seq512_kv2048) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 2;
  constexpr int heads = 16;
  constexpr int seq_q = 512;
  constexpr int seq_kv = 2048;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// DISABLED: Heavy test - workload: 4x4x1024x2048 = 33.6M
TEST(XE4_FMHA_FWD, DISABLED_batch4_heads4_seq1024_kv2048) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 4;
  constexpr int heads = 4;
  constexpr int seq_q = 1024;
  constexpr int seq_kv = 2048;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

/*
=============================================================================
 EDGE CASES
=============================================================================
*/

// Minimum valid configuration
TEST(XE4_FMHA_FWD, minimum_valid_config) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 1;
  constexpr int seq_q = 128;   // Minimum: must be >= M (128)
  constexpr int seq_kv = 512;  // Minimum: must be >= K (512)
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// Square attention (seq_q == seq_kv)
TEST(XE4_FMHA_FWD, DISABLED_square_attention_1024) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 2;
  constexpr int heads = 8;
  constexpr int seq_q = 1024;
  constexpr int seq_kv = 1024;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// Large batch with single head
TEST(XE4_FMHA_FWD, DISABLED_large_batch_single_head) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 8;
  constexpr int heads = 1;
  constexpr int seq_q = 512;
  constexpr int seq_kv = 1024;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// Single batch with many heads (like GPT-style)
// DISABLED: Heavy test - workload: 1x32x1024x1024 = 33.6M
TEST(XE4_FMHA_FWD, DISABLED_gpt_style_32heads) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 32;
  constexpr int seq_q = 1024;
  constexpr int seq_kv = 1024;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// Large KV cache scenario (inference decode)
// DISABLED: Heavy test - workload: 4x8x128x8192 = 33.6M
TEST(XE4_FMHA_FWD, DISABLED_inference_decode_long_context) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 4;
  constexpr int heads = 8;
  constexpr int seq_q = 128;    // Single token decode
  constexpr int seq_kv = 8192;  // Long context
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// BF16 with K=128: seq_kv must be divisible by 128
TEST(XE4_FMHA_FWD, DISABLED_bf16_tile128_seq128_kv128) {
  using TileShape = cute::Shape<_128, _128, _128, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 128;  // Minimum: must be >= K (128)
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_BF16BF16BF16BF16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// BF16 with K=256: seq_kv must be divisible by 256
TEST(XE4_FMHA_FWD, DISABLED_bf16_tile256_seq128_kv256) {
  using TileShape = cute::Shape<_128, _128, _256, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 256;  // Minimum: must be >= K (256)
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_BF16BF16BF16BF16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// BF16 with K=512: seq_kv must be divisible by 512
TEST(XE4_FMHA_FWD, DISABLED_bf16_tile512_seq128_kv512) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 512;  // Minimum: must be >= K (512)
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_BF16BF16BF16BF16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// FP16 with K=256: supported (fp16 gtp_tred_max supports N=16)
TEST(XE4_FMHA_FWD, fp16_tile256_seq128_kv256) {
  using TileShape = cute::Shape<_128, _128, _256, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 256;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

/*
=============================================================================
 BF16 INPUT + FP32 INTERMEDIATE (ElementS = float)
=============================================================================
*/

// BF16 with FP32 intermediate, K=512
// DISABLED: BF16 input + FP32 softmax intermediate causes JIT compilation failure on Xe4
TEST(XE4_FMHA_FWD, DISABLED_bf16_fp32_intermediate_tile512_seq128_kv512) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 512;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_BF16BF16FP32BF16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// BF16 with FP32 intermediate, K=256
TEST(XE4_FMHA_FWD, DISABLED_bf16_fp32_intermediate_tile256_seq128_kv256) {
  using TileShape = cute::Shape<_128, _128, _256, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 256;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_BF16BF16FP32BF16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

// BF16 with FP32 intermediate, larger sequence
TEST(XE4_FMHA_FWD, DISABLED_bf16_fp32_intermediate_seq256_kv1024) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 4;
  constexpr int seq_q = 256;
  constexpr int seq_kv = 1024;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_BF16BF16FP32BF16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}



} // namespace flash_attention_v3
} // namespace test
} // namespace cutlass
