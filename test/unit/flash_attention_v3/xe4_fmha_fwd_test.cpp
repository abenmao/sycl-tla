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

TEST(XE4_FMHA_FWD, smoke_fp16) {
  // Xe4 fwd kernel tile is fixed to 128x128x512x128 (M,N,K,cluster) for this smoke test.
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  // Problem dims expressed via constexprs to keep a single source of truth.
  constexpr int batch = 1;
  constexpr int heads = 1;
  constexpr int seq_q = 64;
  constexpr int seq_kv = 64;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

#if 0 // Disable tests - under development
TEST(XE4_FMHA_FWD, smoke_fp16_fp32acc) {
  using TileShape = cute::Shape<_128, _128, _512, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 1;
  constexpr int seq_q = 64;
  constexpr int seq_kv = 64;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP32FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, tile64x128x128x128_fp32acc_fp16out) {
  using TileShape = cute::Shape<_64, _128, _128, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 1;
  constexpr int seq_q = 64;
  constexpr int seq_kv = 128;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP32FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, tile128x128x128x128_fp32acc_fp16out) {
  using TileShape = cute::Shape<_128, _128, _128, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 1;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 128;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP32FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, tile128x128x256x128_fp32acc_fp16out) {
  using TileShape = cute::Shape<_128, _128, _256, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 1;
  constexpr int seq_q = 128;
  constexpr int seq_kv = 256;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP32FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, tile64x128x128x128_fp16acc_fp16out) {
  using TileShape = cute::Shape<_64, _128, _128, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 1;
  constexpr int seq_q = 64;
  constexpr int seq_kv = 128;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}
#endif

TEST(XE4_FMHA_FWD, tile64x128x256x128_fp16acc_fp16out) {
  using TileShape = cute::Shape<_64, _128, _256, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 1;
  constexpr int seq_q = 64;
  constexpr int seq_kv = 256;
  constexpr int head_dim_qk = 128;
  constexpr int head_dim_vo = 128;

  typename KernelFactory::ProblemShape problem_shape =
      cute::make_tuple(batch, heads, seq_q, seq_kv, head_dim_qk, head_dim_vo);
  const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_qk));

  bool passed = KernelFactory::template run<ProblemConfig_FP16FP16FP16FP16>(problem_shape, softmax_scale);
  EXPECT_TRUE(passed);
}

TEST(XE4_FMHA_FWD, tile128x128x256x128_fp16acc_fp16out) {
  using TileShape = cute::Shape<_128, _128, _256, _128>;
  using KernelFactory = Fmha3KernelFactory<TileShape>;

  constexpr int batch = 1;
  constexpr int heads = 1;
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

} // namespace flash_attention_v3
} // namespace test
} // namespace cutlass
