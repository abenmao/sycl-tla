/***************************************************************************************************
 * Copyright (C) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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
    \brief Flash Attention V2 Prefill for Intel BMG

    This example constructs and executes a Flash Attention Prefill kernel on Intel BMG. The
    definition of the GEMM, options etc for this example are defined in the associated
    bmg_flash_attn_runner.hpp header file.

    See https://arxiv.org/pdf/2307.08691 for details of Flash Attention V2 algorithm

    To run this example:
      $ ./examples/sycl/06_bmg_flash_attention/06_xe_fmha_fwd --seq_len_qo=512
        --seq_len_kv=512 --head_size_vo=128 --head_size_qk=128

    To build & run this example (from your build dir):

      $ ninja 06_xe_fmha_fwd
      $ ./examples/sycl/06_bmg_flash_attention/06_xe_fmha_fwd

    Call with `--help` for information about available options
*/

#include "xe_fmha_fwd_runner.hpp"

int main(int argc, const char **argv) {
  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  // Define the work-group tile shape depending on the head-size of the second matmul

#ifdef PREFILL
#if HEAD_DIM == 16
  /* Tiny config for testing */
  using ShapeQK = Shape<_1, _16, _16>;       // (q,k,d)
  using ShapePV = Shape<_1, _16, _16>;       // (q,v,k)
  using ShapeOut = Shape<_1, _16>;           // (q,v)
  using SubgroupLayoutQK = Layout<Shape<_1, _1, _1>>;

#elif HEAD_DIM == 64
#if defined(IS_INT8) && defined(INT8_KV128)
  /* int8 has 1-byte inputs -> register headroom for a larger KV tile.
     KV tile 64->128 halves the number of fp32 softmax reductions and gives the
     fast int8 DPAS more work per tile to hide the (now-exposed) softmax latency. */
  using ShapeQK = Shape<_128, _128, _32>;
  using ShapePV = Shape<_128, _32, _128>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
#else
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
#endif

#elif HEAD_DIM == 96
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;

#elif HEAD_DIM == 128
#if defined(IS_INT8) && defined(INT8_KV128)
  /* int8 has 1-byte inputs -> register headroom for a larger KV tile.
     KV tile 32->64 halves the number of fp32 softmax reductions and gives the
     fast int8 DPAS more work per tile to hide the (now-exposed) softmax latency. */
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#else
  using ShapeQK = Shape<_256, _32, _32>;
  using ShapePV = Shape<_256, _32, _32>;
  using ShapeOut = Shape<_256, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#endif

#elif HEAD_DIM == 192
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;

#endif
#elif defined(DECODE)

#if PERSISTENT
#define NUM_SG _16
#define KV_TILE_SIZE _256
#else
#define NUM_SG _4
#define KV_TILE_SIZE _256
#endif

#if HEAD_DIM == 16
  /* Tiny config for testing */
  using ShapeQK = Shape<_1, _16, _16>;       // (q,k,d)
  using ShapePV = Shape<_1, _16, _16>;       // (q,v,k)
  using ShapeOut = Shape<_1, _16>;           // (q,v)
  using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;

#elif HEAD_DIM == 64
#ifdef Q_PACKED_DECODE
    // GQA q_packed decode: pack head_group_q query heads (sharing one KV head)
    // as the rows of the QK tile. First dim is the packed query-head count, NOT
    // the (=1) decode query sequence length. This turns the per-head GEMV into a
    // small GEMM and reads each KV head once per WG instead of once per query head.
    // Tile rows (16) are >= head_group_q (e.g. 10 for 80/8); the block-2D copies
    // predicate rows beyond the real head_group_q (built from the runtime Q/O view).
    using ShapeQK = Shape<_16, _64, _64>;
    using ShapePV = Shape<_16, _32, _64>;
    using ShapeOut = Shape<_16, _64>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
#else
    using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
    using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
    using ShapeOut = Shape<_1, _64>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
#endif

#elif HEAD_DIM == 96
    using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
    using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
    using ShapeOut = Shape<_1, _96>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;

#elif HEAD_DIM == 128
#ifdef Q_PACKED_DECODE
    // GQA q_packed decode: pack head_group_q query heads (sharing one KV head)
    // as the rows of the QK tile. First dim is the packed query-head count, NOT
    // the (=1) decode query sequence length. This turns the per-head GEMV into a
    // small GEMM and reads each KV head once per WG instead of once per query head.
    // Tile rows (16) are >= head_group_q (e.g. 10 for 80/8); the block-2D copies
    // predicate rows beyond the real head_group_q (built from the runtime Q/O view).
    using ShapeQK = Shape<_16, _64, _64>;
    using ShapePV = Shape<_16, _32, _64>;
    using ShapeOut = Shape<_16, _128>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
#else
    using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
    using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
    using ShapeOut = Shape<_1, _128>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
#endif

#elif HEAD_DIM == 192
    using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
    using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
    using ShapeOut = Shape<_1, _192>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
#endif
#else
#error Either DECODE or PREFILL should be defined.
#endif

#ifdef DECODE
  constexpr int PipelineStages = 1;
#else
#ifdef ENABLE_SPLIT_BOUNDARY
  constexpr int PipelineStages = 1;
#else
  constexpr int PipelineStages = 2;
#endif
#endif
#ifdef IS_FLOAT_E5M2
  using ElementQ = cutlass::float_e5m2_t;
  using ElementK = cutlass::float_e5m2_t;
  using ElementV = cutlass::float_e5m2_t;
#elif defined(IS_FLOAT_E4M3)
  using ElementQ = cutlass::float_e4m3_t;
  using ElementK = cutlass::float_e4m3_t;
  using ElementV = cutlass::float_e4m3_t;
#elif defined(IS_INT8)
  using ElementQ = int8_t;
  using ElementK = int8_t;
  using ElementV = int8_t;
#else
  using ElementQ = bfloat16_t;
  using ElementK = bfloat16_t;
  using ElementV = bfloat16_t;
#endif

#if PERSISTENT
  return FMHAConfig<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, /*persistent=*/true, ElementQ, ElementK, ElementV>::run(options);
#else
  return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,  /*persistent=*/false, ElementQ, ElementK, ElementV>::run(options)
  : FMHAConfig<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,  /*persistent=*/false, ElementQ, ElementK, ElementV>::run(options);
#endif
}
