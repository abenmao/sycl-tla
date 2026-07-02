/****************************************************************************************************
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
 ***************************************************************************************************/

#include "benchmark_runner.hpp"
#include "benchmarks_sycl_types_tf32.hpp"

CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_512_256_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32FP32_SplitK2_RRR_TileShape_512_256_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32FP32_SplitK4_RRR_TileShape_512_256_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_8_128_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_16_128_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_64_128_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_256_128_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_4_128_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_8_256_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_8_128_16_sg8x16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_SplitK4_RRR_TileShape_512_256_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_SplitK2_RRR_TileShape_512_256_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_512_256_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_16_128_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_64_128_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_256_128_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_8_128_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_4_128_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_8_256_16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_8_128_16_sg8x16);

// ---- TF32 -> TF32 (tuned memory-bound configs) ----
using CriGemm_TF32TF32FP32_TileShape_16_128_32 = Shape<_16, _128, _32>;
using CriGemm_TF32TF32FP32_Tile_16_128_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_TF32TF32FP32_TileShape_16_128_32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmTF32TF32TF32_RRR_TileShape_16_128_32 =
    Gemm_Bench_SrcOut<cutlass::tfloat32_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_TF32TF32FP32_TileShape_16_128_32, CriGemm_TF32TF32FP32_Tile_16_128_32>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_16_128_32);
using CriGemm_TF32TF32FP32_TileShape_64_128_32 = Shape<_64, _128, _32>;
using CriGemm_TF32TF32FP32_Tile_64_128_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_TF32TF32FP32_TileShape_64_128_32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmTF32TF32TF32_RRR_TileShape_64_128_32 =
    Gemm_Bench_SrcOut<cutlass::tfloat32_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_TF32TF32FP32_TileShape_64_128_32, CriGemm_TF32TF32FP32_Tile_64_128_32>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_64_128_32);
using CriGemm_TF32TF32FP32_TileShape_192_128_16 = Shape<_192, _128, _16>;
using CriGemm_TF32TF32FP32_Tile_192_128_16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_TF32TF32FP32_TileShape_192_128_16>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmTF32TF32TF32_RRR_TileShape_192_128_16 =
    Gemm_Bench_SrcOut<cutlass::tfloat32_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_TF32TF32FP32_TileShape_192_128_16, CriGemm_TF32TF32FP32_Tile_192_128_16>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_192_128_16);
using CriGemm_TF32TF32FP32_TileShape_8_128_64 = Shape<_8, _128, _64>;
using CriGemm_TF32TF32FP32_Tile_8_128_64 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_TF32TF32FP32_TileShape_8_128_64>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmTF32TF32TF32_RRR_TileShape_8_128_64 =
    Gemm_Bench_SrcOut<cutlass::tfloat32_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_TF32TF32FP32_TileShape_8_128_64, CriGemm_TF32TF32FP32_Tile_8_128_64>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_8_128_64);
using CriGemm_TF32TF32FP32_Tile_8_256_16_sg8x16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_TF32TF32FP32_TileShape_8_256_16>, Layout<Shape<_1, _16, _1>, Stride<_16, _1, _0>>>::TiledMMA;
using CriGemmTF32TF32TF32_RRR_TileShape_8_256_16_sg8x16 =
    Gemm_Bench_SrcOut<cutlass::tfloat32_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_TF32TF32FP32_TileShape_8_256_16, CriGemm_TF32TF32FP32_Tile_8_256_16_sg8x16>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_8_256_16_sg8x16);
using CriGemm_TF32TF32FP32_TileShape_8_128_128 = Shape<_8, _128, _128>;

// ---- TF32 -> TF32 (tuned compute-bound configs) ----
using CriGemm_TF32TF32FP32_TileShape_128_256_32 = Shape<_128, _256, _32>;
using CriGemm_TF32TF32FP32_Tile_128_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_TF32TF32FP32_TileShape_128_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmTF32TF32TF32_RRR_TileShape_128_256_32 = Gemm_Bench_SrcOut<cutlass::tfloat32_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_TF32TF32FP32_TileShape_128_256_32, CriGemm_TF32TF32FP32_Tile_128_256_32>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_128_256_32);
using CriGemm_TF32TF32FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using CriGemm_TF32TF32FP32_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_TF32TF32FP32_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmTF32TF32TF32_RRR_TileShape_256_256_32 = Gemm_Bench_SrcOut<cutlass::tfloat32_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_TF32TF32FP32_TileShape_256_256_32, CriGemm_TF32TF32FP32_Tile_256_256_32>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_256_256_32);
using CriGemm_TF32TF32FP32_Tile_256_128_16_sg64x16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_TF32TF32FP32_TileShape_256_128_16>, Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>>::TiledMMA;
using CriGemmTF32TF32TF32_RRR_TileShape_256_128_16_sg64x16 = Gemm_Bench_SrcOut<cutlass::tfloat32_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_TF32TF32FP32_TileShape_256_128_16, CriGemm_TF32TF32FP32_Tile_256_128_16_sg64x16>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_256_128_16_sg64x16);
using CriGemm_TF32TF32FP32_TileShape_256_256_16 = Shape<_256, _256, _16>;
using CriGemm_TF32TF32FP32_Tile_256_256_16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_TF32TF32FP32_TileShape_256_256_16>, Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmTF32TF32TF32_RRR_TileShape_256_256_16 = Gemm_Bench_SrcOut<cutlass::tfloat32_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_TF32TF32FP32_TileShape_256_256_16, CriGemm_TF32TF32FP32_Tile_256_256_16>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_256_256_16);
using CriGemm_TF32TF32FP32_Tile_8_128_128 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_TF32TF32FP32_TileShape_8_128_128>, Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>>::TiledMMA;
using CriGemmTF32TF32TF32_SplitK2_RRR_TileShape_8_128_128 = Gemm_Bench_SrcOut<cutlass::tfloat32_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_TF32TF32FP32_TileShape_8_128_128, CriGemm_TF32TF32FP32_Tile_8_128_128>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmTF32TF32TF32_SplitK2_RRR_TileShape_8_128_128);

void register_gemm_benchmarks_tf32() {
  CUTLASS_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_512_256_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32FP32_SplitK2_RRR_TileShape_512_256_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32FP32_SplitK4_RRR_TileShape_512_256_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_8_128_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_16_128_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_64_128_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_256_128_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_4_128_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_8_256_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32FP32_RRR_TileShape_8_128_16_sg8x16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_SplitK4_RRR_TileShape_512_256_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_SplitK2_RRR_TileShape_512_256_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_512_256_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_16_128_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_64_128_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_256_128_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_8_128_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_4_128_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_8_256_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_8_128_16_sg8x16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_16_128_32);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_64_128_32);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_192_128_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_8_128_64);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_8_256_16_sg8x16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_128_256_32);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_256_128_16_sg64x16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_RRR_TileShape_256_256_16);
  CUTLASS_BENCHMARK(CriGemmTF32TF32TF32_SplitK2_RRR_TileShape_8_128_128);
}
