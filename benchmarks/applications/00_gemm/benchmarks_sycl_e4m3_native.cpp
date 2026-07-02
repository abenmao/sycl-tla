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
#include "benchmarks_sycl_types_lowp_float.hpp"
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_256_256_32);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_512_256_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3FP32_SplitK2_RRR_TileShape_512_256_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3FP32_SplitK4_RRR_TileShape_512_256_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_8_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_16_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_64_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_256_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_4_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_8_256_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_8_128_64_sg8x16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemm_W8A8_E4M3E4M3FP16MMA_RRR_TileShape_256_256_32);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3BF16_RRR_TileShape_512_256_128);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_SplitK4_RRR_TileShape_512_256_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_SplitK2_RRR_TileShape_512_256_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_512_256_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_16_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_64_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_256_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_8_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_4_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_8_256_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_8_128_64_sg8x16);

// ---- FP8 (E4M3) -> E4M3 (tuned memory-bound configs) ----
using CriGemm_E4M3E4M3FP32_TileShape_16_128_128 = Shape<_16, _128, _128>;
using CriGemm_E4M3E4M3FP32_Tile_16_128_128 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>>, Layout<CriGemm_E4M3E4M3FP32_TileShape_16_128_128>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE4M3E4M3E4M3_RRR_TileShape_16_128_128 =
    Gemm_Bench_SrcOut<cutlass::float_e4m3_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_E4M3E4M3FP32_TileShape_16_128_128, CriGemm_E4M3E4M3FP32_Tile_16_128_128>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_16_128_128);
using CriGemm_E4M3E4M3FP32_TileShape_64_128_128 = Shape<_64, _128, _128>;
using CriGemm_E4M3E4M3FP32_Tile_64_128_128 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>>, Layout<CriGemm_E4M3E4M3FP32_TileShape_64_128_128>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE4M3E4M3E4M3_RRR_TileShape_64_128_128 =
    Gemm_Bench_SrcOut<cutlass::float_e4m3_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_E4M3E4M3FP32_TileShape_64_128_128, CriGemm_E4M3E4M3FP32_Tile_64_128_128>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_64_128_128);
using CriGemm_E4M3E4M3FP32_TileShape_192_128_64 = Shape<_192, _128, _64>;
using CriGemm_E4M3E4M3FP32_Tile_192_128_64 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>>, Layout<CriGemm_E4M3E4M3FP32_TileShape_192_128_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE4M3E4M3E4M3_RRR_TileShape_192_128_64 =
    Gemm_Bench_SrcOut<cutlass::float_e4m3_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_E4M3E4M3FP32_TileShape_192_128_64, CriGemm_E4M3E4M3FP32_Tile_192_128_64>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_192_128_64);
using CriGemm_E4M3E4M3FP32_TileShape_8_128_256 = Shape<_8, _128, _256>;
using CriGemm_E4M3E4M3FP32_Tile_8_128_256 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>>, Layout<CriGemm_E4M3E4M3FP32_TileShape_8_128_256>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE4M3E4M3E4M3_RRR_TileShape_8_128_256 =
    Gemm_Bench_SrcOut<cutlass::float_e4m3_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_E4M3E4M3FP32_TileShape_8_128_256, CriGemm_E4M3E4M3FP32_Tile_8_128_256>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_8_128_256);
using CriGemm_E4M3E4M3FP32_Tile_8_256_64_sg8x16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>>, Layout<CriGemm_E4M3E4M3FP32_TileShape_8_256_64>, Layout<Shape<_1, _16, _1>, Stride<_16, _1, _0>>>::TiledMMA;
using CriGemmE4M3E4M3E4M3_RRR_TileShape_8_256_64_sg8x16 =
    Gemm_Bench_SrcOut<cutlass::float_e4m3_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_E4M3E4M3FP32_TileShape_8_256_64, CriGemm_E4M3E4M3FP32_Tile_8_256_64_sg8x16>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_8_256_64_sg8x16);
using CriGemm_E4M3E4M3FP32_TileShape_8_128_512 = Shape<_8, _128, _512>;

// ---- FP8 (E4M3) -> E4M3 (tuned compute-bound configs) ----
using CriGemm_E4M3E4M3FP32_TileShape_128_256_128 = Shape<_128, _256, _128>;
using CriGemm_E4M3E4M3FP32_Tile_128_256_128 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>>, Layout<CriGemm_E4M3E4M3FP32_TileShape_128_256_128>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE4M3E4M3E4M3_RRR_TileShape_128_256_128 = Gemm_Bench_SrcOut<cutlass::float_e4m3_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_E4M3E4M3FP32_TileShape_128_256_128, CriGemm_E4M3E4M3FP32_Tile_128_256_128>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_128_256_128);
using CriGemm_E4M3E4M3FP32_TileShape_256_256_128 = Shape<_256, _256, _128>;
using CriGemm_E4M3E4M3FP32_Tile_256_256_128 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>>, Layout<CriGemm_E4M3E4M3FP32_TileShape_256_256_128>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE4M3E4M3E4M3_RRR_TileShape_256_256_128 = Gemm_Bench_SrcOut<cutlass::float_e4m3_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_E4M3E4M3FP32_TileShape_256_256_128, CriGemm_E4M3E4M3FP32_Tile_256_256_128>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_256_256_128);
using CriGemm_E4M3E4M3FP32_Tile_256_128_64_sg64x16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>>, Layout<CriGemm_E4M3E4M3FP32_TileShape_256_128_64>, Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>>::TiledMMA;
using CriGemmE4M3E4M3E4M3_RRR_TileShape_256_128_64_sg64x16 = Gemm_Bench_SrcOut<cutlass::float_e4m3_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_E4M3E4M3FP32_TileShape_256_128_64, CriGemm_E4M3E4M3FP32_Tile_256_128_64_sg64x16>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_256_128_64_sg64x16);
using CriGemm_E4M3E4M3FP32_TileShape_256_256_64 = Shape<_256, _256, _64>;
using CriGemm_E4M3E4M3FP32_Tile_256_256_64 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>>, Layout<CriGemm_E4M3E4M3FP32_TileShape_256_256_64>, Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmE4M3E4M3E4M3_RRR_TileShape_256_256_64 = Gemm_Bench_SrcOut<cutlass::float_e4m3_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_E4M3E4M3FP32_TileShape_256_256_64, CriGemm_E4M3E4M3FP32_Tile_256_256_64>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_256_256_64);
using CriGemm_E4M3E4M3FP32_Tile_8_128_512 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::float_e4m3_t>>, Layout<CriGemm_E4M3E4M3FP32_TileShape_8_128_512>, Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>>::TiledMMA;
using CriGemmE4M3E4M3E4M3_SplitK2_RRR_TileShape_8_128_512 = Gemm_Bench_SrcOut<cutlass::float_e4m3_t, cutlass::layout::RowMajor, Scheduler::Gemm, CriGemm_E4M3E4M3FP32_TileShape_8_128_512, CriGemm_E4M3E4M3FP32_Tile_8_128_512>;
CUTLASS_CREATE_GEMM_BENCHMARK(CriGemmE4M3E4M3E4M3_SplitK2_RRR_TileShape_8_128_512);
#endif
void register_gemm_benchmarks_e4m3_native() {
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3FP32_SplitK2_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3FP32_SplitK4_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_8_128_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_16_128_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_64_128_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_256_128_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_4_128_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_8_256_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3FP32_RRR_TileShape_8_128_64_sg8x16);
  CUTLASS_BENCHMARK(CriGemm_W8A8_E4M3E4M3FP16MMA_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3BF16_RRR_TileShape_512_256_128);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_SplitK4_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_SplitK2_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_16_128_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_64_128_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_256_128_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_8_128_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_4_128_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_8_256_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_8_128_64_sg8x16);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_16_128_128);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_64_128_128);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_192_128_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_8_128_256);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_8_256_64_sg8x16);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_128_256_128);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_256_256_128);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_256_128_64_sg64x16);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_RRR_TileShape_256_256_64);
  CUTLASS_BENCHMARK(CriGemmE4M3E4M3E4M3_SplitK2_RRR_TileShape_8_128_512);
#endif
}
