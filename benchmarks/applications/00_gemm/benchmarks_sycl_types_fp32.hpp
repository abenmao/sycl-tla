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

#pragma once
#include "gemm_configuration_sycl.hpp"
#include "dual_gemm_benchmark_runner.hpp"
#include "cutlass/epilogue/thread/activation.h"

using Scheduler = cutlass::gemm::device::Scheduler;
template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_FP32FP32FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using CriGemm_FP32FP32FP32_TileShape_512_256_16 = Shape<_512, _256, _16>;
using CriGemm_FP32FP32FP32_Tile_512_256_16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_FP32FP32FP32_TileShape_512_256_16>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmFP32FP32FP32_RRR_TileShape_512_256_16 = Gemm_Bench_FP32FP32FP32_RRR<CriGemm_FP32FP32FP32_TileShape_512_256_16, CriGemm_FP32FP32FP32_Tile_512_256_16, void, void>;

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_FP32FP32FP32_RRR_SplitK = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::GemmSplitK, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

template <int Splits>
struct CriGemmFP32FP32FP32_SplitK_RRR_TileShape_512_256_16 :
    Gemm_Bench_FP32FP32FP32_RRR_SplitK<
        CriGemm_FP32FP32FP32_TileShape_512_256_16,
        CriGemm_FP32FP32FP32_Tile_512_256_16,
        void,
        void> {
  using Base = Gemm_Bench_FP32FP32FP32_RRR_SplitK<
      CriGemm_FP32FP32FP32_TileShape_512_256_16,
      CriGemm_FP32FP32FP32_Tile_512_256_16,
      void,
      void>;
  using GemmKernel = typename Base::GemmKernel;

  constexpr static typename GemmKernel::Arguments defaultArguments() {
    using StreamKMode =
        cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams::DecompositionMode;
    typename GemmKernel::Arguments arguments{};
    arguments.scheduler = {Splits, StreamKMode::SplitK};
    return arguments;
  }
};

using CriGemmFP32FP32FP32_SplitK2_RRR_TileShape_512_256_16 =
  CriGemmFP32FP32FP32_SplitK_RRR_TileShape_512_256_16<2>;

using CriGemmFP32FP32FP32_SplitK4_RRR_TileShape_512_256_16 =
    CriGemmFP32FP32FP32_SplitK_RRR_TileShape_512_256_16<4>;

using CriGemm_FP32FP32FP32_TileShape_8_128_16 = Shape<_8, _128, _16>;
using CriGemm_FP32FP32FP32_Tile_8_128_16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_FP32FP32FP32_TileShape_8_128_16>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmFP32FP32FP32_RRR_TileShape_8_128_16 = Gemm_Bench_FP32FP32FP32_RRR<CriGemm_FP32FP32FP32_TileShape_8_128_16, CriGemm_FP32FP32FP32_Tile_8_128_16, void, void>;

using CriGemm_FP32FP32FP32_TileShape_16_128_16 = Shape<_16, _128, _16>;
using CriGemm_FP32FP32FP32_Tile_16_128_16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_FP32FP32FP32_TileShape_16_128_16>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmFP32FP32FP32_RRR_TileShape_16_128_16 = Gemm_Bench_FP32FP32FP32_RRR<CriGemm_FP32FP32FP32_TileShape_16_128_16, CriGemm_FP32FP32FP32_Tile_16_128_16, void, void>;

using CriGemm_FP32FP32FP32_TileShape_64_128_16 = Shape<_64, _128, _16>;
using CriGemm_FP32FP32FP32_Tile_64_128_16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_FP32FP32FP32_TileShape_64_128_16>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmFP32FP32FP32_RRR_TileShape_64_128_16 = Gemm_Bench_FP32FP32FP32_RRR<CriGemm_FP32FP32FP32_TileShape_64_128_16, CriGemm_FP32FP32FP32_Tile_64_128_16, void, void>;

using CriGemm_FP32FP32FP32_TileShape_256_128_16 = Shape<_256, _128, _16>;
using CriGemm_FP32FP32FP32_Tile_256_128_16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_FP32FP32FP32_TileShape_256_128_16>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmFP32FP32FP32_RRR_TileShape_256_128_16 = Gemm_Bench_FP32FP32FP32_RRR<CriGemm_FP32FP32FP32_TileShape_256_128_16, CriGemm_FP32FP32FP32_Tile_256_128_16, void, void>;

using CriGemm_FP32FP32FP32_TileShape_4_128_16 = Shape<_4, _128, _16>;
using CriGemm_FP32FP32FP32_Tile_4_128_16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<4, float, cutlass::tfloat32_t>>, Layout<CriGemm_FP32FP32FP32_TileShape_4_128_16>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using CriGemmFP32FP32FP32_RRR_TileShape_4_128_16 = Gemm_Bench_FP32FP32FP32_RRR<CriGemm_FP32FP32FP32_TileShape_4_128_16, CriGemm_FP32FP32FP32_Tile_4_128_16, void, void>;

using CriGemm_FP32FP32FP32_TileShape_8_256_16 = Shape<_8, _256, _16>;
using CriGemm_FP32FP32FP32_Tile_8_256_16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_FP32FP32FP32_TileShape_8_256_16>, Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>>::TiledMMA;
using CriGemmFP32FP32FP32_RRR_TileShape_8_256_16 = Gemm_Bench_FP32FP32FP32_RRR<CriGemm_FP32FP32FP32_TileShape_8_256_16, CriGemm_FP32FP32FP32_Tile_8_256_16, void, void>;

using CriGemm_FP32FP32FP32_Tile_8_128_16_sg8x16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::tfloat32_t>>, Layout<CriGemm_FP32FP32FP32_TileShape_8_128_16>, Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>>::TiledMMA;
using CriGemmFP32FP32FP32_RRR_TileShape_8_128_16_sg8x16 = Gemm_Bench_FP32FP32FP32_RRR<CriGemm_FP32FP32FP32_TileShape_8_128_16, CriGemm_FP32FP32FP32_Tile_8_128_16_sg8x16, void, void>;
