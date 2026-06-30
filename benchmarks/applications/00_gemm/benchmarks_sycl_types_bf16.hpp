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
using Gemm_Bench_BF16FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using BmgGemm_BF16FP32_TileShape_512_256_32 = Shape<_512, _256, _32>;
using BmgGemm_BF16FP32_Tile_512_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_512_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_512_256_32 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_512_256_32, BmgGemm_BF16FP32_Tile_512_256_32, void, void>;

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_BF16BF16BF16_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    cutlass::bfloat16_t,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t, float>>;

using BmgGemm_BF16BF16BF16_TileShape_512_256_64 = Shape<_512, _256, _64>;
using BmgGemm_BF16BF16BF16_Tile_512_256_64 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16BF16BF16_TileShape_512_256_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16BF16_RRR_TileShape_512_256_64 = Gemm_Bench_BF16BF16BF16_RRR<BmgGemm_BF16BF16BF16_TileShape_512_256_64, BmgGemm_BF16BF16BF16_Tile_512_256_64, void, void>;

using BmgGemm_BF16BF16BF16_TileShape_128_256_64 = Shape<_128, _256, _64>;
using BmgGemm_BF16BF16BF16_Tile_128_256_64 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16BF16BF16_TileShape_128_256_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16BF16_RRR_TileShape_128_256_64 = Gemm_Bench_BF16BF16BF16_RRR<BmgGemm_BF16BF16BF16_TileShape_128_256_64, BmgGemm_BF16BF16BF16_Tile_128_256_64, void, void>;

using BmgGemm_BF16BF16BF16_TileShape_256_256_64 = Shape<_256, _256, _64>;
using BmgGemm_BF16BF16BF16_Tile_256_256_64 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16BF16BF16_TileShape_256_256_64>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16BF16_RRR_TileShape_256_256_64 = Gemm_Bench_BF16BF16BF16_RRR<BmgGemm_BF16BF16BF16_TileShape_256_256_64, BmgGemm_BF16BF16BF16_Tile_256_256_64, void, void>;

// StreamK variant matching example 03_bmg_gemm_streamk (TileShape 256x256x32,
// KernelXeCooperative + StreamKScheduler).
template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_BF16FP32_RRR_StreamK = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::GemmStreamK, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

using BmgGemm_BF16FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using BmgGemm_BF16FP32_Tile_256_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_256_256_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_StreamK_TileShape_256_256_32 = Gemm_Bench_BF16FP32_RRR_StreamK<BmgGemm_BF16FP32_TileShape_256_256_32, BmgGemm_BF16FP32_Tile_256_256_32, void, void>;

template <
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_BF16FP32_RRR_SplitK = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::GemmSplitK, Tiler,
    GmemTiledCopyA, GmemTiledCopyB>;

template <int Splits>
struct BmgGemmBF16BF16FP32_SplitK_RRR_TileShape_512_256_32 :
    Gemm_Bench_BF16FP32_RRR_SplitK<
        BmgGemm_BF16FP32_TileShape_512_256_32,
        BmgGemm_BF16FP32_Tile_512_256_32,
        void,
        void> {
  using Base = Gemm_Bench_BF16FP32_RRR_SplitK<
      BmgGemm_BF16FP32_TileShape_512_256_32,
      BmgGemm_BF16FP32_Tile_512_256_32,
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

using BmgGemmBF16BF16FP32_SplitK2_RRR_TileShape_512_256_32 =
  BmgGemmBF16BF16FP32_SplitK_RRR_TileShape_512_256_32<2>;

using BmgGemmBF16BF16FP32_SplitK4_RRR_TileShape_512_256_32 =
    BmgGemmBF16BF16FP32_SplitK_RRR_TileShape_512_256_32<4>;

using BmgGemm_BF16FP32_TileShape_8_128_32 = Shape<_8, _128, _32>;
using BmgGemm_BF16FP32_Tile_8_128_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_8_128_32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_8_128_32 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_8_128_32, BmgGemm_BF16FP32_Tile_8_128_32, void, void>;

using BmgGemm_BF16FP32_TileShape_16_64_32 = Shape<_16, _64, _32>;
using BmgGemm_BF16FP32_Tile_16_64_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_16_64_32>, Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_16_64_32 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_16_64_32, BmgGemm_BF16FP32_Tile_16_64_32, void, void>;

using BmgGemm_BF16FP32_TileShape_16_128_32 = Shape<_16, _128, _32>;
using BmgGemm_BF16FP32_Tile_16_128_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_16_128_32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_16_128_32 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_16_128_32, BmgGemm_BF16FP32_Tile_16_128_32, void, void>;

using BmgGemm_BF16FP32_TileShape_64_128_32 = Shape<_64, _128, _32>;
using BmgGemm_BF16FP32_Tile_64_128_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_64_128_32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_64_128_32 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_64_128_32, BmgGemm_BF16FP32_Tile_64_128_32, void, void>;

using BmgGemm_BF16FP32_TileShape_256_128_32 = Shape<_256, _128, _32>;
using BmgGemm_BF16FP32_Tile_256_128_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_256_128_32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_256_128_32 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_256_128_32, BmgGemm_BF16FP32_Tile_256_128_32, void, void>;

using BmgGemm_BF16FP32_TileShape_4_128_32 = Shape<_4, _128, _32>;
using BmgGemm_BF16FP32_Tile_4_128_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<4, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_4_128_32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_4_128_32 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_4_128_32, BmgGemm_BF16FP32_Tile_4_128_32, void, void>;

using BmgGemm_BF16FP32_TileShape_8_256_32 = Shape<_8, _256, _32>;
using BmgGemm_BF16FP32_Tile_8_256_32 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_8_256_32>, Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_8_256_32 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_8_256_32, BmgGemm_BF16FP32_Tile_8_256_32, void, void>;

using BmgGemm_BF16FP32_Tile_8_128_32_sg8x16 = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>, Layout<BmgGemm_BF16FP32_TileShape_8_128_32>, Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>>::TiledMMA;
using BmgGemmBF16BF16FP32_RRR_TileShape_8_128_32_sg8x16 = Gemm_Bench_BF16FP32_RRR<BmgGemm_BF16FP32_TileShape_8_128_32, BmgGemm_BF16FP32_Tile_8_128_32_sg8x16, void, void>;

// Dual GEMM (sync from example 07_bmg_dual_gemm): one shared A matrix multiplied by two B
// matrices, fused through a SiLU activation epilogue. Uses MainloopIntelXeXMX16<2> + two
// linear-combination epilogues. TileShape <_128,_128,_64>, MMA XE_8x16x16_F32BF16BF16F32_TT.
using BmgDualGemm_BF16FP32_TileShape_128_128_64 = Shape<_128, _128, _64>;
using BmgDualGemm_BF16FP32_Tile_128_128_64 = typename TiledMMAHelper<
    MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
    Layout<BmgDualGemm_BF16FP32_TileShape_128_128_64>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;
using BmgDualGemmBF16BF16FP32_RRR_TileShape_128_128_64 = cutlass::gemm::device::DualGemmConfiguration<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float,               cutlass::layout::RowMajor,
    BmgDualGemm_BF16FP32_TileShape_128_128_64,
    BmgDualGemm_BF16FP32_Tile_128_128_64,
    XE_2D_U16x16x32_LD_N, XE_2D_U16x32x32_LD_V, 2>;

// ---------------------------------------------------------------------------
// Activation-fused epilogue variants (sync from example 05_bmg_gemm_with_epilogues:
// 05_bmg_gemm_with_epilogue_{relu,silu,gelu}). Same BF16 GEMM as the baseline but
// with D = Act(alpha * A*B + beta * C). Uses MainloopXeL1Staged + IntelXeGeneric
// epilogue, TileShape <_256,_256,_32>, MMA XE_DPAS_TT<8,float,bf16>.
//
// NOTE: the generic benchmark verify() compares against a plain GEMM reference and
// does NOT apply the activation, so it only matches for the baseline. Verification
// is disabled under CUTLASS_TEST_FOR_CRI (the simulator path), so these cases are
// intended to be run on the CRI simulator (see input_files/cri/input_epilogue_gemm.in).
// ---------------------------------------------------------------------------
template <
  template <class> class ActivationFn,
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA,
  typename GmemTiledCopyB>
using Gemm_Bench_BF16FP32_RRR_EltAct = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemTiledCopyA, GmemTiledCopyB,
    cutlass::epilogue::fusion::LinCombEltAct<ActivationFn, float, float, float, float,
        cutlass::FloatRoundStyle::round_to_nearest>>;

using BmgGemm_EltAct_BF16FP32_TileShape_256_256_32 = Shape<_256, _256, _32>;
using BmgGemm_EltAct_BF16FP32_Tile_256_256_32 = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>,
    Layout<BmgGemm_EltAct_BF16FP32_TileShape_256_256_32>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

using BmgGemmReLUBF16BF16FP32_RRR_TileShape_256_256_32 = Gemm_Bench_BF16FP32_RRR_EltAct<
    cutlass::epilogue::thread::ReLu, BmgGemm_EltAct_BF16FP32_TileShape_256_256_32,
    BmgGemm_EltAct_BF16FP32_Tile_256_256_32, void, void>;

using BmgGemmSiLUBF16BF16FP32_RRR_TileShape_256_256_32 = Gemm_Bench_BF16FP32_RRR_EltAct<
    cutlass::epilogue::thread::SiLu, BmgGemm_EltAct_BF16FP32_TileShape_256_256_32,
    BmgGemm_EltAct_BF16FP32_Tile_256_256_32, void, void>;

using BmgGemmGELUBF16BF16FP32_RRR_TileShape_256_256_32 = Gemm_Bench_BF16FP32_RRR_EltAct<
    cutlass::epilogue::thread::GELU, BmgGemm_EltAct_BF16FP32_TileShape_256_256_32,
    BmgGemm_EltAct_BF16FP32_Tile_256_256_32, void, void>;

// "Destination == source" benchmark variants.
//
// These mirror the FP32-output benchmarks above but store the result in the same
// element type as the inputs (e.g. BF16 in -> BF16 out). Accumulation still happens in
// float; the epilogue downcasts to the source element on store. They are backed by the
// generic GemmConfiguration / BlockScalingGemmConfiguration specializations
// (ElementC free, float accumulator) added in gemm_configuration_sycl.hpp.
//
// The TileShape / Tiler (TiledMMA) aliases are output-independent (float accumulator),
// so the existing *_TileShape_* / *_Tile_* aliases are reused.
/////////////////////////////////////////////////////////////////////////////////////

// Generic helper: output element == input element, float accumulator.
template <typename Element, typename LayoutB, Scheduler Sched, typename TileShape, typename Tiler>
using Gemm_Bench_SrcOut = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    Element, cutlass::layout::RowMajor,
    Element, LayoutB,
    Element, cutlass::layout::RowMajor,
    float,
    TileShape, Sched, Tiler,
    void, void>;

template <typename Element, typename LayoutB, typename TileShape, typename Tiler, int Splits>
struct Gemm_Bench_SrcOut_SplitK :
    Gemm_Bench_SrcOut<Element, LayoutB, Scheduler::GemmSplitK, TileShape, Tiler> {
  using Base = Gemm_Bench_SrcOut<Element, LayoutB, Scheduler::GemmSplitK, TileShape, Tiler>;
  using GemmKernel = typename Base::GemmKernel;
  constexpr static typename GemmKernel::Arguments defaultArguments() {
    using StreamKMode =
        cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams::DecompositionMode;
    typename GemmKernel::Arguments arguments{};
    arguments.scheduler = {Splits, StreamKMode::SplitK};
    return arguments;
  }
};

// ---- BF16 -> BF16 ----
using BmgGemmBF16BF16BF16_SplitK4_RRR_TileShape_512_256_32 =
    Gemm_Bench_SrcOut_SplitK<cutlass::bfloat16_t, cutlass::layout::RowMajor, BmgGemm_BF16FP32_TileShape_512_256_32, BmgGemm_BF16FP32_Tile_512_256_32, 4>;
using BmgGemmBF16BF16BF16_SplitK2_RRR_TileShape_512_256_32 =
    Gemm_Bench_SrcOut_SplitK<cutlass::bfloat16_t, cutlass::layout::RowMajor, BmgGemm_BF16FP32_TileShape_512_256_32, BmgGemm_BF16FP32_Tile_512_256_32, 2>;
using BmgGemmBF16BF16BF16_RRR_TileShape_512_256_32 =
    Gemm_Bench_SrcOut<cutlass::bfloat16_t, cutlass::layout::RowMajor, Scheduler::Gemm, BmgGemm_BF16FP32_TileShape_512_256_32, BmgGemm_BF16FP32_Tile_512_256_32>;
using BmgGemmBF16BF16BF16_RRR_TileShape_16_128_32 =
    Gemm_Bench_SrcOut<cutlass::bfloat16_t, cutlass::layout::RowMajor, Scheduler::Gemm, BmgGemm_BF16FP32_TileShape_16_128_32, BmgGemm_BF16FP32_Tile_16_128_32>;
using BmgGemmBF16BF16BF16_RRR_TileShape_64_128_32 =
    Gemm_Bench_SrcOut<cutlass::bfloat16_t, cutlass::layout::RowMajor, Scheduler::Gemm, BmgGemm_BF16FP32_TileShape_64_128_32, BmgGemm_BF16FP32_Tile_64_128_32>;
using BmgGemmBF16BF16BF16_RRR_TileShape_256_128_32 =
    Gemm_Bench_SrcOut<cutlass::bfloat16_t, cutlass::layout::RowMajor, Scheduler::Gemm, BmgGemm_BF16FP32_TileShape_256_128_32, BmgGemm_BF16FP32_Tile_256_128_32>;
using BmgGemmBF16BF16BF16_RRR_TileShape_8_128_32 =
    Gemm_Bench_SrcOut<cutlass::bfloat16_t, cutlass::layout::RowMajor, Scheduler::Gemm, BmgGemm_BF16FP32_TileShape_8_128_32, BmgGemm_BF16FP32_Tile_8_128_32>;
using BmgGemmBF16BF16BF16_RRR_TileShape_4_128_32 =
    Gemm_Bench_SrcOut<cutlass::bfloat16_t, cutlass::layout::RowMajor, Scheduler::Gemm, BmgGemm_BF16FP32_TileShape_4_128_32, BmgGemm_BF16FP32_Tile_4_128_32>;
using BmgGemmBF16BF16BF16_RRR_TileShape_8_256_32 =
    Gemm_Bench_SrcOut<cutlass::bfloat16_t, cutlass::layout::RowMajor, Scheduler::Gemm, BmgGemm_BF16FP32_TileShape_8_256_32, BmgGemm_BF16FP32_Tile_8_256_32>;
using BmgGemmBF16BF16BF16_RRR_TileShape_8_128_32_sg8x16 =
    Gemm_Bench_SrcOut<cutlass::bfloat16_t, cutlass::layout::RowMajor, Scheduler::Gemm, BmgGemm_BF16FP32_TileShape_8_128_32, BmgGemm_BF16FP32_Tile_8_128_32_sg8x16>;

