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
#include "gemm_configuration_sycl.hpp"

using Scheduler = cutlass::gemm::device::Scheduler;

#if defined(SYCL_TARGET_INTEL_GPU_CRI)

template <
  typename ElementC,
  typename ElementD,
  typename AccType,
  typename LayoutB,
  Scheduler Sched,
  typename TileShape,
  typename Tiler,
  typename GmemTiledCopyA = void,
  typename GmemTiledCopyB = void,
  bool UseW8A8 = false>
using Gemm_Bench_E5M2 = cute::conditional_t<UseW8A8,
    cutlass::gemm::device::W8A8GemmConfiguration<
        cutlass::float_e5m2_t, cutlass::layout::RowMajor,
        cutlass::float_e5m2_t, cutlass::layout::RowMajor,
        ElementC, cutlass::layout::RowMajor,
        AccType,
        TileShape,
        Tiler,
        GmemTiledCopyA, GmemTiledCopyB>,
    cutlass::gemm::device::GemmConfiguration<
        cutlass::arch::IntelXe,
        cutlass::float_e5m2_t, cutlass::layout::RowMajor,
        cutlass::float_e5m2_t, LayoutB,
        ElementC, cutlass::layout::RowMajor,
        ElementD,
        TileShape, Sched, Tiler,
        GmemTiledCopyA, GmemTiledCopyB,
        cutlass::epilogue::fusion::LinearCombination<ElementD, AccType>>>;

template <typename ElementC, typename ElementD, typename AccType,
          int WG_M, int WG_N, int WG_K, int SG_M, int SG_N, int Splits = 0,
          bool UseW8A8 = false>
using E5M2_RRR_GEMM_Base = Gemm_Bench_E5M2<ElementC, ElementD, AccType, cutlass::layout::RowMajor,
    (Splits > 0 ? Scheduler::GemmSplitK : Scheduler::Gemm),
    Shape<Int<WG_M>, Int<WG_N>, Int<WG_K>>,
    cute::conditional_t<UseW8A8,
        XeW8A8TiledMMA<WG_M, WG_N, WG_K, SG_M, SG_N>,
        XeTiledMMA<WG_M, WG_N, WG_K, SG_M, SG_N, AccType, cutlass::float_e5m2_t>>,
    cute::conditional_t<UseW8A8, XE_2D_U8x32x32_LD_N, void>,
    cute::conditional_t<UseW8A8, XE_2D_U8x32x32_LD_V, void>,
    UseW8A8>;

template <typename ElementC, typename ElementD, typename AccType,
          int WG_M, int WG_N, int WG_K, int SG_M, int SG_N, int Splits = 0,
          bool UseW8A8 = false>
struct E5M2_RRR_GEMM :
    E5M2_RRR_GEMM_Base<ElementC, ElementD, AccType, WG_M, WG_N, WG_K, SG_M, SG_N, Splits, UseW8A8> {
  using Base = E5M2_RRR_GEMM_Base<ElementC, ElementD, AccType, WG_M, WG_N, WG_K, SG_M, SG_N, Splits, UseW8A8>;
  using GemmKernel = typename Base::GemmKernel;
  static_assert(WG_M % SG_M == 0, "WG_M must be divisible by SG_M");
  static_assert(WG_N % SG_N == 0, "WG_N must be divisible by SG_N");

  constexpr static typename GemmKernel::Arguments defaultArguments() {
    if constexpr (!UseW8A8 && Splits > 0) {
      using StreamKMode =
          cutlass::gemm::kernel::detail::PersistentTileSchedulerXeStreamKParams::DecompositionMode;
      typename GemmKernel::Arguments arguments{};
      arguments.scheduler = {Splits, StreamKMode::SplitK};
      return arguments;
    } else {
      return Base::defaultArguments();
    }
  }
};
#endif

void register_gemm_benchmarks_e5m2() {
#if defined(SYCL_TARGET_INTEL_GPU_CRI)
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32FP32FP32_RRR_WG512x256x64_SG64x64x64", E5M2_RRR_GEMM<float, float, float, 512, 256, 64, 64, 64>);
  CUTLASS_BENCHMARK_T("GemmW8A8_E5M2E5M2FP32FP32FP32_RRR_WG256x256x32_SG32x64x32", E5M2_RRR_GEMM<float, float, float, 256, 256, 32, 32, 64, 0, true>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2BF16BF16BF16_RRR_WG512x256x128_SG64x64x128", E5M2_RRR_GEMM<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, 512, 256, 128, 64, 64>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG128x256x128_SG32x32x128", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 128, 256, 128, 32, 32>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG256x256x64_SG32x64x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 256, 64, 32, 64>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG256x256x64_SG64x64x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 256, 64, 64, 64>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG512x256x64_SG64x64x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 512, 256, 64, 64, 64>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG4x128x64_SG4x32x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 4, 128, 64, 4, 32>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG128x192x64_SG16x96x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 128, 192, 64, 16, 96>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG128x384x64_SG32x96x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 128, 384, 64, 32, 96>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG448x320x64_SG56x80x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 448, 320, 64, 56, 80>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG160x512x64_SG40x64x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 160, 512, 64, 40, 64>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG8x128x128_SG8x16x128", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 8, 128, 128, 8, 16>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG32x128x128_SG16x32x128", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 32, 128, 128, 16, 32>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG96x128x64_SG24x32x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 96, 128, 64, 24, 32>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG1x256x64_SG1x16x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 1, 256, 64, 1, 16>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG1x384x64_SG1x48x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 1, 384, 64, 1, 48>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG4x256x32_SG4x32x32", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 4, 256, 32, 4, 32>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG16x128x64_SG4x16x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 16, 128, 64, 4, 16>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG1x128x512_SG1x16x512", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 1, 128, 512, 1, 16>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG256x192x64_SG64x48x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 192, 64, 64, 48>);
  CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG4x32x256_SG1x16x256", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 4, 32, 256, 1, 16>);

  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG256x128x128_SG16x64x128", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 128, 128, 16, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG256x256x128_SG32x64x128", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 256, 128, 32, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG128x384x64_SG64x48x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 128, 384, 64, 64, 48>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG288x512x32_SG72x64x32", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 288, 512, 32, 72, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG256x512x64_SG64x64x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 512, 64, 64, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG320x128x128_SG40x32x128", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 320, 128, 128, 40, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG320x256x64_SG40x64x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 320, 256, 64, 40, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG128x512x64_SG32x64x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 128, 512, 64, 32, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG448x192x64_SG56x48x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 448, 192, 64, 56, 48>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG224x512x32_SG56x64x32", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 224, 512, 32, 56, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG512x256x32_SG64x64x32", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 512, 256, 32, 64, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG256x512x32_SG64x64x32", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 512, 32, 64, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG64x256x128_SG8x64x128", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 64, 256, 128, 8, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG64x512x128_SG16x64x128", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 64, 512, 128, 16, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG64x512x64_SG16x64x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 64, 512, 64, 16, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG96x512x64_SG24x64x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 96, 512, 64, 24, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32FP32FP32_RRR_WG256x256x32_SG32x64x32", E5M2_RRR_GEMM<float, float, float, 256, 256, 32, 32, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32FP32FP32_RRR_WG512x256x64_SG64x64x64_SplitK2", E5M2_RRR_GEMM<float, float, float, 512, 256, 64, 64, 64, 2>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32FP32FP32_RRR_WG512x256x64_SG64x64x64_SplitK4", E5M2_RRR_GEMM<float, float, float, 512, 256, 64, 64, 64, 4>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32FP32FP32_RRR_WG8x128x64_SG8x32x64", E5M2_RRR_GEMM<float, float, float, 8, 128, 64, 8, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32FP32FP32_RRR_WG16x128x64_SG16x32x64", E5M2_RRR_GEMM<float, float, float, 16, 128, 64, 16, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32FP32FP32_RRR_WG64x128x64_SG64x32x64", E5M2_RRR_GEMM<float, float, float, 64, 128, 64, 64, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32FP32FP32_RRR_WG256x128x64_SG32x32x64", E5M2_RRR_GEMM<float, float, float, 256, 128, 64, 32, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32FP32FP32_RRR_WG4x128x64_SG4x32x64", E5M2_RRR_GEMM<float, float, float, 4, 128, 64, 4, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32FP32FP32_RRR_WG8x256x64_SG8x32x64", E5M2_RRR_GEMM<float, float, float, 8, 256, 64, 8, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32FP32FP32_RRR_WG8x128x64_SG8x16x64", E5M2_RRR_GEMM<float, float, float, 8, 128, 64, 8, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG512x256x64_SG64x64x64_SplitK4", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 512, 256, 64, 64, 64, 4>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG512x256x64_SG64x64x64_SplitK2", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 512, 256, 64, 64, 64, 2>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG512x256x64_SG64x64x64", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 512, 256, 64, 64, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG16x128x64_SG16x32x64", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 16, 128, 64, 16, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG64x128x64_SG64x32x64", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 64, 128, 64, 64, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG256x128x64_SG32x32x64", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 256, 128, 64, 32, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG8x128x64_SG8x32x64", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 8, 128, 64, 8, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG4x128x64_SG4x32x64", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 4, 128, 64, 4, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG8x256x64_SG8x32x64", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 8, 256, 64, 8, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG8x128x64_SG8x16x64", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 8, 128, 64, 8, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG16x128x128_SG16x32x128", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 16, 128, 128, 16, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG64x128x128_SG64x32x128", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 64, 128, 128, 64, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG192x128x64_SG24x32x64", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 192, 128, 64, 24, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG8x128x256_SG8x32x256", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 8, 128, 256, 8, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG8x256x64_SG8x16x64", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 8, 256, 64, 8, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG128x256x128_SG16x64x128", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 128, 256, 128, 16, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG256x256x128_SG32x64x128", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 256, 256, 128, 32, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG256x128x64_SG64x16x64", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 256, 128, 64, 64, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG256x256x64_SG64x64x64", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 256, 256, 64, 64, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG8x128x512_SG8x16x512", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 8, 128, 512, 8, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32E5M2FP32_RRR_WG8x128x512_SG8x16x512_SplitK2", E5M2_RRR_GEMM<float, cutlass::float_e5m2_t, float, 8, 128, 512, 8, 16, 2>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG128x192x128_SG16x48x128", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 128, 192, 128, 16, 48>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG256x192x128_SG32x48x128", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 192, 128, 32, 48>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG256x128x64_SG64x16x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 128, 64, 64, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG256x256x128_SG64x32x128", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 256, 128, 64, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG128x256x128_SG16x64x128", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 128, 256, 128, 16, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG16x256x256_SG16x16x256", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 16, 256, 256, 16, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG64x128x256_SG16x16x256", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 64, 128, 256, 16, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG192x128x64_SG24x32x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 192, 128, 64, 24, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG8x128x256_SG8x32x256", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 8, 128, 256, 8, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG8x256x64_SG8x32x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 8, 256, 64, 8, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG8x256x64_SG8x16x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 8, 256, 64, 8, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG4x128x256_SG4x16x256", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 4, 128, 256, 4, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG8x128x64_SG8x16x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 8, 128, 64, 8, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG256x128x64_SG32x32x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 128, 64, 32, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG16x128x256_SG8x16x256", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 16, 128, 256, 8, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG32x128x256_SG8x32x256", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 32, 128, 256, 8, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG128x256x64_SG16x64x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 128, 256, 64, 16, 64>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG256x192x128_SG64x48x128", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 256, 192, 128, 64, 48>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG1x128x256_SG1x32x256", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 1, 128, 256, 1, 32>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG8x256x64_SG4x16x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 8, 256, 64, 4, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG2x32x256_SG2x16x256", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 2, 32, 256, 2, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG16x128x64_SG8x16x64", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 16, 128, 64, 8, 16>);
  // CUTLASS_BENCHMARK_T("Gemm_E5M2E5M2FP32BF16FP32_RRR_WG1x64x512_SG1x16x512", E5M2_RRR_GEMM<float, cutlass::bfloat16_t, float, 1, 64, 512, 1, 16>);
#endif
}
