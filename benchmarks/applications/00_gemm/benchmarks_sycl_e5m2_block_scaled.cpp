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
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_256_256_32);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2FP32_RRR_TileShape_256_256_32);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_512_256_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2FP32_RRR_TileShape_512_256_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_8_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_16_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_64_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_256_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_4_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2FP32_RRR_TileShape_8_256_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2FP32_RRR_TileShape_8_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2FP32_RRR_TileShape_8_128_64_sg8x16);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2BF16_RRR_TileShape_512_256_128);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2BF16_RRR_TileShape_512_256_128);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2E5M2_RRR_TileShape_512_256_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2E5M2_RRR_TileShape_16_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2E5M2_RRR_TileShape_64_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2E5M2_RRR_TileShape_256_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2E5M2_RRR_TileShape_4_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemm_E5M2E5M2E5M2_RRR_TileShape_8_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2E5M2_RRR_TileShape_8_128_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2E5M2_RRR_TileShape_8_256_64);
CUTLASS_CREATE_GEMM_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2E5M2_RRR_TileShape_8_128_64_sg8x16);
#endif
void register_gemm_benchmarks_e5m2_block_scaled() {
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2FP32_RRR_TileShape_256_256_32);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_8_128_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_16_128_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_64_128_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_256_128_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2FP32_RRR_TileShape_4_128_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2FP32_RRR_TileShape_8_256_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2FP32_RRR_TileShape_8_128_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2FP32_RRR_TileShape_8_128_64_sg8x16);
  CUTLASS_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2FP32_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2BF16_RRR_TileShape_512_256_128);
  CUTLASS_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2BF16_RRR_TileShape_512_256_128);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2E5M2_RRR_TileShape_512_256_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2E5M2_RRR_TileShape_16_128_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2E5M2_RRR_TileShape_64_128_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2E5M2_RRR_TileShape_256_128_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2E5M2_RRR_TileShape_4_128_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemm_E5M2E5M2E5M2_RRR_TileShape_8_128_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2E5M2_RRR_TileShape_8_128_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2E5M2_RRR_TileShape_8_256_64);
  CUTLASS_BENCHMARK(CriBLockScalingGemmNonNative_E5M2E5M2E5M2_RRR_TileShape_8_128_64_sg8x16);
#endif
}
