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

#include "xe_fmha_fwd_runner.hpp"

int main(int argc, const char **argv) {
  Options options;
  options.parse(argc, argv);
  options.varlen = true;

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }
  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

#if defined(PAGED_KV)
  if (options.seq_len_kv_cache <= 0) {
    std::cerr << "Error: this binary only instantiates CachedKV kernels; pass --seq_len_kv_cache." << std::endl;
    return -1;
  }
  constexpr bool IsPagedKV = true;
#else
  if (options.seq_len_kv_cache > 0 || options.use_paged_kv) {
    std::cerr << "Error: CachedKV/PagedKV requested. Use the cached_kv varlen binary." << std::endl;
    return -1;
  }
  constexpr bool IsPagedKV = false;
#endif

#if defined(VARLEN_FP8KVFP16MMA)
  using ElementQ = cutlass::half_t;
  using ElementK = cutlass::float_e4m3_t;
  using ElementV = cutlass::float_e4m3_t;
#if defined(MXFP8_KV)
  using ElementScale = cutlass::float_ue8m0_t;
  constexpr bool BlockScale = true;
#else
  using ElementScale = float;
  constexpr bool BlockScale = false;
#endif
#elif defined(IS_MX_FLOAT_E5M2)
  using ElementType = cutlass::mx_float8_t<float_e5m2_t>;
  using ElementQ = typename ElementType::DataType;
  using ElementK = typename ElementType::DataType;
  using ElementV = typename ElementType::DataType;
  using ElementScale = typename ElementType::ScaleFactorType;
  constexpr bool BlockScale = true;
#elif defined(IS_MX_FLOAT_E4M3)
  using ElementType = cutlass::mx_float8_t<float_e4m3_t>;
  using ElementQ = typename ElementType::DataType;
  using ElementK = typename ElementType::DataType;
  using ElementV = typename ElementType::DataType;
  using ElementScale = typename ElementType::ScaleFactorType;
  constexpr bool BlockScale = true;
#elif defined(IS_MX_FLOAT_E2M1)
  using ElementType = cutlass::mx_float4_t<float_e2m1_t>;
  using ElementQ = typename ElementType::DataType;
  using ElementK = typename ElementType::DataType;
  using ElementV = bfloat16_t;
  using ElementScale = typename ElementType::ScaleFactorType;
  constexpr bool BlockScale = true;
#elif defined(IS_FLOAT_E5M2)
  using ElementQ = cutlass::float_e5m2_t;
  using ElementK = cutlass::float_e5m2_t;
  using ElementV = cutlass::float_e5m2_t;
  using ElementScale = float;
  constexpr bool BlockScale = false;
#elif defined(IS_FLOAT_E4M3)
  using ElementQ = cutlass::float_e4m3_t;
  using ElementK = cutlass::float_e4m3_t;
  using ElementV = cutlass::float_e4m3_t;
  using ElementScale = float;
  constexpr bool BlockScale = false;
#elif defined(IS_FLOAT_E2M1)
  using ElementQ = cutlass::float_e2m1_t;
  using ElementK = cutlass::float_e2m1_t;
  using ElementV = cutlass::bfloat16_t;
  using ElementScale = float;
  constexpr bool BlockScale = false;
#else
  using ElementQ = bfloat16_t;
  using ElementK = bfloat16_t;
  using ElementV = bfloat16_t;
  using ElementScale = float;
  constexpr bool BlockScale = false;
#endif

#if defined(VARLEN_FP8KVFP16MMA)
#if HEAD_DIM == 16
  using ShapeQK = Shape<_16, _16, _32>;
  using ShapePV = Shape<_16, _32, _16>;
  using ShapeOut = Shape<_16, _16>;
  using SubgroupLayoutQK = Layout<Shape<_1, _1, _1>>;
#elif HEAD_DIM == 64
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#elif HEAD_DIM == 96
  using ShapeQK = Shape<_128, _32, _32>;
  using ShapePV = Shape<_128, _32, _32>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#elif HEAD_DIM == 128
  using ShapeQK = Shape<_128, _32, _32>;
  using ShapePV = Shape<_128, _32, _32>;
  using ShapeOut = Shape<_128, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#elif HEAD_DIM == 192
  using ShapeQK = Shape<_128, _32, _32>;
  using ShapePV = Shape<_128, _32, _32>;
  using ShapeOut = Shape<_128, _192>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#endif
#elif defined(IS_MX_FLOAT_E5M2) || defined(IS_MX_FLOAT_E4M3) || defined(IS_MX_FLOAT_E2M1)
#if HEAD_DIM == 16
  using ShapeQK = Shape<_16, _16, _32>;
  using ShapePV = Shape<_16, _32, _16>;
  using ShapeOut = Shape<_16, _16>;
  using SubgroupLayoutQK = Layout<Shape<_1, _1, _1>>;
#elif HEAD_DIM == 64
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
#elif HEAD_DIM == 96
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
#elif HEAD_DIM == 128
#if defined(SYCL_TARGET_INTEL_GPU_CRI) && (defined(IS_MX_FLOAT_E5M2) || defined(IS_MX_FLOAT_E4M3))
  using ShapeQK = Shape<_512, _64, _64>;
  using ShapePV = Shape<_512, _64, _64>;
  using ShapeOut = Shape<_512, _128>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
#else
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#endif
#elif HEAD_DIM == 192
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#endif
#else
#if HEAD_DIM == 16
  using ShapeQK = Shape<_16, _16, _32>;
  using ShapePV = Shape<_16, _32, _16>;
  using ShapeOut = Shape<_16, _16>;
  using SubgroupLayoutQK = Layout<Shape<_1, _1, _1>>;
#elif HEAD_DIM == 64
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
#elif HEAD_DIM == 96
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
#elif HEAD_DIM == 128
#if defined(SYCL_TARGET_INTEL_GPU_CRI)
#if (defined(IS_FLOAT_E5M2) || defined(IS_FLOAT_E4M3)) && !defined(PAGED_KV)
  using ShapeQK = Shape<_512, _64, _128>;
#else
  using ShapeQK = Shape<_512, _64, _64>;
#endif
  using ShapePV = Shape<_512, _64, _64>;
  using ShapeOut = Shape<_512, _128>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
  using ShapeQK_Causal = Shape<_256, _64, _64>;
  using ShapePV_Causal = Shape<_256, _64, _64>;
  using ShapeOut_Causal = Shape<_256, _128>;
  using SubgroupLayoutQK_Causal = Layout<Shape<_16, _1, _1>>;
  using ShapeQK4 = Shape<_128, _64, _64>;
  using ShapePV4 = Shape<_128, _64, _64>;
  using ShapeOut4 = Shape<_128, _128>;
  using SubgroupLayoutQK4 = Layout<Shape<_8, _1, _1>>;
  using ShapeQK8 = Shape<_64, _32, _64>;
  using ShapePV8 = Shape<_64, _64, _32>;
  using ShapeOut8 = Shape<_64, _128>;
  using SubgroupLayoutQK8 = Layout<Shape<_8, _1, _1>>;
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
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#endif
#endif

  constexpr int PipelineStages = 2;
  using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>;

#if HEAD_DIM == 128 && defined(SYCL_TARGET_INTEL_GPU_CRI) && \
    !defined(VARLEN_FP8KVFP16MMA) && !defined(IS_MX_FLOAT_E5M2) && \
    !defined(IS_MX_FLOAT_E4M3) && !defined(IS_MX_FLOAT_E2M1) && \
    (!defined(PAGED_KV) || !(defined(IS_FLOAT_E5M2) || defined(IS_FLOAT_E4M3)))
#if defined(PAGED_KV)
  using FMHACausal = FMHAConfig<true, BlockScale, ShapeQK_Causal, ShapePV_Causal, ShapeOut_Causal,
                                SubgroupLayoutQK_Causal, void, PipelineStages,
                                ElementQ, ElementK, ElementV, ElementScale>;
  using FMHANonCausal = FMHAConfig<false, BlockScale, ShapeQK, ShapePV, ShapeOut,
                                   SubgroupLayoutQK, void, PipelineStages,
                                   ElementQ, ElementK, ElementV, ElementScale>;
  using FMHACausal4 = FMHAConfig<true, BlockScale, ShapeQK4, ShapePV4, ShapeOut4,
                                 SubgroupLayoutQK4, void, PipelineStages,
                                 ElementQ, ElementK, ElementV, ElementScale>;
  using FMHANonCausal4 = FMHAConfig<false, BlockScale, ShapeQK4, ShapePV4, ShapeOut4,
                                    SubgroupLayoutQK4, void, PipelineStages,
                                    ElementQ, ElementK, ElementV, ElementScale>;
  const int num_xe_cores = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  const int num_q_tiles_256 = (options.seq_len_qo + 255) / 256;
  const int total_wgs_256 = num_q_tiles_256 * options.num_heads_q * options.batch;
  const bool use_small = options.seq_len_qo < 512 || total_wgs_256 < 2 * num_xe_cores;

  if (options.is_causal) {
    if (use_small)
      return FMHACausal4::template run<true, IsPagedKV, Scheduler>(options);
    return FMHACausal::template run<true, IsPagedKV, Scheduler>(options);
  }
  if (options.seq_len_qo < 512)
    return FMHANonCausal4::template run<true, IsPagedKV, Scheduler>(options);
  return FMHANonCausal::template run<true, IsPagedKV, Scheduler>(options);
#else
  using CausalSmallScheduler =
    cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false, true, false, true>;
  using CausalScheduler =
    cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false, true, false, false>;
  using FMHACausal = FMHAConfig<true, BlockScale, ShapeQK_Causal, ShapePV_Causal, ShapeOut_Causal,
                                SubgroupLayoutQK_Causal, void, PipelineStages,
                                ElementQ, ElementK, ElementV, ElementScale>;
  using FMHANonCausal = FMHAConfig<false, BlockScale, ShapeQK, ShapePV, ShapeOut,
                                   SubgroupLayoutQK, void, PipelineStages,
                                   ElementQ, ElementK, ElementV, ElementScale>;
  using FMHACausal4 = FMHAConfig<true, BlockScale, ShapeQK4, ShapePV4, ShapeOut4,
                                 SubgroupLayoutQK4, void, 1,
                                 ElementQ, ElementK, ElementV, ElementScale>;
  using FMHANonCausal8 = FMHAConfig<false, BlockScale, ShapeQK8, ShapePV8, ShapeOut8,
                                    SubgroupLayoutQK8, void, 1,
                                    ElementQ, ElementK, ElementV, ElementScale>;
  const int num_xe_cores = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  const int num_q_tiles_256 = (options.seq_len_qo + 255) / 256;
  const int total_wgs_256 = num_q_tiles_256 * options.num_heads_q * options.batch;
  const bool use_small =
      (options.seq_len_qo - 576) * options.num_heads_q <= 11840 || total_wgs_256 < 2 * num_xe_cores;

  if (options.is_causal) {
    if (use_small)
      return FMHACausal4::template run<true, IsPagedKV, CausalSmallScheduler>(options);
    return FMHACausal::template run<true, IsPagedKV, CausalScheduler>(options);
  }
  if (options.seq_len_qo < 512)
    return FMHANonCausal8::template run<
        true, IsPagedKV,
        cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false, false, false, true>>(options);
  return FMHANonCausal::template run<true, IsPagedKV, Scheduler>(options);
#endif
#else
  using FMHACausal = FMHAConfig<true, BlockScale, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK,
                                void, PipelineStages, ElementQ, ElementK, ElementV, ElementScale>;
  using FMHANonCausal = FMHAConfig<false, BlockScale, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK,
                                   void, PipelineStages, ElementQ, ElementK, ElementV, ElementScale>;
  if (options.is_causal)
    return FMHACausal::template run<true, IsPagedKV, Scheduler>(options);
  return FMHANonCausal::template run<true, IsPagedKV, Scheduler>(options);
#endif
}
