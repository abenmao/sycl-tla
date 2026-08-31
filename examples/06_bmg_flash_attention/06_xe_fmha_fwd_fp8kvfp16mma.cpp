/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

    To run one of the examples:
      $ ./examples/06_bmg_flash_attention/06_xe_fmha_fwd_prefill_fp8kvcachefp16mma_hdim128
        --seq_len_qo=1024 --seq_len_kv=1024 --num_heads_q=32 --num_heads_kv=8

    To build & run one example (from your build dir):

      $ ninja 06_xe_fmha_fwd_prefill_fp8kvcachefp16mma_hdim128
      $ ./examples/06_bmg_flash_attention/06_xe_fmha_fwd_prefill_fp8kvcachefp16mma_hdim128

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

  if (options.varlen) {
    std::cerr << "Error: Variable-length FMHA requested. Use the varlen binary." << std::endl;
    return -1;
  }

#if defined(PAGED_KV)
  if (options.seq_len_kv_cache <= 0) {
    std::cerr << "Error: this binary only instantiates CachedKV kernels; pass --seq_len_kv_cache." << std::endl;
    return -1;
  }
#else
  if (options.seq_len_kv_cache > 0 || options.use_paged_kv) {
    std::cerr << "Error: CachedKV/PagedKV requested. Use the cached_kv binary." << std::endl;
    return -1;
  }
#endif

  using ElementQ     = cutlass::half_t;
  using ElementK     = cutlass::float_e4m3_t;
  using ElementV     = cutlass::float_e4m3_t;
#if defined(MXFP8_KV)
  using ElementScale = cutlass::float_ue8m0_t;
  constexpr bool BlockScale = true;
#else
  using ElementScale = float;                                 // one scalar per tensor
  constexpr bool BlockScale = false;
#endif

  // Define the work-group tile shape depending on the head-size of the second matmul

#ifdef PREFILL
#if HEAD_DIM == 16
  /* Tiny config for testing */
  using ShapeQK = Shape<_16, _16, _32>;       // (q,k,d)
  using ShapePV = Shape<_16, _32, _16>;       // (q,v,k)
  using ShapeOut = Shape<_16, _16>;           // (q,v)
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
#if defined(MXFP8_KV) || !defined(SYCL_INTEL_TARGET) || (SYCL_INTEL_TARGET != 35)
  // Non-CRI and MXFP8 configuration.
  using ShapeQK = Shape<_128, _32, _32>;
  using ShapePV = Shape<_128, _32, _32>;
  using ShapeOut = Shape<_128, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#else
  using ShapeQK = Shape<_512, _64, _64>;
  using ShapePV = Shape<_512, _64, _64>;
  using ShapeOut = Shape<_512, _128>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;

  // CRI causal configurations.
  using ShapeQK_Causal = Shape<_256, _64, _64>;
  using ShapePV_Causal = Shape<_256, _64, _64>;
  using ShapeOut_Causal = Shape<_256, _128>;
  using SubgroupLayoutQK_Causal = Layout<Shape<_16, _1, _1>>;

  using ShapeQK8 = Shape<_64, _64, _64>;
  using ShapePV8 = Shape<_64, _64, _64>;
  using ShapeOut8 = Shape<_64, _128>;
  using SubgroupLayoutQK8 = Layout<Shape<_8, _1, _1>>;
#endif

#elif HEAD_DIM == 192
  using ShapeQK = Shape<_128, _32, _32>;
  using ShapePV = Shape<_128, _32, _32>;
  using ShapeOut = Shape<_128, _192>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;

#endif
#elif defined(DECODE)

#if HEAD_DIM == 16
  /* Tiny config for testing */
  using KVTileSize  = _64;
  using SubgroupsK  = _4;
  using PVTileN     = _16;
  using QKTileK     = _16;
  using HeadDimSize = _16;
#elif HEAD_DIM == 64
  using KVTileSize  = _64;
  using SubgroupsK  = _4;
  using PVTileN     = _64;
  using QKTileK     = _64;
  using HeadDimSize = _64;
#elif HEAD_DIM == 96
  using KVTileSize  = _64;
  using SubgroupsK  = _4;
  using PVTileN     = _32;
  using QKTileK     = _32;
  using HeadDimSize = _96;
#elif HEAD_DIM == 128
  using KVTileSize  = _128;
  using SubgroupsK  = _8;
  using PVTileN     = _64;
  using QKTileK     = _128;
  using HeadDimSize = _128;
#elif HEAD_DIM == 192
  using KVTileSize  = _128;
  using SubgroupsK  = _8;
  using PVTileN     = _32;
  using QKTileK     = _64;
  using HeadDimSize = _192;
#endif

  using ShapeQK8  = Shape<_8, KVTileSize, QKTileK>; // (q,k,d)
  using ShapePV8  = Shape<_8, PVTileN, KVTileSize>; // (q,v,k)
  using ShapeOut8 = Shape<_8, HeadDimSize>;         // (q,v)
  using SubgroupLayoutQK8 = Layout<Shape<_1, SubgroupsK, _1>>;

#if HEAD_DIM == 128
  using QKTileK16 = _64;
#else
  using QKTileK16 = QKTileK;
#endif

  using ShapeQK16  = Shape<_16, KVTileSize, QKTileK16>;
  using ShapePV16  = Shape<_16, PVTileN, KVTileSize>;
  using ShapeOut16 = Shape<_16, HeadDimSize>;
  using SubgroupLayoutQK16 = Layout<Shape<_2, SubgroupsK, _1>>;

  using ShapeQK24  = Shape<_24, KVTileSize, QKTileK>;
  using ShapePV24  = Shape<_24, PVTileN, KVTileSize>;
  using ShapeOut24 = Shape<_24, HeadDimSize>;
  using SubgroupLayoutQK24 = Layout<Shape<_3, SubgroupsK, _1>>;

  // Benchmarks favor BF16 tile shapes through 24 Q rows; FP8 shapes win from 32.
  using ShapeQK32  = Shape<_32, _256, _128>;
  using ShapePV32  = Shape<_32, _128, _256>;
  using ShapeOut32 = Shape<_32, HeadDimSize>;
  using SubgroupLayoutQK32 = Layout<Shape<_4, SubgroupsK, _1>>;

  using ShapeQK40  = Shape<_40, _128, _128>;
  using ShapePV40  = Shape<_40, _128, _128>;
  using ShapeOut40 = Shape<_40, HeadDimSize>;
  using SubgroupLayoutQK40 = Layout<Shape<_5, _4, _1>>;

  using ShapeQK48  = Shape<_48, _128, _128>;
  using ShapePV48  = Shape<_48, _128, _128>;
  using ShapeOut48 = Shape<_48, HeadDimSize>;
  using SubgroupLayoutQK48 = Layout<Shape<_6, _4, _1>>;

  using ShapeQK64  = Shape<_64, _64, _128>;
  using ShapePV64  = Shape<_64, _128, _64>;
  using ShapeOut64 = Shape<_64, HeadDimSize>;
  using SubgroupLayoutQK64 = Layout<Shape<_8, _4, _1>>;
#else
#error Either DECODE or PREFILL should be defined.
#endif

#ifdef DECODE
  constexpr int PipelineStages = 1;
#else
  constexpr int PipelineStages = 2;
#endif

#if defined(DECODE)
  const int gqa_group  = options.num_heads_q / options.num_heads_kv;
  const int q_len      = options.seq_len_qo;
  const int total_rows = gqa_group * q_len;

#if defined(PAGED_KV)
  const int kv_tile = int(KVTileSize::value);
  const int kv_blocks = cute::ceil_div(options.seq_len_kv, kv_tile)
                      + cute::ceil_div(options.seq_len_kv_cache, kv_tile);
  const int base_units = options.batch * options.num_heads_kv;
  const int saturation_cores_default = estimate_saturation_cores(base_units, kv_blocks);
  const int saturation_cores =
      cutlass::fmha::kernel::fmha_split_saturation_cores(saturation_cores_default);
  const bool use_split = total_rows <= 64
                      && base_units < saturation_cores
                      && base_units * kv_blocks > saturation_cores;

  #define FMHA_RUN_ONE(QK, PV, OUT, SGL, CAUSAL)                                                        \
    FMHAConfig<CAUSAL, BlockScale, QK, PV, OUT, SGL, void, PipelineStages,                            \
               ElementQ, ElementK, ElementV, ElementScale, /*kGqaFusion=*/true>::                     \
           template run</*isVarLen=*/false, /*PagedKV=*/true,                                     \
               cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>>(options)

  #define FMHA_RUN_SPLIT(QK, PV, OUT, SGL, CAUSAL)                                                      \
    FMHAConfig<CAUSAL, BlockScale, QK, PV, OUT, SGL, void, PipelineStages,                            \
               ElementQ, ElementK, ElementV, ElementScale, /*kGqaFusion=*/false>::                    \
           template run</*isVarLen=*/false, /*PagedKV=*/true,                                     \
               cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>(options)

#define FMHA_RUN_Q(QK, PV, OUT, SGL)                                                                  \
    (use_split                                                                                        \
       ? (options.is_causal ? FMHA_RUN_SPLIT(QK, PV, OUT, SGL, true)                                 \
                : FMHA_RUN_SPLIT(QK, PV, OUT, SGL, false))                               \
       : (options.is_causal ? FMHA_RUN_ONE(QK, PV, OUT, SGL, true)                                  \
                : FMHA_RUN_ONE(QK, PV, OUT, SGL, false)))
#else
#define FMHA_RUN_ONE(QK, PV, OUT, SGL, CAUSAL)                                                        \
    FMHAConfig<CAUSAL, BlockScale, QK, PV, OUT, SGL, void, PipelineStages,                            \
               ElementQ, ElementK, ElementV, ElementScale, /*kGqaFusion=*/true>::                     \
               template run</*isVarLen=*/false, /*PagedKV=*/false,                                    \
               cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>>(options)

#define FMHA_RUN_Q(QK, PV, OUT, SGL)                                                                  \
    (options.is_causal ? FMHA_RUN_ONE(QK, PV, OUT, SGL, true)                                         \
                       : FMHA_RUN_ONE(QK, PV, OUT, SGL, false))
#endif

  if (total_rows <= 8)
    return FMHA_RUN_Q(ShapeQK8,  ShapePV8,  ShapeOut8,  SubgroupLayoutQK8);
  else if (total_rows <= 16)
    return FMHA_RUN_Q(ShapeQK16, ShapePV16, ShapeOut16, SubgroupLayoutQK16);
  else if (total_rows <= 24)
    return FMHA_RUN_Q(ShapeQK24, ShapePV24, ShapeOut24, SubgroupLayoutQK24);
  else if (total_rows <= 32)
    return FMHA_RUN_Q(ShapeQK32, ShapePV32, ShapeOut32, SubgroupLayoutQK32);
  else if (total_rows <= 40)
    return FMHA_RUN_Q(ShapeQK40, ShapePV40, ShapeOut40, SubgroupLayoutQK40);
  else if (total_rows <= 48)
    return FMHA_RUN_Q(ShapeQK48, ShapePV48, ShapeOut48, SubgroupLayoutQK48);
  else
    return FMHA_RUN_Q(ShapeQK64, ShapePV64, ShapeOut64, SubgroupLayoutQK64);

#undef FMHA_RUN_Q
#if defined(PAGED_KV)
#undef FMHA_RUN_SPLIT
#endif
#undef FMHA_RUN_ONE
#else
  using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>;
  using FMHANonCausal = FMHAConfig<false, BlockScale, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, ElementQ, ElementK, ElementV, ElementScale>;

#if defined(PAGED_KV)
#define FMHA_RUN_PREFILL(CFG) CFG::template run<false, true, Scheduler>(options)
#else
#define FMHA_RUN_PREFILL(CFG) CFG::template run<false, false, Scheduler>(options)
#endif

#if HEAD_DIM == 128 && !defined(MXFP8_KV) && \
  defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
  using FMHACausal = FMHAConfig<true, BlockScale, ShapeQK_Causal, ShapePV_Causal, ShapeOut_Causal, SubgroupLayoutQK_Causal, void, PipelineStages, ElementQ, ElementK, ElementV, ElementScale>;
  using FMHACausal8 = FMHAConfig<true, BlockScale, ShapeQK8, ShapePV8, ShapeOut8, SubgroupLayoutQK8, void, PipelineStages, ElementQ, ElementK, ElementV, ElementScale>;

  const int num_xe_cores = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  int num_q_tiles_256 = (options.seq_len_qo + 255) / 256;
  int total_wgs_256 = num_q_tiles_256 * options.num_heads_q * options.batch;
  bool use_small = options.seq_len_qo < 512 || total_wgs_256 < num_xe_cores;

  if (options.is_causal) {
    if (use_small) {
      return FMHA_RUN_PREFILL(FMHACausal8);
    }
    return FMHA_RUN_PREFILL(FMHACausal);
  }
  else {
    return FMHA_RUN_PREFILL(FMHANonCausal);
  }
#else
  using FMHACausal = FMHAConfig<true, BlockScale, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, ElementQ, ElementK, ElementV, ElementScale>;

  if (options.is_causal) {
    return FMHA_RUN_PREFILL(FMHACausal);
  } else {
    return FMHA_RUN_PREFILL(FMHANonCausal);
  }
#endif
#undef FMHA_RUN_PREFILL
#endif
}
