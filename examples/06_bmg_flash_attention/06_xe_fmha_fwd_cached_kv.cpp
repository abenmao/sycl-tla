/***************************************************************************************************
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
    \brief Flash Attention V2 with Cached KV for Intel BMG

    This file instantiates only the CachedKV=true kernel variants,
    split out from the main 06_xe_fmha_fwd.cpp to reduce per-binary compile time.

    Instantiated kernels (2 total):
      - Causal × {true, false}
      - VarLen = false
      - CachedKV = true
      - PagedKV = true (contiguous cache uses one identity-mapped page)

    To build & run (from your build dir):
      $ ninja 06_xe_fmha_fwd_prefill_cached_kv_bfloat16_t_hdim128
      $ ./examples/sycl/06_bmg_flash_attention/06_xe_fmha_fwd_prefill_cached_kv_bfloat16_t_hdim128 \
            --seq_len_kv_cache=256
*/

#include "xe_fmha_fwd_runner.hpp"

#include <cassert>

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

#ifdef IS_FLOAT_E5M2
  using ElementQ = cutlass::float_e5m2_t;
  using ElementK = cutlass::float_e5m2_t;
  using ElementV = cutlass::float_e5m2_t;
#elif defined(IS_FLOAT_E4M3)
  using ElementQ = cutlass::float_e4m3_t;
  using ElementK = cutlass::float_e4m3_t;
  using ElementV = cutlass::float_e4m3_t;
#elif defined(IS_FLOAT_E2M1)
  using ElementQ = cutlass::float_e2m1_t;
  using ElementK = cutlass::float_e2m1_t;
  using ElementV = cutlass::bfloat16_t;
#else
  using ElementQ = bfloat16_t;
  using ElementK = bfloat16_t;
  using ElementV = bfloat16_t;
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
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;

#elif HEAD_DIM == 96
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;

#elif HEAD_DIM == 128
#if !(defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
  using ShapeQK = Shape<_256, _32, _32>;
  using ShapePV = Shape<_256, _32, _32>;
  using ShapeOut = Shape<_256, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#else
  using ShapeQK = Shape<_512, _64, _64>;
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
#endif
#elif HEAD_DIM == 192
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;

#endif
#elif defined(DECODE)

#if (defined(IS_FLOAT_E5M2) || defined(IS_FLOAT_E4M3) || defined(IS_FLOAT_E2M1))
#define KV_TILE_SIZE _256
#else
#define KV_TILE_SIZE _128
#endif

#if HEAD_DIM == 16
  /* Tiny config for testing */
  using KVTileSize = _64;
  using SubgroupsK = _4;
  using PVTileN    = _16;
  using QKTileK    = _16;
  using HeadDimSize = _16;
#elif HEAD_DIM == 64
  using KVTileSize = _64;
  using SubgroupsK = _4;
  using PVTileN    = _64;
  using QKTileK    = _64;
  using HeadDimSize = _64;
#elif HEAD_DIM == 96
  using KVTileSize = _64;
  using SubgroupsK = _4;
  using PVTileN    = _32;
  using QKTileK    = _32;
  using HeadDimSize = _96;
#elif HEAD_DIM == 128
  using KVTileSize = KV_TILE_SIZE;
  using SubgroupsK = _8;
#if defined(IS_BFLOAT16)
  using PVTileN    = _64;
#else
  using PVTileN    = _128;
#endif
  using QKTileK    = _128;
  using HeadDimSize = _128;

  using ShapeQK4  = Shape<_4, _256, _128>;
  using ShapePV4  = Shape<_4, _64, _256>;
  using ShapeOut4 = Shape<_4, _128>;
  using SubgroupLayoutQK4 = Layout<Shape<_1, _8, _1>>;

  using ShapeQK4KV128 = Shape<_4, _128, _128>;
  using ShapePV4KV128 = Shape<_4, _64, _128>;

#elif HEAD_DIM == 192
  using KVTileSize = _128;
  using SubgroupsK = _8;
  using PVTileN    = _32;
  using QKTileK    = _64;
  using HeadDimSize = _192;
#endif

  using ShapeQK8  = Shape<_8, KVTileSize, QKTileK>; // (q,k,d)
  using ShapePV8  = Shape<_8, PVTileN, KVTileSize>; // (q,v,k)
  using ShapeOut8 = Shape<_8,  HeadDimSize>;        // (q,v)
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

  using ShapeQK32  = Shape<_32, KVTileSize, QKTileK>;
  using ShapePV32  = Shape<_32, PVTileN, KVTileSize>;
  using ShapeOut32 = Shape<_32, HeadDimSize>;
  using SubgroupLayoutQK32 = Layout<Shape<_4, SubgroupsK, _1>>;

#if (defined(IS_FLOAT_E5M2) || defined(IS_FLOAT_E4M3) || defined(IS_FLOAT_E2M1))
  using KVTileSizeLargeRows = _128;
#else
  using KVTileSizeLargeRows = _64;
#endif

  using ShapeQK40  = Shape<_40, KVTileSizeLargeRows, QKTileK>;
  using ShapePV40  = Shape<_40, PVTileN, KVTileSizeLargeRows>;
  using ShapeOut40 = Shape<_40, HeadDimSize>;
  using SubgroupLayoutQK40 = Layout<Shape<_5, _4, _1>>;

  using ShapeQK48  = Shape<_48, KVTileSizeLargeRows, QKTileK>;
  using ShapePV48  = Shape<_48, PVTileN, KVTileSizeLargeRows>;
  using ShapeOut48 = Shape<_48, HeadDimSize>;
  using SubgroupLayoutQK48 = Layout<Shape<_6, _4, _1>>;

  using ShapeQK64  = Shape<_64, _64, QKTileK>;
  using ShapePV64  = Shape<_64, PVTileN, _64>;
  using ShapeOut64 = Shape<_64, HeadDimSize>;
  using SubgroupLayoutQK64 = Layout<Shape<_8, _1, _1>>;

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

  const int kv_tile    = int(KVTileSize::value);
  const int kv_blocks  = (options.seq_len_kv + kv_tile - 1) / kv_tile
                       + (options.seq_len_kv_cache + kv_tile - 1) / kv_tile;
  const int base_units = options.batch * options.num_heads_kv;
  const int saturation_cores_default = estimate_saturation_cores(base_units, kv_blocks);
  const int saturation_cores = cutlass::fmha::kernel::fmha_split_saturation_cores(saturation_cores_default);
#if HEAD_DIM == 128 && defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
  const bool short_cache_q8_candidate = options.seq_len_qo <= 8 &&
                                        options.seq_len_kv_cache > 0 &&
                                        options.seq_len_kv_cache <= 1024;
#else
  const bool short_cache_q8_candidate = false;
#endif
  const bool can_use_dynamic_split = total_rows <= 64
                                  && base_units < saturation_cores;
  const bool split_short_cache_q_rows = can_use_dynamic_split
                                     && short_cache_q8_candidate;
  const bool split_large_kv_work = can_use_dynamic_split
                                && base_units * kv_blocks > saturation_cores;
  const bool use_dynamic_split = split_short_cache_q_rows || split_large_kv_work;

  #define FMHA_RUN_DYNAMIC(CAUSAL, QK, PV, OUT, SGL)                                    \
    FMHAConfig<CAUSAL, false, QK, PV, OUT, SGL, void, PipelineStages,                           \
               ElementQ, ElementK, ElementV, float, /*kGqaFusion=*/false>::                    \
               template run<false, true,                                                       \
               cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>(options)

  #define FMHA_RUN_INDIVIDUAL_Q(CAUSAL, QK, PV, OUT, SGL)                      \
    FMHAConfig<CAUSAL, false, QK, PV, OUT, SGL, void, PipelineStages,           \
               ElementQ, ElementK, ElementV, float, /*kGqaFusion=*/false>::     \
               template run<false, true,                                       \
               cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>>(options)

  #define FMHA_RUN_Q(QK, PV, OUT, SGL)                                                 \
    (use_dynamic_split                                                                        \
       ? (options.is_causal                                                                           \
           ? FMHA_RUN_DYNAMIC(true, QK, PV, OUT, SGL)                                           \
           : FMHA_RUN_DYNAMIC(false, QK, PV, OUT, SGL))                                         \
       : (options.is_causal                                                                           \
           ? FMHAConfig</*CausalMask=*/true,  false, QK, PV, OUT, SGL, void, PipelineStages,          \
                        ElementQ, ElementK, ElementV, float, /*kGqaFusion=*/true>::template run<      \
                        false, true, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>>(options) \
           : FMHAConfig</*CausalMask=*/false, false, QK, PV, OUT, SGL, void, PipelineStages,          \
                        ElementQ, ElementK, ElementV, float, /*kGqaFusion=*/true>::template run<      \
                        false, true, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>>(options)))

#if HEAD_DIM == 128 && defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
#if defined(IS_BFLOAT16)
  const int q_row_work = options.batch * options.num_heads_q * options.seq_len_qo;
  if (split_short_cache_q_rows && q_row_work >= 120)
    return options.is_causal
      ? FMHA_RUN_INDIVIDUAL_Q(
          true, ShapeQK4KV128, ShapePV4KV128, ShapeOut4, SubgroupLayoutQK4)
      : FMHA_RUN_INDIVIDUAL_Q(
          false, ShapeQK4KV128, ShapePV4KV128, ShapeOut4, SubgroupLayoutQK4);
#endif
  if (split_short_cache_q_rows)
    return options.is_causal
      ? FMHA_RUN_INDIVIDUAL_Q(true, ShapeQK4, ShapePV4, ShapeOut4, SubgroupLayoutQK4)
      : FMHA_RUN_INDIVIDUAL_Q(false, ShapeQK4, ShapePV4, ShapeOut4, SubgroupLayoutQK4);
#endif

  if (total_rows <= 8)
    return FMHA_RUN_Q(ShapeQK8, ShapePV8, ShapeOut8, SubgroupLayoutQK8);
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
  #undef FMHA_RUN_INDIVIDUAL_Q
  #undef FMHA_RUN_DYNAMIC
#else
  // Directly instantiate only CachedKV=true kernels.
  using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>;

#if HEAD_DIM == 128 && defined(PREFILL) && !(defined(IS_FLOAT_E5M2) || defined(IS_FLOAT_E4M3)) && (defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
  // CRI causal: adaptive Q tile selection to ensure >=2 waves. If too few WGs
  // with BLK_Q=256, use BLK_Q=128 for more waves and finer scheduling granularity.
  using FMHACausal     = FMHAConfig<true, false, ShapeQK_Causal, ShapePV_Causal, ShapeOut_Causal, SubgroupLayoutQK_Causal, void, PipelineStages, ElementQ, ElementK, ElementV>;
  using FMHANonCausal  = FMHAConfig<false, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, ElementQ, ElementK, ElementV>;
  using FMHACausal4    = FMHAConfig<true, false, ShapeQK4, ShapePV4, ShapeOut4, SubgroupLayoutQK4, void, PipelineStages, ElementQ, ElementK, ElementV>;
  using FMHANonCausal4 = FMHAConfig<false, false, ShapeQK4, ShapePV4, ShapeOut4, SubgroupLayoutQK4, void, PipelineStages, ElementQ, ElementK, ElementV>;

  const int num_xe_cores = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  int num_q_tiles_256 = (options.seq_len_qo + 255) / 256;
  int total_wgs_256 = num_q_tiles_256 * options.num_heads_q * options.batch;
  bool use_small = options.seq_len_qo < 512 || total_wgs_256 < 2 * num_xe_cores;

  if (options.is_causal) {
    if (use_small) {
      return FMHACausal4::template run<false, true, Scheduler>(options);
    }
    return FMHACausal::template run<false, true, Scheduler>(options);
  } else {
    if (options.seq_len_qo < 512) {
      return FMHANonCausal4::template run<false, true, Scheduler>(options);
    }
    return FMHANonCausal::template run<false, true, Scheduler>(options);
  }
#else
  using FMHACausal    = FMHAConfig<true, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, ElementQ, ElementK, ElementV>;
  using FMHANonCausal = FMHAConfig<false, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, ElementQ, ElementK, ElementV>;

  if (options.is_causal) {
    return FMHACausal::template run<false, true, Scheduler>(options);
  } else {
    return FMHANonCausal::template run<false, true, Scheduler>(options);
  }
#endif
#endif
}
