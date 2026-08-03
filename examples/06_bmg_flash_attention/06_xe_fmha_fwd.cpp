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
#if defined(IS_FLOAT_E5M2) || defined(IS_FLOAT_E4M3)
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

  // Best-known short-path tile for 64/8 GQA at 128x128 (prefill, TARGET==35).
  using ShapeQK8 = Shape<_64, _32, _64>;
  using ShapePV8 = Shape<_64, _64, _32>;
  using ShapeOut8 = Shape<_64, _128>;
  using SubgroupLayoutQK8 = Layout<Shape<_8, _1, _1>>;

  // Keep a separate small-path tile for broader short-sequence coverage.
  using ShapeQK4 = Shape<_128, _64, _64>;
  using ShapePV4 = Shape<_128, _64, _64>;
  using ShapeOut4 = Shape<_128, _128>;
  using SubgroupLayoutQK4 = Layout<Shape<_8, _1, _1>>;
#endif
#elif HEAD_DIM == 192
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _192>;
#if !(defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
#else
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#endif

#elif HEAD_DIM == 256
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _256>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;

#endif
#elif defined(DECODE)

#define KV_TILE_SIZE _256

#if HEAD_DIM == 16
  /* Tiny config for testing */
  using PVTileN  = _16;
  using QKTileK    = _16;
  using HeadDimSize = _16;
#elif HEAD_DIM == 64
  using PVTileN  = _32;
  using QKTileK    = _64;
  using HeadDimSize = _64;
#elif HEAD_DIM == 96
  using PVTileN  = _32;
  using QKTileK    = _32;
  using HeadDimSize = _96;
#elif HEAD_DIM == 128
#if defined(IS_FLOAT_E5M2) || defined(IS_FLOAT_E4M3)
  using PVTileN  = _64;
#else
  using PVTileN  = _32;
#endif
  using QKTileK    = _64;
  using HeadDimSize = _128;
#elif HEAD_DIM == 192
  using PVTileN  = _32;
  using QKTileK    = _64;
  using HeadDimSize = _192;
#endif

  using ShapeQK8  = Shape<_8,  KV_TILE_SIZE, QKTileK>;   // (q,k,d)
  using ShapePV8  = Shape<_8,  PVTileN, KV_TILE_SIZE>; // (q,v,k)
  using ShapeOut8 = Shape<_8,  HeadDimSize>;        // (q,v)
  using SubgroupLayoutQK8  = Layout<Shape<_1, _8, _1>>;

  using ShapeQK16  = Shape<_16, KV_TILE_SIZE, QKTileK>;
  using ShapePV16  = Shape<_16, PVTileN, KV_TILE_SIZE>;
  using ShapeOut16 = Shape<_16, HeadDimSize>;
  using SubgroupLayoutQK16 = Layout<Shape<_2, _8, _1>>;

  using ShapeQK32  = Shape<_32, KV_TILE_SIZE, QKTileK>;
  using ShapePV32  = Shape<_32, PVTileN, KV_TILE_SIZE>;
  using ShapeOut32 = Shape<_32, HeadDimSize>;
  using SubgroupLayoutQK32 = Layout<Shape<_4, _8, _1>>;

  using ShapeQK64  = Shape<_64, _64, QKTileK>;
  using ShapePV64  = Shape<_64, PVTileN, _64>;
  using ShapeOut64 = Shape<_64, HeadDimSize>;
  using SubgroupLayoutQK64 = Layout<Shape<_8, _1, _1>>;

#else
#error Either DECODE or PREFILL should be defined.
#endif

#ifdef DECODE
  constexpr int PipelineStages = 1;
#elif !(defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
  // BMG (Xe20) prefill: the mainloop is L1/memory-pipe sensitive. Together with the
  // after-GEMM2 K-prefetch (see xe_fmha_fwd_mainloop.hpp), a shallower prefetch depth
  // keeps the K/V L1 footprint small. Stages=1 is the measured optimum on BMG.
  constexpr int PipelineStages = 1;
#else
  constexpr int PipelineStages = 2;
#endif

#if defined(DECODE)
  const int gqa_group = options.num_heads_q / options.num_heads_kv;
  const int total_rows = options.seq_len_qo * options.num_heads_q / options.num_heads_kv;
  const int total_kv_length = options.seq_len_kv + options.seq_len_kv_cache;
  const int base_units = options.batch * options.num_heads_kv;

  bool use_two_kernel_split = options.num_kv_splits > 0;
  if (!use_two_kernel_split && !options.varlen && !options.use_paged_kv &&
      options.seq_len_kv_cache == 0 && (!options.is_causal || options.seq_len_qo == 1)) {
#if defined(IS_FLOAT_E5M2) || defined(IS_FLOAT_E4M3)
    use_two_kernel_split =
        total_kv_length <= 10240 ||
    (total_kv_length <= 65536 && options.batch == 1) ||
        (total_kv_length > 65536 && base_units < 8);
#elif defined(IS_BFLOAT16)
    use_two_kernel_split =
        total_kv_length <= 1024 ||
        (total_kv_length <= 10240 && base_units < 6) ||
        (total_kv_length <= 32768 && base_units == 1) ||
        (total_kv_length > 32768 && gqa_group <= 4);
#endif
  }

  const int kv_tile = int(KV_TILE_SIZE::value);
  const int kv_blocks = cute::ceil_div(total_kv_length, kv_tile);
  const int saturation_cores_default = estimate_saturation_cores(base_units, kv_blocks);
  const bool use_dynamic_split =
      !use_two_kernel_split && total_rows <= 64 &&
      base_units < cutlass::fmha::kernel::fmha_split_saturation_cores(saturation_cores_default);

  auto select_num_kv_splits = [&](int q_tile) {
    if (options.num_kv_splits > 0) {
      return options.num_kv_splits;
    }

    const int q_tiles = cute::ceil_div(total_rows, q_tile);
    const int base_work_groups = options.batch * options.num_heads_kv * q_tiles;
    const int xe_cores = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
#if defined(IS_FLOAT_E5M2) || defined(IS_FLOAT_E4M3)
    const int target_parallel_rows = cute::max(8, xe_cores * 5);
#else
    const int target_parallel_rows = cute::max(8, xe_cores * 8);
#endif
    const int base_parallel_rows = base_work_groups * q_tile;
    const int parallel_splits = cute::max(1, cute::ceil_div(target_parallel_rows, base_parallel_rows));
    const int latency_splits = cute::max(1, cute::ceil_div(kv_blocks, 128));
    const int max_useful_splits = cute::max(1, kv_blocks / 16);
    return cute::min(cute::max(parallel_splits, latency_splits), max_useful_splits);
  };

#define FMHA_RUN_TWO_KERNEL_Q(QK, PV, OUT, SGL, Q_TILE)                                              \
  [&]() {                                                                                            \
    Options tuned_options = options;                                                                 \
    tuned_options.num_kv_splits = select_num_kv_splits(Q_TILE);                                      \
    return (tuned_options.is_causal                                                                  \
      ? FMHAConfig</*CausalMask=*/true, false, QK, PV, OUT, SGL, void, PipelineStages,               \
                   ElementQ, ElementK, ElementV, float, /*kGqaFusion=*/false>::template run<         \
                   false, false, false, cutlass::fmha::kernel::XeFHMASplitKVTileScheduler, true>(    \
                   tuned_options)                                                                    \
      : FMHAConfig</*CausalMask=*/false, false, QK, PV, OUT, SGL, void, PipelineStages,              \
                   ElementQ, ElementK, ElementV, float, /*kGqaFusion=*/false>::template run<         \
                   false, false, false, cutlass::fmha::kernel::XeFHMASplitKVTileScheduler, true>(    \
                   tuned_options));                                                                  \
  }()

#define FMHA_RUN_DYNAMIC_OR_NON_SPLIT_Q(QK, PV, OUT, SGL)                                           \
  (use_dynamic_split                                                                                \
     ? (options.is_causal                                                                            \
         ? FMHAConfig</*CausalMask=*/true, false, QK, PV, OUT, SGL, void, PipelineStages,           \
                      ElementQ, ElementK, ElementV, float, /*kGqaFusion=*/false>::template run<     \
                      false, false, false,                                                           \
                      cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>(options)      \
         : FMHAConfig</*CausalMask=*/false, false, QK, PV, OUT, SGL, void, PipelineStages,          \
                      ElementQ, ElementK, ElementV, float, /*kGqaFusion=*/false>::template run<     \
                      false, false, false,                                                           \
                      cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>(options))     \
     : (options.is_causal                                                                            \
         ? FMHAConfig</*CausalMask=*/true, false, QK, PV, OUT, SGL, void, PipelineStages,           \
                      ElementQ, ElementK, ElementV, float, /*kGqaFusion=*/true>::template run<      \
                      false, false, false,                                                           \
                      cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>>(options)              \
         : FMHAConfig</*CausalMask=*/false, false, QK, PV, OUT, SGL, void, PipelineStages,          \
                      ElementQ, ElementK, ElementV, float, /*kGqaFusion=*/true>::template run<      \
                      false, false, false,                                                           \
                      cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>>(options)))

  // BMG (Xe20): an M=32/64 tile paired with the 256-wide KV tile needs 32 sub-groups and
  // overruns the 256-GRF budget at hdim128, so the launch fails with
  // UR_RESULT_ERROR_OUT_OF_RESOURCES. CRI/Xe3p has the registers for the full ladder.
  // Capping M only costs scheduling granularity, not coverage: the GqaFusion kernel loops
  // ceil_div(total_rows, QK_BLK_M) times over the packed rows.
  if (!use_two_kernel_split) {
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
    if (total_rows <= 8)
      return FMHA_RUN_DYNAMIC_OR_NON_SPLIT_Q(ShapeQK8, ShapePV8, ShapeOut8, SubgroupLayoutQK8);
    else if (total_rows <= 16)
      return FMHA_RUN_DYNAMIC_OR_NON_SPLIT_Q(ShapeQK16, ShapePV16, ShapeOut16, SubgroupLayoutQK16);
    else if (total_rows <= 32)
      return FMHA_RUN_DYNAMIC_OR_NON_SPLIT_Q(ShapeQK32, ShapePV32, ShapeOut32, SubgroupLayoutQK32);
    else
      return FMHA_RUN_DYNAMIC_OR_NON_SPLIT_Q(ShapeQK64, ShapePV64, ShapeOut64, SubgroupLayoutQK64);
#else
    if (total_rows <= 8)
      return FMHA_RUN_DYNAMIC_OR_NON_SPLIT_Q(ShapeQK8, ShapePV8, ShapeOut8, SubgroupLayoutQK8);
    else if (total_rows <= 16)
      return FMHA_RUN_DYNAMIC_OR_NON_SPLIT_Q(ShapeQK16, ShapePV16, ShapeOut16, SubgroupLayoutQK16);
#if HEAD_DIM >= 128
    else if (total_rows <= 32)
      return FMHA_RUN_DYNAMIC_OR_NON_SPLIT_Q(ShapeQK64, ShapePV64, ShapeOut64, SubgroupLayoutQK64);
#else
    else if (total_rows <= 32)
      return FMHA_RUN_DYNAMIC_OR_NON_SPLIT_Q(ShapeQK32, ShapePV32, ShapeOut32, SubgroupLayoutQK32);
#endif
    else
      return FMHA_RUN_DYNAMIC_OR_NON_SPLIT_Q(ShapeQK64, ShapePV64, ShapeOut64, SubgroupLayoutQK64);
#endif
  }

  // Same BMG M-cap as above; the split-KV scheduler packs GQA and query positions into the
  // Q grid dimension, so a smaller tile just yields more work-groups.
#if !(defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
  if (total_rows <= 8)
    return FMHA_RUN_TWO_KERNEL_Q(ShapeQK8, ShapePV8, ShapeOut8, SubgroupLayoutQK8, 8);
  else if (total_rows <= 16)
    return FMHA_RUN_TWO_KERNEL_Q(ShapeQK16, ShapePV16, ShapeOut16, SubgroupLayoutQK16, 16);
#if HEAD_DIM >= 128
  // BMG hdim128: M=32 tile (SubgroupLayoutQK32 = <4,8,1> = 32 SGs, N=256) overruns
  // the 256-GRF budget at launch; skip that rung and fall through to M=64.
  else
    return FMHA_RUN_TWO_KERNEL_Q(ShapeQK64, ShapePV64, ShapeOut64, SubgroupLayoutQK64, 64);
#else
  else if (total_rows <= 32)
    return FMHA_RUN_TWO_KERNEL_Q(ShapeQK32, ShapePV32, ShapeOut32, SubgroupLayoutQK32, 32);
  else
    return FMHA_RUN_TWO_KERNEL_Q(ShapeQK64, ShapePV64, ShapeOut64, SubgroupLayoutQK64, 64);
#endif
#elif defined(IS_FLOAT_E5M2) || defined(IS_FLOAT_E4M3)
  if (total_rows <= 8)
    return FMHA_RUN_TWO_KERNEL_Q(ShapeQK8, ShapePV8, ShapeOut8, SubgroupLayoutQK8, 8);
  else if (total_rows <= 31)
    return FMHA_RUN_TWO_KERNEL_Q(ShapeQK16, ShapePV16, ShapeOut16, SubgroupLayoutQK16, 16);
  else
    return FMHA_RUN_TWO_KERNEL_Q(ShapeQK32, ShapePV32, ShapeOut32, SubgroupLayoutQK32, 32);
#else
  if (total_rows <= 8)
    return FMHA_RUN_TWO_KERNEL_Q(ShapeQK8, ShapePV8, ShapeOut8, SubgroupLayoutQK8, 8);
  else if (total_rows <= 16)
    return FMHA_RUN_TWO_KERNEL_Q(ShapeQK16, ShapePV16, ShapeOut16, SubgroupLayoutQK16, 16);
  else
    return FMHA_RUN_TWO_KERNEL_Q(ShapeQK32, ShapePV32, ShapeOut32, SubgroupLayoutQK32, 32);
#endif

  #undef FMHA_RUN_DYNAMIC_OR_NON_SPLIT_Q
  #undef FMHA_RUN_TWO_KERNEL_Q
#else
#if HEAD_DIM == 128 && defined(PREFILL) && (defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
  if (options.seq_len_kv_cache > 0 || options.use_paged_kv) {
    std::cerr << "Error: CachedKV/PagedKV requested. Use the cached_kv binary." << std::endl;
    return -1;
  }

  // Causal scheduler with DisablePrefetchV=true, used only for small tiles.
  using CausalSmallScheduler =
    cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false, true, false, true>;
  // Causal scheduler without DisablePrefetchV, used for large tiles.
  using CausalScheduler =
    cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false, false, true, false, false>;
  using DefaultScheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>;

  using FMHACausal    = FMHAConfig<true, false, ShapeQK_Causal, ShapePV_Causal, ShapeOut_Causal, SubgroupLayoutQK_Causal, void, PipelineStages, ElementQ, ElementK, ElementV>;
  using FMHANonCausal = FMHAConfig<false, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, ElementQ, ElementK, ElementV>;

  using FMHACausal8    = FMHAConfig<true, false, ShapeQK8, ShapePV8, ShapeOut8, SubgroupLayoutQK8, void, 1, ElementQ, ElementK, ElementV>;
  using FMHANonCausal8 = FMHAConfig<false, false, ShapeQK8, ShapePV8, ShapeOut8, SubgroupLayoutQK8, void, 1, ElementQ, ElementK, ElementV>;

  using FMHACausal4    = FMHAConfig<true, false, ShapeQK4, ShapePV4, ShapeOut4, SubgroupLayoutQK4, void, 1, ElementQ, ElementK, ElementV>;
  using FMHANonCausal4 = FMHAConfig<false, false, ShapeQK4, ShapePV4, ShapeOut4, SubgroupLayoutQK4, void, 1, ElementQ, ElementK, ElementV>;

  // Adaptive smaller Q tile to ensure >=2 waves: if too few WGs with BLK_Q=256,
  // use BLK_Q=128 for more waves and finer scheduling granularity.
  const int num_xe_cores = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  int num_q_tiles_256 = (options.seq_len_qo + 255) / 256;
  int total_wgs_256 = num_q_tiles_256 * options.num_heads_q * options.batch;
  // These magic numbers are from an empirical fit based on observed performance data.
  bool use_small = (options.seq_len_qo - 576) * options.num_heads_q <= 11840 || total_wgs_256 < 2 * num_xe_cores;

  if (options.is_causal) {
    if (use_small) {
      if (options.varlen) {
        return FMHACausal4::template run<true, false, false, CausalSmallScheduler>(options);
      } else {
        return FMHACausal8::template run<false, false, false, CausalSmallScheduler>(options);
      }
    }
    if (options.varlen) {
      return FMHACausal::template run<true, false, false, CausalScheduler>(options);
    } else {
      return FMHACausal::template run<false, false, false, DefaultScheduler>(options);
    }
  } else {
    if (options.seq_len_qo < 512) {
      if (options.varlen) {
        return FMHANonCausal8::template run<true, false, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false,false,false,false,true>>(options);
      } else {
        return FMHANonCausal8::template run<false, false, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<false,false,false,false,true>>(options);
      }
    }
    if (options.varlen) {
      return FMHANonCausal::template run<true, false, false, DefaultScheduler>(options);
    } else {
      return FMHANonCausal::template run<false, false, false, DefaultScheduler>(options);
    }
  }
#else
  if (options.seq_len_kv_cache > 0 || options.use_paged_kv) {
    std::cerr << "Error: CachedKV/PagedKV requested. Use the cached_kv binary." << std::endl;
    return -1;
  }

  using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>;

  using FMHACausal    = FMHAConfig<true, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, ElementQ, ElementK, ElementV>;
  using FMHANonCausal = FMHAConfig<false, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, ElementQ, ElementK, ElementV>;

  if (options.is_causal) {
    if (options.varlen) {
      return FMHACausal::template run<true, false, false, Scheduler>(options);
    } else {
      return FMHACausal::template run<false, false, false, Scheduler>(options);
    }
  } else {
    if (options.varlen) {
      return FMHANonCausal::template run<true, false, false, Scheduler>(options);
    } else {
      return FMHANonCausal::template run<false, false, false, Scheduler>(options);
    }
  }
#endif
#endif
}
