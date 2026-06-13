/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

////////////////////////////////////////////////////////////////////////////////////////////////////
/// XE4 GEMM with ADMA Prefetch and ADMA Store-Reduce
///
/// This example demonstrates two XE4 ADMA extensions to the standard GEMM pipeline:
///
///   1. ADMA PREFETCH in mainloop load:
///      - Before each K-tile ADMA load, the load warp fires a prefetch for the NEXT K-tile
///      - Uses cute::prefetch(adma_load, src) — fire-and-forget, no barrier, no SLM target
///      - Warms the L2 cache so the subsequent ADMA load hits L2 instead of going to memory
///      - The prefetch is derived from the load atom's tensor descriptor (no extra tdesc alloc)
///
///   2. ADMA STORE_REDUCE in epilogue:
///      - Replaces XE4_ADMA_STORE with XE4_ADMA_STORE_REDUCE<T, RedOp::Add> for the D output
///      - Atomically reduces (adds) epilogue results into the existing D matrix in gmem
///      - Enables gradient accumulation, split-K reduction, and multi-pass fusion patterns
///      - Uses async_tensor_fred (float) or async_tensor_ired (integer) asm instructions
///
/// Architecture:
///   The kernel reuses the standard GemmUniversal 5-warp structure:
///     Warp 0: MMA compute        Warp 1: Tile scheduler
///     Warp 2: Mainloop load (+prefetch)   Warp 3: Epilogue load
///     Warps 4+: Epilogue store (+reduce)
///
///   The only differences from the standard xe4_gemm are:
///     - Mainloop load warp issues cute::prefetch() before each ADMA load
///     - Epilogue store uses STORE_REDUCE instead of plain STORE
///
/// Test configurations:
///   - GEMM_PREFETCH: Standard GEMM with prefetch-ahead (TBX-safe, validates correctness)
///   - GEMM_PREFETCH_REDUCE: GEMM with prefetch + reduce-Add (requires real XE4 hardware)
///     Pre-initializes D=0 so reduce-Add produces same result as normal store.
///
/// IMPORTANT: STORE_REDUCE tests require real XE4 hardware.
/// The TBX simulator does not support the send.dma.tensor.reduce_l2g instruction variant.
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cute/tensor.hpp>
#include <cute/algorithm/prefetch.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/cluster_launch.hpp>
#include <sycl/sycl.hpp>

#include "validation.hpp"

using namespace cute;
using namespace sycl;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Epilogue builder variant that uses XE4_ADMA_STORE_REDUCE instead of XE4_ADMA_STORE.
///
/// This is a direct analog of cutlass::epilogue::collective::detail::Xe4AdmaBuilderImpl
/// but with CopyOpS2G = XE4_ADMA_STORE_REDUCE<SmemElementD, RedOp::Add, BarrierType::Abarrier>
/// so the epilogue D store atomically reduces into gmem instead of overwriting.
///
/// All other configuration (StagesC, SmemLayoutAtomC/D, CopyOpG2S, fusion callbacks, etc.)
/// is identical to the standard Xe4AdmaBuilderImpl.
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective::detail {

template <
  class OpClass,
  class MmaTileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class FusionOpOrCallbacks,
  cute::RedOp Rop = cute::RedOp::Add,
  cute::BarrierType BType = cute::BarrierType::Abarrier
>
struct Xe4AdmaReduceBuilderImpl {
private:
  static constexpr int StagesC = 2;
  static constexpr int StagesD = 1;
  static constexpr bool ReuseSmemC = false;
  static constexpr bool DelayTmaStore = false;
  static constexpr int NumControlWarps = 4;
  static constexpr int NumEpilogueWarps = 16;
  static constexpr int EpilogueWarpTileN = 32;
  static constexpr int FragmentSize = 32 / sizeof(ElementD);

  static constexpr bool DisableSource = cute::is_void_v<ElementC_>;
  using ElementC = cute::conditional_t<DisableSource, ElementD, ElementC_>;
  using GmemLayoutTagC = cute::conditional_t<DisableSource, GmemLayoutTagD, GmemLayoutTagC_>;
  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

  constexpr static bool is_fp_postop = is_floating_t<ElementD>::value && (sizeof_bits_v<ElementD> < 16);
  constexpr static bool is_int8_postop = is_integral<ElementD>::value && (sizeof_bits_v<ElementD> == 8);
  using ElementImm = cute::conditional_t<is_fp_postop, bf16, cute::conditional_t<is_int8_postop, int32_t, ElementD>>;

  using CtaTileShape_MNK = MmaTileShape_MNK;
  using TileShape_MN = decltype(select<0,1>(MmaTileShape_MNK{}));

  // SmemElementD is the unpacked SLM element type — same as ElementD for fp16/float
  using SmemElementD = typename cutlass::detail::get_unpacked_element_type<ElementD>::type;

  static constexpr auto
  epilogue_tile() {
    using namespace cute;
    if constexpr (not is_same_v<EpilogueTileType, EpilogueTileAuto>) {
      static_assert(is_tuple_v<EpilogueTileType>, "Shape or Tile");
      return EpilogueTileType{};
    }
    else {
      constexpr int WarpSizeM = cutlass::NumThreadsPerWarp;
      constexpr int ElementsPerWarpN = 32;
      constexpr int TileShapeM = size<0>(CtaTileShape_MNK{});
      constexpr int TileShapeN = size<1>(CtaTileShape_MNK{});

      constexpr int WarpsAlongM = TileShapeM / WarpSizeM;
      constexpr int WarpsAlongN = TileShapeN / ElementsPerWarpN;

      constexpr int NumWarpsAlongN = cute::min(WarpsAlongN, NumEpilogueWarps);
      constexpr int NumWarpsAlongM = cute::min(WarpsAlongM, NumEpilogueWarps / NumWarpsAlongN);
      constexpr int EpilogueTileM = NumWarpsAlongM * WarpSizeM;
      constexpr int EpilogueTileN = NumWarpsAlongN * ElementsPerWarpN;

      return make_tile(Int<EpilogueTileM>{}, Int<EpilogueTileN>{});
    }
  }
  using EpilogueTile = decltype(epilogue_tile());

  using FusionCallbacks = fusion::FusionCallbacks<
    Xe4TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    FusionOpOrCallbacks, CtaTileShape_MNK, EpilogueTile
  >;

  using SmemLayoutAtomC = decltype(make_ordered_layout(EpilogueTile{}, Step<_1, _0>{}));
  using SmemLayoutAtomD = decltype(make_ordered_layout(TileShape_MN{}, Step<_1, _0>{}));

  // KEY DIFFERENCE: Use STORE_REDUCE instead of plain STORE for the D output.
  // This causes the epilogue's copy(adma_store_d.with(abar), smem, gmem) to issue
  // async_tensor_fred/ired instead of async_tensor_copy, atomically reducing SLM data
  // into gmem D with the specified reduction operation.
  using CopyOpS2G = cute::XE4_ADMA_STORE_REDUCE<SmemElementD, Rop, BType>;
  using CopyOpG2S = XE4_ADMA_LOAD;

public:
  using CollectiveOp =
    cutlass::epilogue::collective::CollectiveEpilogue<
      Xe4AdmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore, NumControlWarps, NumEpilogueWarps>,
      CtaTileShape_MNK,
      EpilogueTile,
      ElementC_,
      GmemStrideTypeC,
      ElementD,
      GmemStrideTypeD,
      FusionCallbacks,
      CopyOpG2S,
      SmemLayoutAtomC,
      decltype(xe4_get_smem_load_op<EpilogueWarpTileN, ElementD>()),
      decltype(xe4_get_smem_load_op<EpilogueWarpTileN, ElementImm>()),
      CopyOpS2G,
      SmemLayoutAtomD,
      decltype(xe4_get_smem_store_op<EpilogueWarpTileN, ElementD>()),
      void
    >;
};

} // namespace cutlass::epilogue::collective::detail

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Custom GemmUniversal kernel with ADMA prefetch in the mainloop load path.
///
/// This is a thin extension of the standard GemmUniversal<..., KernelTmaWarpSpecializedXe4> kernel.
/// The only modification is in the MainloopLoad warp: before each K-tile ADMA load, it issues
/// cute::prefetch() for the NEXT K-tile to warm the L2 cache.
///
/// The prefetch is derived from the load atom's existing tensor descriptor via the
/// CopyOp::PREFETCH typedef — no additional tensor descriptor allocation is needed.
///
/// Template parameter EnablePrefetch controls whether prefetch is active (true) or
/// the kernel behaves identically to the standard GemmUniversal (false).
///
/// Template parameter EnableReduce controls whether the epilogue uses STORE_REDUCE (true)
/// or standard STORE (false). When true, the CollectiveEpilogue must be configured with
/// XE4_ADMA_STORE_REDUCE as its CopyOpS2G.
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  bool EnablePrefetch_ = true
>
class GemmPrefetchReduce {
public:
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  static constexpr bool EnablePrefetch = EnablePrefetch_;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using SmemLayoutA = typename CollectiveMainloop::SmemLayoutA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using SmemLayoutB = typename CollectiveMainloop::SmemLayoutB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementD;
  using StrideC = typename CollectiveEpilogue::StrideD;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  // Pipeline depths
  static constexpr uint32_t SchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  static constexpr uint32_t AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;

  // TileID scheduler
  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;
  using TileSchedulerTag = void;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileSchedulerTag, ArchTag, TileShape, ClusterShape, SchedulerPipelineStageCount>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;

  // Warp counts
  static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp;
  static constexpr uint32_t NumMMAThreads          = NumThreadsPerWarp;
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp;
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp;
  static constexpr uint32_t NumEpilogueThreads     = CollectiveEpilogue::ThreadCount;

  static constexpr uint32_t MaxThreadsPerBlock = NumSchedThreads +
                                                 NumMainloopLoadThreads + NumMMAThreads +
                                                 NumEpilogueLoadThreads + NumEpilogueThreads;

  // Pipeline types
  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using MainloopPipelineState = typename CollectiveMainloop::MainloopPipelineState;
  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  using EpiLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;
  // G2S + Fred pipelines exist on the epilogue regardless of HasSkReduce; this DP-only kernel
  // never drives them, but the shared epilogue load()/store() destructure 5-element pipeline
  // tuples, so they must be declared, constructed, and threaded through to match the signature.
  using EpiG2SPipeline = typename CollectiveEpilogue::G2SPipeline;
  using EpiG2SPipelineState = typename CollectiveEpilogue::G2SPipelineState;
  using EpiFredPipeline = typename CollectiveEpilogue::FredPipeline;
  using EpiFredPipelineState = typename CollectiveEpilogue::FredPipelineState;
  using AccumulatorPipeline = cutlass::PipelineTmaAsync<AccumulatorPipelineStageCount>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;
  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;
  using EpiWaveOrderBarrier = typename CollectiveEpilogue::WaveOrderBarrier;
  using CLCPipeline = cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>;
  using CLCPipelineState = typename CLCPipeline::PipelineState;

  // Shared storage
  struct SharedStorage {
    struct TensorStorage {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
      typename TileScheduler::CLCResponse clc_response[SchedulerPipelineStageCount];
    } tensors;

    struct PipelineStorage {
      using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;
      using EpiLoadPipelineStorage = typename EpiLoadPipeline::SharedStorage;
      using EpiG2SPipelineStorage = typename EpiG2SPipeline::SharedStorage;
      using EpiFredPipelineStorage = typename EpiFredPipeline::SharedStorage;
      using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;
      using EpiStorePipelineStorage = typename EpiStorePipeline::SharedStorage;
      using EpiWaveOrderBarrierStorage = typename EpiWaveOrderBarrier::SharedStorage;
      using CLCPipelineStorage = typename CLCPipeline::SharedStorage;
      MainloopPipelineStorage mainloop;
      EpiLoadPipelineStorage epi_load;
      EpiG2SPipelineStorage epi_g2s;
      EpiFredPipelineStorage epi_fred;
      AccumulatorPipelineStorage accumulator;
      EpiStorePipelineStorage epi_store;
      EpiWaveOrderBarrierStorage epi_wave_order;
      CLCPipelineStorage clc;
    } pipelines;
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  static constexpr int TensorStorageSize = sizeof(typename SharedStorage::TensorStorage);

  struct Arguments {
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueArguments epilogue;
    TileSchedulerArguments scheduler{};
  };

  struct Params {
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    KernelHardwareInfo hw_info;
    TileSchedulerParams scheduler;
  };

  enum class WarpCategory : int32_t {
    MMA          = 0,
    Sched        = 1,
    MainloopLoad = 2,
    EpilogueLoad = 3,
    Epilogue     = 4
  };

  struct IsParticipant {
    uint32_t mma       = false;
    uint32_t sched     = false;
    uint32_t main_load = false;
    uint32_t epi_load  = false;
    uint32_t epilogue  = false;
  };

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    int device_id = 0;
    int sm_count = KernelHardwareInfo::query_device_multiprocessor_count(device_id);
    KernelHardwareInfo hw_info{device_id, sm_count, 0};
    return to_underlying_arguments(args, hw_info, workspace);
  }

  static Params
  to_underlying_arguments(Arguments const& args, KernelHardwareInfo const& hw_info, void* workspace) {
    auto problem_shape_MNKL = append<4>(args.problem_shape, Int<1>{});
    return {
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
      hw_info,
      TileScheduler::to_underlying_arguments(
        problem_shape_MNKL, TileShape{}, ClusterShape{},
        hw_info, args.scheduler, workspace
      )
    };
  }

  static dim3
  get_grid_shape(Params const& params) {
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    return TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
  }

  static dim3
  get_block_shape() {
    constexpr uint32_t NumControlWarps = 4;
    constexpr uint32_t NumEpilogueWarps = CollectiveEpilogue::ThreadCount / NumThreadsPerWarp;
    return dim3(1, NumControlWarps + NumEpilogueWarps, NumThreadsPerWarp);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// load_with_prefetch — Mainloop load with L2 prefetch-ahead
  ///
  /// Before each K-tile ADMA load, issues cute::prefetch() for the NEXT K-tile.
  /// The prefetch is fire-and-forget (no barrier, no SLM destination) and reuses
  /// the load atom's tensor descriptor via the CopyOp::PREFETCH typedef.
  ///
  /// This overlaps L2 cache warming with the current tile's pipeline acquire,
  /// so the next tile's ADMA load hits L2 cache instead of going to HBM.
  ///
  /// When EnablePrefetch is false, this function degenerates to the standard load().
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  template <class CollMma, class LoadParams, class TileCoordMNKL, class KTileIterator>
  CUTLASS_DEVICE static auto
  load_with_prefetch(
      CollMma& collective_mainloop,
      MainloopPipeline mainloop_pipeline,
      MainloopPipelineState& slm_pipe_write,
      LoadParams const& load_inputs,
      TileCoordMNKL const& cta_coord_mnkl,
      KTileIterator k_tile_iter,
      int k_tile_count)
  {
    auto [mcast_mask_a, mcast_mask_b] = collective_mainloop.cluster_masks_;
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;
    auto [unused_k_tiles, tAgA_mkl, tBgB_nkl, tAsA, tBsB] = load_inputs;

    // Slice out the work coord from partitioned tensors
    Tensor tAgA = tAgA_mkl(_, m_coord / size(typename TiledMma::AtomThrID{}), _, l_coord);
    Tensor tBgB = tBgB_nkl(_, n_coord, _, l_coord);

    // Issue the mainloop loads with prefetch-ahead
    CUTLASS_PRAGMA_UNROLL
    while (k_tile_count > 0) {

      // ─── PREFETCH: warm L2 for the NEXT K-tile (fire-and-forget) ───
      // Issues async_tensor_prefetch.Nd — no barrier, no SLM target.
      // The prefetch uses the load atom's tensor descriptor (via CopyOp::PREFETCH typedef),
      // so no extra tensor descriptor allocation is needed.
      if constexpr (EnablePrefetch) {
        if (k_tile_count > 1) {
          auto next_iter = k_tile_iter;
          ++next_iter;
          cute::prefetch(*collective_mainloop.observed_adma_load_a_, tAgA(_, *next_iter));
          cute::prefetch(*collective_mainloop.observed_adma_load_b_, tBgB(_, *next_iter));
        }
      }

      // ─── ADMA LOAD: gmem → SLM via async tensor copy with arrival barrier ───
      mainloop_pipeline.producer_acquire(slm_pipe_write);
      uint32_t write_stage = slm_pipe_write.index();
      auto abar_prod = mainloop_pipeline.producer_get_barrier(slm_pipe_write);

      copy(collective_mainloop.observed_adma_load_a_->with(abar_prod, mcast_mask_a),
           tAgA(_, *k_tile_iter), tAsA(_, write_stage));
      copy(collective_mainloop.observed_adma_load_b_->with(abar_prod, mcast_mask_b),
           tBgB(_, *k_tile_iter), tBsB(_, write_stage));

      --k_tile_count;
      ++k_tile_iter;
      ++slm_pipe_write;
    }

    return cute::make_tuple(slm_pipe_write, k_tile_iter);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Kernel operator — 5-warp structure with prefetch in load and reduce in epilogue
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  CUTLASS_DEVICE
  void
  operator()(Params const& params) const {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [M, N, K, L] = problem_shape_MNKL;

    uint32_t warp_idx = get_sg_id();
    WarpCategory warp_category = warp_idx < static_cast<int>(WarpCategory::Epilogue)
                                   ? WarpCategory(warp_idx)
                                   : WarpCategory::Epilogue;

    bool lane_predicate = cute::elect_one_sync();
    auto cluster_shape = ClusterShape{};

    auto ptr = alloc_slm_buffer<uint8_t, TensorStorageSize>(item.get_group());
    auto& shared_tensors = *reinterpret_cast<typename SharedStorage::TensorStorage*>(ptr);

    auto tdesc_a = allocate_tdesc<0>();
    auto tdesc_b = allocate_tdesc<1>();
    auto tdesc_c = allocate_tdesc<2>();
    auto tdesc_d = allocate_tdesc<3>();

    auto& shared_pipelines = allocate_abarrier<typename SharedStorage::PipelineStorage>();

    bool is_first_cta_in_cluster = true;
    IsParticipant is_participant = {
      (warp_category == WarpCategory::MMA),
      (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster,
      (warp_category == WarpCategory::MainloopLoad),
      (warp_category == WarpCategory::EpilogueLoad),
      (warp_category == WarpCategory::Epilogue)
    };

    // ─── Pipeline initialization (identical to standard GemmUniversal) ───

    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (WarpCategory::MainloopLoad == warp_category) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::MMA == warp_category) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = lane_predicate && is_participant.main_load;
    mainloop_pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
    mainloop_pipeline_params.num_consumers = 1;
    mainloop_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::MainloopLoad);
    MainloopPipeline mainloop_pipeline(shared_pipelines.mainloop, mainloop_pipeline_params,
                                       cluster_shape, true_type{}, false_type{});

    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (WarpCategory::EpilogueLoad == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = NumEpilogueThreads;
    epi_load_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::EpilogueLoad);
    EpiLoadPipeline epi_load_pipeline(shared_pipelines.epi_load, epi_load_pipeline_params, true_type{});

    // G2S + Fred pipelines: constructed to satisfy the shared epilogue's 5-tuple signature.
    // This DP-only kernel issues no SK splits, so neither is exercised at runtime.
    typename EpiG2SPipeline::Params epi_g2s_pipeline_params;
    if (WarpCategory::EpilogueLoad == warp_category) {
      epi_g2s_pipeline_params.role = EpiG2SPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      epi_g2s_pipeline_params.role = EpiG2SPipeline::ThreadCategory::Consumer;
    }
    epi_g2s_pipeline_params.transaction_bytes = CollectiveEpilogue::TransactionBytesImm;
    epi_g2s_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_g2s_pipeline_params.consumer_arv_count = NumEpilogueThreads;
    epi_g2s_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::EpilogueLoad);
    EpiG2SPipeline epi_g2s_pipeline(shared_pipelines.epi_g2s, epi_g2s_pipeline_params, true_type{});

    typename EpiFredPipeline::Params epi_fred_pipeline_params;
    epi_fred_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::Epilogue);
    epi_fred_pipeline_params.num_producers = NumEpilogueThreads;
    epi_fred_pipeline_params.num_consumers = 1;
    EpiFredPipeline epi_fred_pipeline(shared_pipelines.epi_fred, epi_fred_pipeline_params,
                                      cluster_shape, true_type{}, false_type{});

    typename AccumulatorPipeline::Params accumulator_pipeline_params;
    if (WarpCategory::MMA == warp_category) {
      accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      accumulator_pipeline_params.role = AccumulatorPipeline::ThreadCategory::Consumer;
    }
    accumulator_pipeline_params.is_leader = lane_predicate && is_participant.mma;
    accumulator_pipeline_params.num_consumers = NumEpilogueThreads;
    accumulator_pipeline_params.transaction_bytes = 1;
    accumulator_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::MMA);
    AccumulatorPipeline accumulator_pipeline(shared_pipelines.accumulator, accumulator_pipeline_params,
                                             cluster_shape, true_type{}, false_type{});

    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::Epilogue);
    epi_store_pipeline_params.num_producers = NumEpilogueThreads;
    epi_store_pipeline_params.num_consumers = 1;
    EpiStorePipeline epi_store_pipeline(shared_pipelines.epi_store, epi_store_pipeline_params,
                                        cluster_shape, true_type{}, false_type{});

    typename EpiWaveOrderBarrier::Params epi_wave_order_barrier_params;
    epi_wave_order_barrier_params.group_id = 0;
    epi_wave_order_barrier_params.group_size = NumEpilogueThreads;
    epi_wave_order_barrier_params.initializing_warp = static_cast<int>(WarpCategory::Epilogue);
    EpiWaveOrderBarrier epi_wave_order_barrier(shared_pipelines.epi_wave_order, epi_wave_order_barrier_params);

    typename CLCPipeline::Params clc_pipeline_params;
    if (WarpCategory::Sched == warp_category) {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::ProducerConsumer;
    } else {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::Consumer;
    }
    clc_pipeline_params.producer_blockid = 0;
    clc_pipeline_params.producer_arv_count = 1;
    clc_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::Sched);
    clc_pipeline_params.consumer_arv_count = NumSchedThreads + TileScheduler::cluster_size * (NumMainloopLoadThreads + NumEpilogueThreads + NumEpilogueLoadThreads + NumMMAThreads);
    CLCPipeline clc_pipeline(shared_pipelines.clc, clc_pipeline_params, cluster_shape);

    // Pipeline states
    auto mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    auto mainloop_pipe_consumer_state = MainloopPipelineState{};
    auto epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    auto epi_load_pipe_consumer_state = EpiLoadPipelineState{};
    auto epi_g2s_pipe_producer_state = cutlass::make_producer_start_state<EpiG2SPipeline>();
    auto epi_g2s_pipe_consumer_state = EpiG2SPipelineState{};
    auto epi_fred_pipe_producer_state = cutlass::make_producer_start_state<EpiFredPipeline>();
    auto epi_fred_pipe_consumer_state = EpiFredPipelineState{};
    auto accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();
    auto accumulator_pipe_consumer_state = AccumulatorPipelineState{};
    auto epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();
    auto epi_store_pipe_consumer_state = EpiStorePipelineState{};
    auto clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();
    auto clc_pipe_consumer_state = CLCPipelineState{};

    auto wg_k = get<2>(TileShape{});
    uint32_t k_tile_count = (K + wg_k - 1) / wg_k;

    // Cluster sync
    auto cluster_wait_fn = [&]() {
      if constexpr (size(ClusterShape{}) > 1) {
        cluster_arrive();
        return []() { cluster_wait(); };
      } else {
        item.barrier(sycl::access::fence_space::local_space);
        return []() {};
      }
    }();
    cluster_wait_fn();

    // Initialize collectives
    CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape);
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_tensors.epilogue,
                                           epi_wave_order_barrier, make_tuple(tdesc_c, tdesc_d));

    auto load_inputs = collective_mainloop.load_init(params.problem_shape, shared_tensors.mainloop,
                                                      make_tuple(tdesc_a, tdesc_b));
    auto intermedia_tensor = CollectiveEpilogue::get_intermedia_tensor(shared_tensors.epilogue);

    dim3 block_id_in_cluster = cute::block_id_in_cluster();
    auto scheduler = TileScheduler(&shared_tensors.clc_response[0], params.scheduler, block_id_in_cluster);
    auto work_tile_info = scheduler.initial_work_tile_info();

    // ─── WARP SPECIALIZATION ───

    if (is_participant.main_load) {
      // ━━━ MAINLOOP LOAD WARP (with prefetch-ahead) ━━━
      do {
        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        auto k_tile_prologue = min(MainloopPipeline::Stages, k_tile_count);
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

        // Prologue loads with prefetch
        auto [mainloop_producer_state_next, k_tile_iter_next] =
            load_with_prefetch(collective_mainloop, mainloop_pipeline, mainloop_pipe_producer_state,
                               load_inputs, cta_coord_mnkl, k_tile_iter, k_tile_prologue);
        mainloop_pipe_producer_state = mainloop_producer_state_next;

        // Remaining loads with prefetch
        auto [mainloop_producer_state_next_, unused_] =
            load_with_prefetch(collective_mainloop, mainloop_pipeline, mainloop_pipe_producer_state,
                               load_inputs, cta_coord_mnkl, k_tile_iter_next, k_tile_count - k_tile_prologue);
        mainloop_pipe_producer_state = mainloop_producer_state_next_;

        auto [next_work_tile_info, increment_pipe] =
            scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        if (increment_pipe) { ++clc_pipe_consumer_state; }
        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());

    } else if (is_participant.sched) {
      // ━━━ SCHEDULER WARP (unchanged) ━━━
      if constexpr (IsSchedDynamicPersistent) {
        bool requires_clc_query = true;
        do {
          if (requires_clc_query) {
            clc_pipe_producer_state = scheduler.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
          }
          auto [next_work_tile_info, increment_pipe] =
              scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
          requires_clc_query = increment_pipe;
          if (increment_pipe) { ++clc_pipe_consumer_state; }
          work_tile_info = next_work_tile_info;
        } while (work_tile_info.is_valid());
      }

    } else if (is_participant.mma) {
      // ━━━ MMA WARP (unchanged) ━━━
      auto mma_inputs = collective_mainloop.mma_init(shared_tensors.mainloop);
      do {
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        mainloop_pipe_consumer_state = collective_mainloop.mma(
            cute::make_tuple(mainloop_pipeline, epi_store_pipeline, accumulator_pipeline),
            cute::make_tuple(mainloop_pipe_consumer_state, epi_store_pipe_producer_state, accumulator_pipe_producer_state),
            intermedia_tensor, mma_inputs, k_tile_count);

        auto [next_work_tile_info, increment_pipe] =
            scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        if (increment_pipe) { ++clc_pipe_consumer_state; }
        ++epi_store_pipe_producer_state;
        ++accumulator_pipe_producer_state;
        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());

    } else if (is_participant.epi_load) {
      // ━━━ EPILOGUE LOAD WARP (unchanged — store uses STORE_REDUCE when configured) ━━━
      int current_wave = 0;
      bool reverse_epi_n = false;
      static constexpr bool IsOverlappingAccum = false;
      auto prev_epi_store_consumer_state = epi_store_pipe_consumer_state;

      do {
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        auto [load_state_next, store_cons_state_next, acc_state_next, g2s_state_next, fred_state_next] =
            collective_epilogue.template load<IsOverlappingAccum>(
                cute::make_tuple(epi_load_pipeline, epi_store_pipeline, accumulator_pipeline,
                                 epi_g2s_pipeline, epi_fred_pipeline),
                cute::make_tuple(epi_load_pipe_producer_state, epi_store_pipe_consumer_state,
                                 accumulator_pipe_consumer_state, epi_g2s_pipe_producer_state,
                                 epi_fred_pipe_consumer_state),
                problem_shape_MNKL, CtaShape_MNK{}, cta_coord_mnkl,
                TileShape{}, TiledMma{}, shared_tensors.epilogue, reverse_epi_n);

        prev_epi_store_consumer_state = epi_store_pipe_consumer_state;
        epi_load_pipe_producer_state = load_state_next;
        epi_store_pipe_consumer_state = store_cons_state_next;
        (void)acc_state_next;  // DP-only: accumulator/g2s/fred states are inert here
        epi_g2s_pipe_producer_state = g2s_state_next;
        epi_fred_pipe_consumer_state = fred_state_next;

        auto [next_work_tile_info, increment_pipe] =
            scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        if (increment_pipe) { ++clc_pipe_consumer_state; }
        work_tile_info = next_work_tile_info;
        current_wave++;
      } while (work_tile_info.is_valid());

      epi_store_pipeline.producer_try_acquire(prev_epi_store_consumer_state);

    } else if (is_participant.epilogue) {
      // ━━━ EPILOGUE COMPUTE WARPS (unchanged) ━━━
      do {
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        auto [load_state_next, store_prod_state_next, acc_state_next, g2s_state_next, fred_state_next] =
            collective_epilogue.store(
                cute::make_tuple(epi_load_pipeline, epi_store_pipeline, accumulator_pipeline,
                                 epi_g2s_pipeline, epi_fred_pipeline),
                cute::make_tuple(epi_load_pipe_consumer_state, epi_store_pipe_producer_state,
                                 accumulator_pipe_consumer_state, epi_g2s_pipe_consumer_state,
                                 epi_fred_pipe_producer_state),
                problem_shape_MNKL, CtaShape_MNK{}, cta_coord_mnkl,
                TileShape{}, TiledMma{}, intermedia_tensor, shared_tensors.epilogue);

        epi_load_pipe_consumer_state = load_state_next;
        epi_store_pipe_producer_state = store_prod_state_next;
        accumulator_pipe_consumer_state = acc_state_next;
        epi_g2s_pipe_consumer_state = g2s_state_next;
        epi_fred_pipe_producer_state = fred_state_next;

        auto [next_work_tile_info, increment_pipe] =
            scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        if (increment_pipe) { ++clc_pipe_consumer_state; }
        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());
    }
  }
};

} // namespace cutlass::gemm::kernel

////////////////////////////////////////////////////////////////////////////////////////////////////
/// GemmUniversalAdapter specialization for GemmPrefetchReduce
///
/// The adapter wraps the kernel class and provides the host-side launch interface.
/// We reuse the standard GemmUniversalAdapter with our custom kernel.
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::device {

template <class GemmKernel_>
struct GemmPrefetchReduceAdapter {
  using GemmKernel = GemmKernel_;
  using Arguments = typename GemmKernel::Arguments;
  using Params = typename GemmKernel::Params;

  GemmKernel kernel;

  Params to_underlying_arguments(Arguments const& args) {
    return kernel.to_underlying_arguments(args, nullptr);
  }
};

} // namespace cutlass::gemm::device

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Shared memory info (same utility as xe4_gemm)
////////////////////////////////////////////////////////////////////////////////////////////////////

#define GET_MEM_INFO(cls, path) std::make_tuple((size_t) & (((cls *)0)->path), sizeof(((cls *)0)->path))

template<typename GemmKernel>
std::string get_shared_memory_info() {
  using TensorStorage = typename GemmKernel::TensorStorage;
  std::ostringstream oss;
  auto bytes2kb = [](size_t bytes) -> std::string {
    double kb = bytes / 1024.0;
    std::ostringstream format;
    if (bytes < 1024) { format << bytes << " Bytes"; }
    else if (bytes >= 1024 * 1024) { format << std::fixed << std::setprecision(2) << kb / 1024.0 << " MB"; }
    else if (bytes % 1024 == 0) { format << static_cast<int>(kb) << " KB"; }
    else { format << std::fixed << std::setprecision(2) << kb << " KB"; }
    return format.str();
  };
  auto [offsetA, sizeA] = GET_MEM_INFO(TensorStorage, mainloop.smem_A);
  auto [offsetB, sizeB] = GET_MEM_INFO(TensorStorage, mainloop.smem_B);
  auto [offsetAcc, sizeAcc] = GET_MEM_INFO(TensorStorage, mainloop.smem_Acc);
  auto [offsetC, sizeC] = GET_MEM_INFO(TensorStorage, epilogue.collective.smem_C);
  auto [offsetD, sizeD] = GET_MEM_INFO(TensorStorage, epilogue.collective.smem_D);
  size_t total_size = sizeof(TensorStorage);
  oss << "Shared Memory Allocation (Total: " << bytes2kb(total_size) << ")\n"
      << "- Mainloop: A=" << bytes2kb(sizeA) << " B=" << bytes2kb(sizeB) << " Acc=" << bytes2kb(sizeAcc) << "\n"
      << "- Epilogue: C=" << bytes2kb(sizeC) << " D=" << bytes2kb(sizeD) << "\n";
  return oss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// run_gemm_prefetch — GEMM with ADMA prefetch in mainloop (standard epilogue)
///
/// Uses the standard CollectiveEpilogue (XE4_ADMA_STORE) but adds prefetch-ahead
/// in the mainloop load path. This test is TBX-safe.
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Config>
void run_gemm_prefetch()
{
  using namespace cute;

  constexpr int NumControlWarps = 4;
  constexpr int NumEpilogueWarps = 16;

  using ElementA = typename Config::ElementA;
  using LayoutA  = typename Config::LayoutA;
  constexpr int AlignmentA = 512;
  using ElementB = typename Config::ElementB;
  using LayoutB  = typename Config::LayoutB;
  constexpr int AlignmentB = 512;
  using ElementC = typename Config::ElementC;
  using ElementD = typename Config::ElementD;
  using LayoutC  = typename Config::LayoutC;
  constexpr int AlignmentC = 512;
  using ElementAccumulator = typename Config::ElementAccumulator;
  using ArchTag = cutlass::arch::Xe4;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = typename Config::CtaTileShape_MNK;
  using ClusterShape = typename Config::ClusterShape_MNK;

  using EpilogueOperation = cutlass::epilogue::fusion::EltAct<
      cutlass::epilogue::thread::Identity, ElementD, ElementD>;
  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;

  // Standard epilogue (XE4_ADMA_STORE)
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutC, AlignmentC,
      EpilogueScheduleType, EpilogueOperation
    >::CollectiveOp;

  // Standard mainloop
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      cute::tuple<ElementAccumulator, ElementD>,
      TileShape, ClusterShape, cutlass::gemm::collective::StageCount<Config::StagesA>,
      cutlass::gemm::KernelTmaWarpSpecializedXe4<Config::StagesA, 1>
    >::CollectiveOp;

  // Custom kernel with prefetch enabled
  using GemmKernel = cutlass::gemm::kernel::GemmPrefetchReduce<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    true  // EnablePrefetch
  >;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  auto problem_shape_mnkl = typename GemmKernel::ProblemShape{};
  cute::fill_int_tuple_from(problem_shape_mnkl, Config::ProblemShape_MNKL);

  uint32_t sizeA = size(select<0,2,3>(problem_shape_mnkl));
  uint32_t sizeB = size(select<1,2,3>(problem_shape_mnkl));
  uint32_t sizeC = size(select<0,1,3>(problem_shape_mnkl));

  auto A_s = malloc_shared<ElementA>(sizeA, q);
  std::generate_n(A_s, sizeA, []() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });
  auto B_s = malloc_shared<ElementB>(sizeB, q);
  std::generate_n(B_s, sizeB, []() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });
  ElementC* C_s = nullptr;
  auto D_s = malloc_shared<ElementD>(sizeC, q);
  std::fill_n(D_s, sizeC, ElementD(0));

  auto [cluster_size_y, cluster_size_x, cluster_size_z] = ClusterShape{};
  sycl::range<3> cluster_size(cluster_size_z, cluster_size_y, cluster_size_x);

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, select<0,2,3>(problem_shape_mnkl));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, select<1,2,3>(problem_shape_mnkl));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, select<0,1,3>(problem_shape_mnkl));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, select<0,1,3>(problem_shape_mnkl));

  auto smem_info = get_shared_memory_info<GemmKernel>();
  std::cout << smem_info << std::endl;

  int device_id = 0;
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  cutlass::KernelHardwareInfo kernel_hw_info{device_id, sm_count, 0};

  using FusionCallbacks = typename CollectiveEpilogue::FusionCallbacks;
  auto args = typename GemmKernel::Arguments{
    problem_shape_mnkl,
    { A_s, stride_A, B_s, stride_B },
    { typename FusionCallbacks::Arguments{}, C_s, stride_C, D_s, stride_D }
  };

  GemmKernel kernel;
  auto params = kernel.to_underlying_arguments(args, kernel_hw_info, nullptr);

  dim3 const grid = GemmKernel::get_grid_shape(params);
  dim3 const block = GemmKernel::get_block_shape();

  range<3> group_range(grid.z, grid.y, grid.x);
  range<3> local_range(block.z, block.y, block.x);

  int smem_size = 0;
  cutlass::SyclClusterLaunchParams launch_params = {group_range, local_range, cluster_size, smem_size, q};
  cutlass::launch_kernel_on_cluster(launch_params, kernel, params).wait();

  // Validate: D = A × B (identity epilogue, no C source)
  auto [mat_m, mat_n, mat_k, mat_l] = problem_shape_mnkl;
  auto as_mem_layout = [](auto layout) {
    if constexpr (std::is_same_v<decltype(layout), cutlass::layout::RowMajor>) {
      return mem_layout::row_major;
    } else {
      return mem_layout::col_major;
    }
  };

  uint32_t err_cnt = validate_gemm_result(
      A_s, B_s, D_s, mat_m, mat_n, mat_k,
      as_mem_layout(LayoutA{}), as_mem_layout(LayoutB{}));

  if (err_cnt > 0) {
    std::cerr << "GEMM Prefetch Test FAILED! error count: " << err_cnt << std::endl;
    exit(1);
  }
  std::cout << "Test Pass!" << std::endl;

  sycl::free(A_s, q);
  sycl::free(B_s, q);
  sycl::free(D_s, q);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// run_gemm_prefetch_reduce — GEMM with ADMA prefetch + STORE_REDUCE epilogue
///
/// Uses XE4_ADMA_STORE_REDUCE<fp16, RedOp::Add> for the D output store.
/// Pre-initializes D=0 so reduce-Add produces the same result as a normal store.
///
/// IMPORTANT: Requires real XE4 hardware. TBX does not support reduce_l2g.
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Config>
void run_gemm_prefetch_reduce()
{
  using namespace cute;

  using ElementA = typename Config::ElementA;
  using LayoutA  = typename Config::LayoutA;
  constexpr int AlignmentA = 512;
  using ElementB = typename Config::ElementB;
  using LayoutB  = typename Config::LayoutB;
  constexpr int AlignmentB = 512;
  using ElementC = typename Config::ElementC;
  using ElementD = typename Config::ElementD;
  using LayoutC  = typename Config::LayoutC;
  constexpr int AlignmentC = 512;
  using ElementAccumulator = typename Config::ElementAccumulator;
  using ArchTag = cutlass::arch::Xe4;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = typename Config::CtaTileShape_MNK;
  using ClusterShape = typename Config::ClusterShape_MNK;

  using EpilogueOperation = cutlass::epilogue::fusion::EltAct<
      cutlass::epilogue::thread::Identity, ElementD, ElementD>;

  // Custom reduce epilogue: uses STORE_REDUCE<Add> instead of plain STORE
  using CollectiveEpilogue = typename cutlass::epilogue::collective::detail::Xe4AdmaReduceBuilderImpl<
      OperatorClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutC, AlignmentC,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueOperation,
      cute::RedOp::Add,
      cute::BarrierType::Abarrier
    >::CollectiveOp;

  // Standard mainloop
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      cute::tuple<ElementAccumulator, ElementD>,
      TileShape, ClusterShape, cutlass::gemm::collective::StageCount<Config::StagesA>,
      cutlass::gemm::KernelTmaWarpSpecializedXe4<Config::StagesA, 1>
    >::CollectiveOp;

  // Custom kernel with both prefetch and reduce
  using GemmKernel = cutlass::gemm::kernel::GemmPrefetchReduce<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    true  // EnablePrefetch
  >;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  auto problem_shape_mnkl = typename GemmKernel::ProblemShape{};
  cute::fill_int_tuple_from(problem_shape_mnkl, Config::ProblemShape_MNKL);

  uint32_t sizeA = size(select<0,2,3>(problem_shape_mnkl));
  uint32_t sizeB = size(select<1,2,3>(problem_shape_mnkl));
  uint32_t sizeC = size(select<0,1,3>(problem_shape_mnkl));

  auto A_s = malloc_shared<ElementA>(sizeA, q);
  std::generate_n(A_s, sizeA, []() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });
  auto B_s = malloc_shared<ElementB>(sizeB, q);
  std::generate_n(B_s, sizeB, []() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });
  ElementC* C_s = nullptr;
  auto D_s = malloc_shared<ElementD>(sizeC, q);
  // Pre-initialize D=0: reduce-Add(0, result) == result, so validation is same as normal GEMM.
  // On real hardware, STORE_REDUCE atomically does D[i] = D[i] + epilogue_result[i].
  std::fill_n(D_s, sizeC, ElementD(0));

  auto [cluster_size_y, cluster_size_x, cluster_size_z] = ClusterShape{};
  sycl::range<3> cluster_size(cluster_size_z, cluster_size_y, cluster_size_x);

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, select<0,2,3>(problem_shape_mnkl));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, select<1,2,3>(problem_shape_mnkl));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, select<0,1,3>(problem_shape_mnkl));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, select<0,1,3>(problem_shape_mnkl));

  auto smem_info = get_shared_memory_info<GemmKernel>();
  std::cout << smem_info << std::endl;

  int device_id = 0;
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  cutlass::KernelHardwareInfo kernel_hw_info{device_id, sm_count, 0};

  using FusionCallbacks = typename CollectiveEpilogue::FusionCallbacks;
  auto args = typename GemmKernel::Arguments{
    problem_shape_mnkl,
    { A_s, stride_A, B_s, stride_B },
    { typename FusionCallbacks::Arguments{}, C_s, stride_C, D_s, stride_D }
  };

  GemmKernel kernel;
  auto params = kernel.to_underlying_arguments(args, kernel_hw_info, nullptr);

  dim3 const grid = GemmKernel::get_grid_shape(params);
  dim3 const block = GemmKernel::get_block_shape();

  range<3> group_range(grid.z, grid.y, grid.x);
  range<3> local_range(block.z, block.y, block.x);

  int smem_size = 0;
  cutlass::SyclClusterLaunchParams launch_params = {group_range, local_range, cluster_size, smem_size, q};
  cutlass::launch_kernel_on_cluster(launch_params, kernel, params).wait();

  // Validate: D = 0 + (A × B) = A × B (identity epilogue, reduce-Add from zero)
  auto [mat_m, mat_n, mat_k, mat_l] = problem_shape_mnkl;
  auto as_mem_layout = [](auto layout) {
    if constexpr (std::is_same_v<decltype(layout), cutlass::layout::RowMajor>) {
      return mem_layout::row_major;
    } else {
      return mem_layout::col_major;
    }
  };

  uint32_t err_cnt = validate_gemm_result(
      A_s, B_s, D_s, mat_m, mat_n, mat_k,
      as_mem_layout(LayoutA{}), as_mem_layout(LayoutB{}));

  if (err_cnt > 0) {
    std::cerr << "GEMM Prefetch+Reduce Test FAILED! error count: " << err_cnt << std::endl;
    exit(1);
  }
  std::cout << "Test Pass!" << std::endl;

  sycl::free(A_s, q);
  sycl::free(B_s, q);
  sycl::free(D_s, q);
}
