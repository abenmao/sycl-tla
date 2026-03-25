#pragma once

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "cute/arch/copy_xe4_adma.hpp"
#include "cute/arch/cluster_xe4.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"

namespace cutlass::gemm::kernel {

using namespace cute;
using namespace cute::xe4;

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<
    cutlass::detail::is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, KernelTmaWarpSpecializedXe4> &&
    cute::is_same_v<TileScheduler_, StaticPersistentScheduler>>
> {
public:
  // Detect if dispatch policy is block-scaled
  template <class DispatchPolicy>
  struct IsBlockScaledDispatchPolicy : cute::false_type {};

  template <int Stages, int SchedulerPipelineStageCount, int AccumulatorPipelineStageCount, class ClusterShape>
  struct IsBlockScaledDispatchPolicy<
    cutlass::gemm::MainloopXe4DmaGmmaWarpSpecializedBlockScaled<
      Stages,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape
    >> : cute::true_type {};

  using OriginalDispatchPolicy = typename CollectiveMainloop_::DispatchPolicy;

  // For block-scaled: use original collective unchanged
  // For regular: rebuild collective to use static dispatch policy variant (matches xe4_mma_warpspecialized_static.hpp)
  using CollectiveMainloop = cute::conditional_t<
    IsBlockScaledDispatchPolicy<OriginalDispatchPolicy>::value,
    CollectiveMainloop_,
    cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopXe4DmaGmmaWarpSpecializedStatic<
        OriginalDispatchPolicy::Stages,
        OriginalDispatchPolicy::Schedule::SchedulerPipelineStageCount,
        OriginalDispatchPolicy::Schedule::AccumulatorPipelineStageCount,
        typename OriginalDispatchPolicy::ClusterShape>,
      typename CollectiveMainloop_::TileShape,
      typename CollectiveMainloop_::ElementA,
      typename CollectiveMainloop_::StrideA,
      typename CollectiveMainloop_::ElementB,
      typename CollectiveMainloop_::StrideB,
      typename CollectiveMainloop_::TiledMma,
      typename CollectiveMainloop_::GmemTiledCopyA,
      typename CollectiveMainloop_::SmemLayoutAtomA,
      typename CollectiveMainloop_::SmemCopyAtomA,
      typename CollectiveMainloop_::TransformA,
      typename CollectiveMainloop_::GmemTiledCopyB,
      typename CollectiveMainloop_::SmemLayoutAtomB,
      typename CollectiveMainloop_::SmemCopyAtomB,
      typename CollectiveMainloop_::TransformB>
  >;

  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

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

  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementD;
  using StrideC = typename CollectiveEpilogue::StrideD;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static constexpr uint32_t SchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  static constexpr uint32_t AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;

  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileSchedulerTag, ArchTag, TileShape, ClusterShape, SchedulerPipelineStageCount>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;

  static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp;
  static constexpr uint32_t NumMMAThreads          = NumThreadsPerWarp;
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp;
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp;
  static constexpr uint32_t NumEpilogueThreads     = CollectiveEpilogue::ThreadCount;
  constexpr static int NumControlWarps = 4;
  constexpr static int NumEpilogueWarps = 16;

  static constexpr uint32_t MaxThreadsPerBlock = NumSchedThreads +
                                                 NumMainloopLoadThreads + NumMMAThreads +
                                                 NumEpilogueLoadThreads + NumEpilogueThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using MainloopPipelineState = typename CollectiveMainloop::MainloopPipelineState;

  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  using EpiLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;

  using AccumulatorPipeline = cutlass::PipelineTmaAsync<AccumulatorPipelineStageCount>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

  using EpiWaveOrderBarrier = typename CollectiveEpilogue::WaveOrderBarrier;

  using CLCPipeline = cutlass::PipelineTmaAsync<SchedulerPipelineStageCount>;
  using CLCPipelineState = typename CLCPipeline::PipelineState;

  struct SharedStorage
  {
    struct TensorStorage
    {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
      typename TileScheduler::CLCResponse clc_response[SchedulerPipelineStageCount];
    } tensors;

    struct PipelineStorage {
      using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;
      using EpiLoadPipelineStorage = typename EpiLoadPipeline::SharedStorage;
      using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;
      using EpiStorePipelineStorage = typename EpiStorePipeline::SharedStorage;
      using EpiWaveOrderBarrierStorage = typename EpiWaveOrderBarrier::SharedStorage;
      using CLCPipelineStorage = typename CLCPipeline::SharedStorage;

      MainloopPipelineStorage mainloop;
      EpiLoadPipelineStorage epi_load;
      AccumulatorPipelineStorage accumulator;
      EpiStorePipelineStorage epi_store;
      EpiWaveOrderBarrierStorage epi_wave_order;
      CLCPipelineStorage clc;
    } pipelines;
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  static constexpr int TensorStorageSize = sizeof(typename SharedStorage::TensorStorage);
  static constexpr int PipelineStorageSize = sizeof(typename SharedStorage::PipelineStorage);

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
    static constexpr int TotalSubGroups = NumControlWarps + NumEpilogueWarps;
    return dim3(cutlass::NumThreadsPerWarp, TotalSubGroups, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params) const {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto& problem_shape = params.problem_shape;

    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [M,N,K,L] = problem_shape_MNKL;

    uint32_t local_id = item.get_local_linear_id();
    uint32_t warp_idx = get_sg_id();
    WarpCategory warp_category = warp_idx < static_cast<int>(WarpCategory::Epilogue) ? WarpCategory(warp_idx)
                                                                                     : WarpCategory::Epilogue;

    bool lane_predicate = cute::elect_one_sync();
    auto cluster_shape = ClusterShape{};

    auto ptr = alloc_slm_buffer<uint8_t, TensorStorageSize>(item.get_group());
    auto& shared_tensors = *reinterpret_cast<typename SharedStorage::TensorStorage*>(ptr);

    auto tdesc_a = allocate_tdesc<0>();
    auto tdesc_b = allocate_tdesc<1>();
    auto tdesc_c = allocate_tdesc<2>();
    auto tdesc_d = allocate_tdesc<3>();

    auto abar_base = allocate_abar_bytes<0, PipelineStorageSize>();
    auto& shared_pipelines = *reinterpret_cast<typename SharedStorage::PipelineStorage*>(abar_base);

    bool is_first_cta_in_cluster = true;
    IsParticipant is_participant = {
      (warp_category == WarpCategory::MMA),
      (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster,
      (warp_category == WarpCategory::MainloopLoad),
      (warp_category == WarpCategory::EpilogueLoad),
      (warp_category == WarpCategory::Epilogue)
    };

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
    MainloopPipeline mainloop_pipeline(shared_pipelines.mainloop, mainloop_pipeline_params, cluster_shape, true_type{}, false_type{});

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
    AccumulatorPipeline accumulator_pipeline(shared_pipelines.accumulator, accumulator_pipeline_params, cluster_shape, true_type{}, false_type{});

    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::Epilogue);
    epi_store_pipeline_params.num_producers = NumEpilogueThreads;
    epi_store_pipeline_params.num_consumers = 1;
    EpiStorePipeline epi_store_pipeline(shared_pipelines.epi_store, epi_store_pipeline_params, cluster_shape, true_type{}, false_type{});

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
    clc_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::Sched);
    clc_pipeline_params.num_producers = NumSchedThreads;
    clc_pipeline_params.num_consumers = NumSchedThreads + NumMMAThreads + NumMainloopLoadThreads + NumEpilogueLoadThreads + NumEpilogueThreads;
    CLCPipeline clc_pipeline(shared_pipelines.clc, clc_pipeline_params, cluster_shape, true_type{}, false_type{});

    auto mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    auto mainloop_pipe_consumer_state = MainloopPipelineState{};

    auto epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    auto epi_load_pipe_consumer_state = EpiLoadPipelineState{};

    auto accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();
    auto accumulator_pipe_consumer_state = AccumulatorPipelineState{};

    auto epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();
    auto epi_store_pipe_consumer_state = EpiStorePipelineState{};

    auto clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();
    auto clc_pipe_consumer_state = CLCPipelineState{};

    auto wg_k = get<2>(TileShape{});
    uint32_t k_tile_count = (K + wg_k -1) / wg_k;

    auto cluster_wait_fn = [&] () {
      if constexpr (size(ClusterShape{}) > 1) {
        cluster_arrive();
        return [] () { cluster_wait(); };
      }
      else {
        item.barrier(sycl::access::fence_space::local_space);
        return [] () {};
      }
    } ();

    cluster_wait_fn();

    CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape);
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_tensors.epilogue, epi_wave_order_barrier, make_tuple(tdesc_c, tdesc_d));

    auto load_inputs = collective_mainloop.load_init(problem_shape, shared_tensors.mainloop, make_tuple(tdesc_a, tdesc_b));
    auto intermedia_tensor = CollectiveEpilogue::get_intermedia_tensor(shared_tensors.epilogue);

    auto scheduler = TileScheduler(&shared_tensors.clc_response[0], params.scheduler);
    auto work_tile_info = scheduler.initial_work_tile_info(cluster_shape);

    if (is_participant.main_load) {
      do {
        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        auto k_tile_prologue = min(MainloopPipeline::Stages, k_tile_count);

        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

        auto [mainloop_producer_state_next, k_tile_iter_next] = collective_mainloop.load(
          params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, cta_coord_mnkl,
          k_tile_iter, k_tile_prologue);
        mainloop_pipe_producer_state = mainloop_producer_state_next;

        auto [mainloop_producer_state_next_, unused_] = collective_mainloop.load(
          params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs, cta_coord_mnkl,
          k_tile_iter_next, k_tile_count - k_tile_prologue);
        mainloop_pipe_producer_state = mainloop_producer_state_next_;

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());
    } else if (is_participant.sched) {
      if constexpr (IsSchedDynamicPersistent) {
        bool requires_clc_query = true;

        do {
          if (requires_clc_query) {
            clc_pipe_producer_state = scheduler.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
          }

          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
          requires_clc_query = increment_pipe;
          if (increment_pipe) {
            ++clc_pipe_consumer_state;
          }

          work_tile_info = next_work_tile_info;
        } while (work_tile_info.is_valid());
      }
    } else if (is_participant.mma) {
      auto mma_inputs = collective_mainloop.mma_init(shared_tensors.mainloop);

      do {
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

        mainloop_pipe_consumer_state = collective_mainloop.mma(
          cute::make_tuple(mainloop_pipeline, epi_store_pipeline, accumulator_pipeline),
          cute::make_tuple(mainloop_pipe_consumer_state, epi_store_pipe_producer_state, accumulator_pipe_producer_state),
          intermedia_tensor, mma_inputs, k_tile_count);

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        ++epi_store_pipe_producer_state;
        ++accumulator_pipe_producer_state;
        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());
    } else if (is_participant.epi_load) {
      bool reverse_epi_n = false;
      static constexpr bool IsOverlappingAccum = false;
      auto prev_epi_store_consumer_state = epi_store_pipe_consumer_state;

      do {
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

        auto [load_state_next, store_cons_state_next] = collective_epilogue.template load<IsOverlappingAccum>(
          cute::make_tuple(epi_load_pipeline, epi_store_pipeline),
          cute::make_tuple(epi_load_pipe_producer_state, epi_store_pipe_consumer_state),
          problem_shape_MNKL,
          CtaShape_MNK{},
          cta_coord_mnkl,
          TileShape{},
          TiledMma{},
          shared_tensors.epilogue,
          reverse_epi_n
        );
        prev_epi_store_consumer_state = epi_store_pipe_consumer_state;
        epi_load_pipe_producer_state = load_state_next;
        epi_store_pipe_consumer_state = store_cons_state_next;

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }

        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());

      epi_store_pipeline.producer_try_acquire(prev_epi_store_consumer_state);
    } else if (is_participant.epilogue)  {
      do {
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

        auto [load_state_next, store_prod_state_next, acc_state_next] = collective_epilogue.store(
          cute::make_tuple(epi_load_pipeline, epi_store_pipeline, accumulator_pipeline),
          cute::make_tuple(epi_load_pipe_consumer_state, epi_store_pipe_producer_state, accumulator_pipe_consumer_state),
          problem_shape_MNKL,
          CtaShape_MNK{},
          cta_coord_mnkl,
          TileShape{},
          TiledMma{},
          intermedia_tensor,
          shared_tensors.epilogue
        );
        epi_load_pipe_consumer_state = load_state_next;
        epi_store_pipe_producer_state = store_prod_state_next;
        accumulator_pipe_consumer_state = acc_state_next;

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }
        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());
    }
  }
};

}