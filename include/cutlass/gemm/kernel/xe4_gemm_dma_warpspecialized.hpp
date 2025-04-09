#pragma once

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "cute/arch/copy_xe4_dma.hpp"
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
    cutlass::detail::is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule, KernelTmaWarpSpecializedXe4>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
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

  // CLC pipeline depth
  // determines how many waves (stages-1) a warp can race ahead
  static constexpr uint32_t SchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  static constexpr uint32_t AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;

  // TileID scheduler
  // Get Blk and Scheduling tile shapes
  using CtaShape_MNK = typename CollectiveMainloop::TileShape;
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileSchedulerTag, ArchTag, TileShape, ClusterShape, SchedulerPipelineStageCount>::Scheduler;

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;

  static constexpr uint32_t NumControlWarps  = CollectiveEpilogue::NumControlWarps;
  static constexpr uint32_t NumEpilogueWarps = CollectiveEpilogue::NumEpilogueWarps;

  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads         = NumThreadsPerWarp; // 1 subgroup
  static constexpr uint32_t NumMMAThreads           = NumThreadsPerWarp; // 1 subgroup
  static constexpr uint32_t NumMainloopLoadThreads  = NumThreadsPerWarp; // 1 subgroup
  static constexpr uint32_t NumEpilogueStoreThreads = NumThreadsPerWarp; // 1 subgroup
  static constexpr uint32_t NumEpilogueThreads      = NumEpilogueWarps * NumThreadsPerWarp;

  static constexpr uint32_t MaxThreadsPerBlock = NumSchedThreads +
                                                 NumMainloopLoadThreads + NumMMAThreads +
                                                 NumEpilogueStoreThreads + NumEpilogueThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // Pipeline and pipeline state types
  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using MainloopPipelineState = typename CollectiveMainloop::MainloopPipelineState;

  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

  using AccumulatorPipeline = cutlass::PipelineTmaAsync<AccumulatorPipelineStageCount>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  using CLCPipeline = cutlass::PipelineTmaAsync<SchedulerPipelineStageCount>;
  using CLCPipelineState = typename CLCPipeline::PipelineState;

  using CLCThrottlePipeline = cutlass::PipelineTmaAsync<SchedulerPipelineStageCount>;
  using CLCThrottlePipelineState = typename CLCThrottlePipeline::PipelineState;

  // Kernel level shared memory storage
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
      using AccumulatorPipelineStorage = typename AccumulatorPipeline::SharedStorage;
      using EpiStorePipelineStorage = typename EpiStorePipeline::SharedStorage;
      using CLCPipelineStorage = typename CLCPipeline::SharedStorage;
      using CLCThrottlePipelineStorage = typename CLCThrottlePipeline::SharedStorage;

      MainloopPipelineStorage mainloop;
      AccumulatorPipelineStorage accumulator;
      EpiStorePipelineStorage epi_store;
      CLCPipelineStorage clc;
      CLCThrottlePipelineStorage clc_throttle;
    } pipelines;
  };

  static constexpr int TensorStorageSize = sizeof(typename SharedStorage::TensorStorage);
  static constexpr int PipelineStorageSize = sizeof(typename SharedStorage::PipelineStorage);

  // Host facing host arguments
  struct Arguments {
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueArguments epilogue;
  };

  // Kernel device entry point API
  struct Params {
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
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

  //
  // Methods
  //

  // Convert to underlying arguments.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace)
    };
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto& problem_shape = params.problem_shape;

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [M,N,K,L] = problem_shape_MNKL;

   // Account for more than one epilogue warp
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

    auto abar_base = allocate_abar_bytes<0, PipelineStorageSize>();
    auto& shared_pipelines = *reinterpret_cast<typename SharedStorage::PipelineStorage*>(abar_base);

    CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape);
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_tensors.epilogue, tdesc_c);

    // Do we load source tensor C or other aux inputs
    bool is_epi_load_needed = false;
    bool is_first_cta_in_cluster = true;
    IsParticipant is_participant = {
      (warp_category == WarpCategory::MMA),                                 // mma
      (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster,    // sched
      (warp_category == WarpCategory::MainloopLoad),                        // main_load
      (warp_category == WarpCategory::EpilogueLoad),                        // epi_load
      (warp_category == WarpCategory::Epilogue)                             // epilogue
    };

    // Mainloop Load pipeline
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
    mainloop_pipeline_params.initializing_warp = 0;
    MainloopPipeline mainloop_pipeline(shared_pipelines.mainloop, mainloop_pipeline_params, cluster_shape, true_type{}, false_type{});

    // Mainloop-Epilogue pipeline
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
    accumulator_pipeline_params.initializing_warp = 1;
    AccumulatorPipeline accumulator_pipeline(shared_pipelines.accumulator, accumulator_pipeline_params, cluster_shape, true_type{}, false_type{});

    // Epilogue Store pipeline
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.initializing_warp = 2;
    epi_store_pipeline_params.num_producers = NumEpilogueThreads;
    epi_store_pipeline_params.num_consumers = 1;
    EpiStorePipeline epi_store_pipeline(shared_pipelines.epi_store, epi_store_pipeline_params, cluster_shape, true_type{}, false_type{});

    // CLC pipeline
    typename CLCPipeline::Params clc_pipeline_params;
    if (WarpCategory::Sched == warp_category) {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::ProducerConsumer;
    } else {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::Consumer;
    }
    clc_pipeline_params.initializing_warp = 3;
    clc_pipeline_params.num_producers = NumSchedThreads;
    clc_pipeline_params.num_consumers = NumSchedThreads + NumMMAThreads + NumMainloopLoadThreads + NumEpilogueStoreThreads + NumEpilogueThreads;
    CLCPipeline clc_pipeline(shared_pipelines.clc, clc_pipeline_params, cluster_shape, true_type{}, false_type{});

    // CLC throttle pipeline
    typename CLCThrottlePipeline::Params clc_throttle_pipeline_params;
    if (WarpCategory::MainloopLoad == warp_category) {
      clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Sched == warp_category)  {
      clc_throttle_pipeline_params.role = CLCThrottlePipeline::ThreadCategory::Consumer;
    }
    clc_throttle_pipeline_params.initializing_warp = 4;
    clc_throttle_pipeline_params.num_producers = NumThreadsPerWarp;
    clc_throttle_pipeline_params.num_consumers = NumThreadsPerWarp;
    CLCThrottlePipeline clc_throttle_pipeline(shared_pipelines.clc_throttle, clc_throttle_pipeline_params, cluster_shape, true_type{}, false_type{});

    auto mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    auto mainloop_pipe_consumer_state = MainloopPipelineState{};

    auto accumulator_pipe_producer_state = cutlass::make_producer_start_state<AccumulatorPipeline>();
    auto accumulator_pipe_consumer_state = AccumulatorPipelineState{};

    auto epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();
    auto epi_store_pipe_consumer_state = EpiStorePipelineState{};

    auto clc_pipe_throttle_producer_state = cutlass::make_producer_start_state<CLCThrottlePipeline>();
    auto clc_pipe_throttle_consumer_state = CLCThrottlePipelineState{};

    auto clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();
    auto clc_pipe_consumer_state = CLCPipelineState{};

    auto wg_k = get<2>(TileShape{});
    uint32_t k_tile_count = (K + wg_k -1) / wg_k;

    auto intermedia_tensor = CollectiveEpilogue::get_intermedia_tensor(shared_tensors.epilogue);

    auto cluster_wait_fn = [&] () {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cbar_arrive();
        return [] () { cbar_wait(); };
      }
      else {
        item.barrier(sycl::access::fence_space::local_space);
        return [] () {}; // do nothing
      }
    } ();

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    auto load_inputs = collective_mainloop.load_init(problem_shape, shared_tensors.mainloop, make_tuple(tdesc_a, tdesc_b));

    auto coop_set_ids = collective_mainloop.coop_set_ids_;
    auto problem_blocks_shape = TileScheduler::calculate_problem_blocks_shape(problem_shape_MNKL, CtaShape_MNK{});
    auto scheduler = TileScheduler(&shared_tensors.clc_response[0], problem_blocks_shape, coop_set_ids);
    auto work_tile_info = scheduler.initial_work_tile_info();

    if (is_participant.main_load) {
      bool requires_clc_query = true;
      do {
        // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
        auto k_tile_prologue = min(MainloopPipeline::Stages, k_tile_count);

        if constexpr (IsSchedDynamicPersistent) {
          if (is_first_cta_in_cluster && requires_clc_query) {
            clc_throttle_pipeline.producer_acquire(clc_pipe_throttle_producer_state);
            clc_throttle_pipeline.producer_commit(clc_pipe_throttle_producer_state, 1);
            ++clc_pipe_throttle_producer_state;
          }
        }

        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

        // Start mainloop prologue loads, arrive on the epilogue residual load barrier, resume mainloop loads
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
        requires_clc_query = increment_pipe;
      } while (work_tile_info.is_valid());
    } else if (is_participant.sched) {
      bool requires_clc_query = true;
      if constexpr (IsSchedDynamicPersistent) {
        do {
          if (requires_clc_query) {
            // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
            clc_throttle_pipeline.consumer_wait(clc_pipe_throttle_consumer_state);
            clc_throttle_pipeline.consumer_release(clc_pipe_throttle_consumer_state);
            ++clc_pipe_throttle_consumer_state;
            // Query next clcID and update producer state
            clc_pipe_producer_state = scheduler.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
          }

          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
          if (increment_pipe) {
            ++clc_pipe_consumer_state;
          }

          work_tile_info = next_work_tile_info;
          requires_clc_query = increment_pipe;
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
    } else if (is_participant.epilogue)  {
      auto prev_epi_store_consumer_state = epi_store_pipe_consumer_state;

      do {
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

        //
        // Epilogue and write to gD
        //
        auto [store_prod_state_next, store_cons_state_next, acc_state_next] = collective_epilogue.store(
          cute::make_tuple(epi_store_pipeline, accumulator_pipeline),
          cute::make_tuple(epi_store_pipe_producer_state, epi_store_pipe_consumer_state, accumulator_pipe_consumer_state),
          problem_shape,
          cta_coord_mnkl,
          shared_tensors.epilogue
        );
        prev_epi_store_consumer_state = epi_store_pipe_consumer_state;
        epi_store_pipe_producer_state = store_prod_state_next;
        epi_store_pipe_consumer_state = store_cons_state_next;
        accumulator_pipe_consumer_state = acc_state_next;

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        if (increment_pipe) {
          ++clc_pipe_consumer_state;
        }
        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());

      epi_store_pipeline.producer_try_acquire(prev_epi_store_consumer_state);
    }
  }
};

}
