#pragma once

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include "cute/arch/copy_xe4_dma_legacy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"

namespace cutlass::conv::kernel {

using namespace cute;
using namespace sycl;
using namespace cute::xe4;

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class Xe4ConvUniversal
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  // Handles the static_assert placed inside the operator()
  // This is also used to decide whether the load_init inside collective mainloop returns rank 4 tensors or rank 5 tensors
  static constexpr bool IsConvProblemShape = not (cute::is_tuple_v<ProblemShape>|| cutlass::gemm::kernel::IsCutlass3ArrayKernel<ProblemShape>::value);
  static_assert( IsConvProblemShape || (cute::rank(ProblemShape{}) == 3 || cute::rank(ProblemShape{}) == 4), "ProblemShape{} should be <M,N,K> or <M,N,K,L> for Gemm");

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
  using TileScheduler = typename cutlass::gemm::kernel::detail::TileSchedulerSelector<
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
  using MainloopPipelineState = typename CollectiveMainloop::PipelineState;

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
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Host facing host arguments
  struct Arguments {
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueArguments epilogue;
  };

  // Kernel entry point API
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
  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape.get_shape_C(), args.epilogue, workspace)
    };
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params) const {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto& problem_shape = params.problem_shape;

   // Account for more than one epilogue warp
    uint32_t local_id = item.get_local_linear_id();
    uint32_t warp_idx = get_sg_id();
    WarpCategory warp_category = warp_idx < static_cast<int>(WarpCategory::Epilogue) ? WarpCategory(warp_idx)
                                                                                     : WarpCategory::Epilogue;
    bool lane_predicate = cute::elect_one_sync();
    auto cluster_shape = ClusterShape{};

    auto ptr = alloc_slm_buffer<uint8_t, SharedStorageSize>(item.get_group());
    auto& shared_tensors = *reinterpret_cast<typename SharedStorage::TensorStorage*>(ptr);

    auto tdesc_b = allocate_tdesc<0>();
    auto tdesc_c = allocate_tdesc<1>();
    auto tdesc_d = allocate_tdesc<2>();

    auto& shared_pipelines = allocate_abarrier<typename SharedStorage::PipelineStorage>();

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
    mainloop_pipeline_params.num_producers = 1;
    mainloop_pipeline_params.num_consumers = 1;
    mainloop_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::MainloopLoad);
    MainloopPipeline mainloop_pipeline(shared_pipelines.mainloop, mainloop_pipeline_params, cluster_shape, true_type{}, false_type{});

    // Epilogue Load pipeline
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (WarpCategory::EpilogueLoad == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = lane_predicate && is_participant.epi_load;
    epi_load_pipeline_params.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = NumEpilogueThreads;
    epi_load_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::EpilogueLoad);
    EpiLoadPipeline epi_load_pipeline(shared_pipelines.epi_load, epi_load_pipeline_params, true_type{});

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
    accumulator_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::MMA);
    AccumulatorPipeline accumulator_pipeline(shared_pipelines.accumulator, accumulator_pipeline_params, cluster_shape, true_type{}, false_type{});

    // Epilogue Store pipeline
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::Epilogue);
    epi_store_pipeline_params.num_producers = NumEpilogueThreads;
    epi_store_pipeline_params.num_consumers = 1;
    EpiStorePipeline epi_store_pipeline(shared_pipelines.epi_store, epi_store_pipeline_params, cluster_shape, true_type{}, false_type{});


    // Epilogue wave order barrier
    typename EpiWaveOrderBarrier::Params epi_wave_order_barrier_params;
    epi_wave_order_barrier_params.group_id = 0;
    epi_wave_order_barrier_params.group_size = NumEpilogueThreads;
    epi_wave_order_barrier_params.initializing_warp = static_cast<int>(WarpCategory::Epilogue);
    EpiWaveOrderBarrier epi_wave_order_barrier(shared_pipelines.epi_wave_order, epi_wave_order_barrier_params);

    // CLC pipeline
    typename CLCPipeline::Params clc_pipeline_params;
    if (WarpCategory::Sched == warp_category) {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::ProducerConsumer;
    } else {
      clc_pipeline_params.role = CLCPipeline::ThreadCategory::Consumer;
    }
    clc_pipeline_params.initializing_warp = static_cast<int>(WarpCategory::Sched);
    clc_pipeline_params.num_producers = NumSchedThreads;
    clc_pipeline_params.num_consumers = NumSchedThreads + NumMMAThreads + NumMainloopLoadThreads + NumEpilogueStoreThreads + NumEpilogueThreads;
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

    using SmemLayoutAtomD = decltype(make_layout(select<0,1>(TileShape{}), GenRowMajor{}));

    using SmemLayoutAcc = decltype(tile_to_shape(
        SmemLayoutAtomD{},
        make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})),
        Step<_2,_1>{}));
    using SmemLayoutDst = decltype(tile_to_shape(
        SmemLayoutAtomD{},
        make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})),
        Step<_2,_1>{}));

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_tensors.epilogue, epi_wave_order_barrier, make_tuple(tdesc_c, tdesc_d));

    auto conv_problem_shape = collective_mainloop.get_problem_shape_MNKL(problem_shape);
    auto load_inputs = collective_mainloop.load_init(conv_problem_shape, params.mainloop, tdesc_b);
    auto intermedia_tensor = CollectiveEpilogue::get_intermedia_tensor(shared_tensors.epilogue);

    auto [gA_mk, gB_nk] = load_inputs;
    auto k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mk));
    auto k_tile_count = size<3>(gA_mk);

    item.barrier(access::fence_space::local_space);

    auto coop_set_ids = make_tuple(uint32_t(0), uint32_t(0));
    auto conv_problem_shape_mnl = replace<3>(conv_problem_shape, 1);
    auto problem_blocks_shape = TileScheduler::calculate_problem_blocks_shape(conv_problem_shape_mnl, TileShape{});
    auto scheduler = TileScheduler(&shared_tensors.clc_response[0], problem_blocks_shape, coop_set_ids);
    auto work_tile_info = scheduler.initial_work_tile_info();

    if (is_participant.main_load) {
      do {
        auto [m_coord, n_coord, _, l_coord] = scheduler.work_tile_to_cta_coord(work_tile_info);
        auto n_coord_ = idx2crd(n_coord, shape<2>(gB_nk), compact_col_major(shape<2>(gB_nk)));
        auto blk_coord = make_tuple(m_coord, n_coord_);

        auto work_id = local_id - NumMMAThreads - NumSchedThreads;
        mainloop_pipe_producer_state = collective_mainloop.load(params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state,
          load_inputs, blk_coord, k_tile_iter, k_tile_count, work_id,
          shared_tensors.mainloop);

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);

        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());
    } else if (is_participant.mma) {
      auto mma_inputs = collective_mainloop.mma_init(shared_tensors.mainloop);

      do {
        mainloop_pipe_consumer_state = collective_mainloop.mma(
          cute::make_tuple(mainloop_pipeline, epi_store_pipeline, accumulator_pipeline),
          cute::make_tuple(mainloop_pipe_consumer_state, epi_store_pipe_producer_state, accumulator_pipe_producer_state),
          intermedia_tensor, mma_inputs, k_tile_count);

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);

        ++epi_store_pipe_producer_state;
        ++accumulator_pipe_producer_state;
        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());
    } else if (is_participant.epi_load)  {
      int current_wave = 0;
      bool reverse_epi_n = false;
      static constexpr bool IsOverlappingAccum = false;
      auto prev_epi_store_consumer_state = epi_store_pipe_consumer_state;

      do {
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

        auto [load_state_next, store_cons_state_next] = collective_epilogue.template load<IsOverlappingAccum>(
          cute::make_tuple(epi_load_pipeline, epi_store_pipeline),
          cute::make_tuple(epi_load_pipe_producer_state, epi_store_pipe_consumer_state),
          conv_problem_shape,
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
        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());

      epi_store_pipeline.producer_try_acquire(prev_epi_store_consumer_state);
    }
    else if (is_participant.epilogue)  {
      do {
        auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

        auto [load_state_next, store_prod_state_next, acc_state_next] = collective_epilogue.store(
          cute::make_tuple(epi_load_pipeline, epi_store_pipeline, accumulator_pipeline),
          cute::make_tuple(epi_load_pipe_consumer_state, epi_store_pipe_producer_state, accumulator_pipe_consumer_state),
          conv_problem_shape,
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
        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());
    }
  }
};

}
