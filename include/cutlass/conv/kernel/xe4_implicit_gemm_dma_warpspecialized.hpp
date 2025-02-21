#pragma once

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include "cute/arch/copy_xe4_dma.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/xe4_epilogue_dma_warpspecialized_implicit_gemm.hpp"

namespace cutlass::conv::kernel {

using namespace cute;
using namespace sycl;
using namespace cute::xe4;
// using namespace cutlass::gemm::kernel;

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
  using SmemLayoutC = decltype(make_layout(take<0,2>(TileShape{}), GenRowMajor{}));
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // CLC pipeline depth
  // determines how many waves (stages-1) a warp can race ahead
  static constexpr uint32_t SchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using ElementD = typename CollectiveEpilogue::ElementD;

  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename cutlass::gemm::kernel::detail::TileSchedulerSelector<
    TileSchedulerTag, ArchTag, TileShape, ClusterShape, SchedulerPipelineStageCount>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;

  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using EpilogueStorePipeline = typename CollectiveEpilogue::EpilogueStorePipeline;
  using MainloopPipelineState = typename CollectiveMainloop::PipelineState;
  using EpilogueStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

  using CLCPipeline = cutlass::xe4::PipelineTmaAsync<SchedulerPipelineStageCount>;
  using CLCPipelineState = typename CLCPipeline::PipelineState;

  using CLCThrottlePipeline = cutlass::xe4::PipelineTmaAsync<SchedulerPipelineStageCount>;
  using CLCThrottlePipelineState = typename CLCThrottlePipeline::PipelineState;

  struct SharedStorage
  {
    using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
    using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
    using CLCPipelineStorage = typename CLCPipeline::SharedStorage;
    using CLCThrottlePipelineStorage = typename CLCThrottlePipeline::SharedStorage;

    struct TensorStorage
    {
      alignas(64) MainloopTensorStorage mainloop;
      alignas(512) EpilogueTensorStorage epilogue;
      alignas(512) cute::array<ElementAccumulator, cute::cosize_v<SmemLayoutC>> smem_Acc;
      alignas(16) typename TileScheduler::CLCResponse clc_response[SchedulerPipelineStageCount];
    } tensors;

    struct PipelineStorage {
      using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;
      using EpiStorePipelineStorage = typename EpilogueStorePipeline::SharedStorage;

      MainloopPipelineStorage mainloop;
      EpiStorePipelineStorage epi_store;
      CLCPipelineStorage clc;
      CLCThrottlePipelineStorage clc_throttle;
    } pipelines;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static constexpr int PipelineStorageSize = sizeof(typename SharedStorage::PipelineStorage);

  struct GroupInfo {
    uint32_t subgroup_size = 0;
    uint32_t mma_subgroup_num = 0;
    uint32_t sched_subgroup_num = 0;
    uint32_t load_subgroup_num = 0;
    uint32_t store_subgroup_num = 0;
    uint32_t mainloop_subgroup_num = 0;
    uint32_t epilogue_subgroup_num = 0;
  };

  // Device side arguments
  struct Arguments {
    GroupInfo group_info;
    ProblemShape problem_shape;
    MainloopArguments mainloop;
    EpilogueArguments epilogue;
  };

  // Kernel entry point API
  struct Params {
    GroupInfo group_info;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  enum class WarpCategory : int32_t {
    MMA          = 0,
    Sched        = 1,
    MainloopLoad = 2,
    EpilogueStore = 3,
    Epilogue     = 4
  };

  struct IsParticipant {
    uint32_t mma       = false;
    uint32_t sched     = false;
    uint32_t main_load = false;
    uint32_t epi_store  = false;
    uint32_t epilogue  = false;
  };

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    auto tdesc_b = allocate_tdesc<0>();

    auto scheduler_args = typename TileScheduler::Arguments {
      {CollectiveMainloop::SlmBytesA, CollectiveMainloop::SlmBytesB}
    };

    CollectiveMainloop collective_mainloop;
    auto conv_problem_shape = collective_mainloop.get_problem_shape_MNKL(args.problem_shape);
    auto problem_shape = replace<3>(conv_problem_shape, 1);

    return {
      args.group_info,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, tdesc_b, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
      TileScheduler::to_underlying_arguments(problem_shape, TileShape{}, ClusterShape{},
        scheduler_args)
    };
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, sycl::nd_item<3> item) {
    auto& group_info = params.group_info;
    auto& problem_shape = params.problem_shape;

    // Warp specialization thread count per threadblock
    uint32_t SubGroupSize            = group_info.subgroup_size;
    uint32_t NumSchedThreads         = group_info.sched_subgroup_num * SubGroupSize;
    uint32_t NumMMAThreads           = group_info.mma_subgroup_num * SubGroupSize;
    uint32_t NumMainloopLoadThreads  = group_info.load_subgroup_num * SubGroupSize;
    uint32_t NumEpilogueStoreThreads = group_info.store_subgroup_num * SubGroupSize;
    uint32_t NumEpilogueThreads      = group_info.epilogue_subgroup_num * SubGroupSize;

    auto ptr = alloc_slm_buffer<uint8_t, SharedStorageSize>(item.get_group());
    auto& shared_tensors = *reinterpret_cast<typename SharedStorage::TensorStorage*>(ptr);

    uint32_t local_id = item.get_local_linear_id();
    uint32_t warp_idx = get_sg_id();
    WarpCategory warp_category;
    if (local_id == 0) {
      warp_category = WarpCategory::MMA;
    } else if (warp_idx < static_cast<int>(WarpCategory::Epilogue))
    {
      warp_category = WarpCategory(warp_idx);
    } else {
      warp_category = WarpCategory::Epilogue;
    }

    // Do we load source tensor C or other aux inputs
    bool is_epi_load_needed = false;
    bool is_first_cta_in_cluster = true;
    IsParticipant is_participant = {
      (warp_category == WarpCategory::MMA),                                 // mma
      (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster,    // sched
      (warp_category == WarpCategory::MainloopLoad),                        // main_load
      (warp_category == WarpCategory::EpilogueStore),                       // epi_store
      (warp_category == WarpCategory::Epilogue)                             // epilogue
    };

    auto abar_base = allocate_abar_bytes<0, PipelineStorageSize>();
    auto& shared_pipelines = *reinterpret_cast<typename SharedStorage::PipelineStorage*>(abar_base);

    typename MainloopPipeline::Params mainloop_pipeline_params;
    mainloop_pipeline_params.initializing_warp = 0;
    MainloopPipeline mainloop_pipeline(shared_pipelines.mainloop, mainloop_pipeline_params);

    typename EpilogueStorePipeline::Params epilogue_store_pipeline_params;
    epilogue_store_pipeline_params.initializing_warp = 1;
    EpilogueStorePipeline epilogue_store_pipeline(shared_pipelines.epi_store, epilogue_store_pipeline_params);

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue;

    // CLC pipeline
    typename CLCPipeline::Params clc_pipeline_params;
    clc_pipeline_params.initializing_warp = 2;
    clc_pipeline_params.num_producers = NumSchedThreads;
    clc_pipeline_params.num_consumers = NumSchedThreads + NumMMAThreads + NumMainloopLoadThreads + NumEpilogueStoreThreads + NumEpilogueThreads;
    CLCPipeline clc_pipeline(shared_pipelines.clc, clc_pipeline_params);

    // CLC throttle pipeline
    typename CLCThrottlePipeline::Params clc_throttle_pipeline_params;
    clc_throttle_pipeline_params.initializing_warp = 3;
    clc_throttle_pipeline_params.num_producers = group_info.subgroup_size;
    clc_throttle_pipeline_params.num_consumers = group_info.subgroup_size;
    CLCThrottlePipeline clc_throttle_pipeline(shared_pipelines.clc_throttle, clc_throttle_pipeline_params);

    MainloopPipelineState mainloop_pipe_consumer_state;
    EpilogueStorePipelineState epilogue_pipe_store_consumer_state;
    auto mainloop_pipe_producer_state = cutlass::xe4::make_producer_start_state<MainloopPipeline>();
    auto epilogue_pipe_store_producer_state = cutlass::xe4::make_producer_start_state<EpilogueStorePipeline>();

    auto clc_pipe_throttle_producer_state = cutlass::xe4::make_producer_start_state<CLCThrottlePipeline>();
    auto clc_pipe_throttle_consumer_state = CLCThrottlePipelineState{};

    auto clc_pipe_producer_state = cutlass::xe4::make_producer_start_state<CLCPipeline>();
    auto clc_pipe_consumer_state = CLCPipelineState{};

    using SmemLayoutAtomAcc = decltype(make_layout(Shape<_32, Int<32 / sizeof(ElementAccumulator)>>{}, GenRowMajor{}));
    using SmemLayoutAtomDst = decltype(make_layout(Shape<_32, Int<32 / sizeof(ElementD)>>{}, GenRowMajor{}));

    using SmemLayoutAcc = decltype(tile_to_shape(
        SmemLayoutAtomAcc{},
        make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})),
        Step<_2,_1>{}));
    using SmemLayoutDst = decltype(tile_to_shape(
        SmemLayoutAtomDst{},
        make_shape(shape<0>(TileShape{}), shape<1>(TileShape{})),
        Step<_2,_1>{}));

    auto accumulator = make_tensor(reinterpret_cast<ElementAccumulator*>(shared_tensors.smem_Acc.data()), SmemLayoutAcc {});
    auto dst = make_tensor(reinterpret_cast<ElementD*>(shared_tensors.epilogue.smem_D.data()), SmemLayoutDst {});

    auto conv_problem_shape = collective_mainloop.get_problem_shape_MNKL(problem_shape);

    auto load_inputs = collective_mainloop.load_init(conv_problem_shape, params.mainloop);
    auto [gA_mk, gB_nk] = load_inputs;
    auto k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mk));
    auto k_tile_count = size<3>(gA_mk);

    item.barrier(access::fence_space::local_space);

    auto scheduler = TileScheduler(&shared_tensors.clc_response[0], params.scheduler);
    auto work_tile_info = scheduler.initial_work_tile_info();

    if (is_participant.main_load) {
      do {
        auto [m_coord, n_coord, _] = scheduler.work_tile_to_cta_coord(work_tile_info);
        n_coord = idx2crd(n_coord, shape<2>(gB_nk), compact_col_major(shape<2>(gB_nk)));
        auto blk_coord = make_tuple(m_coord, n_coord);

        auto work_id = local_id - (group_info.mma_subgroup_num + group_info.sched_subgroup_num) * group_info.subgroup_size;
        mainloop_pipe_producer_state = collective_mainloop.load(params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state,
          load_inputs, blk_coord, k_tile_iter, k_tile_count, work_id,
          shared_tensors.mainloop);

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);

        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());
    } else if (is_participant.mma) {
      do {
        mainloop_pipe_consumer_state = collective_mainloop.mma(mainloop_pipeline, mainloop_pipe_consumer_state,
          epilogue_store_pipeline, epilogue_pipe_store_producer_state, accumulator, dst,
          k_tile_count, ptr);

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);

        ++epilogue_pipe_store_producer_state;
        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());
    } else if (is_participant.epi_store)  {
      uint32_t work_id = local_id - (group_info.mma_subgroup_num + group_info.sched_subgroup_num + group_info.load_subgroup_num) * group_info.subgroup_size;
      do {
        auto [m_coord, n_coord, _] = scheduler.work_tile_to_cta_coord(work_tile_info);
        auto blk_coord = make_coord(m_coord, n_coord);
        collective_epilogue.store(params.epilogue, epilogue_store_pipeline,
          epilogue_pipe_store_consumer_state, conv_problem_shape, blk_coord, work_id,
          shared_tensors.epilogue);

        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);

        ++epilogue_pipe_store_consumer_state;
        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());
    }
  }
};

}
