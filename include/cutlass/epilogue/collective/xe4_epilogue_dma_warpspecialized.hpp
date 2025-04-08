#pragma once

#include "cute/arch/copy_xe4_dma.hpp"
#include "cute/atom/copy_traits_xe4_dma.hpp"
#include "cute/container/array.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/xe4_detail.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cute::detail;
using namespace cutlass::epilogue;
using namespace cutlass::epilogue::thread::detail;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes them out to destination storage.
template <
  int FragmentSize,
  int NumControlWarps_,
  int NumEpilogueWarps_,
  class ElementD_,
  class StrideD_,
  class SmemLayoutD_,
  class TileShape_,
  class FusionCallbacks_
>
class CollectiveEpilogue<
  Xe4DmaWarpSpecialized<FragmentSize, NumControlWarps_, NumEpilogueWarps_>,
  ElementD_,
  StrideD_,
  SmemLayoutD_,
  TileShape_,
  FusionCallbacks_
> {
public:
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using TileShape = TileShape_;
  using FusionCallbacks = FusionCallbacks_;
  using ThreadEpilogueOp = typename fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;

  using TiledCopyD = cute::xe4::ASYNC_TENSOR_STORE<slm_matrix_type::type1>;

  using AccumulatorPipeline = cutlass::PipelineTmaAsync<1>;
  using AccumulatorPipelineState = typename AccumulatorPipeline::PipelineState;

  using StorePipeline = cutlass::PipelineTmaAsync<1>;
  using StorePipelineState = typename StorePipeline::PipelineState;

  using SmemLayoutAtomD = decltype(make_ordered_layout(select<0,1>(TileShape{}), Step<_1, _0>{}));
  using SmemLayoutD = decltype(tile_to_shape(SmemLayoutAtomD{}, take<0,2>(TileShape{}), Step<_2,_1>{}));

  static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

  struct SharedStorage
  {
    struct TensorStorage
    {
      cute::array<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;

      using FusionStorage = typename FusionCallbacks::SharedStorage;
      FusionStorage thread;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;

  static constexpr int NumControlWarps = NumControlWarps_;
  static constexpr int NumEpilogueWarps = NumEpilogueWarps_;

  static constexpr uint32_t TransactionBytesStore = sizeof(ElementD) * size(SmemLayoutD {});

  // Host side epilogue arguments
  struct Arguments {
    ElementD const* ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  struct Params
  {
    using TiledStoreD = decltype(make_tma_copy(
      TiledCopyD{},
      make_tensor(static_cast<ElementD const*>(nullptr), repeat_like(StrideD{}, int32_t(0)), StrideD{}),
      SmemLayoutD{}, take<0,2>(TileShape{})));

    TiledStoreD store_d;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  CUTLASS_HOST
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {

    auto [M, N, K, L] = problem_shape;
    auto D = make_tensor(args.ptr_D, make_layout(make_shape(M, N, L), args.dD));
    auto store_d = make_tma_copy(TiledCopyD{}, D, SmemLayoutD{});

    return {
      store_d
    };
  }

  // Note: SharedStorage is unused for CollectiveEpilogue
  template <class TensorDesc>
  CUTLASS_DEVICE
  CollectiveEpilogue(Params const& params, TensorDesc tensor_desc) : _params(params) {
    params.store_d.cache_.set_tensor_desc(tensor_desc);
  }

  template <class ProblemShape>
  CUTLASS_DEVICE void
  operator()(ProblemShape const& problem_shape,
    cute::tuple<StorePipeline, AccumulatorPipeline> pipelines,
    cute::tuple<StorePipelineState, AccumulatorPipelineState> pipeline_states,
    TensorStorage& shared_tensors, uint32_t local_id)
  {
    uint32_t worker_id = local_id - NumControlWarps * NumThreadsPerWarp;

    auto [store_pipeline, accumulator_pipeline] = pipelines;
    auto [store_pipe_producer_state, accumulator_pipe_consumer_state] = pipeline_states;

    accumulator_pipeline.consumer_try_wait(accumulator_pipe_consumer_state);
    accumulator_pipeline.consumer_release(accumulator_pipe_consumer_state);

    constexpr auto tile_mn = take<0,2>(TileShape{});
    auto tensor_d = make_tensor(shared_tensors.smem_D.data(), CoreMatrix::retile<ElementD>(tile_mn));

    auto empty_tuple = cute::tuple<>{};
    auto dummy_tensor = make_tensor<float>(Int<1>{});
    auto cst_args = fusion::detail::ConsumerStoreArgs(
      Shape<_512,_512,_512,_1>{},   // ProblemShapeMNKL
      append<3>(TileShape{}, _1{}),      // TileShapeMNK
      make_coord(0,0,0,0),           // TileCoordMNKL
      empty_tuple,      // TiledMma
      Shape<_2,_2>{},   // EpilogueTile
      empty_tuple,      // TiledCopy
      empty_tuple,      // CoordTensor
      empty_tuple,      // Residue
      empty_tuple,      // ThrCoordTensor
      empty_tuple,      // ThrResidue
      dummy_tensor,     // ThrSrcTensor
      worker_id          // thread_idx
    );


    auto callback_args = typename FusionCallbacks::Arguments{};
    auto callback_params = FusionCallbacks::to_underlying_arguments(problem_shape, callback_args, nullptr);

    FusionCallbacks fusion_callbacks(callback_params, shared_tensors.thread);
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<true>(cst_args);
    pattern2<FragmentSize, NumEpilogueWarps>(cst_callbacks, tensor_d, tensor_d, worker_id);

    store_pipeline.producer_commit(store_pipe_producer_state, 1);
  }

  template<
    class ProblemShape,
    class BlockCoordMNL
  >
  CUTLASS_DEVICE void
  store(
      StorePipeline store_pipeline,
      StorePipelineState& store_pipe_state,
      ProblemShape const& problem_shape,
      BlockCoordMNL blk_coord_mnl,
      TensorStorage& shared_tensors)
  {
    store_pipeline.consumer_try_wait(store_pipe_state);       // Wait for all postop threads finish their calculation

    auto sD = make_tensor(shared_tensors.smem_D.data(), SmemLayoutD {});
    auto [M, N, K, L] = problem_shape;
    auto mD_mnl = _params.store_d.get_tma_tensor(make_shape(M, N, L)); // (m,n,l)
    auto gD_mnl = flat_divide(mD_mnl, take<0,2>(TileShape {})); // (BLK_M,BLK_N,m,n,l)
    auto block_store_d = _params.store_d.get_slice(0);
    auto [m_coord, n_coord, l_coord] = blk_coord_mnl;

    auto gD = gD_mnl(_, _, m_coord, n_coord, l_coord);  // (BLK_M,BLK_N)
    auto tDgD = block_store_d.partition_S(gD);           // (TMA,TMA_M,TMA_N)
    auto tDsD = block_store_d.partition_D(sD);    // (TMA,TMA_M,TMA_N)

    auto abar_store = store_pipeline.consumer_get_barrier(store_pipe_state);
    copy(_params.store_d.with(abar_store), tDsD, tDgD);
    store_pipeline.consumer_commit(store_pipe_state, TransactionBytesStore);
    store_pipeline.producer_try_acquire(store_pipe_state);
    ++store_pipe_state;
  }

private:
  Params const& _params;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////