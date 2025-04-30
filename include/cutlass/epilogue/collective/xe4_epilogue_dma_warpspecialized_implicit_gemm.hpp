#pragma once

#include "cute/arch/copy_xe4_dma.hpp"
#include "cute/atom/copy_traits_xe4_dma.hpp"
#include "cute/container/array.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/pipeline/pipeline.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;
using namespace cute::detail;

constexpr slm_matrix_type cm_typeD = slm_matrix_type::type1;
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes them out to destination storage.
template <
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmaStore_,
  int NumControlWarps_,
  int NumEpilogueWarps_,
  class CtaTileShape_, // (CTA_M,CTA_N,CTA_K, optional: Tile_L)
  class EpilogueTile_, // (EPI_TILE_M, EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class FusionCallbacks_,
  class CopyOpG2S_,
  class SmemLayoutAtomC_,
  class CopyOpS2R_,
  class CopyOpS2RImm_,
  class CopyOpS2G_,
  class SmemLayoutAtomD_,
  class CopyOpR2S_,
  class CopyOpR2R_
>
class EpilogueConv{
public:
  using ElementD = ElementD_;
  using CtaTileShape = CtaTileShape_;
  using CopyOpS2G = CopyOpS2G_;
  using SmemLayoutAtomD = SmemLayoutAtomD_;
  using StrideD = StrideD_;

private:
  using GmemElementD = ElementD;
  constexpr static int StagesD = StagesD_;

  constexpr static bool is_im2col_D = cute::is_base_of_v<xe4::ASYNC_ROW_IM2COL, CopyOpS2G>;

public:
  using StorePipeline = cutlass::PipelineTmaAsync<StagesD>;
  using StorePipelineState = typename StorePipeline::PipelineState;

  static constexpr int NumControlWarps = 4;
  static constexpr int NumEpilogueWarps = 0;

  struct SharedStorage
  {
    struct TensorStorage
    {
      cute::array<ElementD, cute::cosize_v<SmemLayoutAtomD>> smem_D;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;

  static constexpr uint32_t TmaTransactionBytes =
    (size<0>(SmemLayoutAtomD{}) * size<1>(SmemLayoutAtomD{}) * static_cast<uint32_t>(sizeof(ElementD)));

  // Host side epilogue arguments
  struct Arguments {
    ElementD const* ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  template <class ProblemShapeMNL>
  static constexpr auto
  get_tma_store_d(ProblemShapeMNL const& problem_shape_mnl, Arguments const& args)
  {
    Tensor tensor_d = make_tensor(args.ptr_D, 
                                  make_layout(problem_shape_mnl, args.dD));
    return make_tma_copy<CopyOpS2G_>(CopyOpS2G_{}, tensor_d, SmemLayoutAtomD{}, take<0,2>(CtaTileShape{}), _1{});
  }

  struct Params
  {
    using TMA_D = decltype(get_tma_store_d(repeat_like(take<0,2>(StrideD_{}), int32_t(0)), Arguments{}));

    TMA_D tma_store_d;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
  };

  //
  // Methods
  //
  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    auto shape_D_orig = problem_shape.get_shape_C();
    auto tma_store_d = get_tma_store_d(shape_D_orig, args);
    return {tma_store_d, TmaTransactionBytes};
  }

    //
  // Constructor and Data Members
  //
  template <class Params>
  CUTLASS_DEVICE
  EpilogueConv(
    Params const& params_,
    TensorStorage& shared_tensors)
      : params(params_) {
  }

private:
    Params const& params;

public:
  template<class ProblemShapeMNKL, class CtaCoordMNKL, class CtaTileMNK>
  CUTLASS_DEVICE auto
  store(
    StorePipeline store_pipeline,
    StorePipelineState store_pipe_consumer_state,
    ProblemShapeMNKL const& problem_shape_MNKL,
    CtaTileMNK cta_tile_mnk,
    CtaCoordMNKL const& cta_coord_mnkl,
    TensorStorage& shared_tensors)
  {
    auto sD = make_tensor(shared_tensors.smem_D.data(), SmemLayoutAtomD {});

    auto [M, N, K, L] = problem_shape_MNKL;
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;

    auto coord_shape =
      conditional_return<is_im2col_D>(make_coord(m_coord, n_coord), make_coord(m_coord, n_coord, l_coord));

    uint32_t local_id = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_linear_id();
    uint32_t worker_id = local_id - NumControlWarps * NumThreadsPerWarp;
    uint32_t thread_idx = local_id - 3 * NumThreadsPerWarp;

    Tensor mD_mn = params.tma_store_d.get_tma_tensor(make_shape(M,N));
    auto thr_store_d = params.tma_store_d.get_slice(thread_idx);
    Tensor gD = local_tile(mD_mn, take<0,2>(cta_tile_mnk), coord_shape);

    Tensor tDsD = thr_store_d.partition_S(sD);
    Tensor tDgD = thr_store_d.partition_D(gD);

    store_pipeline.consumer_try_wait(store_pipe_consumer_state);
    if(elect_one_sync()) {
        store_pipeline.consumer_commit(store_pipe_consumer_state, params.tma_transaction_bytes);
    }

    auto abar_store = store_pipeline.consumer_get_barrier(store_pipe_consumer_state);
    copy(params.tma_store_d.with(abar_store), tDsD, tDgD);
    store_pipeline.producer_try_acquire(store_pipe_consumer_state);
    ++store_pipe_consumer_state;

    return store_pipe_consumer_state;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////