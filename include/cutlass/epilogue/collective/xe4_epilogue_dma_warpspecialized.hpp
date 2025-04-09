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
  class CopyOpS2G_,
  class SmemLayoutAtomD_,
  class CopyOpR2S_,
  class CopyOpR2R_
>
class CollectiveEpilogue<
  Xe4DmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_, NumControlWarps_, NumEpilogueWarps_>,
  CtaTileShape_,
  EpilogueTile_,
  ElementC_,
  StrideC_,
  ElementD_,
  StrideD_,
  FusionCallbacks_,
  CopyOpG2S_,
  SmemLayoutAtomC_,
  CopyOpS2R_,
  CopyOpS2G_,
  SmemLayoutAtomD_,
  CopyOpR2S_,
  CopyOpR2R_
> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = Xe4DmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_, NumControlWarps_, NumEpilogueWarps_>;
  using CtaTileShape = CtaTileShape_;
  using EpilogueTile = EpilogueTile_;
  using FusionCallbacks = FusionCallbacks_;
  using ElementC = ElementC_;
  using StrideC = StrideC_;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using CopyOpG2S = CopyOpG2S_;
  using SmemLayoutAtomC = SmemLayoutAtomC_;
  using CopyOpS2R = CopyOpS2R_;
  using CopyOpS2G = CopyOpS2G_;
  using SmemLayoutAtomD = SmemLayoutAtomD_;
  using CopyOpR2S = CopyOpR2S_;
  using CopyOpR2R = CopyOpR2R_;

  using ThreadEpilogueOp = typename fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;
  using GmemTiledCopyD = CopyOpS2G;

  static_assert(!is_layout<EpilogueTile>::value && is_tuple<EpilogueTile>::value, "EpilogueTile must be a cute::Tile or cute::Shape");
  static_assert(rank(EpilogueTile{}) == 2, "EpilogueTile must be rank-2: [EPI_TILE_M, EPI_TILE_N]");

private:
  using GmemElementD = ElementD;
  using GmemElementC = cute::conditional_t<cute::is_void_v<ElementC>,ElementD,ElementC>; // prevents void ref breakages
  using SmemElementD = typename cutlass::detail::get_unpacked_element_type<GmemElementD>::type;
  using SmemElementC = typename cutlass::detail::get_unpacked_element_type<GmemElementC>::type;

  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  static_assert(StagesC >= 1, "StagesC must be >= 1");
  static_assert(StagesD >= 1, "StagesD must be >= 1");

  constexpr static bool ReuseSmemC = ReuseSmemC_;
  constexpr static bool DelayTmaStore = DelayTmaStore_;
  constexpr static bool is_source_supported = not cute::is_void_v<ElementC>;

  constexpr static bool is_m_major_C = detail::is_m_major<StrideC>();
  constexpr static bool is_m_major_D = detail::is_m_major<StrideD>();

  constexpr static bool is_im2col_C = cute::is_base_of_v<xe4::ASYNC_ROW_IM2COL, CopyOpG2S>;
  constexpr static bool is_im2col_D = cute::is_base_of_v<xe4::ASYNC_ROW_IM2COL, CopyOpS2G>;

  using SmemLayoutStageC = decltype(tile_to_shape(SmemLayoutAtomC{}, product_each(shape(EpilogueTile{})),
      cute::conditional_t<is_m_major_C, Step<_1,_2>, Step<_2,_1>>{} ));
  using SmemLayoutStageD = decltype(tile_to_shape(SmemLayoutAtomD{}, product_each(shape(EpilogueTile{})),
    cute::conditional_t<is_m_major_D, Step<_1,_2>, Step<_2,_1>>{} ));

  constexpr static int StageCBits = cosize_v<SmemLayoutStageC> * sizeof_bits_v<SmemElementC>;
  constexpr static int StageDBits = cosize_v<SmemLayoutStageD> * sizeof_bits_v<SmemElementD>;
  constexpr static int MaxStageBits = cute::max(StageCBits, StageDBits);
  constexpr static int StrideStageC = (ReuseSmemC ? MaxStageBits : StageCBits) / sizeof_bits_v<SmemElementC>;
  constexpr static int StrideStageD = (ReuseSmemC ? MaxStageBits : StageDBits) / sizeof_bits_v<SmemElementD>;

  using SmemLayoutC = decltype(cute::append<3>(SmemLayoutStageC{}, Layout<Int<StagesC>,                        Int<StrideStageC>>{}));
  using SmemLayoutD = decltype(cute::append<3>(SmemLayoutStageD{}, Layout<Int<ReuseSmemC ? StagesC : StagesD>, Int<StrideStageD>>{}));

public:
  // TMA pipeline for loading C
  using LoadPipeline = cutlass::PipelineTmaAsync<StagesC>;;
  using LoadPipelineState = typename LoadPipeline::PipelineState;
  constexpr static uint32_t TmaTransactionBytes = cutlass::bits_to_bytes(StageCBits);

  // TMA pipeline for storing D
  using StorePipeline = cutlass::PipelineTmaAsync<StagesD>;
  using StorePipelineState = typename StorePipeline::PipelineState;

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
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
  };

private:
  static constexpr auto
  get_tma_epi_tile() {
    return cute::transform_apply(EpilogueTile{}, seq<0,1>{},
      [] (auto epi_tiler, auto mode) {
        auto cta_tiler_shape = get<mode>(CtaTileShape{});
        // Use a dynamic stride to prevent mode coalescing
        auto cta_tiler_stride = repeat_like(cta_tiler_shape, 0);
        auto cta_tiler = make_layout(cta_tiler_shape, cta_tiler_stride);
        // This is a multimodal CTA tiler, transform before returning
        if constexpr (depth(cta_tiler) > 0) {
          // This is an implicit multimodal tiler, match profile and return
          if constexpr (tuple_size_v<decltype(shape(cta_tiler))> == 1) {
            return make_tile(epi_tiler);
          }
          // This is an explicit multimodal tiler, compose out epi tiler
          else {
            return shape(composition(cta_tiler, epi_tiler));
          }
        }
        // This is a flat CTA tiler, no need for transformation
        else {
          return epi_tiler;
        }
      },
      [] (auto... epi_tilers) {
        return make_tile(epi_tilers...);
      }
    );
  }

  using TmaEpilogueTile = decltype(get_tma_epi_tile());

  template <class ProblemShapeMNL>
  static constexpr auto
  get_tma_load_c(ProblemShapeMNL const& problem_shape_mnl, Arguments const& args) {
    Tensor tensor_c = make_tensor(make_gmem_ptr<GmemElementC>(args.ptr_C),
                                  make_layout(problem_shape_mnl, append<3>(args.dC, _0{})));
    return make_tma_copy(CopyOpG2S{}, tensor_c, SmemLayoutStageC{}, TmaEpilogueTile{}, _1{});
  }

  template <class ProblemShapeMNL>
  static constexpr auto
  get_tma_store_d(ProblemShapeMNL const& problem_shape_mnl, Arguments const& args) {
    Tensor tensor_d = make_tensor(make_gmem_ptr<GmemElementD>(args.ptr_D),
                                  make_layout(problem_shape_mnl, append<3>(args.dD, _0{})));
    return make_tma_copy(CopyOpS2G{}, tensor_d, SmemLayoutStageD{}, TmaEpilogueTile{}, _1{});
  }

public:
  // Device side epilogue params
  struct Params {
    using TMA_C = decltype(get_tma_load_c (repeat_like(append<3>(StrideC{},_1{}), int32_t(0)), Arguments{}));
    using TMA_D = decltype(get_tma_store_d(repeat_like(append<3>(StrideD{},_1{}), int32_t(0)), Arguments{}));

    typename FusionCallbacks::Params thread{};
    TMA_C tma_load_c;
    TMA_D tma_store_d;
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
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_mnl = select<0,1,3>(append<4>(problem_shape, 1));
    typename Params::TMA_C tma_load_c{};
    if constexpr (is_source_supported) {
      tma_load_c = get_tma_load_c(problem_shape_mnl, args);
    }

    typename Params::TMA_D tma_store_d = get_tma_store_d(problem_shape_mnl, args);

    auto callback_args = typename FusionCallbacks::Arguments{};

    return {
      FusionCallbacks::to_underlying_arguments(problem_shape, callback_args, nullptr),
      tma_load_c,
      tma_store_d
    };
  }

  CUTLASS_DEVICE
  static constexpr auto
  get_intermedia_tensor(TensorStorage& shared_tensors) {
    auto tensor_d = make_tensor(shared_tensors.smem_D.data(), SmemLayoutD{});
    return tensor_d;
  }

  // Note: SharedStorage is unused for CollectiveEpilogue
  template <class TensorDesc>
  CUTLASS_DEVICE
  CollectiveEpilogue(Params const& params_, TensorStorage& shared_tensors, TensorDesc tensor_desc)
      : params(params_), fusion_callbacks(params_.thread, shared_tensors.thread) {
    params.tma_store_d.cache_.set_tensor_desc(tensor_desc);
  }

  template<
    class Pipelines,
    class PipelineStates,
    class ProblemShapeMNKL,
    class CtaCoordMNL
  >
  CUTLASS_DEVICE auto
  store(
      Pipelines pipelines,
      PipelineStates pipeline_states,
      ProblemShapeMNKL problem_shape_mnkl,
      CtaCoordMNL cta_coord_mnl,
      TensorStorage& shared_tensors)
  {
    uint32_t local_id = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_linear_id();
    uint32_t worker_id = local_id - NumControlWarps * NumThreadsPerWarp;

    auto [store_pipeline, accumulator_pipeline] = pipelines;
    auto [store_pipe_producer_state, store_pipe_consumer_state, accumulator_pipe_consumer_state] = pipeline_states;

    accumulator_pipeline.consumer_try_wait(accumulator_pipe_consumer_state);
    accumulator_pipeline.consumer_release(accumulator_pipe_consumer_state);

    auto ptr_sD = shared_tensors.smem_D.data();
    auto tensor_d = make_tensor(ptr_sD, CoreMatrix::retile<ElementD>(take<0,2>(CtaTileShape{})));

    auto empty_tuple = cute::tuple<>{};
    auto dummy_tensor = make_tensor<float>(Int<1>{});
    auto cst_args = fusion::detail::ConsumerStoreArgs(
      Shape<_512,_512,_512,_1>{},   // ProblemShapeMNKL
      append<3>(CtaTileShape{}, _1{}),      // TileShapeMNK
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

    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<true>(cst_args);
    pattern2<FragmentSize_, NumEpilogueWarps>(cst_callbacks, tensor_d, tensor_d, worker_id);

    store_pipeline.producer_commit(store_pipe_producer_state, 1);

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, l_coord] = cta_coord_mnl;

    // Represent the full output tensor, slice to get the tile this CTA is responsible for
    auto mD_mnl = params.tma_store_d.get_tma_tensor(make_shape(M, N, L));                                  //       (M,N,L)
    auto gD_mnl = flat_divide(mD_mnl, take<0,2>(CtaTileShape {})); // (BLK_M,BLK_N,m,n,l)

    // Construct the corresponding pipelined smem tensors
    auto sD = make_tensor(ptr_sD, SmemLayoutD{});
    auto block_store_d = params.tma_store_d.get_slice(0);

    auto gD = gD_mnl(_, _, m_coord, n_coord, l_coord);  // (BLK_M,BLK_N)
    auto tDgD = block_store_d.partition_S(gD);           // (TMA,TMA_M,TMA_N)
    auto tDsD = block_store_d.partition_D(sD);    // (TMA,TMA_M,TMA_N)

    store_pipeline.consumer_try_wait(store_pipe_consumer_state);       // Wait for all postop threads finish their calculation
    auto abar_store = store_pipeline.consumer_get_barrier(store_pipe_consumer_state);
    if (worker_id == 0) {
      copy(params.tma_store_d.with(abar_store), tDsD(_,_,_,store_pipe_producer_state.index()), tDgD);
      store_pipeline.consumer_commit(store_pipe_consumer_state, TransactionBytesStore);
    }

    ++store_pipe_producer_state;
    ++store_pipe_consumer_state;
    ++accumulator_pipe_consumer_state;

    return make_tuple(
      store_pipe_producer_state,
      store_pipe_consumer_state,
      accumulator_pipe_consumer_state
    );
  }

private:
  Params const& params;
  FusionCallbacks fusion_callbacks;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////