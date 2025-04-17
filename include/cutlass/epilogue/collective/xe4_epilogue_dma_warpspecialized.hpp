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
  using SmemLayoutStageD = decltype(tile_to_shape(SmemLayoutAtomD{}, take<0,2>(CtaTileShape{}),
    cute::conditional_t<is_m_major_D, Step<_1,_2>, Step<_2,_1>>{} ));

  constexpr static int StageCBits = cosize_v<SmemLayoutStageC> * sizeof_bits_v<SmemElementC>;
  constexpr static int StageDBits = cosize_v<SmemLayoutStageD> * sizeof_bits_v<SmemElementD>;
  constexpr static int MaxStageBits = cute::max(StageCBits, StageDBits);
  constexpr static int StrideStageC = (ReuseSmemC ? MaxStageBits : StageCBits) / sizeof_bits_v<SmemElementC>;
  constexpr static int StrideStageD = (ReuseSmemC ? MaxStageBits : StageDBits) / sizeof_bits_v<SmemElementD>;

  using SmemLayoutC = decltype(cute::append<3>(SmemLayoutStageC{}, Layout<Int<StagesC>,                        Int<StrideStageC>>{}));
  using SmemLayoutD = decltype(cute::append<3>(SmemLayoutStageD{}, Layout<Int<ReuseSmemC ? StagesC : StagesD>, Int<StrideStageD>>{}));

  constexpr static size_t SmemAlignmentC = 512;
  constexpr static size_t SmemAlignmentD = 512;
  constexpr static size_t MaxSmemAlignment = cute::max(SmemAlignmentC, SmemAlignmentD);

  struct CollectiveStorageWithC {
    alignas(SmemAlignmentC) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  union CollectiveStorageWithoutC {
    cute::array<SmemElementC, 0> smem_C;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  union CollectiveStorageReuseC {
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

public:
  // TMA pipeline for loading C
  using LoadPipeline = cutlass::PipelineTransactionAsync<StagesC>;;
  using LoadPipelineState = typename LoadPipeline::PipelineState;
  constexpr static uint32_t TmaTransactionBytes = cutlass::bits_to_bytes(StageCBits);

  // TMA pipeline for storing D
  using StorePipeline = cutlass::PipelineTmaAsync<StagesD>;
  using StorePipelineState = typename StorePipeline::PipelineState;

  static_assert(ReuseSmemC==false, "Not support ReuseSmemC yet");

  struct SharedStorage
  {
    struct TensorStorage
    {
      using CollectiveStorage = cute::conditional_t<not is_source_supported, CollectiveStorageWithoutC,
                                  cute::conditional_t<ReuseSmemC, CollectiveStorageReuseC, CollectiveStorageWithC>>;
      CollectiveStorage collective;

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
    return make_tma_copy(CopyOpS2G{}, tensor_d, SmemLayoutD{}, take<0,2>(CtaTileShape{}), _1{});
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
    auto ptr_sD = shared_tensors.collective.smem_D.begin();
    auto tensor_d = make_tensor(make_smem_ptr(ptr_sD), SmemLayoutD{});
    return tensor_d;
  }

  //
  // Constructor and Data Members
  //
  template <class TensorDescTuple>
  CUTLASS_DEVICE
  CollectiveEpilogue(Params const& params_, TensorStorage& shared_tensors, TensorDescTuple tdesc_tuple)
      : params(params_), fusion_callbacks(params_.thread, shared_tensors.thread) {
    auto [tensor_desc_c, tensor_desc_d] = tdesc_tuple;
    params.tma_load_c.cache_.set_tensor_desc(tensor_desc_c);
    params.tma_store_d.cache_.set_tensor_desc(tensor_desc_d);
  }

private:
  Params const& params;
  FusionCallbacks fusion_callbacks;

  //
  // Non-static Device Functions
  //
public:
  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return fusion_callbacks.is_producer_load_needed();
  }

  template<
    bool ReuseTmem = false,
    class Pipelines,
    class PipelineStates,
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class MmaTileMNK,
    class TiledMma
  >
  CUTLASS_DEVICE auto
  load(
    Pipelines pipelines,
    PipelineStates pipeline_states,
    ProblemShapeMNKL problem_shape_mnkl,
    CtaTileMNK cta_tile_mnk,
    CtaCoordMNKL cta_coord_mnkl,
    MmaTileMNK mma_tile_mnk,
    TiledMma tiled_mma,
    TensorStorage& shared_tensors,
    bool reverse_epi_n = false) {
    using namespace cute;

    int lane_idx = canonical_lane_idx();
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;

    // The tma tensor C under im2col mode only has two modes (M, N) which
    // should be local tiled with only (m_coord, n_coord).
    auto coord_shape =
      conditional_return<is_im2col_C>(make_coord(m_coord, n_coord), make_coord(m_coord, n_coord, l_coord));

    // Represent the full source tensor, slice to get the tile this CTA is currently responsible for
    Tensor mC_mn = params.tma_load_c.get_tma_tensor(make_shape(M,N,L));                                //       (M,N,L)
    Tensor mC = coalesce(mC_mn, take<0,2>(cta_tile_mnk));
    Tensor gC = local_tile(mC, take<0,2>(cta_tile_mnk), coord_shape);                                  // (CTA_M,CTA_N)

    Tensor mD_mnl = params.tma_store_d.get_tma_tensor(make_shape(M,N,L));   // (M,N,L)
    Tensor mD = coalesce(mD_mnl, take<0,2>(cta_tile_mnk));
    Tensor gD = local_tile(mD, take<0,2>(cta_tile_mnk), coord_shape);   // (CTA_M,CTA_N)

    // Apply epilogue subtile, get matching smem tensor
    Tensor gC_epi = flat_divide(gC, EpilogueTile{});                             // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor gD_epi = flat_divide(  gD, take<0,2>(CtaTileShape{}));                // (CTA_TILE_M,CTA_TILE_N,CTA_M,CTA_N)

    // Construct the corresponding pipelined smem tensors
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    auto ptr_sD = shared_tensors.collective.smem_D.begin();
    auto sC_epi = make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{});           // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    auto sD_epi = make_tensor(make_smem_ptr(ptr_sD), SmemLayoutD{});           // (CTA_M,CTA_N,PIPE_D)

    // Prepare the thread(b)lock's (G)mem to (S)mem TMA tiled copy (bGS_)
    ThrCopy thrblk_g2s = params.tma_load_c.get_slice(Int<0>{});
    Tensor bGS_gC = thrblk_g2s.partition_S(gC_epi);                                    // (TMA,TMA_M,TMA_N,EPI_M,EPI_N)
    Tensor bGS_sC = thrblk_g2s.partition_D(sC_epi);                                    // (TMA,TMA_M,TMA_N,PIPE_C)

    // bGS_gC (int, int, int) o (((_512, _32), _1), _1, _1, _8, _1):(((E<0>, E<1>), _0), _0, _0, _32*E<1>, _0)
    // bGS_sC smem_ptr<half*> o (((_512, _32), _1), _1, _1, _1):(((_1, _512), _0), _0, _0, _16384)

    // thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
    ThrCopy thrblk_s2g = params.tma_store_d.get_slice(Int<0>{});
    auto bSG_sD = thrblk_s2g.partition_S(sD_epi);   // (S2G,S2G_M,S2G_N,PIPE_D)
    auto bSG_gD = thrblk_s2g.partition_D(gD_epi);   // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)

    // bSG_sD smem_ptr<half*> o (((_512, _256), _1), _1, _1, _1):(((_1, _512), _0), _0, _0, _131072)
    // bSG_gD (int, int, int) o (((_512, _256), _1), _1, _1, _1, _1):(((E<0>, E<1>), _0), _0, _0, _0, _0)

    // Get the fusion callbacks for the producer load warp
    auto pld_args = cutlass::epilogue::fusion::detail::ProducerLoadArgs{
      problem_shape_mnkl,
      cta_tile_mnk,
      cta_coord_mnkl,
      tiled_mma,
      EpilogueTile{},
      lane_idx
    };
    auto pld_callbacks = fusion_callbacks.get_producer_load_callbacks(pld_args);
    bool is_C_load_needed = is_source_supported && fusion_callbacks.is_C_load_needed();

    // Predication for TMA load (one thread issues TMA load)
    bool lane_predicate = cute::elect_one_sync();

    // Pre-loop fusion callback entry point
    pld_callbacks.begin();

    auto [load_pipeline, store_pipeline] = pipelines;
    auto [load_pipe_producer_state, store_pipe_consumer_state] = pipeline_states;

    CUTLASS_PRAGMA_UNROLL
    for (int iter_m = 0; iter_m < size<2>(gC_epi); ++iter_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter_n = 0; iter_n < size<3>(gC_epi); ++iter_n) {
        int epi_m = iter_m, epi_n = iter_n;

        // Acquire the lock for this stage
        load_pipeline.producer_acquire(load_pipe_producer_state);
        auto tma_barrier = load_pipeline.producer_get_barrier(load_pipe_producer_state);

        // Execute the TMA load for C if needed
        if (lane_predicate && is_C_load_needed) {
          constexpr uint16_t mcast_mask = 0;
          copy(params.tma_load_c.with(tma_barrier, mcast_mask),
              bGS_gC(_,_,_,epi_m,epi_n), bGS_sC(_,_,_,load_pipe_producer_state.index()));
          load_pipeline.producer_expect_transaction(load_pipe_producer_state);
        }

        // Loop fusion callback entry point
        // pld_callbacks.step(tma_barrier, epi_m, epi_n, load_pipe_producer_state.count(), lane_predicate);

        // Commit TMA loads for this stage and release the lock
        load_pipeline.producer_commit(load_pipe_producer_state);
        ++load_pipe_producer_state;
      }
    }

    // Post-loop fusion callback entry point
    pld_callbacks.end();

    store_pipeline.consumer_try_wait(store_pipe_consumer_state);  // ensure all threads have issued their async fence
    if (lane_predicate) {
      auto abar_store = store_pipeline.consumer_get_barrier(store_pipe_consumer_state);
      copy(params.tma_store_d.with(abar_store), bSG_sD(_,_,_,store_pipe_consumer_state.index()), bSG_gD(_,_,_,_0{},_0{}));
      store_pipeline.consumer_commit(store_pipe_consumer_state, TransactionBytesStore);
    }

    ++store_pipe_consumer_state;

    return make_tuple(load_pipe_producer_state, store_pipe_consumer_state);
  }

  template<
    class Pipelines,
    class PipelineStates,
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class MmaTileMNK,
    class TiledMma,
    class AccEngine,
    class AccLayout
  >
  CUTLASS_DEVICE auto
  store(
      Pipelines pipelines,
      PipelineStates pipeline_states,
      ProblemShapeMNKL problem_shape_mnkl,
      CtaTileMNK cta_tile_mnk,
      CtaCoordMNKL cta_coord_mnkl,
      MmaTileMNK mma_tile_mnk,
      TiledMma tiled_mma,
      cute::Tensor<AccEngine,AccLayout> accumulators,
      TensorStorage& shared_tensors)
  {
    using namespace cute;
    using ElementAccumulator = typename AccEngine::value_type;
    using ElementCompute_ = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::ElementCompute;
    using ElementCompute = cute::conditional_t<cute::is_void_v<ElementCompute_>,ElementAccumulator,ElementCompute_>;

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;
    uint32_t local_id = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_linear_id();
    uint32_t worker_id = local_id - NumControlWarps * NumThreadsPerWarp;

    auto [load_pipeline, store_pipeline, accumulator_pipeline] = pipelines;
    auto [load_pipe_consumer_state, store_pipe_producer_state, accumulator_pipe_consumer_state] = pipeline_states;

    // The tma tensor D under im2col mode only has two modes (M, N) which
    // should be local tiled with only (m_coord, n_coord).
    auto coord_shape =
      conditional_return<is_im2col_D>(make_coord(m_coord, n_coord), make_coord(m_coord, n_coord, l_coord));

    auto as_swizzle_tensor = [](auto tensor) {
      using ValType = typename decltype(tensor)::value_type;
      auto swizzled_layout = composition(CoreMatrix::make_swizzle<ValType>(), tensor.layout());
      return make_tensor(make_smem_ptr(tensor.data()), swizzled_layout);
    };

    // Represent the full output tensor, slice to get the tile this CTA is responsible for
    auto mD_mnl = params.tma_store_d.get_tma_tensor(make_shape(M,N,L));   // (M,N,L)
    auto mD = coalesce(mD_mnl, take<0,2>(cta_tile_mnk));
    auto gD = local_tile(mD, take<0,2>(cta_tile_mnk), coord_shape);   // (CTA_M,CTA_N)
    auto sAcc = as_swizzle_tensor(accumulators(_,_,_0{}));            // (CTA_M,CTA_N)

    // Apply epilogue subtiling
    auto sAcc_epi = flat_divide(sAcc, EpilogueTile{});                // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    auto gD_epi   = flat_divide(  gD, take<0,2>(CtaTileShape{}));     // (CTA_TILE_M,CTA_TILE_N,CTA_M,CTA_N)

    // Construct the corresponding pipelined smem tensors
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    auto ptr_sD = shared_tensors.collective.smem_D.begin();
    auto sC_epi = as_swizzle_tensor(make_tensor(ptr_sC, SmemLayoutC{}));   // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    auto sD_epi = as_swizzle_tensor(make_tensor(ptr_sD, SmemLayoutD{}));   // (CTA_M,CTA_N,PIPE_D)

    // (t)hread-partition for (s)mem to (r)egister copy (tSR_)
    TiledCopy tiled_s2r = make_core_matrix_copy(CopyOpS2R{}, sAcc_epi(_,_,_0{},_0{}));
    ThrCopy thread_s2r = tiled_s2r.get_slice(worker_id);
    auto tSR_sC   = thread_s2r.partition_S(sC_epi);           // (S2R, S2R_M, S2R_N, EPI_M, EPI_N)
    auto tSR_sAcc = thread_s2r.partition_S(sAcc_epi);         // (S2R, S2R_M, S2R_N, EPI_M, EPI_N)
    auto tSR_sD = thread_s2r.partition_D(sD_epi(_,_,_0{}));   // (S2R, S2R_M, S2R_N)

    // sAcc_epi smem_ptr<half*> o (_32, _512, _8, _1):(_512, _1, _16384, _0)
    // sD_epi   smem_ptr<half*> o ((_256, _1), (_512, _1), _1):((_512, _0), (_1, _0), _131072)
    // tSR_sAcc smem_ptr<half*> o ((_16, _2), _1, _1, _8, _1):((_1, _512), _0, _0, _16384, _0)
    // tSR_sD   smem_ptr<half*> o ((_16, _2), _8, _1):((_1, _512), _16384, _0)

    // Allocate D and accumulator registers
    // Does directly store the visitor into smem.
    using RegisterElementD = SmemElementD;
    auto tSR_rAcc = make_tensor<ElementAccumulator>(shape(tSR_sAcc(_,_,_,_0{},_0{})));   // (S2R, S2R_M, S2R_N)
    auto tSR_rD = make_tensor<RegisterElementD>(shape(tSR_rAcc));       // (S2R, S2R_M, S2R_N)
    auto tSR_rC = make_tensor<RegisterElementD>(shape(tSR_rAcc));       // (S2R, S2R_M, S2R_N)

    // Vectorized fragment view
    constexpr int FragmentSize = DispatchPolicy::FragmentSize;
    auto tSR_rAcc_frg = recast<Array<ElementAccumulator, FragmentSize>>(coalesce(tSR_rAcc));  // (EPI_V)
    auto tSR_rD_frg = recast<Array<RegisterElementD, FragmentSize>>(coalesce(tSR_rD));        // (EPI_V)
    CUTE_STATIC_ASSERT(size(tSR_rAcc) % DispatchPolicy::FragmentSize == 0, "Fragment size does not vectorize properly");

    // tSR_rAcc_frg Array<half, 16>* o (_2):(_1)
    // tSR_rD_frg   Array<half, 16>* o (_2):(_1)

    // (t)hread-partition for (r)egister to (s)mem copy (tRS_)
    TiledCopy tiled_r2s = make_tiled_copy_D(Copy_Atom<CopyOpR2S, SmemElementD>{}, tiled_s2r);
    ThrCopy thread_r2s = tiled_r2s.get_slice(worker_id);
    auto tRS_sD = thread_r2s.partition_D(sD_epi);   // (R2S, R2S_M, R2S_N, EPI_M, EPI_N)
    auto tRS_rD = make_tensor<SmemElementD>(shape(tRS_sD(_,_,_,_0{})));

    // tRS_sD smem_ptr <half*> o ((_16, _2), _8, _1, _1):((_1, _512), _16384, _0, _131072)
    // tRS_rD   Array<half, 256> o ((_16, _2), _8, _1):((_1, _16), _32, _0)

    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));                                                     // (M,N)
    Tensor cD_mn = local_tile(mD_crd, take<0,2>(cta_tile_mnk), make_coord(m_coord, n_coord));        // (CTA_M,CTA_N)
    Tensor tTR_cD_mn = thread_s2r.partition_D(flat_divide(cD_mn, EpilogueTile{}));     // (T2R,T2R_M,T2R_N,EPI_M,EPI_N)
    // Relative coordinate tensors (static)
    Tensor cD = make_counting_tensor(cD_mn.layout());                                                  // (CTA_M,CTA_N)
    Tensor tTR_cD = make_counting_tensor(tTR_cD_mn.layout());                          // (T2R,T2R_M,T2R_N,EPI_M,EPI_N)
    // Subtract the global "bottom right" corner from the local "top left" corner to get the max relative coordinate
    auto residue_cD = make_coord(M,N) - cD_mn(_0{});                                                           // (m,n)
    auto residue_tTR_cD = make_coord(M,N) - tTR_cD_mn(_0{});                                                   // (m,n)

    // Get the fusion callbacks for the consumer store warps
    auto empty_tuple = cute::tuple<>{};
    auto dummy_tensor = make_tensor<float>(Int<1>{});
    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs(
      problem_shape_mnkl,
      cta_tile_mnk,
      cta_coord_mnkl,
      tiled_mma,
      EpilogueTile{},
      tiled_s2r,
      cD,
      residue_cD,
      tTR_cD,
      residue_tTR_cD,
      tSR_rC,
      worker_id
    );

    bool is_C_load_needed = is_source_supported && fusion_callbacks.is_C_load_needed();
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<true>(cst_args);

    // Begin the wait for the accumulator results
    accumulator_pipeline.consumer_try_wait(accumulator_pipe_consumer_state);

    //
    // BEGIN EPILOGUE
    //
    cst_callbacks.begin();

    // For each epilogue subtile within the CTA tile
    CUTLASS_PRAGMA_UNROLL
    for (int iter_m = 0; iter_m < size<2>(sAcc_epi); ++iter_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter_n = 0; iter_n < size<3>(sAcc_epi); ++iter_n) {
        int epi_m = iter_m, epi_n = iter_n;
        bool is_first_iteration = iter_m == 0 && iter_n == 0;
        bool is_last_iteration = iter_m == size<2>(gD_epi)-1 && iter_n == size<3>(gD_epi)-1;

        cst_callbacks.begin_loop(epi_m, epi_n);

        // Wait for the producer load to fill smem
        load_pipeline.consumer_wait(load_pipe_consumer_state);

        if (is_C_load_needed) {
          // Copy source tile from smem to register
          copy(tiled_s2r, tSR_sC(_,_,_,load_pipe_consumer_state.index()), tSR_rC);
        }

        load_pipeline.consumer_release(load_pipe_consumer_state);
        ++load_pipe_consumer_state;

        // The current tile in smem
        Tensor tSR_sAcc_mn = tSR_sAcc(_,_,_,epi_m,epi_n);

        // Copy accumulator tile from smem to register
        copy(tiled_s2r, tSR_sAcc_mn, tSR_rAcc);

        // Vectorized fragment loop with visitor callback entry point
        CUTLASS_PRAGMA_UNROLL
        for (int epi_v = 0; epi_v < size(tSR_rD_frg); ++epi_v) {
          tSR_rD_frg(epi_v) = cst_callbacks.visit(tSR_rAcc_frg(epi_v), epi_v, epi_m, epi_n);
        }

        auto tRS_rD = tSR_rD;

        // copy output tile from register to smem
        copy(tiled_r2s, tRS_rD(_,_0{},_0{}), tRS_sD(_,epi_m,epi_n,store_pipe_producer_state.index()));
      }
    }

    store_pipeline.producer_commit(store_pipe_producer_state, 1);
    accumulator_pipeline.consumer_release(accumulator_pipe_consumer_state);

    cst_callbacks.end();

    ++store_pipe_producer_state;
    ++accumulator_pipe_consumer_state;

    return make_tuple(load_pipe_consumer_state, store_pipe_producer_state, accumulator_pipe_consumer_state);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////