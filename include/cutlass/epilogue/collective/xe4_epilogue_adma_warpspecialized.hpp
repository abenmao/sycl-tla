#pragma once

#include "cute/arch/copy_xe4_dma_legacy.hpp"
#include "cute/atom/copy_traits_xe4_dma_legacy.hpp"
#include "cute/atom/copy_traits_xe4_ldsm.hpp"
#include "cute/arch/copy_xe4_adma.hpp"
#include "cute/atom/copy_traits_xe4_adma.hpp"
#include "cute/container/array.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/xe4_detail.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/epilogue/fusion/xe4_callbacks_tma_warpspecialized.hpp"

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
// 15-arg forwarding specialization (no CopyOpS2GReduce): delegates to the 16-arg
// specialization with void, preserving backward compatibility with all existing builders.
template <
  int StagesC_, int StagesD_, int FragmentSize_,
  bool ReuseSmemC_, bool DelayTmaStore_,
  int NumControlWarps_, int NumEpilogueWarps_, bool UseMmaAwareLdsm_,
  class CtaTileShape_, class EpilogueTile_,
  class ElementC_, class StrideC_, class ElementD_, class StrideD_,
  class FusionCallbacks_,
  class CopyOpG2S_, class SmemLayoutAtomC_,
  class CopyOpS2R_, class CopyOpS2RImm_,
  class CopyOpS2G_, class SmemLayoutAtomD_,
  class CopyOpR2S_, class CopyOpR2R_
>
class CollectiveEpilogue<
  Xe4AdmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_, NumControlWarps_, NumEpilogueWarps_, UseMmaAwareLdsm_>,
  CtaTileShape_, EpilogueTile_,
  ElementC_, StrideC_, ElementD_, StrideD_,
  FusionCallbacks_,
  CopyOpG2S_, SmemLayoutAtomC_,
  CopyOpS2R_, CopyOpS2RImm_,
  CopyOpS2G_, SmemLayoutAtomD_,
  CopyOpR2S_, CopyOpR2R_
> : public CollectiveEpilogue<
  Xe4AdmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_, NumControlWarps_, NumEpilogueWarps_, UseMmaAwareLdsm_>,
  CtaTileShape_, EpilogueTile_,
  ElementC_, StrideC_, ElementD_, StrideD_,
  FusionCallbacks_,
  CopyOpG2S_, SmemLayoutAtomC_,
  CopyOpS2R_, CopyOpS2RImm_,
  CopyOpS2G_, SmemLayoutAtomD_,
  CopyOpR2S_, CopyOpR2R_,
  void  // CopyOpS2GReduce = void: SK reduction disabled
> {
  using Base = CollectiveEpilogue<
    Xe4AdmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_, NumControlWarps_, NumEpilogueWarps_, UseMmaAwareLdsm_>,
    CtaTileShape_, EpilogueTile_,
    ElementC_, StrideC_, ElementD_, StrideD_,
    FusionCallbacks_,
    CopyOpG2S_, SmemLayoutAtomC_,
    CopyOpS2R_, CopyOpS2RImm_,
    CopyOpS2G_, SmemLayoutAtomD_,
    CopyOpR2S_, CopyOpR2R_,
    void>;
  using Base::Base;  // inherit all constructors
};

// 16-arg specialization: full SK reduction support via CopyOpS2GReduce_.
// Pass void for CopyOpS2GReduce_ to disable (HasSkReduce = false, zero overhead).
template <
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmaStore_,
  int NumControlWarps_,
  int NumEpilogueWarps_,
  bool UseMmaAwareLdsm_,
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
  class CopyOpR2R_,
  class CopyOpS2GReduce_  // XE4_ADMA_STORE_REDUCE for SK; pass void to disable
>
class CollectiveEpilogue<
  Xe4AdmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_, NumControlWarps_, NumEpilogueWarps_, UseMmaAwareLdsm_>,
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
  CopyOpS2RImm_,
  CopyOpS2G_,
  SmemLayoutAtomD_,
  CopyOpR2S_,
  CopyOpR2R_,
  CopyOpS2GReduce_
> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = Xe4AdmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_, NumControlWarps_, NumEpilogueWarps_, UseMmaAwareLdsm_>;
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
  using CopyOpS2RImm = CopyOpS2RImm_;
  using CopyOpS2G = CopyOpS2G_;
  using SmemLayoutAtomD = SmemLayoutAtomD_;
  using CopyOpR2S = CopyOpR2S_;
  using CopyOpR2R = CopyOpR2R_;
  using CopyOpS2GReduce = CopyOpS2GReduce_;
  static constexpr bool HasSkReduce = !cute::is_void_v<CopyOpS2GReduce>;

  using ThreadEpilogueOp = typename fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;
  using GmemTiledCopyD = CopyOpS2G;

  static constexpr int NumControlWarps = NumControlWarps_;
  static constexpr int NumEpilogueWarps = NumEpilogueWarps_;
  constexpr static int ThreadCount = NumEpilogueWarps * NumThreadsPerWarp;

  static_assert(!is_layout<EpilogueTile>::value && is_tuple<EpilogueTile>::value, "EpilogueTile must be a cute::Tile or cute::Shape");
  static_assert(rank(EpilogueTile{}) == 2, "EpilogueTile must be rank-2: [EPI_TILE_M, EPI_TILE_N]");

private:
  using GmemElementD = ElementD;
  using GmemElementC = cute::conditional_t<cute::is_void_v<ElementC>,ElementD,ElementC>; // prevents void ref breakages
  using SmemElementD = typename cutlass::detail::get_unpacked_element_type<GmemElementD>::type;
  using SmemElementC = typename cutlass::detail::get_unpacked_element_type<GmemElementC>::type;

  constexpr static bool is_fp_postop = is_floating_t<SmemElementD>::value && (sizeof_bits_v<SmemElementD> < 16);
  constexpr static bool is_int8_postop = is_integral<SmemElementD>::value && (sizeof_bits_v<SmemElementD> == 8);
  using SmemElementImm = cute::conditional_t<is_fp_postop, bf16, cute::conditional_t<is_int8_postop, int32_t, SmemElementD>>;

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

  using SmemLayoutC = cute::conditional_t<is_im2col_C, decltype(cute::append<3>(SmemLayoutAtomC{}, Layout<Int<StagesC>, Int<StrideStageC>>{})),
                                                       decltype(cute::append<3>(SmemLayoutStageC{}, Layout<Int<StagesC>, Int<StrideStageC>>{}))>;
  using SmemLayoutD = cute::conditional_t<is_im2col_D, decltype(cute::append<3>(SmemLayoutAtomD{}, Layout<Int<ReuseSmemC ? StagesC : StagesD>, Int<StrideStageD>>{})),
                                                       decltype(cute::append<3>(SmemLayoutStageD{}, Layout<Int<ReuseSmemC ? StagesC : StagesD>, Int<StrideStageD>>{}))>;

  constexpr static size_t SmemAlignmentC = 512;
  constexpr static size_t SmemAlignmentD = 512;
  constexpr static size_t MaxSmemAlignment = cute::max(SmemAlignmentC, SmemAlignmentD);

  struct CollectiveStorageWithC {
    alignas(SmemAlignmentC) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    union {
      alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
      alignas(SmemAlignmentD) ArrayEngine<SmemElementImm, cosize_v<SmemLayoutD>> smem_Imm;
    };
  };

  union CollectiveStorageWithoutC {
    cute::array<SmemElementC, 0> smem_C;
    union {
      alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
      alignas(SmemAlignmentD) ArrayEngine<SmemElementImm, cosize_v<SmemLayoutD>> smem_Imm;
    };
  };

  union CollectiveStorageReuseC {
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementImm, cosize_v<SmemLayoutD>> smem_Imm;
  };

public:
  // TMA pipeline for loading C
  using LoadPipeline = cutlass::PipelineTransactionAsync<StagesC>;;
  using LoadPipelineState = typename LoadPipeline::PipelineState;
  constexpr static uint32_t TmaTransactionBytes = cutlass::bits_to_bytes(StageCBits);

  // Dedicated 1-stage sync pipeline for final-SK G2S load-back (gmem D -> smem_Imm).
  using G2SPipeline = cutlass::PipelineTransactionAsync<1>;
  using G2SPipelineState = typename G2SPipeline::PipelineState;

  // Dedicated 1-stage pipeline for the SK fred (smem_Imm -> gmem D atomic S2G reduce).
  using FredPipeline = cutlass::PipelineTmaAsync<1>;
  using FredPipelineState = typename FredPipeline::PipelineState;

  // TMA pipeline for storing D
  using StorePipeline = cutlass::PipelineTmaAsync<StagesD>;
  using StorePipelineState = typename StorePipeline::PipelineState;

  using WaveOrderBarrier = cutlass::OrderedSequenceBarrier<1,1>;

  static_assert(ReuseSmemC==false, "Not support ReuseSmemC yet");

  struct SharedStorage
  {
    struct TensorStorage
    {
      using CollectiveStorage = cute::conditional_t<not is_source_supported, CollectiveStorageWithoutC,
                                  cute::conditional_t<ReuseSmemC, CollectiveStorageReuseC, CollectiveStorageWithC>>;
      CollectiveStorage collective;
      typename FusionCallbacks::SharedStorage thread;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;

  static constexpr uint32_t TransactionBytesStore = sizeof(ElementD) * size(SmemLayoutD {});
  // Bytes transferred when fred'ing smem_Imm -> gmem D workspace (SmemElementImm precision).
  static constexpr uint32_t TransactionBytesImm = sizeof(SmemElementImm) * size(SmemLayoutD {});

  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
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
  get_adma_load_c(ProblemShapeMNL const& problem_shape_mnl, Arguments const& args) {
    if constexpr (is_im2col_C) {
      Tensor tensor_c = make_tensor(make_gmem_ptr<GmemElementC>(args.ptr_C),
                                    make_layout(problem_shape_mnl, args.dC));
      return make_adma_copy(
          CopyOpG2S{}, tensor_c, SmemLayoutAtomC{}, TmaEpilogueTile{}, _1{});
    }
    else {
      Tensor tensor_c = make_tensor(make_gmem_ptr<GmemElementC>(args.ptr_C),
                                    make_layout(problem_shape_mnl, append<3>(args.dC, _0{})));
      return make_adma_copy(
          CopyOpG2S{}, tensor_c, SmemLayoutStageC{}, TmaEpilogueTile{}, _1{});
    }
  }

  template <class ProblemShapeMNL>
  static constexpr auto
  get_adma_store_d(ProblemShapeMNL const& problem_shape_mnl, Arguments const& args) {
    if constexpr(is_im2col_D) {
      Tensor tensor_d = make_tensor(make_gmem_ptr<GmemElementD>(args.ptr_D),
                                    make_layout(problem_shape_mnl, args.dD));
      return make_adma_copy(CopyOpS2G{}, tensor_d, SmemLayoutAtomD{}, take<0,2>(CtaTileShape{}), _1{});
    } else {
      Tensor tensor_d = make_tensor(make_gmem_ptr<GmemElementD>(args.ptr_D),
                                    make_layout(problem_shape_mnl, append<3>(args.dD, _0{})));
      return make_adma_copy(CopyOpS2G{}, tensor_d, SmemLayoutD{}, take<0,2>(CtaTileShape{}), _1{});

    }
  }

  // G2S load: gmem D workspace -> smem_Imm. Used by final SK split to load the fully
  // accumulated partial sum back into smem_Imm so the normal LDSM epilogue path runs.
  // Uses CopyOpG2S (same as adma_load_c) with SmemLayoutD and SmemElementImm element type.
  template <class ProblemShapeMNL>
  static constexpr auto
  get_adma_load_d(ProblemShapeMNL const& problem_shape_mnl, Arguments const& args) {
    Tensor tensor_d = make_tensor(make_gmem_ptr<SmemElementImm>(reinterpret_cast<SmemElementImm const*>(args.ptr_D)),
                                  make_layout(problem_shape_mnl, append<3>(args.dD, _0{})));
    return make_adma_copy(CopyOpG2S{}, tensor_d, SmemLayoutD{}, take<0,2>(CtaTileShape{}), _1{});
  }

  // S2G atomic reduce: smem_Imm -> gmem D workspace. Uses same layout as adma_store_d
  // but with CopyOpS2GReduce (XE4_ADMA_STORE_REDUCE) as the copy atom.
  template <class ProblemShapeMNL>
  static constexpr auto
  get_adma_store_reduce(ProblemShapeMNL const& problem_shape_mnl, Arguments const& args) {
    Tensor tensor_d = make_tensor(make_gmem_ptr<GmemElementD>(args.ptr_D),
                                  make_layout(problem_shape_mnl, append<3>(args.dD, _0{})));
    return make_adma_copy(CopyOpS2GReduce{}, tensor_d, SmemLayoutD{}, take<0,2>(CtaTileShape{}), _1{});
  }


public:
  // Device side epilogue params
  struct Params {
    using ADMA_C = cute::conditional_t<is_im2col_C, decltype(get_adma_load_c (repeat_like(take<0,2>(StrideC{}), int32_t(0)), Arguments{})),
                                                   decltype(get_adma_load_c (repeat_like(append<3>(StrideC{},_1{}), int32_t(0)), Arguments{}))>;
    using ADMA_D = cute::conditional_t<is_im2col_D, decltype(get_adma_store_d(repeat_like(take<0,2>(StrideD{}), int32_t(0)), Arguments{})),
                                                   decltype(get_adma_store_d(repeat_like(append<3>(StrideD{},_1{}), int32_t(0)), Arguments{}))>;

    // SK reduce ADMA types: lazily computed to avoid instantiating helpers when HasSkReduce=false.
    template <bool Enable, class = void>
    struct SkReduceTypes {
      using ADMA_REDUCE  = cute::tuple<>;
      using ADMA_LOAD_D  = cute::tuple<>;
    };
    template <class Dummy>
    struct SkReduceTypes<true, Dummy> {
      using ADMA_REDUCE = decltype(get_adma_store_reduce(
          repeat_like(append<3>(StrideD{},_1{}), int32_t(0)), Arguments{}));
      // G2S load: gmem D (as SmemElementImm) -> smem_Imm for final SK split epilogue.
      using ADMA_LOAD_D = decltype(get_adma_load_d(
          repeat_like(append<3>(StrideD{},_1{}), int32_t(0)), Arguments{}));
    };
    using ADMA_REDUCE = typename SkReduceTypes<HasSkReduce>::ADMA_REDUCE;
    using ADMA_LOAD_D = typename SkReduceTypes<HasSkReduce>::ADMA_LOAD_D;

    typename FusionCallbacks::Params thread{};
    ADMA_C adma_load_c;
    ADMA_D adma_store_d;
    ADMA_REDUCE adma_store_reduce{};  // fred: smem_Imm -> gmem D workspace
    ADMA_LOAD_D adma_load_d{};        // G2S: gmem D -> smem_Imm (final SK split)
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
    if constexpr (is_im2col_C || is_im2col_D) {
      auto problem_shape_mnl = problem_shape;
      typename Params::ADMA_C adma_load_c{};
      if constexpr (is_source_supported) {
        adma_load_c = get_adma_load_c(problem_shape_mnl, args);
      }

      typename Params::ADMA_D adma_store_d = get_adma_store_d(problem_shape_mnl, args);

      return {
        FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, nullptr),
        adma_load_c,
        adma_store_d
      };
    }
    else {
      auto problem_shape_mnl = select<0,1,3>(append<4>(problem_shape, 1));
      typename Params::ADMA_C adma_load_c{};
      if constexpr (is_source_supported) {
        adma_load_c = get_adma_load_c(problem_shape_mnl, args);
      }

      typename Params::ADMA_D adma_store_d = get_adma_store_d(problem_shape_mnl, args);

      if constexpr (HasSkReduce) {
        return {
          FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, nullptr),
          adma_load_c,
          adma_store_d,
          get_adma_store_reduce(problem_shape_mnl, args),
          get_adma_load_d(problem_shape_mnl, args)
        };
      } else {
        return {
          FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, nullptr),
          adma_load_c,
          adma_store_d
        };
      }
    }
  }

  CUTLASS_DEVICE
  static constexpr auto
  get_intermedia_tensor(TensorStorage& shared_tensors) {
    auto ptr_sImm = shared_tensors.collective.smem_Imm.begin();
    auto tensor_imm = make_tensor(make_smem_ptr(ptr_sImm), SmemLayoutD{});
    return tensor_imm;
  }

  //
  // Constructor and Data Members
  //
  template <class Params, class TensorDescTuple>
  CUTLASS_DEVICE
  CollectiveEpilogue(
    Params const& params_,
    TensorStorage& shared_tensors,
    WaveOrderBarrier& wave_order_barrier_,
    TensorDescTuple tdesc_tuple)
      : params(params_)
      , fusion_callbacks(params_.thread, shared_tensors.thread)
      , wave_order_barrier(wave_order_barrier_) {
    if constexpr (!is_im2col_C && !is_im2col_D) {
       auto [tensor_desc_c, tensor_desc_d] = tdesc_tuple;
       params.adma_load_c.set_tensor_desc(tensor_desc_c);
       params.adma_store_d.set_tensor_desc(tensor_desc_d);
       // SK reduce and G2S load both point to ptr_D workspace, share tdesc_d.
       if constexpr (HasSkReduce) {
         params.adma_store_reduce.set_tensor_desc(tensor_desc_d);
         params.adma_load_d.set_tensor_desc(tensor_desc_d);
       }
    }
  }

private:
  Params const& params;
  FusionCallbacks fusion_callbacks;
  WaveOrderBarrier& wave_order_barrier;

  //
  // Non-static Device Functions
  //
public:
  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return fusion_callbacks.is_producer_load_needed();
  }

  // EpiLoad-side dispatch. (IsSkNonFinal, IsSkFinal) select the stream-K role; the pair
  // is mutually exclusive (at most one true) and only meaningful when HasSkReduce is true.
  //   (false, false) : plain DP (or non-SK) tile. Run the full C-load subtile loop and the
  //                     smem_D -> gmem_D store. No SK reduction.
  //   (true,  false) : non-final SK split. fred smem_Imm -> gmem D workspace and bump the
  //                     tile counter; SKIP the C-load loop and the D-store (this unit produces
  //                     no epilogue output). Pairs with store()'s (false, true) case.
  //   (false, true ) : final SK split. fred this split's contribution, then poll the counter
  //                     until all peers have arrived, G2S load-back the fully-accumulated
  //                     gmem D into smem_Imm, then run the normal C-load + store path so the
  //                     reduced sum flows through the standard LDSM epilogue.
  //   (true,  true ) : invalid -- never instantiated.
  template<
    bool ReuseTmem = false,
    bool IsSkNonFinal = false,  // non-final SK split: fred smem_Imm -> gmem D, increment counter
    bool IsSkFinal = false,     // final SK split: poll counter, G2S load-back gmem D -> smem_Imm
    class Pipelines,
    class PipelineStates,
    class ProblemShapeMNKL,
    class CtaTileMNK,
    class CtaCoordMNKL,
    class MmaTileMNK,
    class TiledMma,
    // WorkTileInfo is scheduler-specific; DP schedulers omit k_tile_count/K_idx.
    // Passing the whole struct lets the epilogue extract fields only when HasSkReduce
    // is true, keeping the kernel free of scheduler-type-dependent field accesses.
    class WorkTileInfo_ = cute::tuple<>
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
    bool reverse_epi_n = false,
    int* sk_tile_counter = nullptr,
    uint64_t sk_tile_idx = 0,
    WorkTileInfo_ work_tile_info = {}) {
    using namespace cute;

    int lane_idx = canonical_lane_idx();
    uint32_t local_id = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_linear_id();
    using ThreadIdx = cute::conditional_t<is_im2col_D, uint32_t, Int<0>>;
    ThreadIdx thread_idx; 
    if constexpr (is_im2col_D) {
      thread_idx = local_id - 3 * NumThreadsPerWarp;
    } else {
      thread_idx = Int<0>{};
    }
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;

    // The tma tensor C under im2col mode only has two modes (M, N) which
    // should be local tiled with only (m_coord, n_coord).
    auto coord_shape =
      conditional_return<is_im2col_C>(make_coord(m_coord, n_coord), make_coord(m_coord, n_coord, l_coord));

    // Represent the full source tensor, slice to get the tile this CTA is currently responsible for
    using TMAShapeC = cute::conditional_t<is_im2col_C, decltype(make_shape(M,N)), decltype(make_shape(M,N,L))>;
    using TMAShapeD = cute::conditional_t<is_im2col_D, decltype(make_shape(M,N)), decltype(make_shape(M,N,L))>;
    TMAShapeC shapeC;
    if constexpr (is_im2col_C) {
      shapeC = make_shape(M,N);
    }
    else{
      shapeC = make_shape(M,N,L);
    }
    Tensor mC_mn = params.adma_load_c.get_tma_tensor(shapeC);                                //       (M,N,L)
    Tensor mC = coalesce(mC_mn, take<0,2>(cta_tile_mnk));
    Tensor gC = local_tile(mC, take<0,2>(cta_tile_mnk), coord_shape);                                  // (CTA_M,CTA_N)

    TMAShapeD shapeD;
    if constexpr (is_im2col_C) {
      shapeD = make_shape(M,N);
    }
    else{
      shapeD = make_shape(M,N,L);
    }
    Tensor mD_mnl = params.adma_store_d.get_tma_tensor(shapeD);
    Tensor mD = coalesce(mD_mnl, take<0,2>(cta_tile_mnk));
    Tensor gD = local_tile(mD, take<0,2>(cta_tile_mnk), coord_shape);   // (CTA_M,CTA_N)

    // Apply epilogue subtile, get matching smem tensor
    Tensor gC_epi = flat_divide(gC, EpilogueTile{});                             // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor gD_epi = flat_divide(  gD, take<0,2>(CtaTileShape{}));                // (CTA_TILE_M,CTA_TILE_N,CTA_M,CTA_N)

    // Construct the corresponding pipelined smem tensors
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    auto ptr_sD = shared_tensors.collective.smem_D.begin();
    auto sC_epi = make_slm_tensor<SmemElementC>(ptr_sC, SmemLayoutC{});   // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    auto sD_epi = make_slm_tensor<SmemElementD>(ptr_sD, SmemLayoutD{});   // (CTA_M,CTA_N,PIPE_D)
    
    // Prepare the thread(b)lock's (G)mem to (S)mem TMA tiled copy (bGS_)
    ThrCopy thrblk_g2s = params.adma_load_c.get_slice(thread_idx);
    Tensor bGS_gC = thrblk_g2s.partition_S(gC_epi);                                    // (TMA,TMA_M,TMA_N,EPI_M,EPI_N)
    Tensor bGS_sC = thrblk_g2s.partition_D(sC_epi);                                    // (TMA,TMA_M,TMA_N,PIPE_C)

    // thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
    ThrCopy thrblk_s2g = params.adma_store_d.get_slice(thread_idx);
    auto bSG_sD = thrblk_s2g.partition_S(sD_epi);   // (S2G,S2G_M,S2G_N,PIPE_D)
    auto bSG_gD = thrblk_s2g.partition_D(gD_epi);   // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)

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

    auto [load_pipeline, store_pipeline, accumulator_pipeline, g2s_pipeline, fred_pipeline] = pipelines;
    auto [load_pipe_producer_state, store_pipe_consumer_state, accumulator_pipe_consumer_state, g2s_pipe_producer_state, fred_pipe_consumer_state] = pipeline_states;

    if constexpr ((IsSkNonFinal || IsSkFinal) && HasSkReduce) {
      streamk_fred_to_gmem_d(
        fred_pipeline, fred_pipe_consumer_state,
        thread_idx, shared_tensors, gD_epi, lane_predicate);
      // Extract k_tile_count and K_idx from the scheduler's WorkTileInfo.
      // These fields only exist on the SK scheduler's WorkTileInfo; the epilogue
      // owns this extraction so the kernel never touches scheduler-specific members.
      streamk_arrive_counter(
        sk_tile_counter,
        sk_tile_idx,
        work_tile_info.k_tile_count,
        lane_predicate);

      if constexpr (IsSkFinal) {
        streamk_wait_for_previous_splits(
          sk_tile_counter,
          sk_tile_idx,
          work_tile_info.K_idx + work_tile_info.k_tile_count);
        streamk_load_gmem_d_to_smem(
          g2s_pipeline, g2s_pipe_producer_state,
          thread_idx, shared_tensors, gD_epi, lane_predicate);
      }
    }
    
    // Pre-loop fusion callback entry point
    pld_callbacks.begin();
    
    // Run C-load loop for IsSkFinal and plain DP (not non-final SK).
    if constexpr (!IsSkNonFinal) {
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
            load_pipeline.producer_expect_transaction(load_pipe_producer_state);
            copy(params.adma_load_c.with(tma_barrier, mcast_mask),
                bGS_gC(_,_,_,epi_m,epi_n), bGS_sC(_,_,_,load_pipe_producer_state.index()));
          }

          // Loop fusion callback entry point
          pld_callbacks.step(tma_barrier, epi_m, epi_n, load_pipe_producer_state.count(), lane_predicate);

          // Commit TMA loads for this stage and release the lock
          load_pipeline.producer_commit(load_pipe_producer_state);
          ++load_pipe_producer_state;
        }
      }
    }

    // Post-loop fusion callback entry point
    pld_callbacks.end();

    // IsSkFinal and plain DP both need smem_D -> gmem_D after the epilogue.
    // Non-final SK skips (no epilogue output).
    if constexpr (!IsSkNonFinal) {
      store_pipeline.consumer_try_wait(store_pipe_consumer_state);
      if (lane_predicate) {
        if constexpr (is_im2col_D) {
          store_pipeline.consumer_commit(store_pipe_consumer_state, TransactionBytesStore);
        } else {
          auto abar_store = store_pipeline.consumer_get_barrier(store_pipe_consumer_state);
          copy(params.adma_store_d.with(abar_store),
               bSG_sD(_,_,_,store_pipe_consumer_state.index()),
               bSG_gD(_,_,_,_0{},_0{}));
          store_pipeline.consumer_commit(store_pipe_consumer_state, TransactionBytesStore);
        }
      }
      if constexpr (is_im2col_D) {
        auto abar_store = store_pipeline.consumer_get_barrier(store_pipe_consumer_state);
        copy(params.adma_store_d.with(abar_store),
             bSG_sD(_,_,_,store_pipe_consumer_state.index()),
             bSG_gD(_,_,_,_0{},_0{}));
      }
      ++store_pipe_consumer_state;
    }

    return make_tuple(load_pipe_producer_state, store_pipe_consumer_state, accumulator_pipe_consumer_state, g2s_pipe_producer_state, fred_pipe_consumer_state);
  }

  // EpiStore-side dispatch, mirroring load()'s (IsSkNonFinal, IsSkFinal) tags above. The pair
  // is mutually exclusive (at most one true) and only meaningful when HasSkReduce is true.
  //   (final=false, nonfinal=false) : plain DP (or non-SK) tile. Wait on the accumulator,
  //                                    run the full subtile epilogue (LDSM Acc/C -> visit ->
  //                                    R2S smem_D), and store smem_D -> gmem_D.
  //   (final=false, nonfinal=true ) : non-final SK split. EpiLoad already fred'd smem_Imm ->
  //                                    gmem D, so produce NO output: just release the
  //                                    accumulator stage, commit the fred stage. Pairs with load()'s (true, false).
  //   (final=true,  nonfinal=false) : final SK split. The fully-reduced sum was G2S-loaded
  //                                    into smem_Imm by EpiLoad; commit the fred stage, wait
  //                                    on the dedicated G2S sync stage, then run the normal
  //                                    subtile epilogue + store. Pairs with load()'s (false, true).
  //   (final=true,  nonfinal=true ) : invalid -- never instantiated.
  template<
    bool IsSkFinal = false,
    bool IsSkNonFinal = false,  // non-final SK split: no epilogue output, only keep
                                // accumulator/store pipelines + wave_order barrier in phase
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
      TensorStorage& shared_tensors,
      bool enable_g2s_sync = false)
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

    auto [load_pipeline, store_pipeline, accumulator_pipeline, g2s_pipeline, fred_pipeline] = pipelines;
    auto [load_pipe_consumer_state, store_pipe_producer_state, accumulator_pipe_consumer_state, g2s_pipe_consumer_state, fred_pipe_producer_state] = pipeline_states;

    // The tma tensor D under im2col mode only has two modes (M, N) which
    // should be local tiled with only (m_coord, n_coord).
    auto coord_shape =
      conditional_return<is_im2col_D>(make_coord(m_coord, n_coord), make_coord(m_coord, n_coord, l_coord));

    // Represent the full output tensor, slice to get the tile this CTA is responsible for
    using TMAShapeD = cute::conditional_t<is_im2col_D, decltype(make_shape(M,N)), decltype(make_shape(M,N,L))>;
    TMAShapeD shapeD;
    if constexpr (is_im2col_D) {
      shapeD = make_shape(M,N);
    }
    else{
      shapeD = make_shape(M,N,L);
    }
    auto mD_mnl = params.adma_store_d.get_tma_tensor(shapeD);
    auto mD = coalesce(mD_mnl, take<0,2>(cta_tile_mnk));
    auto gD = local_tile(mD, take<0,2>(cta_tile_mnk), coord_shape);   // (CTA_M,CTA_N)
    auto sAcc = as_cm_tensor(accumulators)(_,_,_0{});                 // (CTA_M,CTA_N)

    // Apply epilogue subtiling
    auto sAcc_epi = flat_divide(sAcc, EpilogueTile{});                // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    auto gD_epi   = flat_divide(  gD, take<0,2>(CtaTileShape{}));     // (CTA_TILE_M,CTA_TILE_N,CTA_M,CTA_N)

    // Construct the corresponding pipelined smem tensors
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    auto ptr_sD = shared_tensors.collective.smem_D.begin();
    auto sC_epi = make_slm_tensor<SmemElementC>(ptr_sC, SmemLayoutC{});   // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    auto sD_epi = make_slm_tensor<SmemElementD>(ptr_sD, SmemLayoutD{});   // (CTA_M,CTA_N,PIPE_D)

    // (t)hread-partition for (s)mem to (r)egister copy (tSR_)
    // Thread and value layouts derived from EpilogueTile and core matrix constants.
    // NumElementsPerThread = EpiTileN / NumWarpsAlongN: N-elements per thread (warp),
    // derived from the epilogue tile shape and warp count rather than hardcoded.
    constexpr int NumWarpsAlongM = get<0>(EpilogueTile{}) / NumThreadsPerWarp;
    constexpr int NumWarpsAlongN = NumEpilogueWarps / NumWarpsAlongM;
    constexpr int NumElementsPerThread = get<1>(EpilogueTile{}) / NumWarpsAlongN;
    auto epi_thr_layout = make_ordered_layout(Shape<Shape<Int<NumThreadsPerWarp>,Int<NumWarpsAlongM>>,Int<NumWarpsAlongN>>{}, Step<Step<_0,_2>,_1>{});
    auto epi_val_layout = make_ordered_layout(Shape<_1,Int<NumElementsPerThread>>{}, Step<_1,_0>{});

    auto [tiled_s2r, tiled_s2r_imm]  = [&]() {
      auto tiled_s2r = make_tiled_copy(Copy_Atom<CopyOpS2R, SmemElementD>{}, epi_thr_layout, epi_val_layout);
      auto tiled_s2r_imm = make_tiled_copy(Copy_Atom<CopyOpS2RImm, SmemElementImm>{}, epi_thr_layout, epi_val_layout);
      return make_tuple(tiled_s2r, tiled_s2r_imm);
    }();

    ThrCopy thread_s2r = tiled_s2r.get_slice(worker_id);
    ThrCopy thread_s2r_imm = tiled_s2r_imm.get_slice(worker_id);
    auto tSR_sC   = thread_s2r.partition_S(sC_epi);           // (S2R, S2R_M, S2R_N, EPI_M, EPI_N)
    auto tSR_sAcc = thread_s2r_imm.partition_S(sAcc_epi);         // (S2R, S2R_M, S2R_N, EPI_M, EPI_N)
    auto tSR_sD = thread_s2r.partition_D(sD_epi(_,_,_0{}));   // (S2R, S2R_M, S2R_N)

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

    // (t)hread-partition for (r)egister to (s)mem copy (tRS_)
    TiledCopy tiled_r2s = make_tiled_copy_D(Copy_Atom<CopyOpR2S, SmemElementD>{}, tiled_s2r);
    ThrCopy thread_r2s = tiled_r2s.get_slice(worker_id);
    auto tRS_sD = thread_r2s.partition_D(sD_epi);   // (R2S, R2S_M, R2S_N, EPI_M, EPI_N)
    auto tRS_rD = make_tensor<SmemElementD>(shape(tRS_sD(_,_,_,_0{})));

    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));                                                     // (M,N)
    Tensor cD_mn = local_tile(mD_crd, take<0,2>(cta_tile_mnk), make_coord(m_coord, n_coord));        // (CTA_M,CTA_N)
    Tensor tTR_cD_mn = thread_s2r.partition_D(flat_divide(cD_mn, EpilogueTile{}));     // (T2R,T2R_M,T2R_N,EPI_M,EPI_N)
    // Relative coordinate tensors (static)
    Tensor cD = make_coord_tensor(cD_mn.layout());                                                  // (CTA_M,CTA_N)
    Tensor tTR_cD = make_coord_tensor(tTR_cD_mn.layout());                          // (T2R,T2R_M,T2R_N,EPI_M,EPI_N)
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

    // Wait for AMMA: smem_Imm filled by MMA.
    accumulator_pipeline.consumer_try_wait(accumulator_pipe_consumer_state);

    if constexpr (IsSkNonFinal) {
      accumulator_pipeline.consumer_release(accumulator_pipe_consumer_state);
      ++accumulator_pipe_consumer_state;
      // Signal the fred pipeline (smem_Imm ready for EpiLoad to reduce). The store pipeline is
      // untouched -- a non-final split writes no smem_D, so it does not D-store.
      fred_pipeline.producer_commit(fred_pipe_producer_state, 1);
      ++fred_pipe_producer_state;

      return make_tuple(load_pipe_consumer_state, store_pipe_producer_state,
                        accumulator_pipe_consumer_state, g2s_pipe_consumer_state,
                        fred_pipe_producer_state);
    }

    if constexpr (IsSkFinal && HasSkReduce) {
      // Signal the fred pipeline (smem_Imm ready for EpiLoad to fred its contribution). The
      // D-store for this final split is committed later on the store pipeline at the end of the
      // subtile loop -- the fred and the D-store are now distinct pipelines, so a final split
      // commits the store pipeline exactly once (matching MMA's single store acquire).
      fred_pipeline.producer_commit(fred_pipe_producer_state, 1);
      ++fred_pipe_producer_state;

      // Only SK-final units consume the dedicated G2S stage.
      if (enable_g2s_sync) {
        g2s_pipeline.consumer_wait(g2s_pipe_consumer_state);
      }
    }

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

        wave_order_barrier.wait();

        load_pipeline.consumer_wait(load_pipe_consumer_state);

        if (is_C_load_needed) {
          // Copy source C tile from SLM to registers (S2R).
          if constexpr (cute::is_ldsm_load_matrix_v<CopyOpS2R>) {
            // --- LDSM descriptor path via make_ldsm_copy_C ---
            // Dispatch: MMA-aware API uses TiledMMA's AtomShape to derive warp
            // distribution; geometry-only API derives purely from SLM tensor shape.
            auto ldsm_s2r = [&]() {
              if constexpr (UseMmaAwareLdsm_) {
                return cute::make_ldsm_copy_C<NumEpilogueWarps>(
                    tiled_mma, sC_epi(_,_,load_pipe_consumer_state.index()));
              } else {
                return cute::make_ldsm_copy_C<NumEpilogueWarps>(
                    sC_epi(_,_,load_pipe_consumer_state.index()));
              }
            }();
            auto coord_sC = make_identity_tensor(make_shape(get<0>(EpilogueTile{}), get<1>(EpilogueTile{})));
            auto thr_ldsm = ldsm_s2r.get_slice(worker_id);
            auto ldsm_src = thr_ldsm.partition_S(coord_sC);
            copy(ldsm_s2r, ldsm_src, tSR_rC);
          } else {
            // Legacy XE4_LDSM vector load path (pointer-based, no descriptor)
            copy(tiled_s2r, tSR_sC(_,_,_,load_pipe_consumer_state.index()), tSR_rC);
          }
        }

        load_pipeline.consumer_release(load_pipe_consumer_state);
        ++load_pipe_consumer_state;

        // Load accumulator tile from smem_Imm:
        // - Plain DP: AMMA output is in smem_Imm.
        // - IsSkFinal: G2S ADMA loaded the fully-accumulated gmem D sum into smem_Imm;
        //   the last load_pipeline.consumer_wait above (for the G2S stage) ensures it's ready.
        {
          Tensor tSR_sAcc_mn = tSR_sAcc(_,_,_,epi_m,epi_n);
          if constexpr (cute::is_ldsm_load_matrix_v<CopyOpS2RImm>) {
            auto ldsm_s2r_imm = cute::make_ldsm_tiled_copy(
                CopyOpS2RImm{}, sAcc_epi(_,_,epi_m,epi_n),
                epi_thr_layout, epi_val_layout, true);
            auto coord_sAcc = make_identity_tensor(make_shape(get<0>(EpilogueTile{}), get<1>(EpilogueTile{})));
            auto thr_ldsm_imm = ldsm_s2r_imm.get_slice(worker_id);
            auto ldsm_src_imm = thr_ldsm_imm.partition_S(coord_sAcc);
            copy(ldsm_s2r_imm, ldsm_src_imm, tSR_rAcc);
          } else {
            copy(tiled_s2r_imm, tSR_sAcc_mn, tSR_rAcc);
          }
        }

        // Vectorized fragment loop with visitor callback entry point
        CUTLASS_PRAGMA_UNROLL
        for (int epi_v = 0; epi_v < size(tSR_rD_frg); ++epi_v) {
          tSR_rD_frg(epi_v) = cst_callbacks.visit(tSR_rAcc_frg(epi_v), epi_v, epi_m, epi_n);
        }

        auto tRS_rD = tSR_rD;

        // Copy output D tile from registers to SLM (R2S).
        if constexpr (cute::is_ldsm_store_matrix_v<CopyOpR2S>) {
          // --- LDSM descriptor path via make_ldsm_copy_D ---
          // See make_ldsm_copy_C dispatch above for MMA-aware vs geometry-only rationale.
          auto sD_stage = sD_epi(_,_,store_pipe_producer_state.index());
          auto sD_subtiled = flat_divide(sD_stage, EpilogueTile{});
          auto ldsm_r2s = [&]() {
            if constexpr (UseMmaAwareLdsm_) {
              return cute::make_ldsm_copy_D<NumEpilogueWarps>(
                  tiled_mma, sD_subtiled(_,_,epi_m,epi_n));
            } else {
              return cute::make_ldsm_copy_D<NumEpilogueWarps>(
                  sD_subtiled(_,_,epi_m,epi_n));
            }
          }();
          auto coord_sD = make_identity_tensor(make_shape(get<0>(EpilogueTile{}), get<1>(EpilogueTile{})));
          auto thr_ldsm_r2s = ldsm_r2s.get_slice(worker_id);
          auto ldsm_dst = thr_ldsm_r2s.partition_D(coord_sD);
          // Rank-matched slicing: both tensors collapse trailing size-1 rest modes
          // to produce matching rank-1 tensors for CuTe's copy().
          copy(ldsm_r2s, tRS_rD(_,_0{},_0{}), ldsm_dst(_,_0{},_0{}));
        } else {
          // Legacy XE4_STSM vector store path
          copy(tiled_r2s, tRS_rD(_,_0{},_0{}), tRS_sD(_,epi_m,epi_n,store_pipe_producer_state.index()));
        }

        wave_order_barrier.arrive();
      }
    }

    store_pipeline.producer_commit(store_pipe_producer_state, 1);
    accumulator_pipeline.consumer_release(accumulator_pipe_consumer_state);

    cst_callbacks.end();

    ++store_pipe_producer_state;
    ++accumulator_pipe_consumer_state;

    if(enable_g2s_sync) {
      g2s_pipeline.consumer_release(g2s_pipe_consumer_state);
      ++g2s_pipe_consumer_state;
    }

    return make_tuple(load_pipe_consumer_state, store_pipe_producer_state, accumulator_pipe_consumer_state, g2s_pipe_consumer_state, fred_pipe_producer_state);
  }

private:
  template <class FredPipeline_, class FredPipelineState_, class ThreadIdx_, class TensorD_>
  CUTLASS_DEVICE void
  streamk_fred_to_gmem_d(
      FredPipeline_& fred_pipeline,
      FredPipelineState_& fred_pipe_consumer_state,
      ThreadIdx_ thread_idx,
      TensorStorage& shared_tensors,
      TensorD_ gD_epi,
      bool lane_predicate) const {
    fred_pipeline.consumer_try_wait(fred_pipe_consumer_state);
    if (lane_predicate) {
      auto ptr_sImm_store = shared_tensors.collective.smem_Imm.begin();
      auto sImm_store = make_slm_tensor<SmemElementImm>(ptr_sImm_store, SmemLayoutD{});
      ThrCopy thrblk_reduce = params.adma_store_reduce.get_slice(thread_idx);
      auto bSG_sImm_reduce = thrblk_reduce.partition_S(sImm_store);
      auto bSG_gD_reduce   = thrblk_reduce.partition_D(gD_epi);
      auto abar_fred = fred_pipeline.consumer_get_barrier(fred_pipe_consumer_state);
      copy(params.adma_store_reduce.with(abar_fred),
           bSG_sImm_reduce(_,_,_,fred_pipe_consumer_state.index()),
           bSG_gD_reduce(_,_,_,_0{},_0{}));
      fred_pipeline.consumer_commit(fred_pipe_consumer_state, TransactionBytesImm);
    }
    ++fred_pipe_consumer_state;
  }

  CUTLASS_DEVICE static void
  streamk_arrive_counter(
      int* sk_tile_counter,
      uint64_t sk_tile_idx,
      int sk_k_tile_count,
      bool lane_predicate) {
    if (lane_predicate) {
      sycl::atomic_ref<int,
          sycl::memory_order_acq_rel,
          sycl::memory_scope_device,
          sycl::access::address_space::global_space> counter(sk_tile_counter[sk_tile_idx]);
      counter += sk_k_tile_count;
    }
  }

  CUTLASS_DEVICE static void
  streamk_wait_for_previous_splits(
      int* sk_tile_counter,
      uint64_t sk_tile_idx,
      int sk_k_idx) {
    sycl::atomic_ref<int,
        sycl::memory_order_acq_rel,
        sycl::memory_scope_device,
        sycl::access::address_space::global_space> counter(sk_tile_counter[sk_tile_idx]);
    auto kcounter = counter.load();
    while (kcounter < sk_k_idx) {
      kcounter = counter.load();
    }
  }

  template <class G2SPipeline_, class G2SPipelineState_, class ThreadIdx_, class TensorD_>
  CUTLASS_DEVICE void
  streamk_load_gmem_d_to_smem(
      G2SPipeline_& g2s_pipeline,
      G2SPipelineState_& g2s_pipe_producer_state,
      ThreadIdx_ thread_idx,
      TensorStorage& shared_tensors,
      TensorD_ gD_epi,
      bool lane_predicate) const {
    g2s_pipeline.producer_acquire(g2s_pipe_producer_state);
    if (lane_predicate) {
      auto ptr_sImm_load = shared_tensors.collective.smem_Imm.begin();
      auto sImm_load = make_slm_tensor<SmemElementImm>(ptr_sImm_load, SmemLayoutD{});
      ThrCopy thrblk_load = params.adma_load_d.get_slice(thread_idx);
      auto bGS_gD_load   = thrblk_load.partition_S(gD_epi);
      auto bGS_sImm_load = thrblk_load.partition_D(sImm_load);
      auto g2s_barrier = g2s_pipeline.producer_get_barrier(g2s_pipe_producer_state);
      g2s_pipeline.producer_expect_transaction(g2s_pipe_producer_state);
      copy(params.adma_load_d.with(g2s_barrier, uint16_t(0)),
           bGS_gD_load(_,_,_,_0{},_0{}),
           bGS_sImm_load(_,_,_,_0{}));
    }
    g2s_pipeline.producer_commit(g2s_pipe_producer_state);
    ++g2s_pipe_producer_state;
  }

  template <class T, class Iterator, class Layout>
  CUTLASS_HOST_DEVICE
  static constexpr auto make_slm_tensor(Iterator const& iter, Layout const& layout) {
    auto retiled_layout = CoreMatrix::retile<T>(layout);
    return make_tensor(make_smem_ptr(iter), retiled_layout);
  }

  template <class Tensor>
  CUTLASS_HOST_DEVICE
  static constexpr auto as_cm_tensor(Tensor tensor) {
    using ValType = typename decltype(tensor)::value_type;
    return make_slm_tensor<ValType>(tensor.data(), tensor.layout());
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
