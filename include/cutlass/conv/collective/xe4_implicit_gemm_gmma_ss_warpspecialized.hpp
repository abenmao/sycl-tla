#pragma once

#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/atom/copy_traits_sm90_im2col.hpp"
#include "cute/arch/mma_xe4_amma.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/conv/detail.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/conv/collective/detail.hpp"

namespace cutlass::conv::collective {

using namespace cute;
using namespace cute::detail;
using namespace cutlass::gemm;

template <class ConvOp>
constexpr auto
Xe4_dispatch_policy_to_stride_A() {
  if constexpr (ConvOp::value == conv::Operator::kFprop) {
    return cute::Stride<cute::Stride<int64_t, int64_t, int64_t>, cute::Int<1>>{};
  }
  else if constexpr (ConvOp::value == conv::Operator::kWgrad) {
    return cute::Stride<cute::Int<1>, int64_t>{};
  }
  else if constexpr (ConvOp::value == conv::Operator::kDgrad) {
    return cute::Stride<cute::Stride<int64_t, int64_t, int64_t>, cute::Int<1>>{};
  }
  else {
    static_assert("Unsupported ConvOp.");
  }
}

template <class ConvOp>
constexpr auto
Xe4_dispatch_policy_to_stride_B() {
  if constexpr (ConvOp::value == conv::Operator::kFprop) {
    return cute::Stride<int64_t, cute::Stride<cute::Int<1>, int64_t, int64_t>>{};
  }
  else if constexpr (ConvOp::value == conv::Operator::kWgrad) {
    return cute::Stride<cute::Int<1>, cute::Stride<int64_t, int64_t, int64_t>>{};
  }
  else if constexpr (ConvOp::value == conv::Operator::kDgrad) {
    return cute::Stride<cute::Int<1>, cute::Stride<int64_t, int64_t, int64_t>>{};
  }
  else {
    static_assert("Unsupported ConvOp.");
  }
}

template <class ConvOp, class TileShape, int Stages>
constexpr auto
Xe4_dispatch_policy_to_layoutSB() {
  if constexpr (ConvOp::value == conv::Operator::kFprop) {
    return cute::Layout<cute::Shape<decltype(size<1>(TileShape{})), decltype(size<2>(TileShape{})), Int<Stages>>,
                        cute::Stride<decltype(size<2>(TileShape{})), Int<1>, decltype(size<1>(TileShape{}) * size<2>(TileShape{}))>>{};
  }
  else if constexpr (ConvOp::value == conv::Operator::kDgrad) {
    return cute::Layout<cute::Shape<decltype(size<1>(TileShape{})), decltype(size<2>(TileShape{})), Int<Stages>>,
                        cute::Stride<Int<1>, decltype(size<1>(TileShape{})), decltype(size<1>(TileShape{}) * size<2>(TileShape{}))>>{};
  }
  else {
    static_assert("Unsupported ConvOp.");
  }
}

template <
  class ConvOp_,
  int Stages,
  int NumSpatialDims,
  class ClusterShape,
  class KernelSchedule,
  int PipelineAsyncMmaStages,
  class TileShape_,
  class ElementA_,
  class ElementB_,
  class TiledMma_,
  class SmemLayoutAtomA_,
  class SmemLayoutAtomB_>
struct CollectiveConv
{
  static_assert(Stages >= 2, "Specialization requires Stages set to value 2 or more.");

  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopXe4DmaGmmaWarpSpecializedImplicitGemm<
    ConvOp_, Stages, NumSpatialDims, ClusterShape, KernelSchedule, PipelineAsyncMmaStages>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using TiledMma = TiledMma_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using ElementAccumulator = typename TiledMma::ValTypeC;

  using ArchTag = typename DispatchPolicy::ArchTag;
  static constexpr int NumSpatialDimensions = DispatchPolicy::NumSpatialDimensions;
  static constexpr int NumTensorDimensions = NumSpatialDimensions + 2;

  using StrideA = decltype(Xe4_dispatch_policy_to_stride_A<ConvOp_>());
  using StrideB = decltype(Xe4_dispatch_policy_to_stride_B<ConvOp_>());
  using LayoutSB = decltype(Xe4_dispatch_policy_to_layoutSB<ConvOp_, TileShape, Stages>());

  static constexpr slm_matrix_type cmTypeA =
    TiledMma::tnspA == cute::SM90::GMMA::Major::K ? slm_matrix_type::type1 : slm_matrix_type::type2;
  using GmemTiledCopyA = cute::xe4::ASYNC_ROW_LOAD_IM2COL<cmTypeA>;
  using GmemTiledCopyB = cute::xe4::ASYNC_TENSOR_LOAD<slm_matrix_type::type1>;

  using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    append(select<0,2>(TileShape{}), Int<Stages>{}),
    cute::conditional_t<TiledMma::tnspA == cute::SM90::GMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{},
    append(select<1,2>(TileShape{}), Int<Stages>{}),
    cute::conditional_t<TiledMma::tnspB == cute::SM90::GMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState  = typename MainloopPipeline::PipelineState;

  static constexpr auto ConvOp = ConvOp_::value;
  using ProblemShape = ConvProblemShape<ConvOp, NumSpatialDimensions>;

  static constexpr bool is_im2col_A = true;
  static constexpr bool is_im2col_B = false;

  struct SharedStorage
  {
    struct TensorStorage
    {
      cute::array<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array<ElementB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;

  static constexpr uint32_t SlmBytesA = size(take<0,2>(SmemLayoutA{}))
    * static_cast<uint32_t>(sizeof(ElementA));
  static constexpr uint32_t SlmBytesB = size(take<0,2>(SmemLayoutB{}))
    * static_cast<uint32_t>(sizeof(ElementB));
  static constexpr uint32_t TmaTransactionBytes = SlmBytesA + SlmBytesB;

  struct Arguments {
    ElementA const* ptr_A {nullptr};
    ElementB const* ptr_B {nullptr};
  };

private:
  template <class TensorA>
  static constexpr auto
  get_tma_load_a_instance(TensorA const& tensor_a, ProblemShape const& problem_shape) {
    // compute the upper and lower corners based on the conv padding
    auto lower_corner_whd = detail::compute_lower_corner_whd(problem_shape);
    auto upper_corner_whd = detail::compute_upper_corner_whd(problem_shape);
    auto lower_srt = detail::compute_lower_srt(problem_shape);

    // The calculation of gbasis strides for dgrad kernel needs perform negate for dilation values.
    cute::array<int32_t, NumSpatialDimensions> stride_srt{};
    for (int i = 0; i < NumSpatialDimensions; ++i) {
      stride_srt[i] = ConvOp == conv::Operator::kDgrad ?
        -problem_shape.dilation[NumSpatialDimensions-1-i] :
        problem_shape.dilation[NumSpatialDimensions-1-i];
    }

    return make_im2col_tma_copy<GmemTiledCopyA>(GmemTiledCopyA{}, 
      tensor_a,
      make_layout(make_shape(size<0>(TileShape{}), size<2>(TileShape{})),
        make_stride(size<2>(TileShape{}), Int<1>{})),
      make_shape(size<0>(TileShape{}), size<2>(TileShape{})),
      1,
      shape(lower_corner_whd),
      shape(upper_corner_whd),
      cute::reverse(shape(problem_shape.lower_padding)),
      cute::reverse(shape(problem_shape.upper_padding)),
      cute::reverse(shape(problem_shape.traversal_stride)),
      shape(lower_srt),
      shape(stride_srt)
    );
  }

  // Get tma_load_b instantce.
  template <class TensorB>
  static constexpr auto
  get_tma_load_b_instance(TensorB const& tensor_b, ProblemShape const& problem_shape) {
    return make_tma_copy<GmemTiledCopyB>(GmemTiledCopyB{},
      tensor_b,
      LayoutSB{}(_, _, 0),
      make_shape(size<1>(TileShape{}), make_shape(size<2>(TileShape{}))),
      _1{}
    );
  }

public:
  // Performs im2col transformations on the input of type ConvProblemShape
  static constexpr auto
  get_problem_shape_MNKL(ProblemShape const& problem_shape) {
    if constexpr (is_im2col_A || is_im2col_B) {
      // transformation + im2col linearization
      return cutlass::conv::detail::get_linearized_problem_shape_MNKL(problem_shape);
    }
    else {
      // transformation
      return cutlass::conv::detail::get_transformed_problem_shape_MNKL(problem_shape);
    }
  }

  // Device side kernel params
  struct Params {
    using _Submode = decltype(take<0, NumTensorDimensions - 1>(typename ProblemShape::TensorExtent{}));
    using TensorShapeA = decltype(make_shape(_Submode{}, int(0)));
    using TensorShapeB = decltype(repeat_like(StrideB{}, int32_t(0)));

    using TMA_A = decltype(get_tma_load_a_instance(
      make_tensor(static_cast<ElementA const*>(nullptr), make_layout(TensorShapeA{}, StrideA{})),
      ProblemShape{})
    );

    using TMA_B = decltype(get_tma_load_b_instance(
      make_tensor(static_cast<ElementB const*>(nullptr), make_layout(TensorShapeB{}, StrideB{})),
      ProblemShape{})
    );

    // Members
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
  };

  //
  //  Methods
  //
  // Lowers the host side user facing arguments to the kernel facing lauch params
  template<class TensorDesc>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, TensorDesc tensor_desc, Arguments const& args, void* workspace) {
    // from the flat problem shape arrays of ConvProblemShape<ConvOp, N>, create a rank-3 MNK problem shape tuple
    // tma desc creation depends on the original untransformed domain.

    // A extents.
    auto shape_A_orig = problem_shape.get_shape_A();
    // B extents.
    auto shape_B_orig = problem_shape.get_shape_B();

    // Fill inferred cute strides from flat stride arrays
    auto dA = make_cute_packed_stride(StrideA{}, problem_shape.stride_A, ConvOp);
    auto dB = make_cute_packed_stride(StrideB{}, problem_shape.stride_B, ConvOp);

    Tensor tensor_a = make_tensor(args.ptr_A, make_layout(shape_A_orig, dA));
    Tensor tensor_b = make_tensor(args.ptr_B, make_layout(shape_B_orig, dB));

    auto tma_load_a = get_tma_load_a_instance(tensor_a, problem_shape);
    auto tma_load_b = get_tma_load_b_instance(tensor_b, problem_shape);
    tma_load_b.cache_.set_tensor_desc(tensor_desc);

    return {
      tma_load_a,
      tma_load_b,
      TmaTransactionBytes
    };
  }

  template <class ProblemShapeMNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShapeMNKL const& problem_shape_MNKL, Params const& mainloop_params){
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mk = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K));                            // (m,k)
    Tensor mB_nk = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K));                            // (n,k)

    // Make tiled views, defer the slice
    Tensor gA_mk = local_tile(mA_mk, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k)
    Tensor gB_nk = local_tile(mB_nk, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k)

    return cute::make_tuple(gA_mk, gB_nk);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE auto
  load(
    Params const& mainloop_params,
    MainloopPipeline pipeline,
    PipelineState smem_pipe_producer_state,
    cute::tuple<TensorA, TensorB> const& load_inputs,
    BlockCoord const& blk_coord,
    KTileIterator k_tile_iter, int k_tile_count,
    int thread_idx,
    TensorStorage& shared_tensors) {

    auto sA = make_tensor(shared_tensors.smem_A.data(), SmemLayoutA {});
    auto sB = make_tensor(shared_tensors.smem_B.data(), SmemLayoutB {});

    auto thr_load_a = mainloop_params.tma_load_a.get_slice(thread_idx);
    auto block_load_b = mainloop_params.tma_load_b.get_slice(get_wgid<0>());

    auto [gA_mk, gB_nk] = load_inputs;

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord] = blk_coord;

    Tensor gA = gA_mk(_,_,m_coord,_);                                                     // (BLK_M,BLK_K,k)
    Tensor gB = gB_nk(_,_,n_coord,_);                                                     // (BLK_N,BLK_K,k)

    // Applies the mapping from block_tma_a
    Tensor tAgA = thr_load_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = thr_load_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_load_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_load_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

    // Mainloop
    CUTLASS_PRAGMA_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      uint32_t write_stage = smem_pipe_producer_state.index();
      auto abar_prod = pipeline.producer_get_barrier(smem_pipe_producer_state);

      pipeline.producer_acquire(smem_pipe_producer_state);
      copy(mainloop_params.tma_load_a.with(abar_prod), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));

      if (elect_one_sync()) {
        copy(mainloop_params.tma_load_b.with(abar_prod), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
      }

      ++k_tile_iter;
      ++smem_pipe_producer_state;
    }
    return smem_pipe_producer_state;
  }

  template <class Pipeline, class PipelineState, class FinalPipeline, class FinalPipelineState,
    class FrgTensorAcc, class FrgTensorC, class SlmPtr>
  CUTLASS_DEVICE auto
  mma(Pipeline pipeline, PipelineState slm_pipe_read, FinalPipeline finalPipeline,
    FinalPipelineState finalPipelineState, FrgTensorAcc& accumulator, FrgTensorC& sC,
    int k_tile_count, SlmPtr slm_ptr) {
    auto shared_tensors = reinterpret_cast<TensorStorage*>(slm_ptr);
    auto sA = make_tensor(shared_tensors->smem_A.data(), SmemLayoutA {});
    auto sB = make_tensor(shared_tensors->smem_B.data(), SmemLayoutB {});

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(0);
    auto tCrA = thread_mma.partition_fragment_A(sA);            // (MMA,MMA_M,MMA_K,PIPE)
    auto tCrB = thread_mma.partition_fragment_B(sB);            // (MMA,MMA_N,MMA_K,PIPE)
    auto accum = thread_mma.partition_fragment_C(accumulator);  // (MMA,MMA_M,MMA_N)
    auto tCrC = thread_mma.partition_fragment_C(sC);            // (MMA,MMA_M,MMA_N)

    uint64_t mma_ctrl = 0x100;

    for (uint32_t i = 0; i < k_tile_count - 1; i++) {
      uint32_t abar_index = slm_pipe_read.index();
      auto abar_cons = pipeline.consumer_get_barrier(slm_pipe_read);
      pipeline.consumer_wait(slm_pipe_read);
      cute::gemm(tiled_mma.with(mma_ctrl, make_tuple(abar_cons, abar_cons)), tCrA(_,_,_,abar_index), tCrB(_,_,_,abar_index), accum);
      pipeline.consumer_commit(slm_pipe_read, 2);
      ++slm_pipe_read;
      mma_ctrl = 0;
    }
    {
      auto abar_store = finalPipeline.producer_get_barrier(finalPipelineState);
      auto abar_cons = pipeline.consumer_get_barrier(slm_pipe_read);
      uint32_t abar_index = slm_pipe_read.index();
      pipeline.consumer_wait(slm_pipe_read);
      finalPipeline.producer_acquire(finalPipelineState);
      cute::gemm(tiled_mma.with(mma_ctrl, make_tuple(abar_store, abar_cons, abar_cons)), tCrC, tCrA(_,_,_,abar_index), tCrB(_,_,_,abar_index), accum);
      pipeline.consumer_commit(slm_pipe_read, 2);
      ++slm_pipe_read;
    }
    return slm_pipe_read;
  }
};
} // namespace cutlass::conv::collective
