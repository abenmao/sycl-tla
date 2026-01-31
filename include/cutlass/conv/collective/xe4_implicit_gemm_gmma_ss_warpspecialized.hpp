#pragma once

#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/atom/copy_traits_xe4_im2col.hpp"
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

template <
  conv::Operator ConvOp,
  int Stages,
  int NumSpatialDims,
  class ClusterShape,
  class KernelSchedule,
  int PipelineAsyncMmaStages,
  class TileShape_,
  class ElementA_,
  class ElementB_,
  class TiledMma_,
  class TileTraitsA_,
  class TileTraitsB_
>
struct CollectiveConv<
    MainloopXe4DmaGmmaWarpSpecializedImplicitGemm<
    ConvOp, Stages, NumSpatialDims, ClusterShape, KernelSchedule, PipelineAsyncMmaStages>,
    TileShape_,
    ElementA_,
    ElementB_,
    TiledMma_,
    TileTraitsA_,
    TileTraitsB_>
{
  static_assert(Stages >= 2, "Specialization requires Stages set to value 2 or more.");

  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopXe4DmaGmmaWarpSpecializedImplicitGemm<
    ConvOp, Stages, NumSpatialDims, ClusterShape, KernelSchedule, PipelineAsyncMmaStages>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using TiledMma = TiledMma_;
  using GmemTiledCopyA = typename TileTraitsA_::GmemTiledCopy;
  using GmemTiledCopyB = typename TileTraitsB_::GmemTiledCopy;
  using SmemLayoutA = typename TileTraitsA_::SmemLayout;
  using SmemLayoutB = typename TileTraitsB_::SmemLayout;
  using ElementAccumulator = typename TiledMma::ValTypeC;

  using ArchTag = typename DispatchPolicy::ArchTag;
  static constexpr int NumSpatialDimensions = DispatchPolicy::NumSpatialDimensions;
  static constexpr int NumTensorDimensions = NumSpatialDimensions + 2;

  using StrideA = decltype(detail::xe4_dispatch_policy_to_stride_A<DispatchPolicy>());
  using StrideB = decltype(detail::xe4_dispatch_policy_to_stride_B<DispatchPolicy>());

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState  = typename MainloopPipeline::PipelineState;

  using ProblemShape = ConvProblemShape<ConvOp, NumSpatialDimensions>;

  static constexpr bool is_im2col_A = detail::is_im2col_load<GmemTiledCopyA>::value;
  static constexpr bool is_im2col_B = detail::is_im2col_load<GmemTiledCopyB>::value;

  using SmemLayoutAcc = decltype(make_layout(take<0,2>(TileShape{}), GenRowMajor{}));

  constexpr static size_t SmemAlignment = 512;

  struct SharedStorage
  {
    struct TensorStorage
    {
      cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>, SmemAlignment> smem_A;
      cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>, SmemAlignment> smem_B;
      cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayoutAcc>, SmemAlignment> smem_Acc;
    } tensors;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;

  static constexpr uint32_t SlmBytesA = size(take<0,2>(SmemLayoutA{}))
    * static_cast<uint32_t>(sizeof(ElementA));
  static constexpr uint32_t SlmBytesB = size(take<0,2>(SmemLayoutB{}))
    * static_cast<uint32_t>(sizeof(ElementB));
  static constexpr uint32_t TmaTransactionBytes = SlmBytesA + SlmBytesB;

  template<class FragmentA, class FragmentB, class FragmentC>
  struct MmaParams {
    TiledMma tiled_mma;
    FragmentA tCrA;
    FragmentB tCrB;
    FragmentC tCrC;

    CUTLASS_DEVICE
    MmaParams (
        TiledMma tiled_mma_,
        FragmentA tCrA_, FragmentB tCrB_, FragmentC tCrC_)
    : tiled_mma(tiled_mma_)
    , tCrA(tCrA_), tCrB(tCrB_), tCrC(tCrC_) {}
  };

  struct Arguments {
    ElementA const* ptr_A {nullptr};
    ElementB const* ptr_B {nullptr};
  };

private:
  template <class TensorA>
  static constexpr auto
  get_tma_load_a_instance(TensorA const& tensor_a, ProblemShape const& problem_shape) {
    if constexpr (is_im2col_A) {
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

      return make_im2col_tma_copy(
        GmemTiledCopyA{},
        tensor_a,
        make_layout(make_shape(size<0>(TileShape{}), size<2>(TileShape{})),
          make_stride(size<2>(TileShape{}), Int<1>{})),
        make_shape(size<0>(TileShape{}), size<2>(TileShape{})),
        size<1>(ClusterShape{}),
        shape(lower_corner_whd),
        shape(upper_corner_whd),
        cute::reverse(shape(problem_shape.lower_padding)),
        cute::reverse(shape(problem_shape.upper_padding)),
        cute::reverse(shape(problem_shape.traversal_stride)),
        shape(lower_srt),
        shape(stride_srt));
    }
    // TMA tiled mode for tensor A in wgrad kernel.
    else {
      return make_tma_copy<ElementA>(
          GmemTiledCopyA{},
          tensor_a,
          SmemLayoutA{}(_,_,_0{}),
          make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
          size<1>(ClusterShape{}));
    }
  }

  // Get tma_load_b instantce.
  template <class TensorB>
  static constexpr auto
  get_tma_load_b_instance(TensorB const& tensor_b, ProblemShape const& problem_shape) {
    if constexpr (is_im2col_B) {
      // compute the upper and lower corners based on the conv padding
      auto lower_corner_whd = detail::compute_lower_corner_whd(problem_shape);
      auto upper_corner_whd = detail::compute_upper_corner_whd(problem_shape);
      auto lower_srt = detail::compute_lower_srt(problem_shape);

      return make_im2col_tma_copy(
          GmemTiledCopyB{},
          tensor_b,
          make_layout(make_shape(size<1>(TileShape{}), size<2>(TileShape{})),
            make_stride(Int<1>{}, size<1>(TileShape{}))),
          make_shape(size<1>(TileShape{}), size<2>(TileShape{})),
          size<0>(ClusterShape{}),
          shape(lower_corner_whd),
          shape(upper_corner_whd),
          cute::reverse(shape(problem_shape.lower_padding)),
          cute::reverse(shape(problem_shape.upper_padding)),
          cute::reverse(shape(problem_shape.traversal_stride)),
          shape(lower_srt),
          cute::reverse(shape(problem_shape.dilation)));
    }
    else {
      return make_tma_copy<ElementB>(
          GmemTiledCopyB{},
          tensor_b,
          SmemLayoutB{}(_,_,_0{}),
          make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
          size<0>(ClusterShape{}));
    }
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
    // Assumption: StrideA is congruent with Problem_MK
    // Select TMA load type according to convolution operator.
    using TensorShapeA = cute::conditional_t<ConvOp == conv::Operator::kWgrad,
        decltype(repeat_like(StrideA{}, int32_t(0))),
        decltype(make_shape(_Submode{}, int(0)))>;

    using TensorShapeB = cute::conditional_t<ConvOp == conv::Operator::kWgrad,
        decltype(make_shape(int(0), _Submode{})),
        decltype(repeat_like(StrideB{}, int32_t(0)))>;

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
  template<class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
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
    
    return {
      tma_load_a,
      tma_load_b,
      TmaTransactionBytes
    };
  }

  template <class ProblemShapeMNKL, class TensorDesc>
  CUTLASS_DEVICE auto
  load_init(ProblemShapeMNKL const& problem_shape_MNKL, Params const& mainloop_params, TensorDesc const& tensor_desc){
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;

    if constexpr (!is_im2col_A) {
      mainloop_params.tma_load_a.cache_.set_tensor_desc(tensor_desc);
    }

    if constexpr (!is_im2col_B) {
      mainloop_params.tma_load_b.cache_.set_tensor_desc(tensor_desc);
    }
    
    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mk = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K));                            // (m,k)
    Tensor mB_nk = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K));                            // (n,k)

    // Make tiled views, defer the slice
    Tensor gA_mk = local_tile(mA_mk, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k)
    Tensor gB_nk = local_tile(mB_nk, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k)

    return cute::make_tuple(gA_mk, gB_nk);
  }

  /// Set up the data needed by this collective for mma compute.
  CUTLASS_DEVICE auto
  mma_init(TensorStorage& shared_tensors) const {
    auto sA = make_tensor(shared_tensors.smem_A.data(), SmemLayoutA {});     // (BLK_M,BLK_K,PIPE) 
    auto sB = make_tensor(shared_tensors.smem_B.data(), SmemLayoutB {});     // (BLK_N,BLK_K,PIPE)
    auto sAcc = make_tensor(shared_tensors.smem_Acc.data(), SmemLayoutAcc{}); // (BLK_M,BLK_N)

    // Allocate "fragments/descriptors" for A and B matrices
    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(0);
    auto tCsA = thread_mma.partition_fragment_A(sA);            // (MMA,MMA_M,MMA_K,PIPE)
    auto tCsB = thread_mma.partition_fragment_B(sB);            // (MMA,MMA_N,MMA_K,PIPE)
    auto tCsAcc = thread_mma.partition_fragment_C(sAcc);        // (MMA,MMA_M,MMA_N)

    MmaParams<decltype(tCsA), decltype(tCsB), decltype(tCsAcc)> mma_params {
      tiled_mma,
      tCsA, tCsB, tCsAcc
    };

    return mma_params;
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

    int slice_idx_A = 0, slice_idx_B = 0;
    if constexpr (is_im2col_A) {
      slice_idx_A = thread_idx;
    }

    if constexpr (is_im2col_B) {
      slice_idx_B = thread_idx;
    }

    auto block_load_a = mainloop_params.tma_load_a.get_slice(slice_idx_A);
    auto block_load_b = mainloop_params.tma_load_b.get_slice(slice_idx_B);

    auto [gA_mk, gB_nk] = load_inputs;

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord] = blk_coord;

    Tensor gA = gA_mk(_,_,m_coord,_);                                                     // (BLK_M,BLK_K,k)
    Tensor gB = gB_nk(_,_,n_coord,_);                                                     // (BLK_N,BLK_K,k)

    // Applies the mapping from block_tma_a
    Tensor tAgA = block_load_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = block_load_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_load_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_load_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

    // Mainloop
    CUTLASS_PRAGMA_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      uint32_t write_stage = smem_pipe_producer_state.index();
      auto abar_prod = pipeline.producer_get_barrier(smem_pipe_producer_state);

      pipeline.producer_acquire(smem_pipe_producer_state);
      if constexpr(is_im2col_A) {
        copy(mainloop_params.tma_load_a.with(abar_prod), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
      } else {
        if (elect_one_sync()) {
          copy(mainloop_params.tma_load_a.with(abar_prod), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        }
      }

      if constexpr(is_im2col_B) {
        copy(mainloop_params.tma_load_b.with(abar_prod), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
      } else {
        if (elect_one_sync()) {
          copy(mainloop_params.tma_load_b.with(abar_prod), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        }
      }

      ++k_tile_iter;
      ++smem_pipe_producer_state;
    }
    return smem_pipe_producer_state;
  }

  template <class Pipelines, class PipelineStates, class FrgTensorC, class MmaParams>
  CUTLASS_DEVICE auto
  mma(Pipelines pipelines, PipelineStates pipeline_states, FrgTensorC& tensor_c, MmaParams const& mma_inputs, int k_tile_count) {
    
    auto [mainloop_pipeline, store_pipeline, accumulator_pipeline] = pipelines;
    auto [mainloop_pipe_consumer_state, store_pipe_producer_state, accumulator_pipe_producer_state] = pipeline_states;
    auto [tiled_mma, tCsA, tCsB, tCsAcc] = mma_inputs;

    auto thread_mma = tiled_mma.get_thread_slice(0);
    auto tCsC = thread_mma.partition_fragment_C(tensor_c);            // (MMA,MMA_M,MMA_N)

    uint64_t mma_ctrl = 0x100;

    while (k_tile_count > 0) {
      mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state);
      mainloop_pipeline.consumer_commit(mainloop_pipe_consumer_state, 2);

      uint32_t read_stage = mainloop_pipe_consumer_state.index();
      auto abar_cons = mainloop_pipeline.consumer_get_barrier(mainloop_pipe_consumer_state);

      // Unroll the K mode manually so we can set mma_ctrl to 0
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCsA); ++k_block) {
        bool is_last_iter = (k_tile_count == 1) && (k_block == size<2>(tCsA) - 1);

        if (is_last_iter) {
          store_pipeline.producer_try_acquire(store_pipe_producer_state);
          accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);

          int write_stage = accumulator_pipe_producer_state.index();
          auto abar_cons_d = accumulator_pipeline.producer_get_barrier(accumulator_pipe_producer_state);
          auto new_tiled_mma = tiled_mma.with(mma_ctrl, make_tuple(abar_cons_d, abar_cons, abar_cons));
          cute::gemm(new_tiled_mma, tCsC(_,_,_,write_stage), tCsA(_,_,k_block,read_stage), tCsB(_,_,k_block,read_stage), tCsAcc);
        } else {
          auto new_tiled_mma = tiled_mma.with(mma_ctrl, make_tuple(abar_cons, abar_cons));
          cute::gemm(new_tiled_mma, tCsA(_,_,k_block,read_stage), tCsB(_,_,k_block,read_stage), tCsAcc);
        }
        mma_ctrl = 0;
      }

      --k_tile_count;
      ++mainloop_pipe_consumer_state;
    }

    return mainloop_pipe_consumer_state;
  }
};
} // namespace cutlass::conv::collective
