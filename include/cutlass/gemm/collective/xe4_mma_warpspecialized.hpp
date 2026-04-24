#pragma once

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_detail.hpp"

#include "cutlass/util/packed_stride.hpp"
#include "cute/atom/copy_traits_xe4_dma_legacy.hpp"
#include <cute/atom/copy_traits_xe4_adma.hpp>

#include "cute/atom/mma_traits_xe4_amma.hpp"
#include "cute/atom/copy_traits_xe4_tma.hpp"

namespace cutlass::gemm::collective {

using namespace cute;
using namespace cute::detail;
using namespace cutlass::gemm;

template <
  int Stages,
  int SchedulerPipelineStageCount,
  int AccumulatorPipelineStageCount,
  class ClusterShape,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
  MainloopXe4DmaGmmaWarpSpecialized<Stages, SchedulerPipelineStageCount, AccumulatorPipelineStageCount, ClusterShape>,
  TileShape_,
  ElementA_,
  StrideA_,
  ElementB_,
  StrideB_,
  TiledMma_,
  GmemTiledCopyA_,
  SmemLayoutAtomA_,
  SmemCopyAtomA_,
  TransformA_,
  GmemTiledCopyB_,
  SmemLayoutAtomB_,
  SmemCopyAtomB_,
  TransformB_>
{
  using TiledMma = TiledMma_;
  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;

  using DispatchPolicy = MainloopXe4DmaGmmaWarpSpecialized<
                          Stages,
                          SchedulerPipelineStageCount,
                          AccumulatorPipelineStageCount,
                          ClusterShape>;
  using TileShape = TileShape_;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;

  CUTE_STATIC_ASSERT_V(evenly_divides(TileShape{}, tile_shape(TiledMma{})),
                      "Static cluster shape used: TileShape should be evenly divided by TiledMma");

  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));

  // Define A and B block shapes for reduced size TMA_LOADs
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));
  using MmaShapeC_MN = decltype(partition_shape_C(TiledMma{}, make_shape(size<0>(TileShape{}), size<1>(TileShape{}))));

  using ElementA = ElementA_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StrideB = StrideB_;

  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemLayoutAtomC = decltype(make_layout(make_shape(get<0>(shape(SmemLayoutAtomA{})), get<0>(shape(SmemLayoutAtomB{}))), GenRowMajor{}));
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
  using MainloopPipelineState = typename MainloopPipeline::PipelineState;

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");

  // Tile along K mode first before tiling over MN. PIPE mode last as usual.
  // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
  // (MMA_TILE_M,MMA_TILE_K),MMA_M,MMA_K,PIPE)
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideA>(), Step<_1,_2,_3>, Step<_2,_1,_3>>{}));
  // (MMA_TILE_N,MMA_TILE_K),MMA_N,MMA_K,PIPE)
  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(MmaShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideB>(), Step<_1,_2,_3>, Step<_2,_1,_3>>{}));
  using SmemLayoutC = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomC{},
      append(MmaShapeC_MN{}, Int<1>{}), Step<_2,_1,_3>{}));

  using SmemLayoutAcc = decltype(make_layout(MmaShapeC_MN{}, GenRowMajor{}));

  constexpr static size_t SmemAlignment = 512;

  struct SharedStorage
  {
    struct TensorStorage : cute::aligned_struct<SmemAlignment, _0>
    {
      cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>, SmemAlignment> smem_A;
      cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>, SmemAlignment> smem_B;
      cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayoutAcc>, SmemAlignment> smem_Acc;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  constexpr static uint32_t SlmBytesA = sizeof(TensorStorage::smem_A) / Stages;
  constexpr static uint32_t SlmBytesB = sizeof(TensorStorage::smem_B) / Stages;
  constexpr static uint32_t TmaTransactionBytes = SlmBytesA + SlmBytesB;

  template<
    class KTileCount,
    class GTensorPartitionedA, class GTensorPartitionedB,
    class STensorA, class STensorB
  >
  struct LoadParams {
    // for scheduler
    KTileCount k_tiles;
    // for input tensor values
    GTensorPartitionedA tAgA_mkl;
    GTensorPartitionedB tBgB_nkl;
    STensorA tAsA;
    STensorB tBsB;

    CUTLASS_DEVICE
    LoadParams (
        KTileCount k_tiles_,
        GTensorPartitionedA tAgA_mkl_, GTensorPartitionedB tBgB_nkl_,
        STensorA tAsA_, STensorB tBsB_)
    : k_tiles(k_tiles_)
    , tAgA_mkl(tAgA_mkl_), tBgB_nkl(tBgB_nkl_)
    , tAsA(tAsA_), tBsB(tBsB_) {}
  };

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

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A {nullptr};
    StrideA dA;
    ElementB const* ptr_B {nullptr};
    StrideB dB;
  };

  // Device side kernel params
  struct Params {
    using ClusterLayout_VMNK = decltype(tiled_divide(make_layout(ClusterShape{}),
                                                     make_tile(typename TiledMma::AtomThrID{})));

    using ADMA_A = decltype(make_adma_atom_A_xe4(
      GmemTiledCopyA{},
      make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
      SmemLayoutA{}(_,_,_,cute::Int<0>{}),
      TileShape{},
      TiledMma{},
      ClusterLayout_VMNK{})
    );

    using ADMA_B = decltype(make_adma_atom_B_xe4(
        GmemTiledCopyB{},
        make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );

    ADMA_A adma_load_a;
    ADMA_B adma_load_b;
  };

  CUTLASS_DEVICE
  CollectiveMma(Params const& params, ClusterShape cluster_shape) : cluster_shape_(cluster_shape) {
    {
      initialize_mcast_masks();
      observed_adma_load_a_ = &params.adma_load_a;
      observed_adma_load_b_ = &params.adma_load_b;
    }
  }

  template<class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    auto [M, N, K, L] = problem_shape;

    auto tensor_a = make_tensor(args.ptr_A, make_layout(make_shape(M,K,L), args.dA));
    auto tensor_b = make_tensor(args.ptr_B, make_layout(make_shape(N,K,L), args.dB));

    auto cluster_shape = ClusterShape{};
    auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));

    auto adma_load_a = make_adma_atom_A_xe4(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    auto adma_load_b = make_adma_atom_B_xe4(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,_,cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    return {adma_load_a, adma_load_b};
  }

  CUTLASS_DEVICE void
  initialize_mcast_masks() {
    uint32_t cluster_wgid_x = get_cluster_wgid<0>();
    uint32_t cluster_wgid_y = get_cluster_wgid<1>();
    auto cluster_layout_mn = make_layout(select<0,1>(ClusterShape{}));
    block_rank_in_cluster_ = cluster_layout_mn(make_coord(cluster_wgid_x, cluster_wgid_y));

    Layout cta_layout_mnk  = make_layout(cluster_shape_);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster_);
    uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
    uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);
    cluster_masks_ = make_tuple(static_cast<uint32_t>(mcast_mask_a), static_cast<uint32_t>(mcast_mask_b));
  }

  template <class ProblemShape, class DescTuple>
  CUTLASS_DEVICE auto
  load_init(ProblemShape const& problem_shape, TensorStorage& shared_tensors, DescTuple const& tdesc_tuple) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape;

    auto [tdesc_a, tdesc_b] = tdesc_tuple;
    observed_adma_load_a_->set_tensor_desc(tdesc_a);
    observed_adma_load_b_->set_tensor_desc(tdesc_b);

    // Represent the full tensors -- get these from TMA
    auto mA_mkl = observed_adma_load_a_->get_tma_tensor(make_shape(M, K, L));   // (m,k,l)
    auto mB_nkl = observed_adma_load_b_->get_tma_tensor(make_shape(N, K, L));   // (n,k,l)

    // Tile the tensors and defer the slice
    auto gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});    // (BLK_M, BLK_K, m, k, l)
    auto gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});    // (BLK_N, BLK_K, n, k, l)

    // Partition for this CTA
    ThrMMA cta_mma = TiledMma{}.get_slice(0);

    Tensor tCgA_mkl = cta_mma.partition_A(gA_mkl);          // (MMA, MMA_M, MMA_K, m, k, l)
    Tensor tCgB_nkl = cta_mma.partition_B(gB_nkl);          // (MMA, MMA_N, MMA_K, n, k, l)

    // Define the CTA-in-cluster Layout and Coord
    Layout cta_layout_mnk  = make_layout(cluster_shape_);
    Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
    auto cta_coord_vmnk  = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster_);

    auto sA = make_tensor(shared_tensors.smem_A.data(), SmemLayoutA{});  // (MMA,MMA_M,MMA_K,PIPE)
    auto sB = make_tensor(shared_tensors.smem_B.data(), SmemLayoutB{});  // (MMA,MMA_N,MMA_K,PIPE)

    // Project the cta_layout for tma_a along the n-modes
    auto [tAgA_mkl, tAsA] = tma_partition(*observed_adma_load_a_,
                                      get<2>(cta_coord_vmnk), make_layout(size<2>(cta_layout_vmnk)),
                                      group_modes<0,3>(sA), group_modes<0,3>(tCgA_mkl));

    // Project the cta_layout for tma_b along the m-modes
    auto [tBgB_nkl, tBsB] = tma_partition(*observed_adma_load_b_,
                                      get<1>(cta_coord_vmnk), make_layout(size<1>(cta_layout_vmnk)),
                                      group_modes<0,3>(sB), group_modes<0,3>(tCgB_nkl));
    
    Tensor rA = TiledMma::make_fragment_A(sA);
    Tensor rB = TiledMma::make_fragment_B(sB);
    Tensor tArA = Tensor{rA.engine(), tAsA.layout()};
    Tensor tBrB = Tensor{rB.engine(), tBsB.layout()};

    LoadParams load_params {
      shape<3>(gA_mkl),                      // for scheduler
      tAgA_mkl, tBgB_nkl, tAsA, tBsB         // for input tensor values
    };

    return load_params;
  }


  /// Set up the data needed by this collective for mma compute.
  CUTLASS_DEVICE auto
  mma_init(TensorStorage& shared_tensors) const {
    auto sA = make_tensor(shared_tensors.smem_A.data(), SmemLayoutA{});     // (BLK_M,BLK_K,PIPE)
    auto sB = make_tensor(shared_tensors.smem_B.data(), SmemLayoutB{});     // (BLK_N,BLK_K,PIPE)
    auto sAcc = make_tensor(shared_tensors.smem_Acc.data(), SmemLayoutAcc{}); // (BLK_M,BLK_N)

    // Allocate "fragments/descriptors" for A and B matrices
    auto tCsA = TiledMma::make_fragment_A(sA);                      // (MMA,MMA_M,MMA_K,PIPE)
    auto tCsB = TiledMma::make_fragment_B(sB);                      // (MMA,MMA_N,MMA_K,PIPE)
    auto tCsAcc = TiledMma::make_fragment_C(sAcc);                  // (MMA,MMA_M,MMA_N)

    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sA));  // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sB));

    TiledMma tiled_mma;

    MmaParams<decltype(tCsA), decltype(tCsB), decltype(tCsAcc)> mma_params {
      tiled_mma,
      tCsA, tCsB, tCsAcc
    };

    return mma_params;
  }

  template <class LoadParams, class TileCoordMNKL, class KTileIterator>
  CUTLASS_DEVICE auto
  load(Params const& mainloop_params, MainloopPipeline mainloop_pipeline, MainloopPipelineState& slm_pipe_write,
    LoadParams const& load_inputs, TileCoordMNKL const& cta_coord_mnkl, KTileIterator k_tile_iter, int k_tile_count) {

    auto [mcast_mask_a, mcast_mask_b] = cluster_masks_;
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;
    auto [unused_k_tiles, tAgA_mkl, tBgB_nkl, tAsA, tBsB] = load_inputs;

    // slice out the work coord from partitioned tensors
    Tensor tAgA = tAgA_mkl(_, m_coord / size(typename TiledMma::AtomThrID{}), _, l_coord);
    Tensor tBgB = tBgB_nkl(_, n_coord, _, l_coord);

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_UNROLL
    while (k_tile_count > 0) {
      // LOCK mainloop_pipe_producer_state for _writing_
      mainloop_pipeline.producer_acquire(slm_pipe_write);

      uint32_t write_stage = slm_pipe_write.index();
      auto abar_prod = mainloop_pipeline.producer_get_barrier(slm_pipe_write);

      copy(observed_adma_load_a_->with(abar_prod, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
      copy(observed_adma_load_b_->with(abar_prod, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_stage));

      --k_tile_count;
      ++k_tile_iter;
      ++slm_pipe_write;
    }

    return cute::make_tuple(slm_pipe_write, k_tile_iter);
  }

  template <class Pipelines, class PipelineStates, class FrgTensorC, class MmaParams>
  CUTLASS_DEVICE auto
  mma(Pipelines pipelines, PipelineStates pipeline_states, FrgTensorC& tensor_c, MmaParams const& mma_inputs, int k_tile_count) {

    auto [mainloop_pipeline, store_pipeline, accumulator_pipeline] = pipelines;
    auto [mainloop_pipe_consumer_state, store_pipe_producer_state, accumulator_pipe_producer_state] = pipeline_states;
    auto [tiled_mma, tCsA, tCsB, tCsAcc] = mma_inputs;

    auto thread_mma = tiled_mma.get_thread_slice(0);
    auto tCsC = thread_mma.partition_fragment_C(tensor_c);        // (MMA,MMA_M,MMA_N)

    auto wg_expect_tx = size<1>(tCsAcc) * size<2>(tCsAcc) * size<2>(tCsA);
    auto cluster_expect_tx = wg_expect_tx * (size<0>(cluster_shape_) + size<1>(cluster_shape_));

    uint64_t mma_ctrl = 0x100;

    while (k_tile_count > 0) {
      mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state);
      mainloop_pipeline.consumer_commit(mainloop_pipe_consumer_state, cluster_expect_tx);

      int read_stage = mainloop_pipe_consumer_state.index();
      auto abar_cons = mainloop_pipeline.consumer_get_barrier(mainloop_pipe_consumer_state);

      // Unroll the K mode manually so we can set mma_ctrl to 0
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCsA); ++k_block) {
        bool is_last_iter = (k_tile_count == 1) && (k_block == size<2>(tCsA) - 1);

        if (is_last_iter) {
          store_pipeline.producer_try_acquire(store_pipe_producer_state);
          accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);

          int write_stage = accumulator_pipe_producer_state.index();
          auto* abar_cons_d = accumulator_pipeline.producer_get_barrier(accumulator_pipe_producer_state);
          cute::gemm(
              tiled_mma.with(
                AMMA::TrackMethod<AMMA::Tracking::DAB>{},
                mma_ctrl, abar_cons_d, abar_cons, abar_cons,
                get<0>(cluster_masks_), get<1>(cluster_masks_)),
              tCsC(_,_,_,write_stage),
              tCsA(_,_,k_block,read_stage),
              tCsB(_,_,k_block,read_stage), tCsAcc);
        } else {
          cute::gemm(
              tiled_mma.with(
                ElementAccumulator {},
                AMMA::TrackMethod<AMMA::Tracking::AB>{},
                mma_ctrl, abar_cons, abar_cons,
                get<0>(cluster_masks_), get<1>(cluster_masks_)),
              tCsA(_,_,k_block,read_stage),
              tCsB(_,_,k_block,read_stage), tCsAcc);
        }
        mma_ctrl = 0;
      }

      --k_tile_count;
      ++mainloop_pipe_consumer_state;
    }

    return mainloop_pipe_consumer_state;
  }

public:
  typename Params::ADMA_A const* observed_adma_load_a_{nullptr};
  typename Params::ADMA_B const* observed_adma_load_b_{nullptr};

  ClusterShape cluster_shape_;
  uint32_t block_rank_in_cluster_;

  cute::tuple<uint32_t, uint32_t> cluster_masks_;
};

}
