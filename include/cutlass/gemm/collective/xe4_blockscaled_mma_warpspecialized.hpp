/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_detail.hpp"
#include "cutlass/float8.h"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/numeric_size.h"
#include "cute/atom/copy_traits_xe4_dma_legacy.hpp"
#include <cute/atom/copy_traits_xe4_adma.hpp>
#include <cute/container/array_subbyte.hpp>

#include "cute/atom/mma_traits_xe4_amma.hpp"
#include "cute/atom/copy_traits_xe4_tma.hpp"
#include <cutlass/detail/xe4_blockscaled_layout.hpp>

namespace cutlass::gemm::collective {

using namespace cute;
using namespace cute::detail;
using namespace cutlass::gemm;

// Maps (ElementSF, SFVecSize) to the hardware block-scale-type encoding.
// Supported: ue8m0+k32(0), ue8m0+k16(1), ue5m3+k32(2), ue5m3+k16(3), ue4m3+k16(5).
template <class ElementSF, int SFVecSize>
struct BlockScaleTypeMap {
  static_assert(cutlass::detail::dependent_false<ElementSF>,
    "Unsupported (ElementSF, SFVecSize) combination for block-scaled MMA.");
};

template <> struct BlockScaleTypeMap<cutlass::float_ue8m0_t, 32> { static constexpr unsigned value = 0; };
template <> struct BlockScaleTypeMap<cutlass::float_ue8m0_t, 16> { static constexpr unsigned value = 1; };
template <> struct BlockScaleTypeMap<cutlass::float_ue5m3_t, 32> { static constexpr unsigned value = 2; };
template <> struct BlockScaleTypeMap<cutlass::float_ue5m3_t, 16> { static constexpr unsigned value = 3; };
template <> struct BlockScaleTypeMap<cutlass::float_ue4m3_t, 16> { static constexpr unsigned value = 5; };

template <class ElementSF, int VS>
CUTLASS_HOST_DEVICE constexpr
unsigned block_scale_type_encoding() {
  return BlockScaleTypeMap<ElementSF, VS>::value;
}

// Block-scaled CollectiveMma for Xe4 (warp-specialized ADMA producer + AMMA consumer).
template <
  int Stages,
  int SchedulerPipelineStageCount,
  int AccumulatorPipelineStageCount,
  class ClusterShape,
  class TileShape_,
  class ElementTupleA_,
  class StridePairA_,
  class ElementTupleB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomPairA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomPairB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
  MainloopXe4DmaGmmaWarpSpecializedBlockScaled<Stages, SchedulerPipelineStageCount, AccumulatorPipelineStageCount, ClusterShape>,
  TileShape_,
  ElementTupleA_,
  StridePairA_,
  ElementTupleB_,
  StridePairB_,
  TiledMma_,
  GmemTiledCopyPairA_,
  SmemLayoutAtomPairA_,
  SmemCopyAtomA_,
  TransformA_,
  GmemTiledCopyPairB_,
  SmemLayoutAtomPairB_,
  SmemCopyAtomB_,
  TransformB_>
{
  // Type aliases
  using TiledMma = TiledMma_;

  using DispatchPolicy = MainloopXe4DmaGmmaWarpSpecializedBlockScaled<
                          Stages,
                          SchedulerPipelineStageCount,
                          AccumulatorPipelineStageCount,
                          ClusterShape>;
  using TileShape = TileShape_;

  // Extract individual types from Tuple bundles
  using ElementTupleA = ElementTupleA_;
  using ElementTupleB = ElementTupleB_;

  using ElementA   = cute::remove_cvref_t<decltype(get<0>(ElementTupleA{}))>;
  using ElementSFA = cute::remove_cvref_t<decltype(get<1>(ElementTupleA{}))>;
  static constexpr int SFVecSize = decltype(get<2>(ElementTupleA{}))::value;

  using ElementB   = cute::remove_cvref_t<decltype(get<0>(ElementTupleB{}))>;
  using ElementSFB = cute::remove_cvref_t<decltype(get<1>(ElementTupleB{}))>;

  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementBMma = typename TiledMma::ValTypeB;
  using ElementAccumulator = typename TiledMma::ValTypeC;

  // StridePairA/B = cute::tuple<StrideData, LayoutSF>
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;
  using StrideA   = cute::remove_cvref_t<decltype(get<0>(StridePairA{}))>;
  using LayoutSFA = cute::remove_cvref_t<decltype(get<1>(StridePairA{}))>;
  using StrideB   = cute::remove_cvref_t<decltype(get<0>(StridePairB{}))>;
  using LayoutSFB = cute::remove_cvref_t<decltype(get<1>(StridePairB{}))>;

  // GmemTiledCopyPairA/B = cute::tuple<CopyOpData, CopyOpSF>
  using GmemTiledCopyA   = cute::remove_cvref_t<decltype(get<0>(GmemTiledCopyPairA_{}))>;
  using GmemTiledCopySFA = cute::remove_cvref_t<decltype(get<1>(GmemTiledCopyPairA_{}))>;
  using GmemTiledCopyB   = cute::remove_cvref_t<decltype(get<0>(GmemTiledCopyPairB_{}))>;
  using GmemTiledCopySFB = cute::remove_cvref_t<decltype(get<1>(GmemTiledCopyPairB_{}))>;

  // SmemLayoutAtomPairA/B = cute::tuple<SmemLayoutData, SmemLayoutSF>
  using SmemLayoutAtomPairA = SmemLayoutAtomPairA_;
  using SmemLayoutAtomPairB = SmemLayoutAtomPairB_;
  using SmemLayoutAtomA   = cute::remove_cvref_t<decltype(get<0>(SmemLayoutAtomPairA{}))>;
  using SmemLayoutAtomSFA = cute::remove_cvref_t<decltype(get<1>(SmemLayoutAtomPairA{}))>;
  using SmemLayoutAtomB   = cute::remove_cvref_t<decltype(get<0>(SmemLayoutAtomPairB{}))>;
  using SmemLayoutAtomSFB = cute::remove_cvref_t<decltype(get<1>(SmemLayoutAtomPairB{}))>;

  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));

  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));
  using MmaShapeC_MN = decltype(partition_shape_C(TiledMma{}, make_shape(size<0>(TileShape{}), size<1>(TileShape{}))));

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using MainloopPipelineState = typename MainloopPipeline::PipelineState;

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");

  // NOTE: This specialization assumes ClusterShape (1,1,1) — single workgroup, no multicast.
  // TODO :
  // When adding cluster support, update: constructor, load_init (tma_partition 5-arg form),
  // load (mcast_mask_a/b), and mma (cluster_expect_tx).
  static_assert(cute::size(ClusterShape{}) == 1,
      "Only ClusterShape (1,1,1) is supported. See NOTE above for multi-CTA checklist.");

  // Data SMEM layouts — hierarchical via UMMA::tile_to_mma_shape (matching regular mainloop).
  // Produces rank-4: ((MMA_TILE_M,MMA_TILE_K),MMA_M,MMA_K,PIPE)
  // The hierarchical rank >= 3 is required so that make_adma_atom_A/B_xe4 uses
  // the layout<0>(slayout) path instead of coalesce(slayout), which avoids a
  // rank-1 collapse for MN-major (ColMajor) SMEM layouts.
  using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomA{},
      append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideA>(),
        Step<_1,_2,_3>, Step<_2,_1,_3>>{}));

  using SmemLayoutB = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtomB{},
      append(MmaShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideB>(),
        Step<_1,_2,_3>, Step<_2,_1,_3>>{}));

  // Accumulator layout
  using SmemLayoutAcc = decltype(make_layout(MmaShapeC_MN{}, GenRowMajor{}));

  // SF (scale-factor) dimensions
  static constexpr int bM = size<0>(TileShape{});
  static constexpr int bN = size<1>(TileShape{});
  static constexpr int bK = size<2>(TileShape{});
  static constexpr int sf_bK = bK / SFVecSize;     // SF factors per tile along K

  // Physical SF elements per pipeline stage
  static constexpr int SfElemsPerPipeA = bM * sf_bK;
  static constexpr int SfElemsPerPipeB = bN * sf_bK;
  static constexpr int SmemSizeSFA = SfElemsPerPipeA * DispatchPolicy::Stages;
  static constexpr int SmemSizeSFB = SfElemsPerPipeB * DispatchPolicy::Stages;

  // SF SMEM layouts: deduced atom + PIPE dimension appended.
  // Rank-4: ((mnBlock, kBlock), _1, (blk_MN, blk_K), PIPE)
  using SmemLayoutSFA = decltype(make_layout(
      append(shape(SmemLayoutAtomSFA{}),  Int<DispatchPolicy::Stages>{}),
      append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))));
  using SmemLayoutSFB = decltype(make_layout(
      append(shape(SmemLayoutAtomSFB{}),  Int<DispatchPolicy::Stages>{}),
      append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))));

  // Shared Storage
  // smem_A, smem_B, smem_Acc for data; smem_SFA, smem_SFB for scale factors.
  constexpr static size_t SmemAlignment = 512;

  struct SharedStorage
  {
    struct TensorStorage : cute::aligned_struct<SmemAlignment, _0>
    {
      // Sub-byte packing: ElementA/B may be 4-bit (float_e2m1_t) where sizeof==1
      // but sizeof_bits==4. Use ceiling division (bits+7)/8 to get packed byte count.
      // For MMA-tile-aligned layouts cosize_v is always a multiple of 2, so this
      // is equivalent to / 8, but guards against odd-element layouts at no runtime cost.
      cute::array_aligned<uint8_t,
          (cute::cosize_v<SmemLayoutA> * cutlass::sizeof_bits<ElementA>::value + 7) / 8,
          SmemAlignment> smem_A;
      cute::array_aligned<uint8_t,
          (cute::cosize_v<SmemLayoutB> * cutlass::sizeof_bits<ElementB>::value + 7) / 8,
          SmemAlignment> smem_B;
      cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayoutAcc>, SmemAlignment> smem_Acc;
      cute::array_aligned<ElementSFA, SmemSizeSFA, SmemAlignment> smem_SFA;
      cute::array_aligned<ElementSFB, SmemSizeSFB, SmemAlignment> smem_SFB;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  constexpr static uint32_t SlmBytesA = sizeof(TensorStorage::smem_A) / DispatchPolicy::Stages;
  constexpr static uint32_t SlmBytesB = sizeof(TensorStorage::smem_B) / DispatchPolicy::Stages;
  constexpr static uint32_t SlmBytesSFA = SfElemsPerPipeA * static_cast<uint32_t>(sizeof(ElementSFA));
  constexpr static uint32_t SlmBytesSFB = SfElemsPerPipeB * static_cast<uint32_t>(sizeof(ElementSFB));
  constexpr static uint32_t TmaTransactionBytes = SlmBytesA + SlmBytesB + SlmBytesSFA + SlmBytesSFB;

  // LoadParams
  template<
    class KTileCount,
    class GTensorPartitionedA, class GTensorPartitionedB,
    class STensorA, class STensorB,
    class GTensorPartitionedSFA, class GTensorPartitionedSFB,
    class STensorSFA, class STensorSFB
  >
  struct LoadParams {
    KTileCount k_tiles;
    // Data tensor partitions
    GTensorPartitionedA tAgA;
    GTensorPartitionedB tBgB;
    STensorA tAsA;
    STensorB tBsB;
    // SF tensor partitions
    GTensorPartitionedSFA tSFAgSFA;
    GTensorPartitionedSFB tSFBgSFB;
    STensorSFA tSFAsSFA;
    STensorSFB tSFBsSFB;

    CUTLASS_DEVICE
    LoadParams(
        KTileCount k_tiles_,
        GTensorPartitionedA tAgA_, GTensorPartitionedB tBgB_,
        STensorA tAsA_, STensorB tBsB_,
        GTensorPartitionedSFA tSFAgSFA_, GTensorPartitionedSFB tSFBgSFB_,
        STensorSFA tSFAsSFA_, STensorSFB tSFBsSFB_)
    : k_tiles(k_tiles_)
    , tAgA(tAgA_), tBgB(tBgB_)
    , tAsA(tAsA_), tBsB(tBsB_)
    , tSFAgSFA(tSFAgSFA_), tSFBgSFB(tSFBgSFB_)
    , tSFAsSFA(tSFAsSFA_), tSFBsSFB(tSFBsSFB_) {}
  };

  // MmaParams
  template<class FragmentA, class FragmentB, class FragmentC, class SfDescTensorA, class SfDescTensorB>
  struct MmaParams {
    TiledMma tiled_mma;
    FragmentA tCsA;
    FragmentB tCsB;
    FragmentC tCsAcc;
    SfDescTensorA tCsSFA;      // SF descriptor tensor for A (DescriptorIterator-based)
    SfDescTensorB tCsSFB;      // SF descriptor tensor for B (DescriptorIterator-based)

    CUTLASS_DEVICE
    MmaParams(
        TiledMma tiled_mma_,
        FragmentA tCsA_, FragmentB tCsB_, FragmentC tCsAcc_,
        SfDescTensorA tCsSFA_, SfDescTensorB tCsSFB_)
    : tiled_mma(tiled_mma_)
    , tCsA(tCsA_), tCsB(tCsB_), tCsAcc(tCsAcc_)
    , tCsSFA(tCsSFA_), tCsSFB(tCsSFB_) {}
  };

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A {nullptr};
    StrideA dA;
    ElementB const* ptr_B {nullptr};
    StrideB dB;
    ElementSFA const* ptr_SFA {nullptr};
    ElementSFB const* ptr_SFB {nullptr};
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

    // SF ADMA atoms reuse make_adma_atom_A/B_xe4 — the stride-0 broadcast modes
    // in the hierarchical SF SMEM layout are detected at compile time, automatically
    // producing Type3 matrix descriptors instead of Type1/Type2.
    // The SF GMEM tensor must use the hierarchical layout from tile_atom_to_shape_SFA/B
    // so that compose(mma_tiler_mk) in construct_tma_gbasis works correctly with bK.
    using Xe4BlkScaledCfg = cutlass::detail::Xe4BlockScaledConfig<SFVecSize>;
    using ADMA_SFA = decltype(make_adma_atom_A_xe4(
        GmemTiledCopySFA{},
        make_tensor(static_cast<ElementSFA const*>(nullptr),
                    Xe4BlkScaledCfg::tile_atom_to_shape_SFA(
                        make_shape(int32_t(0), int32_t(0), int32_t(0)))),
        SmemLayoutSFA{}(_, _, _, cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );

    using ADMA_SFB = decltype(make_adma_atom_B_xe4(
        GmemTiledCopySFB{},
        make_tensor(static_cast<ElementSFB const*>(nullptr),
                    Xe4BlkScaledCfg::tile_atom_to_shape_SFB(
                        make_shape(int32_t(0), int32_t(0), int32_t(0)))),
        SmemLayoutSFB{}(_, _, _, cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        ClusterLayout_VMNK{})
      );

    ADMA_A   adma_load_a;
    ADMA_B   adma_load_b;
    ADMA_SFA adma_load_sfa;
    ADMA_SFB adma_load_sfb;
  };

  CUTLASS_DEVICE
  CollectiveMma(Params const& params, [[maybe_unused]] ClusterShape cluster_shape) {
    observed_adma_load_a_   = &params.adma_load_a;
    observed_adma_load_b_   = &params.adma_load_b;
    observed_adma_load_sfa_ = &params.adma_load_sfa;
    observed_adma_load_sfb_ = &params.adma_load_sfb;
    // Single-CTA cluster (1,1,1): cooperative set IDs are trivially (0,0)
    coop_set_ids_ = make_tuple(uint32_t(0), uint32_t(0));
  }

  // to_underlying_arguments — build ADMA atoms for data + SF from host arguments
  template<class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    auto [M, N, K, L] = problem_shape;

    auto tensor_a = make_tensor(args.ptr_A, make_layout(make_shape(M,K,L), args.dA));
    auto tensor_b = make_tensor(args.ptr_B, make_layout(make_shape(N,K,L), args.dB));

    // SF GMEM tensors: hierarchical layout from tile_atom_to_shape matching the
    // ADMA atom's compose(mma_tiler_mk) expectations.  The SfAtom's stride-0 broadcast
    // mode inflates the logical K extent to match data tiles.
    // TODO: Batched GEMMs (L>1) not supported yet for block-scaled SF tensors.
    using Xe4BlkScaledCfg = cutlass::detail::Xe4BlockScaledConfig<SFVecSize>;
    CUTLASS_ASSERT(L == 1 && "Block-scaled collective does not yet support batched GEMM (L>1)");
    CUTLASS_ASSERT(K % SFVecSize == 0 && "K dimension must be a multiple of SFVecSize for block-scaled GEMM");
    auto tensor_sfa = make_tensor(make_gmem_ptr(args.ptr_SFA),
        Xe4BlkScaledCfg::tile_atom_to_shape_SFA(make_shape(M, N, K)));
    auto tensor_sfb = make_tensor(make_gmem_ptr(args.ptr_SFB),
        Xe4BlkScaledCfg::tile_atom_to_shape_SFB(make_shape(M, N, K)));

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

    // SF ADMA atoms — reuse make_adma_atom_A/B_xe4 with SF SMEM layout slices
    auto adma_load_sfa = make_adma_atom_A_xe4(
        GmemTiledCopySFA{},
        tensor_sfa,
        SmemLayoutSFA{}(_, _, _, cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    auto adma_load_sfb = make_adma_atom_B_xe4(
        GmemTiledCopySFB{},
        tensor_sfb,
        SmemLayoutSFB{}(_, _, _, cute::Int<0>{}),
        TileShape{},
        TiledMma{},
        cluster_layout_vmnk);

    return {adma_load_a, adma_load_b, adma_load_sfa, adma_load_sfb};
  }

  // Set up the data needed by this collective for load.
  // separate SF tensor setup with flat physical SMEM for tma_partition.
  // Accepts the same 2-element tdesc tuple (tdesc_a, tdesc_b) as the regular mainloop,
  // so the existing kernel (xe4_gemm_dma_warpspecialized.hpp) can be reused unchanged.
  // SF tdesc's are allocated internally at indices 4 and 5.
  template <class ProblemShape, class DescTuple>
  CUTLASS_DEVICE auto
  load_init(ProblemShape const& problem_shape, TensorStorage& shared_tensors, DescTuple const& tdesc_tuple) const {
    using X = Underscore;

    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape;

    // --- Set tensor descriptors for data ADMA atoms (from kernel) ---
    auto [tdesc_a, tdesc_b] = tdesc_tuple;
    observed_adma_load_a_->cache_.set_tensor_desc(tdesc_a);
    observed_adma_load_b_->cache_.set_tensor_desc(tdesc_b);

    // --- Allocate SF tensor descriptors internally ---
    // The kernel passes indices 0-3 (A, B, C, D); we use 4 and 5 for SF.
    auto tdesc_sfa = allocate_tdesc<4>();
    auto tdesc_sfb = allocate_tdesc<5>();
    observed_adma_load_sfa_->cache_.set_tensor_desc(tdesc_sfa);
    observed_adma_load_sfb_->cache_.set_tensor_desc(tdesc_sfb);

    // --- Data tensors ---
    // GMEM tensor views from ADMA atoms
    auto mA = observed_adma_load_a_->get_tma_tensor(make_shape(M, K, L));   // (m, k, l)
    auto mB = observed_adma_load_b_->get_tma_tensor(make_shape(N, K, L));   // (n, k, l)

    // Tile the tensors and defer the slice
    auto gA = local_tile(mA, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});    // (BLK_M, BLK_K, m, k, l)
    auto gB = local_tile(mB, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});    // (BLK_N, BLK_K, n, k, l)

    // SMEM tensors: hierarchical ((MMA_TILE,MMA_TILE),MMA_tiles,MMA_tiles,PIPE)
    // recast_ptr handles sub-byte types (e.g. float_e2m1_t = 4-bit) via subbyte_iterator
    // and is a no-op for >= 8-bit types, so this works for any ElementA/B width.
    auto sA = make_tensor(make_smem_ptr(recast_ptr<ElementA>(shared_tensors.smem_A.data())), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(recast_ptr<ElementB>(shared_tensors.smem_B.data())), SmemLayoutB{});

    // 3-arg tma_partition (single-CTA, no cluster cooperation)
    // group_modes<0,3> groups the first 3 modes (MMA tile structure) leaving PIPE ungrouped
    auto [tAgA, tAsA] = tma_partition(*observed_adma_load_a_,
                                      group_modes<0,3>(sA), group_modes<0,2>(gA));

    auto [tBgB, tBsB] = tma_partition(*observed_adma_load_b_,
                                      group_modes<0,3>(sB), group_modes<0,2>(gB));

    // --- SF tensors ---
    // Hierarchical GMEM tensor views from SF ADMA atoms, matching tile_atom_to_shape.
    // Shape: ((1, MN), (SFVecSize, K/SFVecSize)) — stride-0 broadcast in K sub-mode.
    constexpr int SFVecSizeK = SFVecSize;
    auto mSFA = observed_adma_load_sfa_->get_tma_tensor(
        make_shape(make_shape(Int<1>{}, M), make_shape(Int<SFVecSizeK>{}, K / SFVecSizeK)));
    auto mSFB = observed_adma_load_sfb_->get_tma_tensor(
        make_shape(make_shape(Int<1>{}, N), make_shape(Int<SFVecSizeK>{}, K / SFVecSizeK)));

    // Tile SF GMEM with the same cta_tiler as data, using Step patterns to select M/K or N/K.
    // The hierarchical shape's stride-0 broadcast inflates the logical extent to match data tiles.
    auto cta_tiler = TileShape{};
    auto gSFA = local_tile(mSFA, cta_tiler, make_coord(_,_,_), Step<_1, X, _1>{});  // (bM_hier, bK_hier, k_tiles)
    auto gSFB = local_tile(mSFB, cta_tiler, make_coord(_,_,_), Step< X,_1, _1>{});  // (bN_hier, bK_hier, k_tiles)

    // SF SMEM tensors: hierarchical rank-4 ((sf_block, _1, blk), PIPE)
    Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
    Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

    // tma_partition: group_modes<0,3> on rank-4 SF SMEM groups first 3 modes into DATA,
    // keeping PIPE as iteration mode; group_modes<0,2> on gSFA/gSFB groups (bMN_hier, bK_hier)
    // into DATA, keeping k_tiles as iteration mode.
    auto [tSFAgSFA, tSFAsSFA] = tma_partition(*observed_adma_load_sfa_,
                                              group_modes<0,3>(sSFA), group_modes<0,2>(gSFA));
    auto [tSFBgSFB, tSFBsSFB] = tma_partition(*observed_adma_load_sfb_,
                                              group_modes<0,3>(sSFB), group_modes<0,2>(gSFB));

    LoadParams load_params {
      shape<3>(gA),                                        // k_tiles (for scheduler)
      tAgA, tBgB, tAsA, tBsB,                              // data tensor partitions
      tSFAgSFA, tSFBgSFB, tSFAsSFA, tSFBsSFB               // SF tensor partitions
    };

    return load_params;
  }


  // Set up the data needed by this collective for mma compute.
  CUTLASS_DEVICE auto
  mma_init(TensorStorage& shared_tensors) const {
    // Hierarchical SMEM tensors: ((MMA_TILE,MMA_TILE),MMA_tiles,MMA_tiles,PIPE)
    auto sA   = make_tensor(make_smem_ptr(recast_ptr<ElementA>(shared_tensors.smem_A.data())),   SmemLayoutA{});
    auto sB   = make_tensor(make_smem_ptr(recast_ptr<ElementB>(shared_tensors.smem_B.data())),   SmemLayoutB{});
    auto sAcc = make_tensor(shared_tensors.smem_Acc.data(), SmemLayoutAcc{});

    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sA));   // PIPE at mode 3
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sB));

    // Build descriptor fragments directly from hierarchical SMEM (matching regular mainloop)
    auto tCsA   = TiledMma::make_fragment_A(sA);               // (MMA, MMA_M, MMA_K, PIPE)
    auto tCsB   = TiledMma::make_fragment_B(sB);               // (MMA, MMA_N, MMA_K, PIPE)
    auto tCsAcc   = TiledMma::make_fragment_C(sAcc);             // (MMA, MMA_M, MMA_N)

    TiledMma tiled_mma;

    // Create SF descriptor tensors via MakeTensor<AMMA::smem_sf_desc>.
    // Builds DescriptorIterator-based tensors from rank-4 SF SMEM layouts;
    // indexing by pipeline stage yields a MatrixDescriptor with correct
    // Type-3 StartAddress and Pitch for that stage.
    auto sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
    auto sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});
    auto tCsSFA = make_tensor<AMMA::smem_sf_desc>(sSFA);
    auto tCsSFB = make_tensor<AMMA::smem_sf_desc>(sSFB);

    MmaParams<decltype(tCsA), decltype(tCsB), decltype(tCsAcc),
             decltype(tCsSFA), decltype(tCsSFB)> mma_params {
      tiled_mma,
      tCsA, tCsB, tCsAcc,
      tCsSFA, tCsSFB
    };

    return mma_params;
  }


  // Issue ADMA loads for A, B, SFA, SFB into SLM pipeline stages.
  template <class LoadParams, class TileCoordMNKL, class KTileIterator>
  CUTLASS_DEVICE auto
  load(Params const& mainloop_params, MainloopPipeline mainloop_pipeline, MainloopPipelineState& slm_pipe_write,
    LoadParams const& load_inputs, TileCoordMNKL const& cta_coord_mnkl, KTileIterator k_tile_iter, int k_tile_count) {

    constexpr uint32_t mcast_mask_a = 1;   // single-CTA (see ClusterShape assert)
    constexpr uint32_t mcast_mask_b = 1;
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;
    auto [unused_k_tiles, tAgA_full, tBgB_full, tAsA, tBsB,
          tSFAgSFA_full, tSFBgSFB_full, tSFAsSFA, tSFBsSFB] = load_inputs;

    // Slice out the work coord from partitioned tensors.
    Tensor tAgA = tAgA_full(_, m_coord, _, l_coord);
    Tensor tBgB = tBgB_full(_, n_coord, _, l_coord);

    // TODO: Batched GEMMs (L>1) not supported yet — SF tensors lack L mode.
    Tensor tSFAgSFA = tSFAgSFA_full(_, m_coord, _);
    Tensor tSFBgSFB = tSFBgSFB_full(_, n_coord, _);

    // Issue the Mainloop loads
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > 0) {
      // LOCK mainloop_pipe_producer_state for _writing_
      mainloop_pipeline.producer_acquire(slm_pipe_write);

      uint32_t write_stage = slm_pipe_write.index();
      auto abar_prod = mainloop_pipeline.producer_get_barrier(slm_pipe_write);

      // Data loads
      copy(observed_adma_load_a_->with(abar_prod, mcast_mask_a), tAgA(_,*k_tile_iter), tAsA(_,write_stage));
      copy(observed_adma_load_b_->with(abar_prod, mcast_mask_b), tBgB(_,*k_tile_iter), tBsB(_,write_stage));

      // SF loads — SFA multicasts like A (along N-CTAs), SFB like B (along M-CTAs)
      copy(observed_adma_load_sfa_->with(abar_prod, mcast_mask_a), tSFAgSFA(_,*k_tile_iter), tSFAsSFA(_,write_stage));
      copy(observed_adma_load_sfb_->with(abar_prod, mcast_mask_b), tSFBgSFB(_,*k_tile_iter), tSFBsSFB(_,write_stage));

      --k_tile_count;
      ++k_tile_iter;
      ++slm_pipe_write;
    }

    return cute::make_tuple(slm_pipe_write, k_tile_iter);
  }


  // Execute block-scaled AMMA with SF descriptors and MMAControl.
  template <class Pipelines, class PipelineStates, class FrgTensorC, class MmaParams>
  CUTLASS_DEVICE auto
  mma(Pipelines pipelines, PipelineStates pipeline_states, FrgTensorC& tensor_c, MmaParams const& mma_inputs, int k_tile_count) {

    auto [mainloop_pipeline, store_pipeline, accumulator_pipeline] = pipelines;
    auto [mainloop_pipe_consumer_state, store_pipe_producer_state, accumulator_pipe_producer_state] = pipeline_states;
    auto [tiled_mma, tCsA, tCsB, tCsAcc, tCsSFA, tCsSFB] = mma_inputs;

    auto thread_mma = tiled_mma.get_thread_slice(0);
    auto tCsC = thread_mma.partition_fragment_C(tensor_c);        // (MMA,MMA_M,MMA_N)

    auto wg_expect_tx = size<1>(tCsAcc) * size<2>(tCsAcc) * size<2>(tCsA);
    auto cluster_expect_tx = wg_expect_tx * (size<0>(ClusterShape{}) + size<1>(ClusterShape{}));

    // Block-scaled MMA control register:
    //   NullC=1 for first iteration — bypass C read (D = A*B).
    //   A/B_BlockScaleType derived from (ElementSF, SFVecSize) via block_scale_type_encoding.
    //   After first iteration: NullC=0 to accumulate (D = A*B + C).
    constexpr unsigned bst_a = block_scale_type_encoding<ElementSFA, SFVecSize>();
    constexpr unsigned bst_b = block_scale_type_encoding<ElementSFB, SFVecSize>();
    MMAControl mma_ctrl{};
    mma_ctrl.NullC = 1;
    mma_ctrl.A_BlockScaleType = bst_a;
    mma_ctrl.B_BlockScaleType = bst_b;

    while (k_tile_count > 0) {
      mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state);
      mainloop_pipeline.consumer_commit(mainloop_pipe_consumer_state, cluster_expect_tx);

      int read_stage = mainloop_pipe_consumer_state.index();
      auto abar_cons = mainloop_pipeline.consumer_get_barrier(mainloop_pipe_consumer_state);

      // Unroll the K mode manually so we can set mma_ctrl and barrier tracking
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
                mma_ctrl,
                tCsSFA(0, 0, 0, read_stage), tCsSFB(0, 0, 0, read_stage),
                abar_cons_d, abar_cons, abar_cons),
              tCsC(_,_,_,write_stage),
              tCsA(_,_,k_block,read_stage),
              tCsB(_,_,k_block,read_stage), tCsAcc);
        } else {
          // Override d_type to ElementAccumulator (F32) on non-last iterations so
          // intermediate accumulation stays in full precision in smem_Acc.
          // On the last iteration (above), the original d_type from bs_op_selector
          // is used — the hardware does the F32→F16/BF16 downconversion if needed.
          cute::gemm(
              tiled_mma.with(
                ElementAccumulator{},
                AMMA::TrackMethod<AMMA::Tracking::AB>{},
                mma_ctrl,
                tCsSFA(0, 0, 0, read_stage), tCsSFB(0, 0, 0, read_stage),
                abar_cons, abar_cons),
              tCsA(_,_,k_block,read_stage),
              tCsB(_,_,k_block,read_stage), tCsAcc);
        }
        // After first fma, switch to accumulate mode: NullC=0 reads AMMA-written D from SLM.
        // BlockScaleType bits remain set throughout.
        mma_ctrl.NullC = 0;
      }

      --k_tile_count;
      ++mainloop_pipe_consumer_state;
    }

    return mainloop_pipe_consumer_state;
  }

public:
  typename Params::ADMA_A   const* observed_adma_load_a_{nullptr};
  typename Params::ADMA_B   const* observed_adma_load_b_{nullptr};
  typename Params::ADMA_SFA const* observed_adma_load_sfa_{nullptr};
  typename Params::ADMA_SFB const* observed_adma_load_sfb_{nullptr};
  cute::tuple<uint32_t, uint32_t> coop_set_ids_;
};

} // namespace cutlass::gemm::collective
