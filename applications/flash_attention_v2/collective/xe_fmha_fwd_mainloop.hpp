/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/util/sycl_vec.hpp"
#include "fmha_fusion.hpp"

namespace cutlass::fmha {

template <int Stages> class XeDefault {};   // Default FMHA mainloop, P in registers.

};

namespace cutlass::fmha::collective {

using namespace cute;
template <class datatype, size_t height, size_t width, class Stride = cute::Stride<_1, int64_t, int64_t>, class = void>
struct scale_copy_traits {
  static_assert(cute::dependent_false<cute::tuple<datatype, Int<height>, Int<width>, Stride>>, "scale_copy_traits not defined");
};

// 8 bits
template<class datatype, size_t height, size_t width, class stride>
struct scale_copy_traits<datatype, height, width, stride,
          std::enable_if_t<sizeof_bits_v<datatype> == 8 && height <= 1>> {
  using type = XE_2D_U8x1x16_LD_N;
};
template<class datatype, size_t width, class stride>
struct scale_copy_traits<datatype, 2, width, stride,
          std::enable_if_t<sizeof_bits_v<datatype> == 8>> {
  using type = XE_2D_U8x2x16_LD_N;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class DispatchPolicy_,
          bool CausalMask_,
          bool UseScale_,
          bool F8kvF16mma_,
          class TiledMMAQK_,          // Tiling for Q*K GEMM
          class TiledMMAPV_,          // Tiling for P*V GEMM
          int VTiles_,                // # of tiles in V dimension
          class TensorQ_,             // Global Q/K/V tensors
          class TensorK_,
          class TensorV_,
          class TensorScaleQ_,
          class TensorScaleK_,
          class TensorScaleV_,
          class TiledCopyQ_ = void,   // Optional TiledCopy for loading Q
          class TiledCopyK_ = void,   // Optional TiledCopy for loading K
          class TiledCopyV_ = void>   // Optional TiledCopy for loading V
struct FMHAFwdMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages,
          bool CausalMask_, bool UseScale_, bool F8kvF16mma_,
          class TiledMMAQK_, class TiledMMAPV_, int VTiles_,
          class TensorQ_, class TensorK_, class TensorV_,
          class TensorScaleQ_, class TensorScaleK_, class TensorScaleV_,
          class TiledCopyQ_, class TiledCopyK_, class TiledCopyV_>
struct FMHAFwdMainloop<XeDefault<Stages>, CausalMask_, UseScale_, F8kvF16mma_,
                       TiledMMAQK_, TiledMMAPV_, VTiles_,
                       TensorQ_, TensorK_, TensorV_,
                       TensorScaleQ_, TensorScaleK_, TensorScaleV_,
                       TiledCopyQ_, TiledCopyK_, TiledCopyV_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(take<1,4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;
  using ElementQ = typename TensorQ::element_type;
  static constexpr bool FP4Input = cute::is_same_v<ElementQ, cutlass::float_e2m1_t>;
  using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_,_),0)));
  using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_,_),0)));
  using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_,_),0)));
  using TiledCopyQ = conditional_t<is_void_v<TiledCopyQ_>, decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})), TiledCopyQ_>;
  using TiledCopyK = conditional_t<is_void_v<TiledCopyK_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})), TiledCopyK_>;
  using TiledCopyV = conditional_t<is_void_v<TiledCopyV_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})), TiledCopyV_>;
  static constexpr bool UseScale = UseScale_;
  static constexpr bool F8kvF16mma = F8kvF16mma_;

  using TensorScaleQ = TensorScaleQ_;
  using TensorScaleK = TensorScaleK_;
  using TensorScaleV = TensorScaleV_;
  using TensorScaleQ2D = decltype(TensorScaleQ_{}(append<rank_v<TensorScaleQ_>>(make_coord(_,_),0)));
  using TensorScaleK2D = decltype(TensorScaleK_{}(append<rank_v<TensorScaleK_>>(make_coord(_,_),0)));
  using TensorScaleV2D = decltype(TensorScaleV_{}(append<rank_v<TensorScaleV_>>(make_coord(_,_),0)));
  using ElementScaleQ = typename TensorScaleQ::element_type;
  using ElementScaleK = typename TensorScaleK::element_type;
  using ElementScaleV = typename TensorScaleV::element_type;
  using StrideScaleQ = decltype(stride(TensorScaleQ{}));
  using StrideScaleK = decltype(stride(TensorScaleK{}));
  using StrideScaleV = decltype(stride(TensorScaleV{}));

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
                           make_identity_tensor(select<0,1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;                          // (atom val,q',v')
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;     // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool CausalMask = CausalMask_;

  static constexpr int BLK_Q = get<0>(TileShapeQK{});
  static constexpr int BLK_K = get<1>(TileShapeQK{});
  static constexpr int BLK_QK_D = get<2>(TileShapeQK{});

  static constexpr int BLK_P = get<0>(TileShapePV{});
  static constexpr int BLK_V = get<1>(TileShapePV{});
  static constexpr int BLK_PV_D = get<2>(TileShapePV{});

  static constexpr int ATOM_Q = get<1>(typename TiledMMAQK::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_K = get<2>(typename TiledMMAQK::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_QK_D = get<3>(typename TiledMMAQK::ThrLayoutVMNK{}.shape());

  static constexpr int MMA_Q = get<0>(typename TiledMMAQK::Shape_MNK{});
  static constexpr int MMA_K = get<1>(typename TiledMMAQK::Shape_MNK{});
  static constexpr int MMA_QK_D = get<2>(typename TiledMMAQK::Shape_MNK{});

  static constexpr int ATOM_P = get<1>(typename TiledMMAPV::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_V = get<2>(typename TiledMMAPV::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_PV_D = get<3>(typename TiledMMAPV::ThrLayoutVMNK{}.shape());

  static constexpr int MMA_P = get<0>(typename TiledMMAPV::Shape_MNK{});
  static constexpr int MMA_V = get<1>(typename TiledMMAPV::Shape_MNK{});
  static constexpr int MMA_PV_D = get<2>(typename TiledMMAPV::Shape_MNK{});

  static constexpr int SG_Q = ceil_div(BLK_Q, ATOM_Q);
  static constexpr int SG_K = ceil_div(BLK_K, ATOM_K);
  static constexpr int SG_QK_D = ceil_div(BLK_QK_D, ATOM_QK_D);

  static constexpr int SG_P = ceil_div(BLK_P, ATOM_P);
  static constexpr int SG_V = ceil_div(BLK_V, ATOM_V);
  static constexpr int SG_PV_D = ceil_div(BLK_PV_D, ATOM_PV_D);

  static constexpr auto GROUP_K = 32;

  static_assert(SG_QK_D >= 32, "Intel Xe blockscaled MMA requires SG_QK_D to be at least 32.");
  static_assert(SG_PV_D >= 32, "Intel Xe blockscaled MMA requires SG_PV_D to be at least 32.");
  using CopyThreadShape = Shape<_1, Int<intel::sg_size>>;
  using CopyThreadShapeRev = decltype(cute::reverse(CopyThreadShape{}));

  using DefScaleType = cutlass::float_ue8m0_t;
  using NonVoidElementScaleQ = cute::conditional_t<UseScale, ElementScaleQ, DefScaleType>;
  using NonVoidElementScaleK = cute::conditional_t<UseScale, ElementScaleK, DefScaleType>;
  using NonVoidElementScaleV = cute::conditional_t<UseScale, ElementScaleV, DefScaleType>;
  using NonVoidElementScaleP = cute::conditional_t<UseScale, ElementScaleV, DefScaleType>;
  using GmemTiledCopyScaleQ = typename scale_copy_traits<NonVoidElementScaleQ, MMA_QK_D / GROUP_K, SG_Q>::type;
  using GmemTiledCopyScaleK = typename scale_copy_traits<NonVoidElementScaleK, MMA_QK_D / GROUP_K, SG_K>::type;
  using GmemTiledCopyScaleV = typename scale_copy_traits<NonVoidElementScaleV, MMA_PV_D / GROUP_K, SG_V>::type;
  using GmemTiledCopyScaleP = typename scale_copy_traits<NonVoidElementScaleP, MMA_PV_D / GROUP_K, SG_P>::type;
  template<
  class SelectedGmemTiledCopyScale,
  class StrideScale,
  class ElementScale
  >
  struct TiledCopyScaleTraits {
    using traits_load_scale = Copy_Traits<SelectedGmemTiledCopyScale, StrideScale>;
    using atom_load_scale = Copy_Atom<traits_load_scale, ElementScale>;
    using val_layout_load_scale = decltype(make_layout(shape_div(typename traits_load_scale::BlockShape{}, CopyThreadShapeRev{})));
    using Copy_Scale = decltype(make_tiled_copy(atom_load_scale{}, Layout<CopyThreadShapeRev>{}, val_layout_load_scale{}));
  };

  using Copy_ScaleQ = typename TiledCopyScaleTraits<GmemTiledCopyScaleQ, cute::Stride<_1, int64_t>, NonVoidElementScaleQ>::Copy_Scale;
  using Copy_ScaleK = typename TiledCopyScaleTraits<GmemTiledCopyScaleK, cute::Stride<_1, int64_t>, NonVoidElementScaleK>::Copy_Scale;
  using Copy_ScaleV = typename TiledCopyScaleTraits<GmemTiledCopyScaleV, cute::Stride<_1, int64_t>, NonVoidElementScaleV>::Copy_Scale;
  // If IsATransformed, we need modes M_atom, and M_iter from fragment_A layout else we need mode N_iter from fragment_B layout.
  static constexpr auto scaleQ_traits_size = decltype(size(typename GmemTiledCopyScaleQ::BlockShape{}))::value / intel::sg_size;
  static constexpr auto scaleQ_traits_num = SG_Q / size<1>(typename GmemTiledCopyScaleQ::BlockShape{});
  using FragScaleQLayout = Layout<Shape<Int<scaleQ_traits_size>, Int<scaleQ_traits_num>, Int<SG_QK_D / MMA_QK_D>>>;

  static constexpr int scaleK_traits_size = decltype(size(typename GmemTiledCopyScaleK::BlockShape{}))::value / intel::sg_size;
  static constexpr int scaleK_traits_num = SG_K / size<1>(typename GmemTiledCopyScaleK::BlockShape{});
  using FragScaleKLayout = Layout<Shape<Int<scaleK_traits_size>, Int<scaleK_traits_num>, Int<SG_QK_D / MMA_QK_D>>>;

  static constexpr int scaleV_traits_size = decltype(size(typename GmemTiledCopyScaleV::BlockShape{}))::value / intel::sg_size;
  static constexpr int scaleV_traits_num = SG_V / size<1>(typename GmemTiledCopyScaleV::BlockShape{});
  using FragScaleVLayout = Layout<Shape<Int<scaleV_traits_size>, Int<scaleV_traits_num>, Int<SG_PV_D / MMA_PV_D>>>;

  static constexpr int scaleP_traits_size = decltype(size(typename GmemTiledCopyScaleP::BlockShape{}))::value / intel::sg_size;
  static constexpr int scaleP_traits_num = SG_P / size<1>(typename GmemTiledCopyScaleP::BlockShape{});
  using FragScalePLayout = Layout<Shape<Int<scaleP_traits_size>, Int<scaleP_traits_num>, Int<SG_PV_D / MMA_PV_D>>>;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  FMHAFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr
  Params to_underlying_arguments(Arguments const &args, void * /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;            // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{val};
  }

  CUTLASS_HOST_DEVICE static
  bool can_implement(Arguments const&) {
    return true;
  }

  template <
    int scale_traits_size,
    int scale_traits_num,
    class SelectedGmemTiledCopyScale
  >
  CUTLASS_DEVICE static auto
  make_scale_copy_iterator(int coord, int l_coord, int k_tile_count, int sg_d, int mma_d) {
      return make_tensor(make_inttuple_iter(make_coord(coord, 0, l_coord)),
                         make_layout(make_shape(Int<scale_traits_size>{}, Int<scale_traits_num>{}, sg_d / mma_d, k_tile_count),
                                     make_stride(E<0>{} * _16{}, E<0>{} * size<1>(typename SelectedGmemTiledCopyScale::BlockShape{}), E<1>{} * size<0>(typename SelectedGmemTiledCopyScale::BlockShape{}), E<1>{} * (sg_d / GROUP_K))));
  }

  template <typename QVCoord>
  CUTLASS_DEVICE
  void
  operator()(TensorQ2D const& Q_2D,     // (q,d)
             TensorK2D const& K_2D,     // (k,d)
             TensorV2D const& V_2D,     // (d,k)
             FragA          & tArA,     // Output accumulator (q,v)
             FragARow       & tA_max,   // Softmax row-wise max accumulator
             FragARow       & tA_sum,   // Softmax row-wise sum accumulator
             QVCoord          blk_qv,   // WG tile indices: (Q,V)
             int              blk_k0,   // K block range: [K0,K1)
             int              blk_k1,
             int              thr_id,
             int              seq_len,
             int              l_coord,
             int              full_tile_offset,
             int              discard_seq_coord,
             float            scale_k = 1.0f,
             float            scale_v = 1.0f,
             Copy_ScaleQ const& tiled_copy_scaleQ = Copy_ScaleQ{},
             Copy_ScaleK const& tiled_copy_scaleK = Copy_ScaleK{},
             Copy_ScaleV const& tiled_copy_scaleV = Copy_ScaleV{}) {
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    auto tile_shape_v = make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(Q_2D.shape());             // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());             // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());             // (v,k)
    Tensor cP = make_identity_tensor(take<0,2>(TileShapeQK{})); // (q,k)

    /* Partition global tensors into workgroup tiles */
    Tensor gQ       = local_tile(cQ, TileShapeQK{}, append(blk_qv,_),             Step<_1,X,_1>{});   // (q,d,D)
    Tensor gK       = local_tile(cK, TileShapeQK{}, make_coord(_,_,_),            Step<X,_1,_1>{});   // (k,d,K,D)
    Tensor gV       = local_tile(cV, tile_shape_v,  make_coord(get<1>(blk_qv),_));                    // (v,k,K)
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_,_,0),            Step<X,_1,_1>{});   // (v,k,VV,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);                // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);                // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);          // (atom_val,v',k',VV,K)

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_,_,0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_,_,0,0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_,_,0,0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_,_,0,0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_,_,0,0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v = make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);

    Tensor fragment_scaleQ = make_tensor<ElementScaleQ>(FragScaleQLayout{});
    Tensor fragment_scaleK = make_tensor<ElementScaleK>(FragScaleKLayout{});
    Tensor fragment_scaleV = make_tensor<ElementScaleV>(FragScaleVLayout{});
    // P is dummy scale, just the same as V
    Tensor fragment_scaleP = make_tensor<ElementScaleV>(FragScalePLayout{});
    using scaleQSize = decltype(size(fragment_scaleQ));
    using scaleKSize = decltype(size(fragment_scaleK));
    using scalePSize = decltype(size(fragment_scaleP));
    using scaleVSize = decltype(size(fragment_scaleV));

    
    auto scaleQ_layout = make_layout(
        make_shape(size<0>(tSrQ.shape()), size<1>(tSrQ.shape()), size<2>(tSrQ.shape())),
        make_stride(Int<1>{}, Int<0>{}, Int<0>{}));

    auto scaleK_layout = make_layout(
        make_shape(size<0>(tSrK.shape()), size<1>(tSrK.shape()), size<2>(tSrK.shape())),
        make_stride(Int<1>{}, Int<0>{}, Int<0>{}));
    
    auto scaleV_layout = make_layout(
        make_shape(size<0>(tArV.shape()), size<1>(tArV.shape()), size<2>(tArV.shape())),
        make_stride(Int<1>{}, Int<0>{}, Int<0>{}));

    auto scaleP_layout = make_layout(
        make_shape(size<0>(tArP.shape()), size<1>(tArP.shape()), size<2>(tArP.shape())),
        make_stride(Int<1>{}, Int<0>{}, Int<0>{}));

    using MmaRowQM = decltype(size<1>(tSrQ.shape()));
    auto gemm_qm_indices = make_tensor<uint8_t>(make_shape(MmaRowQM{}));
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < MmaRowQM::value; ++m) {
      gemm_qm_indices(m) = sizeof_bits_v<ElementQ> < 8 ? (m / 2) * 32 + (m % 2) * 8 : m * 8;
    }
    auto gemm_qm_offsets = make_tensor(
        gemm_qm_indices.data(),
        make_layout(make_shape(size<0>(tSrQ.shape()), size<1>(tSrQ.shape()), size<2>(tSrQ.shape())),
                    make_stride(Int<0>{}, Int<1>{}, Int<0>{})));
    
    using MmaColKN = decltype(size<1>(tSrK.shape()));
    auto gemm_kn_indices = make_tensor<uint8_t>(make_shape(MmaColKN{}));
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < MmaColKN::value; ++n) {
      gemm_kn_indices(n) = sizeof_bits_v<ElementQ> < 8 ? n * 32 : n * 16;
    }
    auto gemm_kn_offsets = make_tensor(
        gemm_kn_indices.data(),
        make_layout(make_shape(size<0>(tSrK.shape()), size<1>(tSrK.shape()), size<2>(tSrK.shape())),
                    make_stride(Int<0>{}, Int<1>{}, Int<0>{})));

    using MmaQK_K = decltype(size<2>(tSrK.shape()));
    auto gemm_qk_offsets = make_tensor<uint16_t>(Layout<Shape<_1, _1, MmaQK_K>, Stride<_0, _0, _1>>{});
    auto gemm_kk_offsets = make_tensor<uint16_t>(Layout<Shape<_1, _1, MmaQK_K>, Stride<_0, _0, _1>>{});
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < MmaQK_K::value; ++k) {
      gemm_qk_offsets(k) = static_cast<uint16_t>(k * size(typename GmemTiledCopyScaleQ::BlockShape{}) * (SG_Q / 16));
      gemm_kk_offsets(k) = static_cast<uint16_t>(k * size(typename GmemTiledCopyScaleK::BlockShape{}) * (SG_K / 16));
    }

    using MmaColV = decltype(size<1>(tArV.shape()));
    auto gemm_v_indices = make_tensor<uint8_t>(make_shape(MmaColV{}));
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < MmaColV::value; ++n) {
      gemm_v_indices(n) = sizeof_bits_v<ElementQ> < 8 ? n * 32 : n * 16;
    }
    auto gemm_v_offsets = make_tensor(
        gemm_v_indices.data(),
        make_layout(make_shape(size<0>(tArV.shape()), size<1>(tArV.shape()), size<2>(tArV.shape())),
                    make_stride(Int<0>{}, Int<1>{}, Int<0>{})));
    
    using MmaRowP = decltype(size<1>(tArP.shape()));
    auto gemm_p_indices = make_tensor<uint8_t>(make_shape(MmaRowP{}));
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < MmaRowP::value; ++m) {
      gemm_p_indices(m) = sizeof_bits_v<ElementQ> < 8 ? (m / 2) * 32 + (m % 2) * 8 : m * 8;
    }
    auto gemm_p_offsets = make_tensor(
        gemm_p_indices.data(),
        make_layout(make_shape(size<0>(tArP.shape()), size<1>(tArP.shape()), size<2>(tArP.shape())),
                    make_stride(Int<0>{}, Int<1>{}, Int<0>{})));

    fill(fragment_scaleP, ElementScaleV(1));
    using MmaPV_K = decltype(size<2>(tArV.shape()));
    auto gemm_pk_offsets = make_tensor<uint16_t>(Layout<Shape<_1, _1, MmaPV_K>, Stride<_0, _0, _1>>{});
    auto gemm_vk_offsets = make_tensor<uint16_t>(Layout<Shape<_1, _1, MmaPV_K>, Stride<_0, _0, _1>>{});
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < MmaPV_K::value; ++k) {
      gemm_pk_offsets(k) = static_cast<uint16_t>(k * size(typename GmemTiledCopyScaleP::BlockShape{}) * (SG_P / 16));
      gemm_vk_offsets(k) = static_cast<uint16_t>(k * size(typename GmemTiledCopyScaleV::BlockShape{}) * (SG_V / 16));
    }
    // ------
    // Kernel
    // ------

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    if (blk_k0 == 0) {
      for (int D = 0; D < size<3>(pQgQ); D++) {
        prefetch(prefetch_q, pQgQ(_,_,_,D));
      }

      for (int D = 0; D < size<4>(pKgK); D++) {
        CUTLASS_PRAGMA_UNROLL
        for (int K = 0; K < Stages; K++) {
          prefetch(prefetch_k, pKgK(_,_,_,K,D));
        }
      }

      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
    }

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    /* Main loop, blocked in k. */
    for (int K = blk_k0; K < blk_k1; K++) {
      /* Split barrier to keep threads together */
      barrier_arrive(ScopeWorkgroup);

      /* GEMM 1: S = K * Q */
      clear(tSrS);    /* TODO: fuse w/ initial gemm call */
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_,_,_,D),   tQrQ);
        copy(copy_k, tKgK(_,_,_,K,D), tKrK);
        if constexpr (FP4Input) {
          copy(tQrQ, tSrQ);
          copy(tKrK, tSrK);
        } else {
          reorder(tQrQ, tSrQ);
          reorder(tKrK, tSrK);
        }
        if constexpr (UseScale) {
          const int q_coord = get<0>(blk_qv) * BLK_Q + ((thr_id / intel::sg_size) / ATOM_K)  * SG_Q;
          const int k_coord = K * BLK_K + ((thr_id % intel::sg_size) / ATOM_K)  * SG_K;
          auto copy_iter_sQ = make_scale_copy_iterator<scaleQ_traits_size, scaleQ_traits_num, GmemTiledCopyScaleQ>(q_coord, l_coord, size<4>(tKgK), SG_QK_D, MMA_QK_D);
          auto copy_iter_sK = make_scale_copy_iterator<scaleK_traits_size, scaleK_traits_num, GmemTiledCopyScaleK>(k_coord, l_coord, size<4>(tKgK), SG_QK_D, MMA_QK_D);
          copy(tiled_copy_scaleQ, copy_iter_sQ(_, _, _, D), fragment_scaleQ);
          copy(tiled_copy_scaleK, copy_iter_sK(_, _, _, D), fragment_scaleK);
          Tensor scaleQ = recast<intel::vector_t<ElementScaleQ, scaleQSize::value * 2>>(make_tensor(fragment_scaleQ.data(), Shape<scaleQSize>{}));
          Tensor scaleK = recast<intel::vector_t<ElementScaleK, scaleKSize::value * 2>>(make_tensor(fragment_scaleK.data(), Shape<scaleKSize>{}));

          auto scaleQ_view = make_tensor(scaleQ.data(), scaleQ_layout);
          auto scaleK_view = make_tensor(scaleK.data(), scaleK_layout);
          auto zipped_q = make_zip_tensor(tSrQ, scaleQ_view, gemm_qm_offsets, gemm_qk_offsets);
          auto zipped_k = make_zip_tensor(tSrK, scaleK_view, gemm_kn_offsets, gemm_kk_offsets);
          cute::gemm(mma_qk, zipped_q, zipped_k, tSrS);
        } else {
          if constexpr (F8kvF16mma) {
            for (int i = 0; i < tSrK.size(); i++)
              tSrK(i) = static_cast<typename TiledMMAQK::ValTypeB>(scale_k * static_cast<float>(tSrK(i)));
          }
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        }
      }

      /* V prefetch for GEMM 2 */
      prefetch(prefetch_v, pVgV(_,_,_,K));

      /* Causal masking */
      if constexpr (CausalMask) {
        if (K == blk_k1 - 1) {
          // Need to get global col and row indices to mask the elements
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            if (col_idx - full_tile_offset > row_idx - discard_seq_coord) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }
      }
      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSRow k_rem_mask;
        int k = get<0>(tKgK(0,0,0,K,0)) + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) = (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }

      /* Apply softmax and scaling */
      softmax(K == 0, tSrS, tA_max, tA_sum, tArA);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV(_,_,_,VV,K), tVrV);
        reorder(tVrV, tArV);
        if constexpr (UseScale && !FP4Input) {
          const int v_coord = get<1>(blk_qv) * VTiles * BLK_V + VV * BLK_V + ((thr_id % intel::sg_size) / ATOM_V)  * SG_V;
          auto copy_iter_sV = make_scale_copy_iterator<scaleV_traits_size, scaleV_traits_num, GmemTiledCopyScaleV>(v_coord, l_coord, blk_k1, SG_PV_D, MMA_PV_D);
          copy(tiled_copy_scaleV, copy_iter_sV(_, _, _, K), fragment_scaleV);
          Tensor scaleV = recast<intel::vector_t<ElementScaleV, scaleVSize::value * 2>>(make_tensor(fragment_scaleV.data(), Shape<scaleVSize>{}));
          Tensor scaleP = recast<intel::vector_t<ElementScaleV, scalePSize::value * 2>>(make_tensor(fragment_scaleP.data(), Shape<scalePSize>{}));
          auto scaleV_view = make_tensor(scaleV.data(), scaleV_layout);
          auto scaleP_view = make_tensor(scaleP.data(), scaleP_layout);
          auto zipped_v = make_zip_tensor(tArV, scaleV_view, gemm_v_offsets, gemm_vk_offsets);
          auto zipped_p = make_zip_tensor(tArP, scaleP_view, gemm_p_offsets, gemm_pk_offsets);
          cute::gemm(mma_pv, zipped_p, zipped_v, tArA(_,_,_,VV));
        } else {
          if constexpr (F8kvF16mma) {
            for (int i = 0; i < tArV.size(); i++)
              tArV(i) = static_cast<typename TiledMMAQK::ValTypeB>(scale_v * static_cast<float>(tArV(i)));
          }
          cute::gemm(mma_pv, tArP, tArV, tArA(_,_,_,VV));
        }
      }

      /* K prefetch */
      for (int D = 0; D < size<4>(pKgK); D++) {
        prefetch(prefetch_k, pKgK(_,_,_,K+Stages,D));
      }

      barrier_wait(ScopeWorkgroup);
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  void
  softmax(bool       first_block, // First softmax block?
          FragS    & tS,          // Softmax src/dst block
          FragSRow & tS_max,      // Softmax row-wise max accumulator
          FragSRow & tS_sum,      // Softmax row-wise sum accumulator
          FragA    & tA) {        // O accumulator (for rescaling)

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    /* Update (scaled) maxima */
    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), params.scale * tS_bmax(i));
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums and O accumulator */
    if (!first_block) {
      FragSRow rescale;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::native::exp2(tS_prev_max(i) - tS_max(i));
        tS_sum(i) *= rescale(i);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA.size(); i++)
        tA(i) *= broadcast<0>(rescale, tA, i);
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);
  }
};


template <typename SGLayoutQK>
CUTLASS_HOST_DEVICE
constexpr auto
get_sg_layout_pv(SGLayoutQK const&)
{
  return make_layout(
    get<0>(SGLayoutQK{}),
    Layout<_1, _0>{},
    get<1>(SGLayoutQK{})
  );
}

// Get a P*V TiledMMA given K*Q tile size and SG configuration, for mainloops
//   not supporting S data interchange among subgroups (e.g. XeDefault).
template <typename MMAOp,
          typename WGTileQK,
          typename SGLayoutQK,
          typename TileV>
CUTLASS_HOST_DEVICE
constexpr auto
get_tiled_mma_pv(MMAOp const&, WGTileQK const& wg_tile_qk, SGLayoutQK const& sg_layout_qk, TileV const&) {
  using TileQ = decltype(get<0>(wg_tile_qk));
  using TileK = decltype(get<1>(wg_tile_qk));

  using WGTilePV = Shape<TileQ, TileV, TileK>;
  using SGLayoutPV = decltype(get_sg_layout_pv(sg_layout_qk));

  static_assert(size(SGLayoutPV{}) == size(SGLayoutQK{}),
                "Q*K cannot be parallelized in the head size dimension");

  return TiledMMAHelper<MMAOp, WGTilePV, SGLayoutPV>{};
}

} // namespace cutlass::fmha::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
