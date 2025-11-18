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
template<class datatype, size_t width, class stride>
struct scale_copy_traits<datatype, 1, width, stride,
          std::enable_if_t<sizeof_bits_v<datatype> == 8>> {
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
          bool CausalMask_, bool UseScale_,
          class TiledMMAQK_, class TiledMMAPV_, int VTiles_,
          class TensorQ_, class TensorK_, class TensorV_,
          class TensorScaleQ_, class TensorScaleK_, class TensorScaleV_,
          class TiledCopyQ_, class TiledCopyK_, class TiledCopyV_>
struct FMHAFwdMainloop<XeDefault<Stages>, CausalMask_, UseScale_,
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

  using TensorScaleQ = TensorScaleQ_;
  using TensorScaleK = TensorScaleK_;
  using TensorScaleV = TensorScaleV_;
  using TensorScaleQ2D = decltype(TensorScaleQ_{}(append<rank_v<TensorScaleQ_>>(make_coord(_,_),0)));
  using TensorScaleK2D = decltype(TensorScaleK_{}(append<rank_v<TensorScaleK_>>(make_coord(_,_),0)));
  using TensorScaleV2D = decltype(TensorScaleV_{}(append<rank_v<TensorScaleV_>>(make_coord(_,_),0)));
  using ElementScaleQ = typename TensorScaleQ::element_type;
  using ElementScaleK = typename TensorScaleK::element_type;
  using StrideScaleQ = decltype(stride(TensorScaleQ{}));
  using StrideScaleK = decltype(stride(TensorScaleK{}));

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

  using MmaAtomShape = typename TiledMMAQK::AtomShape_MNK;

  static constexpr int BLK_M = get<0>(TileShapeQK{});
  static constexpr int BLK_N = get<1>(TileShapeQK{});
  static constexpr int BLK_K = get<2>(TileShapeQK{});

  static constexpr int ATOM_M = get<1>(typename TiledMMAQK::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_N = get<2>(typename TiledMMAQK::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_K = get<3>(typename TiledMMAQK::ThrLayoutVMNK{}.shape());

  static constexpr int SG_M = ceil_div(BLK_M, ATOM_M);
  static constexpr int SG_N = ceil_div(BLK_N, ATOM_N);
  static constexpr int SG_K = ceil_div(BLK_K, ATOM_K);

  static constexpr auto GROUP_K = 32;

  static_assert(SG_K >= 32, "Intel Xe blockscaled MMA requires SG_K to be at least 32.");

  using CopyThreadShape = Shape<_1, Int<intel::sg_size>>;
  using CopyThreadShapeRev = decltype(cute::reverse(CopyThreadShape{}));


  using DefScaleType = cutlass::float_ue8m0_t;
  using NonVoidElementScaleQ = cute::conditional_t<UseScale, ElementScaleQ, DefScaleType>;
  using NonVoidElementScaleK = cute::conditional_t<UseScale, ElementScaleK, DefScaleType>;
  using GmemTiledCopyScaleQ = typename scale_copy_traits<NonVoidElementScaleQ, SG_K / GROUP_K, SG_M>::type;
  using GmemTiledCopyScaleK = typename scale_copy_traits<NonVoidElementScaleK, SG_K / GROUP_K, SG_N>::type;
  
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
  
  // If IsATransformed, we need modes M_atom, and M_iter from fragment_A layout else we need mode N_iter from fragment_B layout.
  static constexpr auto scaleQ_traits_size = decltype(size(typename GmemTiledCopyScaleQ::BlockShape{}))::value / intel::sg_size;
  static constexpr auto scaleQ_traits_num = SG_M / size<1>(typename GmemTiledCopyScaleQ::BlockShape{});
  using FragScaleQLayout = Layout<Shape<Int<scaleQ_traits_size>, Int<scaleQ_traits_num>, _1>>;

  static constexpr int scaleK_traits_size = decltype(size(typename GmemTiledCopyScaleK::BlockShape{}))::value / intel::sg_size;
  static constexpr int scaleK_traits_num = SG_N / size<1>(typename GmemTiledCopyScaleK::BlockShape{});
  using FragScaleKLayout = Layout<Shape<Int<scaleK_traits_size>, Int<scaleK_traits_num>, _1>>;
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
  make_scale_copy_iterator(int coord, int l_coord, int k_tile_count) {
      return make_tensor(make_inttuple_iter(make_coord(coord, 0, l_coord)),
                         make_layout(make_shape(Int<scale_traits_size>{}, Int<scale_traits_num>{}, _1{}, k_tile_count),
                                     make_stride(E<0>{} * _16{}, E<0>{} * size<1>(typename SelectedGmemTiledCopyScale::BlockShape{}), _0{}, E<1>{} * (SG_K / GROUP_K))));
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
             Copy_ScaleQ const& tiled_copy_scaleQ = Copy_ScaleQ{},
             Copy_ScaleK const& tiled_copy_scaleK = Copy_ScaleK{}) {
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

    const int m_coord = get<0>(blk_qv) * BLK_M + ((thr_id / intel::sg_size) / ATOM_N)  * SG_M;
    const int n_coord = get<1>(blk_qv) * BLK_N + ((thr_id % intel::sg_size) / ATOM_N)  * SG_N;

    auto copy_iter_sQ = make_scale_copy_iterator<scaleQ_traits_size, scaleQ_traits_num, GmemTiledCopyScaleQ>(m_coord, l_coord, blk_k1);
    auto copy_iter_sK = make_scale_copy_iterator<scaleK_traits_size, scaleK_traits_num, GmemTiledCopyScaleK>(n_coord, l_coord, blk_k1);
    
    auto scaleQ_layout = make_layout(
        make_shape(size<0>(tSrQ.shape()), size<1>(tSrQ.shape()), size<2>(tSrQ.shape())),
        make_stride(Int<1>{}, Int<0>{}, Int<0>{}));

    auto scaleK_layout = make_layout(
        make_shape(size<0>(tSrK.shape()), size<1>(tSrK.shape()), size<2>(tSrK.shape())),
        make_stride(Int<1>{}, Int<0>{}, Int<0>{}));

    using scaleQSize = decltype(size(fragment_scaleQ));
    using scaleKSize = decltype(size(fragment_scaleK));

    using MmaRowQ = decltype(size<1>(tSrQ.shape()));
    auto gemm_q_indices = make_tensor<uint8_t>(make_shape(MmaRowQ{}));
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < MmaRowQ::value; ++m) {
      gemm_q_indices(m) = static_cast<uint8_t>(m);
    }
    auto gemm_q_offsets = make_tensor(
        gemm_q_indices.data(),
        make_layout(make_shape(size<0>(tSrQ.shape()), size<1>(tSrQ.shape()), size<2>(tSrQ.shape())),
                    make_stride(Int<0>{}, Int<1>{}, Int<0>{})));
    
    using MmaColK = decltype(size<1>(tSrK.shape()));
    auto gemm_k_indices = make_tensor<uint8_t>(make_shape(MmaColK{}));
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < MmaColK::value; ++n) {
      gemm_k_indices(n) = static_cast<uint8_t>(n);
    }
    auto gemm_k_offsets = make_tensor(
        gemm_k_indices.data(),
        make_layout(make_shape(size<0>(tSrK.shape()), size<1>(tSrK.shape()), size<2>(tSrK.shape())),
                    make_stride(Int<0>{}, Int<1>{}, Int<0>{})));
#if 0
    Tensor cScaleV = make_identity_tensor(ScaleV_2D.shape());   // (v,k)
    Tensor gScaleV   = local_tile(cScaleV, tile_shape_v,  make_coord(get<1>(blk_qv),_));                 // (v,k,K)
    Tensor gScaleV_split = local_tile(gScaleV, TileShapePV{},  make_coord(_,_,0),  Step<X,_1,_1>{});     // (v,k,K)
    TiledCopyScaleV copy_scale_v{ScaleV_2D};
    auto thr_copy_scale_v = copy_scale_v.get_slice(thr_id);
    auto tVgScaleV = thr_copy_scale_v.partition_S(gScaleV_split); // (atom_val,v',k',K)
    auto tArVScaleV = thr_copy_scale_v.partition_sg_fragment_D(gScaleV_split(_,_,0,0));
    auto scale_v_layout = make_layout(
        make_shape(size<0>(tArV.shape()), size<1>(tArV.shape()), size<2>(tArV.shape())),
        make_stride(Int<1>{}, Int<0>{}, Int<0>{}));

    auto scale_p_layout = make_layout(
        make_shape(size<0>(tArP.shape()), size<1>(tArP.shape()), size<2>(tArP.shape())),
        make_stride(Int<1>{}, Int<0>{}, Int<0>{}));
    constexpr int ScaleVElements = cosize_v<typename decltype(tArVScaleV)::layout_type>;
    auto prefetch_scale_v = make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, ScaleV_2D);
    auto pVgScaleV = prefetch_scale_v.get_slice(thr_id).partition_S(gScaleV);
    auto scale_p_fragment = make_tensor<ElementScale>(scale_p_layout);
    fill(scale_p_fragment, ElementScale(1));
    constexpr int ScalePElements = cosize_v<typename decltype(scale_p_fragment)::layout_type>;
    auto scale_p_flat = make_tensor(scale_p_fragment.data(), make_shape(Int<ScalePElements>{}));
    auto scale_p_vec = recast<intel::vector_t<ElementScale, ScalePElements * 2>>(scale_p_flat);
    auto scale_p_view = make_tensor(scale_p_vec.data(), scale_p_layout);
    using MmaColV = decltype(size<1>(tArV.shape()));
    auto gemm_v_indices = make_tensor<uint8_t>(make_shape(MmaColV{}));
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < MmaColV::value; ++n) {
      gemm_v_indices(n) = static_cast<uint8_t>(n);
    }
    auto gemm_v_offsets = make_tensor(
        gemm_v_indices.data(),
        make_layout(make_shape(size<0>(tArV.shape()), size<1>(tArV.shape()), size<2>(tArV.shape())),
                    make_stride(Int<0>{}, Int<1>{}, Int<0>{})));
    
    using MmaRowP = decltype(size<1>(tArP.shape()));
    auto gemm_p_indices = make_tensor<uint8_t>(make_shape(MmaRowP{}));
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < MmaRowP::value; ++m) {
      gemm_p_indices(m) = static_cast<uint8_t>(m);
    }
    auto gemm_p_offsets = make_tensor(
        gemm_p_indices.data(),
        make_layout(make_shape(size<0>(tArP.shape()), size<1>(tArP.shape()), size<2>(tArP.shape())),
                    make_stride(Int<0>{}, Int<1>{}, Int<0>{})));
#endif
    
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
          copy(tiled_copy_scaleQ, copy_iter_sQ(_, _, _, K), fragment_scaleQ);
          copy(tiled_copy_scaleK, copy_iter_sK(_, _, _, K), fragment_scaleK);
          Tensor scaleQ = recast<intel::vector_t<ElementScaleQ, scaleQSize::value * 2>>(make_tensor(fragment_scaleQ.data(), Shape<scaleQSize>{}));
          Tensor scaleK = recast<intel::vector_t<ElementScaleK, scaleKSize::value * 2>>(make_tensor(fragment_scaleK.data(), Shape<scaleKSize>{}));

          auto scaleQ_view = make_tensor(scaleQ.data(), scaleQ_layout);
          auto scaleK_view = make_tensor(scaleK.data(), scaleK_layout);
          auto zipped_q = make_zip_tensor(tSrQ, scaleQ_view, gemm_q_offsets);
          auto zipped_k = make_zip_tensor(tSrK, scaleK_view, gemm_k_offsets);
          cute::gemm(mma_qk, zipped_q, zipped_k, tSrS);
        } else {
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
#if 0
        if constexpr (UseScale) {
          copy(copy_scale_v, tVgScaleV(_,_,_,VV,K), tArVScaleV);
          auto scale_v_flat = make_tensor(tArVScaleV.data(), make_shape(Int<ScaleVElements>{}));
          auto scale_v_vec = recast<intel::vector_t<ElementScale, ScaleVElements * 2>>(scale_v_flat);
          auto scale_v_view = make_tensor(scale_v_vec.data(), scale_v_layout);
          auto zipped_v = make_zip_tensor(tArV, scale_v_view, gemm_v_offsets);
          auto zipped_p = make_zip_tensor(tArP, scale_p_view, gemm_p_offsets);
          cute::gemm(mma_pv, zipped_p, zipped_v, tArA(_,_,_,VV));
        } else {
          cute::gemm(mma_pv, tArP, tArV, tArA(_,_,_,VV));
        }
#else
        cute::gemm(mma_pv, tArP, tArV, tArA(_,_,_,VV));
#endif
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
