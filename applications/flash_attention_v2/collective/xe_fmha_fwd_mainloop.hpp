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
#include "cutlass/gemm/collective/xe_common_blockscaled_mxfp.hpp"
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

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class DispatchPolicy_,
          bool CausalMask_,
          bool UseScale_,
          bool F8kvF16mma_,
          bool CachedKV_,
          bool PagedKV_,
          class TiledMMAQK_,          // Tiling for Q*K GEMM
          class TiledMMAPV_,          // Tiling for P*V GEMM
          int VTiles_,                // # of tiles in V dimension
          class TensorQ_,             // Global Q/K/V tensors
          class TensorK_,
          class TensorV_,
          class TensorScaleQ_,
          class TensorScaleK_,
          class TensorScaleV_,
          class TensorK_cache_,
          class TensorV_cache_,
          class TiledCopyQ_ = void,   // Optional TiledCopy for loading Q
          class TiledCopyK_ = void,   // Optional TiledCopy for loading K
          class TiledCopyV_ = void,   // Optional TiledCopy for loading V
          class TiledCopyK_cache_ = void,
          class TiledCopyV_cache_ = void>   // Optional TiledCopy for loading V_cache
struct FMHAFwdMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages,
          bool CausalMask_, bool UseScale_, bool F8kvF16mma_, bool CachedKV_, bool PagedKV_,
          class TiledMMAQK_, class TiledMMAPV_, int VTiles_,
          class TensorQ_, class TensorK_, class TensorV_,
          class TensorScaleQ_, class TensorScaleK_, class TensorScaleV_,
          class TensorK_cache_, class TensorV_cache_,
          class TiledCopyQ_, class TiledCopyK_, class TiledCopyV_,
          class TiledCopyK_cache_, class TiledCopyV_cache_>
struct FMHAFwdMainloop<XeDefault<Stages>, CausalMask_, UseScale_, F8kvF16mma_,
                       CachedKV_, PagedKV_, TiledMMAQK_, TiledMMAPV_, VTiles_,
                       TensorQ_, TensorK_, TensorV_,
                       TensorScaleQ_, TensorScaleK_, TensorScaleV_,
                       TensorK_cache_, TensorV_cache_,
                       TiledCopyQ_, TiledCopyK_, TiledCopyV_,
                       TiledCopyK_cache_, TiledCopyV_cache_> {
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
  using TensorK_cache = TensorK_cache_;
  using TensorV_cache = TensorV_cache_;
  using TensorK_cache2D = decltype(TensorK_cache_{}(append<rank_v<TensorK_cache_>>(make_coord(_,_),0)));
  using TensorV_cache2D = decltype(TensorV_cache_{}(append<rank_v<TensorV_cache_>>(make_coord(_,_),0)));
  using TiledCopyK_cache = conditional_t<is_void_v<TiledCopyK_cache_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK_cache2D{})), TiledCopyK_cache_>;
  using TiledCopyV_cache = conditional_t<is_void_v<TiledCopyV_cache_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV_cache2D{})), TiledCopyV_cache_>;

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
  static constexpr bool CachedKV = CachedKV_;
  static constexpr bool PagedKV = PagedKV_;

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

  using DefScaleType = cutlass::float_ue8m0_t;
  using ElementScaleP = cute::conditional_t<UseScale, ElementScaleV, DefScaleType>;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    int const* ptr_page_table = nullptr;
    int page_size = 0;
    int const* num_pages_per_seq = nullptr;
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
    return Params{val, args.ptr_page_table, args.page_size, args.num_pages_per_seq};
  }

  CUTLASS_HOST_DEVICE static
  bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_DEVICE
  int get_physical_k_tile(int K, int l_coord, int seq_len_kv_cache) {
    int next_page_logical_idx = K * get<1>(TileShapeQK{}) / params.page_size;
    // get<1>(TileShapeQK{}) usually smaller than page_size.
    // assuming page_size is multiple of get<1>(TileShapeQK{})
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int batch_offset = params.num_pages_per_seq ? params.num_pages_per_seq[l_coord] : l_coord * (seq_len_kv_cache / params.page_size);

    return params.ptr_page_table[
          batch_offset +                  
          next_page_logical_idx] * tiles_per_page +            
          K % tiles_per_page; 
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
             int              total_blk, // Total # of K blocks
             int              thr_id,
             int              seq_len,
             int              seq_len_kv_cache,
             int              l_coord,
             int              full_tile_offset,
             int              discard_seq_coord,
             TensorK_cache2D const& K_cache_2D = TensorK_cache2D{},
             TensorV_cache2D const& V_cache_2D = TensorV_cache2D{},
             float            scale_k = 1.0f,
             float            scale_v = 1.0f,
             TensorScaleQ2D    const& scaleQ = TensorScaleQ2D{},
             TensorScaleK2D    const& scaleK = TensorScaleK2D{},
             TensorScaleV2D    const& scaleV = TensorScaleV2D{}) {
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
    Tensor cK_cache = make_identity_tensor(K_cache_2D.shape()); // (k,d)
    Tensor cV_cache = make_identity_tensor(V_cache_2D.shape()); // (v,k)
    Tensor cP = make_identity_tensor(take<0,2>(TileShapeQK{})); // (q,k)

    /* Partition global tensors into workgroup tiles */
    Tensor gQ       = local_tile(cQ, TileShapeQK{}, append(blk_qv,_),             Step<_1,X,_1>{});   // (q,d,D)
    Tensor gK       = local_tile(cK, TileShapeQK{}, make_coord(_,_,_),            Step<X,_1,_1>{});   // (k,d,K,D)
    Tensor gV       = local_tile(cV, tile_shape_v,  make_coord(get<1>(blk_qv),_));                    // (v,k,K)
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_,_,0),            Step<X,_1,_1>{});   // (v,k,VV,K)

    Tensor gK_cache       = local_tile(cK_cache, TileShapeQK{}, make_coord(_,_,_),            Step<X,_1,_1>{});   // (k,d,K,D)
    Tensor gV_cache       = local_tile(cV_cache, tile_shape_v,  make_coord(get<1>(blk_qv),_));                    // (v,k,K)
    Tensor gV_cache_split = local_tile(gV_cache, TileShapePV{}, make_coord(_,_,0),            Step<X,_1,_1>{});   // (v,k,VV,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};
    TiledCopyK_cache copy_k_cache{K_cache_2D};
    TiledCopyV_cache copy_v_cache{V_cache_2D};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_copy_k_cache = copy_k_cache.get_slice(thr_id);
    auto thr_copy_v_cache = copy_v_cache.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);                // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);                // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);          // (atom_val,v',k',VV,K)
    auto tKgK_cache = thr_copy_k_cache.partition_S(gK_cache);
    auto tVgV_cache = thr_copy_v_cache.partition_S(gV_cache_split);

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
    auto prefetch_v = make_block_2d_prefetch(copy_v);
    auto prefetch_k_cache = make_block_2d_prefetch(copy_k_cache);
    auto prefetch_v_cache = make_block_2d_prefetch(copy_v_cache);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_split);
    auto pKgK_cache = prefetch_k_cache.get_slice(thr_id).partition_S(gK_cache);
    auto pVgV_cache = prefetch_v_cache.get_slice(thr_id).partition_S(gV_cache_split);
    const auto subgroup_id = thr_id / intel::sg_size;

    using ScaleCopyQK = void;
    using ScaleCopyPV = void;
    auto scale_context_qk = [&]() {
      if constexpr (UseScale) {
        auto scale_copy_Q = gemm::collective::make_scaled_copy<ScaleCopyQK, ElementScaleQ, SG_Q, SG_QK_D, GROUP_K>(
                                                      scaleQ, 0, 0, size<4>(tKgK));
        auto scale_copy_K = gemm::collective::make_scaled_copy<ScaleCopyQK, ElementScaleK, SG_K, SG_QK_D, GROUP_K>(
                                                      scaleK, 0, 0, size<4>(tKgK));
        auto scale_prefetch_Q = gemm::collective::make_scaled_prefetch<decltype(get<0>(scale_copy_Q)), SG_Q, SG_QK_D, GROUP_K>(
                                                      get<0>(scale_copy_Q), 0, l_coord, size<4>(tKgK));
        auto scale_prefetch_K = gemm::collective::make_scaled_prefetch<decltype(get<0>(scale_copy_K)), SG_K, SG_QK_D, GROUP_K>(
                                                      get<0>(scale_copy_K), 0, l_coord, size<4>(tKgK));
        auto scale_offsets_qk = gemm::collective::make_scaled_offsets<
                                                      decltype(size<1>(tSrQ.shape()))::value,
                                                      decltype(size<1>(tSrK.shape()))::value,
                                                      decltype(size<2>(tSrK.shape()))::value,
                                                      MMA_QK_D, GROUP_K,
                                                      typename cute::remove_cvref_t<decltype(get<0>(scale_copy_Q))>::BlockShape,
                                                      typename cute::remove_cvref_t<decltype(get<0>(scale_copy_K))>::BlockShape>();
        return cute::make_tuple(scale_copy_Q, scale_copy_K, scale_prefetch_Q, scale_prefetch_K, scale_offsets_qk);
      } else {
        return cute::tuple<>{};
      }
    }();

    auto scale_context_pv = [&]() {
      if constexpr (UseScale) {
        auto scale_copy_P = gemm::collective::make_scaled_copy<ScaleCopyPV, ElementScaleP, SG_P, SG_PV_D, GROUP_K>(scaleV);
        auto scale_copy_V = gemm::collective::make_scaled_copy<ScaleCopyPV, ElementScaleV, SG_V, SG_PV_D, GROUP_K>(
                                                      scaleV, 0, 0, blk_k1);
        auto scale_prefetch_V = gemm::collective::make_scaled_prefetch<decltype(get<0>(scale_copy_V)), SG_V, SG_PV_D, GROUP_K>(
                                                      get<0>(scale_copy_V), 0, l_coord, blk_k1);
        auto scale_offsets_pv = gemm::collective::make_scaled_offsets<
                                                      decltype(size<1>(tArP.shape()))::value,
                                                      decltype(size<1>(tArV.shape()))::value,
                                                      decltype(size<2>(tArV.shape()))::value,
                                                      MMA_PV_D, GROUP_K,
                                                      typename cute::remove_cvref_t<decltype(get<0>(scale_copy_P))>::BlockShape,
                                                      typename cute::remove_cvref_t<decltype(get<0>(scale_copy_V))>::BlockShape>();
        return cute::make_tuple(scale_copy_P, scale_copy_V, scale_prefetch_V, scale_offsets_pv);
      } else {
        return cute::tuple<>{};
      }
    }();

    // ------
    // Kernel
    // ------

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    int kblocks_cache = ceil_div(seq_len_kv_cache, get<1>(TileShapeQK{}));
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_,_,_,D));
    }
    for (int D = 0; D < size<4>(pKgK); D++) {
      CUTLASS_PRAGMA_UNROLL
      for (int K = 0; K < Stages; K++) {
        if (K < kblocks_cache) {
          if constexpr (PagedKV) {
            int physical_K_tile = get_physical_k_tile(K, l_coord, seq_len_kv_cache);
            prefetch(prefetch_k_cache, pKgK_cache(_,_,_,physical_K_tile,D));
          } else {
            prefetch(prefetch_k_cache, pKgK_cache(_,_,_,K,D));
          }
        } else {
          prefetch(prefetch_k, pKgK(_,_,_,K - kblocks_cache,D));
        }
      }
    }
    if constexpr (UseScale) {
      const int q_coord = get<0>(blk_qv) * BLK_Q + (subgroup_id / ATOM_K)  * SG_Q;
      auto& tiled_prefetch_scaleQ = get<0>(get<2>(scale_context_qk));
      auto  prefetch_iter_scaleQ = get<1>(get<2>(scale_context_qk));
      auto& tiled_prefetch_scaleK = get<0>(get<3>(scale_context_qk));
      auto  prefetch_iter_scaleK = get<1>(get<3>(scale_context_qk));
      prefetch_iter_scaleQ.data().coord_ = {q_coord, 0, l_coord};
      for (int D = 0; D < size<3>(pQgQ); D++) {
        prefetch(tiled_prefetch_scaleQ, prefetch_iter_scaleQ(_, _, _, D));
      }

      for (int K = 0; K < Stages; K++) {
        const int k_coord = K * BLK_K + (subgroup_id % ATOM_K)  * SG_K;
        prefetch_iter_scaleK.data().coord_ = {k_coord, 0, l_coord};
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(tiled_prefetch_scaleK, prefetch_iter_scaleK(_, _, _, D));
        }
      }
    }
    if (blk_k0 == 0) {
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
    }

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);
    
    /* Main loop body */
    auto mainloop_body = [&](auto cached_k, int K,
                             auto& copy_k_cur, auto& copy_v_cur,
                             auto& prefetch_v_cur, auto& tKgK_cur,
                             auto& tVgV_cur, auto& pVgV_cur) {
#if not defined(CUTLASS_TEST_FOR_CRI)
      /* Split barrier to keep threads together */
      barrier_arrive(ScopeWorkgroup);
#endif
      constexpr bool is_cache = decltype(cached_k)::value;

      int k_idx;
      if constexpr (is_cache) {
        k_idx = K;
        if constexpr (PagedKV) {
          k_idx = get_physical_k_tile(K, l_coord, seq_len_kv_cache);
        }
      } else {
        k_idx = K - kblocks_cache;
      }

      /* GEMM 1: S = K * Q */
      clear(tSrS);
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_,_,_,D), tQrQ);
        copy(copy_k_cur, tKgK_cur(_,_,_,k_idx,D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);

        if constexpr (UseScale) {
          if constexpr (sizeof_bits_v<ElementQ> <= 8) {
            static_assert(SG_QK_D >= 32, "Intel Xe blockscaled MMA requires SG_QK_D to be at least 32.");
            static_assert(SG_PV_D >= 32, "Intel Xe blockscaled MMA requires SG_PV_D to be at least 32.");
          }

          static_assert(SG_Q == SG_P && SG_K == SG_PV_D && BLK_P == BLK_Q);

          const int q_coord = get<0>(blk_qv) * BLK_Q + (subgroup_id / ATOM_K)  * SG_Q;
          const int k_coord = K * BLK_K + (subgroup_id % ATOM_K)  * SG_K;

          auto& tiled_copy_scaleQ = get<0>(get<0>(scale_context_qk));
          auto  copy_iter_scaleQ = get<1>(get<0>(scale_context_qk));
          auto  fragment_scaleQ = get<2>(get<0>(scale_context_qk));
          auto& tiled_copy_scaleK = get<0>(get<1>(scale_context_qk));
          auto  copy_iter_scaleK = get<1>(get<1>(scale_context_qk));
          auto  fragment_scaleK = get<2>(get<1>(scale_context_qk));
          auto [gemm_qm_offsets, gemm_kn_offsets, gemm_qk_offsets, gemm_kk_offsets] = get<4>(scale_context_qk);

          using scaleQSize = decltype(size(fragment_scaleQ));
          using scaleKSize = decltype(size(fragment_scaleK));

          Tensor scaleQ_view = make_tensor(recast<intel::vector_t<ElementScaleQ, scaleQSize::value>>(fragment_scaleQ).data(),
                                           make_layout(Shape<_1, decltype(size<1>(tSrQ.shape())), _1>{}, Stride<_1, _0, _0>{}));
          Tensor scaleK_view = make_tensor(recast<intel::vector_t<ElementScaleK, scaleKSize::value>>(fragment_scaleK).data(),
                                           make_layout(Shape<_1, decltype(size<1>(tSrK.shape())), _1>{}, Stride<_1, _0, _0>{}));

          auto zipped_q = make_zip_tensor(tSrQ, scaleQ_view, gemm_qm_offsets, gemm_qk_offsets);
          auto zipped_k = make_zip_tensor(tSrK, scaleK_view, gemm_kn_offsets, gemm_kk_offsets);

          copy_iter_scaleQ.data().coord_ = {q_coord, 0, l_coord};
          copy_iter_scaleK.data().coord_ = {k_coord, 0, l_coord};

          copy(tiled_copy_scaleQ, copy_iter_scaleQ(_, _, _, D), fragment_scaleQ);
          copy(tiled_copy_scaleK, copy_iter_scaleK(_, _, _, D), fragment_scaleK);

          cute::gemm(mma_qk, zipped_q, zipped_k, tSrS);
        } else {
          if constexpr (F8kvF16mma) {
            dequantize(tSrK, scale_k);
          }
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        }
      }

      /* V prefetch for GEMM 2 */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        prefetch(prefetch_v_cur, pVgV_cur(_,_,_,VV,k_idx));
      }
      // Prefetch V scale
      if constexpr (UseScale) {
        auto& tiled_prefetch_scaleV = get<0>(get<2>(scale_context_pv));
        auto  prefetch_iter_scaleV = get<1>(get<2>(scale_context_pv));
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          const int v_coord = get<1>(blk_qv) * VTiles * BLK_V + VV * BLK_V + (subgroup_id % ATOM_V) * SG_V;
          prefetch_iter_scaleV.data().coord_ = {v_coord, 0, l_coord};
          prefetch(tiled_prefetch_scaleV, prefetch_iter_scaleV(_, _, _, K - kblocks_cache));
        }
      }
      /* Causal masking - only in non-cache mode */
      if constexpr (!is_cache && CausalMask) {
        if (K == total_blk - 1) {
          // Need to get global col and row indices to mask the elements
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            if (col_idx - seq_len_kv_cache - full_tile_offset > row_idx - discard_seq_coord) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }
      }
      /* k masking for remainder tiles */
      if constexpr (!is_cache) {
        if (check_remainder_k && K == total_blk - 1) {
          FragSRow k_rem_mask;
          int k_val = get<0>(tKgK_cur(0,0,0,k_idx,0)) + kblocks_cache * get<1>(TileShapeQK{});
          int k = k_val + get_sub_group().get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
            k_rem_mask(i) = (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); i++) {
            tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
          }
        }
      }

      /* Apply softmax and scaling (tA rescaling fused into GEMM2 VTile loop) */
      auto rescale = softmax(K == blk_k0, tSrS, tA_max, tA_sum);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension.
        tArA rescaling is fused to per-VTile */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v_cur, tVgV_cur(_,_,_,VV,k_idx), tVrV);
        reorder(tVrV, tArV);
        if (K != blk_k0) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArA.size() / VTiles; i++)
            tArA(_,_,_,VV)(i) *= broadcast<0>(rescale, tArA, i);
        }
        if constexpr (UseScale && !FP4Input) {
          const int v_coord = get<1>(blk_qv) * VTiles * BLK_V + VV * BLK_V + (subgroup_id % ATOM_V) * SG_V;
          auto& tiled_copy_scaleP = get<0>(get<0>(scale_context_pv));
          // P is dummy scale, just the same as V
          auto  fragment_scaleP = get<2>(get<0>(scale_context_pv));
          auto& tiled_copy_scaleV = get<0>(get<1>(scale_context_pv));
          auto  copy_iter_scaleV = get<1>(get<1>(scale_context_pv));
          auto  fragment_scaleV = get<2>(get<1>(scale_context_pv));
          auto [gemm_p_offsets, gemm_v_offsets, gemm_pk_offsets, gemm_vk_offsets] = get<3>(scale_context_pv);

          using scalePSize = decltype(size(fragment_scaleP));
          using scaleVSize = decltype(size(fragment_scaleV));

          Tensor scaleP_view = make_tensor(recast<intel::vector_t<ElementScaleV, scalePSize::value>>(fragment_scaleP).data(),
                                           make_layout(Shape<_1, decltype(size<1>(tArP.shape())), _1>{}, Stride<_1, _0, _0>{}));
          Tensor scaleV_view = make_tensor(recast<intel::vector_t<ElementScaleV, scaleVSize::value>>(fragment_scaleV).data(),
                                           make_layout(Shape<_1, decltype(size<1>(tArV.shape())), _1>{}, Stride<_1, _0, _0>{}));

          auto zipped_p = make_zip_tensor(tArP, scaleP_view, gemm_p_offsets, gemm_pk_offsets);
          auto zipped_v = make_zip_tensor(tArV, scaleV_view, gemm_v_offsets, gemm_vk_offsets);

          copy_iter_scaleV.data().coord_ = {v_coord, 0, l_coord};

          fill(fragment_scaleP, ElementScaleV(1));
          copy(tiled_copy_scaleV, copy_iter_scaleV(_, _, _, K), fragment_scaleV);

          cute::gemm(mma_pv, zipped_p, zipped_v, tArA(_,_,_,VV));
        } else {
          if constexpr (F8kvF16mma) {
            dequantize(tArV, scale_v);
          }
          cute::gemm(mma_pv, tArP, tArV, tArA(_,_,_,VV));
        }
      }

      /* K prefetch */
      int K_next = K + Stages;
      for (int D = 0; D < size<4>(pKgK); D++) {
        if constexpr (is_cache) {
          bool is_cache_next = K_next < kblocks_cache;
          int physical_K_next = K_next;
          if constexpr (PagedKV) {
            if (is_cache_next) {
              physical_K_next = get_physical_k_tile(K_next, l_coord, seq_len_kv_cache);
            }
          }
          if (is_cache_next) {
            prefetch(prefetch_k_cache, pKgK_cache(_,_,_,physical_K_next,D));
          } else {
            prefetch(prefetch_k, pKgK(_,_,_,K_next-kblocks_cache,D));
          }
        } else {
          prefetch(prefetch_k, pKgK(_,_,_,K_next-kblocks_cache,D));
        }
      }
      // Prefetch K scale
      if constexpr (UseScale) {
        auto& tiled_prefetch_scaleK = get<0>(get<3>(scale_context_qk));
        auto  prefetch_iter_scaleK = get<1>(get<3>(scale_context_qk));
        const int k_coord_next = (K_next-kblocks_cache) * BLK_K + (subgroup_id % ATOM_K) * SG_K;
        prefetch_iter_scaleK.data().coord_ = {k_coord_next, 0, l_coord};
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(tiled_prefetch_scaleK, prefetch_iter_scaleK(_, _, _, D));
        }
      }
#if not defined(CUTLASS_TEST_FOR_CRI)
      barrier_wait(ScopeWorkgroup);
#endif
    };

    /* Main loop, blocked in k. */
    if constexpr (CachedKV) {
      for (int K = blk_k0; K < kblocks_cache; K++) {
        mainloop_body(std::bool_constant<true>{}, K,
                      copy_k_cache, copy_v_cache,
                      prefetch_v_cache, tKgK_cache,
                      tVgV_cache, pVgV_cache);
      }
    }

    for (int K = (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache); K < blk_k1; K++) {
      mainloop_body(std::bool_constant<false>{}, K,
                    copy_k, copy_v,
                    prefetch_v, tKgK,
                    tVgV, pVgV);
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  FragSRow
  softmax(bool       first_block, // First softmax block?
          FragS    & tS,          // Softmax src/dst block
          FragSRow & tS_max,      // Softmax row-wise max accumulator
          FragSRow & tS_sum) {    // Softmax row-wise sum accumulator
    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums */
    if (!first_block) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) {
        tS_sum(i) *= rescale(i);
      }
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);

    return rescale;
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
