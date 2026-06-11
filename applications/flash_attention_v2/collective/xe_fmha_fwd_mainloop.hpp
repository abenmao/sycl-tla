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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "fmha_fusion.hpp"

namespace cutlass::fmha {

template <int Stages> class XeDefault {};   // Default FMHA mainloop, P in registers.

};

namespace cutlass::fmha::collective {

using namespace cute;

// Rebind the element type of a subgroup fragment while preserving its TV layout.
// Used by the int8 FMHA path to derive a float "scores" / "output" fragment that
// shares the layout of the int32 DPAS accumulator fragment.
template <typename NewT, typename E, typename L, typename TV>
CUTE_HOST_DEVICE constexpr auto
sg_fragment_rebind(cute::SubgroupTensor<E, L, TV> const&) {
  return cute::make_subgroup_tensor(cute::make_tensor<NewT>(L{}), TV{});
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class DispatchPolicy_,
          bool CausalMask_,
          bool CachedKV_,
          bool PagedKV_,
          class TiledMMAQK_,          // Tiling for Q*K GEMM
          class TiledMMAPV_,          // Tiling for P*V GEMM
          int VTiles_,                // # of tiles in V dimension
          class TensorQ_,             // Global Q/K/V tensors
          class TensorK_,
          class TensorV_,
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
          bool CausalMask_, bool CachedKV_, bool PagedKV_,
          class TiledMMAQK_, class TiledMMAPV_, int VTiles_,
          class TensorQ_, class TensorK_, class TensorV_,
          class TensorK_cache_, class TensorV_cache_,
          class TiledCopyQ_, class TiledCopyK_, class TiledCopyV_,
          class TiledCopyK_cache_, class TiledCopyV_cache_>
struct FMHAFwdMainloop<XeDefault<Stages>, CausalMask_, CachedKV_, PagedKV_,
                       TiledMMAQK_, TiledMMAPV_, VTiles_,
                       TensorQ_, TensorK_, TensorV_,
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

  using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_,_),0)));
  using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_,_),0)));
  using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_,_),0)));

  using TiledCopyQ = conditional_t<is_void_v<TiledCopyQ_>, decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})), TiledCopyQ_>;
  using TiledCopyK = conditional_t<is_void_v<TiledCopyK_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})), TiledCopyK_>;
  using TiledCopyV = conditional_t<is_void_v<TiledCopyV_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})), TiledCopyV_>;
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

#ifdef IS_INT8
  // int8 FMHA: Q/K/V are int8 and the QK/PV DPAS accumulate to int32. Softmax and
  // the online O accumulation are still done in fp32, so the "scores" (S) and
  // output (A) fragments are float, derived by rebinding the int32 accumulator
  // fragment's element type while keeping its layout. This entire block is gated
  // behind IS_INT8 so the bf16/fp8 translation units see the pristine original
  // definitions below (zero codegen perturbation).
  static constexpr bool IsInt8 = cute::is_same_v<cute::remove_cv_t<typename TensorQ::value_type>, int8_t>;

  using FragSAcc = FragC<TiledMMAQK>;                             // QK DPAS accumulator (int32 for int8)
  using ElementSAcc = typename TiledMMAQK::ValTypeD;
  using ElementS = cute::conditional_t<IsInt8, float, ElementSAcc>;
  using FragS = cute::conditional_t<IsInt8, decltype(sg_fragment_rebind<float>(FragSAcc{})), FragSAcc>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));

  using SingleFragAAcc = FragC<TiledMMAPV>;                       // PV DPAS accumulator (int32 for int8)
  using ElementAAcc = typename TiledMMAPV::ValTypeD;
  using SingleFragA = cute::conditional_t<IsInt8, decltype(sg_fragment_rebind<float>(SingleFragAAcc{})), SingleFragAAcc>;
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;     // (atom val,q',v',VV) -- fp32 O accumulator
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = cute::conditional_t<IsInt8, float, ElementAAcc>;
  using FragAAcc = expand_sg_fragment_t<SingleFragAAcc, 1, VTiles>;  // int32 per-K-block PV partial (int8 only)

  // Static quantization scale for P (softmax probabilities) before the int8 P*V GEMM.
  // P lies in [0,1] (un-normalized exp2 values), mapped to int8 via round(P * kPQuant).
  static constexpr float kPQuant = 127.0f;
  // log2(kPQuant): the P-quant scale is applied by lowering the max subtracted inside
  // exp2 (P = exp2(scale*S - max + log2(kPQuant)) = kPQuant * exp2(scale*S - max)), so P
  // is produced already in [0,kPQuant]. This removes a separate full-tile multiply pass
  // between softmax and the float->int8 reorder and uses the full int8 range (better
  // quant resolution), mirroring flash-attention's fp8 "max_offset" trick.
  static constexpr float kPQuantLog2 = 6.988684686772166f;  // log2(127)
#else
  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;                          // (atom val,q',v')
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;     // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;
#endif

  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool CachedKV = CachedKV_;
  static constexpr bool PagedKV = PagedKV_;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
#ifdef IS_INT8
    float const v_scale = 1.0f;   // int8: per-tensor V dequant scale (applied via tA_sum)
#endif
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
#ifdef IS_INT8
    return Params{val, args.v_scale, args.ptr_page_table, args.page_size, args.num_pages_per_seq};
#else
    return Params{val, args.ptr_page_table, args.page_size, args.num_pages_per_seq};
#endif
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
            TensorV_cache2D const& V_cache_2D = TensorV_cache2D{}) {
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
#ifdef INT8_FP8_MIMIC
    /* EXPERIMENT (INT8_FP8_MIMIC): make int8 mimic fp8 by REMOVING the int8-specific
       int32->float conversions. PV accumulates in a persistent int32 fragment across
       ALL K blocks (like a plain GEMM), with no per-block dequant and no online
       accumulator rescale; a single int32->float convert happens after the K-loop.
       NUMERICALLY WRONG (no flash online-softmax rescale of O) -- perf measurement only. */
    FragAAcc tArA_i32;
#endif
    if (blk_k0 == 0) {
      clear(tArA);
#ifdef INT8_FP8_MIMIC
      if constexpr (IsInt8) { clear(tArA_i32); }
#endif
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
    }

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    /* Main loop body.
       is_boundary: compile-time flag — true only for the last K tile where
       causal masking and k-remainder masking may apply. Splitting this out
       eliminates masking branches from the hot inner loop, producing a
       tighter instruction stream for the non-boundary iterations.
       For non-causal mode, a single loop with runtime checks is used to
       avoid the code bloat of two template instantiations. */
    auto mainloop_body = [&](auto cached_k, auto is_boundary, int K,
                            auto& copy_k_cur, auto& copy_v_cur,
                            auto& prefetch_v_cur, auto& tKgK_cur,
                            auto& tVgV_cur, auto& pVgV_cur) {
      barrier_arrive(ScopeWorkgroup);
      constexpr bool is_cache = decltype(cached_k)::value;
      constexpr bool boundary = decltype(is_boundary)::value;

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

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* V prefetch for GEMM 2 */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        prefetch(prefetch_v_cur, pVgV_cur(_,_,_,VV,k_idx));
      }

#ifdef IS_INT8
      /* ---- int8 path ----
         QK accumulated to int32 in tSrS; dequant of Q/K (q_scale*k_scale) is
         folded into params.scale, so promote the raw int32 scores to fp32 and
         run masking + softmax in fp32 exactly like the bf16 path.

         Key: alias tSrS's registers as a float tile and convert IN PLACE, instead
         of allocating a separate float tile. bf16 runs softmax in place on tSrS;
         a separate int8 tile would keep BOTH the int32 and float S-tiles live,
         ~doubling the largest intermediate's register footprint and pushing the
         kernel into 256-GRF pressure (the reason int8 trailed bf16). The float
         view and the int32 fragment share one storage, so only one S-tile is live. */
      auto tPf = make_subgroup_tensor(
          make_tensor(recast_ptr<float>(tSrS.data()), tSrS.layout()),
          tSrS.tv_layout());
#ifdef INT8_FP8_MIMIC
      /* Model fp8: QK outputs float natively, so there is NO score conversion.
         tPf aliases the int32 QK accumulator storage; a bit_cast loop would just
         read and write the same 32 bits back (a redundant full-tile load/store),
         so we skip it. The QK GEMM result is still consumed by softmax() through
         the float alias, so it is not dead-code eliminated. NUMERICALLY WRONG. */
#else
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tSrS.size(); i++) {
        tPf(i) = static_cast<float>(tSrS(i));   // read int slot i, write float slot i
      }
#endif

      /* Causal masking — only on boundary tile in non-cache mode */
      if constexpr (!is_cache && CausalMask && boundary) {
        if (K == total_blk - 1) {
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tPf.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            if (col_idx - seq_len_kv_cache - full_tile_offset > row_idx - discard_seq_coord) {
              tPf(i) = -INFINITY;
            }
          }
        }
      }
      /* k masking for remainder tiles — only on boundary tile */
      if constexpr (!is_cache && boundary) {
        if (check_remainder_k && K == total_blk - 1) {
          FragSCol k_rem_mask;
          int k_val = get<0>(tKgK_cur(0,0,0,k_idx,0)) + kblocks_cache * get<1>(TileShapeQK{});
          int k = k_val + get_sub_group().get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
            k_rem_mask(i) = (k < seq_len) ? float(sycl::nan(0u)) : float(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tPf.size(); i++) {
            tPf(i) = sycl::fmin(tPf(i), broadcast<1>(k_rem_mask, tPf, i));
          }
        }
      }

      /* softmax in fp32; the static P-quant scale (kPQuant) is folded into the
         exp2 offset inside softmax(), so P comes out already in [0,kPQuant] and is
         cast straight to int8 by reorder(). Folding removes a separate full-tile
         multiply pass that would otherwise sit between softmax and the cast (IGC
         does not fuse two independent passes), keeping the cast pipeline-friendly. */
      auto rescale = softmax(K == blk_k0, tPf, tA_max, tA_sum);
      reorder(tPf, tArP);

#ifdef INT8_FP8_MIMIC
      /* EXPERIMENT: accumulate int8 P*V straight into a persistent int32 fragment
         (no clear, no dequant, no rescale) -- removes the per-VTile int32->float
         conversion + rescale-add pass that is the int8-specific overhead vs fp8. */
      (void)rescale;
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v_cur, tVgV_cur(_,_,_,VV,k_idx), tVrV);
        reorder(tVrV, tArV);
        cute::gemm(mma_pv, tArP, tArV, tArA_i32(_,_,_,VV));
      }
#else
      /* GEMM 2: int8 P*V into an int32 per-block partial, then dequant to fp32
         and online-rescale-accumulate into the fp32 O accumulator. */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v_cur, tVgV_cur(_,_,_,VV,k_idx), tVrV);
        reorder(tVrV, tArV);

        SingleFragAAcc tA_blk;             // int32 partial for this K block / VTile
        clear(tA_blk);
        cute::gemm(mma_pv, tArP, tArV, tA_blk);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tArA.size() / VTiles; i++) {
          float prev = (K != blk_k0) ? (tArA(_,_,_,VV)(i) * broadcast<0>(rescale, tArA, i)) : 0.0f;
          tArA(_,_,_,VV)(i) = prev + static_cast<float>(tA_blk(i));
        }
      }
#endif
#else
      /* Causal masking — only on boundary tile in non-cache mode */
      if constexpr (!is_cache && CausalMask && boundary) {
        if (K == total_blk - 1) {
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
      /* k masking for remainder tiles — only on boundary tile */
      if constexpr (!is_cache && boundary) {
        if (check_remainder_k && K == total_blk - 1) {
          FragSCol k_rem_mask;
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

        cute::gemm(mma_pv, tArP, tArV, tArA(_,_,_,VV));
      }
#endif

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
      barrier_wait(ScopeWorkgroup);
    };

    /* Main loop, blocked in k. */
    if constexpr (CachedKV) {
      for (int K = blk_k0; K < kblocks_cache; K++) {
        mainloop_body(std::bool_constant<true>{}, std::false_type{}, K,
                      copy_k_cache, copy_v_cache,
                      prefetch_v_cache, tKgK_cache,
                      tVgV_cache, pVgV_cache);
      }
    }

    /* Non-cache K loop. */
    {
#ifdef ENABLE_SPLIT_BOUNDARY
      int K_start = (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache);
      int K_end = blk_k1;

      if constexpr (CausalMask) {
        /* Causal: split into clean inner loop + boundary tile.
           The inner loop has zero masking code (compiled away via is_boundary=false_type),
           giving a tighter instruction stream for the majority of iterations. */
        for (int K = K_start; K < K_end - 1; K++) {
          mainloop_body(std::bool_constant<false>{}, std::false_type{}, K,
                        copy_k, copy_v,
                        prefetch_v, tKgK,
                        tVgV, pVgV);
        }
        if (K_start < K_end) {
          mainloop_body(std::bool_constant<false>{}, std::true_type{}, K_end - 1,
                        copy_k, copy_v,
                        prefetch_v, tKgK,
                        tVgV, pVgV);
        }
      } else {
        /* Non-causal: single loop, single template instantiation.
           Remainder masking uses runtime check (K == total_blk - 1). */
        for (int K = K_start; K < K_end; K++) {
          mainloop_body(std::bool_constant<false>{}, std::true_type{}, K,
                        copy_k, copy_v,
                        prefetch_v, tKgK,
                        tVgV, pVgV);
        }
      }
#else
      for (int K = (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache); K < blk_k1; K++) {
        mainloop_body(std::bool_constant<false>{}, std::true_type(), K,
        //mainloop_body(std::bool_constant<false>{}, K,
                      copy_k, copy_v,
                      prefetch_v, tKgK,
                      tVgV, pVgV);
      }
#endif
    }

    /* int8: undo the V dequant scale in the normalization. tArA = sum_k (kPQuant*P)*V_i8
       and tA_sum = sum_k (kPQuant*P) (the kPQuant factor is already baked into both by
       the exp2 offset fold), so O = tArA / (tA_sum / v_scale) = v_scale * weighted_avg(V),
       i.e. the kPQuant factors cancel and only the V dequant (v_scale) remains. */
#ifdef IS_INT8
    {
#ifdef INT8_FP8_MIMIC
      /* Single int32->float convert of the whole O accumulator after the K-loop
         (replaces the per-block convert/rescale removed above). REQUIRED: this is the
         only reader of the persistent int32 PV accumulator tArA_i32 -- without it IGC
         dead-code-eliminates the entire P*V GEMM, invalidating the measurement. It is a
         once-after-loop pass, mirroring real fp8's epilogue read of its float acc. */
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tArA.size(); i++) {
        tArA(i) = static_cast<float>(tArA_i32(i));
      }
#endif
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA_sum.size(); i++) {
        tA_sum(i) *= (1.0f / params.v_scale);
      }
    }
#endif
  }

#ifdef ENABLE_FUSED_SOFTMAX
  // Fully-fused softmax: max + exp2 + sum in one function.
  // Layout reshaping done once; temp[] array reused between max and sum phases.
#ifdef IS_INT8
  // int8: templated on the scores tensor so it accepts an aliasing float view over
  // the int8 path's int32 QK accumulator (in-place softmax).
  template <class TensorS>
  CUTLASS_DEVICE
  FragSRow
  softmax(bool       first_block,
          TensorS  & tS,
          FragSRow & tS_max,
          FragSRow & tS_sum) {
    using namespace cute;
    using T = typename TensorS::element_type;
#else
  CUTLASS_DEVICE
  FragSRow
  softmax(bool       first_block,
          FragS    & tS,
          FragSRow & tS_max,
          FragSRow & tS_sum) {
    using namespace cute;
    using T = typename FragS::element_type;
#endif
    using TVLayout = decltype(tS.tv_layout());
    using TVToV = Layout<Shape<intel::_SGSize,int>, Stride<_0,_1>>;

    constexpr int Mode = 1;
    constexpr auto shape = atuple_coshape(TVLayout{});
    constexpr auto coord_to_tv = right_inverse(project_strides(TVLayout{})).with_shape(shape);
    constexpr auto rcoord_to_tv = make_layout(select<Mode>(coord_to_tv), remove<Mode>(coord_to_tv));
    constexpr auto rcoord_to_v = filter(composition(TVToV{}, rcoord_to_tv), Step<_1,_1>{});

    Tensor src_r = make_tensor(tS.data(), rcoord_to_v);

    constexpr bool horizontal = (size<0>(rcoord_to_tv) == intel::_SGSize{} * size<0>(rcoord_to_v));
    constexpr bool align16 = is_constant_v<0, decltype(size<1>(rcoord_to_v) % _16{})>;
    constexpr bool align8  = is_constant_v<8, decltype(size<1>(rcoord_to_v))>;
    constexpr bool hadd16 = horizontal && std::is_same_v<T, float> && align16;
    constexpr bool hadd8  = horizontal && std::is_same_v<T, float> && align8;

    [[maybe_unused]] T temp[size<1>(rcoord_to_v)];

    // ---- Phase 1: Vertical max accumulation ----
    CUTE_UNROLL
    for (int j = 0; j < size<1>(rcoord_to_v); j++) {
      T max_acc = src_r(0, j);
      CUTE_UNROLL
      for (int i = 1; i < size<0>(rcoord_to_v); i++) {
        max_acc = sycl::max(max_acc, src_r(i, j));
      }
      if constexpr (hadd16 || hadd8)
        temp[j] = max_acc;
      else if constexpr (horizontal) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        temp[j] = reduce_over_group(sg, max_acc, sycl::maximum<void>{});
      } else {
        temp[j] = max_acc;
      }
    }

    // ---- Cross-lane max reduction ----
    FragSRow tS_bmax;
    if constexpr (hadd16) {
      CUTE_UNROLL
      for (int j = 0; j < size<1>(rcoord_to_v); j += 16) {
        tS_bmax(j/16) = hreduce16_float_max(&temp[j]);
      }
    } else if constexpr (hadd8) {
      tS_bmax(0) = hreduce8_float_max(&temp[0]);
    } else {
      CUTE_UNROLL
      for (int j = 0; j < size<1>(rcoord_to_v); j++) {
        set_single_value(tS_bmax, j, temp[j]);
      }
    }

    // ---- Update global max and compute rescale ----
    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    // ---- Phase 2: Fused exp2 + vertical sum accumulation ----
    CUTE_UNROLL
    for (int j = 0; j < size<1>(rcoord_to_v); j++) {
      int flat_idx = rcoord_to_v(0, j);
      T max_val = broadcast<0>(tS_max, tS, flat_idx);
#ifdef IS_INT8
      // int8: lower the subtracted max by log2(kPQuant) so exp2 yields kPQuant*P
      // directly (P pre-scaled into [0,kPQuant] for int8 quant). tS_max is untouched,
      // so the cross-tile rescale (exp2(old_max-new_max)) is unaffected.
      max_val -= T(kPQuantLog2);
#endif

      src_r(0, j) = sycl::native::exp2(params.scale * src_r(0, j) - max_val);
      T acc = src_r(0, j);
      CUTE_UNROLL
      for (int i = 1; i < size<0>(rcoord_to_v); i++) {
        src_r(i, j) = sycl::native::exp2(params.scale * src_r(i, j) - max_val);
        acc += src_r(i, j);
      }

      if constexpr (hadd16 || hadd8)
        temp[j] = acc;
      else if constexpr (horizontal) {
        auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
        temp[j] = reduce_over_group(sg, acc, sycl::plus<void>{});
      } else {
        temp[j] = acc;
      }
    }

    // ---- Cross-lane sum reduction + update tS_sum ----
    FragSRow tS_bsum;
    if constexpr (hadd16) {
      CUTE_UNROLL
      for (int j = 0; j < size<1>(rcoord_to_v); j += 16) {
        tS_bsum(j/16) = hreduce16_float_add(&temp[j]);
      }
    } else if constexpr (hadd8) {
      tS_bsum(0) = hreduce8_float_add(&temp[0]);
    } else {
      CUTE_UNROLL
      for (int j = 0; j < size<1>(rcoord_to_v); j++) {
        set_single_value(tS_bsum, j, temp[j]);
      }
    }

    if (!first_block) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) {
        tS_sum(i) = tS_sum(i) * rescale(i) + tS_bsum(i);
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) {
        tS_sum(i) = tS_bsum(i);
      }
    }

    return rescale;
  }
#else
#ifdef IS_INT8
  // int8: templated on the scores tensor so it accepts an aliasing float view over
  // the int8 path's int32 QK accumulator (in-place softmax).
  template <class TensorS>
  // Single step of blocked softmax.
  CUTLASS_DEVICE
  FragSRow
  softmax(bool       first_block, // First softmax block?
          TensorS    & tS,          // Softmax src/dst block
          FragSRow & tS_max,      // Softmax row-wise max accumulator
          FragSRow & tS_sum) {    // Softmax row-wise sum accumulator
#else
  // Single step of blocked softmax.
  CUTLASS_DEVICE
  FragSRow
  softmax(bool       first_block, // First softmax block?
          FragS    & tS,          // Softmax src/dst block
          FragSRow & tS_max,      // Softmax row-wise max accumulator
          FragSRow & tS_sum) {    // Softmax row-wise sum accumulator
#endif
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
#endif
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
