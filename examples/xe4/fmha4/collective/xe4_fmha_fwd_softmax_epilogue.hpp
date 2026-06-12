/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation, All rights reserved.
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
// =============================================================================
//  FMHA4 SOFTMAX EPILOGUE — CuTe LDSM-factory migration
// =============================================================================
//
//  All SLM↔register transfers in this file go through the unified LDSM
//  warp-row factory API from `cute/atom/copy_traits_xe4_ldsm.hpp`:
//      cute::make_ldsm_copy_warp_row_C<NumWarps>(slm_tensor) — load  (S2R)
//      cute::make_ldsm_copy_warp_row_D<NumWarps>(slm_tensor) — store (R2S)
//
//  The factory builds a TiledCopy that encodes the within-warp-row
//  (T, V) → (M, N) coord mapping (so row-mates land within ONE warp,
//  satisfying the within-warp `fred.max` reduction's invariant) while still
//  emitting the HW cooperative-row PISA instruction
//  (`ld_matrix.unordered.al<N>.cooprow.<bw>`).
//
//  Equivalent to the legacy `cm_vrow_*_unordered` formula:
//      m = wi_lo + 8·eu_id + 32·eu_sg_id + RowsPerWi·iter
//      n = wi_hi · NumValPerWIPerIter + atom_val
//
//  At each call site, user code does:
//      auto tc       = cute::make_ldsm_copy_warp_row_C<NumSoftmaxWarps>(sX_packed);
//      auto thr_part = tc.get_slice(worker_id).partition_S(coord_full_tile);
//      copy(tc, thr_part(_, i, _0{}), reg_slot);    // per iter i
//
//  partition_S(...) returns a (CPY, CPY_M=TotalRowsPerThread, CPY_N=1) tensor.
//  CPY_M indexes the outer softmax iteration in row-stride RowsPerWi (= 2)
//  units, matching the cross-method invariant that
//  sum_reg[i] / max_reg[i] / exp_reg[i] indices stay aligned across
//  update / rescale_O / final_rescale_O.
//
//  No XE4_LOAD_MATRIX / XE4_STORE_MATRIX op types appear in this file.
//  No matrix_desc_t.  No EU-aware constants.  No cm_vrow_* intrinsics.
// =============================================================================

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cute/atom/copy_traits_xe4_ldsm.hpp"

// Gate for the packed fp16 FMAD register-compute path inside update().
#define USE_PACKED_FMAD

namespace cutlass::flash_attention::collective {

using namespace cute;

template <
  class TileShape_,
  class ElementAccum_,
  class ElementOutput_,
  class ElementS_,
  class ElementP_,
  int NumSoftmaxWarps_ = 16,
  int NumThreadPerRow_ = 16,
  int Unroll_ = 2,
  int NumStage_ = 2>
class CollectiveSoftmaxEpilogue {
public:

  using TileShape = TileShape_;                                   // <BLK_M_Q, BLK_N_V, BLK_N_QK, BLK_K_QK>
  using TileShapeQK_MNK = decltype(select<0, 2, 3>(TileShape{})); // <BLK_M_Q, BLK_N_QK, BLK_K_QK>
  using TileShapePV_MNK = decltype(select<0, 1, 2>(TileShape{}));

  using ElementAccum = ElementAccum_;
  using ElementOutput = ElementOutput_;
  using ElementS = ElementS_;
  using ElementP = ElementP_;

  static constexpr int SP_tile_M = CUTE_STATIC_V(get<0>(TileShapeQK_MNK{}));
  static constexpr int O_tile_M = CUTE_STATIC_V(get<0>(TileShapePV_MNK{}));

  static constexpr int Unroll = Unroll_;
  static constexpr int NumStage = NumStage_;

  static constexpr int NumSoftmaxWarps = NumSoftmaxWarps_;
  static constexpr int NumThreadPerRow = NumThreadPerRow_;
  static constexpr int numRowsPerWarp = cutlass::NumThreadsPerWarp / NumThreadPerRow;
  static constexpr int TotalRowsPerThread = SP_tile_M / (NumSoftmaxWarps * numRowsPerWarp);
  static constexpr int numRowsPerIteration = NumSoftmaxWarps * numRowsPerWarp;

  // Host side epilogue arguments
  struct Arguments {
    ElementAccum const softmax_scale; // 1/sqrt(head dim)
  };

  // Device side epilogue params.
  //   scale         = softmax_scale * log2(e), used to fold exp() into exp2()
  //                   for the tensor-pipe gtp_texp_red_sum primitive.
  //   softmax_scale = 1/sqrt(head_dim).
  struct Params {
    ElementAccum const scale;
    ElementAccum const softmax_scale;
  };

  static constexpr Params to_underlying_arguments(Arguments const &args) {
    constexpr double log2e = 1.4426950408889634074f;
    ElementAccum scale = args.softmax_scale * static_cast<ElementAccum>(log2e);
    return { scale, args.softmax_scale };
  }

  CUTLASS_HOST_DEVICE
  CollectiveSoftmaxEpilogue(Params const &params_) : params(params_) {}

  template <int numRowsPerSubgroup>
  CUTLASS_DEVICE uint32_t generate_reduce_mask(int local_row_id){
    uint32_t mask = 0;
    if constexpr(numRowsPerSubgroup == 1) {
      mask = 0xFFFFFFFF;
    } else if constexpr (numRowsPerSubgroup == 2){
      mask = 0x55555555 << local_row_id;
    } else if constexpr (numRowsPerSubgroup == 4){
      mask = 0x11111111 << local_row_id;
    } else if constexpr (numRowsPerSubgroup == 8){
      mask = 0x01010101 << local_row_id;
    } else if constexpr (numRowsPerSubgroup == 16){
      mask = 0x00010001 << local_row_id;
    } else {
      static_assert(0, "Unsupported");
    }
    return mask;
  }

  // reduce_max — within-thread tred.max + cross-lane fred.max.
  // Mirrors master_next reduce_max but operates on the packed uint32 register
  // slot used by the LDSM-factory load path (instead of a Tensor wrapper).
  template <int numRowsPerSubgroup, int numElemPerThread, class PackedT>
  CUTLASS_DEVICE ElementS reduce_max(sycl::sub_group sg, uint32_t worker_id,
                                     PackedT const* tRS_rS_packed,
                                     ElementS max_prev, int local_row_id) {
    ElementS max = max_prev;
    if constexpr (std::is_same_v<ElementS, float>) {
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < numElemPerThread; ++k) {
        ElementS val = sycl::bit_cast<ElementS>(tRS_rS_packed[k]);
        max = sycl::max(max, val);
      }
    } else {
      max = gtp_tred_max<ElementS, numElemPerThread>(tRS_rS_packed, max);
    }

    for (int k = 0; k < numRowsPerSubgroup; ++k) {
      auto mask = generate_reduce_mask<numRowsPerSubgroup>(k);
      if constexpr (std::is_same_v<ElementS, float>) {
        INLINE_PISA("fred.max.f %0, %0, %1;" : "+r"(max) : "i"(mask));
      } else if constexpr (std::is_same_v<ElementS, fp16>) {
        INLINE_PISA("fred.max.hf %0, %0, %1;" : "+r"(max) : "i"(mask));
      } else if constexpr (std::is_same_v<ElementS, bf16>) {
        INLINE_PISA("fred.max.bf %0, %0, %1;" : "+r"(max) : "i"(mask));
      } else {
        static_assert(sizeof(ElementS) == 0, "unsupported case");
      }
    }
    return max;
  }

  template <int numRowsPerSubgroup>
  CUTLASS_DEVICE float reduce_sum(sycl::sub_group sg, uint32_t worker_id,
                                  float sum, int local_row_id) {
    float row_sum = sum;
    for (int i = numRowsPerSubgroup; i < cutlass::NumThreadsPerWarp; i *= 2) {
      row_sum += permute_group_by_xor(sg, row_sum, i);
    }
    return row_sum;
  }

  // scale_apply_exp2 — packed FMAD on uint32 + tensor-pipe gtp_texp_red_sum.
  // Mirrors master_next scale_apply_exp2 but takes the packed uint32 S/P slots
  // used by the LDSM-factory pipeline (master_next operates on a Tensor).
  template <int numRowsPerSubgroup, int numElemPerThread,
            int packedNumS, int packedArrLenS,
            int packedNumP, class PackedT>
  CUTLASS_DEVICE float scale_apply_exp2(sycl::sub_group sg, uint32_t worker_id,
                                        PackedT* tRS_rS_packed,
                                        PackedT* tRS_rP_packed,
                                        ElementS max_curr) {
    ElementAccum max_acc_type;
    cvt<ElementAccum, ElementS>(max_acc_type, max_curr);
    const ElementAccum src2 = -(max_acc_type * params.softmax_scale);

    ElementS tmp_src1[packedNumS];
    ElementS tmp_src2[packedNumS];
    for (int k = 0; k < packedNumS; ++k) {
      cvt<ElementS, ElementAccum>(tmp_src1[k], params.softmax_scale);
      cvt<ElementS, ElementAccum>(tmp_src2[k], src2);
    }

    PackedT packed_src1, packed_src2;
    pack_data<packedNumS>(&packed_src1, tmp_src1);
    pack_data<packedNumS>(&packed_src2, tmp_src2);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < packedArrLenS; ++k) {
      tRS_rS_packed[k] = packed_fmad<ElementS>(
          tRS_rS_packed[k], packed_src1, packed_src2);
    }

    constexpr int maxElemPerCall = cute::min(32, numElemPerThread);
    constexpr int numCall = numElemPerThread / maxElemPerCall;

    float sum = 0.0f;
    for (int j = 0; j < numCall; ++j) {
      auto per_packed_call_offset_s = j * maxElemPerCall / packedNumS;
      auto per_packed_call_offset_p = j * maxElemPerCall / packedNumP;
      sum += gtp_texp_red_sum<ElementP, ElementS, maxElemPerCall, PackedT>(
          &tRS_rP_packed[per_packed_call_offset_p],
          &tRS_rS_packed[per_packed_call_offset_s]);
    }
    return sum;
  }

  //
  // update<Init>() — online softmax iteration over one KV tile per Q-half.
  //
  // For each per-call iter i ∈ [0, TotalRowsPerThread):
  //   (1) Load S-stripe from SLM into the (curr_stage,inner) register slot.
  //   (2) Compute new rowmax (cross-WI fred.max) and update max_reg[i].
  //   (3) Apply scale-exp-rowsum (packed FMAD on uint32 + tensor-pipe
  //       gtp_texp_red_sum), produce P-stripe in the same register slot.
  //   (4) Prefetch S-stripe for iter i+unroll into next_stage.
  //   (5) Store P-stripe back to SLM.
  //
  // The (num_stage × unroll) register pipeline is rotated via
  // curr_stage = next_stage at the bottom of each outer loop.
  //
  // Init=true on the first KV tile per Q-half (sum_reg is initialized
  // earlier; exp_reg is unused).  Init=false on subsequent tiles (computes
  // exp_reg[i] from max_prev−max_curr for the rescale_O correction).
  //
  template <bool Init, class TensorSlmS, class TensorSlmP>
  CUTLASS_DEVICE void update(sycl::sub_group sg, uint32_t worker_id,
                             TensorSlmS const& sS_stage,
                             TensorSlmP const& sP_stage,
                             ElementS*         max_reg,
                             ElementAccum*     sum_reg,
                             ElementAccum*     exp_reg) {
    constexpr auto numRowsPerSubgroup = numRowsPerWarp;

    constexpr int SP_tile_N = CUTE_STATIC_V(get<1>(TileShapeQK_MNK{}));
    constexpr int numElemPerThread = SP_tile_N / NumThreadPerRow;

    constexpr auto unroll    = cute::min(Unroll, TotalRowsPerThread);
    constexpr auto num_stage = cute::min(NumStage, TotalRowsPerThread);

    using dtype_packed = uint32_t;
    constexpr auto packedNumS    = sizeof(dtype_packed) / sizeof(ElementS);
    constexpr auto packedArrLenS = numElemPerThread / packedNumS;
    constexpr auto packedNumP    = sizeof(dtype_packed) / sizeof(ElementP);
    constexpr auto packedArrLenP = numElemPerThread / packedNumP;

    static_assert(numElemPerThread % packedNumS == 0,
                  "numElemPerThread must be divisible by packedNumS");
    static_assert(numElemPerThread % packedNumP == 0,
                  "numElemPerThread must be divisible by packedNumP");

    // SLM-tensor recast to uint32:
    //   The factory's descriptor builder (`make_ldsm_matrix_descriptor`) does
    //   `Pitch = stride_in_elements >> 2`, which is correct only when
    //   sizeof(element)==4.  For fp16 SLM (sizeof=2) the implicit assumption
    //   is broken — the resulting Pitch is 2× too large.  Recasting to uint32
    //   halves the column count, making `stride>>2` produce HW-correct units.
    //   Concretely:    fp16 row 512×2B = 1024 B  → Pitch=64 (16 B units)
    //                  vs uint32 row 256×4B = 1024 B → Pitch=64    ✓
    //   This is the same convention master_next legacy uses
    //   (`packed_row_size_s = kv_stride * sizeof(fp16) / sizeof(uint32_t)`).
    //   Bonus: the resulting op type operates on uint32 — the same dtype as
    //   the packed-FMAD compute path below, so register fragments are shared
    //   without further reinterpret_cast.
    auto sS_packed = recast<dtype_packed>(sS_stage);
    auto sP_packed = recast<dtype_packed>(sP_stage);

    // Build the warp-row TiledCopies via the unified factory.  See the big
    // comment block on `make_ldsm_copy_warp_row_CD_int` in
    // `cute/atom/copy_traits_xe4_ldsm.hpp` for the (T,V)→(M,N) derivation;
    // the short story is that this factory bypasses `make_tiled_copy`'s
    // injectivity check and bakes the within-warp-row coord formula into
    // the TiledCopy so HW cooperative-row + within-warp `fred.max` coexist.
    auto tc_load  = cute::make_ldsm_copy_warp_row_C<NumSoftmaxWarps>(sS_packed);
    auto tc_store = cute::make_ldsm_copy_warp_row_D<NumSoftmaxWarps>(sP_packed);

    // Identity coord tensor over the per-stage SLM tile (already recast to
    // uint32 columns).  partition_S/D resolves
    //     (T, V, RestM, RestN) → original (m, n)
    // with shape (CPY = NumValPerWIPerIter, CPY_M = TotalRowsPerThread, CPY_N = 1).
    // Indexing thr(_, i, _0{}) below selects the i-th iter at row-stride
    // RowsPerWi (= 2) — matching the legacy `2*iter` y-stride.
    constexpr int SP_tile_N_packed = SP_tile_N / packedNumS;
    auto coord_sp = make_identity_tensor(make_shape(Int<SP_tile_M>{},
                                                      Int<SP_tile_N_packed>{}));
    auto thr_src  = tc_load .get_slice(worker_id).partition_S(coord_sp);
    auto thr_dst  = tc_store.get_slice(worker_id).partition_D(coord_sp);

    // Register-pipeline storage.  Sized as packed uint32 elements per
    // (stage, inner) slot so the factory's copy() can write directly into
    // the same array used by the packed-FMAD path (no reinterpret_cast).
    //   packedArrLenS = numElemPerThread / packedNumS = NEPT*2/4 = NEPT/2 fp16
    //   For FMHA fp16: 32 fp16 = 16 uint32 per slot.
    dtype_packed tRS_rS[num_stage][unroll][packedArrLenS];
    dtype_packed tRS_rP[num_stage][unroll][packedArrLenP];

    int32_t curr_stage = 0;

    // ── Prologue: load iters [0, unroll) into curr_stage ──
    // Each `copy(tc, src, dst)` issues exactly one HW
    //   ld_matrix.unordered.al<Alen>.cooprow.<bw>
    // per WI, delivering `Vlen·Alen` packed uint32 = NEPT fp16 in one shot.
    // `thr_src(_, i, _0{})` provides the iter-i (idx_x, idx_y) coord that
    // each WI submits to the HW cohort (the coord is read from
    // `src.data().coord_` inside copy_unpack — see Copy_Traits<XE4_LOAD_MATRIX>).
    CUTLASS_PRAGMA_UNROLL
    for (int inner = 0; inner < unroll; ++inner) {
      int i = inner;
      Tensor reg_view = make_tensor(
          &tRS_rS[curr_stage][inner][0],
          make_layout(make_shape(Int<packedArrLenS>{}), GenRowMajor{}));
      copy(tc_load, thr_src(_, i, _0{}), reg_view);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int outer = 0; outer < TotalRowsPerThread / unroll; ++outer) {
      ElementS max_prev[unroll];

      // (1) Rowmax — intra-thread tred.max + cross-lane fred.max.
      auto local_lane_id = worker_id % cutlass::NumThreadsPerWarp;
      auto local_row_id  = local_lane_id % numRowsPerSubgroup;
      CUTLASS_PRAGMA_UNROLL
      for (int inner = 0; inner < unroll; ++inner) {
        int i = outer * unroll + inner;
        max_prev[inner] = max_reg[i];
        max_reg[i] = reduce_max<numRowsPerSubgroup, numElemPerThread>(
            sg, worker_id, tRS_rS[curr_stage][inner],
            max_prev[inner], local_row_id);
      }

      // (2) Scale + exp + rowsum (packed FMAD + fused texp_red_sum).
      CUTLASS_PRAGMA_UNROLL
      for (int inner = 0; inner < unroll; ++inner) {
        int i = outer * unroll + inner;

        float sum_prev = sum_reg[i];

        if constexpr (!Init) {
          ElementAccum max_acc_type, max_prev_acc_type;
          cvt<ElementAccum, ElementS>(max_acc_type, max_reg[i]);
          cvt<ElementAccum, ElementS>(max_prev_acc_type, max_prev[inner]);
          ElementAccum sub_acc_type = max_prev_acc_type - max_acc_type;
          const float exp_scale = sycl::native::exp2(sub_acc_type * params.scale);
          exp_reg[i] = exp_scale;
          sum_prev *= exp_scale;
        }

        float sum = scale_apply_exp2<numRowsPerSubgroup, numElemPerThread,
                                     packedNumS, packedArrLenS, packedNumP>(
            sg, worker_id,
            tRS_rS[curr_stage][inner], tRS_rP[curr_stage][inner],
            max_reg[i]);

        sum_reg[i] = sum_prev + sum;
      }

      // (3) Prefetch next iteration's S-stripe into next_stage.
      auto next_stage = (curr_stage == num_stage - 1) ? 0 : curr_stage + 1;
      if (outer < TotalRowsPerThread / unroll - 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int inner = 0; inner < unroll; ++inner) {
          int i = (outer + 1) * unroll + inner;
          Tensor reg_view = make_tensor(
              &tRS_rS[next_stage][inner][0],
              make_layout(make_shape(Int<packedArrLenS>{}), GenRowMajor{}));
          copy(tc_load, thr_src(_, i, _0{}), reg_view);
        }
      }

      // (4) Store current iteration's P-stripe back to SLM.
      CUTLASS_PRAGMA_UNROLL
      for (int inner = 0; inner < unroll; ++inner) {
        int i = outer * unroll + inner;
        Tensor reg_view = make_tensor(
            &tRS_rP[curr_stage][inner][0],
            make_layout(make_shape(Int<packedArrLenP>{}), GenRowMajor{}));
        copy(tc_store, reg_view, thr_dst(_, i, _0{}));
      }

      curr_stage = next_stage;
    }
  }

  //
  // rescale_O() — apply per-iter exp_reg[i] correction to the running
  // O-accumulator stored in SLM.  Three-pass: load all iters → scale all
  // iters → store all iters.  Called once per KV tile (after `update<false>`
  // on that tile produces a non-trivial exp_reg).
  //
  // Oacc is fp32, so no SLM recast needed — `make_ldsm_matrix_descriptor`'s
  // `Pitch = stride>>2` is correct as-is for 4-byte elements.  The factory's
  // selector picks UnorderedVector for fp32 NEPT=8 (coop_vlen<32>=8) →
  // emits `ld_matrix.unordered.al1.cooprow.32b` per WI.
  //
  // Cross-method invariant: the `i` index here matches the `i` used in
  // update<>().  Both partition_S/D the full per-stage SLM tile through the
  // same TV layout + interspersed M-tiler, so iter i touches the same
  // physical rows in update / rescale_O / final_rescale_O.
  //
  template <class TensorSlmOacc>
  CUTLASS_DEVICE void rescale_O(sycl::sub_group sg, uint32_t worker_id,
                                TensorSlmOacc const& sOacc_stage,
                                ElementAccum*        exp_reg) {
    constexpr int O_tile_N = CUTE_STATIC_V(get<1>(TileShapePV_MNK{}));
    constexpr int numElemPerThread = O_tile_N / NumThreadPerRow;

    // Same TiledCopy used for both load and store — Oacc has the same
    // layout in both directions.  The factory builds two distinct atoms
    // (XE4_LOAD_MATRIX vs XE4_STORE_MATRIX) so `tc_load` ≠ `tc_store` even
    // though the tensor is the same.
    auto tc_load  = cute::make_ldsm_copy_warp_row_C<NumSoftmaxWarps>(sOacc_stage);
    auto tc_store = cute::make_ldsm_copy_warp_row_D<NumSoftmaxWarps>(sOacc_stage);
    auto coord    = make_identity_tensor(make_shape(Int<O_tile_M>{}, Int<O_tile_N>{}));
    auto thr_src  = tc_load .get_slice(worker_id).partition_S(coord);
    auto thr_dst  = tc_store.get_slice(worker_id).partition_D(coord);

    ElementAccum rOacc[TotalRowsPerThread][numElemPerThread];

    // Pass 1: load
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < TotalRowsPerThread; ++i) {
      Tensor reg_view = make_tensor(&rOacc[i][0],
          make_layout(make_shape(Int<numElemPerThread>{}), GenRowMajor{}));
      copy(tc_load, thr_src(_, i, _0{}), reg_view);
    }

    // Pass 2: scale
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < TotalRowsPerThread; ++i) {
      float exp_scale = exp_reg[i];
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < numElemPerThread; ++k) {
        rOacc[i][k] *= exp_scale;
      }
    }

    // Pass 3: store
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < TotalRowsPerThread; ++i) {
      Tensor reg_view = make_tensor(&rOacc[i][0],
          make_layout(make_shape(Int<numElemPerThread>{}), GenRowMajor{}));
      copy(tc_store, reg_view, thr_dst(_, i, _0{}));
    }
  }

  //
  // final_rescale_O() — normalize Oacc by the global rowsum, cast to the
  // output dtype, and store to sO.  Called once per Q-half after the KV loop
  // closes.
  //
  // Three passes:
  //   1. Cooperative-row load fp32 Oacc into per-iter rOacc registers.
  //   2. For each iter i: cross-warp reduce sum_reg[i] (XOR-permute across
  //      the lanes of the warp that share a row), scale by 1/global_sum,
  //      cast Oacc → ElementOutput.
  //   3. Store ElementOutput rO to sO.  For fp16/bf16 output the factory
  //      auto-falls-back to per-warp Vector mode (NEPT=8 < coop_vlen<16>=16
  //      can't satisfy cooperative-row); for fp32 output it stays
  //      cooperative-row (matches the legacy ordered fallback at master_next
  //      softmax_epilogue.hpp:738).
  //
  template <class TensorSlmOacc, class TensorSlmO, class FragSum>
  CUTLASS_DEVICE void final_rescale_O(sycl::sub_group sg, uint32_t worker_id,
                                      TensorSlmOacc const& sOacc_stage,
                                      TensorSlmO    const& sO_stage,
                                      FragSum&             sum_reg) {
    constexpr auto numRowsPerSubgroup = numRowsPerWarp;
    constexpr int O_tile_N = CUTE_STATIC_V(get<1>(TileShapePV_MNK{}));
    constexpr int numElemPerThread = O_tile_N / NumThreadPerRow;

    auto tc_oacc = cute::make_ldsm_copy_warp_row_C<NumSoftmaxWarps>(sOacc_stage);
    auto tc_o    = cute::make_ldsm_copy_warp_row_D<NumSoftmaxWarps>(sO_stage);
    auto coord   = make_identity_tensor(make_shape(Int<O_tile_M>{}, Int<O_tile_N>{}));
    auto thr_src_oacc = tc_oacc.get_slice(worker_id).partition_S(coord);
    auto thr_dst_o    = tc_o   .get_slice(worker_id).partition_D(coord);

    auto local_lane_id = worker_id % cutlass::NumThreadsPerWarp;
    auto local_row_id  = local_lane_id % numRowsPerSubgroup;

    ElementAccum  rOacc[TotalRowsPerThread][numElemPerThread];
    ElementOutput rO[TotalRowsPerThread][numElemPerThread];

    // Pass 1: cooperative-row load fp32 Oacc.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < TotalRowsPerThread; ++i) {
      Tensor reg_view = make_tensor(&rOacc[i][0],
          make_layout(make_shape(Int<numElemPerThread>{}), GenRowMajor{}));
      copy(tc_oacc, thr_src_oacc(_, i, _0{}), reg_view);
    }

    // Pass 2: cross-warp rowsum reduction → normalize → cast.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < TotalRowsPerThread; ++i) {
      float global_sum = reduce_sum<numRowsPerSubgroup>(
          sg, worker_id, sum_reg[i], local_row_id);
      float scale = 1.f / global_sum;
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < numElemPerThread; ++k) {
        rOacc[i][k] *= scale;
      }
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < numElemPerThread; ++k) {
        cvt<ElementOutput, ElementAccum>(rO[i][k], rOacc[i][k]);
      }
    }

    // Pass 3: store ElementOutput O.  The factory's selector picks
    // cooperative-row for fp32 output and falls back to per-warp Vector
    // mode for fp16/bf16 output where WarpTileN < coop_vlen — matching
    // the legacy ordered fallback at master_next softmax_epilogue.hpp:738.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < TotalRowsPerThread; ++i) {
      Tensor reg_view = make_tensor(&rO[i][0],
          make_layout(make_shape(Int<numElemPerThread>{}), GenRowMajor{}));
      copy(tc_o, reg_view, thr_dst_o(_, i, _0{}));
    }
  }

private:
  Params const& params;
};
} // namespace cutlass::flash_attention::collective
