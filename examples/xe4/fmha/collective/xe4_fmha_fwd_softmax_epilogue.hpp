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
#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"

namespace cutlass::flash_attention::collective {

// Selects the largest vectorized smem store atom available
template <int EpilogueWarpTileN, class ElementD>
constexpr auto
xe4_get_smem_store_op() {
  constexpr int CoreMatrixRowSize = 32;  // 32B
  constexpr int VS = cute::min(CoreMatrixRowSize/sizeof(ElementD), EpilogueWarpTileN);
  return cute::xe4::XE4_STSM<VS, ElementD, ElementD>{};
}

// Selects the largest vectorized smem load atom available
template <int EpilogueWarpTileN, class ElementD>
constexpr auto
xe4_get_smem_load_op() {
  constexpr int CoreMatrixRowSize = 32;  // 32B
  constexpr int VS = cute::min(CoreMatrixRowSize/sizeof(ElementD), EpilogueWarpTileN);
  return cute::xe4::XE4_LDSM<VS, ElementD, ElementD>{};
}

template <bool S2R, class ValType, int NumEpilogueWarps, int NumThreadPerRow, int TotalRowPerThread, class TileShape>
CUTLASS_HOST_DEVICE constexpr auto make_tiled_copy(TileShape const& tile_shape) {
  static_assert(is_static<TileShape>::value, "Tile shape must be static");

  constexpr int tile_M = CUTE_STATIC_V(get<0>(tile_shape));
  constexpr int tile_N = CUTE_STATIC_V(get<1>(tile_shape)); // row size

  constexpr uint32_t numElemPerThread = tile_N / NumThreadPerRow; // 128/16=8
  constexpr uint32_t numRowsPerWarp = cutlass::NumThreadsPerWarp / NumThreadPerRow; // 32/16=2
  constexpr uint32_t numRowsPerIteration = NumEpilogueWarps * numRowsPerWarp; // 8*2=16

  static_assert((tile_M / numRowsPerIteration) == TotalRowPerThread);

  // The thread layout is:
  // thr0, thr2, thr4, ..., thr30
  // thr1, thr3, thr5, ..., thr31
  auto thr_layout = make_ordered_layout(
    Shape<Shape<Int<numRowsPerWarp>, Int<NumEpilogueWarps>>, Int<NumThreadPerRow>>{},
    Step<Step<_0,_2>,_1>{}
  );

  auto val_layout = make_layout(Shape<_1,Int<numElemPerThread>>{}, GenRowMajor{});

  if constexpr (S2R){
    // slm load
    using SlmVOp = decltype(xe4_get_smem_load_op<numElemPerThread, ValType>());
    using Atom = Copy_Atom<SlmVOp, ValType>;
    auto tiled_copy = make_tiled_copy(Atom{}, thr_layout, val_layout);
    return tiled_copy;
  }else {
    // slm store
    using SlmVOp = decltype(xe4_get_smem_store_op<numElemPerThread, ValType>());
    using Atom = Copy_Atom<SlmVOp, ValType>;
    auto tiled_copy = make_tiled_copy(Atom{}, thr_layout, val_layout);
    return tiled_copy;
  }
}

template <
  class TileShape_,
  class ElementAccum_,
  class ElementOutput_,
  class ElementS_,
  class ElementP_>
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

  // static_assert(SP_tile_M == O_tile_M);

  static constexpr int NumSoftmaxWarps = 16;
  static constexpr int NumThreadPerRow = 8;
  static constexpr int numRowsPerWarp = cutlass::NumThreadsPerWarp / NumThreadPerRow;
  static constexpr int TotalRowsPerThread = SP_tile_M / (NumSoftmaxWarps * numRowsPerWarp);
  static constexpr int numRowsPerIteration = NumSoftmaxWarps * numRowsPerWarp;

  using TiledCopyS2R_Update = decltype(make_tiled_copy</*S2R*/true, ElementS, NumSoftmaxWarps, NumThreadPerRow, TotalRowsPerThread>(TileShapeQK_MNK{}));
  using TiledCopyR2S_Update = decltype(make_tiled_copy</*S2R*/false, ElementS, NumSoftmaxWarps, NumThreadPerRow, TotalRowsPerThread>(TileShapeQK_MNK{}));
  using TiledCopyS2R_RescaleO = decltype(make_tiled_copy</*S2R*/true, ElementAccum, NumSoftmaxWarps, NumThreadPerRow, TotalRowsPerThread>(TileShapePV_MNK{}));
  using TiledCopyR2S_RescaleO = decltype(make_tiled_copy</*S2R*/false, ElementAccum, NumSoftmaxWarps, NumThreadPerRow, TotalRowsPerThread>(TileShapePV_MNK{}));
  using TiledCopyS2R_FinalRescaleO = decltype(make_tiled_copy</*S2R*/true, ElementOutput, NumSoftmaxWarps, NumThreadPerRow, TotalRowsPerThread>(TileShapePV_MNK{}));
  using TiledCopyR2S_FinalRescaleO = decltype(make_tiled_copy</*S2R*/false, ElementOutput, NumSoftmaxWarps, NumThreadPerRow, TotalRowsPerThread>(TileShapePV_MNK{}));

  // Host side epilogue arguments
  struct Arguments {
    ElementAccum const softmax_scale; // 1/sqrt(head dim)
  };

  // Device side epilogue params
  struct Params {
    TiledCopyS2R_Update tiled_copy_s2r_update;
    TiledCopyR2S_Update tiled_copy_r2s_update;
    TiledCopyS2R_RescaleO tiled_copy_s2r_rescale_o;
    TiledCopyR2S_RescaleO tiled_copy_r2s_rescale_o;
    TiledCopyS2R_FinalRescaleO tiled_copy_s2r_final_rescale_o;
    TiledCopyR2S_FinalRescaleO tiled_copy_r2s_final_rescale_o;
    ElementAccum const scale;
    ElementAccum const softmax_scale;
  };

  static constexpr Params to_underlying_arguments(Arguments const &args) {
    TiledCopyS2R_Update tiled_copy_s2r_update = TiledCopyS2R_Update{};
    TiledCopyR2S_Update tiled_copy_r2s_update = TiledCopyR2S_Update{};
    TiledCopyS2R_RescaleO tiled_copy_s2r_rescale_o = TiledCopyS2R_RescaleO{};
    TiledCopyR2S_RescaleO tiled_copy_r2s_rescale_o = TiledCopyR2S_RescaleO{};
    TiledCopyS2R_FinalRescaleO tiled_copy_s2r_final_rescale_o = TiledCopyS2R_FinalRescaleO{};
    TiledCopyR2S_FinalRescaleO tiled_copy_r2s_final_rescale_o = TiledCopyR2S_FinalRescaleO{};

    constexpr double log2e = 1.4426950408889634074f;
    ElementAccum scale = args.softmax_scale * static_cast<ElementAccum>(log2e);
    return {
      tiled_copy_s2r_update, tiled_copy_r2s_update, 
      tiled_copy_s2r_rescale_o, tiled_copy_r2s_rescale_o, 
      tiled_copy_s2r_final_rescale_o, tiled_copy_r2s_final_rescale_o, 
      scale, args.softmax_scale};
  }

  CUTLASS_HOST_DEVICE
  CollectiveSoftmaxEpilogue(Params const &params_) : params(params_) {}

  CUTLASS_DEVICE const Params& get_params() {
    return params;
  }

  // Refer to https://curly-invention-299nr7q.pages.github.io/eu/fred.html for
  // the mask generation.
  template <int numRowsPerSubgroup>
  CUTLASS_DEVICE uint32_t generate_reduce_mask(int local_row_id){
    uint32_t mask = 0;
    if constexpr(numRowsPerSubgroup == 1) {
      mask = 0xFFFFFFFF; // reduce cross all threads in sg
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

  template <int numRowsPerSubgroup, typename FragRmem>
  CUTLASS_DEVICE ElementS reduce_max(sycl::sub_group sg, uint32_t worker_id, 
                                 FragRmem& tRS_rS, ElementS max_prev, int local_row_id) {
    ElementS max = max_prev;
    // Get the max of current thread
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size(tRS_rS); ++k) {
      max = sycl::max(max, tRS_rS[k]);
    }

    ElementS row_max[numRowsPerSubgroup];
    for (int k = 0; k < numRowsPerSubgroup; k++){
      auto mask = generate_reduce_mask<numRowsPerSubgroup>(k);
      if constexpr (std::is_same_v<ElementS, float>) {
        INLINE_PISA("fred.max.sync.f %0, %1, %2;" : "=r"(row_max[k]) : "r"(max), "r"(mask));
      } else if constexpr (std::is_same_v<ElementS, fp16>) {
        INLINE_PISA("fred.max.sync.hf %0, %1, %2;" : "=r"(row_max[k]) : "r"(max), "r"(mask));
      } else {
        static_assert(sizeof(ElementS) == 0, "unsupported case");
      }
    }
    return row_max[local_row_id];
  }

  template <int numRowsPerSubgroup>
  CUTLASS_DEVICE float reduce_sum(sycl::sub_group sg, uint32_t worker_id, float sum, int local_row_id) {
    float row_sum[numRowsPerSubgroup];
    for (int k = 0; k < numRowsPerSubgroup; k++){
      auto mask = generate_reduce_mask<numRowsPerSubgroup>(k);
      row_sum[k] = sycl::reduce_over_group(sg, (mask & (1 << worker_id % 32)) ? sum : 0.0f, sycl::plus<float>{});
    }
    return row_sum[local_row_id];
  }

  template <int numRowsPerSubgroup, typename FragRmem>
  CUTLASS_DEVICE float scale_apply_exp2(sycl::sub_group sg, uint32_t worker_id, 
                                       FragRmem& tRS_rS, 
                                       ElementS max_curr) {
    const ElementS src1 = static_cast<ElementS>(params.softmax_scale);
    const ElementS src2 = -(max_curr * params.softmax_scale);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size(tRS_rS); ++k) {
      if constexpr (std::is_same_v<ElementS, float>) {
        INLINE_PISA("fmad.f %0, %1, %2, %3;" : "=r"(tRS_rS[k]) : "r"(tRS_rS[k]), "r"(src1), "r"(src2));
      } else if constexpr (std::is_same_v<ElementS, fp16>) {
        INLINE_PISA("fmad.hf %0, %1, %2, %3;" : "=r"(tRS_rS[k]) : "r"(tRS_rS[k]), "r"(src1), "r"(src2));
      } else {
        static_assert(sizeof(ElementS) == 0, "unsupported case");
      }
    }

    constexpr int SP_tile_N = CUTE_STATIC_V(get<1>(TileShapeQK_MNK{}));
    constexpr int numElemPerThread = SP_tile_N / NumThreadPerRow; // 128/16=8
    constexpr int maxElemPerCall = cute::min(32, numElemPerThread);
    constexpr int numCall = numElemPerThread / maxElemPerCall;

    float sum = 0.0;
    for (int j = 0; j < numCall; j++) {
      auto per_call_offset = j * maxElemPerCall;
      sum += gtp_texp_red_sum<ElementS, ElementS, maxElemPerCall, ElementS>(tRS_rS.data() + per_call_offset, tRS_rS.data() + per_call_offset);
    }
    return sum;
  }

  template <bool Init, typename FragSmemS, typename FragRmemS, typename FragMax, typename FragSum, typename FragExp, typename mat_desc_t = uint32_t>
  CUTLASS_DEVICE void update(sycl::sub_group sg, uint32_t worker_id, 
                             FragSmemS const& tSR_sS, FragRmemS& tRS_rS,
                             FragMax& max_reg,
                             FragSum& sum_reg,
                             FragExp& exp_reg,
                             const mat_desc_t& p_desc) {
    constexpr auto numRowsPerSubgroup = numRowsPerWarp;

    auto sg_id = worker_id / cutlass::NumThreadsPerWarp;
    auto local_lane_id = worker_id % cutlass::NumThreadsPerWarp;
    auto local_row_id = local_lane_id % numRowsPerSubgroup;

    for (int i = 0; i < size<1>(tSR_sS); ++i) {
      // TODO: to use ld_matrix to reduce the address calculation
      copy(params.tiled_copy_s2r_update, tSR_sS(_, i), tRS_rS);

      ElementS max_prev = max_reg[i];
      float sum_prev = sum_reg[i];
      ElementS max_curr = 0.f;
      float sum_curr = 0.f;

      // the global max of the row of current tile
      max_curr = reduce_max<numRowsPerSubgroup>(sg, worker_id, tRS_rS, max_prev, local_row_id);
      max_reg[i] = max_curr;

      if constexpr (!Init){
        const float exp_scale = sycl::native::exp2((max_prev - max_curr) * params.scale);
        exp_reg[i] = exp_scale;
        sum_prev *= exp_scale;
      }

      // the local sum of the row in 1 thread
      sum_curr = sum_prev + scale_apply_exp2<numRowsPerSubgroup>(sg, worker_id, tRS_rS, max_curr);
      sum_reg[i] = sum_curr;

      // Convert type
      // TODO: remove magic numbers
      constexpr int SP_tile_N = CUTE_STATIC_V(get<1>(TileShapeQK_MNK{}));
      constexpr int numElemPerThread = SP_tile_N / NumThreadPerRow;
      constexpr int maxElemPerStore = cute::min(32 / sizeof(ElementP), numElemPerThread);
      constexpr int numStore = numElemPerThread / maxElemPerStore;

      uint32_t row_base = (worker_id / cutlass::NumThreadsPerWarp) * numRowsPerSubgroup + (worker_id % numRowsPerSubgroup);
      uint32_t col_base = ((worker_id % cutlass::NumThreadsPerWarp) / numRowsPerSubgroup) * numElemPerThread;

      for (int j = 0; j < numStore; j++) {
        auto per_store_offset = j * maxElemPerStore;
        sycl::marray<uint16_t, 2> coord = {static_cast<uint16_t>(col_base + per_store_offset), static_cast<uint16_t>(i * numRowsPerIteration + row_base)};

        ElementP st_vec[maxElemPerStore];
        for (int k = 0; k < maxElemPerStore; ++k) {
          ElementP val = static_cast<ElementP>(tRS_rS[per_store_offset + k]);
          st_vec[k] = *reinterpret_cast<ElementP*>(&val);
        }
        // TODO: Add Xe4 copy atom for ld_matrix/st_matrix
        cm_vrow_store<ElementP, maxElemPerStore>(p_desc, st_vec, coord);
      }
    }
  }

  template <typename FragSmemO, typename FragExp>
  CUTLASS_DEVICE void rescale_O(sycl::sub_group sg, uint32_t worker_id, 
                               FragSmemO& tSR_sOacc,
                               FragExp& exp_reg) {
    Tensor rOacc = make_tensor<ElementAccum>(tSR_sOacc(_, 0).shape());
    
    for (int i = 0; i < size<1>(tSR_sOacc); ++i) {
      copy(params.tiled_copy_s2r_rescale_o, tSR_sOacc(_, i), rOacc);

      float exp_scale = exp_reg[i];
      for (int k = 0; k < size(rOacc); ++k) {
        rOacc[k] *= exp_scale;
      }
      copy(params.tiled_copy_r2s_rescale_o, rOacc, tSR_sOacc(_, i));
    }
  }

  template <typename FragSmemOacc, typename FragSmemO, typename FragSum>
  CUTLASS_DEVICE void final_rescale_O(sycl::sub_group sg, uint32_t worker_id,
                                     FragSmemOacc& tSR_sOacc, FragSmemO& tRS_sO, 
                                     FragSum& sum_reg) {
    constexpr auto numRowsPerSubgroup = numRowsPerWarp;

    auto sg_id = worker_id / cutlass::NumThreadsPerWarp;
    auto local_lane_id = worker_id % cutlass::NumThreadsPerWarp;
    auto local_row_id = local_lane_id % numRowsPerSubgroup;

    static_assert(CUTE_STATIC_V(size<0>(tSR_sOacc)) == CUTE_STATIC_V(size<0>(tRS_sO)), "tSR_sOacc and tRS_sO should have same size");
    static_assert(CUTE_STATIC_V(size<1>(tSR_sOacc)) == CUTE_STATIC_V(size<1>(tRS_sO)), "tSR_sOacc and tRS_sO should have same size");
    
    Tensor rOacc = make_tensor<ElementAccum>(tSR_sOacc(_, 0).shape());
    Tensor rO = make_tensor<ElementOutput>(tRS_sO(_, 0).shape());

    for (int i = 0; i < size<1>(tSR_sOacc); ++i) {
      copy(params.tiled_copy_s2r_rescale_o, tSR_sOacc(_, i), rOacc);

      // reduce the local sum cross threads in same row to get the global sum
      float global_sum = reduce_sum<numRowsPerSubgroup>(sg, worker_id, sum_reg[i], local_row_id);

      float scale = (global_sum == 0.f || global_sum != global_sum) ? 1.f : 1.f / global_sum;
      for (int k = 0; k < size(rOacc); ++k) {
        rO[k] = static_cast<ElementOutput>(rOacc[k] * scale);
      }
      copy(params.tiled_copy_r2s_final_rescale_o, rO, tRS_sO(_, i));
    }
  }

private:
  Params const& params;
};
} // namespace cutlass::flash_attention::collective
