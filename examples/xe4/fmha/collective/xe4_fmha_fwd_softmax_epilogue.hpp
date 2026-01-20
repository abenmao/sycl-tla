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

#define USE_PACKED_FMAD
#define USE_LD_ST_MATRIX
#define USE_UNORDERED_LOAD_STORE

namespace cutlass::flash_attention::collective {

using namespace cute;

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

  // static_assert(SP_tile_M == O_tile_M);

  static constexpr int Unroll = Unroll_;
  static constexpr int NumStage = NumStage_;

  static constexpr int NumSoftmaxWarps = NumSoftmaxWarps_;
  static constexpr int NumThreadPerRow = NumThreadPerRow_;
  static constexpr int numRowsPerWarp = cutlass::NumThreadsPerWarp / NumThreadPerRow;
  static constexpr int TotalRowsPerThread = SP_tile_M / (NumSoftmaxWarps * numRowsPerWarp);
  static constexpr int numRowsPerIteration = NumSoftmaxWarps * numRowsPerWarp;

  static constexpr uint32_t TYPE1_CM_ELEM_Y = 32;
  static constexpr uint32_t eu_num = 4;
  static constexpr uint32_t eu_sg_num = NumSoftmaxWarps / eu_num;
  static constexpr uint32_t type1_cm_y_per_eu = TYPE1_CM_ELEM_Y / eu_num; 
  static constexpr uint32_t num_itr_per_cm = type1_cm_y_per_eu / numRowsPerWarp;

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
                                 FragRmem const& tRS_rS, ElementS max_prev, int local_row_id) {
    ElementS max = max_prev;
    // Get the max of current thread
    if constexpr (std::is_same_v<ElementS, float>) {
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size(tRS_rS); ++k) {
        max = sycl::max(max, tRS_rS[k]);
      }
    } else {
      // For 16 bits or lower precision dtype, use tensor pipe
      constexpr auto vecLen = CUTE_STATIC_V(size(tRS_rS));
      max = gtp_tred_max<ElementS, vecLen>(tRS_rS.data(), max);
    }

    for (int k = 0; k < numRowsPerSubgroup; k++){
      auto mask = generate_reduce_mask<numRowsPerSubgroup>(k);
      // max register value won't be modified if this lane is masked out
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
  CUTLASS_DEVICE float reduce_sum(sycl::sub_group sg, uint32_t worker_id, float sum, int local_row_id) {
    float row_sum = sum;
    for (int i = numRowsPerSubgroup; i < cutlass::NumThreadsPerWarp; i *= 2) {
      row_sum += permute_group_by_xor(sg, row_sum, i);
    }
    return row_sum;
  }

  template <int numRowsPerSubgroup, typename FragRmem>
  CUTLASS_DEVICE float scale_apply_exp2(sycl::sub_group sg, uint32_t worker_id, 
                                       FragRmem& tRS_rS, 
                                       ElementS max_curr) {
    const ElementS src1 = static_cast<ElementS>(params.softmax_scale);
    const ElementS src2 = -(max_curr * params.softmax_scale);

#ifdef USE_PACKED_FMAD
    using dtype_packed = uint32_t;

    constexpr auto packedNum = sizeof(dtype_packed) / sizeof(ElementS);
    ElementS tmp_src1[packedNum];
    ElementS tmp_src2[packedNum];
    for (int k = 0; k < packedNum; k++) {
      tmp_src1[k] = src1;
      tmp_src2[k] = src2;
    }

    dtype_packed packed_src1, packed_src2;
    pack_data<packedNum>(&packed_src1, tmp_src1);
    pack_data<packedNum>(&packed_src2, tmp_src2);

    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size(tRS_rS); k+=packedNum) {
      dtype_packed packed_src0, packed_dst;
      pack_data<packedNum>(&packed_src0, &tRS_rS[k]);
      packed_dst = packed_fmad<ElementS>(packed_src0, packed_src1, packed_src2);
      unpack_data<packedNum>(&tRS_rS[k], &packed_dst);
    }
#else
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
#endif

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

  template <bool Init, typename FragSmemS, typename FragMax, typename FragSum, typename FragExp, typename mat_desc_t = uint32_t>
  CUTLASS_DEVICE void update(sycl::sub_group sg, uint32_t worker_id, 
                             FragSmemS const& tSR_sS,
                             const mat_desc_t& s_desc,
                             FragMax& max_reg,
                             FragSum& sum_reg,
                             FragExp& exp_reg,
                             const mat_desc_t& p_desc) {
    constexpr auto numRowsPerSubgroup = numRowsPerWarp;

    constexpr int SP_tile_N = CUTE_STATIC_V(get<1>(TileShapeQK_MNK{}));
    constexpr int numElemPerThread = SP_tile_N / NumThreadPerRow;
    constexpr int maxElemPerStore = cute::min(32 / sizeof(ElementP), numElemPerThread);
    constexpr int numStore = numElemPerThread / maxElemPerStore;
    constexpr int maxElemPerLoad = cute::min(32 / sizeof(ElementS), numElemPerThread);
    constexpr int numLoad = numElemPerThread / maxElemPerLoad;

    auto sg_id = worker_id / cutlass::NumThreadsPerWarp;
    auto local_lane_id = worker_id % cutlass::NumThreadsPerWarp;
    auto local_row_id = local_lane_id % numRowsPerSubgroup;

#ifndef USE_LD_ST_MATRIX
    Tensor tRS_rS = make_tensor<ElementS>(tSR_sS(_, 0).shape());

    for (int i = 0; i < size<1>(tSR_sS); ++i) {
      // Load data from slm to reg
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
#else

    uint32_t wi_id = worker_id % cutlass::NumThreadsPerWarp;

    uint32_t eu_id = sg_id % eu_num;
    uint32_t eu_sg_id = sg_id / eu_num;

    constexpr auto unroll = cute::min(Unroll, TotalRowsPerThread);
    constexpr auto num_stage = cute::min(NumStage, TotalRowsPerThread);

    // Tensor tRS_rS = make_tensor<ElementS>(tSR_sS(_, 0).shape());
    using dtype_packed = uint32_t;
    constexpr auto packedNumS = sizeof(dtype_packed) / sizeof(ElementS);
    constexpr auto packedArrLenS = numElemPerThread / packedNumS;

    constexpr auto packedNumP = sizeof(dtype_packed) / sizeof(ElementP);
    constexpr auto packedArrLenP = numElemPerThread / packedNumP;

    dtype_packed tRS_rS[num_stage][unroll][packedArrLenS];
    dtype_packed tRS_rP[num_stage][unroll][packedArrLenP];

    int32_t curr_stage = 0;

#ifndef USE_UNORDERED_LOAD_STORE
    uint32_t col_base = (local_lane_id / numRowsPerSubgroup) * numElemPerThread / packedNum;
    uint32_t row_base = sg_id * numRowsPerSubgroup + (worker_id % numRowsPerSubgroup);
#else 
    uint32_t idx_x_base_s = (wi_id / numRowsPerSubgroup) * numElemPerThread / packedNumS;
    uint32_t idx_x_base_p = (wi_id / numRowsPerSubgroup) * numElemPerThread / packedNumP;
    uint32_t idx_y = eu_sg_id * TYPE1_CM_ELEM_Y + eu_id * type1_cm_y_per_eu + (wi_id % numRowsPerSubgroup);
#endif

    // load data for the 0-th iteration
    int outer = 0;
    for (int inner = 0; inner < unroll; inner++){
      int i = outer * unroll + inner;

      // Load data from slm to reg
#ifndef USE_UNORDERED_LOAD_STORE
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < numLoad; j++) {
        auto per_load_offset = j * maxElemPerLoad / packedNum;;
        sycl::marray<uint16_t, 2> coord = {static_cast<uint16_t>(col_base + per_load_offset), static_cast<uint16_t>(i* numRowsPerIteration + row_base)};
        cm_vrow_load<dtype_packed, maxElemPerLoad / packedNum>(&tRS_rS[curr_stage][inner][per_load_offset], s_desc, coord);
      }
#else
      uint16_t curr_idx_x = idx_x_base_s;
      uint32_t cur_row_i = i; // the iter index
      uint16_t curr_idx_y =
          idx_y + (cur_row_i % num_itr_per_cm) * numRowsPerWarp + (cur_row_i / num_itr_per_cm) * (NumSoftmaxWarps * type1_cm_y_per_eu);
      sycl::marray<uint16_t, 2> coord = {curr_idx_x, curr_idx_y};
      cm_vrow_load_unordered<dtype_packed, maxElemPerLoad / packedNumS, numLoad>(tRS_rS[curr_stage][inner], s_desc.get(), coord);
#endif
    }

    CUTLASS_PRAGMA_UNROLL
    for (int outer = 0; outer < size<1>(tSR_sS) / unroll; outer++) {
      ElementS max_prev[unroll];

      for (int inner = 0; inner < unroll; inner++){
        int i = outer * unroll + inner;

        max_prev[inner] = max_reg[i];

        /////////////////////////////////////////////////////////////////////////
        // the global max of the row of current tile
        ElementS max = max_prev[inner];
        // Get the max of current thread
        // tred.max instr doesn't support f32 dtype
        if constexpr (std::is_same_v<ElementS, float>) {
          static_assert(numElemPerThread == packedArrLenS);
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < numElemPerThread; ++k) {
            ElementS val = sycl::bit_cast<ElementS>(tRS_rS[curr_stage][inner][k]);
            max = sycl::max(max, val);
          }
        } else {
          max = gtp_tred_max<ElementS, numElemPerThread>(tRS_rS[curr_stage][inner], max);
        }

        for (int k = 0; k < numRowsPerSubgroup; k++){
          auto mask = generate_reduce_mask<numRowsPerSubgroup>(k);
          // max register value won't be modified if this lane is masked out
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
        /////////////////////////////////////////////////////////////////

        max_reg[i] = max;
      }

      for (int inner = 0; inner < unroll; inner++){
        int i = outer * unroll + inner;

        float sum_prev = sum_reg[i];

        ElementAccum max_acc_type, max_prev_acc_type;
        cvt<ElementAccum, ElementS>(max_acc_type, max_reg[i]);

        if constexpr (!Init){
          cvt<ElementAccum, ElementS>(max_prev_acc_type, max_prev[inner]);
          ElementAccum sub_acc_type = max_prev_acc_type - max_acc_type;
          const float exp_scale = sycl::native::exp2(sub_acc_type * params.scale);
          exp_reg[i] = exp_scale;
          sum_prev *= exp_scale;
        }

        const ElementAccum src2 = -(max_acc_type * params.softmax_scale);
    
        ElementS tmp_src1[packedNumS];
        ElementS tmp_src2[packedNumS];
        for (int k = 0; k < packedNumS; k++) {
          cvt<ElementS, ElementAccum>(tmp_src1[k], params.softmax_scale);
          cvt<ElementS, ElementAccum>(tmp_src2[k], src2);
        }
    
        dtype_packed packed_src1, packed_src2;
        pack_data<packedNumS>(&packed_src1, tmp_src1);
        pack_data<packedNumS>(&packed_src2, tmp_src2);
    
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < packedArrLenS; k++) {
          tRS_rS[curr_stage][inner][k] = packed_fmad<ElementS>(tRS_rS[curr_stage][inner][k], packed_src1, packed_src2);
        }

        constexpr int maxElemPerCall = cute::min(32, numElemPerThread);
        constexpr int numCall = numElemPerThread / maxElemPerCall;

        float sum = 0.0;
        for (int j = 0; j < numCall; j++) {
          auto per_packed_call_offset_s = j * maxElemPerCall / packedNumS;
          auto per_packed_call_offset_p = j * maxElemPerCall / packedNumP;
          sum += gtp_texp_red_sum<ElementP, ElementS, maxElemPerCall, dtype_packed>(&tRS_rP[curr_stage][inner][per_packed_call_offset_p], &tRS_rS[curr_stage][inner][per_packed_call_offset_s]);
        }

        sum_reg[i] = sum_prev + sum;
      }

      // load data for the next iteration
      auto next_stage = (curr_stage == num_stage - 1)? 0: curr_stage + 1;
      if (outer < size<1>(tSR_sS) / unroll - 1) {
        for (int inner = 0; inner < unroll; inner++){
          int i = (outer + 1) * unroll + inner;

          // Load data from slm to reg
#ifndef USE_UNORDERED_LOAD_STORE
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < numLoad; j++) {
            auto per_load_offset = j * maxElemPerLoad / packedNum;;
            sycl::marray<uint16_t, 2> coord = {static_cast<uint16_t>(col_base + per_load_offset), static_cast<uint16_t>(i* numRowsPerIteration + row_base)};
            cm_vrow_load<dtype_packed, maxElemPerLoad / packedNum>(&tRS_rS[next_stage][inner][per_load_offset], s_desc, coord);
          }
#else
          uint16_t curr_idx_x = idx_x_base_s;
          uint32_t cur_row_i = i;
          uint16_t curr_idx_y =
              idx_y + (cur_row_i % num_itr_per_cm) * numRowsPerWarp + (cur_row_i / num_itr_per_cm) * (NumSoftmaxWarps * type1_cm_y_per_eu);
          sycl::marray<uint16_t, 2> coord = {curr_idx_x, curr_idx_y};
          cm_vrow_load_unordered<dtype_packed, maxElemPerLoad / packedNumS, numLoad>(tRS_rS[next_stage][inner], s_desc.get(), coord);
#endif
        }
      }

      // Store data for the current iteration
      for (int inner = 0; inner < unroll; inner++){
        int i = outer * unroll + inner;

        // Convert type and store reg data to slm
#ifndef USE_UNORDERED_LOAD_STORE
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < numStore; j++) {
          auto per_store_offset = j * maxElemPerStore / packedNum;
          sycl::marray<uint16_t, 2> coord = {static_cast<uint16_t>(col_base + per_store_offset), static_cast<uint16_t>(i * numRowsPerIteration + row_base)};
          cm_vrow_store<dtype_packed, maxElemPerStore / packedNum>(p_desc, &tRS_rS[curr_stage][inner][per_store_offset], coord);
        }
#else
        uint16_t curr_idx_x = idx_x_base_p;
        uint32_t cur_row_i = i;
        uint16_t curr_idx_y =
            idx_y + (cur_row_i % num_itr_per_cm) * numRowsPerWarp + (cur_row_i / num_itr_per_cm) * (NumSoftmaxWarps * type1_cm_y_per_eu);
        sycl::marray<uint16_t, 2> coord = {curr_idx_x, curr_idx_y};
        cm_vrow_store_unordered<dtype_packed, maxElemPerStore / packedNumP, numStore>(p_desc.get(), tRS_rP[curr_stage][inner], coord);
#endif
      }

      curr_stage = next_stage;
    }
#endif
  }

  template <typename FragSmemO, typename FragExp, typename mat_desc_t = uint32_t>
  CUTLASS_DEVICE void rescale_O(sycl::sub_group sg, uint32_t worker_id, 
                               FragSmemO& tSR_sOacc,
                               const mat_desc_t& o_acc_desc,
                               FragExp& exp_reg) {
    constexpr int O_tile_N = CUTE_STATIC_V(get<1>(TileShapePV_MNK{}));
    constexpr int numElemPerThread = O_tile_N / NumThreadPerRow;
    constexpr int maxElemPerLoadStore = cute::min(32 / sizeof(ElementAccum), numElemPerThread);
    constexpr int numLoadStore = numElemPerThread / maxElemPerLoadStore;

#ifndef USE_LD_ST_MATRIX
    Tensor rOacc = make_tensor<ElementAccum>(tSR_sOacc(_, 0).shape());
    
    for (int i = 0; i < size<1>(tSR_sOacc); ++i) {
      copy(params.tiled_copy_s2r_rescale_o, tSR_sOacc(_, i), rOacc);

      float exp_scale = exp_reg[i];
      for (int k = 0; k < size(rOacc); ++k) {
        rOacc[k] *= exp_scale;
      }
      copy(params.tiled_copy_r2s_rescale_o, rOacc, tSR_sOacc(_, i));
    }
#else
    auto sg_id = worker_id / cutlass::NumThreadsPerWarp;
    uint32_t wi_id = worker_id % cutlass::NumThreadsPerWarp;

    uint32_t eu_id = sg_id % eu_num;
    uint32_t eu_sg_id = sg_id / eu_num;

#ifndef USE_UNORDERED_LOAD_STORE
    uint32_t row_base = (worker_id / cutlass::NumThreadsPerWarp) * numRowsPerWarp + (worker_id % numRowsPerWarp);
    uint32_t col_base = ((worker_id % cutlass::NumThreadsPerWarp) / numRowsPerWarp) * numElemPerThread;
#else
    uint32_t idx_x_base = (wi_id / numRowsPerWarp) * numElemPerThread;
    uint32_t idx_y = eu_sg_id * TYPE1_CM_ELEM_Y + eu_id * type1_cm_y_per_eu + (wi_id % numRowsPerWarp);
#endif

    ElementAccum rOacc[TotalRowsPerThread][numElemPerThread];
    
    for (int i = 0; i < size<1>(tSR_sOacc); ++i) {
#ifndef USE_UNORDERED_LOAD_STORE
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < numLoadStore; j++) {
        auto per_load_offset = j * maxElemPerLoadStore;
        sycl::marray<uint16_t, 2> coord = {static_cast<uint16_t>(col_base + per_load_offset), static_cast<uint16_t>(i * numRowsPerIteration + row_base)};
        // TODO: Add Xe4 copy atom for ld_matrix/st_matrix
        cm_vrow_load<ElementAccum, maxElemPerLoadStore>(&rOacc[i][per_load_offset], o_acc_desc, coord);
      }
#else
      uint16_t curr_idx_x = idx_x_base;
      uint16_t curr_idx_y =
          idx_y + (i % num_itr_per_cm) * numRowsPerWarp + (i / num_itr_per_cm) * (NumSoftmaxWarps * type1_cm_y_per_eu);
      sycl::marray<uint16_t, 2> coord = {curr_idx_x, curr_idx_y};
      cm_vrow_load_unordered<ElementAccum, maxElemPerLoadStore, numLoadStore>(rOacc[i], o_acc_desc.get(), coord);
#endif
    }

    for (int i = 0; i < size<1>(tSR_sOacc); ++i) {
      float exp_scale = exp_reg[i];
      for (int k = 0; k < numElemPerThread; ++k) {
        rOacc[i][k] *= exp_scale;
      }
    }

    for (int i = 0; i < size<1>(tSR_sOacc); ++i) {
#ifndef USE_UNORDERED_LOAD_STORE
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < numLoadStore; j++) {
        auto per_load_offset = j * maxElemPerLoadStore;
        sycl::marray<uint16_t, 2> coord = {static_cast<uint16_t>(col_base + per_load_offset), static_cast<uint16_t>(i * numRowsPerIteration + row_base)};
        // TODO: Add Xe4 copy atom for ld_matrix/st_matrix
        cm_vrow_store<ElementAccum, maxElemPerLoadStore>(o_acc_desc, &rOacc[i][per_load_offset], coord);
      }
#else
      uint16_t curr_idx_x = idx_x_base;
      uint16_t curr_idx_y =
          idx_y + (i % num_itr_per_cm) * numRowsPerWarp + (i / num_itr_per_cm) * (NumSoftmaxWarps * type1_cm_y_per_eu);
      sycl::marray<uint16_t, 2> coord = {curr_idx_x, curr_idx_y};
      cm_vrow_store_unordered<ElementAccum, maxElemPerLoadStore, numLoadStore>(o_acc_desc.get(), rOacc[i], coord);
#endif
    }
#endif
  }

  template <typename FragSmemOacc, typename FragSmemO, typename FragSum, typename mat_desc_t = uint32_t>
  CUTLASS_DEVICE void final_rescale_O(sycl::sub_group sg, uint32_t worker_id,
                                     FragSmemOacc& tSR_sOacc, FragSmemO& tRS_sO, 
                                     const mat_desc_t& o_acc_desc, const mat_desc_t& o_desc,
                                     FragSum& sum_reg) {
    constexpr auto numRowsPerSubgroup = numRowsPerWarp;
    constexpr int O_tile_N = CUTE_STATIC_V(get<1>(TileShapePV_MNK{}));
    constexpr int numElemPerThread = O_tile_N / NumThreadPerRow;
    constexpr int maxElemPerLoad = cute::min(32 / sizeof(ElementAccum), numElemPerThread);
    constexpr int numLoad = numElemPerThread / maxElemPerLoad;
    constexpr int maxElemPerStore = cute::min(32 / sizeof(ElementOutput), numElemPerThread);
    constexpr int numStore = numElemPerThread / maxElemPerStore;

    auto sg_id = worker_id / cutlass::NumThreadsPerWarp;
    auto local_lane_id = worker_id % cutlass::NumThreadsPerWarp;
    auto local_row_id = local_lane_id % numRowsPerSubgroup;

    static_assert(CUTE_STATIC_V(size<0>(tSR_sOacc)) == CUTE_STATIC_V(size<0>(tRS_sO)), "tSR_sOacc and tRS_sO should have same size");
    static_assert(CUTE_STATIC_V(size<1>(tSR_sOacc)) == CUTE_STATIC_V(size<1>(tRS_sO)), "tSR_sOacc and tRS_sO should have same size");

#if !defined(USE_LD_ST_MATRIX) || !defined(USE_UNORDERED_LOAD_STORE)
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
#else
    uint32_t wi_id = worker_id % cutlass::NumThreadsPerWarp;

    uint32_t eu_id = sg_id % eu_num;
    uint32_t eu_sg_id = sg_id / eu_num;

    uint32_t idx_x_base = (wi_id / numRowsPerWarp) * numElemPerThread;
    uint32_t idx_y = eu_sg_id * TYPE1_CM_ELEM_Y + eu_id * type1_cm_y_per_eu + (wi_id % numRowsPerWarp);
    ElementAccum rOacc[TotalRowsPerThread][numElemPerThread];
    ElementOutput rO[TotalRowsPerThread][numElemPerThread];

    for (int i = 0; i < size<1>(tSR_sOacc); ++i) {
      uint16_t curr_idx_x = idx_x_base;
      uint16_t curr_idx_y =
          idx_y + (i % num_itr_per_cm) * numRowsPerWarp + (i / num_itr_per_cm) * (NumSoftmaxWarps * type1_cm_y_per_eu);
      sycl::marray<uint16_t, 2> coord = {curr_idx_x, curr_idx_y};
      cm_vrow_load_unordered<ElementAccum, maxElemPerLoad, numLoad>(rOacc[i], o_acc_desc.get(), coord);
    }

    for (int i = 0; i < size<1>(tSR_sOacc); ++i) {
      // reduce the local sum cross threads in same row to get the global sum
      float global_sum = reduce_sum<numRowsPerSubgroup>(sg, worker_id, sum_reg[i], local_row_id);

      float scale = 1.f / global_sum;
      for (int k = 0; k < numElemPerThread; ++k) {
        rOacc[i][k] = rOacc[i][k] * scale;
      }

      for (int k = 0; k < numElemPerThread; ++k) {
        cvt<ElementOutput, ElementAccum>(rO[i][k], rOacc[i][k]);
      }
    }
    
    for (int i = 0; i < size<1>(tSR_sOacc); ++i) {
      uint16_t curr_idx_x = idx_x_base;
      uint16_t curr_idx_y =
          idx_y + (i % num_itr_per_cm) * numRowsPerWarp + (i / num_itr_per_cm) * (NumSoftmaxWarps * type1_cm_y_per_eu);
      sycl::marray<uint16_t, 2> coord = {curr_idx_x, curr_idx_y};

      if constexpr (sizeof(ElementOutput) * maxElemPerStore >= 32) {
        cm_vrow_store_unordered<ElementOutput, maxElemPerStore, numStore>(o_desc.get(), rO[i], coord);
      } else {
        static_assert(numStore == 1, "Invalid case");
        cm_vrow_store<ElementOutput, maxElemPerStore>(o_desc.get(), rO[i], coord);
      }
    }
#endif
  }

private:
  Params const& params;
};
} // namespace cutlass::flash_attention::collective
