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

template <
  class ProblemShape_,
  class TileShape_,
  class ElementOutput_, 
  class StrideO_, 
  class SmemLayoutOutput_,
  class TMACopyAtomO_>
class CollectiveEpilogueAttention {
public:

  using ProblemShape = ProblemShape_;

  using TileShape = TileShape_;
  using TileShapePV_MNK = decltype(select<0, 1, 2>(TileShape{})); // <BLK_M_PV, BLK_N_V, BLK_N_QK>

  using ElementOutput = ElementOutput_;
  using StrideO = StrideO_;
  using SmemLayoutOutput = SmemLayoutOutput_;

  using TMACopyAtomO = TMACopyAtomO_;

  using TMA_O = decltype(make_tma_copy(
    TMACopyAtomO{},
    make_tensor(make_gmem_ptr(static_cast<ElementOutput const*>(nullptr)),
                repeat_like(StrideO{}, int32_t(0)), 
                StrideO{}),
    SmemLayoutOutput{},
    select<0, 1>(TileShapePV_MNK{}),
    _1{}
  ));

  static constexpr size_t SmemAlignment = 512;
  struct SharedStorage
  {
    struct TensorStorage : cute::aligned_struct<SmemAlignment, _0>
    {
      cute::array_aligned<ElementOutput, cute::cosize_v<SmemLayoutOutput>, SmemAlignment> smem_O;
    };
    struct PipelineStorage {
      cutlass::arch::ClusterTransactionBarrier barrier_O;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr uint32_t TmaTransactionBytesO = sizeof(TensorStorage::smem_O);

  // Host side epilogue arguments
  struct Arguments {
    ElementOutput const* ptr_O;
    StrideO dO;
  };

  // Device side epilogue params
  struct Params {
    ProblemShape problem_shape;
    TMA_O tma_store_O;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
    ProblemShape const& problem_shape,
    Arguments const& args) 
  {
    auto [batch, num_heads, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo] = problem_shape;

    Tensor mO = make_tensor(
      make_gmem_ptr(args.ptr_O),
      make_layout(make_shape(seq_len_qo, head_size_vo, batch * num_heads),
                  args.dO));
    
    TMA_O tma_store_O = make_tma_copy(
      TMACopyAtomO{},
      mO,
      SmemLayoutOutput{},
      select<0, 1>(TileShapePV_MNK{}),
      _1{}
    );

    return {problem_shape, tma_store_O};
  }

  CUTLASS_HOST_DEVICE
  CollectiveEpilogueAttention(Params const& params_) : params(params_) {}

  template <typename TensorStorage, typename DescTuple, typename BlockCoord>
  CUTLASS_DEVICE void store(
    Params const& params,
    TensorStorage& shared_tensors,
    PipelineStorage& shared_pipelines,
    DescTuple const& tdesc_tuple,
    BlockCoord const& block_coord)
  {
    
    bool lane_predicate = cute::elect_one_sync();
    uint32_t sg_id = get_sg_id();

    if (lane_predicate && sg_id == 2) {
      auto [batch, num_heads, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo] = params.problem_shape;

      auto blk_m_coord = get<1>(block_coord); // seq_len_blk_idx
      auto blk_n_coord = get<0>(block_coord); // head_size_blk_idx
      auto batch_coord = get<2>(block_coord); // batch_blk_idx
      auto num_heads_coord = get<3>(block_coord); // num_heads_blk_idx
      auto blk_l_coord = batch_coord * num_heads + num_heads_coord;
   
      // Initialize matrix descriptor
      auto [tdesc_o] = tdesc_tuple;
      params.tma_store_O.cache_.set_tensor_desc(tdesc_o);

      Tensor mO_mnl = params.tma_store_O.get_tma_tensor(
        make_shape(seq_len_qo, head_size_vo, batch * num_heads)); // (M,N,L)

      Tensor gO = local_tile(mO_mnl(_, _, blk_l_coord), TileShapePV_MNK{},
        make_coord(blk_m_coord, blk_n_coord, _), Step<_1, _1, X>{}); // (BLK_M,BLK_N)

      Tensor sO = make_tensor(make_smem_ptr(shared_tensors.smem_O.data()), SmemLayoutOutput{}); // (BLK_M,BLK_N)

      auto [tOgO, tOsO] = tma_partition(params.tma_store_O, _0{}, Layout<_1>{},
                                        group_modes<0, 2>(sO), group_modes<0, 2>(gO)); // (TMA), (TMA)

      shared_pipelines.barrier_O.arrive_and_expect_tx(TmaTransactionBytesO);
      copy(params.tma_store_O.with(
           reinterpret_cast<uint64_t*>(&shared_pipelines.barrier_O), 0),
           tOsO, tOgO);
      shared_pipelines.barrier_O.wait(0);
    }
  }

private:
  Params const &params;
};

} // namespace cutlass::flash_attention::collective
