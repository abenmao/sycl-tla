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

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"

#include "xe4_fmha_fwd_softmax_epilogue.hpp"

namespace cutlass::flash_attention::collective {

using namespace cute;

template <
  class ProblemShape_,
  class TileShape_,
  class ElementQ_,
  class ElementK_,
  class ElementV_,
  class ElementS_,
  class ElementP_,
  class ElementAccum_,
  class ElementOutput_,
  class StrideQ_,
  class StrideK_,
  class StrideV_,
  class TiledMmaQK_,
  class TiledMmaPV_,
  class SmemLayoutQ_,
  class SmemLayoutK_,
  class SmemLayoutV_,
  class SmemLayoutOutputAccum_,
  class SmemLayoutOutput_,
  class TMACopyAtomQ_,
  class TMACopyAtomK_,
  class TMACopyAtomV_>
struct CollectiveMmaAttention {

  using ProblemShape = ProblemShape_;

  using TileShape = TileShape_;                                   // <BLK_M_Q, BLK_N_V, BLK_N_QK, BLK_K_QK>
  using TileShapeQK_MNK = decltype(select<0, 2, 3>(TileShape{})); // <BLK_M_Q, BLK_N_QK, BLK_K_QK>
  using TileShapePV_MNK = decltype(select<0, 1, 2>(TileShape{})); // <BLK_M_PV, BLK_N_V, BLK_N_QK>

  using ElementQ = ElementQ_;
  using ElementK = ElementK_;
  using ElementV = ElementV_;
  using ElementS = ElementS_;
  using ElementP = ElementP_;
  using ElementAccum = ElementAccum_;
  using ElementOutput = ElementOutput_;

  using StrideQ = StrideQ_;
  using StrideK = StrideK_;
  using StrideV = StrideV_;

  using TiledMmaQK = TiledMmaQK_;
  using TiledMmaPV = TiledMmaPV_;

  using SmemLayoutQ = SmemLayoutQ_;
  using SmemLayoutK = SmemLayoutK_;
  using SmemLayoutV = SmemLayoutV_;
  using SmemLayoutOutputAccum = SmemLayoutOutputAccum_;
  using SmemLayoutOutput = SmemLayoutOutput_;

  using TMACopyAtomQ = TMACopyAtomQ_;
  using TMACopyAtomK = TMACopyAtomK_;
  using TMACopyAtomV = TMACopyAtomV_;

  // TODO add assert to check unsupported configs

  // number of QO stages per workgroup; tunable based on tile shapes
  static constexpr int NumStageQO = 2;
  static constexpr int NumStageKV = 2;

  using ClusterShape = Shape<_1, _1, _1>;

  // from load q to mma
  using PipelineQ = cutlass::PipelineTmaAsync<NumStageQO>;

  // from load k to mma
  using PipelineK = cutlass::PipelineTmaAsync<NumStageKV>;

  // from load v to mma
  using PipelineV = cutlass::PipelineTmaAsync<NumStageKV>;

  // from mma (QK) to softmax
  using PipelineS = cutlass::PipelineTmaAsync<2>;

  // from mma (PV) to softmax
  using PipelineO = cutlass::PipelineTmaAsync<2>;

  // from softmax to epilogue
  using PipelineEpi = cutlass::PipelineTmaAsync<2>;

  using TMA_Q = decltype(make_tma_copy(
    TMACopyAtomQ{},
    make_tensor(make_gmem_ptr(static_cast<ElementQ const*>(nullptr)),
                repeat_like(StrideQ{}, int32_t(0)),
                StrideQ{}),
    SmemLayoutQ{}(_, _, cute::Int<0>{}),
    select<0, 2>(TileShapeQK_MNK{}),
    _1{}
  ));

  using TMA_K = decltype(make_tma_copy(
    TMACopyAtomK{},
    make_tensor(make_gmem_ptr(static_cast<ElementK const*>(nullptr)),
                repeat_like(StrideK{}, int32_t(0)),
                StrideK{}),
    SmemLayoutK{}(_, _, cute::Int<0>{}),
    select<1, 2>(TileShapeQK_MNK{}),
    _1{}
  ));

  using TMA_V = decltype(make_tma_copy(
    TMACopyAtomV{},
    make_tensor(make_gmem_ptr(static_cast<ElementV const*>(nullptr)),
                repeat_like(StrideV{}, int32_t(0)),
                StrideV{}),
    SmemLayoutV{}(_, _, cute::Int<0>{}),
    select<1, 2>(TileShapePV_MNK{}),
    _1{}
  ));

  using SmemLayoutStageS = decltype(make_layout(
      cute::select<0, 1>(TileShapeQK_MNK{}), GenRowMajor{}));
  using SmemLayoutS = decltype(tile_to_shape(
      SmemLayoutStageS{},
      make_shape(shape<0>(TileShapeQK_MNK{}),
                 shape<1>(TileShapeQK_MNK{}),
                 Int<NumStageQO>{})));

  using SmemLayoutStageP = decltype(make_layout(
      cute::select<0, 1>(TileShapeQK_MNK{}), GenRowMajor{}));
  using SmemLayoutP = decltype(tile_to_shape(
      SmemLayoutStageP{},
      make_shape(shape<0>(TileShapeQK_MNK{}),
                 shape<1>(TileShapeQK_MNK{}),
                 Int<NumStageQO>{})));

  static constexpr size_t SmemAlignment = 512;
  struct SharedStorage
  {
    struct TensorStorage : cute::aligned_struct<SmemAlignment, _0>
    {
      cute::array_aligned<ElementQ, cute::cosize_v<SmemLayoutQ>, SmemAlignment> smem_Q;
      cute::array_aligned<ElementK, cute::cosize_v<SmemLayoutK>, SmemAlignment> smem_K;
      cute::array_aligned<ElementV, cute::cosize_v<SmemLayoutV>, SmemAlignment> smem_V;
      cute::array_aligned<ElementS, cute::cosize_v<SmemLayoutS>, SmemAlignment> smem_S;
      cute::array_aligned<ElementP, cute::cosize_v<SmemLayoutP>, SmemAlignment> smem_P;
      cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutOutputAccum>, SmemAlignment> smem_Oacc;
    };
    struct PipelineStorage {
      // TODO: add AMMA atom with different combinations of .dtm, .atm, and .btm
      cutlass::arch::ClusterTransactionBarrier barrier_dummy;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr uint32_t TmaTransactionBytesQ = sizeof(TensorStorage::smem_Q) / NumStageQO;
  static constexpr uint32_t TmaTransactionBytesK = sizeof(TensorStorage::smem_K) / NumStageKV;
  static constexpr uint32_t TmaTransactionBytesV = sizeof(TensorStorage::smem_V) / NumStageKV;

  // Host side kernel arguments
  struct Arguments {
    ElementQ const* ptr_Q;
    StrideQ dQ;
    ElementK const* ptr_K;
    StrideK dK;
    ElementV const* ptr_V;
    StrideV dV;
  };

  // Device side kernel params
  struct Params {
    ProblemShape problem_shape;
    TMA_Q tma_load_Q;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
  };

  CollectiveMmaAttention() = default;

  static constexpr Params to_underlying_arguments(
    ProblemShape const& problem_shape, 
    Arguments const& args) 
  {
    auto [batch, num_heads, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo] = problem_shape;

    Tensor mQ = make_tensor(
      make_gmem_ptr(args.ptr_Q),
      make_layout(make_shape(seq_len_qo, head_size_qk, batch * num_heads),
                  args.dQ));

    TMA_Q tma_load_Q = make_tma_copy(
      TMACopyAtomQ{},
      mQ, 
      SmemLayoutQ{}(_, _, _0{}),
      select<0, 2>(TileShapeQK_MNK{}), 
      _1{}
    );

    Tensor mK = make_tensor(
      make_gmem_ptr(args.ptr_K),
      make_layout(make_shape(seq_len_kv, head_size_qk, batch * num_heads),
                  args.dK));

    TMA_K tma_load_K = make_tma_copy(
      TMACopyAtomK{},
      mK, 
      SmemLayoutK{}(_, _, _0{}),
      select<1, 2>(TileShapeQK_MNK{}), 
      _1{}
    );

    Tensor mV = make_tensor(
      make_gmem_ptr(args.ptr_V),
      make_layout(make_shape(head_size_vo, seq_len_kv, batch * num_heads),
                  args.dV));

    TMA_V tma_load_V = make_tma_copy(
      TMACopyAtomV{},
      mV, 
      SmemLayoutV{}(_, _, _0{}),
      select<1, 2>(TileShapePV_MNK{}), 
      _1{}
    );

    return {problem_shape, tma_load_Q, tma_load_K, tma_load_V};
  }

  template<class ProblemShape, class TileShape>
  static bool
  can_implement(
    ProblemShape const& problem_shape,
    TileShape const& tile_shape,
    Arguments const& args) {

    bool implementable = true;

    auto seq_len_qo = size<2>(problem_shape);
    auto tile_shape_qo = size<0>(tile_shape);

    implementable = implementable && (seq_len_qo % (tile_shape_qo * 2) == 0);
    if (!(seq_len_qo % (tile_shape_qo * 2) == 0)) {
      CUTLASS_TRACE_HOST("seq_len qo=" << seq_len_qo
        << " must be divisible by (tile_shape QO * 2)=" << (tile_shape_qo * 2)
        << " for QO double buffering\n");
    }
    return implementable;
  }

  template <typename DescTuple, typename BlockCoord>
  CUTLASS_DEVICE void load(
    Params const& params,
    TensorStorage& shared_tensors,
    PipelineStorage& shared_pipelines,
    DescTuple const& tdesc_tuple,
    BlockCoord const& block_coord,
    int const num_kv_tiles, 
    PipelineQ& pipeline_q, PipelineQ::PipelineState& pipeline_q_producer_state,
    PipelineK& pipeline_k, PipelineK::PipelineState& pipeline_k_producer_state,
    PipelineV& pipeline_v, PipelineV::PipelineState& pipeline_v_producer_state)
  {
    bool lane_predicate = cute::elect_one_sync();

    auto [batch, num_heads, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo] = params.problem_shape;

    auto blk_m_coord = get<1>(block_coord); // seq_len_blk_idx
    auto blk_n_coord = get<0>(block_coord); // head_size_blk_idx
    auto batch_coord = get<2>(block_coord); // batch_blk_idx
    auto num_heads_coord = get<3>(block_coord); // num_heads_blk_idx
    auto blk_l_coord = batch_coord * num_heads + num_heads_coord;

    // Initialize matrix descriptor
    auto [tdesc_q, tdesc_k, tdesc_v] = tdesc_tuple;
    params.tma_load_Q.cache_.set_tensor_desc(tdesc_q);
    params.tma_load_K.cache_.set_tensor_desc(tdesc_k);
    params.tma_load_V.cache_.set_tensor_desc(tdesc_v);

    Tensor mQ_mkl = params.tma_load_Q.get_tma_tensor(
      make_shape(seq_len_qo, head_size_qk, batch * num_heads)); // (M,K,L)
    Tensor mK_nkl = params.tma_load_K.get_tma_tensor(
      make_shape(seq_len_kv, head_size_qk, batch * num_heads)); // (N,K,L)
    Tensor mV_nkl = params.tma_load_V.get_tma_tensor(
      make_shape(head_size_vo, seq_len_kv, batch * num_heads)); // (N,K,L)
    
    Tensor gQ = local_tile(mQ_mkl(_, _, blk_l_coord), TileShapeQK_MNK{},
                            make_coord(_, _, blk_n_coord), Step<_1, X, _1>{}); // (BLK_M,BLK_K,q)
    Tensor gK = local_tile(mK_nkl(_, _, blk_l_coord), TileShapeQK_MNK{},
                            make_coord(_, _, blk_n_coord), Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gV = local_tile(mV_nkl(_, _, blk_l_coord), TileShapePV_MNK{},
                            make_coord(_, blk_n_coord, _), Step<X, _1, _1>{}); // (BLK_N,BLK_K,k)

    Tensor sQ = make_tensor(make_smem_ptr(shared_tensors.smem_Q.data()), SmemLayoutQ{}); // (BLK_M,BLK_K,PIPE)
    Tensor sK = make_tensor(make_smem_ptr(shared_tensors.smem_K.data()), SmemLayoutK{}); // (BLK_N,BLK_K,PIPE)
    Tensor sV = make_tensor(make_smem_ptr(shared_tensors.smem_V.data()), SmemLayoutV{}); // (BLK_N,BLK_K,PIPE)

    auto [tQgQ, tQsQ] = tma_partition(params.tma_load_Q, _0{}, Layout<_1>{},
                                      group_modes<0, 2>(sQ), group_modes<0, 2>(gQ)); // (TMA,q), (TMA,PIPE)
    auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{},
                                      group_modes<0, 2>(sK), group_modes<0, 2>(gK)); // (TMA,k), (TMA,PIPE)
    auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{},
                                      group_modes<0, 2>(sV), group_modes<0, 2>(gV)); // (TMA,k), (TMA,PIPE)

    constexpr uint32_t mcast_mask = 0;

    int q0_index = NumStageQO * blk_m_coord;
    int q1_index = NumStageQO * blk_m_coord + 1;

    pipeline_q.producer_acquire(pipeline_q_producer_state);
    copy(params.tma_load_Q.with(
         pipeline_q.producer_get_barrier(pipeline_q_producer_state), mcast_mask),
         tQgQ(_, q0_index), tQsQ(_, pipeline_q_producer_state.index())
    );
    ++pipeline_q_producer_state;
    
    pipeline_k.producer_acquire(pipeline_k_producer_state);
    copy(params.tma_load_K.with(
         pipeline_k.producer_get_barrier(pipeline_k_producer_state), mcast_mask),
         tKgK(_, _0{}), tKsK(_, pipeline_k_producer_state.index())
    );
    ++pipeline_k_producer_state;
    
    pipeline_q.producer_acquire(pipeline_q_producer_state);
    copy(params.tma_load_Q.with(
         pipeline_q.producer_get_barrier(pipeline_q_producer_state), mcast_mask),
         tQgQ(_, q1_index), tQsQ(_, pipeline_q_producer_state.index())
    );
    ++pipeline_q_producer_state;

    pipeline_v.producer_acquire(pipeline_v_producer_state);
    copy(params.tma_load_V.with(
         pipeline_v.producer_get_barrier(pipeline_v_producer_state), mcast_mask),
         tVgV(_, _0{}), tVsV(_, pipeline_v_producer_state.index())
    );
    ++pipeline_v_producer_state;
    
    for (int i = 1; i < num_kv_tiles; ++i) {
      pipeline_k.producer_acquire(pipeline_k_producer_state);
      copy(params.tma_load_K.with(
           pipeline_k.producer_get_barrier(pipeline_k_producer_state), mcast_mask),
           tKgK(_, i), tKsK(_, pipeline_k_producer_state.index())
      );
      ++pipeline_k_producer_state;

      pipeline_v.producer_acquire(pipeline_v_producer_state);
      copy(params.tma_load_V.with(
           pipeline_v.producer_get_barrier(pipeline_v_producer_state), mcast_mask),
           tVgV(_, i), tVsV(_, pipeline_v_producer_state.index())
      );
      ++pipeline_v_producer_state;
    }
  }

  CUTLASS_DEVICE void mma(
    TensorStorage& shared_tensors,
    PipelineStorage& shared_pipelines,
    int const num_kv_tiles,
    PipelineQ& pipeline_q, PipelineQ::PipelineState& pipeline_q_consumer_state,
    PipelineK& pipeline_k, PipelineK::PipelineState& pipeline_k_consumer_state,
    PipelineV& pipeline_v, PipelineV::PipelineState& pipeline_v_consumer_state,
    PipelineS& pipeline_s, PipelineS::PipelineState& pipeline_s_producer_state,
    PipelineO& pipeline_corr, PipelineO::PipelineState& pipeline_corr_producer_state)
  {
    int thread_idx = static_cast<int>(ThreadIdxX());

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;

    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

    Tensor sQ = make_tensor(make_smem_ptr(shared_tensors.smem_Q.data()), SmemLayoutQ{}); // (BLK_M,BLK_K,PIPE)
    Tensor sK = make_tensor(make_smem_ptr(shared_tensors.smem_K.data()), SmemLayoutK{}); // (BLK_N,BLK_K,PIPE)
    Tensor sV = make_tensor(make_smem_ptr(shared_tensors.smem_V.data()), SmemLayoutV{}); // (BLK_N,BLK_K,PIPE)
    Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_S.data()), SmemLayoutS{}); // (BLK_M,BLK_N,PIPE)
    Tensor sP = make_tensor(make_smem_ptr(shared_tensors.smem_P.data()), SmemLayoutP{}); // (BLK_M,BLK_K,PIPE)
    Tensor sOacc = make_tensor(make_smem_ptr(shared_tensors.smem_Oacc.data()), SmemLayoutOutputAccum{}); // (BLK_M,BLK_N,PIPE)

    // Matrix descriptors
    Tensor tSsQ = thr_mma_qk.partition_fragment_A(sQ); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tSsK = thr_mma_qk.partition_fragment_B(sK); // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tOsV = thr_mma_pv.partition_fragment_B(sV); // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tSsS = thr_mma_qk.partition_fragment_C(sS); // (MMA,MMA_M,MMA_N,PIPE)
    Tensor tOsP = thr_mma_pv.partition_fragment_A(sP); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tOsOacc = thr_mma_pv.partition_fragment_C(sOacc); // (MMA,MMA_M,MMA_N)

    constexpr uint64_t qk_mma_ctrl = 0x100;
    uint64_t pv_mma_ctrl_0 = 0x100;
    uint64_t pv_mma_ctrl_1 = 0x100;
    
    constexpr uint32_t mcast_mask = 0;

    constexpr int qk_mma_k_itr = size<2>(tSsQ);
    constexpr int pv_mma_k_itr = size<2>(tOsP);

    pipeline_q.consumer_wait(pipeline_q_consumer_state);
    ++pipeline_q_consumer_state;

    pipeline_k.consumer_wait(pipeline_k_consumer_state);

    pipeline_s.producer_acquire(pipeline_s_producer_state);

    // gemm Q1 * K1 -> S1
    CUTE_UNROLL
    for (int k = 0; k < qk_mma_k_itr; ++k) {
      cute::gemm(tiled_mma_qk.with(
	      AMMA::TrackMethod<AMMA::Tracking::DAB>{},
        qk_mma_ctrl,
        pipeline_s.producer_get_barrier(pipeline_s_producer_state),
        reinterpret_cast<uint64_t*>(&shared_pipelines.barrier_dummy),
        pipeline_k.consumer_get_barrier(pipeline_k_consumer_state),
        mcast_mask, mcast_mask
      ), tSsQ(_, _, k, _0{}), tSsK(_, _, k, pipeline_k_consumer_state.index()), tSsS(_, _, _, _0{}));
    }

    ++pipeline_s_producer_state;

    pipeline_q.consumer_wait(pipeline_q_consumer_state);

    pipeline_s.producer_acquire(pipeline_s_producer_state);

    // gemm Q2 * K1 -> S2
    CUTE_UNROLL
    for (int k = 0; k < qk_mma_k_itr; ++k) {
      cute::gemm(tiled_mma_qk.with(
        AMMA::TrackMethod<AMMA::Tracking::DAB>{},
        qk_mma_ctrl,
        pipeline_s.producer_get_barrier(pipeline_s_producer_state),
        reinterpret_cast<uint64_t*>(&shared_pipelines.barrier_dummy),
        pipeline_k.consumer_get_barrier(pipeline_k_consumer_state),
        mcast_mask, mcast_mask
      ), tSsQ(_, _, k, _1{}), tSsK(_, _, k, pipeline_k_consumer_state.index()), tSsS(_, _, _, _1{}));
    }

    ++pipeline_s_producer_state;

    pipeline_k.consumer_commit(pipeline_k_consumer_state, 2 * qk_mma_k_itr);
    ++pipeline_k_consumer_state;

    pipeline_v.consumer_wait(pipeline_v_consumer_state);
    
    pipeline_corr.producer_acquire(pipeline_corr_producer_state);

    pipeline_s.producer_acquire(pipeline_s_producer_state);

    // gemm P1 * V1 -> O1
    CUTE_UNROLL
    for (int k = 0; k < pv_mma_k_itr; ++k) {
      cute::gemm(tiled_mma_pv.with(
        AMMA::TrackMethod<AMMA::Tracking::DAB>{},
        pv_mma_ctrl_0,
        pipeline_corr.producer_get_barrier(pipeline_corr_producer_state),
        reinterpret_cast<uint64_t*>(&shared_pipelines.barrier_dummy),
        pipeline_v.consumer_get_barrier(pipeline_v_consumer_state),
        mcast_mask, mcast_mask
      ), tOsP(_, _, k, _0{}), tOsV(_, _, k, pipeline_v_consumer_state.index()), tOsOacc(_, _, _, _0{}));

      pv_mma_ctrl_0 = 0x0;
    }

    ++pipeline_corr_producer_state;
    
    for (int i = 1; i < num_kv_tiles; ++i) {
      pipeline_k.consumer_wait(pipeline_k_consumer_state);

      // gemm Q1 * Ki -> S1
      CUTE_UNROLL
      for (int k = 0; k < qk_mma_k_itr; ++k) {
        cute::gemm(tiled_mma_qk.with(
          AMMA::TrackMethod<AMMA::Tracking::DAB>{},
          qk_mma_ctrl,
          pipeline_s.producer_get_barrier(pipeline_s_producer_state),
          reinterpret_cast<uint64_t*>(&shared_pipelines.barrier_dummy),
          pipeline_k.consumer_get_barrier(pipeline_k_consumer_state),
          mcast_mask, mcast_mask
        ), tSsQ(_, _, k, _0{}), tSsK(_, _, k, pipeline_k_consumer_state.index()), tSsS(_, _, _, _0{}));
      }

      ++pipeline_s_producer_state;
      
      pipeline_corr.producer_acquire(pipeline_corr_producer_state);

      pipeline_s.producer_acquire(pipeline_s_producer_state);

      // gemm P2 * V(i-1) -> O2
      CUTE_UNROLL
      for (int k = 0; k < pv_mma_k_itr; ++k) {
        cute::gemm(tiled_mma_pv.with(
          AMMA::TrackMethod<AMMA::Tracking::DAB>{},
          pv_mma_ctrl_1,
          pipeline_corr.producer_get_barrier(pipeline_corr_producer_state),
          reinterpret_cast<uint64_t*>(&shared_pipelines.barrier_dummy),
          pipeline_v.consumer_get_barrier(pipeline_v_consumer_state),
          mcast_mask, mcast_mask
        ), tOsP(_, _, k, _1{}), tOsV(_, _, k, pipeline_v_consumer_state.index()), tOsOacc(_, _, _, _1{}));

        pv_mma_ctrl_1 = 0x0;
      }

      ++pipeline_corr_producer_state;

      pipeline_v.consumer_commit(pipeline_v_consumer_state, 2 * pv_mma_k_itr);
      ++pipeline_v_consumer_state;

      // gemm Q2 * Ki -> S2
      CUTE_UNROLL
      for (int k = 0; k < qk_mma_k_itr; ++k) {
        cute::gemm(tiled_mma_qk.with(
          AMMA::TrackMethod<AMMA::Tracking::DAB>{},
          qk_mma_ctrl,
          pipeline_s.producer_get_barrier(pipeline_s_producer_state),
          reinterpret_cast<uint64_t*>(&shared_pipelines.barrier_dummy),
          pipeline_k.consumer_get_barrier(pipeline_k_consumer_state),
          mcast_mask, mcast_mask
        ), tSsQ(_, _, k, _1{}), tSsK(_, _, k, pipeline_k_consumer_state.index()), tSsS(_, _, _, _1{}));
      }

      ++pipeline_s_producer_state;

      pipeline_k.consumer_commit(pipeline_k_consumer_state, 2 * qk_mma_k_itr);
      ++pipeline_k_consumer_state;

      pipeline_v.consumer_wait(pipeline_v_consumer_state);

      pipeline_corr.producer_acquire(pipeline_corr_producer_state);

      pipeline_s.producer_acquire(pipeline_s_producer_state);

      // gemm P1 * Vi -> O1
      CUTE_UNROLL
      for (int k = 0; k < pv_mma_k_itr; ++k) {
        cute::gemm(tiled_mma_pv.with(
          AMMA::TrackMethod<AMMA::Tracking::DAB>{},
          pv_mma_ctrl_0,
          pipeline_corr.producer_get_barrier(pipeline_corr_producer_state),
          reinterpret_cast<uint64_t*>(&shared_pipelines.barrier_dummy),
          pipeline_v.consumer_get_barrier(pipeline_v_consumer_state),
          mcast_mask, mcast_mask
        ), tOsP(_, _, k, _0{}), tOsV(_, _, k, pipeline_v_consumer_state.index()), tOsOacc(_, _, _, _0{}));
      }

      ++pipeline_corr_producer_state;
    }

    pipeline_corr.producer_acquire(pipeline_corr_producer_state);

    // gemm P2 * Vn -> O2
    CUTE_UNROLL
    for (int k = 0; k < pv_mma_k_itr; ++k) {
      cute::gemm(tiled_mma_pv.with(
        AMMA::TrackMethod<AMMA::Tracking::DAB>{},
        pv_mma_ctrl_1,
        pipeline_corr.producer_get_barrier(pipeline_corr_producer_state),
        reinterpret_cast<uint64_t*>(&shared_pipelines.barrier_dummy),
        pipeline_v.consumer_get_barrier(pipeline_v_consumer_state),
        mcast_mask, mcast_mask
      ), tOsP(_, _, k, _1{}), tOsV(_, _, k, pipeline_v_consumer_state.index()), tOsOacc(_, _, _, _1{}));

      pv_mma_ctrl_1 = 0x0;
    }

    ++pipeline_corr_producer_state;

    pipeline_v.consumer_commit(pipeline_v_consumer_state, 2 * pv_mma_k_itr);
    ++pipeline_v_consumer_state;
  }

  template <typename CollectiveSoftmax, typename EpiTensorStorage>
  CUTLASS_DEVICE void softmax(
    CollectiveSoftmax& collective_softmax,
    TensorStorage& shared_tensors,
    EpiTensorStorage& epi_shared_tensors,
    int const num_kv_tiles,
    uint32_t worker_id,
    PipelineS& pipeline_s, PipelineS::PipelineState& pipeline_s_consumer_state,
    PipelineO& pipeline_o, PipelineO::PipelineState& pipeline_o_consumer_state,
    PipelineEpi& pipeline_epi, PipelineEpi::PipelineState& pipeline_epi_producer_state)
  {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = item.get_sub_group();

    auto tiled_copy_s2r_update = collective_softmax.get_params().tiled_copy_s2r_update;
    auto thr_copy_s2r_update = tiled_copy_s2r_update.get_slice(worker_id);

    auto tiled_copy_r2s_rescale_o = collective_softmax.get_params().tiled_copy_r2s_rescale_o;
    auto thr_copy_r2s_rescale_o = tiled_copy_r2s_rescale_o.get_slice(worker_id);

    auto tiled_copy_r2s_final_rescale_o = collective_softmax.get_params().tiled_copy_r2s_final_rescale_o;
    auto thr_copy_r2s_final_rescale_o = tiled_copy_r2s_final_rescale_o.get_slice(worker_id);

    // Matrix descriptors
    using dtype_packed = uint32_t;
    constexpr auto kv_stride = shape<1>(TileShapeQK_MNK{});
    constexpr auto q_stride = shape<0>(TileShapeQK_MNK{});
    constexpr uint32_t packed_row_size_s = kv_stride * sizeof(ElementS) / sizeof(dtype_packed);
    constexpr uint32_t packed_row_size_p = kv_stride * sizeof(ElementP) / sizeof(dtype_packed);
    constexpr auto slm_bytes_per_s_stage = q_stride * kv_stride * sizeof(ElementS);
    constexpr auto slm_bytes_per_p_stage = q_stride * kv_stride * sizeof(ElementP);

    constexpr auto o_row_len = shape<1>(TileShapePV_MNK{});
    constexpr uint32_t row_size_oacc = o_row_len; // matrix stride in elements
    constexpr uint32_t row_size_o = o_row_len;
    constexpr auto slm_bytes_per_oacc_stage = q_stride * row_size_oacc * sizeof(ElementAccum);
    constexpr auto slm_bytes_per_o_stage = q_stride * row_size_o * sizeof(ElementOutput);

    auto tOsS = matrix_desc_t(shared_tensors.smem_S.data(), packed_row_size_s, slm_matrix_type::type1);
#ifdef USE_LD_ST_MATRIX
    auto tOsP = matrix_desc_t(shared_tensors.smem_P.data(), packed_row_size_p, slm_matrix_type::type1);
#else
    auto tOsP = matrix_desc_t(shared_tensors.smem_P.data(), kv_stride, slm_matrix_type::type1);
#endif
    auto tOsOacc = matrix_desc_t(shared_tensors.smem_Oacc.data(), row_size_oacc, slm_matrix_type::type1);
    auto tOsO = matrix_desc_t(epi_shared_tensors.smem_O.data(), row_size_o, slm_matrix_type::type1);

    auto retiled_layout_sS = cutlass::epilogue::thread::detail::CoreMatrix::retile<ElementS>(SmemLayoutS{});
    Tensor sS_post_process = make_tensor(make_smem_ptr(shared_tensors.smem_S.begin()), retiled_layout_sS);
    Tensor tSR_sS = group_modes<1, 3>(thr_copy_s2r_update.partition_S(sS_post_process));
    
    auto retiled_layout_sOacc = cutlass::epilogue::thread::detail::CoreMatrix::retile<ElementAccum>(SmemLayoutOutputAccum{});
    Tensor sOacc_post_process = make_tensor(make_smem_ptr(shared_tensors.smem_Oacc.begin()), retiled_layout_sOacc);
    Tensor tSR_sOacc = group_modes<1, 3>(thr_copy_r2s_rescale_o.partition_D(sOacc_post_process)); // (VEC,VEC_M,PIPE)

    auto retiled_layout_sO = cutlass::epilogue::thread::detail::CoreMatrix::retile<ElementOutput>(SmemLayoutOutput{});
    Tensor sO_post_process = make_tensor(make_smem_ptr(epi_shared_tensors.smem_O.begin()), retiled_layout_sO);
    Tensor tSR_sO = group_modes<1, 3>(thr_copy_r2s_final_rescale_o.partition_D(sO_post_process)); // (VEC,VEC_M,PIPE)

    constexpr uint32_t total_rows_per_wi = CollectiveSoftmax::TotalRowsPerThread;

    constexpr int SP_tile_N = CUTE_STATIC_V(get<1>(TileShapeQK_MNK{}));
    constexpr int numElemPerThread = SP_tile_N / CollectiveSoftmax::NumThreadPerRow;
    constexpr auto packedNum = sizeof(uint32_t) / sizeof(ElementS);
    constexpr auto packedArrLen = numElemPerThread / packedNum;

    ElementS max_reg[NumStageQO][total_rows_per_wi];
    ElementAccum sum_reg[NumStageQO][total_rows_per_wi];
    ElementAccum exp_reg[total_rows_per_wi];

    for (int stage = 0; stage < NumStageQO; ++stage) {
      for (int i = 0; i < total_rows_per_wi; ++i) {
        max_reg[stage][i] = static_cast<ElementS>(-INFINITY);
        sum_reg[stage][i] = static_cast<ElementAccum>(0);
      }
    }

    pipeline_s.consumer_wait(pipeline_s_consumer_state);
    
    collective_softmax.template update<true>(
      sg, worker_id,
      tSR_sS(_, _, _0{}),
      tOsS,
      max_reg[0],
      sum_reg[0],
      exp_reg,
      tOsP
    );

    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;

    pipeline_s.consumer_wait(pipeline_s_consumer_state);

    collective_softmax.template update<true>(
      sg, worker_id,
      tSR_sS(_, _, _1{}),
      tOsS + slm_bytes_per_s_stage,
      max_reg[1],
      sum_reg[1],
      exp_reg,
      tOsP + slm_bytes_per_p_stage
    );

    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;

    for (int kv_tile = 1; kv_tile < num_kv_tiles; ++kv_tile) {
      pipeline_s.consumer_wait(pipeline_s_consumer_state);
      
      collective_softmax.template update<false>(
        sg, worker_id,
        tSR_sS(_, _, _0{}),
        tOsS,
        max_reg[0],
        sum_reg[0],
        exp_reg,
        tOsP
      );

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      collective_softmax.rescale_O(
        sg, worker_id,
        tSR_sOacc(_, _, _0{}),
        tOsOacc,
        exp_reg
      );

      pipeline_s.consumer_release(pipeline_s_consumer_state);
      ++pipeline_s_consumer_state;

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;

      pipeline_s.consumer_wait(pipeline_s_consumer_state);

      collective_softmax.template update<false>(
        sg, worker_id,
        tSR_sS(_, _, _1{}),
        tOsS + slm_bytes_per_s_stage,
        max_reg[1],
        sum_reg[1],
        exp_reg,
        tOsP + slm_bytes_per_p_stage
      );

      pipeline_o.consumer_wait(pipeline_o_consumer_state);

      collective_softmax.rescale_O(
        sg, worker_id,
        tSR_sOacc(_, _, _1{}),
        tOsOacc + slm_bytes_per_oacc_stage,
        exp_reg
      );

      pipeline_s.consumer_release(pipeline_s_consumer_state);
      ++pipeline_s_consumer_state;

      pipeline_o.consumer_release(pipeline_o_consumer_state);
      ++pipeline_o_consumer_state;
    }

    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    collective_softmax.final_rescale_O(
      sg, worker_id,
      tSR_sOacc(_, _, _0{}),
      tSR_sO(_, _, _0{}),
      tOsOacc,
      tOsO,
      sum_reg[0]
    );

    pipeline_o.consumer_release(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;

    pipeline_epi.producer_commit(pipeline_epi_producer_state, 1);
    ++pipeline_epi_producer_state;

    pipeline_o.consumer_wait(pipeline_o_consumer_state);
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    collective_softmax.final_rescale_O(
      sg, worker_id,
      tSR_sOacc(_, _, _1{}),
      tSR_sO(_, _, _1{}),
      tOsOacc + slm_bytes_per_oacc_stage,
      tOsO + slm_bytes_per_o_stage,
      sum_reg[1]
    );

    pipeline_o.consumer_release(pipeline_o_consumer_state);
    ++pipeline_o_consumer_state;

    pipeline_epi.producer_commit(pipeline_epi_producer_state, 1);
    ++pipeline_epi_producer_state;
  }
};
} // namespace cutlass::flash_attention::collective
