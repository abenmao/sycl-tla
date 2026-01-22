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
#include "cutlass/pipeline/xe4_pipeline.hpp"
#include "cutlass/epilogue/thread/xe4_detail.hpp"

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
  class TMACopyAtomV_,
  uint32_t PipelineStages_,
  uint32_t PipelineStagesQ_>
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

  static constexpr uint32_t PipelineStages = PipelineStages_;
  using MainloopPipeline = cutlass::PipelineTmaAsync<PipelineStages>;
  using PipelineState = typename cutlass::PipelineState<PipelineStages>;

  static constexpr uint32_t PipelineStagesQ = PipelineStagesQ_;
  using MainloopPipelineQ = cutlass::PipelineTmaAsync<PipelineStagesQ>;
  using PipelineStateQ = typename cutlass::PipelineState<PipelineStagesQ>;

  static constexpr int NumProducerWarps = 1;
  static constexpr int NumMMAWarps = 1;

  static constexpr int NumControlerWarps = 4;

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
  static constexpr uint32_t sg_size = cutlass::NumThreadsPerWarp;
  static constexpr int StagesSP = PipelineStages;

  using SmemLayoutStageS = decltype(make_layout(
      cute::select<0, 1>(TileShapeQK_MNK{}), GenRowMajor{}));
  using SmemLayoutS = decltype(tile_to_shape(
      SmemLayoutStageS{},
      make_shape(shape<0>(TileShapeQK_MNK{}),
                 shape<1>(TileShapeQK_MNK{}),
                 Int<StagesSP>{})));

  using SmemLayoutStageP = decltype(make_layout(
      cute::select<0, 1>(TileShapeQK_MNK{}), GenRowMajor{}));
  using SmemLayoutP = decltype(tile_to_shape(
      SmemLayoutStageP{},
      make_shape(shape<0>(TileShapeQK_MNK{}),
                 shape<1>(TileShapeQK_MNK{}),
                 Int<StagesSP>{})));

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
      typename MainloopPipelineQ::SharedStorage storage_Q;

      typename MainloopPipeline::SharedStorage storage_K;
      typename MainloopPipeline::SharedStorage storage_V;

      typename MainloopPipeline::SharedStorage storage_S;
      typename MainloopPipeline::SharedStorage storage_P;

      cutlass::arch::ClusterTransactionBarrier barrier_Q;
      cutlass::arch::ClusterTransactionBarrier barrier_O;
      cutlass::arch::ClusterTransactionBarrier barrier_O_empty;

      // TODO: Add async_gmma PISA with only .dtm.btm, without .atm
      cutlass::arch::ClusterTransactionBarrier barrier_q_dummy;
    };
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr uint32_t TmaTransactionBytesQ = sizeof(TensorStorage::smem_Q) / PipelineStagesQ;
  static constexpr uint32_t TmaTransactionBytesK = sizeof(TensorStorage::smem_K) / PipelineStages;
  static constexpr uint32_t TmaTransactionBytesV = sizeof(TensorStorage::smem_V) / PipelineStages;

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

  template <typename DescTuple, typename BlockCoord>
  CUTLASS_DEVICE void load(
    Params const& params,
    TensorStorage& shared_tensors,
    PipelineStorage& shared_pipelines,
    DescTuple const& tdesc_tuple,
    BlockCoord const& block_coord,
    BlockCoord const& block_coord_next,
    int const num_kv_tiles, 
    MainloopPipelineQ pipeline_q, PipelineStateQ& smem_pipe_write_q,
    MainloopPipeline pipeline_k, PipelineState& smem_pipe_write_k,
    MainloopPipeline pipeline_v, PipelineState& smem_pipe_write_v,
    bool is_first_wave, bool is_last_wave)
  {
    bool lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      auto [batch, num_heads, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo] = params.problem_shape;

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
    
      Tensor sQ = make_tensor(make_smem_ptr(shared_tensors.smem_Q.data()), SmemLayoutQ{}); // (BLK_M,BLK_K,PIPE_Q)
      Tensor sK = make_tensor(make_smem_ptr(shared_tensors.smem_K.data()), SmemLayoutK{}); // (BLK_N,BLK_K,PIPE)
      Tensor sV = make_tensor(make_smem_ptr(shared_tensors.smem_V.data()), SmemLayoutV{}); // (BLK_N,BLK_K,PIPE)

      auto blk_m_coord = get<1>(block_coord); // seq_len_blk_idx
      auto blk_n_coord = get<0>(block_coord); // head_size_blk_idx
      auto batch_coord = get<2>(block_coord); // batch_blk_idx
      auto num_heads_coord = get<3>(block_coord); // num_heads_blk_idx
      auto blk_l_coord = batch_coord * num_heads + num_heads_coord;

      Tensor gQ = local_tile(mQ_mkl(_, _, blk_l_coord), TileShapeQK_MNK{},
                             make_coord(blk_m_coord, _, blk_n_coord), Step<_1, X, _1>{}); // (BLK_M,BLK_K)
      Tensor gK = local_tile(mK_nkl(_, _, blk_l_coord), TileShapeQK_MNK{},
                             make_coord(_, _, blk_n_coord), Step<X, _1, _1>{});           // (BLK_N,BLK_K,k)
      Tensor gV = local_tile(mV_nkl(_, _, blk_l_coord), TileShapePV_MNK{},
                             make_coord(_, blk_n_coord, _), Step<X, _1, _1>{});           // (BLK_N,BLK_K,k)

      auto [tQgQ, tQsQ] = tma_partition(params.tma_load_Q, _0{}, Layout<_1>{},
                                        group_modes<0, 2>(sQ), group_modes<0, 2>(gQ)); // (TMA), (TMA,PIPE_Q)
      auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{},
                                        group_modes<0, 2>(sK), group_modes<0, 2>(gK)); // (TMA,k), (TMA,PIPE)
      auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{},
                                        group_modes<0, 2>(sV), group_modes<0, 2>(gV)); // (TMA,k), (TMA,PIPE)


      // load Q and K tile for the iter_0 of the first wave
      if (is_first_wave) {
        // shared_pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
        pipeline_q.producer_acquire(smem_pipe_write_q);
        copy(params.tma_load_Q.with(
            pipeline_q.producer_get_barrier(smem_pipe_write_q), 0),
            tQgQ, tQsQ(_, smem_pipe_write_q.index())
        );
        ++smem_pipe_write_q;

        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(params.tma_load_K.with(
              pipeline_k.producer_get_barrier(smem_pipe_write_k), 0),
              tKgK(_, 0), tKsK(_, smem_pipe_write_k.index())
        );
        ++smem_pipe_write_k;
      }

      for (int i = 1; i < num_kv_tiles; ++i) {
        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(params.tma_load_K.with(
             pipeline_k.producer_get_barrier(smem_pipe_write_k), 0),
             tKgK(_, i), tKsK(_, smem_pipe_write_k.index())
        );
        ++smem_pipe_write_k;
        
        pipeline_v.producer_acquire(smem_pipe_write_v);
        copy(params.tma_load_V.with(
             pipeline_v.producer_get_barrier(smem_pipe_write_v), 0),
             tVgV(_, i - 1), tVsV(_, smem_pipe_write_v.index())
        );
        ++smem_pipe_write_v;
      }

      // load Q and K tile for the iter_0 of the next wave
      if (!is_last_wave) {
        auto blk_m_coord_next = get<1>(block_coord_next); // seq_len_blk_idx
        auto blk_n_coord_next = get<0>(block_coord_next); // head_size_blk_idx
        auto batch_coord_next = get<2>(block_coord_next); // batch_blk_idx
        auto num_heads_coord_next = get<3>(block_coord_next); // num_heads_blk_idx
        auto blk_l_coord_next = batch_coord_next * num_heads + num_heads_coord_next;

        Tensor gQ_next = local_tile(mQ_mkl(_, _, blk_l_coord_next), TileShapeQK_MNK{},
                                make_coord(blk_m_coord_next, _, blk_n_coord_next), Step<_1, X, _1>{}); // (BLK_M,BLK_K)
        Tensor gK_next = local_tile(mK_nkl(_, _, blk_l_coord_next), TileShapeQK_MNK{},
                                make_coord(_, _, blk_n_coord_next), Step<X, _1, _1>{});           // (BLK_N,BLK_K,k)

        auto [tQgQ_next, tQsQ_next] = tma_partition(params.tma_load_Q, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sQ), group_modes<0, 2>(gQ_next)); // (TMA), (TMA,PIPE_Q)
        auto [tKgK_next, tKsK_next] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sK), group_modes<0, 2>(gK_next)); // (TMA,k), (TMA,PIPE)
        
        pipeline_q.producer_acquire(smem_pipe_write_q);
        copy(params.tma_load_Q.with(
            pipeline_q.producer_get_barrier(smem_pipe_write_q), 0),
            tQgQ_next, tQsQ(_, smem_pipe_write_q.index())
        );
        ++smem_pipe_write_q;

        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(params.tma_load_K.with(
              pipeline_k.producer_get_barrier(smem_pipe_write_k), 0),
              tKgK_next(_, 0), tKsK(_, smem_pipe_write_k.index())
        );
        ++smem_pipe_write_k;
      }

      pipeline_v.producer_acquire(smem_pipe_write_v);
      copy(params.tma_load_V.with(
            pipeline_v.producer_get_barrier(smem_pipe_write_v), 0),
            tVgV(_, num_kv_tiles - 1), tVsV(_, smem_pipe_write_v.index())
      );
      ++smem_pipe_write_v;
    }
  }

  template <typename EpilogueTensorStorage, typename CollectiveSoftmax>
  CUTLASS_DEVICE void mma(
    Params const& params,
    TensorStorage& shared_tensors,
    EpilogueTensorStorage& epi_shared_tensors,
    PipelineStorage& shared_pipelines,
    int const num_kv_tiles,
    MainloopPipelineQ pipeline_q, PipelineStateQ& smem_pipe_read_q,
    MainloopPipeline pipeline_k, PipelineState& smem_pipe_read_k,
    MainloopPipeline pipeline_v, PipelineState& smem_pipe_read_v,
    MainloopPipeline pipeline_s, PipelineState& smem_pipe_write_s,
    MainloopPipeline pipeline_p, PipelineState& smem_pipe_read_p,
    CollectiveSoftmax& collective_softmax,
    bool is_first_wave, bool is_last_wave)
  {
    bool lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
      auto sg = item.get_sub_group();
      uint32_t item_id = item.get_local_linear_id();
      uint32_t sg_id = get_sg_id();
      int thread_idx = static_cast<int>(ThreadIdxX());

      TiledMmaQK tiled_mma_qk;
      TiledMmaPV tiled_mma_pv;

      auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
      auto thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

      Tensor sQ = make_tensor(make_smem_ptr(shared_tensors.smem_Q.data()), SmemLayoutQ{});      // (BLK_M,BLK_K,PIPE_Q)
      Tensor sK = make_tensor(make_smem_ptr(shared_tensors.smem_K.data()), SmemLayoutK{});      // (BLK_N,BLK_K,PIPE)
      Tensor sV = make_tensor(make_smem_ptr(shared_tensors.smem_V.data()), SmemLayoutV{});      // (BLK_N,BLK_K,PIPE)
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_S.data()), SmemLayoutS{});  // (BLK_M,BLK_N,PIPE)
      Tensor sP = make_tensor(make_smem_ptr(shared_tensors.smem_P.data()), SmemLayoutP{});  // (BLK_M,BLK_N,PIPE)
      Tensor sOacc = make_tensor(make_smem_ptr(shared_tensors.smem_Oacc.data()), SmemLayoutOutputAccum{});  // (BLK_M,BLK_N)
      Tensor sO = make_tensor(make_smem_ptr(epi_shared_tensors.smem_O.data()), SmemLayoutOutput{}); // (BLK_M,BLK_N)

      // Matrix descriptors
      Tensor tSsQ = thr_mma_qk.partition_fragment_A(sQ); // (MMA,MMA_M,MMA_K,PIPE_Q)
      Tensor tSsK = thr_mma_qk.partition_fragment_B(sK); // (MMA,MMA_N,MMA_K,PIPE)
      Tensor tOsP = thr_mma_pv.partition_fragment_A(sP); // (MMA,MMA_M,MMA_K,PIPE)
      Tensor tOsV = thr_mma_pv.partition_fragment_B(sV); // (MMA,MMA_N,MMA_K,PIPE)
      Tensor tSsS = thr_mma_qk.partition_fragment_C(sS); // (MMA,MMA_M,MMA_N,PIPE)
      Tensor tOsOacc = thr_mma_pv.partition_fragment_C(sOacc); // (MMA,MMA_M,MMA_N)
      Tensor tOsO = thr_mma_pv.partition_fragment_C(sO); // (MMA,MMA_M,MMA_N)

      constexpr uint64_t qk_mma_ctrl = 0x100;
      uint64_t pv_mma_ctrl = 0x100;
      constexpr uint32_t mcast_mask = 0;

      // QK for the iter_0 of the first wave
      if (is_first_wave) {
        // shared_pipelines.barrier_Q.wait(/*phase=*/0);
        pipeline_q.consumer_wait(smem_pipe_read_q);
        pipeline_k.consumer_wait(smem_pipe_read_k);
        pipeline_s.producer_acquire(smem_pipe_write_s);

        // Compute S = Q * K^T
        cute::gemm(tiled_mma_qk.with(
          AMMA::TrackMethod<AMMA::Tracking::DAB>{},
          qk_mma_ctrl,
          pipeline_s.producer_get_barrier(smem_pipe_write_s),
          pipeline_q.consumer_get_barrier(smem_pipe_read_q),
          pipeline_k.consumer_get_barrier(smem_pipe_read_k),
          mcast_mask, mcast_mask
        ), tSsQ(_, _, _, smem_pipe_read_q.index()), tSsK(_, _, _, smem_pipe_read_k.index()), tSsS(_, _, _, smem_pipe_write_s.index()));
        
        pipeline_k.consumer_commit(smem_pipe_read_k, 1);

        ++smem_pipe_read_k;
        ++smem_pipe_write_s;
      }

      for (int i = 1; i < num_kv_tiles; ++i) {
        pipeline_k.consumer_wait(smem_pipe_read_k);
        pipeline_s.producer_acquire(smem_pipe_write_s);

        // Compute S = Q * K^T
        cute::gemm(tiled_mma_qk.with(
          AMMA::TrackMethod<AMMA::Tracking::DAB>{},
          qk_mma_ctrl,
          pipeline_s.producer_get_barrier(smem_pipe_write_s),
          pipeline_q.consumer_get_barrier(smem_pipe_read_q),
          pipeline_k.consumer_get_barrier(smem_pipe_read_k),
          mcast_mask, mcast_mask
        ), tSsQ(_, _, _, smem_pipe_read_q.index()), tSsK(_, _, _, smem_pipe_read_k.index()), tSsS(_, _, _, smem_pipe_write_s.index()));

        pipeline_k.consumer_commit(smem_pipe_read_k, 1);

        ++smem_pipe_read_k;
        ++smem_pipe_write_s;

        // Finish the use of Q tile for current wave
        if (i == num_kv_tiles - 1) {
          pipeline_q.consumer_commit(smem_pipe_read_q, num_kv_tiles);
          ++smem_pipe_read_q;
        }

        pipeline_v.consumer_wait(smem_pipe_read_v);
        pipeline_p.consumer_wait(smem_pipe_read_p);

        shared_pipelines.barrier_O_empty.wait(/*phase*/i % 2);
        
        // Compute O = P * V
        auto K = size<2>(tOsP);
        CUTE_UNROLL
        for (int k = 0; k < K; ++k) {
          cute::gemm(tiled_mma_pv.with(
            AMMA::TrackMethod<AMMA::Tracking::DAB>{},
            pv_mma_ctrl,
            reinterpret_cast<uint64_t*>(&shared_pipelines.barrier_O),
            pipeline_p.consumer_get_barrier(smem_pipe_read_p),
            pipeline_v.consumer_get_barrier(smem_pipe_read_v),
            mcast_mask, mcast_mask
          ), tOsP(_, _, k, smem_pipe_read_p.index()), tOsV(_, _, k, smem_pipe_read_v.index()), tOsOacc);

          pv_mma_ctrl = 0x0;
        }

        pipeline_v.consumer_commit(smem_pipe_read_v, K);
        pipeline_p.consumer_commit(smem_pipe_read_p, K);
        shared_pipelines.barrier_O.arrive_and_expect_tx(K);

        ++smem_pipe_read_v;
        ++smem_pipe_read_p;
      }

      // QK for the iter_0 of the next wave
      if (!is_last_wave) {
        // shared_pipelines.barrier_Q.wait(/*phase=*/0);
        pipeline_q.consumer_wait(smem_pipe_read_q);
        pipeline_k.consumer_wait(smem_pipe_read_k);
        pipeline_s.producer_acquire(smem_pipe_write_s);

        // Compute S = Q * K^T
        cute::gemm(tiled_mma_qk.with(
          AMMA::TrackMethod<AMMA::Tracking::DAB>{},
          qk_mma_ctrl,
          pipeline_s.producer_get_barrier(smem_pipe_write_s),
          pipeline_q.consumer_get_barrier(smem_pipe_read_q),
          pipeline_k.consumer_get_barrier(smem_pipe_read_k),
          mcast_mask, mcast_mask
        ), tSsQ(_, _, _, smem_pipe_read_q.index()), tSsK(_, _, _, smem_pipe_read_k.index()), tSsS(_, _, _, smem_pipe_write_s.index()));
        
        pipeline_k.consumer_commit(smem_pipe_read_k, 1);

        ++smem_pipe_read_k;
        ++smem_pipe_write_s;
      }

      pipeline_v.consumer_wait(smem_pipe_read_v);
      pipeline_p.consumer_wait(smem_pipe_read_p);
      
      shared_pipelines.barrier_O_empty.wait(/*phase*/num_kv_tiles % 2);

      // Compute O = P * V
      auto K = size<2>(tOsP);
      CUTE_UNROLL
      for (int k = 0; k < K; ++k) {
        cute::gemm(tiled_mma_pv.with(
          AMMA::TrackMethod<AMMA::Tracking::DAB>{},
          pv_mma_ctrl,
          reinterpret_cast<uint64_t*>(&shared_pipelines.barrier_O),
          pipeline_p.consumer_get_barrier(smem_pipe_read_p),
          pipeline_v.consumer_get_barrier(smem_pipe_read_v),
          mcast_mask, mcast_mask
        ), tOsP(_, _, k, smem_pipe_read_p.index()), tOsV(_, _, k, smem_pipe_read_v.index()), tOsOacc);

        pv_mma_ctrl = 0x0;
      }

      pipeline_v.consumer_commit(smem_pipe_read_v, K);
      pipeline_p.consumer_commit(smem_pipe_read_p, K);
      shared_pipelines.barrier_O.arrive_and_expect_tx(K);

      ++smem_pipe_read_v;
      ++smem_pipe_read_p;
    }
  }

  template <typename EpilogueTensorStorage, typename EpiloguePipelineStorage, typename CollectiveSoftmax,
            typename FragMax, typename FragExp, typename FragSum>
  CUTLASS_DEVICE void softmax(
    Params const& params,
    TensorStorage& shared_tensors,
    EpilogueTensorStorage& epi_shared_tensors,
    PipelineStorage& shared_pipelines,
    EpiloguePipelineStorage& epi_shared_pipelines,
    int const num_kv_tiles,
    MainloopPipeline pipeline_s, PipelineState& smem_pipe_read_s,
    MainloopPipeline pipeline_p, PipelineState& smem_pipe_write_p,
    CollectiveSoftmax& collective_softmax,
    FragMax& max_reg, FragExp& exp_reg, FragSum& sum_reg,
    bool is_first_wave, bool is_last_wave)
  {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto sg = item.get_sub_group();
    uint32_t sg_id = get_sg_id();

    uint32_t item_id = item.get_local_linear_id();
    int thread_idx = static_cast<int>(ThreadIdxX());
    
    // also minus the mma and store warp
    uint32_t worker_id = item_id - sg_size * NumControlerWarps;
    bool lane_predicate = cute::elect_one_sync();

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;

    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);
    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);

    auto tiled_copy_s2r_update = collective_softmax.get_params().tiled_copy_s2r_update;
    auto thr_copy_s2r_update = tiled_copy_s2r_update.get_slice(worker_id);

    auto tiled_copy_r2s_rescale_o = collective_softmax.get_params().tiled_copy_r2s_rescale_o;
    auto thr_copy_r2s_rescale_o = tiled_copy_r2s_rescale_o.get_slice(worker_id);

    auto tiled_copy_r2s_final_rescale_o = collective_softmax.get_params().tiled_copy_r2s_final_rescale_o;
    auto thr_copy_r2s_final_rescale_o = tiled_copy_r2s_final_rescale_o.get_slice(worker_id);

    Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_S.data()), SmemLayoutS{});  // (BLK_M,BLK_N,PIPE)
    Tensor sP = make_tensor(make_smem_ptr(shared_tensors.smem_P.data()), SmemLayoutP{});  // (BLK_M,BLK_N,PIPE)
    Tensor sOac = make_tensor(make_smem_ptr(shared_tensors.smem_Oacc.data()), SmemLayoutOutputAccum{});  // (BLK_M,BLK_N)

    // Matrix descriptors
    using dtype_packed = uint32_t;
    constexpr auto kv_stride = shape<1>(TileShapeQK_MNK{});
    constexpr auto q_stride = shape<0>(TileShapeQK_MNK{});
    constexpr auto slm_bytes_per_s_stage = q_stride * kv_stride * sizeof(ElementS);
    constexpr auto slm_bytes_per_p_stage = q_stride * kv_stride * sizeof(ElementP);
    constexpr uint32_t packed_row_size_s = kv_stride * sizeof(ElementS) / sizeof(dtype_packed);
    constexpr uint32_t packed_row_size_p = kv_stride * sizeof(ElementP) / sizeof(dtype_packed);
    auto tOsS = matrix_desc_t(shared_tensors.smem_S.data(), packed_row_size_s, slm_matrix_type::type1);
#ifdef USE_LD_ST_MATRIX
    auto tOsP = matrix_desc_t(shared_tensors.smem_P.data(), packed_row_size_p, slm_matrix_type::type1);
#else
    auto tOsP = matrix_desc_t(shared_tensors.smem_P.data(), kv_stride, slm_matrix_type::type1);
#endif

    constexpr auto o_row_len = shape<1>(TileShapePV_MNK{});
    constexpr uint32_t row_size_oacc = o_row_len; // matrix stride in elements
    constexpr uint32_t row_size_o = o_row_len;
    auto tOsOacc = matrix_desc_t(shared_tensors.smem_Oacc.data(), row_size_oacc, slm_matrix_type::type1);
    auto tOsO = matrix_desc_t(epi_shared_tensors.smem_O.data(), row_size_o, slm_matrix_type::type1);

    auto retiled_layout_sS = cutlass::epilogue::thread::detail::CoreMatrix::retile<ElementS>(SmemLayoutS{});
    Tensor sS_post_process = make_tensor(make_smem_ptr(shared_tensors.smem_S.begin()), retiled_layout_sS);
    Tensor tSR_sS = group_modes<1, 3>(thr_copy_s2r_update.partition_S(sS_post_process));

    auto retiled_layout_sOacc = cutlass::epilogue::thread::detail::CoreMatrix::retile_2d<ElementAccum>(SmemLayoutOutputAccum{});
    Tensor sOacc_post_process = make_tensor(make_smem_ptr(shared_tensors.smem_Oacc.begin()), retiled_layout_sOacc);
    Tensor tSR_sOacc = group_modes<1, -1>(thr_copy_r2s_rescale_o.partition_D(sOacc_post_process));

    auto retiled_layout_sO = cutlass::epilogue::thread::detail::CoreMatrix::retile_2d<ElementOutput>(SmemLayoutOutput{});
    Tensor sO_post_process = make_tensor(make_smem_ptr(epi_shared_tensors.smem_O.begin()), retiled_layout_sO);
    Tensor tSR_sO = group_modes<1, -1>(thr_copy_r2s_final_rescale_o.partition_D(sO_post_process));

    constexpr uint32_t total_rows_per_wi = CollectiveSoftmax::TotalRowsPerThread;

    // softmax for the iter_0 of the first wave
    if (is_first_wave) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < total_rows_per_wi; ++i) {
        max_reg[i] = -INFINITY;
        sum_reg[i] = ElementAccum(0);
      }

      pipeline_s.consumer_wait(smem_pipe_read_s);
      pipeline_p.producer_acquire(smem_pipe_write_p);

      collective_softmax.template update</*init=*/true>(sg, worker_id, 
                                                        tSR_sS(_, _, smem_pipe_read_s.index()), 
                                                        // *tOsS(_, _, _, smem_pipe_read_s.index()).data(),
                                                        tOsS + smem_pipe_read_s.index() * slm_bytes_per_s_stage,
                                                        max_reg, sum_reg, exp_reg, 
                                                        // *tOsP(_, _, _, smem_pipe_write_p.index()).data(),
                                                        tOsP + smem_pipe_write_p.index() * slm_bytes_per_p_stage);

      pipeline_s.consumer_release(smem_pipe_read_s);
      pipeline_p.producer_commit(smem_pipe_write_p, 1);

      ++smem_pipe_read_s;
      ++smem_pipe_write_p;
    }

    for (int i = 1; i < num_kv_tiles; ++i) {
      pipeline_s.consumer_wait(smem_pipe_read_s);
      pipeline_p.producer_acquire(smem_pipe_write_p);

      // softmax for iter_i
      collective_softmax.template update</*init=*/false>(sg, worker_id, 
                                                         tSR_sS(_, _, smem_pipe_read_s.index()), 
                                                        //  *tOsS(_, _, _, smem_pipe_read_s.index()).data(),
                                                         tOsS + smem_pipe_read_s.index() * slm_bytes_per_s_stage,
                                                         max_reg, sum_reg, exp_reg, 
                                                        //  *tOsP(_, _, _, smem_pipe_write_p.index()).data(),
                                                         tOsP + smem_pipe_write_p.index() * slm_bytes_per_p_stage);

      // rescale O for iter_i-1
      shared_pipelines.barrier_O.wait(/*phase=*/1 - (i % 2)); // phase = (i-1)%2
      collective_softmax.rescale_O(sg, worker_id, tSR_sOacc, tOsOacc, exp_reg);

      shared_pipelines.barrier_O_empty.arrive();
      pipeline_p.producer_commit(smem_pipe_write_p, 1);
      pipeline_s.consumer_release(smem_pipe_read_s);

      ++smem_pipe_read_s;
      ++smem_pipe_write_p;
    }

    // softmax for the iter_0 of the next wave
    Tensor sum_reg_next = make_tensor<ElementAccum>(Shape<Int<total_rows_per_wi>>{});
    if (!is_last_wave) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < total_rows_per_wi; ++i) {
        max_reg[i] = -INFINITY;
        sum_reg_next[i] = ElementAccum(0);
      }

      pipeline_s.consumer_wait(smem_pipe_read_s);
      pipeline_p.producer_acquire(smem_pipe_write_p);

      collective_softmax.template update</*init=*/true>(sg, worker_id, 
                                                        tSR_sS(_, _, smem_pipe_read_s.index()), 
                                                        // *tOsS(_, _, _, smem_pipe_read_s.index()).data(),
                                                        tOsS + smem_pipe_read_s.index() * slm_bytes_per_s_stage,
                                                        max_reg, sum_reg_next, exp_reg, 
                                                        // *tOsP(_, _, _, smem_pipe_write_p.index()).data(),
                                                        tOsP + smem_pipe_write_p.index() * slm_bytes_per_p_stage);

      pipeline_s.consumer_release(smem_pipe_read_s);
      pipeline_p.producer_commit(smem_pipe_write_p, 1);

      ++smem_pipe_read_s;
      ++smem_pipe_write_p;
    }

    shared_pipelines.barrier_O.wait(/*phase=*/(num_kv_tiles - 1) % 2);

    collective_softmax.final_rescale_O(sg, worker_id, tSR_sOacc, tSR_sO, 
                                       tOsOacc, tOsO, 
                                       sum_reg);

    if (!is_last_wave) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < total_rows_per_wi; ++i) {
        sum_reg[i] = sum_reg_next[i];
      }
    }

    shared_pipelines.barrier_O_empty.arrive();
    epi_shared_pipelines.barrier_O_final.arrive();
  }

};
} // namespace cutlass::flash_attention::collective
