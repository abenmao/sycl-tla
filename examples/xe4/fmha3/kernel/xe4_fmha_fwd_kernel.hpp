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
#include "xe4_tile_scheduler.hpp"
#include "../collective/xe4_fmha_fwd_mainloop.hpp"

namespace cutlass::flash_attention::kernel {

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveSoftmaxEpilogue_,
  class CollectiveEpilogue_,
  class TileScheduler_>
class GemmUniversalAttention {
public:

  using ProblemShape = ProblemShape_;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TileShapeQK_MNK = typename CollectiveMainloop::TileShapeQK_MNK;
  using TileShapePV_MNK = typename CollectiveMainloop::TileShapePV_MNK;
  using ElementQ = typename CollectiveMainloop::ElementQ;
  using ElementK = typename CollectiveMainloop::ElementK;
  using ElementV = typename CollectiveMainloop::ElementV;
  using ElementS = typename CollectiveMainloop::ElementS;
  using ElementAccum = typename CollectiveMainloop::ElementAccum;
  using StrideQ = typename CollectiveMainloop::StrideQ;
  using StrideK = typename CollectiveMainloop::StrideK;
  using StrideV = typename CollectiveMainloop::StrideV;

  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using PipelineState = typename CollectiveMainloop::PipelineState;
  using MainloopPipelineQ = typename CollectiveMainloop::MainloopPipelineQ;
  using PipelineStateQ = typename CollectiveMainloop::PipelineStateQ;
  static constexpr int NumProducerWarps = CollectiveMainloop::NumProducerWarps;
  static constexpr int NumMMAWarps = CollectiveMainloop::NumMMAWarps;
  static constexpr int NumControlerWarps = CollectiveMainloop::NumControlerWarps;
  static constexpr int SgSize = CollectiveMainloop::sg_size;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using StrideO = typename CollectiveEpilogue::StrideO;

  using CollectiveSoftmaxEpilogue = CollectiveSoftmaxEpilogue_;

  static constexpr int NumSoftmaxWarps = CollectiveSoftmaxEpilogue::NumSoftmaxWarps;

  static constexpr int NumSGs = NumControlerWarps + NumSoftmaxWarps;

  using TileScheduler = TileScheduler_;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct TensorStorage {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    };
    struct PipelineStorage {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiloguePipelineStorage = typename CollectiveEpilogue::PipelineStorage;

      MainloopPipelineStorage mainloop;
      EpiloguePipelineStorage epilogue;
    };
  };

  static constexpr int TensorStorageSize = sizeof(typename SharedStorage::TensorStorage);

  // Host side kernel arguments
  struct Arguments {
    ProblemShape problem_shape{};
    typename CollectiveMainloop::Arguments mainloop{};
    typename CollectiveSoftmaxEpilogue::Arguments softmax{};
    typename CollectiveEpilogue::Arguments epilogue{};
  };

  // Device side kernel params
  struct Params {
    ProblemShape problem_shape;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveSoftmaxEpilogue::Params softmax;
    typename CollectiveEpilogue::Params epilogue;
    typename TileScheduler::Params scheduler;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop),
      CollectiveSoftmaxEpilogue::to_underlying_arguments(args.softmax),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue),
      TileScheduler::to_underlying_arguments(args.problem_shape, KernelHardwareInfo{}, TileShape{})
    };
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<NumSGs>(params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SgSize, NumControlerWarps + NumSoftmaxWarps, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const &params) const {
    CollectiveMainloop collective_mainloop;
    CollectiveSoftmaxEpilogue collective_softmax(params.softmax);
    CollectiveEpilogue collective_epilogue{params.epilogue};
    
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    uint32_t sg_id = get_sg_id();
    bool lane_predicate = cute::elect_one_sync();

    // Allocate shared memory
    auto slm_ptr = alloc_slm_buffer<uint8_t, TensorStorageSize>(item.get_group());
    auto& shared_tensors = *reinterpret_cast<typename SharedStorage::TensorStorage*>(slm_ptr);

    // Allocate pipeline storage
    auto& shared_pipelines = allocate_abarrier<typename SharedStorage::PipelineStorage>();

    // Allocate matrix descriptor
    auto tdesc_q = allocate_tdesc<0>();
    auto tdesc_k = allocate_tdesc<1>();
    auto tdesc_v = allocate_tdesc<2>();
    auto tdesc_o = allocate_tdesc<3>();

    typename MainloopPipelineQ::Params pipeline_q_params;
    pipeline_q_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ;
    if (sg_id == 0) {
      pipeline_q_params.role = MainloopPipelineQ::ThreadCategory::Producer;
    }
    else if (sg_id == 1) {
      pipeline_q_params.role = MainloopPipelineQ::ThreadCategory::Consumer;
    }
    pipeline_q_params.is_leader = lane_predicate && sg_id == 0;
    pipeline_q_params.num_producers = 1;
    pipeline_q_params.num_consumers = 1;

    typename MainloopPipeline::Params pipeline_k_params;
    pipeline_k_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    if (sg_id == 0) {
      pipeline_k_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    else if (sg_id == 1) {
      pipeline_k_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    pipeline_k_params.is_leader = lane_predicate && sg_id == 0;
    pipeline_k_params.num_producers = 1;
    pipeline_k_params.num_consumers = 1;

    typename MainloopPipeline::Params pipeline_v_params;
    pipeline_v_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
    if (sg_id == 0) {
      pipeline_v_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    else if (sg_id == 1) {
      pipeline_v_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    pipeline_v_params.is_leader = lane_predicate && sg_id == 0;
    pipeline_v_params.num_producers = 1;
    pipeline_v_params.num_consumers = 1;

    MainloopPipelineQ pipeline_q = MainloopPipelineQ(shared_pipelines.mainloop.storage_Q, pipeline_q_params, /*cluster_shape=*/Shape<_1, _1, _1>{});
    MainloopPipeline pipeline_k = MainloopPipeline(shared_pipelines.mainloop.storage_K, pipeline_k_params, /*cluster_shape=*/Shape<_1, _1, _1>{});
    MainloopPipeline pipeline_v = MainloopPipeline(shared_pipelines.mainloop.storage_V, pipeline_v_params, /*cluster_shape=*/Shape<_1, _1, _1>{});

    typename MainloopPipeline::Params pipeline_s_params;
    pipeline_s_params.transaction_bytes = 1; // 1 mma in s producer side
    if (sg_id == 1) {
      pipeline_s_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    else if (sg_id >= NumControlerWarps) {
      pipeline_s_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    pipeline_s_params.is_leader = lane_predicate && sg_id == 1;
    pipeline_s_params.num_producers = 1;
    pipeline_s_params.num_consumers = NumSoftmaxWarps * cutlass::NumThreadsPerWarp;

    typename MainloopPipeline::Params pipeline_p_params;
    pipeline_p_params.transaction_bytes = 0; // no dma/mma in p producer side
    if (sg_id >= NumControlerWarps) {
      pipeline_p_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    else if (sg_id == 1) {
      pipeline_p_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    pipeline_p_params.is_leader = 0; // no leader for softmax
    pipeline_p_params.num_producers = NumSoftmaxWarps * cutlass::NumThreadsPerWarp;
    pipeline_p_params.num_consumers = 1;

    MainloopPipeline pipeline_s = MainloopPipeline(shared_pipelines.mainloop.storage_S, pipeline_s_params, /*cluster_shape=*/Shape<_1, _1, _1>{});
    MainloopPipeline pipeline_p = MainloopPipeline(shared_pipelines.mainloop.storage_P, pipeline_p_params, /*cluster_shape=*/Shape<_1, _1, _1>{});

    // KV tiles along seq_len_kv
    int num_kv_tiles = cute::ceil_div(get<3>(params.problem_shape), size<1>(typename CollectiveMainloop::TileShapeQK_MNK{}));

    // Initialize abarriers
    if (sg_id == 0) {
      if (lane_predicate) {
        shared_pipelines.mainloop.barrier_Q.init(1);
        shared_pipelines.epilogue.barrier_O.init(/*arrival_count=*/1);
        shared_pipelines.mainloop.barrier_O.init(/*arrival_count=*/1);
        shared_pipelines.mainloop.barrier_O_empty.init(/*arrival_count=*/NumSoftmaxWarps * cutlass::NumThreadsPerWarp);

        // TODO: Add async_gmma PISA with only .dtm.btm, without .atm
        shared_pipelines.mainloop.barrier_q_dummy.init(/*arrival_count=*/1);

        shared_pipelines.epilogue.barrier_O_final.init(/*arrival_count=*/NumSoftmaxWarps * cutlass::NumThreadsPerWarp);
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (sg_id == 0) { // Producer
      TileScheduler tile_scheduler{params.scheduler};

      PipelineStateQ smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipelineQ>();

      PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
      PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

      bool is_first_wave = true, is_last_wave = false;
      bool valid = tile_scheduler.is_valid();
      while (valid) {
        auto block_coord = tile_scheduler.get_block_coord();

        ++tile_scheduler;
        valid = tile_scheduler.is_valid();
        is_last_wave = !valid;
        auto block_coord_next = tile_scheduler.get_block_coord();

        collective_mainloop.load(
          params.mainloop,
          shared_tensors.mainloop,
          shared_pipelines.mainloop,
          make_tuple(tdesc_q, tdesc_k, tdesc_v),
          block_coord,
          block_coord_next,
          num_kv_tiles,
          pipeline_q, smem_pipe_write_q,
          pipeline_k, smem_pipe_write_k,
          pipeline_v, smem_pipe_write_v,
          is_first_wave, is_last_wave
        );

        is_first_wave = false;
      }
    }
    else if (sg_id == 1) { // MMA
      TileScheduler tile_scheduler{params.scheduler};

      PipelineStateQ smem_pipe_read_q;

      PipelineState smem_pipe_read_k;
      PipelineState smem_pipe_read_v;

      PipelineState smem_pipe_write_s = cutlass::make_producer_start_state<MainloopPipeline>();
      PipelineState smem_pipe_read_p;

      bool is_first_wave = true, is_last_wave = false;
      bool valid = tile_scheduler.is_valid();
      while (valid) {
        auto block_coord = tile_scheduler.get_block_coord();

        ++tile_scheduler;
        valid = tile_scheduler.is_valid();
        is_last_wave = !valid;

        collective_mainloop.mma(
          params.mainloop,
          shared_tensors.mainloop,
          shared_tensors.epilogue,
          shared_pipelines.mainloop,
          num_kv_tiles,
          pipeline_q, smem_pipe_read_q,
          pipeline_k, smem_pipe_read_k,
          pipeline_v, smem_pipe_read_v,
          pipeline_s, smem_pipe_write_s,
          pipeline_p, smem_pipe_read_p,
          collective_softmax,
          is_first_wave, is_last_wave
        );

        is_first_wave = false;
      }
    } 
    else if (sg_id == 2) { // ADMA store
      TileScheduler tile_scheduler{params.scheduler};
      uint32_t phase = 0;

      bool valid = tile_scheduler.is_valid();
      while (valid) {
        auto block_coord = tile_scheduler.get_block_coord();

        ++tile_scheduler;
        valid = tile_scheduler.is_valid();

        collective_epilogue.store(
          params.epilogue,
          shared_tensors.epilogue,
          shared_pipelines.epilogue,
          make_tuple(tdesc_o),
          block_coord,
          phase
        );
      }
    }
    else if (sg_id >= NumControlerWarps) { // Softmax
      TileScheduler tile_scheduler{params.scheduler};

      PipelineState smem_pipe_read_s;
      PipelineState smem_pipe_write_p = cutlass::make_producer_start_state<MainloopPipeline>();

      constexpr uint32_t total_rows_per_wi = CollectiveSoftmaxEpilogue::TotalRowsPerThread;

      Tensor max_reg = make_tensor<ElementS>(Shape<Int<total_rows_per_wi>>{});
      Tensor exp_reg = make_tensor<ElementAccum>(Shape<Int<total_rows_per_wi>>{});
      Tensor sum_reg = make_tensor<ElementAccum>(Shape<Int<total_rows_per_wi>>{});

      bool is_first_wave = true, is_last_wave = false;
      bool valid = tile_scheduler.is_valid();
      while (valid) {
        auto block_coord = tile_scheduler.get_block_coord();

        ++tile_scheduler;
        valid = tile_scheduler.is_valid();
        is_last_wave = !valid;

        collective_mainloop.softmax(
          params.mainloop,
          shared_tensors.mainloop,
          shared_tensors.epilogue,
          shared_pipelines.mainloop,
          shared_pipelines.epilogue,
          num_kv_tiles,
          pipeline_s, smem_pipe_read_s,
          pipeline_p, smem_pipe_write_p,
          collective_softmax,
          max_reg, exp_reg, sum_reg,
          is_first_wave, is_last_wave
        );

        is_first_wave = false;
      }
    }
  }

};
} // namespace cutlass::flash_attention::kernel
