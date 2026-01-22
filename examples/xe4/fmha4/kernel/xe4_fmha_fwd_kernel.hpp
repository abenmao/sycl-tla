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
  using ClusterShape = typename CollectiveMainloop::ClusterShape;
  using TiledMmaQK = typename CollectiveMainloop::TiledMmaQK;
  using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TileShapeQK_MNK = typename CollectiveMainloop::TileShapeQK_MNK;
  using TileShapePV_MNK = typename CollectiveMainloop::TileShapePV_MNK;
  using ElementQ = typename CollectiveMainloop::ElementQ;
  using ElementK = typename CollectiveMainloop::ElementK;
  using ElementV = typename CollectiveMainloop::ElementV;
  using ElementAccum = typename CollectiveMainloop::ElementAccum;
  using StrideQ = typename CollectiveMainloop::StrideQ;
  using StrideK = typename CollectiveMainloop::StrideK;
  using StrideV = typename CollectiveMainloop::StrideV;

  static constexpr int qk_mma_k_itr = shape<2>(TileShapeQK_MNK{}) / cute::size<2>(typename TiledMmaQK::Shape_MNK{});
  static constexpr int pv_mma_k_itr = shape<2>(TileShapePV_MNK{}) / cute::size<2>(typename TiledMmaPV::Shape_MNK{});

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using StrideO = typename CollectiveEpilogue::StrideO;

  using CollectiveSoftmaxEpilogue = CollectiveSoftmaxEpilogue_;

  using TileScheduler = TileScheduler_;

  enum class SgRole {MMA, Load, Epilogue, Softmax, Empty};

  static constexpr int NumControlSgs = 4;

  static constexpr int NumMmaSgs = 1;
  static constexpr int NumLoadSgs = 1;
  static constexpr int NumEpilogueSgs = 1;
  static constexpr int NumSoftmaxSgs = 16;

  CUTLASS_DEVICE
    static constexpr SgRole sg_id_to_SgRole(int sg_id) {
    if (sg_id == 0) return SgRole::Load;
    if (sg_id == 1) return SgRole::MMA;
    if (sg_id == 2) return SgRole::Epilogue;
    if (sg_id == 3) return SgRole::Empty;
    if (sg_id >= 4 && sg_id < 20) return SgRole::Softmax;

    return SgRole::Empty;
  }

  // Kernel level shared memory storage
  struct SharedStorage {
    struct TensorStorage {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    };
    struct PipelineStorage {
      typename CollectiveMainloop::PipelineQ::SharedStorage load_q;
      typename CollectiveMainloop::PipelineK::SharedStorage load_k;
      typename CollectiveMainloop::PipelineV::SharedStorage load_v;
      typename CollectiveMainloop::PipelineS::SharedStorage mma_s;
      typename CollectiveMainloop::PipelineO::SharedStorage mma_corr;
      typename CollectiveMainloop::PipelineEpi::SharedStorage epi;

      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;

      MainloopPipelineStorage mainloop;
    };
  };

  static constexpr int TensorStorageSize = sizeof(typename SharedStorage::TensorStorage);
  static constexpr int PipelineStorageSize = sizeof(typename SharedStorage::PipelineStorage);

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
      TileScheduler::to_underlying_arguments(args.problem_shape, TileShape{})
    };
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.problem_shape, TileShape{}, args.mainloop);
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler);
  }

  static dim3 get_block_shape() {
    static constexpr int TotalSgs = NumControlSgs + NumSoftmaxSgs;

    return dim3(cutlass::NumThreadsPerWarp, TotalSgs, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const &params) const {
    CollectiveMainloop collective_mainloop;
    CollectiveSoftmaxEpilogue collective_softmax(params.softmax);
    CollectiveEpilogue collective_epilogue{params.epilogue};
    TileScheduler tile_scheduler{params.scheduler};

    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    uint32_t sg_id = get_sg_id();

    uint32_t worker_id = item.get_local_linear_id();
    SgRole role = sg_id_to_SgRole(sg_id);

    bool lane_predicate = cute::elect_one_sync();

    // Allocate shared memory
    auto slm_ptr = alloc_slm_buffer<uint8_t, TensorStorageSize>(item.get_group());
    auto& shared_tensors = *reinterpret_cast<typename SharedStorage::TensorStorage*>(slm_ptr);

    // Allocate pipeline storage
    auto abar_base = allocate_abar_bytes<0, PipelineStorageSize>();
    auto& shared_pipelines = *reinterpret_cast<typename SharedStorage::PipelineStorage*>(abar_base);

    // Allocate matrix descriptor
    auto tdesc_q = allocate_tdesc<0>();
    auto tdesc_k = allocate_tdesc<1>();
    auto tdesc_v = allocate_tdesc<2>();
    auto tdesc_o = allocate_tdesc<3>();

    // from load Q to mma
    typename CollectiveMainloop::PipelineQ::Params pipeline_load_q_params;
    if (role == SgRole::Load) {
      pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Producer;
    }
    if (role == SgRole::MMA) {
      pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Consumer;
    }
    pipeline_load_q_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesQ;
    pipeline_load_q_params.is_leader = lane_predicate && (role == SgRole::Load);
    pipeline_load_q_params.num_producers = 1;
    pipeline_load_q_params.num_consumers = 1;
    typename CollectiveMainloop::PipelineQ pipeline_load_q(shared_pipelines.load_q, pipeline_load_q_params, ClusterShape{});

    // from load K to mma
    typename CollectiveMainloop::PipelineK::Params pipeline_load_k_params;
    if (role == SgRole::Load) {
      pipeline_load_k_params.role = CollectiveMainloop::PipelineK::ThreadCategory::Producer;
    }
    if (role == SgRole::MMA) {
      pipeline_load_k_params.role = CollectiveMainloop::PipelineK::ThreadCategory::Consumer;
    }
    pipeline_load_k_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
    pipeline_load_k_params.is_leader = lane_predicate && (role == SgRole::Load);
    pipeline_load_k_params.num_producers = 1;
    pipeline_load_k_params.num_consumers = 1;
    typename CollectiveMainloop::PipelineK pipeline_load_k(shared_pipelines.load_k, pipeline_load_k_params, ClusterShape{});

    // from load V to mma
    typename CollectiveMainloop::PipelineV::Params pipeline_load_v_params;
    if (role == SgRole::Load) {
      pipeline_load_v_params.role = CollectiveMainloop::PipelineV::ThreadCategory::Producer;
    }
    if (role == SgRole::MMA) {
      pipeline_load_v_params.role = CollectiveMainloop::PipelineV::ThreadCategory::Consumer;
    }
    pipeline_load_v_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
    pipeline_load_v_params.is_leader = lane_predicate && (role == SgRole::Load);
    pipeline_load_v_params.num_producers = 1;
    pipeline_load_v_params.num_consumers = 1;
    typename CollectiveMainloop::PipelineV pipeline_load_v(shared_pipelines.load_v, pipeline_load_v_params, ClusterShape{});

    // from mma (QK) to softmax
    typename CollectiveMainloop::PipelineS::Params pipeline_mma_s_params;
    if (role == SgRole::MMA) {
      pipeline_mma_s_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Producer;
    }
    if (role == SgRole::Softmax) {
      pipeline_mma_s_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Consumer;
    }
    pipeline_mma_s_params.transaction_bytes = qk_mma_k_itr;
    pipeline_mma_s_params.is_leader = lane_predicate && (role == SgRole::MMA);
    pipeline_mma_s_params.num_producers = 1;
    pipeline_mma_s_params.num_consumers = NumSoftmaxSgs * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineS pipeline_mma_s(shared_pipelines.mma_s, pipeline_mma_s_params, ClusterShape{});

    // from mma (PV) to softmax
    typename CollectiveMainloop::PipelineO::Params pipeline_mma_corr_params;
    if (role == SgRole::MMA) {
      pipeline_mma_corr_params.role = CollectiveMainloop::PipelineO::ThreadCategory::Producer;
    }
    if (role == SgRole::Softmax) {
      pipeline_mma_corr_params.role = CollectiveMainloop::PipelineO::ThreadCategory::Consumer;
    }
    pipeline_mma_corr_params.transaction_bytes = pv_mma_k_itr;
    pipeline_mma_corr_params.is_leader = lane_predicate && (role == SgRole::MMA);
    pipeline_mma_corr_params.num_producers = 1;
    pipeline_mma_corr_params.num_consumers = NumSoftmaxSgs * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineO pipeline_mma_corr(shared_pipelines.mma_corr, pipeline_mma_corr_params, ClusterShape{});

    // from softmax to epilogue
    typename CollectiveMainloop::PipelineEpi::Params pipeline_epi_params;
    if (role == SgRole::Softmax) {
      pipeline_epi_params.role = CollectiveMainloop::PipelineEpi::ThreadCategory::Producer;
    }
    if (role == SgRole::Epilogue) {
      pipeline_epi_params.role = CollectiveMainloop::PipelineEpi::ThreadCategory::Consumer;
    }
    pipeline_epi_params.transaction_bytes = 0;
    pipeline_epi_params.is_leader = 0;
    pipeline_epi_params.num_producers = NumSoftmaxSgs * cutlass::NumThreadsPerWarp;
    pipeline_epi_params.num_consumers = 1;
    typename CollectiveMainloop::PipelineEpi pipeline_epi(shared_pipelines.epi, pipeline_epi_params, ClusterShape{});


    typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_consumer_state;
    typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_producer_state =
      cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineQ>();

    typename CollectiveMainloop::PipelineK::PipelineState pipeline_load_k_consumer_state;
    typename CollectiveMainloop::PipelineK::PipelineState pipeline_load_k_producer_state =
      cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineK>();

    typename CollectiveMainloop::PipelineV::PipelineState pipeline_load_v_consumer_state;
    typename CollectiveMainloop::PipelineV::PipelineState pipeline_load_v_producer_state =
      cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineV>();

    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s_consumer_state;
    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s_producer_state =
      cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineS>();

    typename CollectiveMainloop::PipelineO::PipelineState pipeline_mma_corr_consumer_state;
    typename CollectiveMainloop::PipelineO::PipelineState pipeline_mma_corr_producer_state =
      cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineO>();

    typename CollectiveMainloop::PipelineEpi::PipelineState pipeline_epi_consumer_state;
    typename CollectiveMainloop::PipelineEpi::PipelineState pipeline_epi_producer_state =
      cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineEpi>();

    // KV tiles along seq_len_kv
    int num_kv_tiles = cute::ceil_div(get<3>(params.problem_shape), size<1>(typename CollectiveMainloop::TileShapeQK_MNK{}));

    // Initialize abarriers
    if (role == SgRole::MMA && lane_predicate) {
      // TODO: add AMMA atom with different combinations of .dtm, .atm, and .btm
      shared_pipelines.mainloop.barrier_dummy.init(1);
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (role == SgRole::Load && lane_predicate) {
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto block_coord = tile_scheduler.get_block_coord();

        collective_mainloop.load(
          params.mainloop,
          shared_tensors.mainloop,
          shared_pipelines.mainloop,
          make_tuple(tdesc_q, tdesc_k, tdesc_v),
          block_coord,
          num_kv_tiles,
          pipeline_load_q, pipeline_load_q_producer_state,
          pipeline_load_k, pipeline_load_k_producer_state,
          pipeline_load_v, pipeline_load_v_producer_state
        );
      }
    }
    else if (role == SgRole::MMA && lane_predicate) {
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto block_coord = tile_scheduler.get_block_coord();

        collective_mainloop.mma(
          shared_tensors.mainloop,
          shared_pipelines.mainloop,
          num_kv_tiles,
          pipeline_load_q, pipeline_load_q_consumer_state,
          pipeline_load_k, pipeline_load_k_consumer_state,
          pipeline_load_v, pipeline_load_v_consumer_state,
          pipeline_mma_s, pipeline_mma_s_producer_state,
          pipeline_mma_corr, pipeline_mma_corr_producer_state
        );
      }
    } 
    else if (role == SgRole::Epilogue && lane_predicate) {
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto block_coord = tile_scheduler.get_block_coord();

        collective_epilogue.store(
          params.epilogue,
          shared_tensors.epilogue,
          make_tuple(tdesc_o),
          block_coord,
          pipeline_epi, pipeline_epi_consumer_state
        );
      }
    }
    else if (role == SgRole::Softmax) {
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto block_coord = tile_scheduler.get_block_coord();

        collective_mainloop.softmax(
          collective_softmax,
          shared_tensors.mainloop,
          shared_tensors.epilogue,
          num_kv_tiles,
          worker_id - cutlass::NumThreadsPerWarp * NumControlSgs,
          pipeline_mma_s, pipeline_mma_s_consumer_state,
          pipeline_mma_corr, pipeline_mma_corr_consumer_state,
          pipeline_epi , pipeline_epi_producer_state
        );
      }
    }
  }

};
} // namespace cutlass::flash_attention::kernel
