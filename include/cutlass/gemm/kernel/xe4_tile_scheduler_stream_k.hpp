/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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

#include "cute/tensor.hpp"
#include "cute/arch/xe4_inline_pisa.hpp"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm_coord.hpp"
#include "cutlass/gemm/kernel/xe4_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp"
#include "cutlass/barrier.h"
#include "cutlass/block_striped.h"

namespace cutlass::gemm::kernel::detail {

using namespace cute;

// Persistent tile scheduler for XE4 with stream-K decomposition.
// Uses PersistentTileSchedulerXe4 (CLC-based) for grid iteration and
// PersistentTileSchedulerSm90StreamK for K-decomposition, workspace, and fixup.
// Follows the SM100 StreamK pattern.
template <
  class TileShape,
  class ClusterShape,
  uint32_t Stages_
>
class PersistentTileSchedulerXe4StreamK {
  static constexpr uint32_t NumThreadsPerWarp = 32;
  using UnderlyingScheduler = PersistentTileSchedulerXe4<Stages_, ClusterShape>;
  using UnderlyingStreamKScheduler = PersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>;
  using InternalWorkTileInfo = typename UnderlyingScheduler::WorkTileInfo;

public:
  static_assert(cute::is_static_v<ClusterShape>);

  static constexpr uint32_t Stages = Stages_;
  static constexpr bool IsDynamicPersistent = true;

  using CLCResponse = typename UnderlyingScheduler::CLCResponse;
  using WorkTileInfo = typename UnderlyingStreamKScheduler::WorkTileInfo;

  using Params = PersistentTileSchedulerXe4StreamKParams;
  using ReductionMode = typename Params::ReductionMode;
  using DecompositionMode = typename Params::DecompositionMode;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;
  using Pipeline = typename UnderlyingScheduler::Pipeline;
  using PipelineState = typename Pipeline::PipelineState;

  static constexpr uint32_t cluster_size = size(ClusterShape{});

  // Number of sub blocks in the kernel epilogue
  static constexpr int EpilogueSubtiles = 1;

  struct Arguments {
    Arguments() = default;
    Arguments(Arguments const&) = default;
    Arguments(Arguments&&) = default;

    CUTLASS_HOST_DEVICE Arguments& operator=(Arguments const& args) {
      splits = args.splits;
      max_swizzle_size = args.max_swizzle_size;
      raster_order = args.raster_order;
      reduction_mode = args.reduction_mode;
      decomposition_mode = args.decomposition_mode;
      return *this;
    }

    CUTLASS_HOST_DEVICE Arguments& operator=(Arguments&& args) noexcept {
      splits = args.splits;
      max_swizzle_size = args.max_swizzle_size;
      raster_order = args.raster_order;
      reduction_mode = args.reduction_mode;
      decomposition_mode = args.decomposition_mode;
      return *this;
    }

    int splits = 1;
    int max_swizzle_size = 1;
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic;
    ReductionMode reduction_mode = ReductionMode::Deterministic;
    DecompositionMode decomposition_mode = DecompositionMode::Heuristic;
  };

  //
  // Static helpers
  //

  template <class ProblemShapeMNKL, class TileShapeMNK>
  CUTLASS_HOST_DEVICE static dim3
  get_tiled_cta_shape_mnl(ProblemShapeMNKL problem_shape_mnkl, TileShapeMNK tile_shape, ClusterShape cluster_shape) {
    return UnderlyingScheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
  }

  template <class ProblemShapeMNKL, class TileShapeMNK, class AtomThrShape>
  CUTLASS_HOST_DEVICE static dim3
  get_tiled_cta_shape_mnl(ProblemShapeMNKL problem_shape_mnkl,
                          TileShapeMNK tile_shape_mnk,
                          AtomThrShape atom_thr_shape_mnk,
                          ClusterShape cluster_shape_mnk) {
    return UnderlyingScheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk, cluster_shape_mnk);
  }

  template <class ProblemShapeMNKL, class TileShapeMNK>
  static auto
  calculate_problem_blocks_shape(ProblemShapeMNKL problem_shape_mnkl, TileShapeMNK tile_shape) {
    return UnderlyingScheduler::calculate_problem_blocks_shape(problem_shape_mnkl, tile_shape);
  }

  template <class ProblemShape>
  static Params
  to_underlying_arguments(
      ProblemShape problem_shape,
      TileShape tile_shape,
      [[maybe_unused]] ClusterShape cluster_shape,
      KernelHardwareInfo const& hw_info,
      Arguments const& args,
      void* workspace,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {

    auto cs = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);
    auto problem_shape_mnkl = cute::append<4>(problem_shape, cute::Int<1>{});
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cs);
    uint32_t k_tile_per_output_tile = cute::size(
      cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    Params params;
    params.initialize(
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(cs),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      args.reduction_mode,
      args.decomposition_mode,
      workspace
    );
    return params;
  }

  template <class ProblemShapeMNKL, class TileShapeMNK, class AtomThrShape>
  static Params
  to_underlying_arguments(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_mnk,
      AtomThrShape atom_thr_shape_mnk,
      ClusterShape cluster_shape_mnk,
      KernelHardwareInfo const& hw_info,
      Arguments const& args,
      void* workspace = nullptr) {

    auto cs = cutlass::detail::select_cluster_shape(cluster_shape_mnk, hw_info.cluster_shape);
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk, cs);
    uint32_t k_tile_per_output_tile = cute::size(
      cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    Params params;
    params.initialize(
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(cs),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      args.reduction_mode,
      args.decomposition_mode,
      workspace
    );
    return params;
  }

  static bool
  can_implement(Arguments const& args) {
    return UnderlyingStreamKScheduler::can_implement(args);
  }

  template <class ProblemShapeMNKL, class BlockShape>
  CUTLASS_HOST_DEVICE
  static dim3
  get_grid_shape(
    Params const& params,
    ProblemShapeMNKL problem_shape_mnk,
    BlockShape cta_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo hw_info,
    [[maybe_unused]] Arguments arguments) {

    auto problem_shape_MNKL = cute::append<4>(problem_shape_mnk, cute::Int<1>{});
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_MNKL, cta_shape, cluster_shape);
    return params.get_grid_shape(problem_blocks, to_gemm_coord(cluster_shape));
  }

  //
  // Constructors
  //

  CUTLASS_DEVICE
  explicit
  PersistentTileSchedulerXe4StreamK(Params const& params)
    : xe4_scheduler_(params.xe4_params_),
      params_(params),
      block_id_in_cluster_(cute::block_id_in_cluster()) {}

  CUTLASS_DEVICE
  PersistentTileSchedulerXe4StreamK(CLCResponse* clc_response_ptr, Params const& params, dim3 block_id_in_cluster)
    : xe4_scheduler_(clc_response_ptr, params.xe4_params_, block_id_in_cluster),
      params_(params),
      block_id_in_cluster_(block_id_in_cluster) {}

  template <class ProblemShapeMNKL, class TileShapeMNK>
  CUTLASS_DEVICE
  PersistentTileSchedulerXe4StreamK(CLCResponse* clc_response_ptr, Params const& params,
      ProblemShapeMNKL, TileShapeMNK, dim3 block_id_in_cluster)
    : PersistentTileSchedulerXe4StreamK(clc_response_ptr, params, block_id_in_cluster) {}

  //
  // Work tile scheduling
  //

  CUTLASS_DEVICE
  WorkTileInfo
  initial_work_tile_info(ClusterShape cluster_shape) {
    InternalWorkTileInfo internal = xe4_scheduler_.initial_work_tile_info(cluster_shape);
    return convert_work(internal);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  initial_work_tile_info() {
    return initial_work_tile_info(ClusterShape{});
  }

  CUTLASS_DEVICE
  auto
  work_tile_to_cta_coord(WorkTileInfo const& work_tile_info) {
    if (is_dp_only()) {
      // For data-parallel decompositions, use the M,N from the stream-K work tile info
      // but route through the XE4 scheduler's coordinate logic
      InternalWorkTileInfo underlying{
        work_tile_info.M_idx,
        work_tile_info.N_idx,
        work_tile_info.L_idx,
        work_tile_info.is_valid()
      };
      return xe4_scheduler_.work_tile_to_cta_coord(underlying);
    }
    else {
      // The SM90 stream-K scheduler already operates at CTA level,
      // so the returned work tile info already contains CTA offsets
      return cute::make_coord(
        work_tile_info.M_idx,
        work_tile_info.N_idx,
        int32_t(0),
        work_tile_info.L_idx
      );
    }
  }

  CUTLASS_DEVICE
  PipelineState
  advance_to_next_work(Pipeline& clc_pipeline, PipelineState clc_pipe_producer_state) {
    return xe4_scheduler_.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
  }

  // Returns whether the current work_tile_info should continue to be used.
  // This occurs for stream-K work units that span multiple output tiles.
  CUTLASS_DEVICE
  bool
  continue_current_work(WorkTileInfo& work_tile_info) {
    return UnderlyingStreamKScheduler::continue_current_work_for_linear_idx(
      current_work_linear_idx_, unit_iter_start_, block_id_in_cluster_, work_tile_info, params_.sk_params_);
  }

  template <class CLCPipeline, class CLCPipelineState>
  CUTLASS_DEVICE
  cute::tuple<WorkTileInfo, bool>
  fetch_next_work(
      WorkTileInfo work_tile_info,
      CLCPipeline& clc_pipeline,
      CLCPipelineState clc_pipe_consumer_state) {

    // Check whether we should continue on with the current work unit.
    // If so, the work unit has been updated to reflect the next tile to compute.
    // Return false to indicate that the CLC pipeline state need not be advanced.
    if (continue_current_work(work_tile_info)) {
      return cute::make_tuple(work_tile_info, false);
    }

    // Fetch next CLC query result via the underlying XE4 scheduler
    auto [work_tile, _] = xe4_scheduler_.fetch_next_work(InternalWorkTileInfo{}, clc_pipeline, clc_pipe_consumer_state);
    if (!work_tile.is_valid()) {
      return cute::make_tuple(invalid_work_tile(), true);
    }

    auto converted_work_tile = convert_work(work_tile);

    // Return true to indicate that the CLC pipeline state should be advanced
    return cute::make_tuple(converted_work_tile, true);
  }

  CUTLASS_DEVICE
  void
  set_data_ptr(CLCResponse* clc_response_ptr) {
    xe4_scheduler_.set_data_ptr(clc_response_ptr);
  }

  //
  // K-tile API
  //

  template <class ProblemShapeMNKL, class TileShapeMNK>
  CUTLASS_DEVICE
  auto
  get_k_tile_iterator(WorkTileInfo const& work_tile_info, ProblemShapeMNKL problem_shape, TileShapeMNK tile_shape) {
    auto k_tiles = cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape));
    auto k_tile_start = get_work_k_tile_start(work_tile_info);
    return cute::make_coord_iterator(idx2crd(k_tile_start, k_tiles), k_tiles);
  }

  template <class ProblemShape, class TileShapeMNK>
  CUTLASS_HOST_DEVICE
  static int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape, TileShapeMNK) {
    return work_tile_info.k_tile_count;
  }

  CUTLASS_HOST_DEVICE
  static uint32_t
  get_work_k_tile_start(WorkTileInfo const& work_tile_info) {
    return work_tile_info.K_idx;
  }

  //
  // Epilogue and fixup
  //

  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const& work_tile_info, Params const& params) {
    return UnderlyingStreamKScheduler::compute_epilogue(work_tile_info, params.sk_params_);
  }

  CUTLASS_HOST_DEVICE
  bool
  compute_epilogue(WorkTileInfo const& work_tile_info) const {
    return UnderlyingStreamKScheduler::compute_epilogue(work_tile_info, params_.sk_params_);
  }

  CUTLASS_HOST_DEVICE
  static bool
  requires_fixup(Params const& params, WorkTileInfo const& work_tile_info) {
    return UnderlyingStreamKScheduler::requires_fixup(params.sk_params_, work_tile_info);
  }

  // Returns a pointer to the per-tile atomic counter array in the SK workspace.
  // Layout: [counter_data: int32_t * num_sk_tiles]
  // XE4 SK uses fred directly to gmem D; no accumulator scratch is needed.
  // EpiLoad increments the counter after fred completes; the final split polls it.
  CUTLASS_HOST_DEVICE
  static int*
  get_sk_tile_counter_ptr(Params const& params) {
    return reinterpret_cast<int*>(params.sk_params_.reduction_workspace_);
  }

  // Returns the linear tile index for a given work tile (used to index into counters).
  // Computed in uint64_t to avoid overflow when multiplying tiles_mn * L_idx.
  // Follows SM90 pattern of returning uint64_t for consistency across architectures.
  CUTLASS_DEVICE
  static uint64_t
  get_tile_idx(Params const& params, WorkTileInfo const& work_tile_info) {
    auto const& sk = params.sk_params_;
    uint64_t linear_in_batch = PersistentTileSchedulerSm90::get_linear_idx_from_m_and_n(
        work_tile_info.M_idx, work_tile_info.N_idx,
        sk.divmod_cluster_shape_major_, sk.divmod_cluster_shape_minor_,
        sk.divmod_cluster_blk_major_, sk.log_swizzle_size_, sk.raster_order_);
    uint64_t tiles_mn = sk.divmod_batch_.divisor;
    return tiles_mn * work_tile_info.L_idx + linear_in_batch;
  }

  CUTLASS_DEVICE
  static bool
  valid_warpgroup_in_work_tile(WorkTileInfo const&) {
    return true;
  }

  CUTLASS_DEVICE
  static bool
  requires_separate_reduction(Params const&) {
    return false;
  }

  //
  // Workspace
  //

  // One int32 atomic counter per output tile.
  // EpiLoad increments it after fred; the final split polls until it reaches K_idx.
  static constexpr uint32_t NumFixupBarriers = 1;

  template <class ProblemShape, class ElementAccumulator>
  static size_t
  get_workspace_size(
      Arguments const& args,
      ProblemShape problem_shape,
      KernelHardwareInfo const& hw_info,
      [[maybe_unused]] uint32_t reduction_warp_groups = NumFixupBarriers,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t num_accumulator_mtxs = 1) {
    auto problem_shape_mnkl = cute::append<4>(problem_shape, cute::Int<1>{});

    auto cs = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);
    TileShape tile_shape;

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cs);
    uint32_t k_tile_per_output_tile = cute::size(
      cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    return Params::get_workspace_size(
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(tile_shape),
      to_gemm_coord(cs),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      args.decomposition_mode,
      args.reduction_mode,
      reduction_warp_groups,
      sizeof_bits<int32_t>::value,
      sizeof_bits<ElementAccumulator>::value,
      EpilogueSubtiles,
      num_accumulator_mtxs
    );
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(
      Arguments const& args,
      void* workspace,
      cudaStream_t stream,
      ProblemShape const& problem_shape,
      KernelHardwareInfo const& hw_info,
      [[maybe_unused]] uint32_t reduction_warp_groups = NumFixupBarriers,
      [[maybe_unused]] uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t num_accumulator_mtxs = 1,
      [[maybe_unused]] CudaHostAdapter* cuda_adapter = nullptr) {
    auto problem_shape_mnkl = cute::append<4>(problem_shape, cute::Int<1>{});

    auto cs = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);
    TileShape tile_shape;

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cs);
    uint32_t k_tile_per_output_tile = cute::size(
      cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    return Params::initialize_workspace(
      workspace,
      stream,
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(tile_shape),
      to_gemm_coord(cs),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      args.decomposition_mode,
      args.reduction_mode,
      reduction_warp_groups,
      sizeof_bits<int32_t>::value,
      sizeof_bits<ElementAccumulator>::value,
      EpilogueSubtiles,
      num_accumulator_mtxs,
      cuda_adapter
    );
  }

private:

  CUTLASS_HOST_DEVICE
  WorkTileInfo invalid_work_tile() const {
    return WorkTileInfo::invalid_work_tile();
  }

  // Converts the work tile info returned by the XE4 scheduler to a linear index
  // for the SM90 stream-K scheduler.
  // XE4 grid layout: X=M, Y=N (unlike SM100 where grid is oriented differently).
  CUTLASS_DEVICE
  uint64_t
  to_linear_idx(InternalWorkTileInfo const& work_tile_info) {
    // On XE4, the CLC query returns (M_idx, N_idx, L_idx) which map to grid (X, Y, Z).
    // The SM90 stream-K scheduler expects a linear index over all CTAs.
    // Linear index = wave_idx * sm_count + cluster_idx
    uint64_t sm_count = uint64_t(GridDimX()) * uint64_t(GridDimY());
    uint64_t wave_idx = work_tile_info.L_idx;
    uint64_t cluster_idx = uint64_t(GridDimY()) * uint64_t(work_tile_info.M_idx) + uint64_t(work_tile_info.N_idx);

    return sm_count * wave_idx + cluster_idx;
  }

  // Convert underlying XE4 scheduler work tile to SM90 StreamK WorkTileInfo
  CUTLASS_DEVICE
  WorkTileInfo
  convert_work(InternalWorkTileInfo const& work_tile_info) {
    if (has_sk_work()) {
      current_work_linear_idx_ = to_linear_idx(work_tile_info);
      auto work = UnderlyingStreamKScheduler::get_current_work_for_linear_idx(
        unit_iter_start_, current_work_linear_idx_, block_id_in_cluster_, params_.sk_params_);
      if (!work.is_valid()) {
        return invalid_work_tile();
      }
      return work;
    }
    else if (is_split_k()) {
      // Split-K: CLC returns (M_idx, N_idx, L_idx) where L_idx encodes batch*splits + split_idx
      int32_t M_idx = work_tile_info.M_idx;
      int32_t N_idx = work_tile_info.N_idx;

      int L_idx, Split_idx;
      params_.sk_params_.divmod_splits_(L_idx, Split_idx, work_tile_info.L_idx);

      int additional_k_tiles = 0;
      int split_start_offset = params_.sk_params_.big_units_;

      if (Split_idx < static_cast<int>(params_.sk_params_.big_units_)) {
        additional_k_tiles = 1;
        split_start_offset = Split_idx;
      }

      uint32_t k_tiles = params_.sk_params_.divmod_k_tiles_per_sk_unit_.divisor;
      uint32_t K_idx = Split_idx * k_tiles;

      K_idx += split_start_offset;
      k_tiles += additional_k_tiles;

      return {
        M_idx,
        N_idx,
        static_cast<int32_t>(K_idx),
        static_cast<int32_t>(L_idx),
        k_tiles,
        k_tiles
      };
    }
    else {
      // Data-parallel case
      return {
        static_cast<int32_t>(work_tile_info.M_idx),
        static_cast<int32_t>(work_tile_info.N_idx),
        static_cast<int32_t>(0),                   // K_idx
        static_cast<int32_t>(work_tile_info.L_idx),
        static_cast<uint32_t>(params_.sk_params_.divmod_tiles_per_output_tile_.divisor),
        static_cast<uint32_t>(params_.sk_params_.divmod_tiles_per_output_tile_.divisor)
      };
    }
  }

  CUTLASS_HOST_DEVICE
  bool is_dp_only() const {
    return params_.sk_params_.sk_units_ == 0 && params_.sk_params_.divmod_splits_.divisor == 1;
  }

  CUTLASS_HOST_DEVICE
  bool is_split_k() const {
    return params_.sk_params_.divmod_splits_.divisor > 1;
  }

  CUTLASS_HOST_DEVICE
  bool has_sk_work() const {
    return params_.sk_params_.sk_units_ > 0;
  }

  //
  // Members
  //

  UnderlyingScheduler xe4_scheduler_;
  Params const& params_;
  dim3 block_id_in_cluster_;
  uint64_t current_work_linear_idx_ = 0;
  uint32_t unit_iter_start_ = 0;
};

} // namespace cutlass::gemm::kernel::detail
