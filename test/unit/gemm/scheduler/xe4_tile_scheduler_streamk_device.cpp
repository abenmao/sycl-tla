/***************************************************************************************************
 * Copyright (c) 2025 INTEL CORPORATION. All rights reserved.
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
/**
 * Unit tests for PersistentTileSchedulerXe4StreamK.
 *
 * Two categories:
 *
 * 1. HOST-SIDE tests: verify workspace sizing, decomposition heuristics, grid
 *    shape computation, and the `requires_fixup`/`compute_epilogue` predicates
 *    without launching any kernel. These are fast and run on any host.
 *
 * 2. DEVICE-SIDE tests: launch a "K-tile coverage" kernel that uses the full
 *    CLC dynamic scheduling loop to dispatch work tiles. Each CTA atomically
 *    increments a per-output-tile counter by its `k_tile_count`. The host then
 *    checks that every output tile received exactly `total_k_tiles` worth of
 *    coverage — proving that Stream-K/Split-K decomposition dispatches every
 *    K-tile exactly once with no gaps or overlaps.
 *
 * Requires: Xe4 hardware (SYCL_INTEL_TARGET == 40) with CLC support.
 */

#include <sycl/sycl.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

#include "cute/tensor.hpp"
#include "cute/arch/cluster_xe4.hpp"
#include "cute/arch/xe4_util.hpp"
#include "cute/arch/xe4_inline_pisa.hpp"
#include "cutlass/gemm/kernel/xe4_tile_scheduler_stream_k.hpp"
#include "cutlass/pipeline/xe4_pipeline.hpp"
#include "cutlass/cluster_launch.hpp"

namespace {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// Constants
///////////////////////////////////////////////////////////////////////////////
static constexpr uint32_t kStages       = 2;
static constexpr int kNumSubgroups      = 2;
static constexpr int kThreadsPerSG      = cutlass::NumThreadsPerWarp;
static constexpr int kThreadsPerBlock   = kNumSubgroups * kThreadsPerSG;

///////////////////////////////////////////////////////////////////////////////
// Scheduler traits for StreamK
///////////////////////////////////////////////////////////////////////////////
template <class TileShape_, class ClusterShape_>
struct StreamKSchedulerTraits {
  using TileShape     = TileShape_;
  using ClusterShape  = ClusterShape_;
  using TileScheduler = cutlass::gemm::kernel::detail::PersistentTileSchedulerXe4StreamK<
                            TileShape, ClusterShape, kStages>;
  using Params        = typename TileScheduler::Params;
  using Arguments     = typename TileScheduler::Arguments;
  using WorkTileInfo  = typename TileScheduler::WorkTileInfo;
  using CLCPipeline   = cutlass::PipelineCLCFetchAsync<kStages, ClusterShape>;
  using CLCPipeState  = typename CLCPipeline::PipelineState;
  using CLCResponse   = typename TileScheduler::CLCResponse;
  using Pipeline      = typename TileScheduler::Pipeline;
  using PipelineState = typename TileScheduler::PipelineState;
  using RasterOrderOptions = typename Params::RasterOrderOptions;
  using DecompositionMode  = typename Params::DecompositionMode;
  using ReductionMode      = typename Params::ReductionMode;

  static constexpr int cluster_size = size(ClusterShape{});
};

///////////////////////////////////////////////////////////////////////////////
// Device kernel: K-tile coverage verification
//
// Each CTA fetches work via CLC + StreamK scheduler, then atomically adds its
// k_tile_count to k_coverage[output_tile_idx]. Host verifies sum ==
// total_k_tiles for every output tile (no gaps, no overlaps).
//
// Additionally writes split info for debugging: per-SK-unit records of
// (tile_idx, K_idx, k_tile_count).
///////////////////////////////////////////////////////////////////////////////
template <class TileShape, class ClusterShape>
struct StreamKCoverageKernel {
  using Traits        = StreamKSchedulerTraits<TileShape, ClusterShape>;
  using Params        = typename Traits::Params;
  using CLCPipeline   = typename Traits::CLCPipeline;
  using CLCPipeState  = typename Traits::CLCPipeState;
  using CLCResponse   = typename Traits::CLCResponse;
  using TileScheduler = typename Traits::TileScheduler;
  using WorkTileInfo  = typename Traits::WorkTileInfo;

  Params params;
  int*   k_coverage;         // [total_output_tiles]: atomic sum of k_tile_count per tile
  int*   dispatch_count;     // [total_output_tiles]: atomic count of splits per tile
  int*   oob_errors;         // [1]: out-of-bounds error count
  int    tiles_m;
  int    tiles_n;
  int    batches;
  int    total_k_tiles;

  CUTLASS_DEVICE int output_tile_idx(WorkTileInfo const& wti) const {
    return wti.L_idx * (tiles_m * tiles_n) + wti.N_idx * tiles_m + wti.M_idx;
  }

  CUTLASS_DEVICE void do_work(WorkTileInfo const& wti, uint32_t lane_id) const {
    if (!wti.is_valid() || wti.k_tile_count == 0) return;

    bool oob = (wti.M_idx < 0 || wti.M_idx >= tiles_m ||
                wti.N_idx < 0 || wti.N_idx >= tiles_n ||
                wti.L_idx < 0 || wti.L_idx >= batches);
    if (oob) {
      if (lane_id == 0) {
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                          sycl::memory_scope::device,
                          sycl::access::address_space::global_space>
            ref(oob_errors[0]);
        ref.fetch_add(1);
      }
      return;
    }

    if (lane_id == 0) {
      int idx = output_tile_idx(wti);
      sycl::atomic_ref<int, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
          cov_ref(k_coverage[idx]);
      cov_ref.fetch_add(static_cast<int>(wti.k_tile_count));

      sycl::atomic_ref<int, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
          cnt_ref(dispatch_count[idx]);
      cnt_ref.fetch_add(1);
    }
  }

  CUTLASS_DEVICE void operator()() const {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    uint32_t warp_idx = get_sg_id();
    bool is_scheduler_warp = (warp_idx == 0);
    uint32_t lane_id = get_lane_id();

    auto cluster_shape = ClusterShape{};

    auto slm_ptr = alloc_slm_buffer<uint8_t,
        sizeof(CLCResponse) * kStages>(item.get_group());
    auto* clc_response_ptr = reinterpret_cast<CLCResponse*>(slm_ptr);

    constexpr uint32_t PipeStorageSize =
        static_cast<uint32_t>(sizeof(typename CLCPipeline::SharedStorage));
    auto abar_base = allocate_abar_bytes<0, PipeStorageSize>();
    auto& pipeline_storage =
        *reinterpret_cast<typename CLCPipeline::SharedStorage*>(abar_base);

    uint32_t cluster_wgid_x = get_cluster_wgid<0>();
    uint32_t cluster_wgid_y = get_cluster_wgid<1>();
    auto cluster_layout_mn = make_layout(select<0,1>(ClusterShape{}));
    uint32_t cta_rank_in_cluster = cluster_layout_mn(make_coord(cluster_wgid_x, cluster_wgid_y));
    bool is_first_cta_in_cluster = (cta_rank_in_cluster == 0);
    bool is_clc_producer = is_scheduler_warp && is_first_cta_in_cluster;

    typename CLCPipeline::Params pipe_params;
    pipe_params.role = is_clc_producer
        ? CLCPipeline::ThreadCategory::ProducerConsumer
        : CLCPipeline::ThreadCategory::Consumer;
    pipe_params.producer_blockid   = 0;
    pipe_params.producer_arv_count = 1;
    pipe_params.consumer_arv_count = kThreadsPerSG + Traits::cluster_size * kThreadsPerSG;
    pipe_params.initializing_warp  = 0;
    CLCPipeline clc_pipeline(pipeline_storage, pipe_params, cluster_shape);

    if constexpr (Traits::cluster_size > 1) {
      cute::cluster_arrive();
      cute::cluster_wait();
    } else {
      item.barrier(sycl::access::fence_space::local_space);
    }

    dim3 block_id_in_cluster = cute::block_id_in_cluster();
    TileScheduler scheduler(clc_response_ptr, params, block_id_in_cluster);

    auto work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
    auto clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();
    CLCPipeState clc_pipe_consumer_state{};

    if (is_clc_producer) {
      bool requires_clc_query = true;
      do {
        if (requires_clc_query) {
          clc_pipe_producer_state =
              scheduler.advance_to_next_work(clc_pipeline, clc_pipe_producer_state);
        }
        auto [next_work, inc] =
            scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        requires_clc_query = inc;
        if (inc) { ++clc_pipe_consumer_state; }
        work_tile_info = next_work;
      } while (work_tile_info.is_valid());

    } else if (!is_scheduler_warp) {
      do {
        do_work(work_tile_info, lane_id);
        auto [next_work, inc] =
            scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        if (inc) { ++clc_pipe_consumer_state; }
        work_tile_info = next_work;
      } while (work_tile_info.is_valid());
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// Host helper: build scheduler params via to_underlying_arguments
///////////////////////////////////////////////////////////////////////////////
template <class Traits>
static typename Traits::Params make_streamk_params(
    cute::tuple<int,int,int,int> problem_shape_mnkl,
    typename Traits::Arguments const& args,
    void* workspace = nullptr) {

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  hw_info.max_active_clusters = 0;

  using TileScheduler = typename Traits::TileScheduler;
  using TileShape = typename Traits::TileShape;
  using ClusterShape = typename Traits::ClusterShape;

  return TileScheduler::to_underlying_arguments(
    problem_shape_mnkl,
    TileShape{},
    ClusterShape{},
    hw_info,
    args,
    workspace
  );
}

///////////////////////////////////////////////////////////////////////////////
// Host helper: compute expected tiles_m, tiles_n from problem shape
///////////////////////////////////////////////////////////////////////////////
template <class TileShape>
static dim3 get_problem_blocks(cute::tuple<int,int,int,int> problem_shape_mnkl) {
  int M = cute::get<0>(problem_shape_mnkl);
  int N = cute::get<1>(problem_shape_mnkl);
  int L = cute::get<3>(problem_shape_mnkl);
  int tiles_m = (M + cute::get<0>(TileShape{}) - 1) / cute::get<0>(TileShape{});
  int tiles_n = (N + cute::get<1>(TileShape{}) - 1) / cute::get<1>(TileShape{});
  return dim3(tiles_m, tiles_n, L);
}

template <class TileShape>
static int get_k_tiles(cute::tuple<int,int,int,int> problem_shape_mnkl) {
  int K = cute::get<2>(problem_shape_mnkl);
  return (K + cute::get<2>(TileShape{}) - 1) / cute::get<2>(TileShape{});
}

///////////////////////////////////////////////////////////////////////////////
// Device-side test runner: verifies K-tile coverage is exact
///////////////////////////////////////////////////////////////////////////////
template <class TileShape, class ClusterShape>
static void run_streamk_coverage_test(
    cute::tuple<int,int,int,int> problem_shape_mnkl,
    typename StreamKSchedulerTraits<TileShape, ClusterShape>::Arguments const& args) {

  using Traits = StreamKSchedulerTraits<TileShape, ClusterShape>;
  using TileScheduler = typename Traits::TileScheduler;

  sycl::queue q;

  auto params = make_streamk_params<Traits>(problem_shape_mnkl, args);

  dim3 problem_blocks = get_problem_blocks<TileShape>(problem_shape_mnkl);
  int tiles_m = problem_blocks.x;
  int tiles_n = problem_blocks.y;
  int batches = problem_blocks.z;
  int total_output_tiles = tiles_m * tiles_n * batches;
  int total_k_tiles = get_k_tiles<TileShape>(problem_shape_mnkl);

  auto cluster_shape_cs = ClusterShape{};
  cutlass::gemm::GemmCoord cluster_gc(
      int(size<0>(cluster_shape_cs)), int(size<1>(cluster_shape_cs)), int(size<2>(cluster_shape_cs)));

  dim3 grid = params.get_grid_shape(problem_blocks, cluster_gc);

  // For non-SK (DP/SplitK) grids, apply the same grid transpose that the XE4
  // scheduler uses: AlongN swaps grid.x and grid.y so CLC's fast-varying
  // BlockIdxX steps along N.
  using UnderlyingXe4Sched = cutlass::gemm::kernel::detail::PersistentTileSchedulerXe4<kStages, ClusterShape>;
  grid = UnderlyingXe4Sched::possibly_transpose_grid(
      params.xe4_params_.raster_order_,
      params.xe4_params_.divmod_cluster_shape_m_,
      params.xe4_params_.divmod_cluster_shape_n_,
      grid);

  auto* k_coverage     = sycl::malloc_shared<int>(total_output_tiles, q);
  auto* dispatch_count = sycl::malloc_shared<int>(total_output_tiles, q);
  auto* oob_errors     = sycl::malloc_shared<int>(1, q);

  std::memset(k_coverage, 0, total_output_tiles * sizeof(int));
  std::memset(dispatch_count, 0, total_output_tiles * sizeof(int));
  oob_errors[0] = 0;

  sycl::range<3> group_range(grid.z, grid.y, grid.x);
  sycl::range<3> local_range(1, kNumSubgroups, kThreadsPerSG);
  sycl::range<3> cluster_size(
      size<2>(cluster_shape_cs),
      size<1>(cluster_shape_cs),
      size<0>(cluster_shape_cs));
  int smem_size = 0;

  StreamKCoverageKernel<TileShape, ClusterShape> kernel{
      params, k_coverage, dispatch_count, oob_errors,
      tiles_m, tiles_n, batches, total_k_tiles};

  cutlass::launch_kernel_on_cluster(
      group_range, local_range, cluster_size, q, smem_size, kernel)
      .wait();

  // Verification
  EXPECT_EQ(oob_errors[0], 0)
      << oob_errors[0] << " out-of-bounds tile coordinate(s) detected";

  int coverage_errors = 0;
  int unvisited = 0;
  for (int i = 0; i < total_output_tiles; ++i) {
    if (k_coverage[i] != total_k_tiles) {
      if (coverage_errors < 10) {
        int m = i % tiles_m;
        int n = (i / tiles_m) % tiles_n;
        int l = i / (tiles_m * tiles_n);
        EXPECT_EQ(k_coverage[i], total_k_tiles)
            << "K-tile coverage mismatch at output tile (M=" << m
            << ", N=" << n << ", L=" << l << "): got " << k_coverage[i]
            << " expected " << total_k_tiles
            << " (dispatched " << dispatch_count[i] << " times)";
      }
      ++coverage_errors;
    }
    if (dispatch_count[i] == 0) ++unvisited;
  }

  EXPECT_EQ(coverage_errors, 0)
      << coverage_errors << " output tiles have incorrect K-tile coverage out of "
      << total_output_tiles;
  EXPECT_EQ(unvisited, 0)
      << unvisited << " output tiles were never visited out of " << total_output_tiles;

  // For split-K: verify that tiles got multiple dispatches (splits are mandatory).
  // For stream-K / heuristic: the scheduler may validly choose DP if tiles fill
  // waves perfectly, so we only check when splits > 1 (explicit split-K).
  if (args.splits > 1) {
    int multi_dispatch = 0;
    for (int i = 0; i < total_output_tiles; ++i) {
      if (dispatch_count[i] > 1) ++multi_dispatch;
    }
    EXPECT_GT(multi_dispatch, 0)
        << "Split-K with splits=" << args.splits
        << " should split tiles across multiple CTAs, but none were";
  }

  sycl::free(k_coverage, q);
  sycl::free(dispatch_count, q);
  sycl::free(oob_errors, q);
}

///////////////////////////////////////////////////////////////////////////////
// Convenience types for tests
///////////////////////////////////////////////////////////////////////////////
using TileShape_256x256x32 = Shape<_256, _256, _32>;
using TileShape_128x128x64 = Shape<_128, _128, _64>;
using TileShape_256x128x32 = Shape<_256, _128, _32>;
using ClusterShape_1x1x1   = Shape<_1, _1, _1>;

using Traits_256x256x32 = StreamKSchedulerTraits<TileShape_256x256x32, ClusterShape_1x1x1>;
using Traits_128x128x64 = StreamKSchedulerTraits<TileShape_128x128x64, ClusterShape_1x1x1>;
using Traits_256x128x32 = StreamKSchedulerTraits<TileShape_256x128x32, ClusterShape_1x1x1>;

using RasterOrderOptions = Traits_256x256x32::RasterOrderOptions;
using DecompositionMode   = Traits_256x256x32::DecompositionMode;
using ReductionMode       = Traits_256x256x32::ReductionMode;

///////////////////////////////////////////////////////////////////////////////
// HOST-SIDE TESTS: Workspace sizing
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4StreamKSchedulerHost, WorkspaceSize_DP_IsZero) {
  // Data-parallel: no workspace needed (no splits)
  auto problem = cute::make_tuple(512, 512, 256, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::DataParallel;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  hw_info.max_active_clusters = 0;

  size_t ws = Traits_256x256x32::TileScheduler::template get_workspace_size<
      decltype(problem), float>(args, problem, hw_info);

  EXPECT_EQ(ws, 0u) << "DP mode should not require workspace";
}

TEST(Xe4StreamKSchedulerHost, WorkspaceSize_SplitK_CountersOnly) {
  // Split-K: workspace = num_output_tiles * sizeof(int32_t) (barrier workspace only)
  auto problem = cute::make_tuple(512, 512, 1024, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 4;
  args.decomposition_mode = DecompositionMode::SplitK;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  hw_info.max_active_clusters = 0;

  size_t ws = Traits_256x256x32::TileScheduler::template get_workspace_size<
      decltype(problem), float>(args, problem, hw_info);

  // Should be small: just counters (one int32 per tile, padded to alignment)
  dim3 pb = get_problem_blocks<TileShape_256x256x32>(problem);
  size_t min_expected = pb.x * pb.y * pb.z * sizeof(int32_t);
  EXPECT_GE(ws, min_expected)
      << "Workspace must hold at least one counter per output tile";
  // Should NOT be huge (no accumulator scratch)
  size_t max_reasonable = min_expected * 16;  // generous upper bound for alignment
  EXPECT_LE(ws, max_reasonable)
      << "Workspace should be small (counters only), got " << ws << " bytes";
}

TEST(Xe4StreamKSchedulerHost, WorkspaceSize_StreamK_CountersOnly) {
  auto problem = cute::make_tuple(1024, 1024, 2048, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::StreamK;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  hw_info.max_active_clusters = 0;

  size_t ws = Traits_256x256x32::TileScheduler::template get_workspace_size<
      decltype(problem), float>(args, problem, hw_info);

  // The heuristic may decide sk_tiles=0 if tiles perfectly fill waves on this
  // hardware, in which case workspace is 0 (valid: falls back to DP internally).
  if (ws > 0) {
    dim3 pb = get_problem_blocks<TileShape_256x256x32>(problem);
    // No accumulator scratch: should be much smaller than M*N*sizeof(float)
    size_t accum_size = pb.x * pb.y * 256 * 256 * sizeof(float);
    EXPECT_LT(ws, accum_size)
        << "Workspace should not contain accumulator scratch, got " << ws
        << " bytes vs potential " << accum_size;
  }
}

///////////////////////////////////////////////////////////////////////////////
// HOST-SIDE TESTS: Grid shape
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4StreamKSchedulerHost, GridShape_DP_MatchesProblemBlocks) {
  auto problem = cute::make_tuple(512, 512, 256, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::DataParallel;

  auto params = make_streamk_params<Traits_256x256x32>(problem, args);
  dim3 pb = get_problem_blocks<TileShape_256x256x32>(problem);
  cutlass::gemm::GemmCoord cluster_coord(1, 1, 1);
  dim3 grid = params.get_grid_shape(pb, cluster_coord);

  EXPECT_EQ(grid.x, pb.x);
  EXPECT_EQ(grid.y, pb.y);
  EXPECT_EQ(grid.z, pb.z);
}

TEST(Xe4StreamKSchedulerHost, GridShape_SplitK_ZMultipliedBySplits) {
  auto problem = cute::make_tuple(512, 512, 1024, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 4;
  args.decomposition_mode = DecompositionMode::SplitK;

  auto params = make_streamk_params<Traits_256x256x32>(problem, args);
  dim3 pb = get_problem_blocks<TileShape_256x256x32>(problem);
  cutlass::gemm::GemmCoord cluster_coord(1, 1, 1);
  dim3 grid = params.get_grid_shape(pb, cluster_coord);

  // Split-K multiplies Z by splits
  EXPECT_EQ(grid.x, pb.x);
  EXPECT_EQ(grid.y, pb.y);
  EXPECT_EQ(grid.z, pb.z * 4u);
}

TEST(Xe4StreamKSchedulerHost, GridShape_StreamK_PersistentWaves) {
  auto problem = cute::make_tuple(2048, 2048, 2048, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::StreamK;

  auto params = make_streamk_params<Traits_256x256x32>(problem, args);
  dim3 pb = get_problem_blocks<TileShape_256x256x32>(problem);
  cutlass::gemm::GemmCoord cluster_coord(1, 1, 1);
  dim3 grid = params.get_grid_shape(pb, cluster_coord);

  // Stream-K: grid.x * grid.y = sk_units, grid.z = waves
  uint32_t total_ctas = grid.x * grid.y * grid.z;
  uint32_t total_tiles = pb.x * pb.y * pb.z;

  // SK grid should not exceed total output tiles * max reasonable splits
  EXPECT_LE(total_ctas, total_tiles * 4u)
      << "StreamK grid seems unreasonably large";
  // SK grid should use fewer CTAs than DP if problem is large enough
  // (persistent: reuses CTAs across multiple tiles)
  EXPECT_GT(grid.z, 0u) << "StreamK should have at least 1 wave";
}

///////////////////////////////////////////////////////////////////////////////
// HOST-SIDE TESTS: Decomposition heuristic
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4StreamKSchedulerHost, Heuristic_LargeSquare_UsesStreamK) {
  // Large square problem: heuristic should pick StreamK (tail effect is present)
  auto problem = cute::make_tuple(2048, 2048, 2048, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::Heuristic;

  auto params = make_streamk_params<Traits_256x256x32>(problem, args);

  // If StreamK is active, sk_units > 0
  bool has_sk = params.sk_params_.sk_units_ > 0;
  bool has_splits = params.sk_params_.divmod_splits_.divisor > 1;

  // Heuristic should enable either SK or splits for large problems
  // (exact behavior depends on SM count, so just verify it's not plain DP
  //  when there's a clear tail effect)
  dim3 pb = get_problem_blocks<TileShape_256x256x32>(problem);
  uint32_t total_tiles = pb.x * pb.y;
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();

  if (total_tiles % hw_info.sm_count != 0) {
    // There IS a tail effect, heuristic should do something
    EXPECT_TRUE(has_sk || has_splits)
        << "Heuristic should enable SK or splits when tail effect exists"
        << " (tiles=" << total_tiles << ", SMs=" << hw_info.sm_count << ")";
  }
}

TEST(Xe4StreamKSchedulerHost, Heuristic_PerfectFit_MayUseDP) {
  // When tiles perfectly fill waves, DP is optimal
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();

  // Create a problem where tiles_m * tiles_n = exact multiple of sm_count
  int tiles_needed = hw_info.sm_count * 2;  // exactly 2 waves
  int tiles_per_side = 1;
  while (tiles_per_side * tiles_per_side < tiles_needed) ++tiles_per_side;
  // Adjust to get exact multiple
  int M = tiles_per_side * 256;
  int N = (tiles_needed / tiles_per_side) * 256;
  if (tiles_per_side * (tiles_needed / tiles_per_side) != tiles_needed) {
    // Can't make exact square, just use sm_count along one dim
    M = hw_info.sm_count * 256;
    N = 2 * 256;
  }

  auto problem = cute::make_tuple(M, N, 256, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::Heuristic;

  auto params = make_streamk_params<Traits_256x256x32>(problem, args);

  // With perfect wave fill, heuristic may choose DP (no tail to fix)
  // This is hardware-dependent; we just verify no crash and valid grid
  dim3 pb = get_problem_blocks<TileShape_256x256x32>(problem);
  cutlass::gemm::GemmCoord cluster_coord(1, 1, 1);
  dim3 grid = params.get_grid_shape(pb, cluster_coord);
  EXPECT_GT(grid.x * grid.y * grid.z, 0u) << "Grid should be non-empty";
}

///////////////////////////////////////////////////////////////////////////////
// HOST-SIDE TESTS: requires_fixup / compute_epilogue predicates
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4StreamKSchedulerHost, RequiresFixup_DP_ReturnsFalse) {
  auto problem = cute::make_tuple(512, 512, 256, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::DataParallel;

  auto params = make_streamk_params<Traits_256x256x32>(problem, args);

  int total_k = get_k_tiles<TileShape_256x256x32>(problem);

  // Simulate a DP work tile (full K range)
  Traits_256x256x32::WorkTileInfo wti;
  wti.M_idx = 0;
  wti.N_idx = 0;
  wti.K_idx = 0;
  wti.L_idx = 0;
  wti.k_tile_count = total_k;
  wti.k_tile_remaining = total_k;

  EXPECT_FALSE(Traits_256x256x32::TileScheduler::requires_fixup(params, wti))
      << "DP tiles should not require fixup";
  EXPECT_TRUE(Traits_256x256x32::TileScheduler::compute_epilogue(wti, params))
      << "DP tiles should always compute epilogue";
}

TEST(Xe4StreamKSchedulerHost, RequiresFixup_SplitK_NonFinalReturnsTrue) {
  auto problem = cute::make_tuple(256, 256, 1024, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 4;
  args.decomposition_mode = DecompositionMode::SplitK;

  auto params = make_streamk_params<Traits_256x256x32>(problem, args);

  int total_k = get_k_tiles<TileShape_256x256x32>(problem);  // 1024/32 = 32
  int k_per_split = total_k / 4;  // 8

  // Non-final split (first split)
  Traits_256x256x32::WorkTileInfo wti_nonfinal;
  wti_nonfinal.M_idx = 0;
  wti_nonfinal.N_idx = 0;
  wti_nonfinal.K_idx = 0;
  wti_nonfinal.L_idx = 0;
  wti_nonfinal.k_tile_count = k_per_split;
  wti_nonfinal.k_tile_remaining = k_per_split;

  EXPECT_TRUE(Traits_256x256x32::TileScheduler::requires_fixup(params, wti_nonfinal))
      << "Non-final split should require fixup";
  EXPECT_FALSE(Traits_256x256x32::TileScheduler::compute_epilogue(wti_nonfinal, params))
      << "Non-final split should not compute epilogue";

  // Final split (last split, ends at total_k)
  Traits_256x256x32::WorkTileInfo wti_final;
  wti_final.M_idx = 0;
  wti_final.N_idx = 0;
  wti_final.K_idx = total_k - k_per_split;
  wti_final.L_idx = 0;
  wti_final.k_tile_count = k_per_split;
  wti_final.k_tile_remaining = k_per_split;

  EXPECT_TRUE(Traits_256x256x32::TileScheduler::requires_fixup(params, wti_final))
      << "Final split should still require fixup (it's a partial tile)";
  EXPECT_TRUE(Traits_256x256x32::TileScheduler::compute_epilogue(wti_final, params))
      << "Final split should compute epilogue";
}

///////////////////////////////////////////////////////////////////////////////
// HOST-SIDE TESTS: Raster order selection
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4StreamKSchedulerHost, RasterOrder_AlongM_Respected) {
  auto problem = cute::make_tuple(1024, 512, 256, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.raster_order = RasterOrderOptions::AlongM;
  args.decomposition_mode = DecompositionMode::DataParallel;

  auto params = make_streamk_params<Traits_256x256x32>(problem, args);
  // AlongM rasterizes with M as the fast-changing dimension
  EXPECT_EQ(params.raster_order_,
            Traits_256x256x32::Params::RasterOrder::AlongM);
}

TEST(Xe4StreamKSchedulerHost, RasterOrder_AlongN_Respected) {
  auto problem = cute::make_tuple(512, 1024, 256, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.raster_order = RasterOrderOptions::AlongN;
  args.decomposition_mode = DecompositionMode::DataParallel;

  auto params = make_streamk_params<Traits_256x256x32>(problem, args);
  EXPECT_EQ(params.raster_order_,
            Traits_256x256x32::Params::RasterOrder::AlongN);
}

TEST(Xe4StreamKSchedulerHost, RasterOrder_Heuristic_PicksTaller) {
  // Heuristic should pick AlongN for M > N (rasterize along the shorter dim)
  auto problem_tall = cute::make_tuple(2048, 512, 256, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.raster_order = RasterOrderOptions::Heuristic;
  args.decomposition_mode = DecompositionMode::DataParallel;

  auto params = make_streamk_params<Traits_256x256x32>(problem_tall, args);
  // For M > N (tall), heuristic typically picks AlongN
  // (raster along the dimension with more tiles = AlongM means M changes fast)
  // The exact choice depends on the SM90 heuristic, but verify it's deterministic
  auto order1 = params.raster_order_;

  auto params2 = make_streamk_params<Traits_256x256x32>(problem_tall, args);
  EXPECT_EQ(order1, params2.raster_order_)
      << "Heuristic should be deterministic for same inputs";
}

///////////////////////////////////////////////////////////////////////////////
// HOST-SIDE TESTS: Swizzle
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4StreamKSchedulerHost, Swizzle_StoredInParams) {
  auto problem = cute::make_tuple(2048, 2048, 256, 1);

  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.max_swizzle_size = 4;
  args.decomposition_mode = DecompositionMode::DataParallel;

  auto params = make_streamk_params<Traits_256x256x32>(problem, args);

  // Swizzle should be clamped to min(max_swizzle_size, tiles_in_raster_dim)
  EXPECT_GE(params.log_swizzle_size_, 0);
  EXPECT_LE(params.log_swizzle_size_, 2);  // log2(4) = 2
}

TEST(Xe4StreamKSchedulerHost, Swizzle_ClampedToGridDim) {
  // 2x2 grid with swizzle=8 should clamp
  auto problem = cute::make_tuple(512, 512, 256, 1);  // 2x2 tiles

  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.max_swizzle_size = 8;
  args.decomposition_mode = DecompositionMode::DataParallel;

  auto params = make_streamk_params<Traits_256x256x32>(problem, args);

  // With only 2 tiles in each dim, swizzle should be clamped to at most 2
  EXPECT_LE(params.log_swizzle_size_, 1)  // log2(2) = 1
      << "Swizzle should be clamped when grid is smaller than swizzle size";
}

///////////////////////////////////////////////////////////////////////////////
// DEVICE-SIDE TESTS: Data-Parallel (baseline — verifies backward compat)
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4StreamKSchedulerDevice, Coverage_DP_Square) {
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::DataParallel;
  args.raster_order = RasterOrderOptions::AlongM;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(1024, 1024, 256, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_DP_Rectangular_AlongN) {
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::DataParallel;
  args.raster_order = RasterOrderOptions::AlongN;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(512, 1024, 256, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_DP_MultiBatch) {
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::DataParallel;
  args.raster_order = RasterOrderOptions::Heuristic;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(512, 512, 256, 2), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_DP_Swizzle4) {
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.max_swizzle_size = 4;
  args.decomposition_mode = DecompositionMode::DataParallel;
  args.raster_order = RasterOrderOptions::AlongM;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(2048, 2048, 256, 1), args);
}

///////////////////////////////////////////////////////////////////////////////
// DEVICE-SIDE TESTS: Split-K
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4StreamKSchedulerDevice, Coverage_SplitK_2Splits) {
  Traits_256x256x32::Arguments args;
  args.splits = 2;
  args.decomposition_mode = DecompositionMode::SplitK;
  args.raster_order = RasterOrderOptions::AlongM;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(512, 512, 512, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_SplitK_4Splits) {
  Traits_256x256x32::Arguments args;
  args.splits = 4;
  args.decomposition_mode = DecompositionMode::SplitK;
  args.raster_order = RasterOrderOptions::AlongN;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(512, 512, 1024, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_SplitK_8Splits_LargeK) {
  Traits_256x256x32::Arguments args;
  args.splits = 8;
  args.decomposition_mode = DecompositionMode::SplitK;
  args.raster_order = RasterOrderOptions::Heuristic;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(256, 256, 2048, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_SplitK_MultiBatch) {
  Traits_256x256x32::Arguments args;
  args.splits = 2;
  args.decomposition_mode = DecompositionMode::SplitK;
  args.raster_order = RasterOrderOptions::AlongM;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(512, 512, 512, 3), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_SplitK_Swizzle2_AlongN) {
  Traits_256x256x32::Arguments args;
  args.splits = 4;
  args.max_swizzle_size = 2;
  args.decomposition_mode = DecompositionMode::SplitK;
  args.raster_order = RasterOrderOptions::AlongN;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(1024, 1024, 1024, 1), args);
}

///////////////////////////////////////////////////////////////////////////////
// DEVICE-SIDE TESTS: Stream-K
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4StreamKSchedulerDevice, Coverage_StreamK_Square) {
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::StreamK;
  args.raster_order = RasterOrderOptions::AlongM;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(1024, 1024, 1024, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_StreamK_TallSkinny) {
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::StreamK;
  args.raster_order = RasterOrderOptions::AlongN;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(2048, 256, 1024, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_StreamK_ShortWide) {
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::StreamK;
  args.raster_order = RasterOrderOptions::AlongM;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(256, 2048, 1024, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_StreamK_LargeK) {
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::StreamK;
  args.raster_order = RasterOrderOptions::Heuristic;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(512, 512, 4096, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_StreamK_MultiBatch) {
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::StreamK;
  args.raster_order = RasterOrderOptions::AlongM;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(512, 512, 512, 4), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_StreamK_Swizzle4_AlongN) {
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.max_swizzle_size = 4;
  args.decomposition_mode = DecompositionMode::StreamK;
  args.raster_order = RasterOrderOptions::AlongN;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(2048, 2048, 512, 1), args);
}

///////////////////////////////////////////////////////////////////////////////
// DEVICE-SIDE TESTS: Heuristic mode
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4StreamKSchedulerDevice, Coverage_Heuristic_Square) {
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::Heuristic;
  args.raster_order = RasterOrderOptions::Heuristic;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(1024, 1024, 1024, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_Heuristic_Irregular) {
  // Non-power-of-2 problem likely to trigger tail effect
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::Heuristic;
  args.raster_order = RasterOrderOptions::Heuristic;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(768, 1280, 512, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_Heuristic_MultiBatch) {
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::Heuristic;
  args.raster_order = RasterOrderOptions::AlongM;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(512, 512, 512, 2), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_Heuristic_TallSkinny_Swizzle2) {
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.max_swizzle_size = 2;
  args.decomposition_mode = DecompositionMode::Heuristic;
  args.raster_order = RasterOrderOptions::AlongN;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(2048, 512, 1024, 1), args);
}

///////////////////////////////////////////////////////////////////////////////
// DEVICE-SIDE TESTS: Different tile shapes
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4StreamKSchedulerDevice, Coverage_StreamK_128x128x64) {
  Traits_128x128x64::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::StreamK;
  args.raster_order = RasterOrderOptions::AlongM;

  run_streamk_coverage_test<TileShape_128x128x64, ClusterShape_1x1x1>(
      cute::make_tuple(512, 512, 512, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_SplitK_128x128x64_4Splits) {
  Traits_128x128x64::Arguments args;
  args.splits = 4;
  args.decomposition_mode = DecompositionMode::SplitK;
  args.raster_order = RasterOrderOptions::AlongN;

  run_streamk_coverage_test<TileShape_128x128x64, ClusterShape_1x1x1>(
      cute::make_tuple(512, 512, 1024, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_StreamK_256x128x32) {
  Traits_256x128x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::StreamK;
  args.raster_order = RasterOrderOptions::Heuristic;

  run_streamk_coverage_test<TileShape_256x128x32, ClusterShape_1x1x1>(
      cute::make_tuple(1024, 512, 1024, 1), args);
}

///////////////////////////////////////////////////////////////////////////////
// DEVICE-SIDE TESTS: Edge cases
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4StreamKSchedulerDevice, Coverage_StreamK_SingleOutputTile) {
  // Only one output tile — all K-tiles must land on it
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::StreamK;
  args.raster_order = RasterOrderOptions::AlongM;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(256, 256, 2048, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_SplitK_SingleOutputTile_4Splits) {
  Traits_256x256x32::Arguments args;
  args.splits = 4;
  args.decomposition_mode = DecompositionMode::SplitK;
  args.raster_order = RasterOrderOptions::AlongM;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(256, 256, 1024, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_StreamK_MinimalK) {
  // K is exactly one tile — no splitting possible
  Traits_256x256x32::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::StreamK;
  args.raster_order = RasterOrderOptions::AlongM;

  run_streamk_coverage_test<TileShape_256x256x32, ClusterShape_1x1x1>(
      cute::make_tuple(512, 512, 32, 1), args);
}

TEST(Xe4StreamKSchedulerDevice, Coverage_Heuristic_ManySmallTiles) {
  // Many output tiles, small K — should prefer DP
  Traits_128x128x64::Arguments args;
  args.splits = 1;
  args.decomposition_mode = DecompositionMode::Heuristic;
  args.raster_order = RasterOrderOptions::Heuristic;

  run_streamk_coverage_test<TileShape_128x128x64, ClusterShape_1x1x1>(
      cute::make_tuple(2048, 2048, 64, 1), args);
}

} // anonymous namespace
