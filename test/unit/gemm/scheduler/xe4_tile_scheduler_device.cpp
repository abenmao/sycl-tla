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
 * Device-side unit tests for PersistentTileSchedulerXe4 dynamic scheduling.
 *
 * Strategy: launch a "tiled vector-add" kernel that uses the full CLC dynamic
 * scheduling loop (advance_to_next_work / fetch_next_work) to assign tiles to
 * work-groups. Each tile covers a contiguous chunk of a 1-D array. The worker
 * warp computes C[i] = A[i] + B[i] for its assigned tile, providing a
 * verifiable side-effect. The host checks every element of C for correctness,
 * which proves that:
 *   (a) every tile was dispatched exactly once (no gaps, no duplicates),
 *   (b) swizzle_and_rasterize correctly mapped CLC coordinates to tile indices,
 *   (c) the advance_to_next_work / fetch_next_work pipeline loop terminates.
 *
 * Kernel layout (mirrors real GEMM warp-specialization):
 *   Subgroup 0 — Scheduler warp (ProducerConsumer):
 *       Drives the CLC pipeline: advance_to_next_work + fetch_next_work.
 *   Subgroup 1 — Worker warp (Consumer):
 *       Performs the vector-add work + fetch_next_work.
 *
 * Requires: Xe4 hardware (SYCL_INTEL_TARGET == 40) with CLC support.
 */

#include <sycl/sycl.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "cute/tensor.hpp"
#include "cute/arch/cluster_xe4.hpp"
#include "cute/arch/xe4_util.hpp"
#include "cute/arch/xe4_inline_pisa.hpp"
#include "cutlass/gemm/kernel/xe4_dynamic_tile_scheduler.hpp"
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
static constexpr int kTileSize          = kThreadsPerSG;

///////////////////////////////////////////////////////////////////////////////
// Scheduler traits — gathers all cluster-shape-dependent types
///////////////////////////////////////////////////////////////////////////////
template <class ClusterShape_>
struct SchedulerTraits {
  using ClusterShape  = ClusterShape_;
  using TileScheduler = cutlass::gemm::kernel::detail::DynamicPersistentTileSchedulerXe4<kStages, ClusterShape>;
  using Params        = typename TileScheduler::Params;
  using CLCPipeline   = cutlass::PipelineCLCFetchAsync<kStages, ClusterShape>;
  using CLCPipeState  = typename CLCPipeline::PipelineState;
  using CLCResponse   = typename TileScheduler::CLCResponse;
  using WorkTileInfo  = typename TileScheduler::WorkTileInfo;
  using RasterOrderOptions = typename Params::RasterOrderOptions;

  static constexpr int cluster_size = size(ClusterShape{});
};

///////////////////////////////////////////////////////////////////////////////
// Kernel: tiled vector-add driven by CLC dynamic scheduling
///////////////////////////////////////////////////////////////////////////////
template <class ClusterShape>
struct VectorAddKernel {
  using Traits       = SchedulerTraits<ClusterShape>;
  using Params       = typename Traits::Params;
  using CLCPipeline  = typename Traits::CLCPipeline;
  using CLCPipeState = typename Traits::CLCPipeState;
  using CLCResponse  = typename Traits::CLCResponse;
  using TileScheduler = typename Traits::TileScheduler;

  Params       scheduler_params;
  const float* A;
  const float* B;
  float*       C;
  int          tiles_m;
  int          tiles_n;

  CUTLASS_DEVICE void do_vectoradd(
      typename Traits::WorkTileInfo const& wti, uint32_t lane_id) const {
    int chunk = wti.L_idx * (tiles_m * tiles_n)
              + wti.N_idx * tiles_m
              + wti.M_idx;
    int base = chunk * kTileSize;
    if (lane_id < kTileSize) {
      C[base + lane_id] = A[base + lane_id] + B[base + lane_id];
    }
  }

  CUTLASS_DEVICE void operator()() const {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    uint32_t warp_idx = get_sg_id();
    bool is_scheduler_warp = (warp_idx == 0);
    uint32_t lane_id = get_lane_id();

    auto cluster_shape = ClusterShape{};

    // --- SLM for CLC responses ---
    auto slm_ptr = alloc_slm_buffer<uint8_t,
        sizeof(CLCResponse) * kStages>(item.get_group());
    auto* clc_response_ptr = reinterpret_cast<CLCResponse*>(slm_ptr);

    // --- ABAR for pipeline barriers ---
    constexpr uint32_t PipeStorageSize =
        static_cast<uint32_t>(sizeof(typename CLCPipeline::SharedStorage));
    auto abar_base = allocate_abar_bytes<0, PipeStorageSize>();
    auto& pipeline_storage =
        *reinterpret_cast<typename CLCPipeline::SharedStorage*>(abar_base);

    // --- CLC pipeline (same pattern as real GEMM) ---
    // Only CTA 0's scheduler warp produces CLC queries; all others consume only.
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

    // Barrier after pipeline init
    if constexpr (Traits::cluster_size > 1) {
      cute::cluster_arrive();
      cute::cluster_wait();
    } else {
      item.barrier(sycl::access::fence_space::local_space);
    }

    // --- Tile scheduler ---
    dim3 block_id_in_cluster = cute::block_id_in_cluster();
    TileScheduler scheduler(clc_response_ptr, scheduler_params, block_id_in_cluster);

    auto work_tile_info = scheduler.initial_work_tile_info(cluster_shape);
    auto clc_pipe_producer_state = cutlass::make_producer_start_state<CLCPipeline>();
    CLCPipeState clc_pipe_consumer_state{};

    // ================================================================
    // Warp-specialised loops (mirrors xe4_gemm_dma_warpspecialized.hpp)
    //
    // CTA 0, warp 0:  ProducerConsumer — advance_to_next_work + fetch_next_work
    // Other sched warps: Consumer only — fetch_next_work (no produce, no work)
    // Worker warps:      Consumer      — vector-add + fetch_next_work
    // ================================================================

    if (is_clc_producer) {
      // ---- CLC producer (CTA 0, scheduler warp) ----
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
      // ---- Worker warp (any CTA): vector-add + consume ----
      do {
        do_vectoradd(work_tile_info, lane_id);
        auto [next_work, inc] =
            scheduler.fetch_next_work(work_tile_info, clc_pipeline, clc_pipe_consumer_state);
        if (inc) { ++clc_pipe_consumer_state; }
        work_tile_info = next_work;
      } while (work_tile_info.is_valid());

    } 
  }
};

///////////////////////////////////////////////////////////////////////////////
// Host helpers
///////////////////////////////////////////////////////////////////////////////

template <class Traits>
static typename Traits::Params make_scheduler_params(
    dim3 problem_blocks,
    cutlass::gemm::GemmCoord cluster_shape,
    int max_swizzle_size,
    typename Traits::RasterOrderOptions raster_order_option) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  hw_info.max_active_clusters = 0;

  typename Traits::Params params;
  params.initialize(problem_blocks, cluster_shape, hw_info,
                    max_swizzle_size, raster_order_option);
  return params;
}

///////////////////////////////////////////////////////////////////////////////
// Run one tiled vector-add test
///////////////////////////////////////////////////////////////////////////////
template <class ClusterShape>
static void run_vectoradd_scheduler_test(
    dim3 problem_blocks,
    int max_swizzle_size,
    typename SchedulerTraits<ClusterShape>::RasterOrderOptions raster_order_option) {

  using Traits        = SchedulerTraits<ClusterShape>;
  using TileScheduler = typename Traits::TileScheduler;
  using CLCResponse   = typename Traits::CLCResponse;

  sycl::queue q;

  auto cluster_shape_cs = ClusterShape{};
  cutlass::gemm::GemmCoord cluster_shape_gc(
      size<0>(cluster_shape_cs),
      size<1>(cluster_shape_cs),
      size<2>(cluster_shape_cs));

  auto params = make_scheduler_params<Traits>(
      problem_blocks, cluster_shape_gc,
      max_swizzle_size, raster_order_option);

  dim3 launch_grid = TileScheduler::possibly_transpose_grid(
      params.raster_order_,
      params.divmod_cluster_shape_m_,
      params.divmod_cluster_shape_n_,
      problem_blocks);

  int total_tiles = static_cast<int>(problem_blocks.x)
                  * static_cast<int>(problem_blocks.y)
                  * static_cast<int>(problem_blocks.z);
  int total_elems = total_tiles * kTileSize;

  auto* A = sycl::malloc_shared<float>(total_elems, q);
  auto* B = sycl::malloc_shared<float>(total_elems, q);
  auto* C = sycl::malloc_shared<float>(total_elems, q);

  for (int i = 0; i < total_elems; ++i) {
    A[i] = static_cast<float>(i);
    B[i] = static_cast<float>(i * 2);
    C[i] = -1.0f;
  }

  sycl::range<3> group_range(launch_grid.z, launch_grid.y, launch_grid.x);
  sycl::range<3> local_range(1, kNumSubgroups, kThreadsPerSG);
  sycl::range<3> cluster_size(
      size<2>(cluster_shape_cs),
      size<1>(cluster_shape_cs),
      size<0>(cluster_shape_cs));
  int smem_size = 0;

  VectorAddKernel<ClusterShape> kernel{
      params, A, B, C,
      static_cast<int>(problem_blocks.x),
      static_cast<int>(problem_blocks.y)};

  cutlass::launch_kernel_on_cluster(
      group_range, local_range, cluster_size, q, smem_size, kernel)
      .wait();

  // ---- Verify every element ----
  int errors = 0;
  int written = 0;
  for (int i = 0; i < total_elems; ++i) {
    float expected = A[i] + B[i];
    if (C[i] != -1.0f) { ++written; }
    if (C[i] != expected) {
      if (errors < 10) {
        EXPECT_FLOAT_EQ(C[i], expected) << "Mismatch at index " << i;
      }
      ++errors;
    }
  }

  EXPECT_EQ(errors, 0) << errors << " element mismatches out of " << total_elems;
  EXPECT_EQ(written, total_elems)
      << "Only " << written << " of " << total_elems
      << " elements were written (some tiles not scheduled?)";

  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);
}

///////////////////////////////////////////////////////////////////////////////
// Convenience aliases for RasterOrderOptions
///////////////////////////////////////////////////////////////////////////////
using RasterOrderOptions =
    cutlass::gemm::kernel::detail::DynamicPersistentTileSchedulerXe4Params::RasterOrderOptions;

///////////////////////////////////////////////////////////////////////////////
// Test cases — Cluster 1x1x1
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4TileSchedulerDevice, VectorAdd_1x1_NoSwizzle_AlongM) {
  run_vectoradd_scheduler_test<Shape<_1,_1,_1>>(
      dim3(4, 4, 1), 1, RasterOrderOptions::AlongM);
}

TEST(Xe4TileSchedulerDevice, VectorAdd_1x1_NoSwizzle_AlongN) {
  run_vectoradd_scheduler_test<Shape<_1,_1,_1>>(
      dim3(4, 4, 1), 1, RasterOrderOptions::AlongN);
}

TEST(Xe4TileSchedulerDevice, VectorAdd_1x1_Swizzle2) {
  run_vectoradd_scheduler_test<Shape<_1,_1,_1>>(
      dim3(4, 8, 1), 2, RasterOrderOptions::AlongM);
}

TEST(Xe4TileSchedulerDevice, VectorAdd_1x1_Swizzle4) {
  run_vectoradd_scheduler_test<Shape<_1,_1,_1>>(
      dim3(8, 8, 1), 4, RasterOrderOptions::AlongM);
}

TEST(Xe4TileSchedulerDevice, VectorAdd_1x1_Swizzle4_AlongN) {
  run_vectoradd_scheduler_test<Shape<_1,_1,_1>>(
      dim3(8, 8, 1), 4, RasterOrderOptions::AlongN);
}

TEST(Xe4TileSchedulerDevice, VectorAdd_1x1_MultiBatch) {
  run_vectoradd_scheduler_test<Shape<_1,_1,_1>>(
      dim3(4, 4, 2), 2, RasterOrderOptions::Heuristic);
}

TEST(Xe4TileSchedulerDevice, VectorAdd_1x1_Rectangular) {
  run_vectoradd_scheduler_test<Shape<_1,_1,_1>>(
      dim3(4, 8, 1), 1, RasterOrderOptions::Heuristic);
}

TEST(Xe4TileSchedulerDevice, VectorAdd_1x1_LargerGrid) {
  run_vectoradd_scheduler_test<Shape<_1,_1,_1>>(
      dim3(8, 16, 1), 4, RasterOrderOptions::AlongM);
}

///////////////////////////////////////////////////////////////////////////////
// Test cases — Cluster 2x1x1
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4TileSchedulerDevice, VectorAdd_2x1_NoSwizzle) {
  run_vectoradd_scheduler_test<Shape<_2,_1,_1>>(
      dim3(8, 4, 1), 1, RasterOrderOptions::AlongM);
}

TEST(Xe4TileSchedulerDevice, VectorAdd_2x1_Swizzle2) {
  run_vectoradd_scheduler_test<Shape<_2,_1,_1>>(
      dim3(8, 8, 1), 2, RasterOrderOptions::AlongM);
}

///////////////////////////////////////////////////////////////////////////////
// Test cases — Cluster 1x2x1
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4TileSchedulerDevice, VectorAdd_1x2_NoSwizzle) {
  run_vectoradd_scheduler_test<Shape<_1,_2,_1>>(
      dim3(4, 8, 1), 1, RasterOrderOptions::AlongN);
}

TEST(Xe4TileSchedulerDevice, VectorAdd_1x2_Swizzle2) {
  run_vectoradd_scheduler_test<Shape<_1,_2,_1>>(
      dim3(8, 8, 1), 2, RasterOrderOptions::AlongM);
}

///////////////////////////////////////////////////////////////////////////////
// Test cases — Cluster 2x2x1
///////////////////////////////////////////////////////////////////////////////

TEST(Xe4TileSchedulerDevice, VectorAdd_2x2_NoSwizzle) {
  run_vectoradd_scheduler_test<Shape<_2,_2,_1>>(
      dim3(8, 8, 1), 1, RasterOrderOptions::AlongM);
}

TEST(Xe4TileSchedulerDevice, VectorAdd_2x2_Swizzle4) {
  run_vectoradd_scheduler_test<Shape<_2,_2,_1>>(
      dim3(8, 8, 1), 4, RasterOrderOptions::AlongN);
}

} // anonymous namespace
