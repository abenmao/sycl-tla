#pragma once

#include "cute/tensor.hpp"
#include "cute/arch/copy_xe4_dma_legacy.hpp"
#include "cute/arch/xe4_inline_pisa.hpp"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm_coord.hpp"
#include "cutlass/gemm/kernel/xe4_static_tile_scheduler.hpp"

namespace cutlass::gemm::kernel::detail {

using namespace cute;

template<uint32_t Stages_, class ClusterShape>
class PersistentTileSchedulerXe4 {
public:
  static_assert(cute::is_static_v<ClusterShape>);

  using UnderlyingTileScheduler = StaticPersistentTileSchedulerXe4<ClusterShape>;

  static constexpr uint32_t CLC_VS = 4;
  struct alignas(16) CLCResponse { uint32_t data[CLC_VS]; };

  static constexpr uint32_t Stages = Stages_;
  static constexpr bool IsDynamicPersistent = true;

  using Pipeline = cutlass::PipelineCLCFetchAsync<Stages, ClusterShape>;
  using PipelineState = typename Pipeline::PipelineState;
  using WorkTileInfo = typename UnderlyingTileScheduler::WorkTileInfo;
  using Params = PersistentTileSchedulerXe4Params;
  using Arguments = typename UnderlyingTileScheduler::Arguments;
  using RasterOrder = typename UnderlyingTileScheduler::RasterOrder;
  using RasterOrderOptions = typename UnderlyingTileScheduler::RasterOrderOptions;
  static constexpr uint32_t cluster_size = size(ClusterShape{});
  static constexpr bool is_cluster = cluster_size > 1;

  template <class ProblemShapeMNKL, class TileShape>
  static auto
  calculate_problem_blocks_shape(ProblemShapeMNKL problem_shape_mnkl, TileShape tile_shape) {
    return UnderlyingTileScheduler::calculate_problem_blocks_shape(problem_shape_mnkl, tile_shape);
  }

  template <class ProblemShapeMNKL, class TileShape, class ClusterShapeMNK>
  static Params
  to_underlying_arguments(
    ProblemShapeMNKL problem_shape_mnkl,
    TileShape tile_shape,
    [[maybe_unused]] ClusterShapeMNK cluster_shape,
    KernelHardwareInfo const& hw_info,
    Arguments const& args,
    void* workspace = nullptr,
    uint32_t epilogue_subtile = 1,
    uint32_t ktile_start_alignment_count = 1u) {
    CUTLASS_UNUSED(workspace);
    CUTLASS_UNUSED(epilogue_subtile);
    CUTLASS_UNUSED(ktile_start_alignment_count);

    auto cs = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cs);

    Params params;
    params.initialize(
      problem_blocks,
      to_gemm_coord(cs),
      hw_info,
      args.max_swizzle_size,
      args.raster_order);
    return params;
  }

  template <class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShapeMNK>
  static Params
  to_underlying_arguments(
    ProblemShapeMNKL problem_shape_mnkl,
    TileShape tile_shape_mnk,
    AtomThrShape atom_thr_shape_mnk,
    ClusterShapeMNK cluster_shape_mnk,
    KernelHardwareInfo const& hw_info,
    Arguments const& args,
    void* workspace = nullptr) {
    CUTLASS_UNUSED(workspace);

    auto selected_cluster_shape = cutlass::detail::select_cluster_shape(cluster_shape_mnk, hw_info.cluster_shape);
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape_mnk, atom_thr_shape_mnk, selected_cluster_shape);

    Params params;
    params.initialize(
      problem_blocks,
      to_gemm_coord(selected_cluster_shape),
      hw_info,
      args.max_swizzle_size,
      args.raster_order);
    return params;
  }

  template<class ProblemShapeMNKL, class BlockShape, class ClusterShapeMNK>
  CUTLASS_HOST_DEVICE static dim3
  get_tiled_cta_shape_mnl(ProblemShapeMNKL problem_shape_mnkl, BlockShape blk_shape, ClusterShapeMNK cluster_shape) {
    auto grid_shape    = shape(ceil_div(problem_shape_mnkl, blk_shape));
    auto grid_shape_up = round_up(product_each(grid_shape), cluster_shape);
    return dim3(size<0>(grid_shape_up), size<1>(grid_shape_up), size<3>(grid_shape_up));
  }

  template<class ProblemShapeMNKL, class TileShapeMNK, class AtomThrShape, class ClusterShapeMNK>
  CUTLASS_HOST_DEVICE static dim3
  get_tiled_cta_shape_mnl(ProblemShapeMNKL problem_shape_mnkl,
                          TileShapeMNK tile_shape_mnk,
                          AtomThrShape atom_thr_shape_mnk,
                          ClusterShapeMNK cluster_shape_mnk) {
    auto [tiles_m, tiles_n, tiles_l] = product_each(ceil_div(select<0,1,3>(problem_shape_mnkl), take<0,2>(tile_shape_mnk)));
    auto ctas_m = round_nearest(tiles_m * size<0>(atom_thr_shape_mnk), size<0>(cluster_shape_mnk));
    auto ctas_n = round_nearest(tiles_n * size<1>(atom_thr_shape_mnk), size<1>(cluster_shape_mnk));
    auto ctas_l = tiles_l;

    return {static_cast<uint32_t>(ctas_m),
            static_cast<uint32_t>(ctas_n),
            static_cast<uint32_t>(ctas_l)};
  }

  template <class ProblemShapeMNKL, class BlockShape, class ClusterShapeMNK>
  CUTLASS_HOST_DEVICE
  static dim3
  get_grid_shape(
    Params const& params,
    ProblemShapeMNKL problem_shape_mnk,
    BlockShape cta_shape,
    ClusterShapeMNK cluster_shape,
    [[maybe_unused]] KernelHardwareInfo hw_info,
    [[maybe_unused]] Arguments arguments) {
    auto problem_shape_MNKL = append<4>(problem_shape_mnk, Int<1>{});
    auto grid = get_tiled_cta_shape_mnl(problem_shape_MNKL, cta_shape, cluster_shape);
    return possibly_transpose_grid(params.raster_order_, params.divmod_cluster_shape_m_, params.divmod_cluster_shape_n_, grid);
  }

  // For RasterOrder::AlongN, swap grid.x (M) and grid.y (N) so that hardware's
  // fast-varying BlockIdxX steps along N. Swap operates at cluster granularity to
  // preserve cluster-aligned grid dimensions.
  CUTLASS_HOST_DEVICE
  static dim3
  possibly_transpose_grid(RasterOrder raster_order, FastDivmod divmod_cluster_shape_m, FastDivmod divmod_cluster_shape_n, dim3 grid) {
    if (raster_order == RasterOrder::AlongN) {
      auto tmp = grid.x;
      grid.x = divmod_cluster_shape_n.divide(grid.y) * divmod_cluster_shape_m;
      grid.y = divmod_cluster_shape_m.divide(tmp) * divmod_cluster_shape_n;
    }
    return grid;
  }

  template <class ProblemShape, class ElementAccumulator>
  static size_t
  get_workspace_size(
    Arguments const& args,
    ProblemShape problem_shape,
    KernelHardwareInfo const& hw_info,
    uint32_t reduction_warp_groups,
    const uint32_t epilogue_subtile = 1,
    uint32_t num_accumulator_mtxs = 1) {
    return UnderlyingTileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args,
      problem_shape,
      hw_info,
      reduction_warp_groups,
      epilogue_subtile,
      num_accumulator_mtxs);
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(
    Arguments const& args,
    void* workspace,
    cudaStream_t stream,
    ProblemShape const& problem_shape,
    KernelHardwareInfo const& hw_info,
    uint32_t reduction_warp_groups,
    uint32_t epilogue_subtile = 1,
    uint32_t num_accumulator_mtxs = 1,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return UnderlyingTileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
      args,
      workspace,
      stream,
      problem_shape,
      hw_info,
      reduction_warp_groups,
      epilogue_subtile,
      num_accumulator_mtxs,
      cuda_adapter);
  }

  static bool
  can_implement(Arguments const& args) {
    return UnderlyingTileScheduler::can_implement(args);
  }

  CUTLASS_DEVICE
  explicit
  PersistentTileSchedulerXe4(Params const& params)
    : params_(params) {}

  CUTLASS_DEVICE
  PersistentTileSchedulerXe4(CLCResponse* clc_response_ptr, Params const& params)
    : clc_response_ptr_(clc_response_ptr), params_(params) {}

  CUTLASS_DEVICE
  PersistentTileSchedulerXe4(CLCResponse* clc_response_ptr, Params const& params, dim3 block_id_in_cluster)
    : clc_response_ptr_(clc_response_ptr), params_(params), block_id_in_cluster_(block_id_in_cluster) {}

  template <class ProblemShapeMNKL, class TileShape>
  CUTLASS_DEVICE
  PersistentTileSchedulerXe4(CLCResponse* clc_response_ptr, Params const& params, ProblemShapeMNKL, TileShape, dim3 block_id_in_cluster)
    : PersistentTileSchedulerXe4(clc_response_ptr, params, block_id_in_cluster) {}

  // For RasterOrder::AlongN, the grid was launched with M and N swapped (see possibly_transpose_grid),
  // so BlockIdxX carries a raw N cluster index and BlockIdxY carries a raw M cluster index.
  // This function corrects for that by swapping the cluster-level indices while keeping each
  // CTA's intra-cluster remainder in its own dimension (remainder_m stays in M, remainder_n in N).
  template <class DivmodM, class DivmodN>
  CUTLASS_DEVICE
  static cute::tuple<int32_t, int32_t>
  possibly_transpose_work_tile(RasterOrder raster_order, int32_t M_idx, int32_t N_idx, DivmodM divmod_cluster_shape_m, DivmodN divmod_cluster_shape_n) {
    if (raster_order == RasterOrder::AlongN) {
      int cluster_m, remainder_m, cluster_n, remainder_n;
      divmod_cluster_shape_m(cluster_m, remainder_m, M_idx);
      divmod_cluster_shape_n(cluster_n, remainder_n, N_idx);
      M_idx = cluster_n * divmod_cluster_shape_m.divisor + remainder_m;
      N_idx = cluster_m * divmod_cluster_shape_n.divisor + remainder_n;
    }
    return cute::make_tuple(M_idx, N_idx);
  }


  CUTLASS_DEVICE
  static void
  possibly_transpose_work_tile(WorkTileInfo& work_tile_info, Params const& params) {
    auto [M_idx, N_idx] = possibly_transpose_work_tile(
      params.raster_order_, work_tile_info.M_idx, work_tile_info.N_idx, params.divmod_cluster_shape_m_, params.divmod_cluster_shape_n_);
    work_tile_info.M_idx = M_idx;
    work_tile_info.N_idx = N_idx;
  }

  CUTLASS_DEVICE
  void
  possibly_transpose_work_tile(WorkTileInfo& work_tile_info) {
    possibly_transpose_work_tile(work_tile_info, params_);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  swizzle_and_rasterize(
      int cta_coord_m,
      int cta_coord_n,
      int cta_coord_l,
      bool valid,
      int cta_in_cluster_offset_m,
      int cta_in_cluster_offset_n) const {

    // Swizzling is enabled if the swizzle size is greater than 0
    if (params_.divmod_swizzle_size_.divisor > 0) {
      //
      // Swizzling enabled
      //

      // Swizzling is performed in terms of clusters. Convert the major and minor CTA coordinates
      // into cluster coordinates.
      int32_t cluster_coord_major, cluster_coord_minor, cluster_offset_m, cluster_offset_n;
      params_.divmod_cluster_shape_m_(cluster_coord_major, cluster_offset_m, cta_coord_m);
      params_.divmod_cluster_shape_n_(cluster_coord_minor, cluster_offset_n, cta_coord_n);

      // The general swizzling transformation:
      //   Given a grid of (M,N) clusters with swizzle size S,
      //   break the grid into (N/S) sub-grids of size MxS and remap:
      //     new_m_local = (m / S) + ((M / S) * (n % S))
      //     new_n_local = (m % S)
      //   Then map back to the full grid:
      //     new_n_global = new_n_local + ((n / S) * S)
      //   Residual clusters (outside the (M/S)*S x (N/S)*S subgrid) are not remapped.

      int32_t minor_div_swizz, minor_mod_swizz;
      params_.divmod_swizzle_size_(minor_div_swizz, minor_mod_swizz, cluster_coord_minor);

      int32_t major_clusters = params_.divmod_cluster_shape_m_.divide(GridDimX());

      // Determine the first IDs in the major and minor mode that constitute "residual" space
      int32_t major_clusters_div_swizzle = params_.divmod_swizzle_size_.divide(major_clusters);
      int32_t first_residual_major_cluster_id = major_clusters_div_swizzle * params_.divmod_swizzle_size_.divisor;
      int32_t minor_clusters_div_swizzle = params_.divmod_swizzle_size_.divide(params_.divmod_cluster_shape_n_.divide(GridDimY()));
      int32_t first_residual_minor_cluster_id = minor_clusters_div_swizzle * params_.divmod_swizzle_size_.divisor;

      // Only schedule via the swizzle if we're not within the residual space in either the major or minor mode.
      int32_t new_major_coord = cluster_coord_major, new_minor_coord = cluster_coord_minor;
      if (cluster_coord_major < first_residual_major_cluster_id && cluster_coord_minor < first_residual_minor_cluster_id) {
        // Not a residual cluster
        int32_t major_div_swizz, major_mod_swizz;
        params_.divmod_swizzle_size_(major_div_swizz, major_mod_swizz, cluster_coord_major);

        new_major_coord = major_div_swizz + (major_clusters_div_swizzle * minor_mod_swizz);
        new_minor_coord = major_mod_swizz + (minor_div_swizz * params_.divmod_swizzle_size_.divisor);
      }

      // Map the swizzled cluster tile back to a CTA tile
      cta_coord_m = new_major_coord * params_.divmod_cluster_shape_m_.divisor + cluster_offset_m;
      cta_coord_n = new_minor_coord * params_.divmod_cluster_shape_n_.divisor + cluster_offset_n;
    }

    // Since we swap the grid x and y modes if raster order is AlongN, swap the M and N tile offsets when
    // raster order is AlongN.
    auto [new_cta_coord_m, new_cta_coord_n] = possibly_transpose_work_tile(
      params_.raster_order_, cta_coord_m, cta_coord_n, params_.divmod_cluster_shape_m_, params_.divmod_cluster_shape_n_);

    new_cta_coord_m += cta_in_cluster_offset_m;
    new_cta_coord_n += cta_in_cluster_offset_n;

    return {new_cta_coord_m, new_cta_coord_n, static_cast<int32_t>(cta_coord_l), valid};
  }
  
  CUTLASS_DEVICE
  WorkTileInfo
  initial_work_tile_info(ClusterShape) {
    return swizzle_and_rasterize(
      BlockIdxX(), BlockIdxY(), BlockIdxZ(),
      /*valid=*/true,
      /*cluster_offset_m=*/0,
      /*cluster_offset_n=*/0);
  }

  CUTLASS_DEVICE
  WorkTileInfo
  initial_work_tile_info() {
    return initial_work_tile_info(ClusterShape{});
  }

  CUTLASS_DEVICE
  auto
  work_tile_to_cta_coord(WorkTileInfo work_tile_info) {
    return cute::make_coord(
      work_tile_info.M_idx,
      work_tile_info.N_idx,
      int32_t(0),
      work_tile_info.L_idx
    );
  }

  CUTLASS_DEVICE
  static void
  issue_clc_query(PipelineState state, CLCResponse* clc_response_ptr) {
    if (clc_response_ptr != nullptr) {
      auto status_coord = cluster_scheduler_get_next();
      CLCResponse response{};
      response.data[0] = status_coord[0];
      response.data[1] = status_coord[1];
      response.data[2] = status_coord[2];
      response.data[3] = status_coord[3];
      if constexpr (is_cluster) {
        // Distribute CLC response to all WGs in the cluster via DSM
        #pragma unroll
        for (uint32_t wg_id = 0; wg_id < cluster_size; ++wg_id) {
          dsm_vstore<CLC_VS>(clc_response_ptr[state.index()].data, response.data, wg_id);
        }
      } else {
        // Write CLC response to local SLM
        clc_response_ptr[state.index()] = response;
      }
    }
  }

  CUTLASS_DEVICE
  static WorkTileInfo
  work_tile_info_from_clc_response(CLCResponse const& response) {
    bool valid = cluster_scheduler_query(response.data[3]);
    if (!valid) {
      return WorkTileInfo::invalid_work_tile();
    }

    return {
      static_cast<int32_t>(response.data[0]),
      static_cast<int32_t>(response.data[1]),
      static_cast<int32_t>(response.data[2]),
      true
    };
  }

  CUTLASS_DEVICE
  PipelineState
  advance_to_next_work(Pipeline& clc_pipeline, PipelineState clc_pipe_producer_state) {
    if (cute::elect_one_sync()) {
      clc_pipeline.producer_acquire(clc_pipe_producer_state);
      issue_clc_query(clc_pipe_producer_state, clc_response_ptr_);
      clc_pipeline.producer_commit(clc_pipe_producer_state);
      ++clc_pipe_producer_state;
    }
    return clc_pipe_producer_state;
  }

  // Kernel helper function to get next work tile
  template <class TileSchedulerPipeline, class TileSchedulerPipelineState>
  CUTLASS_DEVICE
  auto
  fetch_next_work(
    WorkTileInfo work_tile_info,
    TileSchedulerPipeline& scheduler_pipeline,
    TileSchedulerPipelineState scheduler_pipe_consumer_state) {
    scheduler_pipeline.consumer_wait(scheduler_pipe_consumer_state);
    auto work_tile = work_tile_info_from_clc_response(clc_response_ptr_[scheduler_pipe_consumer_state.index()]);
    scheduler_pipeline.consumer_release(scheduler_pipe_consumer_state);

    auto next_work_tile_info = swizzle_and_rasterize(
      work_tile.M_idx, work_tile.N_idx, work_tile.L_idx, work_tile.is_valid(),
      block_id_in_cluster_.x, block_id_in_cluster_.y);
    // Return true to indicate that the tile scheduler pipeline state should be advanced
    return cute::make_tuple(next_work_tile_info, true);
  }

  CUTLASS_DEVICE
  void
  set_data_ptr(CLCResponse* clc_response_ptr) {
    clc_response_ptr_ = clc_response_ptr;
  }

  //
  // K Tile API
  //
  // Permute K iteration loading order from [C, S, R, T] to [S, R, T, C] for better L2 locality
  template <class ProblemShapeMNKL, class TileShape>
  CUTLASS_DEVICE
  auto
  get_k_tile_iterator(WorkTileInfo const& work_tile_info, ProblemShapeMNKL problem_shape_MNKL, TileShape tile_shape) {
    constexpr int32_t rank_t = cute::rank<2>(ProblemShapeMNKL{});
    auto k_tiles = cute::ceil_div(cute::get<2>(problem_shape_MNKL), cute::get<2>(tile_shape));
    if constexpr (rank_t == 4) {
      return cute::make_coord_iterator<cute::Step<_3, _0, _1, _2>>(k_tiles);
    }
    else if constexpr (rank_t == 3) {
      return cute::make_coord_iterator<cute::Step<_2, _0, _1>>(k_tiles);
    }
    else if constexpr (rank_t == 2) {
      return cute::make_coord_iterator<cute::Step<_1, _0>>(k_tiles);
    }
    else {
      return cute::make_coord_iterator(k_tiles);
    }
  }

  template <class ProblemShape, class TileShape>
  CUTLASS_HOST_DEVICE
  static int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape problem_shape, TileShape tile_shape) {
    // All work units returned by this scheduler cover the entire K iteration
    // space of the output tile assigned to the work unit.
    return cute::size(cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape)));
  }

  CUTLASS_HOST_DEVICE
  static uint32_t
  get_work_k_tile_start(WorkTileInfo const& work_tile_info) {
    // All work units returned by this scheduler start from K tile 0
    return 0u;
  }

  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const& work_tile_info, Params const& params) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const& work_tile_info) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  static bool
  requires_fixup(Params const& params, WorkTileInfo const work_tile_info) {
    return false;
  }

  CUTLASS_HOST_DEVICE
  static int*
  get_sk_tile_counter_ptr(Params const&) {
    return nullptr;
  }

  CUTLASS_HOST_DEVICE
  static uint64_t
  get_tile_idx(Params const&, WorkTileInfo const&) {
    return 0;
  }

  template <class FrgTensorC>
  CUTLASS_DEVICE
  static void
  fixup(Params const& params, WorkTileInfo const& work_tile_info, FrgTensorC& accumulators, uint32_t num_barriers, uint32_t barrier_idx) {
    UnderlyingTileScheduler::fixup(params.underlying_params_, work_tile_info, accumulators, num_barriers, barrier_idx);
  }

  CUTLASS_DEVICE
  static bool
  continue_current_work(WorkTileInfo& work_tile_info) {
    return UnderlyingTileScheduler::continue_current_work(work_tile_info);
  }

  CUTLASS_DEVICE
  static bool
  valid_warpgroup_in_work_tile(WorkTileInfo const& work_tile_info) {
    return true;
  }

  CUTLASS_DEVICE
  static bool
  requires_separate_reduction(Params const& params) {
    return false;
  }

private:
  CLCResponse *clc_response_ptr_ = nullptr;
  Params const& params_;
  dim3 block_id_in_cluster_ = {0, 0, 0};
};

}
