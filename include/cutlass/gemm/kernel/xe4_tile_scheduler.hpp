#pragma once

#include "cute/arch/cluster_xe4.hpp"
#include "cute/arch/copy_xe4_dma.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm_coord.hpp"

namespace cutlass::gemm::kernel::detail {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////

static auto make_coord_tensor(cute::tuple<int, int, int> problem_blocks_shape, cute::tuple<uint32_t, uint32_t> cluster_shape) {
  auto group_range = cute::make_shape(get_wgcount<1>(), get_wgcount<0>());
  auto wg_gride_layout = make_layout(group_range, make_stride(cute::E<0>{}, cute::E<1>{}));
  auto tiled_wg_gride_layout = zipped_divide(wg_gride_layout, cluster_shape);

  auto problem_blocks_layout = cute::make_layout(problem_blocks_shape, cute::make_stride(cute::E<0>{}, cute::E<1>{}, cute::E<2>{}));
  auto tiled_problem_blocks_layout = zipped_divide(problem_blocks_layout, group_range);

  auto coord_tensor_layout = replace<0>(tiled_problem_blocks_layout, tiled_wg_gride_layout);
  auto coord_tensor = cute::make_tensor(cute::make_inttuple_iter(0,0), coord_tensor_layout);

  return coord_tensor;
}

template<uint32_t Stages_>
class PersistentTileSchedulerXe4 {
public:
  static constexpr uint32_t CLC_VS = 4;
  struct alignas(16) CLCResponse { uint32_t data[CLC_VS]; };

  static constexpr uint32_t Stages = Stages_;
  static constexpr bool IsDynamicPersistent = true;

  using Pipeline = xe4::PipelineTmaAsync<Stages>;
  using PipelineState = typename Pipeline::PipelineState;

  struct Arguments {
    cute::tuple<uint32_t, uint32_t> slm_bytes;
  };

  struct Params {
    cute::tuple<int, int, int> problem_blocks_range;
    cute::tuple<int, int, int> problem_blocks_shape;
    cute::tuple<uint32_t, uint32_t> cluster_shape;
    cute::tuple<uint32_t, uint32_t> cluster_masks;
    cute::tuple<uint32_t, uint32_t> coop_set_ids;
    cute::tuple<uint32_t, uint32_t> coop_ids;
    uint32_t wg_linear_id_in_cluster;
  };

  template <class ProblemShapeMNKL, class TileShape, class ClusterShape>
  static Params
  to_underlying_arguments(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape,
      ClusterShape cluster_shape,
      Arguments const& arguments,
      [[maybe_unused]] void* workspace=nullptr) {

    // We only need the tile and cluster shape during scheduler setup, so let FTAD do the magic
    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    auto problem_shape_mnl = cute::select<0, 1, 3>(problem_shape_mnkl);
    auto problem_blocks_range = cute::ceil_div(flatten(problem_shape_mnl), flatten(tile_shape));

    auto [cluster_size_m, cluster_size_n, _] = cluster_shape;
    auto problem_blocks_m = cute::round_up(cute::get<0>(problem_blocks_range), cluster_size_m);
    auto problem_blocks_n = cute::round_up(cute::get<1>(problem_blocks_range), cluster_size_n);
    auto problem_blocks_shape = cute::make_shape(problem_blocks_m, problem_blocks_n, cute::get<2>(problem_blocks_range));

    uint32_t cluster_wgid_x = get_cluster_wgid<0>();
    uint32_t cluster_wgid_y = get_cluster_wgid<1>();

    uint32_t cluster_mask_a_ = 0;
    uint32_t cluster_mask_b_ = 0;
    uint32_t coop_id_a_ = cluster_wgid_x;
    uint32_t coop_id_b_ = cluster_wgid_y;
    uint32_t coop_set_id_a_ = cluster_wgid_y;
    uint32_t coop_set_id_b_ = cluster_wgid_x;
    uint32_t wg_linear_id_in_cluster = 0;

    uint32_t coop_num_a = cluster_size_n;
    uint32_t coop_num_b = cluster_size_m;
    uint32_t multicast_size_a = get<0>(arguments.slm_bytes) / coop_num_a;
    uint32_t multicast_size_b = get<1>(arguments.slm_bytes) / coop_num_b;

    if (multicast_size_a >= multicast_size_b) {
      wg_linear_id_in_cluster = cluster_wgid_x * cluster_size_m + cluster_wgid_y;
      cluster_mask_a_ = ((1u << coop_num_a) - 1) << (coop_set_id_a_ * coop_num_a); //0011
      uint32_t cluster_mask_b_base = 1u << coop_set_id_b_;
      #pragma unroll
      for (uint32_t i = 0; i < coop_num_b; i++) {
        cluster_mask_b_ |= cluster_mask_b_base << (i * coop_num_a);
      }
    } else {
      wg_linear_id_in_cluster = cluster_wgid_y * cluster_size_n + cluster_wgid_x;
      cluster_wgid_x = wg_linear_id_in_cluster % coop_num_b;
      cluster_wgid_y = wg_linear_id_in_cluster / coop_num_b;
      coop_set_id_a_ = cluster_wgid_x;
      coop_set_id_b_ = cluster_wgid_y;
      coop_id_a_ = cluster_wgid_y;
      coop_id_b_ = cluster_wgid_x;

      cluster_mask_b_ = ((1u << coop_num_b) - 1) << (coop_set_id_b_ * coop_num_b); //0011
      uint32_t cluster_mask_a_base = 1u << coop_set_id_a_;
      #pragma unroll
      for (uint32_t i = 0; i < coop_num_a; i++) {
        cluster_mask_a_ |= cluster_mask_a_base << (i * coop_num_b);
      } //0101
    }

    return {
      problem_blocks_range,
      problem_blocks_shape,
      {cluster_size_m, cluster_size_n},
      {cluster_mask_a_, cluster_mask_b_},
      {coop_set_id_a_, coop_set_id_b_},
      {coop_id_a_, coop_id_b_},
      wg_linear_id_in_cluster
    };
  }

  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t L_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_HOST_DEVICE
    bool
    is_valid() const {
      return is_valid_tile;
    }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo
    invalid_work_tile() {
      return {-1, -1, -1, false};
    }

    CUTLASS_HOST_DEVICE
    bool
    is_final_split(uint32_t k_tiles_per_output_tile) const {
      return true;
    }

    CUTLASS_HOST_DEVICE
    int32_t
    reduction_subtile_idx() const {
      return -1;
    }
  };

  PersistentTileSchedulerXe4(CLCResponse* clc_response_ptr, Params const& params)
    : clc_response_ptr_(clc_response_ptr)
    , params_(params)
    , coord_tensor_(make_coord_tensor(params.problem_blocks_shape, params.cluster_shape))
    , wgid_(get_wgid<1>(), get_wgid<0>()) {}

  CUTLASS_DEVICE
  WorkTileInfo initial_work_tile_info() {
    return get_current_work({});
  }

  CUTLASS_DEVICE
  auto
  work_tile_to_cta_coord(WorkTileInfo work_tile_info) {
    return cute::make_coord(
      work_tile_info.M_idx,
      work_tile_info.N_idx,
      work_tile_info.L_idx
    );
  }

  CUTLASS_DEVICE
  void
  issue_clc_query(PipelineState state, uint64_t* mbarrier_addr, CLCResponse* clc_response_ptr) {
  }

  CUTLASS_DEVICE
  WorkTileInfo
  work_tile_info_from_clc_response(uint32_t result_addr) {
    ++iter_id_;
    auto work_tile_info = get_current_work({});
    return work_tile_info;
  }

  CUTLASS_DEVICE
  PipelineState
  advance_to_next_work(Pipeline& clc_pipeline, PipelineState clc_pipe_producer_state) {
    // Wait for clcID buffer to become empty with a flipped phase
    issue_clc_query(clc_pipe_producer_state, nullptr, clc_response_ptr_);
    ++clc_pipe_producer_state;
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

    auto new_work_tile_info = work_tile_info_from_clc_response(0);

    // Return true to indicate that the tile scheduler pipeline state should be advanced
    return cute::make_tuple(new_work_tile_info, true);
  }

  CUTLASS_DEVICE
  WorkTileInfo get_current_work(PipelineState state) const {
    if (iter_id_ >= size<1>(coord_tensor_)) {
      return WorkTileInfo::invalid_work_tile();
    }

    const auto& cluster_local_id = params_.coop_set_ids;
    const auto cluster_id = cute::transform(wgid_, params_.cluster_shape, [](auto x, auto y) { return x / y; });
    auto [coord_m, coord_n, coord_l] = coord_tensor_(cute::make_coord(cluster_local_id, cluster_id), iter_id_);

    return {
      static_cast<int32_t>(coord_m),
      static_cast<int32_t>(coord_n),
      static_cast<int32_t>(coord_l),
      true
    };
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

private:
  using CoordTensor = decltype(make_coord_tensor(cute::make_tuple(0,0,0), cute::make_tuple((uint32_t)0,(uint32_t)0)));

  CLCResponse *clc_response_ptr_ = nullptr;
  Params params_;
  uint32_t iter_id_ {0};
  CoordTensor coord_tensor_;
  cute::tuple<uint32_t, uint32_t> wgid_;
};

}
