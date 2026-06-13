
#pragma once
// Enable printing of transformation of CLC IDs into swizzled tile coordinates
#define CUTLASS_SWIZZLE_DEVICE_DEBUG_PRINT 0

#include "cute/int_tuple.hpp"

#include "cutlass/arch/config.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/detail/cluster.hpp" 
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm_coord.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm100_static_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/conv/convnd_problem_shape.hpp"
#include "cutlass/conv/detail.hpp"
#include "cute/arch/xe4_inline_pisa.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel::detail {

///////////////////////////////////////////////////////////////////////////////

template <class ClusterShape_>
class StaticPersistentTileSchedulerXe4:
  public PersistentTileSchedulerSm90 {

public:
  using BaseScheduler = PersistentTileSchedulerSm90;
  using ClusterShape = ClusterShape_;
  
  // Inherit all constructors from the base class
  using BaseScheduler::BaseScheduler;

  using RasterOrder = typename BaseScheduler::RasterOrder;
  using RasterOrderOptions = typename BaseScheduler::RasterOrderOptions;
  static constexpr uint32_t CLC_VS = 4;
  struct alignas(16) CLCResponse { uint32_t data[CLC_VS]; };

  using Params = typename BaseScheduler::Params;
  using Arguments = typename BaseScheduler::Arguments;

  template <class ProblemShapeMNKL, class TileShape, class ClusterShapeMNK>
  static Params
  to_underlying_arguments(
      ProblemShapeMNKL problem_shape_mnkl,
      TileShape tile_shape,
      ClusterShapeMNK cluster_shape,
      [[maybe_unused]] KernelHardwareInfo const& hw_info,
      Arguments const& arguments,
      [[maybe_unused]] void* workspace = nullptr,
      [[maybe_unused]] const uint32_t epilogue_subtile = 1,
      [[maybe_unused]] uint32_t ktile_start_alignment_count = 1u) {

    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShapeMNK>::value);

    dim3 problem_blocks = BaseScheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);

    Params params;
    params.initialize(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.max_swizzle_size,
      arguments.raster_order);

    return params;
  }

  // clc_response_ptr is a placeholder; it is just to make the StaticPersistentTileSchedulerXe4 and PersistentTileSchedulerXe4 constructor interfaces consistent
  CUTLASS_DEVICE explicit
  StaticPersistentTileSchedulerXe4(CLCResponse* /* clc_response_ptr */, Params const& params, dim3 block_id_in_cluster)
    : BaseScheduler(params) {}
  
  CUTLASS_DEVICE explicit
  StaticPersistentTileSchedulerXe4(CLCResponse* /* clc_response_ptr */, Params const& params)
    : BaseScheduler(params) {}

  // 3-arg overload that delegates to base class 4-arg version
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
  static bool
  requires_fixup(Params const& params, WorkTileInfo const work_tile_info) {
    return false;
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
  static int*
  get_sk_tile_counter_ptr(Params const&) {
    return nullptr;
  }

  CUTLASS_HOST_DEVICE
  static uint64_t
  get_tile_idx(Params const&, WorkTileInfo const&) {
    return 0;
  }

};

}