
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

  // clc_response_ptr is a placeholder; it is just to make the StaticPersistentTileSchedulerXe4 and PersistentTileSchedulerXe4 constructor interfaces consistent
  CUTLASS_DEVICE explicit
  StaticPersistentTileSchedulerXe4(CLCResponse* /* clc_response_ptr */, Params const& params)
    : BaseScheduler(params) {}

  // 3-arg overload that delegates to base class 4-arg version
  template <class ProblemShapeMNKL, class TileShape>
  CUTLASS_DEVICE
  auto
  get_k_tile_iterator(WorkTileInfo const& work_tile_info, ProblemShapeMNKL problem_shape_MNKL, TileShape tile_shape) {
    return BaseScheduler::get_k_tile_iterator(work_tile_info, problem_shape_MNKL, tile_shape, cute::tuple<>{});
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

};

}