#pragma once

#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include "cute/arch/copy_xe4_dma_legacy.hpp"
#include "cute/atom/copy_traits_xe4_dma_legacy.hpp"

namespace cutlass {
namespace epilogue {
namespace thread {
namespace detail {

using namespace cute;

struct CoreMatrix {
  static constexpr int kRowsPerCmTile = 2;
  static constexpr int kEsubBanksPerBank = 4;
  static constexpr int kSlmBanks = 4;
  static constexpr int kBytesPerCmRow = 32;

  static constexpr auto kCmLayoutRaw = make_ordered_layout(
    Shape<Shape<Int<kRowsPerCmTile>,Int<kEsubBanksPerBank>,Int<kSlmBanks>>,Int<kBytesPerCmRow>>{},
    Step<Step<_1,_3,_2>,_0>{}
  );

  template<typename ValType, typename EpilogueTile>
  CUTLASS_HOST_DEVICE static constexpr auto retile(EpilogueTile const& epilogue_tile) {
    auto cm_layout = recast_layout<uint8_t, ValType>(kCmLayoutRaw);
    auto retiled_layout = tile_to_shape(cm_layout, epilogue_tile, Step<_1,_0,_2>{});
    auto swizzled_layout = composition(make_swizzle<ValType>(), retiled_layout);
    return swizzled_layout;
  }

  template<typename ValType, typename EpilogueTile>
  CUTLASS_HOST_DEVICE static constexpr auto retile_2d(EpilogueTile const& epilogue_tile) {
    auto cm_layout = recast_layout<uint8_t, ValType>(kCmLayoutRaw);
    auto retiled_layout = tile_to_shape(cm_layout, epilogue_tile, Step<_1,_0>{});
    auto swizzled_layout = composition(make_swizzle<ValType>(), retiled_layout);
    return swizzled_layout;
  }

  template<typename ValType>
  CUTLASS_HOST_DEVICE static constexpr auto make_swizzle() {
    constexpr int kSwizzleB = 1;
    constexpr int kSwizzleM = countr_zero(kBytesPerCmRow / sizeof(ValType));
    constexpr int kSwizzleS = countr_zero(size(kCmLayoutRaw) / sizeof(ValType)) - kSwizzleM;
    return Swizzle<1, kSwizzleM, kSwizzleS>{};
  }
};

enum class EpilogueAccessPattern {
  Pattern1, // TODO: Each subgroup processes one core matrix
  Pattern2, // Each subgroup processes one or more rows of data across multiple core matrices
  Pattern3, // TODO
};

template <template <int, class, class> class SlmVOp, class ValType, int NumEpilogueWarps, class TileShape>
CUTLASS_HOST_DEVICE constexpr auto make_pattern2_tiled_copy(TileShape const& tile_shape) {
  static_assert(is_static<TileShape>::value, "Tile shape must be static");

  constexpr int tile_M = CUTE_STATIC_V(get<0>(tile_shape));
  constexpr int tile_N = CUTE_STATIC_V(get<1>(tile_shape));

  constexpr int kMaxLanesPerRow = 16;
  constexpr int kMaxBytesPerLoad = 32;
  constexpr int maxValuesPerLoad = kMaxBytesPerLoad / sizeof(ValType);
  static_assert(tile_N % maxValuesPerLoad == 0, "tile_N must be divisible by maxValuesPerLoad!");
  /**
   * @brief Use as many as lanes to process one row then calculate the minimum rows a subgroups would
   * process (`minRowsPerSubgroup`).
   */
  constexpr int maxLanesPerRow = cute::min(tile_N / maxValuesPerLoad, kMaxLanesPerRow);
  static_assert(cute::popcount(maxLanesPerRow) == 1, "maxLanesPerRow must be a power of 2!");
  constexpr int minRowsPerSubgroupPerIter = NumThreadsPerWarp / maxLanesPerRow;

  static_assert(tile_M % NumEpilogueWarps == 0, "tile_M must be divisible by NumEpilogueWarps!");
  constexpr int totalRowsPerSubgroup = tile_M / NumEpilogueWarps;

  /**
   * @brief Lanes are supposed to load SLM with `kMaxBytesPerLoad`, but if this cause some subgroups
   * in idle, lanes could load less than `kMaxBytesPerLoad` bytes.
   */
  constexpr int numRowsPerSubgroup = cute::min(totalRowsPerSubgroup, minRowsPerSubgroupPerIter);
  static_assert(tile_M % numRowsPerSubgroup == 0, "tile_M must be divisible by numRowsPerSubgroup!");
  static_assert(cute::popcount(numRowsPerSubgroup) == 1, "numRowsPerSubgroup must be a power of 2!");

  constexpr int numLanesPerRow = NumThreadsPerWarp / numRowsPerSubgroup;
  static_assert(tile_N % numLanesPerRow == 0, "tile_N must be divisible by numLanesPerRow!");
  constexpr int numValuesPerLane = cute::min(tile_N / numLanesPerRow, maxValuesPerLoad);

  auto thr_layout = make_ordered_layout(
    Shape<Shape<Int<numRowsPerSubgroup>, Int<NumEpilogueWarps>>, Int<numLanesPerRow>>{},
    Step<Step<_0,_2>,_1>{}
  );

  auto val_layout = make_layout(Shape<_1,Int<numValuesPerLane>>{}, GenRowMajor{});

  using Copy_Traits = Copy_Traits<SlmVOp<numValuesPerLane,ValType,ValType>>;
  using Atom = Copy_Atom<Copy_Traits, ValType>;
  auto tiled_copy = make_tiled_copy(Atom{}, thr_layout, val_layout);

  return tiled_copy;
}

#if 0
template <
  int FragmentSize,
  int NumEpilogueWarps,
  typename CstCallbacks,
  typename STensor,
  typename DTensor
>
CUTLASS_HOST_DEVICE
void pattern2(CstCallbacks& cst_callbacks, STensor const& src_tensor, DTensor& dst_tensor, uint32_t worker_id) {
  using SType = typename STensor::value_type;
  using DType = typename DTensor::value_type;

  auto tile_shape = product_each(shape(dst_tensor));

  auto tiled_s2r = make_pattern2_tiled_copy<cute::xe4::XE4_LDSM, SType, NumEpilogueWarps>(tile_shape);
  auto thread_s2r = tiled_s2r.get_thread_slice(worker_id);
  Tensor tSR_src = thread_s2r.partition_S(src_tensor);

  auto tiled_r2s = make_pattern2_tiled_copy<cute::xe4::XE4_STSM, SType, NumEpilogueWarps>(tile_shape);
  auto thread_r2s = tiled_r2s.get_thread_slice(worker_id);
  Tensor tRS_dst = thread_r2s.partition_D(dst_tensor);

  Tensor src_v = group_modes<1,-1>(tSR_src);
  Tensor dst_v = group_modes<1,-1>(tRS_dst);
}
#endif

} // namespace detail
} // namespace thread
} // namespace epilogue
} // namespace cutlass
