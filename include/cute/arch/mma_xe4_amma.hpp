#pragma once

#include "inline_pisa.hpp"
#include "cute/arch/mma_xe4_desc.hpp"

namespace cute {

namespace xe4 {
  using MatDesc = uint32_t;
  using Abarrier = uint64_t*;
}

template <class TupleC, class TA, class TB, class Shape_MNK_, xe4::GMMA::Major tnspA, xe4::GMMA::Major tnspB>
struct XE4_ASYNC_GMMA
{
  using MatDesc = xe4::MatDesc;
  using Abarrier = xe4::Abarrier;
  using Shape_MNK = Shape_MNK_;

  using DRegisters = MatDesc[1];
  using ARegisters = MatDesc[1];
  using BRegisters = MatDesc[1];
  using CRegisters = MatDesc[1];

  template<typename DstType, typename... Args>
  CUTE_HOST_DEVICE static void
  fma(MatDesc const& mat_desc_d,
      MatDesc const& mat_desc_c,
      MatDesc const& mat_desc_a,
      MatDesc const& mat_desc_b,
      DstType const& dst_type,
      Args&&... args)
  {
    constexpr auto Tile_M = get<0>(Shape_MNK{});
    constexpr auto Tile_N = get<1>(Shape_MNK{});
    constexpr auto Tile_K = get<2>(Shape_MNK{});

    constexpr mem_layout layout_a = (tnspA == xe4::GMMA::Major::K) ? mem_layout::row_major: mem_layout::col_major;
    constexpr mem_layout layout_b = (tnspB == xe4::GMMA::Major::MN) ? mem_layout::row_major: mem_layout::col_major;

    using TC = tuple_element_t<0, TupleC>;
    using TD = tuple_element_t<static_cast<int>(DstType::value), TupleC>;
    async_gmma<TD, TC, TA, TB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, static_cast<Args&&>(args)...);
  }
};

template <class TupleC, class TA, class TB, class Shape_MNK_, xe4::GMMA::Major tnspA, xe4::GMMA::Major tnspB>
struct XE4_ASYNC_GMMA_MULTICAST
{
  using MatDesc = xe4::MatDesc;
  using Abarrier = xe4::Abarrier;
  using Shape_MNK = Shape_MNK_;

  using DRegisters = MatDesc[1];
  using ARegisters = MatDesc[1];
  using BRegisters = MatDesc[1];
  using CRegisters = MatDesc[1];

  template<typename DstType, typename MmaCtrl, typename... Args>
  CUTE_HOST_DEVICE static void
  fma(MatDesc const& mat_desc_d,
      MatDesc const& mat_desc_c,
      MatDesc const& mat_desc_a,
      MatDesc const& mat_desc_b,
      DstType const& dst_type,
      MmaCtrl mma_ctrl,
      Args&&... args)
  {

    constexpr auto Tile_M = get<0>(Shape_MNK{});
    constexpr auto Tile_N = get<1>(Shape_MNK{});
    constexpr auto Tile_K = get<2>(Shape_MNK{});

    constexpr mem_layout layout_a = (tnspA == xe4::GMMA::Major::K) ? mem_layout::row_major: mem_layout::col_major;
    constexpr mem_layout layout_b = (tnspB == xe4::GMMA::Major::MN) ? mem_layout::row_major: mem_layout::col_major;

    using TC = tuple_element_t<0, TupleC>;
    using TD = tuple_element_t<static_cast<int>(DstType::value), TupleC>;

    constexpr auto args_count = sizeof...(args);
    auto args_tuple = std::make_tuple(static_cast<Args&&>(args)...);

    if constexpr (args_count == 4) {
      auto [abar_a, abar_b, mask_a, mask_b] = args_tuple;
      async_gmma<TD, TC, TA, TB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, mma_ctrl, abar_a, mask_a, abar_b, mask_b);
    } else if constexpr (args_count == 5) {
      auto [abar_d, abar_a, abar_b, mask_a, mask_b] = args_tuple;
      async_gmma<TD, TC, TA, TB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, mma_ctrl, abar_d, abar_a, mask_a, abar_b, mask_b);
    } else {
      static_assert(args_count == 4 || args_count == 5, "Invalid number of arguments for async_gmma_multicast");
    }
  }
};

template <class TupleC, class TA, class TB, class TMeta, class Shape_MNK_, xe4::GMMA::Major tnspA, xe4::GMMA::Major tnspB, bool ScaleA, bool ScaleB>
struct XE4_ASYNC_GMMA_SCALE
{
  using MetaDesc = uint64_t;
  using MatDesc = xe4::MatDesc;
  using Abarrier = xe4::Abarrier;
  using Shape_MNK = Shape_MNK_;

  using DRegisters = MatDesc[1];
  using ARegisters = MatDesc[1];
  using BRegisters = MatDesc[1];
  using CRegisters = MatDesc[1];

  template<typename DstType, typename... Args>
  CUTE_HOST_DEVICE static void
  fma(MatDesc const& mat_desc_d,
      MatDesc const& mat_desc_c,
      MatDesc const& mat_desc_a,
      MatDesc const& mat_desc_b,
      MetaDesc const& mat_desc_meta_a,
      MetaDesc const& mat_desc_meta_b,
      DstType const& dst_type,
      Args&&... args)
  {
    constexpr auto Tile_M = get<0>(Shape_MNK{});
    constexpr auto Tile_N = get<1>(Shape_MNK{});
    constexpr auto Tile_K = get<2>(Shape_MNK{});

    constexpr mem_layout layout_a = (tnspA == xe4::GMMA::Major::K) ? mem_layout::row_major: mem_layout::col_major;
    constexpr mem_layout layout_b = (tnspB == xe4::GMMA::Major::MN) ? mem_layout::row_major: mem_layout::col_major;

    using TC = tuple_element_t<0, TupleC>;
    using TD = tuple_element_t<static_cast<int>(DstType::value), TupleC>;
    async_gmma<TD, TC, TA, TB, Tile_M, Tile_N, Tile_K, layout_a, layout_b, ScaleA, ScaleB>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, mat_desc_meta_a, mat_desc_meta_b, static_cast<Args&&>(args)...);
  }
};

} // namespace cute
