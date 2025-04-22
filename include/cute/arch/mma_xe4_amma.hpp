#pragma once

#include "inline_pisa.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"

namespace cute {

namespace xe4 {
  using MatDesc = uint32_t;
  using Abarrier = uint64_t*;
}

template <class TupleC, class TA, class TB, class Shape_MNK_, SM90::GMMA::Major tnspA, SM90::GMMA::Major tnspB>
struct XE4_ASYNC_GMMA
{
  using MatDesc = xe4::MatDesc;
  using Abarrier = xe4::Abarrier;
  using Shape_MNK = Shape_MNK_;

  template<typename MatDescD, typename MatDescC, typename MatDescA, typename MatDescB, typename... Args>
  CUTE_HOST_DEVICE static void
  fma(MatDescD const& mat_desc_d,
      MatDescC const& mat_desc_c,
      MatDescA const& mat_desc_a,
      MatDescB const& mat_desc_b,
      Args&&... args)
  {
    constexpr auto Tile_M = get<0>(Shape_MNK{});
    constexpr auto Tile_N = get<1>(Shape_MNK{});
    constexpr auto Tile_K = get<2>(Shape_MNK{});

    constexpr mem_layout layout_a = (tnspA == SM90::GMMA::Major::K) ? mem_layout::row_major: mem_layout::col_major;
    constexpr mem_layout layout_b = (tnspB == SM90::GMMA::Major::MN) ? mem_layout::row_major: mem_layout::col_major;

    using dtypeA = typename MatDescA::underlying_type;
    using dtypeB = typename MatDescB::underlying_type;
    using dtypeC = typename MatDescC::underlying_type;
    using dtypeD = typename MatDescD::underlying_type;
    async_gmma<dtypeD, dtypeC, dtypeA, dtypeB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(
      *mat_desc_d, *mat_desc_c, *mat_desc_a, *mat_desc_b, static_cast<Args&&>(args)...);
  }
};

template <class TupleC, class TA, class TB, class Shape_MNK_, SM90::GMMA::Major tnspA, SM90::GMMA::Major tnspB>
struct XE4_ASYNC_GMMA_MULTICAST
{
  using MatDesc = xe4::MatDesc;
  using Abarrier = xe4::Abarrier;
  using Shape_MNK = Shape_MNK_;

  template<typename MatDescD, typename MatDescC, typename MatDescA, typename MatDescB, typename MmaCtrl, typename... Args>
  CUTE_HOST_DEVICE static void
  fma(MatDescD const& mat_desc_d,
      MatDescC const& mat_desc_c,
      MatDescA const& mat_desc_a,
      MatDescB const& mat_desc_b,
      MmaCtrl mma_ctrl,
      Args&&... args)
  {
    constexpr auto Tile_M = get<0>(Shape_MNK{});
    constexpr auto Tile_N = get<1>(Shape_MNK{});
    constexpr auto Tile_K = get<2>(Shape_MNK{});

    constexpr mem_layout layout_a = (tnspA == SM90::GMMA::Major::K) ? mem_layout::row_major: mem_layout::col_major;
    constexpr mem_layout layout_b = (tnspB == SM90::GMMA::Major::MN) ? mem_layout::row_major: mem_layout::col_major;

    using dtypeA = typename MatDescA::underlying_type;
    using dtypeB = typename MatDescB::underlying_type;
    using dtypeC = typename MatDescC::underlying_type;
    using dtypeD = typename MatDescD::underlying_type;

    constexpr auto args_count = sizeof...(args);
    auto args_tuple = std::make_tuple(static_cast<Args&&>(args)...);

    if constexpr (args_count == 4) {
      auto [abar_a, abar_b, mask_a, mask_b] = args_tuple;
      async_gmma<dtypeD, dtypeC, dtypeA, dtypeB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(
        *mat_desc_d, *mat_desc_c, *mat_desc_a, *mat_desc_b, mma_ctrl, abar_a, mask_a, abar_b, mask_b);
    } else if constexpr (args_count == 5) {
      auto [abar_d, abar_a, abar_b, mask_a, mask_b] = args_tuple;
      async_gmma<dtypeD, dtypeC, dtypeA, dtypeB, Tile_M, Tile_N, Tile_K, layout_a, layout_b>(
        *mat_desc_d, *mat_desc_c, *mat_desc_a, *mat_desc_b, mma_ctrl, abar_d, abar_a, mask_a, abar_b, mask_b);
    } else {
      static_assert(args_count == 4 || args_count == 5, "Invalid number of arguments for async_gmma_multicast");
    }
  }
};

} // namespace cute
