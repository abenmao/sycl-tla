#pragma once

#include "cute/config.hpp"
#include "cute/numeric/math.hpp"
#include "cute/layout.hpp"
#include "inline_pisa.hpp"
#include "mma_xe4_amma.hpp"

namespace cute::xe4::GMMA {

using namespace cute;

enum class OpType {
  Cluster,
  NoneCluster
};

template<typename T>
constexpr uint32_t getMinMmaK() {
  if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, fp16>) {
    return 16;
  }
  if constexpr (std::is_same_v<T, bf8> || std::is_same_v<T, int8_t>) {
    return 32;
  }

  CUTE_GCC_UNREACHABLE;
}

template<typename T>
constexpr uint32_t getMaxMmaK() {
  if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, fp16>) {
    return 128;
  }
  if constexpr (std::is_same_v<T, bf8> || std::is_same_v<T, int8_t>) {
    return 256;
  }

  CUTE_GCC_UNREACHABLE;
}

template <
  class ElementA,
  class ElementB,
  class ElementTupleC,
  class TileShape_MNK,
  class ClusterShape_MNK,
  SM90::GMMA::Major majorA,
  SM90::GMMA::Major majorB,
  auto... Args
>
CUTE_HOST_DEVICE constexpr
auto
ss_op_selector()
{
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");

  constexpr uint32_t MMA_M_MIN = 32;
  constexpr uint32_t MMA_M_MAX = 256;
  constexpr uint32_t MMA_N_MIN = 32;
  constexpr uint32_t MMA_N_MAX = 512;
  constexpr uint32_t MMA_K_MIN = cute::max(getMinMmaK<ElementA>(), getMinMmaK<ElementB>());
  constexpr uint32_t MMA_K_MAX = cute::min(getMaxMmaK<ElementA>(), getMaxMmaK<ElementB>());

  constexpr uint32_t Tile_M = size<0>(TileShape_MNK{});
  constexpr uint32_t Tile_N = size<1>(TileShape_MNK{});
  constexpr uint32_t Tile_K = size<2>(TileShape_MNK{});

  constexpr uint32_t MMA_M = cute::gcd(Tile_M, MMA_M_MAX);
  constexpr uint32_t MMA_N = cute::gcd(Tile_N, MMA_N_MAX);
  constexpr uint32_t MMA_K = cute::gcd(Tile_K, MMA_K_MAX);

  static_assert(MMA_M % 32 == 0, "MMA_M must be a multiple of 32.");
  static_assert(MMA_N % 32 == 0, "MMA_N must be a multiple of 32.");
  static_assert((MMA_K % 32 == 0) || (MMA_K == 16), "Tile_K must be a multiple of 32.");

  using MMA_Shape = Shape<Int<MMA_M>, Int<MMA_N>, Int<MMA_K>>;

  if constexpr (size(ClusterShape_MNK{}) == 1) {
    return XE4_ASYNC_GMMA<ElementTupleC, ElementA, ElementB, MMA_Shape, majorA, majorB>();
  } else {
    return XE4_ASYNC_GMMA_MULTICAST<ElementTupleC, ElementA, ElementB, MMA_Shape, majorA, majorB>();
  }

  CUTE_GCC_UNREACHABLE;
}

} // namespace cute::xe4::GMMA
