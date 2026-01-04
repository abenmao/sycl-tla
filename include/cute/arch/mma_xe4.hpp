#pragma once

#include "cute/config.hpp"
#include "cute/numeric/math.hpp"
#include "cute/layout.hpp"
#include "xe4_inline_pisa.hpp"
#include "mma_xe4_amma.hpp"

namespace cute {
namespace AMMA {

enum class OpType {
  Cluster,
  NoneCluster
};

template<typename T>
constexpr uint32_t getMinMmaK() {
  if constexpr (std::is_same_v<T, float>) {
    return 8;
  }
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
  if constexpr (std::is_same_v<T, float>) {
    return 64;
  }
  if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, fp16>) {
    return 128;
  }
  if constexpr (std::is_same_v<T, bf8> || std::is_same_v<T, int8_t>) {
    return 256;
  }

  CUTE_GCC_UNREACHABLE;
}


template <
  class ElementD,
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  class ClusterShape_MNK,
  AMMA::Major MajorA,
  AMMA::Major MajorB
>
CUTE_HOST_DEVICE constexpr
auto
ss_op_selector()
{
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");

  constexpr uint32_t M_MIN = 32;
  constexpr uint32_t M_MAX = 256;
  constexpr uint32_t N_MIN = 32;
  constexpr uint32_t N_MAX = 512;
  constexpr uint32_t K_MIN = cute::max(getMinMmaK<ElementA>(), getMinMmaK<ElementB>());
  constexpr uint32_t K_MAX = cute::min(getMaxMmaK<ElementA>(), getMaxMmaK<ElementB>());

  constexpr uint32_t Tile_M = size<0>(TileShape_MNK{});
  constexpr uint32_t Tile_N = size<1>(TileShape_MNK{});
  constexpr uint32_t Tile_K = size<2>(TileShape_MNK{});

  constexpr uint32_t M = cute::gcd(Tile_M, M_MAX);
  constexpr uint32_t N = cute::gcd(Tile_N, N_MAX);
  constexpr uint32_t K = cute::gcd(Tile_K, K_MAX);

  static_assert(M % 32 == 0, "MMA_M must be a multiple of 32.");
  static_assert(N % 32 == 0, "MMA_N must be a multiple of 32.");
  static_assert((K % 32 == 0) || (K == 16), "Tile_K must be a multiple of 32.");

  if constexpr (size(ClusterShape_MNK{}) == 1) {
    return XE4_AMMA<
      ElementD, ElementA, ElementB, ElementC, M, N, K, MajorA, MajorB>();
  } else {
    return XE4_AMMA_AB_CLUSTER<
      ElementD, ElementA, ElementB, ElementC, M, N, K, MajorA, MajorB>();
  }

  CUTE_GCC_UNREACHABLE;
}

}
}
