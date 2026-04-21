#pragma once

#include "cute/config.hpp"
#include "cute/numeric/math.hpp"
#include "cute/layout.hpp"
#include "cutlass/float_subbyte.h"
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
  if constexpr (std::is_same_v<T, tfloat32_t>) {
    return 8;
  }
  if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, fp16>) {
    return 16;
  }
  if constexpr (std::is_same_v<T, bf8> || std::is_same_v<T, int8_t> || std::is_same_v<T, cutlass::float_e4m3_t>) {
    return 32;
  }
  if constexpr (std::is_same_v<T, cutlass::float_e2m1_t>) {
    return 64;
  }

  CUTE_GCC_UNREACHABLE;
}

template<typename T>
constexpr uint32_t getMaxMmaK() {
  if constexpr (std::is_same_v<T, tfloat32_t>) {
    return 64;
  }
  if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, fp16>) {
    return 128;
  }
  if constexpr (std::is_same_v<T, bf8> || std::is_same_v<T, int8_t> || std::is_same_v<T, cutlass::float_e4m3_t>) {
    return 256;
  }
  if constexpr (std::is_same_v<T, cutlass::float_e2m1_t>) {
    return 768;
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

// Block-scaled MMA operation selector for XE4
template <
  class ElementD,
  class ElementA,
  class ElementB,
  class ElementC,
  class ElementSF,    // Scale factor type
  int VS,             // Vector size (16 or 32)
  class TileShape_MNK,
  class ClusterShape_MNK,
  AMMA::Major MajorA,
  AMMA::Major MajorB,
  bool EnableCooperativeSF = false
>
CUTE_HOST_DEVICE constexpr
auto
bs_op_selector()
{
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");
  static_assert(VS == 16 || VS == 32, "Vector size must be 16 or 32.");

  constexpr uint32_t M_MIN = 64;
  constexpr uint32_t M_MAX = 256;
  constexpr uint32_t N_MIN = 64;
  constexpr uint32_t N_MAX = 512;

  // K_MIN accounts for the element-type hardware minimum.
  // The cm_8x32B core-matrix constraint (K_sf >= 8) is handled by padding sf_bK
  // in the collective mainloop, not by inflating K_MIN here.
  constexpr uint32_t K_MIN = cute::max(getMinMmaK<ElementA>(), getMinMmaK<ElementB>());
  constexpr uint32_t K_MAX = cute::min(getMaxMmaK<ElementA>(), getMaxMmaK<ElementB>());

  constexpr uint32_t Tile_M = size<0>(TileShape_MNK{});
  constexpr uint32_t Tile_N = size<1>(TileShape_MNK{});
  constexpr uint32_t Tile_K = size<2>(TileShape_MNK{});

  constexpr uint32_t M = cute::gcd(Tile_M, M_MAX);
  constexpr uint32_t N = cute::gcd(Tile_N, N_MAX);

  // For cooperative SF loading with VS=32, use Tile_K directly (when it fits) so the
  // atom K is large enough for ADMA box truncation (each CTA's K_sf >= 8 after split).
  // Without cooperative SF, padding handles the cm_8x32B constraint, so gcd is fine.
  constexpr bool is_cluster = size(ClusterShape_MNK{}) > 1;
  constexpr uint32_t K = (is_cluster && EnableCooperativeSF && VS == 32 && Tile_K <= K_MAX)
      ? Tile_K : cute::gcd(Tile_K, K_MAX);

  // According to the whitepaper, M_MIN/N_MIN was originally set to 32, but when using
  // block scaling, we encounter the error: "failed on (matrix_stride_in_elems % 64 == 0
  // && 'Type3 matrix_stride needs to be 64-elem aligned')". Therefore, M_MIN/N_MIN
  // is currently set to 64 to satisfy the alignment requirement.
  static_assert(M >= M_MIN, "MMA_M must be >= 64 as Type3 matrix_stride needs to be 64-elem aligned");
  static_assert(N >= N_MIN, "MMA_N must be >= 64 as Type3 matrix_stride needs to be 64-elem aligned");
  static_assert(K >= K_MIN, "MMA_K must be >= K_MIN (element type hardware minimum).");
  static_assert(K % VS == 0, "MMA_K must be a multiple of the scale factor vector size (VS).");

  if constexpr (size(ClusterShape_MNK{}) == 1) {
    return XE4_AMMA_FP4FP8<
      ElementD, ElementA, ElementB, ElementC, ElementSF, M, N, K, VS, MajorA, MajorB>();
  } else {
    return XE4_AMMA_FP4FP8_AB_CLUSTER<
      ElementD, ElementA, ElementB, ElementC, ElementSF, M, N, K, VS, MajorA, MajorB>();
  }

  CUTE_GCC_UNREACHABLE;
}

}
}