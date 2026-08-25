// Tile list. Every entry maps to a config typedef in moe_types.hpp / the runner.
// The X-macro drives both benchmark registration and the runtime dtype->config
// dispatch (moe_api.cpp).
//
//   X(DTYPE, NAME, CONFIG)                        -- greedy (uniform or dynamic-M)
//   X_DOUBLE_BUFFER(DTYPE, NAME, CONFIG)          -- double-buffer (plain)
//   X_DOUBLE_BUFFER_SCALED(DTYPE, NAME, CONFIG)   -- double-buffer (scaled)
// DTYPE = .in first token, NAME = unique thunk id, CONFIG = the typedef.
//
// Per dtype: a greedy binary (uniform + dynamic-M tiles) via -DMOE_DTYPE_<TAG>,
// plus a DB binary via -DMOE_DTYPE_DOUBLE_BUFFER_<TAG> for dtypes the
// tile-select routes to double buffer. -DMOE_BENCH_ALL enables every tag so
// tile-select sees both greedy and DB candidates. No PART slicing --
// each dtype has <=2 tiles.

#pragma once

// -DMOE_BENCH_ALL enables every dtype tag (one binary with all device kernels).
#ifdef MOE_BENCH_ALL
#define MOE_DTYPE_FP8_TENSOR_E4M3
#define MOE_DTYPE_MXFP8_E4M3
#define MOE_DTYPE_MXFP4_E2M1
#define MOE_DTYPE_BF16
#define MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3
#define MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1
#endif

// ---- GREEDY tiles (one uniform + one dynamic-M per dtype) ----

#ifdef MOE_DTYPE_FP8_TENSOR_E4M3
#define MOE_TILE_LIST_FP8_TENSOR_E4M3 \
  X(fp8_tensor_moe, GreedyMoE_Fp8Tensor_256_512_64,      cutlass::moe::Fp8TensorGreedy) \
  X(fp8_tensor_moe, GreedyMoE_Fp8Tensor_256_512_64_dynm, cutlass::moe::Fp8TensorGreedyDynM)
#else
#define MOE_TILE_LIST_FP8_TENSOR_E4M3
#endif

#ifdef MOE_DTYPE_MXFP8_E4M3
#define MOE_TILE_LIST_MXFP8_E4M3 \
  X(mxfp8_e4m3_moe, GreedyMoE_MxFp8_256_512_64,      cutlass::moe::MxFp8Greedy) \
  X(mxfp8_e4m3_moe, GreedyMoE_MxFp8_256_512_64_dynm, cutlass::moe::MxFp8GreedyDynM)
#else
#define MOE_TILE_LIST_MXFP8_E4M3
#endif

#ifdef MOE_DTYPE_MXFP4_E2M1
#define MOE_TILE_LIST_MXFP4_E2M1 \
  X(mxfp4_moe, GreedyMoE_MxFp4_256_512_128,      cutlass::moe::MxFp4Greedy) \
  X(mxfp4_moe, GreedyMoE_MxFp4_256_512_128_dynm, cutlass::moe::MxFp4GreedyDynM) \
  X_GREEDY_BIGK(mxfp4_moe, GreedyMoE_MxFp4_bigk_tinyK256,      cutlass::moe::MxFp4GreedyBigK) \
  X_GREEDY_BIGK(mxfp4_moe, GreedyMoE_MxFp4_bigk_tinyK256_dynm, cutlass::moe::MxFp4GreedyBigKDynM)
#else
#define MOE_TILE_LIST_MXFP4_E2M1
#endif

#ifdef MOE_DTYPE_BF16
#define MOE_TILE_LIST_BF16 \
  X(bf16_moe, GreedyMoE_Bf16_256_512_32,      cutlass::moe::Bf16Greedy) \
  X(bf16_moe, GreedyMoE_Bf16_256_512_32_dynm, cutlass::moe::Bf16GreedyDynM)
#else
#define MOE_TILE_LIST_BF16
#endif

// ---- DOUBLE-BUFFER tiles (only the ones tile-select selects) ----

#ifdef MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP8_E4M3 \
    X_DOUBLE_BUFFER_SCALED(mxfp8_e4m3_moe, CriBLockScalingGroupedGemmDoubleBuffer_E4M3E4M3BF16_RRR_TileShape_224_256_64, cutlass::moe::MxFp8DoubleBuffer_224_256_64)
#else
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP8_E4M3
#endif

#ifdef MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP4_E2M1 \
    X_DOUBLE_BUFFER_SCALED(mxfp4_moe, CriBLockScalingGroupedGemmDoubleBuffer_E2M1E2M1BF16_RCR_TileShape_192_256_128, cutlass::moe::MxFp4DoubleBuffer_192_256_128) \
    X_DOUBLE_BUFFER_SCALED(mxfp4_moe, CriBLockScalingGroupedGemmDoubleBuffer_E2M1E2M1BF16_RCR_TileShape_224_256_128, cutlass::moe::MxFp4DoubleBuffer_224_256_128)
#else
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP4_E2M1
#endif

// Combined X-list = greedy tiles + double-buffer tiles for whatever this binary enabled.
#if defined(MOE_DTYPE_FP8_TENSOR_E4M3) || defined(MOE_DTYPE_MXFP8_E4M3) || \
    defined(MOE_DTYPE_MXFP4_E2M1) || defined(MOE_DTYPE_BF16) || \
    defined(MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3) || defined(MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1)
#define MOE_TILE_X_LIST \
  MOE_TILE_LIST_FP8_TENSOR_E4M3 \
  MOE_TILE_LIST_MXFP8_E4M3 \
  MOE_TILE_LIST_MXFP4_E2M1 \
  MOE_TILE_LIST_BF16 \
  MOE_TILE_LIST_DOUBLE_BUFFER_MXFP8_E4M3 \
  MOE_TILE_LIST_DOUBLE_BUFFER_MXFP4_E2M1

// Dtype tag list = one F(DTYPE) per active dtype (greedy and/or DB both count;
// select_tile ranks both kinds under the same tag).
#if defined(MOE_DTYPE_FP8_TENSOR_E4M3)
#define MOE_DTYPE_TAG_LIST_FP8_TENSOR_E4M3 F(fp8_tensor_moe)
#else
#define MOE_DTYPE_TAG_LIST_FP8_TENSOR_E4M3
#endif
#if defined(MOE_DTYPE_MXFP8_E4M3) || defined(MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3)
#define MOE_DTYPE_TAG_LIST_MXFP8_E4M3 F(mxfp8_e4m3_moe)
#else
#define MOE_DTYPE_TAG_LIST_MXFP8_E4M3
#endif
#if defined(MOE_DTYPE_MXFP4_E2M1) || defined(MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1)
#define MOE_DTYPE_TAG_LIST_MXFP4_E2M1 F(mxfp4_moe)
#else
#define MOE_DTYPE_TAG_LIST_MXFP4_E2M1
#endif
#if defined(MOE_DTYPE_BF16)
#define MOE_DTYPE_TAG_LIST_BF16 F(bf16_moe)
#else
#define MOE_DTYPE_TAG_LIST_BF16
#endif

#define MOE_DTYPE_TAG_LIST \
  MOE_DTYPE_TAG_LIST_FP8_TENSOR_E4M3 \
  MOE_DTYPE_TAG_LIST_MXFP8_E4M3 \
  MOE_DTYPE_TAG_LIST_MXFP4_E2M1 \
  MOE_DTYPE_TAG_LIST_BF16
#endif
