// Auto-generated tile list. Every entry maps to a config typedef in
// moe_gemm_runner.hpp. The X-macro drives both benchmark registration and the
// runtime dtype->config dispatch (moe_api.cpp).
//
//   X(DTYPE, NAME, CONFIG)                        -- regular, tile auto-selected
//   X_DOUBLE_BUFFER(DTYPE, NAME, CONFIG)          -- double-buffer (plain)
//   X_DOUBLE_BUFFER_SCALED(DTYPE, NAME, CONFIG)   -- double-buffer (scaled)
// DTYPE is the .in first token, NAME a unique thunk identifier, CONFIG the typedef.

#pragma once

// -DMOE_BENCH_ALL enables every dtype tag (one binary with all device kernels).
#ifdef MOE_BENCH_ALL
#define MOE_DTYPE_FP8_TENSOR_E4M3
#define MOE_DTYPE_MXFP8_E4M3
#define MOE_DTYPE_MXFP4_E2M1
#define MOE_DTYPE_BF16
#define MOE_DTYPE_DOUBLE_BUFFER_BF16
#define MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3
#define MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1
#define MOE_DTYPE_DOUBLE_BUFFER_FP8_TENSOR_E4M3
#endif

// A coarse -DMOE_DTYPE_<TAG> enables every PART<n> of that basket.
#ifdef MOE_DTYPE_FP8_TENSOR_E4M3
#define MOE_DTYPE_FP8_TENSOR_E4M3_PART0
#endif
#ifdef MOE_DTYPE_MXFP8_E4M3
#define MOE_DTYPE_MXFP8_E4M3_PART0
#endif
#ifdef MOE_DTYPE_MXFP4_E2M1
#define MOE_DTYPE_MXFP4_E2M1_PART0
#endif
#ifdef MOE_DTYPE_BF16
#define MOE_DTYPE_BF16_PART0
#endif
#ifdef MOE_DTYPE_DOUBLE_BUFFER_BF16
#define MOE_DTYPE_DOUBLE_BUFFER_BF16_PART0
#define MOE_DTYPE_DOUBLE_BUFFER_BF16_PART1
#define MOE_DTYPE_DOUBLE_BUFFER_BF16_PART2
#endif
#ifdef MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3
#define MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3_PART0
#define MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3_PART1
#endif
#ifdef MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1
#define MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1_PART0
#define MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1_PART1
#endif
#ifdef MOE_DTYPE_DOUBLE_BUFFER_FP8_TENSOR_E4M3
#define MOE_DTYPE_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART0
#define MOE_DTYPE_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART1
#endif

// ---- Per-dtype named lists, sliced into <=3-tile PART<n> ----

// -- Single-buffer fp8_tensor (3 tiles -> 1 part) --
#ifdef MOE_DTYPE_FP8_TENSOR_E4M3_PART0
#define MOE_TILE_LIST_FP8_TENSOR_E4M3_PART0 \
  X(fp8_tensor, CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_192_512_64, cutlass::moe::Fp8Tensor_192_512_64) \
  X(fp8_tensor, CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_256_512_64, cutlass::moe::Fp8Tensor_256_512_64) \
  X(fp8_tensor, CriGroupedGemm_E4M3E4M3BF16_RRR_TileShape_320_384_64, cutlass::moe::Fp8Tensor_320_384_64)
#else
#define MOE_TILE_LIST_FP8_TENSOR_E4M3_PART0
#endif
#define MOE_TILE_LIST_FP8_TENSOR_E4M3 \
  MOE_TILE_LIST_FP8_TENSOR_E4M3_PART0

// -- Single-buffer mxfp8_e4m3 (3 tiles -> 1 part) --
#ifdef MOE_DTYPE_MXFP8_E4M3_PART0
#define MOE_TILE_LIST_MXFP8_E4M3_PART0 \
  X(mxfp8_e4m3, CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_192_512_64, cutlass::moe::MxFp8_192_512_64) \
  X(mxfp8_e4m3, CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_256_512_64, cutlass::moe::MxFp8_256_512_64) \
  X(mxfp8_e4m3, CriBLockScalingGroupedGemm_E4M3E4M3BF16_RRR_TileShape_320_384_64, cutlass::moe::MxFp8_320_384_64)
#else
#define MOE_TILE_LIST_MXFP8_E4M3_PART0
#endif
#define MOE_TILE_LIST_MXFP8_E4M3 \
  MOE_TILE_LIST_MXFP8_E4M3_PART0

// -- Single-buffer mxfp4_e2m1 (3 tiles -> 1 part) --
#ifdef MOE_DTYPE_MXFP4_E2M1_PART0
#define MOE_TILE_LIST_MXFP4_E2M1_PART0 \
  X(mxfp4, CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_192_512_128, cutlass::moe::MxFp4_192_512_128) \
  X(mxfp4, CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_256_512_128, cutlass::moe::MxFp4_256_512_128) \
  X(mxfp4, CriBLockScalingGroupedGemm_E2M1E2M1BF16_RCR_TileShape_320_384_128, cutlass::moe::MxFp4_320_384_128)
#else
#define MOE_TILE_LIST_MXFP4_E2M1_PART0
#endif
#define MOE_TILE_LIST_MXFP4_E2M1 \
  MOE_TILE_LIST_MXFP4_E2M1_PART0

// -- Single-buffer bf16 (3 tiles -> 1 part) --
#ifdef MOE_DTYPE_BF16_PART0
#define MOE_TILE_LIST_BF16_PART0 \
  X(bf16, CriGroupedGemmBF16BF16BF16_RRR_TileShape_192_512_32, cutlass::moe::Bf16_192_512_32) \
  X(bf16, CriGroupedGemmBF16BF16BF16_RRR_TileShape_256_512_32, cutlass::moe::Bf16_256_512_32) \
  X(bf16, CriGroupedGemmBF16BF16BF16_RRR_TileShape_320_384_32, cutlass::moe::Bf16_320_384_32)
#else
#define MOE_TILE_LIST_BF16_PART0
#endif
#define MOE_TILE_LIST_BF16 \
  MOE_TILE_LIST_BF16_PART0

// ---- Double-buffer tile lists, sliced into <=3-tile PART<n> ----

// -- Double-buffer bf16 (7 tiles -> 3 parts) --
#ifdef MOE_DTYPE_DOUBLE_BUFFER_BF16_PART0
#define MOE_TILE_LIST_DOUBLE_BUFFER_BF16_PART0 \
    X_DOUBLE_BUFFER(bf16, CriGroupedGemmDoubleBufferBF16BF16BF16_RRR_TileShape_192_256_32, cutlass::moe::Bf16DoubleBuffer_192_256_32) \
    X_DOUBLE_BUFFER(bf16, CriGroupedGemmDoubleBufferBF16BF16BF16_RRR_TileShape_192_384_32, cutlass::moe::Bf16DoubleBuffer_192_384_32) \
    X_DOUBLE_BUFFER(bf16, CriGroupedGemmDoubleBufferBF16BF16BF16_RRR_TileShape_224_256_32, cutlass::moe::Bf16DoubleBuffer_224_256_32)
#else
#define MOE_TILE_LIST_DOUBLE_BUFFER_BF16_PART0
#endif
#ifdef MOE_DTYPE_DOUBLE_BUFFER_BF16_PART1
#define MOE_TILE_LIST_DOUBLE_BUFFER_BF16_PART1 \
    X_DOUBLE_BUFFER(bf16, CriGroupedGemmDoubleBufferBF16BF16BF16_RRR_TileShape_256_256_32, cutlass::moe::Bf16DoubleBuffer_256_256_32) \
    X_DOUBLE_BUFFER(bf16, CriGroupedGemmDoubleBufferBF16BF16BF16_RRR_TileShape_288_256_32, cutlass::moe::Bf16DoubleBuffer_288_256_32) \
    X_DOUBLE_BUFFER(bf16, CriGroupedGemmDoubleBufferBF16BF16BF16_RRR_TileShape_320_192_32, cutlass::moe::Bf16DoubleBuffer_320_192_32)
#else
#define MOE_TILE_LIST_DOUBLE_BUFFER_BF16_PART1
#endif
#ifdef MOE_DTYPE_DOUBLE_BUFFER_BF16_PART2
#define MOE_TILE_LIST_DOUBLE_BUFFER_BF16_PART2 \
    X_DOUBLE_BUFFER(bf16, CriGroupedGemmDoubleBufferBF16BF16BF16_RRR_TileShape_384_192_32, cutlass::moe::Bf16DoubleBuffer_384_192_32)
#else
#define MOE_TILE_LIST_DOUBLE_BUFFER_BF16_PART2
#endif
#define MOE_TILE_LIST_DOUBLE_BUFFER_BF16 \
    MOE_TILE_LIST_DOUBLE_BUFFER_BF16_PART0 \
    MOE_TILE_LIST_DOUBLE_BUFFER_BF16_PART1 \
    MOE_TILE_LIST_DOUBLE_BUFFER_BF16_PART2

// -- Double-buffer mxfp8_e4m3 (6 tiles -> 2 parts) --
#ifdef MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3_PART0
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP8_E4M3_PART0 \
    X_DOUBLE_BUFFER_SCALED(mxfp8_e4m3, CriBLockScalingGroupedGemmDoubleBuffer_E4M3E4M3BF16_RRR_TileShape_192_256_64, cutlass::moe::MxFp8DoubleBuffer_192_256_64) \
    X_DOUBLE_BUFFER_SCALED(mxfp8_e4m3, CriBLockScalingGroupedGemmDoubleBuffer_E4M3E4M3BF16_RRR_TileShape_192_384_64, cutlass::moe::MxFp8DoubleBuffer_192_384_64) \
    X_DOUBLE_BUFFER_SCALED(mxfp8_e4m3, CriBLockScalingGroupedGemmDoubleBuffer_E4M3E4M3BF16_RRR_TileShape_224_256_64, cutlass::moe::MxFp8DoubleBuffer_224_256_64)
#else
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP8_E4M3_PART0
#endif
#ifdef MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3_PART1
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP8_E4M3_PART1 \
    X_DOUBLE_BUFFER_SCALED(mxfp8_e4m3, CriBLockScalingGroupedGemmDoubleBuffer_E4M3E4M3BF16_RRR_TileShape_256_256_64, cutlass::moe::MxFp8DoubleBuffer_256_256_64) \
    X_DOUBLE_BUFFER_SCALED(mxfp8_e4m3, CriBLockScalingGroupedGemmDoubleBuffer_E4M3E4M3BF16_RRR_TileShape_288_256_64, cutlass::moe::MxFp8DoubleBuffer_288_256_64) \
    X_DOUBLE_BUFFER_SCALED(mxfp8_e4m3, CriBLockScalingGroupedGemmDoubleBuffer_E4M3E4M3BF16_RRR_TileShape_320_192_64, cutlass::moe::MxFp8DoubleBuffer_320_192_64)
#else
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP8_E4M3_PART1
#endif
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP8_E4M3 \
    MOE_TILE_LIST_DOUBLE_BUFFER_MXFP8_E4M3_PART0 \
    MOE_TILE_LIST_DOUBLE_BUFFER_MXFP8_E4M3_PART1

// -- Double-buffer mxfp4_e2m1 (6 tiles -> 2 parts) --
#ifdef MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1_PART0
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP4_E2M1_PART0 \
    X_DOUBLE_BUFFER_SCALED(mxfp4, CriBLockScalingGroupedGemmDoubleBuffer_E2M1E2M1BF16_RCR_TileShape_192_256_128, cutlass::moe::MxFp4DoubleBuffer_192_256_128) \
    X_DOUBLE_BUFFER_SCALED(mxfp4, CriBLockScalingGroupedGemmDoubleBuffer_E2M1E2M1BF16_RCR_TileShape_192_384_128, cutlass::moe::MxFp4DoubleBuffer_192_384_128) \
    X_DOUBLE_BUFFER_SCALED(mxfp4, CriBLockScalingGroupedGemmDoubleBuffer_E2M1E2M1BF16_RCR_TileShape_224_256_128, cutlass::moe::MxFp4DoubleBuffer_224_256_128)
#else
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP4_E2M1_PART0
#endif
#ifdef MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1_PART1
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP4_E2M1_PART1 \
    X_DOUBLE_BUFFER_SCALED(mxfp4, CriBLockScalingGroupedGemmDoubleBuffer_E2M1E2M1BF16_RCR_TileShape_256_256_128, cutlass::moe::MxFp4DoubleBuffer_256_256_128) \
    X_DOUBLE_BUFFER_SCALED(mxfp4, CriBLockScalingGroupedGemmDoubleBuffer_E2M1E2M1BF16_RCR_TileShape_288_256_128, cutlass::moe::MxFp4DoubleBuffer_288_256_128) \
    X_DOUBLE_BUFFER_SCALED(mxfp4, CriBLockScalingGroupedGemmDoubleBuffer_E2M1E2M1BF16_RCR_TileShape_320_192_128, cutlass::moe::MxFp4DoubleBuffer_320_192_128)
#else
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP4_E2M1_PART1
#endif
#define MOE_TILE_LIST_DOUBLE_BUFFER_MXFP4_E2M1 \
    MOE_TILE_LIST_DOUBLE_BUFFER_MXFP4_E2M1_PART0 \
    MOE_TILE_LIST_DOUBLE_BUFFER_MXFP4_E2M1_PART1

// -- Double-buffer fp8_tensor (6 tiles -> 2 parts) --
#ifdef MOE_DTYPE_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART0
#define MOE_TILE_LIST_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART0 \
    X_DOUBLE_BUFFER_SCALED(fp8_tensor, CriGroupedGemmDoubleBuffer_E4M3E4M3BF16_RRR_TileShape_192_256_64, cutlass::moe::Fp8TensorDoubleBuffer_192_256_64) \
    X_DOUBLE_BUFFER_SCALED(fp8_tensor, CriGroupedGemmDoubleBuffer_E4M3E4M3BF16_RRR_TileShape_192_384_64, cutlass::moe::Fp8TensorDoubleBuffer_192_384_64) \
    X_DOUBLE_BUFFER_SCALED(fp8_tensor, CriGroupedGemmDoubleBuffer_E4M3E4M3BF16_RRR_TileShape_224_256_64, cutlass::moe::Fp8TensorDoubleBuffer_224_256_64)
#else
#define MOE_TILE_LIST_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART0
#endif
#ifdef MOE_DTYPE_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART1
#define MOE_TILE_LIST_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART1 \
    X_DOUBLE_BUFFER_SCALED(fp8_tensor, CriGroupedGemmDoubleBuffer_E4M3E4M3BF16_RRR_TileShape_256_256_64, cutlass::moe::Fp8TensorDoubleBuffer_256_256_64) \
    X_DOUBLE_BUFFER_SCALED(fp8_tensor, CriGroupedGemmDoubleBuffer_E4M3E4M3BF16_RRR_TileShape_288_256_64, cutlass::moe::Fp8TensorDoubleBuffer_288_256_64) \
    X_DOUBLE_BUFFER_SCALED(fp8_tensor, CriGroupedGemmDoubleBuffer_E4M3E4M3BF16_RRR_TileShape_320_192_64, cutlass::moe::Fp8TensorDoubleBuffer_320_192_64)
#else
#define MOE_TILE_LIST_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART1
#endif
#define MOE_TILE_LIST_DOUBLE_BUFFER_FP8_TENSOR_E4M3 \
    MOE_TILE_LIST_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART0 \
    MOE_TILE_LIST_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART1

// Combined X-list = regular tiles + double-buffer tiles.
#if defined(MOE_DTYPE_FP8_TENSOR_E4M3_PART0) || \
    defined(MOE_DTYPE_MXFP8_E4M3_PART0) || \
    defined(MOE_DTYPE_MXFP4_E2M1_PART0) || \
    defined(MOE_DTYPE_BF16_PART0) || \
    defined(MOE_DTYPE_DOUBLE_BUFFER_BF16_PART0) || defined(MOE_DTYPE_DOUBLE_BUFFER_BF16_PART1) || \
    defined(MOE_DTYPE_DOUBLE_BUFFER_BF16_PART2) || \
    defined(MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3_PART0) || defined(MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3_PART1) || \
    defined(MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1_PART0) || defined(MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1_PART1) || \
    defined(MOE_DTYPE_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART0) || defined(MOE_DTYPE_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART1)
#define MOE_TILE_X_LIST \
  MOE_TILE_LIST_FP8_TENSOR_E4M3 \
  MOE_TILE_LIST_MXFP8_E4M3 \
  MOE_TILE_LIST_MXFP4_E2M1 \
  MOE_TILE_LIST_BF16 \
  MOE_TILE_LIST_DOUBLE_BUFFER_BF16 \
  MOE_TILE_LIST_DOUBLE_BUFFER_MXFP8_E4M3 \
  MOE_TILE_LIST_DOUBLE_BUFFER_MXFP4_E2M1 \
  MOE_TILE_LIST_DOUBLE_BUFFER_FP8_TENSOR_E4M3

// Dtype list = one F(DTYPE) per active basket (drives benchmark registration).
// A tag fires when any part (regular or DB) of that dtype is in this binary,
// since pick_tile() ranks both kinds under the same tag.
#if defined(MOE_DTYPE_FP8_TENSOR_E4M3_PART0) || \
    defined(MOE_DTYPE_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART0) || defined(MOE_DTYPE_DOUBLE_BUFFER_FP8_TENSOR_E4M3_PART1)
#define MOE_DTYPE_TAG_LIST_FP8_TENSOR_E4M3 F(fp8_tensor)
#else
#define MOE_DTYPE_TAG_LIST_FP8_TENSOR_E4M3
#endif
#if defined(MOE_DTYPE_MXFP8_E4M3_PART0) || \
    defined(MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3_PART0) || defined(MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3_PART1)
#define MOE_DTYPE_TAG_LIST_MXFP8_E4M3 F(mxfp8_e4m3)
#else
#define MOE_DTYPE_TAG_LIST_MXFP8_E4M3
#endif
#if defined(MOE_DTYPE_MXFP4_E2M1_PART0) || \
    defined(MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1_PART0) || defined(MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1_PART1)
#define MOE_DTYPE_TAG_LIST_MXFP4_E2M1 F(mxfp4)
#else
#define MOE_DTYPE_TAG_LIST_MXFP4_E2M1
#endif
#if defined(MOE_DTYPE_BF16_PART0) || \
    defined(MOE_DTYPE_DOUBLE_BUFFER_BF16_PART0) || defined(MOE_DTYPE_DOUBLE_BUFFER_BF16_PART1) || \
    defined(MOE_DTYPE_DOUBLE_BUFFER_BF16_PART2)
#define MOE_DTYPE_TAG_LIST_BF16 F(bf16)
#else
#define MOE_DTYPE_TAG_LIST_BF16
#endif

#define MOE_DTYPE_TAG_LIST \
  MOE_DTYPE_TAG_LIST_FP8_TENSOR_E4M3 \
  MOE_DTYPE_TAG_LIST_MXFP8_E4M3 \
  MOE_DTYPE_TAG_LIST_MXFP4_E2M1 \
  MOE_DTYPE_TAG_LIST_BF16
#endif
