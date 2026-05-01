# Xe4 Feature Implementation Status

This document provides a developer-oriented summary of the Xe4 CuTe atom and API implementations in this repository.

| # | Feature | Summary |
|---|---|---|
| 1 | [MMA (AMMA/TMM)](#1-mma-ammatmm--compute-apis) | AMMA async atoms and TMM synchronous tensor-matrix atoms |
| 2 | [Data Types for MMA](#2-data-types-for-mma) | Standard and block-scaled MMA data types (TF32, BF16, FP16, FP8, FP4, MXFP4/8) |
| 3 | [Async DMA (ADMA)](#3-async-dma-adma--data-movement) | Tensor + linear ADMA load/store/prefetch/reduce and multicast flows |
| 4 | [LDSM/STSM](#4-slm--register-ldsmstsm--eu-access-to-core-matrix) | Load/store matrix atoms for SLM ↔ register data movement |
| 5 | [A-Barriers](#5-addressable-barriers-a-barriers) | Addressable barrier init, arrive, wait, and transaction APIs |
| 6 | [Cluster APIs](#6-cluster-apis) | Cluster synchronization, relaxed barriers, and leader election |
| 7 | [Tile Scheduler](#7-tile-scheduler) | Static and dynamic persistent tile schedulers with CLC support |
| 8 | [Collective Builder & Block-Scaled GEMM](#8-collective-builder--block-scaled-gemm-support) | Collective MMA builders for standard and block-scaled GEMM |
| 9 | [Asymmetric Register Allocation](#9-asymmetric-register-allocation) | Control/worker sub-group register partitioning for epilogue |
| 10 | [EU Copy Atoms](#10-eu-copy-atoms) | EU-based copy atoms for GMEM/SLM/register data movement |
| 11 | [Tensor Pipe Quantize/Downconvert](#11-tensor-pipe-quantizedownconvert) | Register-level tensor-pipe quantize/downconvert APIs and XE4 tests |

## 1. MMA (AMMA/TMM) — Compute APIs

| API / Atom | Header | Status |
|---|---|---|
| `XE4_AMMA` (no barrier tracking) | [`mma_xe4_amma.hpp`](../../../include/cute/arch/mma_xe4_amma.hpp) | ✅ Implemented |
| `XE4_AMMA_GROUPSYNC` | [`mma_xe4_amma.hpp`](../../../include/cute/arch/mma_xe4_amma.hpp) | ✅ Implemented |
| `XE4_AMMA_D` (D-barrier tracking) | [`mma_xe4_amma.hpp`](../../../include/cute/arch/mma_xe4_amma.hpp) | ✅ Implemented |
| `XE4_AMMA_AB` (AB-barrier tracking) | [`mma_xe4_amma.hpp`](../../../include/cute/arch/mma_xe4_amma.hpp) | ✅ Implemented |
| `XE4_AMMA_DAB` (DAB-barrier tracking) | [`mma_xe4_amma.hpp`](../../../include/cute/arch/mma_xe4_amma.hpp) | ✅ Implemented |
| `XE4_AMMA_DB` (DB-barrier tracking) | [`mma_xe4_amma.hpp`](../../../include/cute/arch/mma_xe4_amma.hpp) | ✅ Implemented |
| `XE4_AMMA_AB_CLUSTER` (AB multicast) | [`mma_xe4_amma.hpp`](../../../include/cute/arch/mma_xe4_amma.hpp) | ✅ Implemented |
| `XE4_AMMA_DAB_CLUSTER` (DAB multicast) | [`mma_xe4_amma.hpp`](../../../include/cute/arch/mma_xe4_amma.hpp) | ✅ Implemented |
| `XE4_AMMA_FP4FP8` (block-scaled) | [`mma_xe4_amma.hpp`](../../../include/cute/arch/mma_xe4_amma.hpp) | ✅ Implemented |
| `XE4_TMM<d_type, a_type, b_type, c_type, N>` (Tensor MMA) | [`mma_xe4_tmm.hpp`](../../../include/cute/arch/mma_xe4_tmm.hpp) | ✅ Implemented |

## 2. Data Types for MMA

Supported data types are determined by the `getMinMmaK`/`getMaxMmaK` specializations in [`mma_xe4.hpp`](../../../include/cute/arch/mma_xe4.hpp), `XE4_TMM` type-to-K mappings in [`mma_xe4_tmm.hpp`](../../../include/cute/arch/mma_xe4_tmm.hpp), and block-scaled AMMA atoms.

| Data Type | Atom Coverage | Status |
|---|---|---|
| **TF32** (`tfloat32_t`) | `ss_op_selector` | ✅ Implemented |
| **BF16** | `ss_op_selector` | ✅ Implemented |
| **FP16** | `ss_op_selector` | ✅ Implemented |
| **FP8** (E4M3 / BF8 / HF8) | `ss_op_selector` | ✅ Implemented |
| **INT8** | `ss_op_selector` | ✅ Implemented |
| **FP4** (E2M1) | `ss_op_selector` | ✅ Implemented |
| **MXFP4** (block-scaled, ue8m0 k=32/k=16, ue5m3 k=32/k=16, ue4m3 k=16) | `bs_op_selector` + `XE4_AMMA_FP4FP8` | ✅ Implemented |
| **MXFP8** (block-scaled, ue8m0 k=32) | `bs_op_selector` + `XE4_AMMA_FP4FP8` | ✅ Implemented |
| **FP6** (E3M2 / E2M3) | Data type enums only; no MMA atom | ⚠️ Partial |
| **Block-scaled MXFP6** | — | ❌ Not Implemented |

> **Examples:** See [`examples/xe4/gemm/`](../../../examples/xe4/gemm/) for standard GEMM examples, and [`gemm_blockscaled_fp4.cpp`](../../../examples/xe4/gemm/gemm_blockscaled_fp4.cpp) / [`gemm_blockscaled_fp8.cpp`](../../../examples/xe4/gemm/gemm_blockscaled_fp8.cpp) for block-scaled GEMM examples.

## 3. Async DMA (ADMA) — Data Movement

| Atom | Header | Status |
|---|---|---|
| `XE4_ADMA_LOAD` (global→SLM) | [`copy_xe4_adma.hpp`](../../../include/cute/arch/copy_xe4_adma.hpp) | ✅ Implemented |
| `XE4_ADMA_STORE` (SLM→global) | [`copy_xe4_adma.hpp`](../../../include/cute/arch/copy_xe4_adma.hpp) | ✅ Implemented |
| `XE4_ADMA_LOAD_MULTICAST` | [`copy_xe4_adma.hpp`](../../../include/cute/arch/copy_xe4_adma.hpp) | ✅ Implemented |
| `XE4_ADMA_LINEAR_LOAD` / `XE4_ADMA_LINEAR_STORE` | [`copy_xe4_adma.hpp`](../../../include/cute/arch/copy_xe4_adma.hpp) + [`copy_traits_xe4_adma.hpp`](../../../include/cute/atom/copy_traits_xe4_adma.hpp) | ✅ Implemented |
| `XE4_ADMA_LINEAR_PREFETCH` | [`copy_xe4_adma.hpp`](../../../include/cute/arch/copy_xe4_adma.hpp) + [`copy_traits_xe4_adma.hpp`](../../../include/cute/atom/copy_traits_xe4_adma.hpp) | ✅ Implemented |
| `XE4_ADMA_LINEAR_REDUCE` / `XE4_ADMA_STORE_REDUCE` | [`copy_xe4_adma.hpp`](../../../include/cute/arch/copy_xe4_adma.hpp) + [`copy_traits_xe4_adma.hpp`](../../../include/cute/atom/copy_traits_xe4_adma.hpp) | ✅ Implemented |
| `XE4_ADMA_LINEAR_LOAD_MULTICAST_CLUSTER` | [`copy_xe4_adma.hpp`](../../../include/cute/arch/copy_xe4_adma.hpp) + [`copy_traits_xe4_adma.hpp`](../../../include/cute/atom/copy_traits_xe4_adma.hpp) | ✅ Implemented |
| `XE4_ADMA_LINEAR_LOAD_LOCAL_TO_REMOTE_SLM_CLUSTER` | [`copy_xe4_adma.hpp`](../../../include/cute/arch/copy_xe4_adma.hpp) + [`copy_traits_xe4_adma.hpp`](../../../include/cute/atom/copy_traits_xe4_adma.hpp) | ✅ Implemented |

> **Runtime tuning support:** ADMA load/multicast traits support runtime `.with(detail::CacheHint<...>, detail::FillMode<...>)` for cache/fill behavior selection.

## 4. SLM ↔ Register (LDSM/STSM) — EU Access to Core Matrix

| Atom | Description | Header | Status |
|---|---|---|---|
| `XE4_LDSM_Scaler` | Load matrix scalar | [`copy_xe4_ldsm.hpp`](../../../include/cute/arch/copy_xe4_ldsm.hpp) | ✅ Implemented |
| `XE4_LDSM_Vector` | Load matrix simple vector | [`copy_xe4_ldsm.hpp`](../../../include/cute/arch/copy_xe4_ldsm.hpp) | ✅ Implemented |
| `XE4_LDSM_CoopVector` | Load matrix cooperative vector | [`copy_xe4_ldsm.hpp`](../../../include/cute/arch/copy_xe4_ldsm.hpp) | ✅ Implemented |
| `XE4_LDSM_UVector` | Load matrix unordered vector | [`copy_xe4_ldsm.hpp`](../../../include/cute/arch/copy_xe4_ldsm.hpp) | ✅ Implemented |
| `XE4_LDSM_UAOfVector` | Load matrix unordered array-of-vectors | [`copy_xe4_ldsm.hpp`](../../../include/cute/arch/copy_xe4_ldsm.hpp) | ✅ Implemented |
| `XE4_STSM_Scaler` | Store matrix scalar | [`copy_xe4_ldsm.hpp`](../../../include/cute/arch/copy_xe4_ldsm.hpp) | ✅ Implemented |
| `XE4_STSM_Vector` | Store matrix simple vector | [`copy_xe4_ldsm.hpp`](../../../include/cute/arch/copy_xe4_ldsm.hpp) | ✅ Implemented |
| `XE4_STSM_CoopVector` | Store matrix cooperative vector | [`copy_xe4_ldsm.hpp`](../../../include/cute/arch/copy_xe4_ldsm.hpp) | ✅ Implemented |
| `XE4_STSM_UVector` | Store matrix unordered vector | [`copy_xe4_ldsm.hpp`](../../../include/cute/arch/copy_xe4_ldsm.hpp) | ✅ Implemented |
| `XE4_STSM_UAOfVector` | Store matrix unordered array-of-vectors | [`copy_xe4_ldsm.hpp`](../../../include/cute/arch/copy_xe4_ldsm.hpp) | ✅ Implemented |
| `XE4_REDUCE_MATRIX` | Reduce matrix (fred/ired) | [`copy_xe4_ldsm.hpp`](../../../include/cute/arch/copy_xe4_ldsm.hpp) + [`copy_traits_xe4_ldsm.hpp`](../../../include/cute/atom/copy_traits_xe4_ldsm.hpp) | ✅ Implemented |

## 5. Addressable Barriers (A-Barriers)

| API | Header | Status |
|---|---|---|
| `xe4_initialize_barrier()` | [`mma_xe4_desc.hpp`](../../../include/cute/arch/mma_xe4_desc.hpp) | ✅ Implemented |
| `xe4_arrive_barrier()` | [`mma_xe4_desc.hpp`](../../../include/cute/arch/mma_xe4_desc.hpp) | ✅ Implemented |
| `xe4_wait_barrier()` | [`mma_xe4_desc.hpp`](../../../include/cute/arch/mma_xe4_desc.hpp) | ✅ Implemented |
| `xe4_set_barrier_transaction_bytes()` | [`mma_xe4_desc.hpp`](../../../include/cute/arch/mma_xe4_desc.hpp) | ✅ Implemented |

## 6. Cluster APIs

| API | Header | Status |
|---|---|---|
| `cluster_arrive()` / `cluster_wait()` / `cluster_sync()` | [`cluster_xe4.hpp`](../../../include/cute/arch/cluster_xe4.hpp) | ✅ Implemented |
| `cluster_arrive_relaxed()` / `cluster_wait_relaxed()` | [`cluster_xe4.hpp`](../../../include/cute/arch/cluster_xe4.hpp) | ✅ Implemented |
| `elect_one_sync()` | [`cluster_xe4.hpp`](../../../include/cute/arch/cluster_xe4.hpp) | ✅ Implemented |

## 7. Tile Scheduler

| Feature | Implementation | Status |
|---|---|---|
| **Static Persistent Scheduler** | [`StaticPersistentTileSchedulerXe4`](../../../include/cutlass/gemm/kernel/xe4_static_tile_scheduler.hpp) — precomputed coord_tensor maps iteration IDs to tile coordinates | ✅ Implemented |
| **Dynamic Persistent Scheduler (CLC)** | [`PersistentTileSchedulerXe4`](../../../include/cutlass/gemm/kernel/xe4_tile_scheduler.hpp) — uses hardware CLC `cscheduler.get_next` for dynamic tile dispatch | ✅ Implemented (PR #322) |
| Raster order (AlongM / AlongN / Heuristic) | `PersistentTileSchedulerXe4::possibly_transpose_grid()` | ✅ Implemented (PR #322) |
| Other scheduler features | `PipelineCLCFetchAsync` pipeline, `PersistentTileSchedulerXe4Params`, tile swizzling (`swizzle_and_rasterize()`) | ✅ Implemented (PR #322) |
| Scheduler validation (device tests) | [`xe4_tile_scheduler_device.cpp`](../../../test/unit/gemm/scheduler/xe4_tile_scheduler_device.cpp) | ✅ Implemented |
> **Tests:** Device-side unit tests in [`test/unit/gemm/scheduler/xe4_tile_scheduler_device.cpp`](../../../test/unit/gemm/scheduler/xe4_tile_scheduler_device.cpp) validate CLC scheduling with cluster shapes 1×1, 2×1, 1×2, 2×2; swizzle sizes 1/2/4; and all raster orders.

## 8. Collective Builder & Block-Scaled GEMM Support

| Feature | Implementation | Status |
|---|---|---|
| **Block-scaled Collective Builder** | [`xe4_blockscaled_amma_builder.inl`](../../../include/cutlass/gemm/collective/builders/xe4_blockscaled_amma_builder.inl) — selects `bs_op_selector`, SF SMEM layouts, ADMA copy ops | ✅ Implemented |
| **Block-scaled Collective MMA** | [`xe4_blockscaled_mma_warpspecialized.hpp`](../../../include/cutlass/gemm/collective/xe4_blockscaled_mma_warpspecialized.hpp) — ADMA load for data + scale factors, AMMA with `MMAControl` block-scale bits | ✅ Implemented |
| `MainloopXe4DmaGmmaWarpSpecializedBlockScaled` dispatch tag | Dispatch policy for block-scaled ADMA+AMMA warp-specialized mainloop | ✅ Implemented |
| `Xe4BlockScaledConfig` (SF layout helpers) | Scale factor SMEM layout and tiling configuration | ✅ Implemented |
| `BlockScaleTypeMap` (SF type → HW encoding) | Maps scale factor element type to hardware MMAControl encoding | ✅ Implemented |
| FP4xFP8 and FP4/FP8xBF16/FP16 block-scaled paths | Block-scaled data-type expansion on XE4 kernels/examples | ✅ Implemented |
| **Standard Collective Builder** | [`xe4_amma_builder.inl`](../../../include/cutlass/gemm/collective/builders/xe4_amma_builder.inl) | ✅ Implemented |
| **Standard Collective MMA** | [`xe4_mma_warpspecialized.hpp`](../../../include/cutlass/gemm/collective/xe4_mma_warpspecialized.hpp) | ✅ Implemented |

> **Tested configurations (block-scaled):**
>
> | A/B Data Type | Scale Factor Type | Output Types |
> |---|---|---|
> | MXFP8 (e4m3) | ue8m0 (k=32) | fp32, fp16, bf16 |
> | MXFP4 (e2m1) | ue8m0 (k=32), ue8m0 (k=16) | fp32, fp16, bf16 |
> | NVFP4 (e2m1) | ue4m3 (k=16) | fp32, fp16, bf16 |
> | NVFP4+ (e2m1) | ue5m3 (k=32), ue5m3 (k=16) | fp32, fp16, bf16 |
>
> **Examples:** [`gemm_blockscaled_fp4.cpp`](../../../examples/xe4/gemm/gemm_blockscaled_fp4.cpp), [`gemm_blockscaled_fp8.cpp`](../../../examples/xe4/gemm/gemm_blockscaled_fp8.cpp)

## 9. Asymmetric Register Allocation

| Feature | Implementation | Status |
|---|---|---|
| Asymmetric register allocation (`is_control_sub_group()`) | [`gemm_epilogue_asymmetric_adma_amma.cpp`](../../../examples/cute/tutorial/xe4/gemm_epilogue_asymmetric_adma_amma.cpp) — control sub-groups use reduced register files, worker sub-groups use full registers for epilogue | ✅ Implemented |

> **Note:** Asymmetric allocation allows workgroup sizes of 640 or 384 with control sub-groups (ADMA/AMMA producers) and worker sub-groups (epilogue consumers) running different register allocations.

## 10. EU Copy Atoms

| Atom | Description | Header | Status |
|---|---|---|---|
| `XE4_EU_COPY_G2S` | GMEM → SLM (Type-1 K-major, swizzled) | [`copy_xe4_eu_copy.hpp`](../../../include/cute/arch/copy_xe4_eu_copy.hpp) + [`copy_traits_xe4_eu_copy.hpp`](../../../include/cute/atom/copy_traits_xe4_eu_copy.hpp) | ✅ Implemented |
| `XE4_EU_COPY_S2G` | SLM → GMEM (unswizzled) | [`copy_xe4_eu_copy.hpp`](../../../include/cute/arch/copy_xe4_eu_copy.hpp) + [`copy_traits_xe4_eu_copy.hpp`](../../../include/cute/atom/copy_traits_xe4_eu_copy.hpp) | ✅ Implemented |
| `XE4_EU_COPY_S2R` | SLM → Registers | [`copy_xe4_eu_copy.hpp`](../../../include/cute/arch/copy_xe4_eu_copy.hpp) + [`copy_traits_xe4_eu_copy.hpp`](../../../include/cute/atom/copy_traits_xe4_eu_copy.hpp) | ✅ Implemented |
| `XE4_EU_COPY_G2R` | GMEM → Registers | [`copy_xe4_eu_copy.hpp`](../../../include/cute/arch/copy_xe4_eu_copy.hpp) + [`copy_traits_xe4_eu_copy.hpp`](../../../include/cute/atom/copy_traits_xe4_eu_copy.hpp) | ✅ Implemented |
| `XE4_EU_COPY_R2G` | Registers → GMEM | [`copy_xe4_eu_copy.hpp`](../../../include/cute/arch/copy_xe4_eu_copy.hpp) + [`copy_traits_xe4_eu_copy.hpp`](../../../include/cute/atom/copy_traits_xe4_eu_copy.hpp) | ✅ Implemented |
| `make_xe4_g2s_tiled_copy()` | Factory for GMEM→SLM tiled copy | [`copy_traits_xe4_eu_copy.hpp`](../../../include/cute/atom/copy_traits_xe4_eu_copy.hpp) | ✅ Implemented |
| `make_xe4_s2r_tiled_copy()` | Factory for SLM→Register tiled copy | [`copy_traits_xe4_eu_copy.hpp`](../../../include/cute/atom/copy_traits_xe4_eu_copy.hpp) | ✅ Implemented |
| `make_xe4_r2g_tiled_copy()` | Factory for Register→GMEM tiled copy | [`copy_traits_xe4_eu_copy.hpp`](../../../include/cute/atom/copy_traits_xe4_eu_copy.hpp) | ✅ Implemented |

> **Usage:**
> ```cpp
> auto tiled_copy = make_xe4_g2s_tiled_copy<T, M, K>(gmem_tensor, slm_tensor);
> auto thr_copy   = tiled_copy.get_thread_slice(thread_id);
> auto tSrc = thr_copy.partition_S(gmem_tensor);
> auto tDst = thr_copy.partition_D(slm_tensor);
> copy(tiled_copy, tSrc, tDst);
> ```
>
> **Examples:** [`gemm_eu_copy_matrix_atoms.cpp`](../../../examples/cute/tutorial/xe4/gemm_eu_copy_matrix_atoms.cpp)
> **Tests:** [`test/unit/cute/xe4/slm_copy_test.cpp`](../../../test/unit/cute/xe4/slm_copy_test.cpp)

## 11. Tensor Pipe Quantize/Downconvert

| API | Header | Status |
|---|---|---|
| `cute::tensor_pipe_quantize<TcvdDstType, TcvdSrcType>(src, dst)` | [`tensor_processing.hpp`](../../../include/cute/algorithm/tensor_processing.hpp) | ✅ Implemented |
| XE4 Tensor Pipe quantize API validation | [`tensor_pipe_quantize_api_test.cpp`](../../../test/unit/cute/xe4/tensor_pipe_quantize_api_test.cpp) + [`tensor_pipe_quantize_api.hpp`](../../../test/unit/cute/xe4/tensor_pipe_quantize_api.hpp) | ✅ Implemented |

> **Notes:** Tensor Pipe quantize flow is validated in XE4 unit tests and used for register-level downconvert (TCVD) paths.
