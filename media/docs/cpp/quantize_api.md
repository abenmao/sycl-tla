# Block-wise Quantization for SYCL-TLA

## Motivation

### Problem Statement

Current quantization workflows require:
1. Store high-precision GEMM output to global memory
2. Launch separate quantization kernel
3. Load quantized data for next operation

This incurs unnecessary global memory traffic and kernel launch overhead.

### Solution

Provide a `quantize_block_wise` API that:
- Operates directly on `SubgroupTensor` (data in GPU registers)
- Enables fusion with GEMM prologue/epilogue
- Supports MX floating-point formats (MXFP8, MXFP4) and native FP8 with flexible scale types (E8M0, BF16, FP32)
- Leverages hardware instructions (TensorPipe on JGS, FPU on CRI)

### Target fusion patterns

| Pattern | Data flow | When |
|---------|-----------|------|
| **Post-op** (epilogue) | GEMM → Activation → **Quantize** → Store | FFN projections, attention output |
| **Pre-op** (prologue) | Load → Norm → **Quantize** → GEMM | Activation re-quantization before GEMM |

Both patterns keep the high-precision → low-precision conversion entirely in registers. The quantized tensor and its scale factors are written to global memory (post-op) or fed directly to the next DPAS instruction (pre-op) without an extra kernel launch.

---

## API and Implementation Considerations

### 1. API

```cpp
template <int BlockSize,
          class SrcEngine, class SrcLayout, class SrcTVLayout,
          class DstEngine, class DstLayout, class DstTVLayout,
          class ScaleEngine, class ScaleLayout, class ScaleTVLayout>
CUTE_HOST_DEVICE
void quantize_block_wise(
    SubgroupTensor<SrcEngine, SrcLayout, SrcTVLayout> const& src,
    SubgroupTensor<DstEngine, DstLayout, DstTVLayout>      & dst,
    SubgroupTensor<ScaleEngine, ScaleLayout, ScaleTVLayout> & scale
);
```

**File location:** `include/cute/quantization/quantize.hpp` (CuTe
level, reusable across epilogue, prologue, and standalone kernels).

**Design considerations:**

- **Why `dst` and `scale` are input parameters, not return values.**  The caller
  constructs `dst` and `scale` with the desired TV layout *before* calling
  `quantize_block_wise`. This lets the caller choose layouts that match the next
  operation — a store-friendly layout for an epilogue (`XE_2D_STORE`) or an
  MMA-friendly layout for a prologue (`XE_DPAS`). Returning them would force the
  API to pick a layout, adding an extra `reorder()` later.
- **Rounding modes** are hardcoded: **round-toward-zero** for scales,
  `cutlass::FloatRoundStyle::round_to_nearest` for elements.
- All intermediate computation uses **float32**. This matches the MX spec and
  has no performance penalty on Xe (native 32-bit FPU).
- The current implementation uses the software FPU path (`NumericConverter` +
  `reduce_over_group`). A future path can dispatch to `tcvdmx` (TensorPipe
  block-quantize instruction) on architectures that support it.

**Template parameters:**

| Parameter | Description |
|-----------|-------------|
| `BlockSize` | Elements per quantization block (compile-time; must be ≥ `sg_size` = 16) |

**Supported types**:

| Role | Types |
|------|-------|
| Source (`src`) | `bfloat16_t`, `half_t`, `float` |
| Destination (`dst`) | `float_e4m3_t`, `float_e5m2_t`, `float_e2m1_t` |
| Scale (`scale`) | `float_ue8m0_t`, `bfloat16_t`, `float` |

**Shape constraints:**

| Tensor | Logical shape |
|--------|--------------|
| `src` | $(M, N)$ |
| `dst` | $(M, N)$ — must match `src` |
| `scale` | $(M, N / \text{BlockSize})$ — N/BlockSize scales per row |


### 2. Algorithm

The core algorithm is:

1. Compute per-block `amax = max(abs(x_i))` for all elements in the block.
2. Compute `scale = round_toward_zero(target_max(DstType) / amax)`.
3. Quantize: `dst_i = round_to_nearest(x_i * scale)` (using `cutlass::FloatRoundStyle::round_to_nearest`, i.e., round-to-nearest ties-to-even).
4. Reorder elements from source TV layout to destination TV layout via `reorder()`.


#### Quantize along N → needs cross-lane reduction

Consider a 2×32 logical tensor with `sg_size = 16` and `BlockSize = 32`.  The TV layout distributes columns across threads:

```
Logical (M=2, N=32):

         n=0  n=1  n=2  ...  n=15 │ n=16  n=17  ...  n=31
       ┌─────────────────────────────┼──────────────────────────┐
  m=0  │ T0   T1   T2  ...  T15   │ T0    T1   ...   T15     │
  m=1  │ T0   T1   T2  ...  T15   │ T0    T1   ...   T15     │
       └─────────────────────────────┴──────────────────────────┘
                              one quantization block (32 cols)

  Thread 0 owns: (0,0), (1,0), (0,16), (1,16)  — 4 values
  Thread 1 owns: (0,1), (1,1), (0,17), (1,17)  — 4 values
  ...
```

A block of 32 elements along N spans **all 16 threads**.  No single thread sees every element in a block, so computing `amax` requires:
1. **Phase 1 (vertical):** each thread computes a local abs-max over its private values belonging to each (row, block) pair — no cross-lane communication.
2. **Phase 2 (horizontal):** `reduce_over_group` across the subgroup to produce the block-wide `amax` — one cross-lane reduction per (row, block).

This two-phase design minimises expensive shuffles to exactly one `reduce_over_group` call per (row, block) pair.

#### Reorder step

After quantization the element bit-width shrinks (e.g. 16-bit → 8-bit), so the number of elements packed per 32-bit register doubles or quadruples.  The implementation constructs a temporary `SubgroupTensor` with the *source* TV layout and calls `reorder()` to transform it into the caller-supplied *destination* TV layout.  `reorder()` handles cross-lane shuffles, packing, and layout conversion.

#### Preconditions

All of the above leads to the following preconditions. Some are validated via `static_assert` in `quantize.hpp`, while others (notably the thread-stride constraint) are currently usage requirements that are **not** checked at compile time:

- **`(BlockSize % sg_size) == 0`** — each quantization block must be multiple of subgroup width so the compile-time block-index derivation from `tv_layout(0, v)` is valid.
- **`BlockSize` evenly divides the quantized dimension** (N).
- **Thread stride maps only to dimension 1 (N)** — the row index (M) must be thread-independent. This is what makes `tv_layout(0, v)` yield correct (m, block_id) coordinates for all threads; this constraint is required for correctness but is not currently enforced by a `static_assert`.
- **Source and destination have the same logical shape.**
- **Scale shape matches the blocking** — $(M, N/\text{BlockSize})$.
