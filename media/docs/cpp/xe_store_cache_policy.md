# Xe block2D output-store cache policies

SYCL\*TLA exposes the cache-control suffix of Xe `lsc_store_block2d.ugm`
instructions as a compile-time policy. This allows a kernel to select the cache
behavior of its output store without replacing the automatically selected block
shape or adding runtime dispatch.

## Design

`cute::XEStoreCachePolicy` contains `Default` and the explicit policies accepted
by the Xe assembler:

| Family | Policies |
|---|---|
| Default | `Default` |
| UC | `UC_UC_UC`, `UC_UC_WB`, `UC_WB_UC`, `UC_WB_WB` |
| WT | `WT_UC_UC`, `WT_UC_WB`, `WT_WB_UC`, `WT_WB_WB` |
| WB | `WB_UC_UC`, `WB_UC_WB`, `WB_WB_UC` |

`Default` emits `lsc_store_block2d.ugm` without a cache suffix and preserves the
behavior of existing code. An explicit policy emits the corresponding suffix,
for example `WT_WB_WB` emits `lsc_store_block2d.ugm.wt.wb.wb`.

The policy is a template argument because the vISA instruction is fixed during
device compilation. A kernel binary therefore contains one selected policy and
has no runtime policy-selection cost. Changing the policy requires recompiling
the affected kernel instance.

## CuTe APIs

At the instruction level, pass the policy as the final `XE_STORE_2D` template
argument:

```cpp
using Store = cute::XE_STORE_2D<
    32, 4, 16, cute::XEStoreCachePolicy::WT_WB_WB>;
```

The block2D D-store builders retain automatic block-shape selection while
accepting the policy explicitly:

```cpp
auto tiled_copy = cute::make_block_2d_copy_D<
    cute::XEStoreCachePolicy::WT_WB_WB>(tiled_mma, tensor_d);

auto subtiled_copy = cute::make_block_2d_copy_D_subtiled<
    cute::XEStoreCachePolicy::WT_WB_WB>(
        tiled_mma, subgroup_tv_layout, subgroup_layout, tensor_d);
```

Calls without the template argument remain source-compatible and select
`Default`.

## GEMM and Flash Attention integration

The generic Xe GEMM epilogue accepts `cute::C<XeStoreCachePolicy>` as its
register-to-global copy operation. This changes only the cache policy; the
epilogue continues to derive the block2D store shape from the output tensor and
tiled MMA:

```cpp
using CopyOpR2G = cute::C<
    cute::XeStoreCachePolicy::kWT_WB_WB>;
```

The Flash Attention forward epilogue exposes `StoreCachePolicy_` as its final
template argument. Both the normal and subtiled output-store paths receive the
same policy.

Header-only consumers, including downstream vLLM kernels, instantiate their own
kernel binary from these templates. They do not call a precompiled SYCL\*TLA
example at runtime. A downstream epilogue that calls the policy-less builder
continues to use `Default`; it must pass its selected policy to
`make_block_2d_copy_D` and `make_block_2d_copy_D_subtiled` explicitly.

## Flash Attention example policy

The Flash Attention runner selects `WT_WB_WB` directly as the default
`StoreCachePolicyO` template argument. A normal build therefore receives the
tuned policy without a CMake option and still builds exactly one policy for each
existing kernel instance. Developers can pass another `StoreCachePolicyO`
template argument in source when evaluating a different policy.

The default was selected from a reduced CRI prefill sweep covering five BF16
and five FP8 E4M3 cases. Relative to `Default`, `WT_WB_WB` improved all ten
cases, with a 4.80% geometric-mean speedup overall (3.95% for BF16 and 5.66%
for FP8 E4M3). These results apply to the measured Flash Attention workload and
do not imply that the same policy is optimal for GEMM or every downstream
workload.

Downstream projects with their own epilogue or runner must select and propagate
their policy explicitly; changing this example runner does not alter downstream
kernel instantiations.
