# Xe35 Chunkwise Gated DeltaNet (GDN) Attention

A direct CuTe-on-SYCL port of the Gated DeltaNet (GDN) **chunkwise
attention forward pass** from the vllm-xpu-kernels project for Intel Xe GPUs. The kernel runs as four raw
device kernels submitted back-to-back on a single in-order SYCL queue — there
is **no** CUTLASS collective/kernel scaffolding. (The upstream port had a
separate Stage-1 `chunk_prepare` kernel for the cumulative-sum gate; it is now
**fused into** `chunk_compute_A_o2`, so the pipeline is four stages.)

> ⚠️ **Status — initial port, not yet Xe3-optimized.**
> This is the **first step** of bringing the upstream Xe2 GDN kernels into the
> SYCL-TLA environment: getting them to **build and run correctly** here. The
> code is a straight functional port — it does **not** yet exploit Xe3 (CRI)
> hardware features, except for setting GRF size at compilation (CRI: 512 vs. BMG 256).
> Running it on CRI as-is is expected to work but **not** to
> be performant; a follow-up pass is needed to adapt the kernels (tiling,
> register-file usage, DPAS shapes, etc.) to take advantage of Xe3 before any
> CRI performance numbers are meaningful.

> **At a glance**
> - **What:** GDN chunkwise bf16 activations + fp32 SSM state.
> - **Targets:** Intel **CRI** (Xe-3.5) and **BMG** (Xe20).
> - **Origin:** a CuTe/SYCL chunkwise GDN attention implementation for Intel Xe, with a caller-visible ABI modeled on common GDN attention kernels.

## Contents

- [Where the code lives](#where-the-code-lives)
- [The four-stage pipeline](#the-four-stage-pipeline)
  - [Data flow through the stages](#data-flow-through-the-stages)
- [Work hierarchy & GPU mapping](#work-hierarchy--gpu-mapping)
- [Public API](#public-api)
  - [`GDNArguments` reference](#gdnarguments-reference)
    - [Symbols & dtypes](#symbols--dtypes)
    - [Shape fields](#shape-fields)
    - [Input tensors](#input-tensors)
    - [Output tensors](#output-tensors)
    - [Workspace tensors](#workspace-tensors)
  - [Return status](#return-status)
  - [Mutability contract](#mutability-contract)
  - [`has_initial_state` contract](#has_initial_state-contract)
  - [The sigmoid of `b`](#the-sigmoid-of-b)
- [Sizing the workspaces](#sizing-the-workspaces)
- [Integrating the launcher](#integrating-the-launcher)
- [Constraints](#constraints)
- [File map](#file-map)

## Where the code lives

The kernel itself is **header-only** and lives under
[`applications/gdn_attention/`](../../../applications/gdn_attention), alongside
a single shared host harness,
[`gdn_runner.hpp`](../../../applications/gdn_attention/gdn_runner.hpp). The
runner is the one source of truth for device allocation, input initialization
(including the [sigmoid of `b`](#the-sigmoid-of-b)), `GDNArguments`
construction, kernel launch, and the host verification oracles. All three
consumers build on it rather than re-implementing setup:

| Consumer | Path | Adds on top of `gdn_runner.hpp` |
|---|---|---|
| Example | [`examples/14_xe35_gdn_attention/`](../../../examples/14_xe35_gdn_attention) | CLI `Options` + perf timing; verifies via the recurrent oracle |
| Benchmark | [`benchmarks/applications/03_gdn`](../../../benchmarks/applications/03_gdn) | FLOP/byte model + Google Benchmark harness + configuration sweep |
| Unit test | [`test/unit/gdn_attention/`](../../../test/unit/gdn_attention) | GoogleTest pass gate (`cutlass_test_unit_gdn_attention_chunkwise`) |

`gdn_runner.hpp` also hosts the shared CLI shape helpers (`parse_gdn_shape` /
`validate_gdn_shape`) and the `ExampleOptions` / `BenchmarkOptions` structs, so
all command-line parsing has one home; the unit test sets shape fields on the
runner directly and never touches them. See the full [file map](#file-map)
below for every header and its purpose.


## The four-stage pipeline

Each call to
[`cutlass::gdn::chunk_gated_delta_rule_launch<T, StateT>`](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule.hpp)
submits four device kernels on one in-order SYCL queue — no host wait between
stages.

| # | Stage | What it computes |
|---|---|---|
| 1 | `chunk_compute_A_o2` | **(a) Cumulative gate (fused, was `chunk_prepare`):** compute `a[t] = cumsum(softplus(a + dt_bias) * -exp(A_log))` in place — hoisted ahead of the norm, the `kv_ratio` v-heads of this k-head striped one-per-sub-group so each whole-chunk scan stays inside one sub-group. **(b) L2-normalize** Q (scaled by `1/sqrt(D)`) and K in place — per k-head, on the rows this work item is about to consume. **(c) Fused dual GEMM** sharing the `K` operand: build the lower-triangular transition matrix `L[m,n] = (K_m·K_n) * exp(a[m] - a[n]) * b[m]` into `A_workspace`, **and** the decay-gated intra-chunk score `O2[m,n] = (Q_m·K_n) * exp(a[m] - a[n])` (causal, `m>=n`) into `o2_workspace`. |
| 2 | `chunk_inverse` | Invert `L` in place, one 64×64 chunk matrix per sub-group: a 4×4 grid of 16×16 blocks, diagonal blocks inverted in registers, off-diagonal blocks filled by 16×16×16 DPAS (block forward substitution). |
| 3 | `chunk_compute_wu` | `U = L^-1 * V * diag(b)` and `W = L^-1 * K * diag(exp(a) * b)`. |
| 4 | `chunk_fwd_o` | `O = Q * S^T * exp(g) + O2 * U` (reads the precomputed `O2`); update SSM state `S_out = exp(g_last) * S_prev + U^T * K_scaled`. |

**Workspaces.** Four workspace tensors are supplied by the caller
(`A_workspace`, `o2_workspace`, `w_workspace`, `u_workspace`); element counts
come from
[`cutlass::gdn::get_workspace_sizes()`](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule.hpp).
They are sized by `total_virtual_seqlen` (the chunk-padded extent), **not**
`total_seqlen` (real tokens), because that is the stride the kernels use when
indexing them.

### Data flow through the stages

Each stage reads the tensors produced upstream and writes the next. `q`/`k`/`a`
are mutated **in place** (see [mutability contract](#mutability-contract)); the
`A`/`o2`/`w`/`u` workspaces are scratch buffers reused only within a launch.

```
  STAGE              READS                          WRITES
  ─────              ─────                          ──────
  1 compute_A_o2     a, A_log, dt_bias              a                  (cumsum gate, in place)
    (incl. fused     q, k, a, b                     q, k               (L2-norm, in place)
     prepare gate)                                  A  (L, lower-tri)  [workspace]
        │                                           o2 (O2, causal)    [workspace]
        ▼
  2 inverse          A                              A  (L^-1)          [workspace, in place]
        │
        ▼
  3 compute_wu       A, q, k, v, b, a, A_log,       w (W), u (U)       [workspace]
        │            dt_bias, has_initial_state
        ▼
  4 fwd_o            o2, w, u, q, k, a, ssm_state   core_attn_out  [total_seqlen, n_vh, hv]
                                                    ssm_state      (recurrent state, in place)

  Legend:  in place = mutates a caller tensor   ·   workspace = A/o2/w/u scratch (per launch)
```

## Work hierarchy & GPU mapping

GDN decomposes the problem along three nested axes, and the kernels map them
onto the Xe execution hierarchy. Understanding this mapping explains the launch
geometry of each stage.

```
  PROBLEM DECOMPOSITION                         Xe EXECUTION HIERARCHY
  ─────────────────────                         ──────────────────────

  batch (variable-length sequences)
   └─ v_head           (num_v_heads)            sycl::nd_range<3>
       └─ chunk        (seq padded to kChunkSize)│
           └─ token    (kChunkSize = 64)         ├─ work-group  (Xe-core)
               └─ head_dim element               │   └─ co-resident group¹ (size(mma) lanes)
                                                 │       └─ sub-group (16 lanes, 1 DPAS row)
                                                 │           └─ work-item (SIMD lane)
  Tile math per chunk: kChunkSize×kChunkSize (64x64) transition matrix,
  inverted by ONE sub-group as a 4×4 grid of 16×16 DPAS blocks.
```

> ¹ **Stage 1 only.** Its Xe-core work-group is subdivided into
> `GroupsPerWg = MaxThreadsPerXeCore / size(mma)` co-resident cooperative
> groups, each `size(mma)` lanes and each owning one `(chunk, k_head)` pair per
> persistent iteration. Stages 2–4 map one work-group directly onto the DPAS
> tile with no intermediate group level.

**How each stage is launched.** All four stages share one in-order queue.
`xe_core_count` is the device Xe-core count (floored — see [Constraints](#constraints));
`MaxThreadsPerXeCore == 512`, `sub_group_size == 16`.

```
  Stage               grid (global, dim-1 axis)        work-group   work split
  ──────────────────  ───────────────────────────────  ───────────  ─────────────────────────
  1 chunk_compute_A_o2 xe_core_count (one WG / Xe-core)  size(mma)·   GroupsPerWg co-resident groups of
                                                         GroupsPerWg  size(mma) lanes; each group ↦ one
                                                                      (chunk,k_head) pair per persistent
                                                                      iteration; fused cumsum gate +
                                                                      per-k_head L2-norm + L + O2 dual
                                                                      GEMM
  2 chunk_inverse     max(xe_core_count·512/16,         16 (1 sub-   1 work-group ↦ (chunk,v_head);
                          ⌈tvs/64⌉·num_v_heads)         group)       4×4 block forward-substitution
  3 chunk_compute_wu  xe_core_count                     512 threads  1 work-group ↦ one chunk; holds
                                                                      num_tiles MMA tiles, one v_head
                                                                      per tile
  4 chunk_fwd_o       grid = (batch, num_v_heads,       MMA wg_size  1 work-group ↦ (batch,v_head,dv);
                             head_v_dim / kChunkSize)                 sequential scan over chunks
```

Stages 1–3 launch a *persistent* grid (sized to fill the device) and use an
internal grid-stride loop to stride over all their work units, so a short grid
still covers every unit of work. Stage 1 differs in how that work is indexed:
instead of a 2D per-k_head grid it launches **one work-group per Xe-core**
(`global(1, xe_core_count, 1)`) and fills each work-group with
`GroupsPerWg = MaxThreadsPerXeCore / size(mma)` **co-resident cooperative
groups** of `size(mma)` lanes each (the WG size is the group-aligned
`GroupsPerWg · size(mma)`, so there are never remainder lanes whose `group_id`
would index past the SLM — enforced by a `static_assert` on divisibility).
Every group independently pulls one `(chunk, k_head)` pair per persistent
iteration from a single flat work-list of every such pair
(`w % num_k_heads → k_head`, `w / num_k_heads → chunk`); a `has_work` guard
lets trailing groups with no pair left idle through the shared barriers while
their busy siblings finish. Any group can pull any pair, so none idles behind
its own head's exhausted chunk list; and because consecutive `w` share a chunk
across heads, co-resident groups reuse that chunk's K/Q rows in L1. Each group
owns its own `kv_ratio · kChunkSize`-float SLM slot
(`group_slm_ptr = slm_mem + group_id · kv_ratio · kChunkSize`, total SLM
`GroupsPerWg · kv_ratio · kChunkSize`) so co-resident groups never alias each
other's gate scratch, and the shared-B GEMM runs through the barrier-free
`gemm_TTS_shareB_multi_tile` (register-private per lane) so idle groups skip it
cleanly. The fused cumsum gate runs at the
top of each chunk iteration: the `kv_ratio` v-heads of the chunk's k-head are
striped one-per-sub-group so each whole-chunk prefix-sum stays inside a single
sub-group, and the gate's fence folds into the normalization barrier (no added
barrier). Stage 4 instead launches one work-group per `(batch, v_head, dv)`
tile: its chunk loop must walk that head's chunks **in order** (the SSM-state
recurrence carries `S` across chunks), but the `head_v_dim` tiling axis (`dv`,
one 64-wide `head_v_dim` slice per tile) is independent across tiles because the
recurrence only touches `S[dv, :]`, so it is mapped to the third grid dimension
(`head_v_dim / kChunkSize` tiles) instead of an in-kernel loop.


## Public API

To **launch** the kernel, include the launch header — it carries the inline
launcher definition and pulls in the device kernels. The lightweight public
header `xe35_chunk_gated_delta_rule.hpp` only *declares* the entry point and is
for host-only consumers that just need `GDNArguments` / `get_workspace_sizes`.

```cpp
#include "gdn_attention/xe35_chunk_gated_delta_rule_launch.hpp"

cutlass::gdn::GDNArguments args = ...;          // shapes + raw device pointers
auto sizes = cutlass::gdn::get_workspace_sizes(args);
// caller allocates A_workspace / w_workspace / u_workspace
// of sizes.* elements of T

sycl::queue queue{sycl::gpu_selector_v,
                  sycl::property::queue::in_order()};

cutlass::Status s = cutlass::gdn::chunk_gated_delta_rule_launch<
    cutlass::bfloat16_t, float>(queue, args);
queue.wait_and_throw();
```

### `GDNArguments` reference

[`GDNArguments`](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule.hpp)
is one plain struct holding every input the kernel needs — all the shapes and
device pointers in one place. There is no builder, no handle, and no hidden
state: you fill in the shape scalars and raw device pointers, then pass it by
`const&` to the launcher. Default-initialize it
(`GDNArguments a{};`) so any field you forget is a null/zero you can catch,
rather than garbage.

The fields fall into four groups — [shape](#shape-fields),
[inputs](#input-tensors), [outputs](#output-tensors), and
[workspaces](#workspace-tensors). **All tensors are row-major.** The symbols and
dtypes used in the tables below are defined first, so you can read every layout
column without jumping ahead.

#### Symbols & dtypes

| Symbol | Definition |
|---|---|
| `T` | Activation dtype — `cutlass::bfloat16_t`. A `void*` field tagged `→ T` points at `T` elements. |
| `StateT` | SSM-state dtype — `float`. |
| `FP32` / `int32` / `bool` | Element types of the side arrays (gates, offsets, the `has_initial_state` flags). |
| `tvs` | `total_virtual_seqlen` — real tokens after padding **each** sequence to a multiple of 64. |
| `kChunkSize` | Fixed chunk length, **64** (compile-time constant). The 64 is the baseline Xe2 (BMG) value inherited from the upstream port, not retuned for Xe3. |
| `cache_batch` | Number of `ssm_state` slots the caller manages (leading extent of `ssm_state`). |

#### Shape fields

These scalars describe the problem. They must be consistent with the pointer
layouts below — the kernel trusts them and indexes accordingly.

| Field | Type | Meaning |
|---|---|---|
| `batch_size` | `int` | Number of sequences. Equals `query_start_loc.size() − 1`. |
| `total_seqlen` | `int` | Total **real** (unpadded) tokens across all sequences. Stride of `core_attn_out`. |
| `total_virtual_seqlen` | `int` | Total tokens after padding **each** sequence up to a multiple of `kChunkSize` (64). Stride of `q`/`k`/`v` and the workspaces. See [`tvs`](#symbols--dtypes). |
| `num_k_heads` | `int` | Number of key/query heads. Must divide `num_v_heads`. |
| `num_v_heads` | `int` | Number of value heads (the GQA-expanded count). |
| `head_k_dim` | `int` | Per-head Q/K dimension. Positive multiple of 64 (validated: 128). |
| `head_v_dim` | `int` | Per-head V/output dimension. Positive multiple of 64 (validated: 128). |
| `ssm_state_stride_0` | `int` | Element stride between batch slots in `ssm_state`. Typically `num_v_heads · head_v_dim · head_k_dim`, but may be larger for a padded/aligned slot layout (matches upstream `ssm_state.stride(0)`) |

> **Why two seqlen fields?** `total_seqlen` is what the model actually produced;
> `total_virtual_seqlen` is the chunk-padded extent the kernel iterates over.
> Outputs are indexed by the former, the chunkwise machinery (and every
> workspace) by the latter. Mixing them up is the most common sizing bug — hence
> [`get_workspace_sizes`](#sizing-the-workspaces) takes the padded one for you.

#### Input tensors

| Field | Type | Layout | In/Out |
|---|---|---|---|
| `q` | `void*` → `T` | `[tvs, num_k_heads, head_k_dim]` | **in/out** — L2-normalized & scaled in place by stage 1 |
| `k` | `void*` → `T` | `[tvs, num_k_heads, head_k_dim]` | **in/out** — L2-normalized in place by stage 1 |
| `v` | `const void*` → `T` | `[tvs, num_v_heads, head_v_dim]` | in |
| `b` | `const float*` | `[num_v_heads, tvs]` | in — per-token gate `b` |
| `a` | `float*` | `[num_v_heads, tvs]` | **in/out** — per-token log-scale gate; cumsum'd in place by stage 1 (fused prepare) |
| `A_log` | `const float*` | `[num_v_heads]` | in — per-head log decay |
| `dt_bias` | `const void*` → `T` | `[num_v_heads]` | in — per-head timestep bias |
| `query_start_loc` | `const int*` | `[batch_size + 1]` | in — packed sequence boundaries (prefix-sum offsets) |
| `cache_indices` | `const int*` | `[batch_size]` | in — `ssm_state` slot for each batch |
| `has_initial_state` | `const bool*` | `[batch_size]` or `nullptr` | in — see [contract](#has_initial_state-contract) |

The three **in/out** pointers (`q`, `k`, `a`) are the [mutability
contract](#mutability-contract): snapshot them on the host first if you need the
originals afterward.

#### Output tensors

| Field | Type | Layout | Notes |
|---|---|---|---|
| `core_attn_out` | `void*` → `T` | `[total_seqlen, num_v_heads, head_v_dim]` | The attention output. Sized by **real** tokens. |
| `ssm_state` | `void*` → `StateT` | `[cache_batch, num_v_heads, head_v_dim, head_k_dim]` | Recurrent state, **read then written in place** per batch. |

`ssm_state`'s leading extent `cache_batch` is the caller's slot pool, **not** a
`GDNArguments` field: batch `batch_id` reads/writes slot
`cache_indices[batch_id]`, and `ssm_state_stride_0` is the element stride between
slots. This decouples the on-device state cache from the per-launch batch order
(the vLLM paged-cache pattern).

#### Workspace tensors

Scratch buffers, allocated by the caller, reused only within a single launch.
Size them with [`get_workspace_sizes`](#sizing-the-workspaces) — do not hand-roll
the arithmetic.

| Field | Type | Layout | Produced by → consumed by |
|---|---|---|---|
| `A_workspace` | `void*` → `T` | `[num_v_heads, tvs, kChunkSize]` | stage 1 (`L`) → stage 2 (`L⁻¹`) → stage 3 |
| `o2_workspace` | `void*` → `T` | `[num_v_heads, tvs, kChunkSize]` | stage 1 (`O2`) → stage 4 |
| `w_workspace` | `void*` → `T` | `[num_v_heads, tvs, head_k_dim]` | stage 3 (`W`) → stage 4 |
| `u_workspace` | `void*` → `T` | `[num_v_heads, tvs, head_v_dim]` | stage 3 (`U`) → stage 4 |

### Return status

`chunk_gated_delta_rule_launch` is **asynchronous**: it validates the problem
shape on the host, submits the four stages, and returns immediately. The caller
must `queue.wait_and_throw()` (or otherwise synchronize) before reading
`core_attn_out` / `ssm_state`. The returned `cutlass::Status` reflects only the
host-side validation that happens *before* dispatch:

| Condition | Result |
|---|---|
| `batch_size <= 0` or `total_virtual_seqlen <= 0` | `Status::kSuccess` (valid no-op — nothing is submitted) |
| Any of `num_k_heads / num_v_heads / head_k_dim / head_v_dim <= 0` | `Status::kErrorInvalidProblem` (guards the GQA modulo) |
| `num_v_heads % num_k_heads != 0` | `Status::kErrorInvalidProblem` |
| otherwise | `Status::kSuccess` (four stages submitted) |

A `kSuccess` return therefore means **submitted**, not **finished** — runtime
faults (e.g. a null required pointer) surface from `queue.wait_and_throw()`, not
from the return value.

Device pointers are **not** null-checked — a null required pointer faults
inside the kernels by design, matching the raw-pointer ABI of the upstream
entry point. `has_initial_state` is the one documented nullable pointer.

### Mutability contract

Stages 1+ mutate `q`, `k`, `a`, and `ssm_state` **in place** on the device.
The caller MUST NOT rely on their pre-launch contents after the call. The
shared [`GdnRunner`](../../../applications/gdn_attention/gdn_runner.hpp)
snapshots the originals into host vectors before the launch precisely because
the host reference oracles need them to replay the pipeline.

### `has_initial_state` contract

`GDNArguments::has_initial_state` is a nullable `const bool*` of length
`batch_size`

- **non-null:** `has_initial_state[batch_id]` selects whether stages 4-5 load
  the prior SSM state from `ssm_state[cache_indices[batch_id], ...]` or start
  from zero.
- **null:** treated as "every batch has carry-over state". The caller is then
  responsible for pre-zeroing the corresponding `ssm_state` slots before the
  launch — mirrors the Python idiom
  `initial_state[~has_initial_state, ...] = 0` in
  `vllm/model_executor/layers/mamba/gdn`.

Both kernel-side load sites null-check the pointer with
`(has_initial_state == nullptr) || has_initial_state[batch_id]`.

### The sigmoid of `b`

The kernel reads `b` **verbatim** as the delta-rule write strength `beta` and
assumes it already lies in `(0, 1)`. It does **not** apply a sigmoid itself: in
a full GDN model the causal conv1d front-end emits an in-range `b`, and the
chunk kernel is the stage *after* that front-end. This mirrors upstream
[`vllm-xpu-kernels`](https://github.com/vllm-project/vllm-xpu-kernels/tree/main/csrc/xpu/gdn_attn),
where the chunk kernel (`chunk_gated_delta_rule_kernels_xe2.hpp`) likewise reads
`b` raw and the sigmoid is applied one stage earlier by the conv1d front-end
(`chunk_causal_conv1d_xe2.hpp`, `b_value = act_sigmoid(b_value)`). The contract
is documented at the kernel's `b` load site in `chunk_compute_wu` (see the
`SIGMOID CONTRACT` comment in
[`xe35_chunk_gated_delta_rule_kernels.hpp`](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule_kernels.hpp)).

**Callers that bypass the conv1d front-end must sigmoid `b` on the host before
the launch.** The shared
[`GdnRunner`](../../../applications/gdn_attention/gdn_runner.hpp) does this in
`initialize()`: it fills `b` with raw values in `[-2, 2]`, snapshots that raw
copy (`h_b_raw`) for the oracles, then applies
[`apply_sigmoid_b`](../../../tools/util/include/cutlass/util/reference/host/xe35_gdn_attention_stage_references.hpp)
in place before handing `b` to the kernel. The raw snapshot matters because the
host oracles expect an already-sigmoided `b`, which `verify_*()` re-derives
from the raw copy — snapshotting after the in-place sigmoid would double-apply it.

> **Why it matters numerically:** raw, un-sigmoided `b` is `O(1)` or larger, so
> the chunk transition matrix `L[m,n] = (K_m·K_n)·exp(a[m]−a[n])·b[m]` grows
> unbounded and the stage-3 inverse overflows to ±inf, propagating NaN through
> `compute_wu` and `fwd_o`. Keeping `b ∈ (0,1)` is what keeps `L`
> well-conditioned for the 64-step forward substitution.

## Sizing the workspaces

[`get_workspace_sizes(args)`](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule.hpp)
returns a `GDNWorkspaceSizes` with the **element counts** (of `T`) for the four
scratch buffers. It reads only the shape fields, so you can call it on a
shape-only `GDNArguments` before any device pointer exists:

```cpp
struct GDNWorkspaceSizes {
  size_t A_elems;   // num_v_heads · total_virtual_seqlen · kChunkSize
  size_t o2_elems;  // num_v_heads · total_virtual_seqlen · kChunkSize
  size_t w_elems;   // num_v_heads · total_virtual_seqlen · head_k_dim
  size_t u_elems;   // num_v_heads · total_virtual_seqlen · head_v_dim
};
```

```cpp
cutlass::gdn::GDNArguments shape = /* shape scalars only */;
auto ws = cutlass::gdn::get_workspace_sizes(shape);

// allocate `ws.A_elems` / `ws.o2_elems` / `ws.w_elems` / `ws.u_elems` elements
// **of T** (e.g. bytes = ws.A_elems * sizeof(T)), then store the pointers back:
shape.A_workspace  = /* device alloc of ws.A_elems  × sizeof(T) */;
shape.o2_workspace = /* device alloc of ws.o2_elems × sizeof(T) */;
shape.w_workspace  = /* device alloc of ws.w_elems  × sizeof(T) */;
shape.u_workspace  = /* device alloc of ws.u_elems  × sizeof(T) */;
```

The counts use `total_virtual_seqlen` (the chunk-padded extent), because that is
exactly the stride the kernels apply when indexing the workspaces (e.g.
`v_head_id · total_virtual_seqlen · kChunkSize`). The shared `GdnRunner` follows
this pattern: it builds a `make_arguments_shape_only()` struct, sizes the
workspaces from it, then fills in pointers in `make_arguments()`.

## Integrating the launcher

End-to-end, a caller does five things. Steps 1–2 are pure host work (no device
code), so they can live in a translation unit that includes only the lightweight
[`xe35_chunk_gated_delta_rule.hpp`](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule.hpp);
the launch in step 4 needs the [launch
header](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule_launch.hpp).

```cpp
#include "gdn_attention/xe35_chunk_gated_delta_rule_launch.hpp"

using T      = cutlass::bfloat16_t;   // activations
using StateT = float;                 // SSM state

// 1. Describe the problem (shape scalars).
cutlass::gdn::GDNArguments args{};
args.batch_size           = batch;
args.total_seqlen         = total_seqlen;            // real tokens
args.total_virtual_seqlen = total_virtual_seqlen;    // chunk-padded tokens
args.num_k_heads          = num_k_heads;
args.num_v_heads          = num_v_heads;             // % num_k_heads == 0
args.head_k_dim           = 128;                     // multiple of 64
args.head_v_dim           = 128;                     // multiple of 64
args.ssm_state_stride_0   = num_v_heads * head_v_dim * head_k_dim;

// 2. Size + allocate the four workspaces (host-only math).
auto ws = cutlass::gdn::get_workspace_sizes(args);
//   allocate ws.A_elems / ws.o2_elems / ws.w_elems / ws.u_elems elements of T on the device

// 3. Fill in every device pointer (inputs, outputs, workspaces).
args.q = d_q;  args.k = d_k;  args.v = d_v;
args.b = d_b;  args.a = d_a;  args.A_log = d_A_log;  args.dt_bias = d_dt_bias;
args.query_start_loc = d_qsl;  args.cache_indices = d_cache;
args.has_initial_state = d_has_init;   // or nullptr (see contract)
args.core_attn_out = d_out;  args.ssm_state = d_state;
args.A_workspace = d_A;  args.o2_workspace = d_o2;
args.w_workspace = d_w;  args.u_workspace = d_u;

// 4. Launch on an IN-ORDER queue and synchronize.
sycl::queue queue{sycl::gpu_selector_v, sycl::property::queue::in_order()};
cutlass::Status st = cutlass::gdn::chunk_gated_delta_rule_launch<T, StateT>(queue, args);
if (st != cutlass::Status::kSuccess) { /* bad problem shape — see Return status */ }

// 5. Wait, then read core_attn_out / ssm_state.
queue.wait_and_throw();
```

**Checklist before you launch**

- [ ] Queue created `in_order()` — non-negotiable ([Constraints](#constraints)).
- [ ] `num_v_heads % num_k_heads == 0` and all head dims are positive multiples of 64.
- [ ] `q`/`k`/`v`/`b`/`a` sized by `total_virtual_seqlen`; `core_attn_out` by `total_seqlen`.
- [ ] Workspaces sized via `get_workspace_sizes` (in elements of `T`).
- [ ] Originals of `q`/`k`/`a` snapshotted if you still need them — they are clobbered.
- [ ] `ssm_state` slots for batches **without** carry-over pre-zeroed when `has_initial_state == nullptr`.

## Constraints

- **`num_v_heads` must be an exact integer multiple of `num_k_heads`
  (`num_v_heads >= num_k_heads`, GQA-style head grouping).** The kernel maps each
  v-head back to its owning k/q-head via integer division
  (`k_head = v_head / (num_v_heads / num_k_heads)`), so a non-multiple would
  misalign v-heads to the wrong k/q-head. The launcher rejects
  `num_v_heads % num_k_heads != 0` with `Status::kErrorInvalidProblem`.
- **`kChunkSize == 64` is baked in.** The value 64 is the baseline Xe2 (BMG)
  choice inherited from the upstream port
  ([`vllm-xpu-kernels` `csrc/xpu/gdn_attn/xe_2`](https://github.com/vllm-project/vllm-xpu-kernels/tree/main/csrc/xpu/gdn_attn/xe_2));
  it has **not** been retuned for Xe3 (CRI). Mirrored in the kernel namespace as
  `cutlass::gdn::detail::chunk_size`. Sequences are padded per batch to a
  multiple of 64. Changing the constant requires touching every fixed-trip
  inner loop in the kernels.
- **`head_k_dim` / `head_v_dim` must be positive multiples of `kChunkSize`
  (64).** The runner and benchmark reject any other value; `128 / 128` is the
  only configuration currently validated end-to-end. The `compute_wu` and `fwd_o` stages
  walk the head dims in 64-wide column tiles, so a non-multiple would silently
  drop the trailing `head_dim % 64` columns rather than fault. Supporting other
  dims is a partial-tile change to those loops, not an algorithmic one.
- **Correctness is validated only for full-chunk sequences.** Per-batch
  `seq_len` **must be a multiple of `kChunkSize`** (currently 64: valid lengths
  are 64, 128, 256, 512, ...). The kernel will not fault on non-multiples
  (padding is applied automatically), but numerical correctness is **not
  guaranteed**: partial-tail sequences where `seq_len % kChunkSize != 0` (e.g.,
  200 = 3×64 + 8-row tail) exhibit differences between the kernel output and
  both the recurrent and chunkwise fp32 reference oracles on some seeds (~0.05%
  of SSM state elements exceed the 5% tolerance band, typically within 10-20%).
  Accordingly, the example driver, benchmark, and unit test all use only
  chunk-aligned `seq_len` values.
- **In-order queue required.** The launcher submits four stages without any
  host wait; an out-of-order queue would corrupt the pipeline.
- **No framework dependency.** Inputs are raw `void*` device pointers; this
  kernel does NOT plug into the CUTLASS collective/kernel `GemmUniversal`
  scaffolding. It is intentionally a direct CuTe-on-SYCL kernel.
- **Inverse-stage grid is a multiple of `num_v_heads`.** Stage 2
  (`chunk_inverse`) derives its `(chunk, v_head)` assignment from
  `group(1) % num_v_heads` and `group(1) / num_v_heads`, so the launcher rounds 
  its machine-sized grid **up** to a multiple of `num_v_heads`
  (`⌈machine_groups / num_v_heads⌉ · num_v_heads`) so the head id and the
  persistent chunk stride divide exactly. (The former Stage-1 `chunk_prepare`
  also constrained the grid this way; with prepare fused into Stage 1 the flat
  `(chunk, k_head)` work-list no longer divides by `num_v_heads`, so that floor
  is gone.)

### Constraints of the `compute_wu` intra-work-group tile fan-out

Stage 3 now launches a full-Xe-core work-group (`MaxThreadsPerXeCore == 512`
work-items) instead of one MMA tile, so it holds `num_tiles =
MaxThreadsPerXeCore / size(mma)` independent MMA tiles (currently 4), each
driving a different `v_head`:

```
  tile_id       = local_id / size(mma)     // which tile
  tile_local_id = local_id % size(mma)     // position within that tile
```

Grid mapping is unchanged (one work-group per `chunk_id`); tile `T` starts at
`v_head = T` and strides by `num_tiles`. The design is barrier-free at
work-group scope, which is what lets tiles run different numbers of v_heads
without deadlocking:

- **No work-group-scope barrier in the per-tile region.** Any such barrier
  would deadlock once one tile runs out of v_heads before another.
- **Use `gemm_TTS_k_multi_tile`**, which slices by `tile_local_id` and has no
  k-loop barriers (the pipelined prefetch they served is not worthwhile at
  `k_tile_count == 2`).
- **SLM scale arrays are per sub-group, not per tile** — every sub-group of a
  tile reads the entire `chunk_size` diagonal span, so a per-tile slice would
  create a cross-sub-group dependency. The fill is lane-private (stride
  `sub_group_size` from `sg_local_id`), so no barrier is needed. Costs 16 KB
  of SLM.
- **An MMA tile must be a whole number of sub-groups, and the core a whole
  number of tiles** — both enforced by `static_assert` in the launcher.
- **Launcher SLM size and kernel slicing must stay in sync** — the launcher
  allocates `(local_range / sub_group_size) * chunk_size * 2`; the kernel
  derives the same split from `sg.get_group_linear_id()`.


- the public `GDNArguments` wrapper (this repo's preferred ABI) and its
  validation gates,
- the `xe_core_count` floor described above.

## File map

The headers form two layers: a lightweight **API surface** that host-only code
can include freely, and the **device layer** that pulls in SYCL kernel code.
Callers that launch include the launcher; callers that only need shapes include
the public header.

```
  CONSUMERS                              GDN HEADERS (applications/gdn_attention/)
  ─────────                              ─────────────────────────────────────────

  examples/14_xe35_gdn_attention ─┐
  benchmarks/applications/03_gdn ─┼─include─▶ gdn_runner.hpp
  test/unit/gdn_attention ────────┘            (shared host harness: GdnRunner,
                                                CLI Options, host oracles)
                                                  │ includes
                                                  ▼
                                             xe35_chunk_gated_delta_rule_launch.hpp
                                                  (inline launcher: validate + dispatch)
                                                  │ includes
                                                  ▼
                                             xe35_chunk_gated_delta_rule_kernels.hpp
                                             (4 device kernels + kernel_launcher)
                                                  │ uses
                                                  ▼
                                             xe35_chunk_gated_delta_rule_gemm.hpp
                                                  (CuTe GEMM helpers)

  host reference / shape-only ──include─▶ xe35_chunk_gated_delta_rule.hpp
                                          (public API: GDNArguments,
                                           get_workspace_sizes — NO device code)
```

| File | Purpose |
|---|---|
| [xe35_chunk_gated_delta_rule.hpp](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule.hpp) | Lightweight public API: `GDNArguments`, `get_workspace_sizes`, `chunk_gated_delta_rule_launch` declaration |
| [xe35_chunk_gated_delta_rule_launch.hpp](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule_launch.hpp) | Header-only launcher: argument validation + `chunk_gated_delta_rule_launch<T, StateT>` definition (inline template, instantiated at each call site) |
| [xe35_chunk_gated_delta_rule_kernels.hpp](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule_kernels.hpp) | Four device kernels + `detail::kernel_launcher` (upstream-aligned signature; the Stage-1 `chunk_prepare` gate is fused into `chunk_compute_A_o2`) |
| [xe35_chunk_gated_delta_rule_gemm.hpp](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule_gemm.hpp) | CuTe GEMM helpers (`gemm_TTS`, `gemm_STS`, `gemm_TSS`, `gemm_TTS_k_multi`, `gemm_TTS_k_multi_tile`, `gemm_TTS_shareB_multi_tile`) |
| [gdn_runner.hpp](../../../applications/gdn_attention/gdn_runner.hpp) | Shared host harness: `GdnRunner` (alloc + init + sigmoid(b) + args + launch + oracles), the `parse_gdn_shape`/`validate_gdn_shape` CLI helpers, and the `ExampleOptions`/`BenchmarkOptions` structs. Used by all three consumers |
| [xe35_gdn_attention_stage_references.hpp](../../../tools/util/include/cutlass/util/reference/host/xe35_gdn_attention_stage_references.hpp) | Per-stage host reference + `apply_sigmoid_b` (in `cutlass/util/reference/host/`), driving the chunkwise oracle |
| [xe35_gdn_attention_recurrent_reference.hpp](../../../tools/util/include/cutlass/util/reference/host/xe35_gdn_attention_recurrent_reference.hpp) | Token-by-token fp32 recurrent reference oracle + `kTolE2E` tolerance |
| [xe35_gdn_attention_compare.hpp](../../../tools/util/include/cutlass/util/reference/host/xe35_gdn_attention_compare.hpp) | Host verification comparator (`compare_with_stats`/`print_compare_stats`, in `cutlass/util/reference/host/`) used by `GdnRunner` |
| [examples/14_xe35_gdn_attention/](../../../examples/14_xe35_gdn_attention) | Example driver + thin `GdnExampleRunner` (CLI + perf timing) over the shared `GdnRunner` |
| [benchmarks/applications/03_gdn/](../../../benchmarks/applications/03_gdn) | Google Benchmark harness + FLOP/byte model over the shared `GdnRunner`, configuration sweep |
| [test/unit/gdn_attention/](../../../test/unit/gdn_attention) | GoogleTest unit coverage (`cutlass_test_unit_gdn_attention_chunkwise`); thin pass gate over the shared `GdnRunner` |
