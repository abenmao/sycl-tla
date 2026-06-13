# XE4 Stream-K Tile Scheduler — Design Document

## 1. Motivation

Data-parallel (DP) GEMM assigns one output tile per CTA. When the number of output tiles is not a multiple of the GPU's SM count, some SMs sit idle during the final "wave" — the **tail effect**. For tall-skinny or irregular shapes this can waste 30-50% of compute.

Stream-K eliminates tail effects by splitting the K-reduction dimension across CTAs. Instead of each CTA owning an entire output tile, a CTA owns a *range of K-tiles* and may contribute to one or more output tiles. Multiple CTAs collaborate on the same output tile, each reducing a partial accumulator into the final result.

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Host (gemm_streamk.hpp)                      │
│  1. Compute decomposition (how many SK units, which tiles)      │
│  2. Allocate workspace: int32[num_sk_tiles] atomic counters     │
│  3. Build scheduler params, launch persistent grid              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│            PersistentTileSchedulerXe4StreamK (device)           │
│                                                                 │
│  Wraps:                                                         │
│    • XE4 CLC-based dynamic scheduler (grid iteration; the       │
│      single-kernel persistent model mirrors SM100's hardware    │
│      Cluster Launch Control — see §10)                          │
│    • SM90 StreamK decomposition logic (K-tile partitioning)     │
│                                                                 │
│  Produces WorkTileInfo { M_idx, N_idx, K_idx, L_idx,            │
│                          k_tile_count }                         │
│                                                                 │
│  Key methods:                                                   │
│    initial_work_tile_info()    — first tile for this CTA        │
│    fetch_next_work()           — next tile (may continue in SK) │
│    get_k_tile_iterator()       — K-tile range for mainloop      │
│    get_sk_tile_counter_ptr()   — atomic counter array           │
│    get_tile_idx()              — linear index for counter lookup│
│    requires_fixup()            — is this a split tile?          │
│    compute_epilogue()          — is this the final split?       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
   │ Non-final   │    │ Final split  │    │ Data-parallel│
   │ SK split    │    │ (SK)         │    │ (DP tile)    │
   │             │    │              │    │              │
   │ MMA partial │    │ MMA partial  │    │ MMA full K   │
   │ Fred→gmem D │    │ Fred→gmem D  │    │ Full epilogue│
   │ Counter++   │    │ Counter++    │    │ Store D      │
   │ No epilogue │    │ Poll counter │    │              │
   │             │    │ G2S load-back│    │              │
   │             │    │ Full epilogue│    │              │
   └─────────────┘    └──────────────┘    └──────────────┘
```

## 3. Decomposition Modes

The scheduler supports three decomposition modes (selected at host):

| Mode | Behavior |
|------|----------|
| `StreamK` | All output tiles processed via Stream-K decomposition. Every tile may be split across multiple CTAs. |
| `SplitK` | Each output tile split into `splits` equal pieces along K. Grid is `tiles_mn * splits` CTAs. |
| `Heuristic` | Auto-selects: uses Stream-K for the first wave's worth of tiles (eliminating the tail), data-parallel for the rest. |

### Split-K as a special case

Split-K reuses the same workspace, counter, and fred mechanism as Stream-K. The difference is purely in how work is assigned:
- Stream-K: arbitrary K-ranges, variable per CTA
- Split-K: uniform K-ranges, `ceil(K_tiles / splits)` per split

Both use the same non-final/final dispatch in the epilogue.

## 4. Scheduler Internals

### 4.1 WorkTileInfo

```cpp
struct WorkTileInfo {
  int32_t M_idx;         // Output tile row
  int32_t N_idx;         // Output tile column
  int32_t K_idx;         // Starting K-tile index (0 for DP)
  int32_t L_idx;         // Batch index
  int32_t k_tile_count;  // K-tiles assigned to this unit
};
```

For DP tiles: `K_idx = 0`, `k_tile_count = total_k_tiles`.
For SK splits: `K_idx > 0` possible, `k_tile_count < total_k_tiles`.

### 4.2 convert_work() — the core dispatch

When the XE4 CLC scheduler produces a grid coordinate, `convert_work()` translates it based on decomposition state:

```
Grid coordinate (M_idx, N_idx, wave_idx)
         │
         ▼
┌─ has_sk_work() ─────────────────────────────────────────┐
│  Convert to linear CTA index:                           │
│    linear_idx = wave_idx * sm_count + cluster_idx       │
│  Feed into SM90 StreamK decomposition:                  │
│    get_current_work_for_linear_idx(linear_idx)          │
│  Extract: M/N from output tile, K_idx, k_tile_count     │
└─────────────────────────────────────────────────────────┘
         │
         ▼ (if linear_idx >= sk_units)
┌─ is_split_k() ──────────────────────────────────────────┐
│  Split L_idx into (batch, split_idx):                   │
│    batch = L_idx / splits                               │
│    split_idx = L_idx % splits                           │
│  Compute K range:                                       │
│    k_tiles_per_split = total_k_tiles / splits           │
│    K_idx = split_idx * k_tiles_per_split                │
│    k_tile_count = k_tiles_per_split (+ remainder)       │
└─────────────────────────────────────────────────────────┘
         │
         ▼ (if pure DP)
┌─ is_dp_only() ──────────────────────────────────────────┐
│  K_idx = 0, k_tile_count = total_k_tiles                │
│  M/N/L from grid coordinate                             │
└─────────────────────────────────────────────────────────┘
```

### 4.3 CLC continuation (multi-tile SK units)

A single SK unit may span multiple output tiles (when `k_tiles_per_unit > k_tiles_per_output_tile`). The scheduler handles this transparently:

```cpp
auto fetch_next_work(work_tile_info, clc_pipeline, clc_state) {
  if (continue_current_work(work_tile_info)) {
    // Same SK unit, next output tile — no CLC query consumed
    return {work_tile_info, false};  // false = don't increment CLC
  }
  // Done with this unit, get next CLC result
  auto next = xe4_scheduler_.fetch_next_work(...);
  return {convert_work(next), true};  // true = consumed CLC slot
}
```

This is critical: the MainloopLoad, Scheduler, and MMA warps all call `fetch_next_work()`. When `increment_pipe=false`, they skip the CLC pipeline advance, keeping all warps synchronized on the same logical work boundary.

### 4.4 Tile counter addressing

Each SK output tile has one `int32` atomic counter. The counter tracks how many K-tiles have been reduced into gmem D for that tile.

```cpp
static int* get_sk_tile_counter_ptr(Params const& params) {
  return reinterpret_cast<int*>(params.sk_params_.reduction_workspace_);
}

static int get_tile_idx(Params const& params, WorkTileInfo const& work_tile_info) {
  // Linearize (M_idx, N_idx, L_idx) accounting for swizzle and raster order
  auto tiles_mn = problem_blocks.x * problem_blocks.y;
  return tiles_mn * L_idx + linear_tile_in_batch;
}
```

### 4.5 Small-split prevention

The scheduler enforces `min_iters_per_sk_unit` (typically 8). If a split boundary would create a fragment smaller than this, it merges the fragment with the adjacent unit. This prevents CTAs from doing trivially small work that doesn't amortize launch overhead.

## 5. Kernel Integration

### 5.1 Warp roles

The XE4 warp-specialized GEMM has five roles:

| Warp | Role | SK-specific behavior |
|------|------|----------------------|
| MMA | Matrix multiply-accumulate | Processes `k_tile_count` tiles (partial for SK); acquires the EpiStore stage only when `writes_d_output` (§12) |
| MainloopLoad | Load A/B tiles from gmem | Uses `get_k_tile_iterator()` for K-range |
| Scheduler | Issue CLC queries | Handles continuation (`increment_pipe=false`) |
| EpilogueLoad | Load C, coordinate SK reduction | **Primary SK dispatch point** |
| Epilogue | Compute + store D | Non-final: signal only. Final/DP: full store. |

### 5.2 EpilogueLoad SK dispatch

```cpp
// In kernel operator(), EpiLoad warp:
if constexpr (IsStreamK && CollectiveEpilogue::HasSkReduce) {
  bool is_sk_split = TileScheduler::requires_fixup(params.scheduler, work_tile_info);
  if (is_sk_split) {
    bool is_sk_final = TileScheduler::compute_epilogue(work_tile_info, params.scheduler);
    int* counter = TileScheduler::get_sk_tile_counter_ptr(params.scheduler);
    int  tile_idx = TileScheduler::get_tile_idx(params.scheduler, work_tile_info);

    // Template instantiation: exactly one of IsSkNonFinal/IsSkFinal is true
    auto result = is_sk_final
      ? epilogue.load<false, false, true>(...)   // final split
      : epilogue.load<false, true, false>(...);  // non-final split
  }
}
```

The template dispatch ensures dead code elimination: non-final path never compiles C-load or store logic; final path never compiles fred-only logic.

### 5.3 Backward compatibility

The kernel's SK code is guarded by:
```cpp
if constexpr (IsStreamK && CollectiveEpilogue::HasSkReduce)
```

When using `xe4_static_tile_scheduler` or `xe4_tile_scheduler`:
- `IsStreamK = false` (scheduler tag is not `StreamKScheduler`)
- Entire SK block is eliminated at compile time
- `work_tile_info` is the simpler struct without `K_idx`/`k_tile_count`
- Epilogue `load()` is called with default `WorkTileInfo_ = cute::tuple<>{}` — the SK extraction code inside epilogue is also dead-code-eliminated

## 6. Epilogue Fred Reduction

### 6.1 The problem

Multiple CTAs produce partial accumulators for the same output tile. These must be combined before the final epilogue (which applies activation, fuses C, stores D).

### 6.2 The solution: atomic fred + counter + load-back

```
CTA 0 (non-final)         CTA 1 (non-final)         CTA 2 (final)
─────────────────         ─────────────────         ─────────────────
MMA: partial acc          MMA: partial acc          MMA: partial acc
     │                         │                         │
     ▼                         ▼                         ▼
Fred: smem_Imm ──atomic──► gmem D workspace    Fred: smem_Imm ──atomic──► gmem D
Counter += k_tile_count    Counter += k_tile_count    Counter += k_tile_count
     │                         │                         │
     ▼                         ▼                         ▼
  (done)                    (done)               Poll: counter >= K_idx?
                                                        │ yes
                                                        ▼
                                                 G2S: gmem D → smem_Imm
                                                        │
                                                        ▼
                                                 Full epilogue:
                                                   smem_Imm + C → D (final store)
```

### 6.3 Fred operation (`streamk_fred_to_gmem_d`)

Uses `XE4_ADMA_STORE_REDUCE<ElementImm, RedOp::Add>` — an async DMA copy atom that performs atomic floating-point addition from shared memory to global memory.

```cpp
void streamk_fred_to_gmem_d(fred_pipeline, fred_state, ...) {
  fred_pipeline.consumer_try_wait(fred_state);    // Wait for smem_Imm ready
  copy(params.adma_store_reduce.with(abar),       // Atomic add: smem -> gmem
       bSG_sImm(...), bSG_gD(...));
  fred_pipeline.consumer_commit(fred_state, TransactionBytesImm);
}
```

This runs on the **EpiLoad** warp and consumes the dedicated `FredPipeline` (§8) — the Epilogue
warps signal `smem_Imm` ready via `fred_pipeline.producer_commit`. It is intentionally separate
from the EpiStore pipeline, which carries only the `smem_D → gmem D` store (§12).

The atomic add is hardware-accelerated on XE4 — it's not a CAS loop.

### 6.4 Counter protocol

```cpp
// Non-final and final splits both arrive:
void streamk_arrive_counter(int* counter, int tile_idx, int k_tile_count, ...) {
  atomic_ref<int, memory_order::relaxed, memory_scope::device> ref(counter[tile_idx]);
  ref += k_tile_count;
}

// Final split polls:
void streamk_wait_for_previous_splits(int* counter, int tile_idx, int K_idx) {
  atomic_ref<int, memory_order::relaxed, memory_scope::device> ref(counter[tile_idx]);
  while (ref.load() < K_idx) { /* spin */ }
}
```

The counter accumulates the *total number of K-tiles* that have been fred'd into gmem D. The final split knows its own `K_idx` (starting K-tile offset), so it waits until the counter reaches that value — meaning all prior splits have completed their freds.

**Why relaxed ordering is sufficient**: The fred operation (`async_tensor_fred`) provides its own memory ordering guarantee — once the DMA completes, the data is globally visible. The counter increment happens *after* fred completion (via pipeline ordering). The final split's poll guarantees it sees the counter only after all freds land.

### 6.5 G2S load-back (`streamk_load_gmem_d_to_smem`)

Once the counter confirms all freds are done, the final split loads the accumulated result back into shared memory:

```cpp
void streamk_load_gmem_d_to_smem(g2s_pipeline, g2s_state, ...) {
  g2s_pipeline.producer_acquire(g2s_state);
  auto tma_barrier = g2s_pipeline.producer_get_barrier(g2s_state);
  copy(params.adma_load_d.with(tma_barrier), gD(...), sImm(...));
  g2s_pipeline.producer_expect_transaction(g2s_state);
  g2s_pipeline.producer_commit(g2s_state);
}
```

After this, `smem_Imm` contains the fully-reduced accumulator for all K-tiles. The epilogue then proceeds normally: applies activation, fuses with C, stores final D.

## 7. Workspace Layout

```
┌──────────────────────────────────────────────────────────────┐
│  int32 counters[num_sk_tiles]                                │
│  (one per output tile that participates in Stream-K)         │
│                                                              │
│  Size = tiles_m * tiles_n * tiles_l * sizeof(int32_t)        │
│  Typical: 128 bytes for a 4x4x2 tile grid                    │
│                                                              │
│  Initialized to 0 before kernel launch (host memset)         │
└──────────────────────────────────────────────────────────────┘
```

No accumulator scratch. No reduction buffer. The gmem D output buffer itself serves as the reduction target via atomic fred.
## 8. Pipeline Interactions

```
Pipeline             Producer          Consumer         Purpose
─────────────────────────────────────────────────────────────────
MainloopPipeline     MainloopLoad      MMA              A/B tile delivery
AccumulatorPipeline  MMA               Epilogue         smem_Imm ready signal
EpiStorePipeline     Epilogue + MMA*   EpiLoad          smem_D -> gmem D store
FredPipeline         Epilogue          EpiLoad          smem_Imm -> gmem D reduce (fred)
EpiLoadPipeline      EpiLoad           Epilogue         C tile delivery
G2SPipeline          EpiLoad           Epilogue         SK final: gmem D → smem_Imm
CLCPipeline          Scheduler         All loads        Work tile coordination
```

`*` The EpiStore pipeline has a **split producer**: the Epilogue warps do the `producer_commit`
(smem_D filled), while the MMA warp does the `producer_acquire` (the empty-barrier wait that
gates smem_D reuse). MMA places the acquire before its final MMA so the wait latency overlaps
matmul compute, and the epilogue never stalls on smem_D availability. This split is a
deliberate latency-hiding technique; §12 explains the invariant it imposes.

### Why the fred has its own pipeline

The fred (`smem_Imm → gmem D` atomic reduce) and the D-store (`smem_D → gmem D`) are two
*distinct* EpiLoad gmem events that must not share one pipeline:

- A **non-final SK split** produces no epilogue output and no `smem_D` — it only freds.
- A **final SK split** does **both**: it freds its own partial contribution, then (after the
  counter poll + G2S load-back) runs the full epilogue and D-stores.
- A **DP tile** only D-stores.

If the fred rode the EpiStore pipeline, a final split would commit it **twice** (fred + D-store)
while a non-final committed once and DP once — a non-uniform per-tile count. Because MMA is the
split-producer's acquirer (see §12), MMA would then have to mirror that 1-or-2 count, which is
fragile. Giving the fred its **own** pipeline makes the EpiStore pipeline carry exactly the
`smem_D` handoff, so it is touched by exactly the tiles that write D output
(`compute_epilogue == true`: final SK + DP), exactly once each.

Per-tile pipeline activity after this split:

| tile type      | EpiStore (acquire/commit/consume) | FredPipeline (commit/consume) |
|----------------|-----------------------------------|-------------------------------|
| non-final SK   | 0 / 0 / 0                         | 1 / 1                         |
| final SK       | 1 / 1 / 1                         | 1 / 1                         |
| DP             | 1 / 1 / 1                         | 0 / 0                         |

For non-final SK splits the Epilogue warp does no `store()` compute and never touches the
EpiStore pipeline; it only signals the FredPipeline (smem_Imm ready), EpiLoad performs the fred for that tile.

## 9. Host-Side Setup

```cpp
// 1. Configure scheduler
TileSchedulerArguments scheduler_args{
  .splits = 1,                           // or >1 for split-K
  .max_swizzle_size = 1,
  .raster_order = RasterOrderOptions::AlongN,
  .reduction_mode = ReductionMode::Deterministic,
  .decomposition_mode = DecompositionMode::StreamK
};

// 2. Compute workspace size (just counters)
size_t workspace_size = TileScheduler::get_workspace_size<ProblemShape, ElementAcc>(
    scheduler_args, problem_shape_mnkl, kernel_hw_info);

// 3. Allocate and zero
void* workspace = sycl::malloc_shared<uint8_t>(workspace_size, queue);
std::memset(workspace, 0, workspace_size);

// 4. Build kernel params (workspace pointer stored in scheduler params)
auto params = kernel.to_underlying_arguments(args, kernel_hw_info, workspace);

// 5. Launch persistent grid
dim3 grid = GemmKernel::get_grid_shape(params);  // Sized by SM occupancy, not tile count
launch_kernel_on_cluster(launch_params, kernel, params);
```

The grid shape for Stream-K is determined by hardware occupancy (how many CTAs can be resident), not by the number of output tiles. This is what enables the tail-effect elimination.

## 10. Comparison with SM100 Stream-K

The XE4 scheduler borrows SM90's **decomposition math** (it literally wraps
`PersistentTileSchedulerSm90StreamKParams` for K-tile partitioning), but its **runtime shape**
is modeled on SM100: a single persistent kernel driven by hardware-assisted dynamic work
distribution. The table below compares XE4 against SM100.

| Aspect | SM100 (Blackwell) | XE4 (SYCL) |
|--------|-------------------|-------------|
| Grid scheduling | **Hardware CLC** — Cluster Launch Control issues next work tile via `PipelineCLCFetchAsync` + a CLC-throttle pipeline | **CLC-based dynamic persistent** — software CLC scheduler (`PersistentTileSchedulerXe4`) feeding the SM90 decomposition |
| Work-tile delivery | Async CLC response consumed through a dedicated pipeline stage | Async CLC response in `shared_tensors.clc_response`, consumed by `fetch_next_work()` |
| Multi-tile continuation | CLC response re-used in place (no new query for in-progress SK unit) | `fetch_next_work()` returns `increment_pipe=false` to reuse the work tile without consuming a CLC slot |
| Reduction target | Separate accumulator/reduction workspace; accumulators live in **TMEM** before fixup | **gmem D buffer directly** via atomic fred (no scratch accumulator) |
| Reduction mechanism | `scheduler.fixup()` — barrier-gated reduction over TMEM accumulators, upstream of the epilogue | Hardware atomic float-add DMA (`XE4_ADMA_STORE_REDUCE`) + atomic tile counter |
| Reduction workspace size | `num_tiles * tile_M * tile_N * sizeof(acc)` + barrier workspace | `num_tiles * sizeof(int32)` (counters only) |
| Kernel launches | 1 (fixup folded into the persistent kernel) | 1 (fred + load-back folded into the persistent kernel) |
| Cross-split sync | Named-barrier–gated `fixup()` over peer splits | Atomic counter poll (`streamk_wait_for_previous_splits`) in the same kernel |
| Non-final split handling | **Skips load *and* store together** (`compute_epilogue==false`); fixup reduces TMEM upstream | Skips epilogue *output* and the EpiStore pipeline; EpiLoad freds `smem_Imm → gmem D` via the dedicated FredPipeline |

## 12. The EpiStore Split-Producer Invariant

The EpiStore pipeline (depth `StagesD = 1`) guards reuse of the CTA-tile-sized `smem_D`
buffer. Its producer protocol is **split across two warp roles**:

- The **Epilogue** warps call `producer_commit` once they have filled `smem_D` (R2S of the
  final result).
- The **MMA** warp calls `producer_acquire` — the empty-barrier wait that blocks until EpiLoad
  has finished draining the *previous* tile's `smem_D` to gmem, so the buffer is safe to
  overwrite.

MMA owns the acquire (rather than the epilogue) on purpose: it issues the acquire *before* its
final `cute::gemm`, so the empty-barrier latency overlaps matmul compute and the epilogue never
stalls waiting on `smem_D`. This is the same latency-hiding rationale behind keeping the fred on
the EpiLoad warp.

### 12.1 The invariant

A split producer only works if both halves advance the pipeline state the **same number of
times per tile**. The empty-barrier on a depth-1 pipeline is a single parity bit; if MMA
acquires N times while the epilogue commits M times with `N != M`, the producer and consumer
parities diverge and the next `try_wait` blocks forever.

> **Invariant:** MMA acquires (and `++store_pipe_producer_state`) **iff** the Epilogue commits
> (and `++store_pipe_producer_state`) — namely, exactly on tiles that write D output.

### 12.2 How the invariant is enforced

The predicate `TileScheduler::compute_epilogue(work_tile, params)` — "does this tile write D
output" — is the **single source of truth** that both warps key off:

- **MMA** (`xe4_mma_warpspecialized.hpp`): `mma()` takes a `bool writes_d_output`. The kernel
  computes it as `compute_epilogue(work_tile_info, …)` and passes it in. MMA does
  `producer_try_acquire` and the kernel does `++epi_store_pipe_producer_state` **only when
  `writes_d_output`**.
- **Epilogue** (`xe4_epilogue_adma_warpspecialized.hpp`): the non-final `store()` branch
  (`compute_epilogue == false`) never touches the store pipeline (§11.3); the final-SK and DP
  branches (`compute_epilogue == true`) `producer_commit` exactly once.

Because both sides gate on the *same predicate*, the counts are equal by construction — no
commit-count mirroring, no per-tile arithmetic to keep in sync.

### 12.3 What this replaced

The earlier working-but-fragile fix had the non-final `store()` commit the store pipeline once
(to keep EpiLoad's fred unblocked) and compensated by advancing the **accumulator** producer
state twice (`+2`) on the MMA side so the divergent store count happened to re-align. That
`+2` was opaque: it coupled two unrelated pipelines and silently encoded "a final SK split
commits the store pipeline twice." Splitting the fred onto its own pipeline (§8) removed the
double-commit at its source, so the store pipeline now carries exactly one event per D-output
tile and the accumulator pipeline advances uniformly (`+1`) for every tile. The invariant above
is what remains, and it is enforced by one shared predicate rather than a numeric fudge.
