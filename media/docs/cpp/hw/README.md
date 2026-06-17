# Hardware Reference Docs

Authoritative Intel Xe hardware specifications for **verifying** kernels and
changes — DPAS/systolic shapes, XeCore layout, L2, copy engines, alignment
rules, FP8/FP4 TOPS, power/perf envelopes, and "what is *not* present" limits.

This is a **pointer index**, not vendored content. The specs are large and
versioned, and they live in Artifactory as the single source of truth. Fetch
them on demand into a local cache; do not commit them here.

## Source of truth (Artifactory)

```
gfx-sandbox-ba/rishiyad/arch-docs/<arch>/<version>
```

| Arch | Meaning | Pinned version |
|---|---|---|
| `cri` | Crescent Island — Xe3p v1c, PCIe AI inference GPU | `2026-04-27` |
| `jgs` | JGS variant | `2026-04-27-v4` |
| `nvl` | NVL variant | `2026-04-27` |

Browse (CRI): <https://gfx-assets.iind.intel.com/artifactory/gfx-sandbox-ba/rishiyad/arch-docs/cri/2026-04-27/>

## How to fetch

These docs ship with [gpuforge](https://github.com/intel-sandbox/applications.ai.toolkits.services).
With gpuforge installed:

```bash
# CRI only (or 'all' for cri + jgs + nvl)
~/.gpuforge/tool/helpers/fetch-arch-docs.sh cri
```

- Caches to `~/.gpuforge/docs/<arch>/` (idempotent — re-runs report a cache
  hit; add `--force` to refetch).
- Requires Artifactory credentials in `~/.netrc`:
  ```
  machine gfx-assets.iind.intel.com login <idsid> password <token>
  ```
  (`chmod 600 ~/.netrc`)

Without gpuforge installed, pull the same path directly from the Artifactory
URL above.

## What CRI provides

Start at `~/.gpuforge/docs/cri/cri-INDEX.md` (Tier-1 cheatsheet), then the
topic file that answers your question:

| File | Key content |
|---|---|
| `cri-hw-config.md` | XeCore layout, L2, systolic, config summary |
| `cri-hw-limits.md` | Structures NOT in CRI, fusing, security |
| `cri-copy-engines.md` | 4× paging CEs, MMIO, MCR, resets, latency |
| `cri-power-perf.md` | Fmax, VF curve, PnP, TDP, TOPS targets |
| `cri-workloads.md` | AI workload table, models, TTFT/TPOT, scale |
| `cri-quantization.md` | FP8/FP4/MX, llm-compressor, INC |
| `cri-sw-stack.md` | vLLM, SGLang, SW bounding box, BMG reuse |
| `cri-mlperf.md` | MLPerf targets, accuracy, competitive scope |
| `cri-containers.md` | OOB containers, DoD criteria, Day0 readiness |
| `cri-media.md` | Decode/encode streams, codecs, framework |

Plus `images/` (block diagrams, latency tables, VF curves).

## Architecture neutrality

The same repo (SYCL\*TLA) targets multiple Intel Xe architectures. CRI / Xe3p,
JGS, and NVL are all available via the same helper. When it isn't obvious from
context which architecture a verification targets, **ask** rather than assuming
CRI. Use the repo's existing Xe arch-tag vocabulary
([`../cute/10_intel_overview.md`](../cute/10_intel_overview.md)).

## Related

- Invokable playbook: [`.claude/skills/cutlass-arch-docs/SKILL.md`](../../../../.claude/skills/cutlass-arch-docs/SKILL.md)
- Cross-tool routing index: [`AGENTS.md`](../../../../AGENTS.md)
