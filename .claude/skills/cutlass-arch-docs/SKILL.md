---
name: cutlass-arch-docs
description: Use when verifying a kernel or change against Intel Xe hardware specs, or when you need authoritative hardware numbers for SYCL*TLA work — DPAS/systolic shapes, XeCore layout, L2, copy engines, alignment rules, FP8/FP4 TOPS, power/perf envelopes, and "what is NOT present" limits. Covers Crescent Island (CRI / Xe3p), JGS, and NVL. The docs live in Artifactory and are fetched on demand into a local cache; this skill is pointers, not vendored content.
---

# CUTLASS Architecture Hardware Docs

Authoritative Intel Xe hardware reference docs for **verifying** SYCL\*TLA
kernels and changes against the real silicon — not duplicated in-repo. The
specs live in Artifactory (the single source of truth) and are fetched on
demand into a local cache. Companion to
[`media/docs/cpp/hw/README.md`](../../../media/docs/cpp/hw/README.md), which is
the in-repo index; see [`AGENTS.md`](../../../AGENTS.md) for the cross-tool
routing index.

Use this when a task asks "does this match the HW?" — e.g. confirming a tile
shape against the systolic depth, an alignment against L2 banking, a copy
pattern against the paging copy engines, or a perf claim against the TOPS /
bandwidth envelope.

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

## Fetch the docs

These docs ship with [gpuforge](https://github.com/intel-sandbox/applications.ai.toolkits.services).

```bash
# CRI only — or 'all' for cri + jgs + nvl
~/.gpuforge/tool/helpers/fetch-arch-docs.sh cri
```

- Caches to `~/.gpuforge/docs/<arch>/`. Idempotent — a re-run reports a cache
  hit; pass `--force` to refetch a newer version.
- Override the version with `FETCH_ARCH_DOCS_VERSION=<ver>`.
- Requires Artifactory credentials in `~/.netrc` (never commit these):
  ```
  machine gfx-assets.iind.intel.com login <idsid> password <token>
  ```
  `chmod 600 ~/.netrc`

**No gpuforge?** Pull the same `<arch>/<version>` path directly from the
Artifactory URL above.

## Read the right file (CRI)

Start with the Tier-1 cheatsheet, then drill into the topic file:

```bash
cat ~/.gpuforge/docs/cri/cri-INDEX.md
```

| Question | File |
|---|---|
| XeCore layout, L2, systolic depth, config summary | `cri-hw-config.md` |
| What structures are NOT in CRI, fusing, security | `cri-hw-limits.md` |
| Copy engines (4× paging CEs), MMIO, MCR, resets, latency | `cri-copy-engines.md` |
| Fmax, VF curve, PnP, TDP, FP8/FP4 TOPS targets | `cri-power-perf.md` |
| AI workload table, models, TTFT/TPOT, scale | `cri-workloads.md` |
| FP8/FP4/MX quantization, llm-compressor, INC | `cri-quantization.md` |
| vLLM, SGLang, SW bounding box, BMG reuse | `cri-sw-stack.md` |
| MLPerf targets, accuracy, competitive scope | `cri-mlperf.md` |
| OOB containers, DoD criteria, Day0 readiness | `cri-containers.md` |
| Decode/encode streams, codecs, media framework | `cri-media.md` |

Diagrams and latency/VF tables are under `~/.gpuforge/docs/cri/images/`.

## CRI quick reference

```
Product:      Crescent Island (CRI) — Xe3p v1c XPC, PCIe AI inference GPU
XeCores:      32 (1 XeCu × 4 slices × 8 cores/slice)
Systolic:     16-deep, FP8/FP4 support
L2 Cache:     32 MB (4 slices, 2 nodes/slice, 4 banks/node, 1MB/bank)
Copy Engines: 4× paging CEs (no main copy), PCIe Gen5 (256 GB/s)
Fmax:         2.4 GHz
TDP:          300 W
FP8 TOPS:     915 @ 300W (1250 peak)
FP4 TOPS:     1805 @ 300W (2500 peak)
Memory BW:    1.37 TB/s per direction @ 2 GHz
NOT present:  3D structures, encryption (PXP), lossless compression, KCRS
```

Always confirm against the fetched docs — this cheatsheet is a starting point,
not the citation.

## Conventions

- **Pick the right architecture.** SYCL\*TLA targets multiple Intel Xe arches.
  If the target (CRI / Xe3p, JGS, NVL, mainline) isn't obvious from context or
  the working directory, **ask the user** before fetching — verifying against
  the wrong arch gives wrong answers. Use the repo's Xe arch-tag vocabulary
  (`media/docs/cpp/cute/10_intel_overview.md`).
- **Never commit fetched docs or credentials.** The cache lives under
  `~/.gpuforge/`, outside the repo. `~/.netrc` is workstation-local.
- **Cite the version.** When you state a HW number in a PR, review, or ticket,
  name the arch + version you read it from (e.g. "CRI `2026-04-27`,
  `cri-power-perf.md`") so the claim stays traceable.
- **AI-assistance disclaimer.** When this skill produces a user-facing artifact
  (PR comment, Jira/Confluence note), append a one-line
  "_created with AI assistance_" footer.
