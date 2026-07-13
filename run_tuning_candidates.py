#!/usr/bin/env python3
"""Build & benchmark projection-shortlisted FA tile candidates on the device.

This is the real-HW side of the projection -> tuning loop. It reads the JSON written by
the projection model's export_tuning_candidates.py (gpu_projection .../OpPerf), then for
each shortlisted tile config it builds the matching env-guarded sweep target and runs it
on the attention problem, parsing the kernel's "Performance:" line. Results (projected
vs measured) are written to CSV.

It only builds the SPECIFIC candidate targets the projection selected -- not the full
tile cartesian product -- which is the whole point of using the projection as a filter.

Guarded by design: nothing here runs unless you invoke this script explicitly, and the
tile-sweep CMake targets only exist when the build is configured with -DFMHA_TUNE=ON
(this script does that configure for you). Normal example builds are unaffected.

Usage:
    python3 run_tuning_candidates.py --candidates tuning_candidates.json --build-dir build
    # optional: --dry-run (configure + print plan, no build/run), --iterations N
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

EXAMPLE_SUBDIR = "examples/06_bmg_flash_attention"
PERF_RE = re.compile(
    r"Performance:\s+([\d.]+)\s+GB/s,\s+([\d.]+)\s+TFlop/s,\s+([\d.]+)\s+ms")

# IGC / GRF tuning environment for BMG FA kernels. These MUST match the reference
# tuning regime (examples/06_bmg_flash_attention/run.sh): the FA kernels are
# register-pressure bound and the projection assumes LARGE-GRF (256 reg) mode, so
# without these the AOT-compiled kernels run in a different (slower) codegen regime
# than both the reference and the projection assume. Applied at BUILD time (AOT
# codegen reads IGC_*) AND run time. Measured impact: hd72 256x32x16 goes 52->56 TF.
IGC_ENV = {
    "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file",
    "IGC_VectorAliasBBThreshold": "10000",
    "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
    "IGC_TotalGRFNum": "256",
    "IGC_EnableVISAPreSched": "1",
    "IGC_SetVISAOption":
        "-vISA_LocalBankConflictReduction 1 -vISA_EnableGroupScheduleForBC 1",
}


def build_env(use_igc):
    env = os.environ.copy()
    if use_igc:
        env.update(IGC_ENV)
    return env


def prefill_target_name(head_dim, c, variant="sb_fs"):
    # ENABLE_SPLIT_BOUNDARY + ENABLE_FUSED_SOFTMAX are IMPLEMENTATION optimizations the
    # projection ASSUMES are beneficial ("ideal impl") -- but empirically they are NOT
    # always: e.g. hd72 256x32x16 measures 56.1 TF (base) vs 52.4 TF (_sb_fs), a -7%
    # REGRESSION. Since this is a codegen detail the projection cannot predict, the tuner
    # measures BOTH variants and lets the hardware decide.
    #   variant "sb_fs" -> _sb_fs target (both flags on)
    #   variant "base"  -> plain target  (neither flag)
    suffix = "_sb_fs" if variant == "sb_fs" else ""
    return (f"06_xe_fmha_fwd_prefill_bfloat16_t_hdim{head_dim}"
            f"_q{c['tile_q']}_seq{c['tile_seq_len']}"
            f"_hdim{c['tile_hdim']}_qsg{c['tile_q_sg']}{suffix}")


def decode_target_name(head_dim, c):
    suffix = f"kv{c['tile_kv']}_kvsg{c['tile_kv_sg']}"
    if c.get("persistent"):
        return f"06_xe_fmha_fwd_decode_persistent_bfloat16_t_hdim{head_dim}_{suffix}"
    else:
        return f"06_xe_fmha_fwd_decode_bfloat16_t_hdim{head_dim}_{suffix}"


def asym_target_name(c):
    # Matches CMakeLists.txt's FMHA_TUNE_ASYM_* sweep naming (ASYM_TARGET string).
    return (f"06_xe_fmha_fwd_prefill_bfloat16_t_hdim192_vo128"
            f"_kv{c['tile_kv']}_q{c['tile_q']}_d{c['tile_d_step']}"
            f"_v{c['tile_v_step']}_sg{c['tile_sg']}")


def target_name(problem, c, variant="sb_fs"):
    if c["mode"] == "prefill_asym":
        return asym_target_name(c)
    hd = c["kernel_head_dim"]
    if c["mode"] == "decode":
        return decode_target_name(hd, c)
    return prefill_target_name(hd, c, variant)


def run(cmd, env=None, **kw):
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, text=True, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)


def configure(build_dir, data):
    """Configure cmake with FMHA_TUNE=ON, scoped to the union of tile values used."""
    # Prefill dims
    pf_heads, tq, tseq, thdim, tqsg = set(), set(), set(), set(), set()
    need_split_boundary = False
    need_fused_softmax  = False
    # Decode dims
    dec_heads, tkv, tkvsg = set(), set(), set()
    # Asym (head_size_qk != head_size_vo) candidates. No -D overrides collected here:
    # sdpa.py's fa_asym_*_search lists are kept in sync with CMakeLists.txt's
    # FMHA_TUNE_ASYM_* defaults (see sdpa.py's comment above those lists), so the
    # default CMake ranges already cover every value the projection can shortlist --
    # unlike prefill/decode above, whose sdpa.py search lists are deliberately BROADER
    # than the CMake defaults and so must be threaded through via -D.
    asym_present = False

    for p in data["problems"]:
        for c in p["candidates"]:
            if c["mode"] == "prefill_asym":
                asym_present = True
                continue
            hd = c["kernel_head_dim"]
            if c["mode"] == "prefill":
                pf_heads.add(hd)
                tq.add(c["tile_q"]); tseq.add(c["tile_seq_len"])
                thdim.add(c["tile_hdim"]); tqsg.add(c["tile_q_sg"])
                # split_boundary and fused_softmax are always-on for prefill
                need_split_boundary = True
                need_fused_softmax  = True
            else:
                dec_heads.add(hd)
                tkv.add(c["tile_kv"]); tkvsg.add(c["tile_kv_sg"])

    if not pf_heads and not dec_heads and not asym_present:
        print("No candidates to build.")
        return False

    def lst(s):
        return ";".join(str(x) for x in sorted(s))

    parts = [f"cmake {build_dir} -DFMHA_TUNE=ON"]
    if pf_heads:
        parts += [f'-DFMHA_TUNE_HEAD_DIMS="{lst(pf_heads)}"',
                  f'-DFMHA_TUNE_TILE_Q="{lst(tq)}"',
                  f'-DFMHA_TUNE_TILE_SEQ_LEN="{lst(tseq)}"',
                  f'-DFMHA_TUNE_TILE_HDIM="{lst(thdim)}"',
                  f'-DFMHA_TUNE_TILE_Q_SG="{lst(tqsg)}"']
        # kernel variant flags: generate variant targets when any candidate needs them
        if need_split_boundary:
            parts.append('-DFMHA_TUNE_SPLIT_BOUNDARY=ON')
        if need_fused_softmax:
            parts.append('-DFMHA_TUNE_FUSED_SOFTMAX=ON')
    if dec_heads:
        parts += [f'-DFMHA_TUNE_DECODE_HEAD_DIMS="{lst(dec_heads)}"',
                  f'-DFMHA_TUNE_TILE_KV="{lst(tkv)}"',
                  f'-DFMHA_TUNE_TILE_KV_SG="{lst(tkvsg)}"']

    r = run(" ".join(parts))
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print("ERROR: cmake configure failed.")
        return False
    return True


def bench_args(problem, iterations):
    causal = "--is_causal" if problem.get("causal") else ""
    varlen = "--varlen" if problem.get("varlen") else ""
    length = problem["length"]
    kv = problem.get("context", 0) + length
    if problem.get("mode") == "prefill_asym":
        # Asym problems carry independent QK/VO widths (e.g. DeepSeek MLA 192/128) --
        # must NOT collapse to the same value (that was a bug: the symmetric branch
        # below passes problem["head_dim"] for both, which is correct only when the
        # two widths are equal).
        head_qk = problem["head_dim_qk"]
        head_vo = problem["head_dim_vo"]
    else:
        # Pass the LOGICAL head_dim as the runtime head size. The binary is compiled with
        # HEAD_DIM = kernel_head_dim (pad32 of the logical head, e.g. 72 -> 96); the runner
        # pads the true head size up to that kernel shape but reports FLOPs on the true head.
        # Passing the logical head_dim here makes the benchmark measure the SAME logical
        # problem the projection costed (critical for ragged heads like 72; a no-op for clean
        # heads where head_dim == kernel_head_dim). Without it the runner defaults head_size
        # to HEAD_DIM (the padded 96), measuring a different problem than projected.
        head_qk = head_vo = problem["head_dim"]
    return (f"--iterations={iterations} --batch=1 --verify=0 "
            f"--num_heads_q={problem['q_heads_per_rank']} "
            f"--num_heads_kv={problem['kv_heads_per_rank']} "
            f"--head_size_qk={head_qk} --head_size_vo={head_vo} "
            f"--seq_len_qo={length} --seq_len_kv={kv} {causal} {varlen}").strip()


def candidate_summary(c):
    if c["mode"] == "prefill":
        return (f"prefill tile q{c['tile_q']} seq{c['tile_seq_len']} "
                f"hdim{c['tile_hdim']} qsg{c['tile_q_sg']}  "
                f"(proj {c['proj_time_us']}us / MFU {c['proj_mfu'] * 100:.1f}%)")
    elif c["mode"] == "prefill_asym":
        return (f"asym    tile q{c['tile_q']} kv{c['tile_kv']} sg{c['tile_sg']} "
                f"d{c['tile_d_step']} v{c['tile_v_step']}  "
                f"(proj {c['proj_time_us']}us / MFU {c['proj_mfu'] * 100:.1f}%)")
    else:
        return (f"decode  tile kv{c['tile_kv']} kvsg{c['tile_kv_sg']} "
                f"{'persistent' if c.get('persistent') else 'non-persistent'}  "
                f"(proj {c['proj_time_us']}us / MBU {c['proj_mbu'] * 100:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default="tuning_candidates.json",
                    help="candidate JSON from export_tuning_candidates.py")
    ap.add_argument("--build-dir", default="build",
                    help="configured cmake build directory (default: build)")
    ap.add_argument("--iterations", type=int, default=100,
                    help="benchmark iterations passed to the kernel (default 100)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(),
                    help="parallel compile jobs (targets built concurrently via ninja)")
    ap.add_argument("--output", default="tuning_results.csv",
                    help="CSV output path")
    ap.add_argument("--no-igc-env", action="store_true",
                    help="do NOT set the BMG large-GRF / VISA IGC env (default: set it, "
                         "matching examples/.../run.sh)")
    ap.add_argument("--prefill-variant", choices=["auto", "both", "sb_fs", "base"], default="auto",
                    help="which prefill codegen variant(s) to measure: 'auto' (default) "
                         "follows each candidate's split_boundary/fused_softmax flag from "
                         "the export -- sb_fs+base for reference-validated heads (64,192), "
                         "base-only for heads where the reference build doesn't enable "
                         "these (72/96/128; measured: sb_fs regresses hd128's optimum by "
                         "~1%%, and hd72's by ~7%%); 'both' forces both variants for every "
                         "candidate regardless (exploration); 'sb_fs'/'base' pins one for all")
    ap.add_argument("--ze-affinity", default=None,
                    help="ZE_AFFINITY_MASK value to pin the GPU tile for the benchmark run "
                         "(e.g. 3). Default: unset (device 0).")
    ap.add_argument("--dry-run", action="store_true",
                    help="configure and print the build/run plan only")
    args = ap.parse_args()

    if not os.path.isfile(args.candidates):
        sys.exit(f"candidate file not found: {args.candidates}")
    if not os.path.isfile(os.path.join(args.build_dir, "CMakeCache.txt")):
        sys.exit(f"{args.build_dir} is not a configured cmake build dir; run cmake first.")

    with open(args.candidates) as f:
        data = json.load(f)
    print(f"Loaded {args.candidates} (platform={data.get('platform')}, "
          f"{len(data['problems'])} problems)")

    use_igc = not args.no_igc_env
    env = build_env(use_igc)
    if use_igc:
        print("IGC env: large-GRF (256 reg) + VISA scheduling ON "
              "(match run.sh; note the FMHA kernel already declares grf_size<256>, so "
              "IGC_TotalGRFNum is belt-and-suspenders). Disable with --no-igc-env.")

    if not configure(build_dir=args.build_dir, data=data):
        sys.exit(1)

    bin_dir = os.path.join(args.build_dir, EXAMPLE_SUBDIR)

    # Prefill variants to measure. sb_fs (ENABLE_SPLIT_BOUNDARY+FUSED_SOFTMAX) is NOT
    # always beneficial -- measured: hd72's optimum regresses ~7% with sb_fs, hd128's
    # regresses slightly too, matching the reference CMakeLists policy (only HEAD_DIM in
    # {64,192} enable these flags by default). 'auto' (default) follows each candidate's
    # own split_boundary/fused_softmax flag from the export. Decode has no such flags.
    def variants_for(c):
        if c["mode"] != "prefill":
            return ["-"]
        if args.prefill_variant == "base":
            return ["base"]
        if args.prefill_variant == "sb_fs":
            return ["sb_fs"]
        if args.prefill_variant == "both":
            return ["sb_fs", "base"]
        # auto: respect the export's per-candidate policy
        return ["sb_fs", "base"] if c.get("split_boundary") else ["base"]

    # Build one row per (problem, candidate, variant) and collect the unique target set.
    rows = []
    meta = []             # parallel to rows: (problem_dict, target_name)
    targets = []          # ordered unique target names to build
    seen_tgt = set()
    for p in data["problems"]:
        for c in p["candidates"]:
            for variant in variants_for(c):
                tgt = target_name(p, c, variant)
                rows.append({
                    "problem": p["name"], "mode": c["mode"],
                    "head_dim": c.get("head_dim", ""),
                    "head_dim_qk": c.get("head_dim_qk", ""),
                    "head_dim_vo": c.get("head_dim_vo", ""),
                    "variant": variant,
                    "tile_q":       c.get("tile_q", ""),
                    "tile_seq_len": c.get("tile_seq_len", ""),
                    "tile_hdim":    c.get("tile_hdim", ""),
                    "tile_q_sg":    c.get("tile_q_sg", ""),
                    "tile_kv":      c.get("tile_kv", ""),
                    "tile_kv_sg":   c.get("tile_kv_sg", ""),
                    "tile_sg":      c.get("tile_sg", ""),
                    "tile_d_step":  c.get("tile_d_step", ""),
                    "tile_v_step":  c.get("tile_v_step", ""),
                    "persistent":   c.get("persistent", ""),
                    "proj_time_us": c["proj_time_us"],
                    "proj_mfu":     c["proj_mfu"], "proj_mbu": c["proj_mbu"],
                    "meas_gbps": None, "meas_tflops": None, "meas_time_ms": None,
                    "status": "",
                })
                meta.append((p, tgt))
                if tgt not in seen_tgt:
                    seen_tgt.add(tgt)
                    targets.append(tgt)

    print(f"\n{len(rows)} row(s), {len(targets)} unique target(s) to build "
          f"(prefill variant mode: {args.prefill_variant}).")

    if args.dry_run:
        for r in rows:
            r["status"] = "dry-run"
        write_csv(args.output, rows)
        print(f"[dry-run] plan written to {args.output}")
        return

    # --- PARALLEL BUILD: all targets in ONE ninja invocation. Each FMHA target is a
    #     single heavy .cpp compile, so per-target -j does nothing; passing every target
    #     to one build lets ninja run up to `jobs` compiles CONCURRENTLY. ---
    print(f"\nBuilding {len(targets)} targets in parallel (-j{args.jobs})...")
    build_cmd = (f"cmake --build {args.build_dir} -j{args.jobs} --target "
                 + " ".join(targets))
    b = run(build_cmd, env=env)
    if b.returncode != 0:
        print(b.stdout[-3000:])
        print("WARNING: parallel build reported errors; per-target status resolved by "
              "binary presence below.")

    # --- SERIAL BENCHMARK: run each on the GPU one at a time (avoid GPU contention). ---
    for r, (p, tgt) in zip(rows, meta):
        binary = os.path.join(bin_dir, tgt)
        print(f"\n[{p['name']}] {tgt}")
        if not os.path.isfile(binary):
            r["status"] = "build-failed"
            write_csv(args.output, rows); continue
        prefix = f"ZE_AFFINITY_MASK={args.ze_affinity} " if args.ze_affinity else ""
        rr = run(f"{prefix}{binary} {bench_args(p, args.iterations)}", env=env)
        m = PERF_RE.search(rr.stdout or "")
        if m:
            r["meas_gbps"]    = float(m.group(1))
            r["meas_tflops"]  = float(m.group(2))
            r["meas_time_ms"] = float(m.group(3))
            r["status"] = "ok"
            print(f"  -> {r['meas_tflops']} TFlop/s  {r['meas_gbps']} GB/s  {r['meas_time_ms']} ms")
        else:
            print((rr.stdout or "")[-1000:])
            r["status"] = "run-failed"
        write_csv(args.output, rows)   # incremental save

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nDone. {ok}/{len(rows)} measured. Results -> {args.output}")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
