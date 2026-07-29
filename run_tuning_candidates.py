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
GEMM_SUBDIR = "examples/00_bmg_gemm"
GROUP_GEMM_SUBDIR = "examples/04_bmg_grouped_gemm"
PERF_RE = re.compile(
    r"Performance:\s+([\d.]+)\s+GB/s,\s+([\d.]+)\s+TFlop/s,\s+([\d.]+)\s+ms")
# The GEMM examples print a DIFFERENT line than the FMHA ones (no GB/s term):
#   Cutlass GEMM Performance:     [85.633]TFlop/s  (1.6050)ms
# Both 00_bmg_gemm.cpp and 00_bmg_gemm_tune.cpp emit this exact format, so the
# reference anchor and the swept configs stay directly comparable.
GEMM_PERF_RE = re.compile(
    r"Cutlass GEMM Performance:\s+\[([\d.]+)\]TFlop/s\s+\(([\d.]+)\)ms")

# Grouped GEMM prints a THIRD format, different from both FMHA and plain GEMM:
#     Groups      : 8
#     Avg runtime : 6.18596 ms
#     GFLOPS      : 88871.6
# so it needs its own regex, and the two fields arrive on separate lines in the
# opposite order (time first, throughput second). Group order below is normalised to
# (tflops, ms) to match GEMM_PERF_RE, so the shared median/reporting path downstream
# does not need to know which mode produced the sample.
GROUP_GEMM_PERF_RE = re.compile(
    r"Avg runtime\s*:\s*([\d.eE+-]+)\s*ms\s*\n\s*GFLOPS\s*:\s*([\d.eE+-]+)")

# IGC / GRF tuning environment for BMG FA kernels. These MUST match the reference
# tuning regime (examples/06_bmg_flash_attention/run.sh): the FA kernels are
# register-pressure bound and the projection assumes LARGE-GRF (256 reg) mode, so
# without these the AOT-compiled kernels run in a different (slower) codegen regime
# than both the reference and the projection assume. Applied at BUILD time (AOT
# codegen reads IGC_*) AND run time. Measured impact: hd72 256x32x16 goes 52->56 TF.
#
# For GEMM the BUILD-time half of that is not a refinement, it is the difference
# between a working and a broken measurement. Measured on BMG, 4096^3 bf16, the SAME
# binary source: built WITHOUT this env -> 4.3 TFlop/s; built WITH it -> 85.6 TFlop/s,
# a 20x gap. A 256x256 WG tile with an 8x4 sub-group layout gives each sub-group a
# 32x64 fp32 accumulator = 8192 B = exactly the 256-GRF large-GRF file; compiled in
# the default (128-reg) small-GRF mode it spills catastrophically. Setting the env only
# at run time does NOT help, because these binaries are AOT-compiled.
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


def gemm_target_name(c):
    # Matches examples/00_bmg_gemm/CMakeLists.txt's GEMM_TUNE sweep naming.
    return (f"00_bmg_gemm_tune_{c['data_type']}"
            f"_m{c['tile_m']}_n{c['tile_n']}_k{c['tile_k']}"
            f"_sgm{c['tile_sg_m']}_sgn{c['tile_sg_n']}_st{c['stages']}")


def group_gemm_target_name(c):
    # Matches examples/04_bmg_grouped_gemm/CMakeLists.txt's GROUP_GEMM_TUNE naming.
    # No dtype in the name: the grouped tune source is bf16-only (see the exporter's
    # SUPPORTED_GROUP_GEMM_DTYPES note).
    return (f"04_bmg_grouped_gemm_tune"
            f"_m{c['tile_m']}_n{c['tile_n']}_k{c['tile_k']}"
            f"_sgm{c['tile_sg_m']}_sgn{c['tile_sg_n']}_st{c['stages']}")


def group_gemm_anchor_target(_dtype):
    """Untouched reference grouped GEMM -- same anchor role as gemm_anchor_target.

    Note this target's CMake link options were fixed to carry the same large-GRF +
    perfmodel flags as the sweep; without that the anchor is ~8x under-measured and
    every candidate spuriously 'wins' (section 11.1 of the methodology doc)."""
    return "04_bmg_grouped_gemm"


def gemm_anchor_target(dtype):
    """Reference-default anchor to measure alongside a GEMM shortlist.

    Methodology (optimization_skills/11_projection_guided_autotuning.md section 1.2):
    the cheapest way to detect a broken shortlist is to measure the untouched reference
    default; if it beats everything you shortlisted, the search or the model is wrong,
    not just unlucky."""
    return "00_bmg_gemm" if dtype == "bf16" else "00_bmg_gemm_int8"


def decode_anchor_target(head_dim):
    """Untouched reference decode binary -- anchor for decode shortlist cross-validation.

    The default decode target is built without TILE_KV/TILE_KV_SG overrides and uses the
    kernel's compile-time defaults (KV_TILE_SIZE=256, NUM_SG=4). If this beats every
    shortlisted config, the model's config space is too narrow."""
    return f"06_xe_fmha_fwd_decode_bfloat16_t_hdim{head_dim}"


def target_name(problem, c, variant="sb_fs"):
    if c["mode"] == "prefill_asym":
        return asym_target_name(c)
    if c["mode"] == "gemm":
        return gemm_target_name(c)
    if c["mode"] == "group_gemm":
        return group_gemm_target_name(c)
    hd = c["kernel_head_dim"]
    if c["mode"] == "decode":
        return decode_target_name(hd, c)
    return prefill_target_name(hd, c, variant)


def bin_dir_for(build_dir, mode):
    """Each op family's targets land in its own example subdir."""
    sub = {"gemm": GEMM_SUBDIR, "group_gemm": GROUP_GEMM_SUBDIR}.get(mode, EXAMPLE_SUBDIR)
    return os.path.join(build_dir, sub)


def run(cmd, env=None, **kw):
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, text=True, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)


def configure(build_dir, data):
    """Configure cmake with FMHA_TUNE / GEMM_TUNE, scoped to the union of tile values used."""
    # Prefill dims
    pf_heads, tq, tseq, thdim, tqsg = set(), set(), set(), set(), set()
    need_split_boundary = False
    need_fused_softmax  = False
    # Decode dims
    dec_heads, tkv, tkvsg = set(), set(), set()
    # GEMM dims (bf16 / int8 W8A8 tile sweep)
    gm_dtypes, gm_m, gm_n, gm_k = set(), set(), set(), set()
    gm_sgm, gm_sgn, gm_st = set(), set(), set()
    # Grouped-GEMM dims (bf16-only sweep; no dtype axis)
    gg_present = False
    gg_m, gg_n, gg_k, gg_sgm, gg_sgn, gg_st = set(), set(), set(), set(), set(), set()
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
            if c["mode"] == "group_gemm":
                gg_present = True
                gg_m.add(c["tile_m"]);   gg_n.add(c["tile_n"]);   gg_k.add(c["tile_k"])
                gg_sgm.add(c["tile_sg_m"]); gg_sgn.add(c["tile_sg_n"]); gg_st.add(c["stages"])
                continue
            if c["mode"] == "gemm":
                gm_dtypes.add(c["data_type"])
                gm_m.add(c["tile_m"]); gm_n.add(c["tile_n"]); gm_k.add(c["tile_k"])
                gm_sgm.add(c["tile_sg_m"]); gm_sgn.add(c["tile_sg_n"])
                gm_st.add(c["stages"])
                continue
            hd = c["kernel_head_dim"]
            if c["mode"] == "prefill":
                pf_heads.add(hd)
                tq.add(c["tile_q"]); tseq.add(c["tile_seq_len"])
                thdim.add(c["tile_hdim"]); tqsg.add(c["tile_q_sg"])
                if c.get("split_boundary"):
                    need_split_boundary = True
                if c.get("fused_softmax"):
                    need_fused_softmax  = True
            else:
                dec_heads.add(hd)
                tkv.add(c["tile_kv"]); tkvsg.add(c["tile_kv_sg"])

    if not pf_heads and not dec_heads and not asym_present and not gm_dtypes and not gg_present:
        print("No candidates to build.")
        return False

    def lst(s):
        return ";".join(str(x) for x in sorted(s))

    fmha_present = bool(pf_heads or dec_heads or asym_present)
    parts = [f"cmake {build_dir}"]
    # Keep BOTH switches explicit on every configure. CMake cache variables are sticky,
    # so a run that omitted -DGEMM_TUNE after an earlier run set it ON would silently
    # keep regenerating the whole GEMM sweep (and vice-versa for FMHA_TUNE) -- slow, and
    # confusing when a stale target from a previous shortlist still resolves.
    parts.append(f"-DFMHA_TUNE={'ON' if fmha_present else 'OFF'}")
    parts.append(f"-DGEMM_TUNE={'ON' if gm_dtypes else 'OFF'}")
    # Always pass GROUP_GEMM_TUNE explicitly: the CMake cache is sticky, so omitting it
    # would silently leave a previous run's ON in place (same reason GEMM_TUNE is
    # explicit here).
    parts.append(f"-DGROUP_GEMM_TUNE={'ON' if gg_present else 'OFF'}")
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
    if gm_dtypes:
        # The CMake sweep takes the CARTESIAN PRODUCT of these lists, so it generates a
        # superset of the shortlist (plus whatever the divisibility guard drops). That is
        # fine and intentional -- only the named targets are ever built.
        parts += [f'-DGEMM_TUNE_DTYPES="{lst(gm_dtypes)}"',
                  f'-DGEMM_TUNE_TILE_M="{lst(gm_m)}"',
                  f'-DGEMM_TUNE_TILE_N="{lst(gm_n)}"',
                  f'-DGEMM_TUNE_TILE_K="{lst(gm_k)}"',
                  f'-DGEMM_TUNE_TILE_SG_M="{lst(gm_sgm)}"',
                  f'-DGEMM_TUNE_TILE_SG_N="{lst(gm_sgn)}"',
                  f'-DGEMM_TUNE_STAGES="{lst(gm_st)}"']

    if gg_present:
        parts += [f'-DGROUP_GEMM_TUNE_TILE_M="{lst(gg_m)}"',
                  f'-DGROUP_GEMM_TUNE_TILE_N="{lst(gg_n)}"',
                  f'-DGROUP_GEMM_TUNE_TILE_K="{lst(gg_k)}"',
                  f'-DGROUP_GEMM_TUNE_TILE_SG_M="{lst(gg_sgm)}"',
                  f'-DGROUP_GEMM_TUNE_TILE_SG_N="{lst(gg_sgn)}"',
                  f'-DGROUP_GEMM_TUNE_STAGES="{lst(gg_st)}"']

    r = run(" ".join(parts))
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print("ERROR: cmake configure failed.")
        return False
    return True


def bench_args(problem, iterations):
    if problem.get("mode") == "gemm":
        # GEMM/quant-matmul: a different CLI than the FMHA runner entirely.
        # --verify=0 to keep the measured loop clean (the reference GemmComplex check is
        # far slower than the kernel and is not what we are timing).
        # alpha=1/beta=0 keeps the epilogue at its cheapest; for int8 alpha is the
        # per-tensor dequant scale, whose VALUE cannot change the kernel's cost, so
        # leaving it at 1.0 measures the same work a real dequant would do.
        return (f"--m={problem['m']} --n={problem['n']} --k={problem['k']} "
                f"--l={problem.get('batch_l', 1)} "
                f"--iterations={iterations} --verify=0 --alpha=1 --beta=0")
    if problem.get("mode") == "group_gemm":
        # 04_bmg_grouped_gemm replicates a single --m across --groups; the exporter has
        # already rejected non-uniform ms, so problem["m"] is the shared per-expert M.
        return (f"--m={problem['m']} --n={problem['n']} --k={problem['k']} "
                f"--groups={problem['groups']} "
                f"--iterations={iterations} --verify=0 --alpha=1 --beta=0")
    causal = "--is_causal" if problem.get("causal") else ""
    varlen = "--varlen" if problem.get("varlen") else ""
    length = problem["length"]
    # For decode problems length=0 (projection decode path); use q_len for the benchmark.
    bench_q_len = problem.get("q_len", length) if length == 0 else length
    kv = problem.get("context", 0) + bench_q_len
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
    return (f"--iterations={iterations} --batch={problem.get('batch', 1)} --verify=0 "
            f"--num_heads_q={problem['q_heads_per_rank']} "
            f"--num_heads_kv={problem['kv_heads_per_rank']} "
            f"--head_size_qk={head_qk} --head_size_vo={head_vo} "
            f"--seq_len_qo={bench_q_len} --seq_len_kv={kv} {causal} {varlen}").strip()


def candidate_summary(c):
    if c["mode"] == "group_gemm":
        return (f"ggemm tile {c['tile_m']}x{c['tile_n']}x{c['tile_k']} "
                f"sg{c['tile_sg_m']}x{c['tile_sg_n']} st{c['stages']}  "
                f"(proj {c['proj_time_us']}us / MFU {c['proj_mfu'] * 100:.1f}%)")
    if c["mode"] == "prefill":
        return (f"prefill tile q{c['tile_q']} seq{c['tile_seq_len']} "
                f"hdim{c['tile_hdim']} qsg{c['tile_q_sg']}  "
                f"(proj {c['proj_time_us']}us / MFU {c['proj_mfu'] * 100:.1f}%)")
    elif c["mode"] == "prefill_asym":
        return (f"asym    tile q{c['tile_q']} kv{c['tile_kv']} sg{c['tile_sg']} "
                f"d{c['tile_d_step']} v{c['tile_v_step']}  "
                f"(proj {c['proj_time_us']}us / MFU {c['proj_mfu'] * 100:.1f}%)")
    elif c["mode"] == "gemm":
        return (f"gemm    {c['data_type']} tile {c['tile_m']}x{c['tile_n']}x{c['tile_k']} "
                f"sg {c['tile_sg_m']}x{c['tile_sg_n']} st{c['stages']}  "
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
    ap.add_argument("--repeats", type=int, default=1,
                    help="re-launch each benchmark binary N times and keep the MEDIAN "
                         "sample. Default 1. Short kernels on this box have a heavily "
                         "RIGHT-SKEWED per-launch distribution, not symmetric noise: "
                         "2048x384x3584 bf16 over 9 launches gave 42.8 42.8 42.8 42.9 "
                         "42.9 44.7 44.8 53.0 62.3 TFlop/s -- a tight mode at ~42.8 with "
                         "a long fast tail. --iterations does NOT average this out (the "
                         "mode is fixed per process). With N=3 the median lands in the "
                         "tail often enough to fabricate a 40%% 'win' between identical "
                         "candidate sets, which is exactly what happened once here. Use "
                         "--repeats 9 for sub-millisecond kernels, 1 for shapes whose "
                         "per-launch spread is already under ~1%% (2048x8192x8192 holds "
                         "92.28-92.30, so N=1 is fine there). The printed "
                         "'[n=.. lo-hi, spread ..%%]' tells you which regime you are in.")
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

    # (binary directory is resolved per-row via bin_dir_for(): GEMM targets live under
    #  examples/00_bmg_gemm, FMHA targets under examples/06_bmg_flash_attention)

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

    def add_row(p, c, variant, tgt):
        rows.append({
            "problem": p["name"], "mode": c["mode"],
            "data_type":   c.get("data_type", ""),
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
            # GEMM problem extent + tile
            "m": p.get("m", ""), "n": p.get("n", ""), "k": p.get("k", ""),
            "l": p.get("batch_l", ""),
            "tile_m":     c.get("tile_m", ""),
            "tile_n":     c.get("tile_n", ""),
            "tile_k":     c.get("tile_k", ""),
            "tile_sg_m":  c.get("tile_sg_m", ""),
            "tile_sg_n":  c.get("tile_sg_n", ""),
            "stages":     c.get("stages", ""),
            "proj_time_us": c.get("proj_time_us", ""),
            "proj_mfu":     c.get("proj_mfu", ""),
            "proj_mbu":     c.get("proj_mbu", ""),
            "groups": p.get("groups"),   # grouped GEMM only (number of experts)
            "meas_gbps": None, "meas_tflops": None, "meas_time_ms": None,
            "status": "",
        })
        meta.append((p, tgt))
        if tgt not in seen_tgt:
            seen_tgt.add(tgt)
            targets.append(tgt)

    for p in data["problems"]:
        # Reference-default ANCHOR row, measured on the same problem shape as the
        # shortlist. Methodology section 1.2: if the untouched default beats every
        # shortlisted config, the shortlist/model has a bug -- and you only find that out
        # if you actually measure the default in the same run, under the same env.
        if p.get("mode") == "gemm" and p["candidates"]:
            anchor = {"mode": "gemm", "data_type": p["data_type"]}
            add_row(p, anchor, "anchor", gemm_anchor_target(p["data_type"]))
        if p.get("mode") == "group_gemm" and p["candidates"]:
            anchor = {"mode": "group_gemm", "data_type": p["data_type"]}
            add_row(p, anchor, "anchor", group_gemm_anchor_target(p["data_type"]))
        if p.get("mode") == "decode" and p["candidates"]:
            anchor = {"mode": "decode", "head_dim": p["head_dim"]}
            add_row(p, anchor, "anchor", decode_anchor_target(p["kernel_head_dim"]))
        for c in p["candidates"]:
            for variant in variants_for(c):
                add_row(p, c, variant, target_name(p, c, variant))

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
        binary = os.path.join(bin_dir_for(args.build_dir, r["mode"]), tgt)
        print(f"\n[{p['name']}] {tgt}"
              + ("  (reference-default ANCHOR)" if r["variant"] == "anchor" else ""))
        if not os.path.isfile(binary):
            r["status"] = "build-failed"
            write_csv(args.output, rows); continue
        prefix = f"ZE_AFFINITY_MASK={args.ze_affinity} " if args.ze_affinity else ""
        cmd = f"{prefix}{binary} {bench_args(p, args.iterations)}"

        # Repeat the whole binary invocation and keep the MEDIAN, not a single sample.
        #
        # Why: short kernels are BIMODAL on this box, not merely noisy. Measured on a
        # quiesced BMG B60 (device 4), bf16 2048x384x3584 / 160x128x32 sg2x8, six
        # back-to-back runs of --iterations=100 gave
        #     44.9  44.7  44.9  64.3  44.9  42.9  TFlop/s
        # i.e. five samples at ~44.9 and one 43% high outlier. The same binary on
        # 2048x8192x8192 gave 92.287 92.276 92.304 92.300 -- a 0.03% spread. The
        # in-binary iteration loop does NOT average this away, because the mode is
        # picked once per process (clock/power state at launch), not per iteration.
        #
        # A single sample therefore fabricates differences between configs that are
        # actually identical: an earlier run of this script reported a +40.7% "win" for
        # bf16 2048x384x3584 across two passes whose bf16 candidate lists were verified
        # byte-identical -- pure outlier. Median-of-N kills that; mean would not.
        #
        # Cost is bounded: large shapes are where wall time lives and they only need
        # one sample, so --repeats defaults to 1 and you opt in for small shapes.
        samples, last_stdout = [], ""
        for i in range(max(1, args.repeats)):
            rr = run(cmd, env=env) if i == 0 else subprocess.run(
                cmd, shell=True, text=True, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            last_stdout = rr.stdout or ""
            if r["mode"] == "group_gemm":
                m = GROUP_GEMM_PERF_RE.search(last_stdout)
                if m:
                    # regex captures (ms, GFLOPS); normalise to the (tflops, ms) tuple
                    # the GEMM path uses so the median/report code stays mode-agnostic.
                    ms_v, gflops = float(m.group(1)), float(m.group(2))
                    samples.append((gflops / 1000.0, ms_v))
            else:
                m = (GEMM_PERF_RE if r["mode"] == "gemm" else PERF_RE).search(last_stdout)
                if m:
                    samples.append(tuple(float(g) for g in m.groups()))

        if not samples:
            print(last_stdout[-1000:])
            r["status"] = "run-failed"
            write_csv(args.output, rows); continue

        # Median per field, ranked by TFlop/s so the reported time/GB-s stay consistent
        # with the reported throughput (a per-field median could mix two different runs).
        # GEMM & grouped GEMM: (tflops, ms). FMHA: (gbps, tflops, ms).
        tf_idx = 0 if r["mode"] in ("gemm", "group_gemm") else 1
        chosen = sorted(samples, key=lambda s: s[tf_idx])[len(samples) // 2]

        if r["mode"] in ("gemm", "group_gemm"):
            # Both print throughput+time and no GB/s -- meas_gbps stays empty.
            r["meas_tflops"], r["meas_time_ms"] = chosen
        else:
            r["meas_gbps"], r["meas_tflops"], r["meas_time_ms"] = chosen
        r["status"] = "ok"

        spread = ""
        if len(samples) > 1:
            tfs = [s[tf_idx] for s in samples]
            spread = (f"  [n={len(samples)} {min(tfs):.1f}-{max(tfs):.1f}, "
                      f"spread {(max(tfs) - min(tfs)) / max(tfs) * 100:.1f}%]")
        gbps = "" if r["mode"] in ("gemm", "group_gemm") else f"  {r['meas_gbps']} GB/s"
        print(f"  -> {r['meas_tflops']} TFlop/s{gbps}  {r['meas_time_ms']} ms{spread}")
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
