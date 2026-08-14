#!/bin/bash
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
BUILD_DIR="${BUILD_DIR:-${WORKSPACE}/build}"
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
ONEAPI_SET_VARS_SCRIPT="${ONEAPI_SET_VARS_SCRIPT:-${ONEAPI_ROOT}/setvars.sh}"
SIM_PORT="${CRI_SIM_PORT:-6117}"
SIM_DIR="${CRI_SIM_DIR:-/opt/intel/crisim}"
SIM_RUN_SCRIPT="${CRI_SIM_RUN_SCRIPT:-${SIM_DIR}/runsim.sh}"
SIM_ENV_SCRIPT="${CRI_SIM_ENV_SCRIPT:-${SIM_DIR}/env.sh}"
SIM_RUN_PATTERN="${CRI_SIM_RUN_PATTERN:-runsim\\.sh}"
SIM_STARTUP_WAIT="${CRI_SIM_STARTUP_WAIT:-30}"
SIM_MAX_RETRIES="${CRI_SIM_MAX_RETRIES:-2}"

EXAMPLES_CTESTS=(
    ctest_examples_03_bmg_gemm_streamk
    ctest_examples_04_bmg_grouped_gemm_test_groups_2
    ctest_examples_04_bmg_grouped_gemm_test_groups_4
    ctest_examples_05_bmg_gemm_with_epilogue_relu
    ctest_examples_06_xe_fmha_fwd_prefill_cached_kv_bfloat16_t_hdim64
    ctest_examples_08_bmg_gemm_f8
    ctest_examples_cute_tutorial_tiled_copy
    ctest_examples_cute_tutorial_bmg
    ctest_examples_50_xe35_block_scaled_gemm_e2m1
    ctest_examples_51_xe35_block_scaled_grouped_gemm_e5m2
    ctest_examples_06_xe_fmha_fwd_decode_mx_float_e4m3_t_hdim64
    ctest_examples_06_xe_fmha_fwd_decode_mx_float_e2m1_t_hdim64
    ctest_examples_14_xe35_gdn_attention_bfloat16
)
UT_CTESTS=(
    ctest_unit_flash_attention_decode_h128_xe
    ctest_unit_cute_core
    ctest_unit_flash_attention_prefill_fp8e4m3_fp32_fp32_h96_xe
    ctest_unit_flash_attention_prefill_fp8e4m3_fp32_fp8e4m3_h96_xe
    ctest_unit_gemm_device_tensorop_cooperative_xe
    ctest_unit_gemm_device_tensorop_epilogue_fusion_xe
    ctest_unit_gemm_device_mixed_input_tensorop_xe
    ctest_unit_gemm_device_tensorop_xe_group_gemm
    ctest_unit_gemm_device_mixed_dtype_tensorop_xe_group_gemm
    ctest_unit_gdn_attention_chunkwise
)

validate_config() {
    [[ "$SIM_PORT" =~ ^[0-9]+$ ]] && (( SIM_PORT > 0 && SIM_PORT <= 65535 )) || {
        echo "ERROR: CRI_SIM_PORT must be between 1 and 65535, got '$SIM_PORT'."
        return 1
    }
    [[ "$SIM_STARTUP_WAIT" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: CRI_SIM_STARTUP_WAIT must be a positive integer."
        return 1
    }
    [[ "$SIM_MAX_RETRIES" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: CRI_SIM_MAX_RETRIES must be a positive integer."
        return 1
    }
}

real_cri_available() {
    local devices
    devices=$(sycl-ls 2>/dev/null || true)
    printf '%s\n' "$devices"
    printf '%s\n' "$devices" | grep -Eiq 'intel_gpu_cri|\bcri\b|crescent[[:space:]_-]*island|xe3p|xe[- ]?3\.5'
}

stop_runsim() {
    local pids pattern="${SIM_RUN_PATTERN} +${SIM_PORT}( |$)"
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    [ -z "$pids" ] && return 0
    echo "Stopping runsim supervisor(s) on port ${SIM_PORT}: ${pids}"
    echo "$pids" | xargs -r sudo -n kill 2>/dev/null || true
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    [ -z "$pids" ] || echo "$pids" | xargs -r sudo -n kill -9 2>/dev/null || true
}

clear_port() {
    local pids
    pids=$(sudo -n lsof -ti :"$SIM_PORT" -sTCP:LISTEN 2>/dev/null || true)
    [ -z "$pids" ] && return 0
    echo "$pids" | xargs -r sudo -n kill 2>/dev/null || true
    pids=$(sudo -n lsof -ti :"$SIM_PORT" -sTCP:LISTEN 2>/dev/null || true)
    [ -z "$pids" ] || echo "$pids" | xargs -r sudo -n kill -9 2>/dev/null || true
}

wait_ready() {
    local pid="$1"
    for ((second = 1; second <= SIM_STARTUP_WAIT; second++)); do
        sudo -n kill -0 "$pid" 2>/dev/null || return 1
        if sudo -n lsof -i :"$SIM_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
            echo "CRI simulator is listening on port ${SIM_PORT} after ${second}s."
            return 0
        fi
        sleep 1
    done
    return 1
}

ensure_device() {
    validate_config
    if ! command -v sycl-ls >/dev/null 2>&1 && [ -f "$ONEAPI_SET_VARS_SCRIPT" ]; then
        set +u
        source "$ONEAPI_SET_VARS_SCRIPT" --force >/dev/null
        set -u
    fi
    if real_cri_available; then
        echo "Real CRI device detected; simulator is not required."
        [ -z "${GITHUB_ENV:-}" ] || echo "CRI_USING_SIMULATOR=false" >> "$GITHUB_ENV"
        return 0
    fi

    command -v lsof >/dev/null 2>&1 || { echo "ERROR: lsof is required to manage the CRI simulator."; return 1; }
    [ -f "$SIM_RUN_SCRIPT" ] || { echo "ERROR: ${SIM_RUN_SCRIPT} is missing."; return 1; }
    [ -f "$SIM_ENV_SCRIPT" ] || { echo "ERROR: ${SIM_ENV_SCRIPT} is missing."; return 1; }

    echo "No real CRI device detected; starting simulator on port ${SIM_PORT}."
    stop_runsim
    clear_port
    for ((attempt = 1; attempt <= SIM_MAX_RETRIES; attempt++)); do
        (
            cd "$SIM_DIR"
            exec sudo -n "$SIM_RUN_SCRIPT" "$SIM_PORT"
        ) >/dev/null 2>&1 &
        local simulator_pid=$!
        if wait_ready "$simulator_pid"; then
            export CRI_USING_SIMULATOR=true
            [ -z "${GITHUB_ENV:-}" ] || echo "CRI_USING_SIMULATOR=true" >> "$GITHUB_ENV"
            set +u
            source "$SIM_ENV_SCRIPT" "$SIM_PORT"
            set -u
            if [ -n "${GITHUB_ENV:-}" ]; then
                env >> "$GITHUB_ENV"
            fi
            return 0
        fi
        echo "WARNING: CRI simulator attempt ${attempt}/${SIM_MAX_RETRIES} failed."
        stop_runsim
        clear_port
    done
    echo "ERROR: CRI simulator failed to start after ${SIM_MAX_RETRIES} attempts."
    return 1
}

cleanup_device() {
    if [ "${CRI_USING_SIMULATOR:-false}" != "true" ]; then
        echo "CRI simulator cleanup skipped; this job used a real device."
        return 0
    fi
    stop_runsim
    clear_port
    echo "CRI simulator cleanup complete."
}

ctest_regex() {
    local IFS='|'
    echo "^($*)$"
}

cri_example_targets() {
    local test_name
    for test_name in "${EXAMPLES_CTESTS[@]}"; do
        test_name="${test_name#ctest_examples_}"
        echo "${test_name%%_test_*}"
    done | awk '!seen[$0]++'
}

cri_unit_targets() {
    local test_name
    for test_name in "${UT_CTESTS[@]}"; do
        echo "cutlass_test_unit_${test_name#ctest_unit_}"
    done
}

run_suite() {
    cd "$BUILD_DIR"
    case "$1" in
        examples) ctest -V -R "$(ctest_regex "${EXAMPLES_CTESTS[@]}")" --output-on-failure ;;
        unit|ut) ctest -V -R "$(ctest_regex "${UT_CTESTS[@]}")" --output-on-failure ;;
        benchmarks|benchmark|bm) echo "No CRI benchmark tests are configured; skipping." ;;
        *) echo "ERROR: Unknown CRI suite '$1'."; return 1 ;;
    esac
}

main() {
    case "${1:-}" in
        ensure) ensure_device ;;
        cleanup) cleanup_device ;;
        test) run_suite "${2:?Usage: $0 test <examples|unit|benchmarks>}" ;;
        *) echo "Usage: $0 <ensure|cleanup|test>"; return 1 ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
