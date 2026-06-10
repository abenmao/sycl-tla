#!/bin/bash
set -euo pipefail

# ============================================================
# CRI Simulator Build & Test Script
#
# Usage:
#   cri_simulator_compile.sh build
#   cri_simulator_compile.sh test <examples|ut|benchmark>
#
# build      — setup environment, configure & compile.
# test <suite> — setup environment, ensure simulator is running
#                (reuse if alive, clean & start if not), run
#                the specified test suite, then clean up.
#
# Supported test suites:
#   examples  — example programs (gemm, attention, tutorials…)
#   ut        — unit tests (flash_attention, gemm_device…)
#   benchmark — benchmark tests (placeholder, no tests yet)
#
# Each invocation sources the Intel toolchain independently,
# so the script works from separate workflow steps.
# ============================================================

MODE="${1:?Usage: $0 <build|test>}"
case "$MODE" in
    build) ;;
    test)
        TEST_SUITE="${2:?Usage: $0 test <examples|ut|benchmark>}"
        case "$TEST_SUITE" in
            examples|ut|benchmark) ;;
            *) echo "ERROR: Unknown test suite '$TEST_SUITE'. Use: examples, ut, or benchmark."; exit 1 ;;
        esac
        ;;
    *) echo "ERROR: Unknown mode '$MODE'. Use: build or test."; exit 1 ;;
esac

# --- configurable parameters ---
SIM_PORT="${CRI_SIM_PORT:-6117}"
SIM_DIR="${CRI_SIM_DIR:-/opt/intel/crisim}"
SIM_STARTUP_WAIT="${CRI_SIM_STARTUP_WAIT:-30}"
SIM_MAX_RETRIES="${CRI_SIM_MAX_RETRIES:-2}"
# Validate simulator configuration parameters.
if ! [[ "${SIM_PORT}" =~ ^[0-9]+$ ]] || (( SIM_PORT <= 0 || SIM_PORT > 65535 )); then
    echo "WARNING: Invalid SIM_PORT='${SIM_PORT}'; falling back to default 6117."
    SIM_PORT=6117
fi
if ! [[ "${SIM_STARTUP_WAIT}" =~ ^[0-9]+$ ]] || (( SIM_STARTUP_WAIT <= 0 )); then
    echo "WARNING: Invalid SIM_STARTUP_WAIT='${SIM_STARTUP_WAIT}'; falling back to default 30."
    SIM_STARTUP_WAIT=30
fi
if ! [[ "${SIM_MAX_RETRIES}" =~ ^[0-9]+$ ]] || (( SIM_MAX_RETRIES <= 0 )); then
    echo "WARNING: Invalid SIM_MAX_RETRIES='${SIM_MAX_RETRIES}'; falling back to default 2."
    SIM_MAX_RETRIES=2
fi
# Validate simulator directory and required scripts.
if [[ ! -d "${SIM_DIR}" ]]; then
    echo "ERROR: SIM_DIR='${SIM_DIR}' does not exist. Check CRI_SIM_DIR environment variable."
    exit 1
fi
if [[ ! -f "${SIM_DIR}/runsim.sh" ]]; then
    echo "ERROR: '${SIM_DIR}/runsim.sh' is missing."
    exit 1
fi
if [[ ! -f "${SIM_DIR}/env.sh" ]]; then
    echo "ERROR: '${SIM_DIR}/env.sh' is missing."
    exit 1
fi
if [[ "$MODE" == "test" ]] && ! command -v lsof &>/dev/null; then
    echo "ERROR: 'lsof' is required for test mode but not found. Install it or check PATH."
    exit 1
fi
# Auto-calculate safe parallelism (build mode only): each icpx -O3
# process uses ~4 GB.  Use (available_memory / 5GB) to leave headroom,
# capped between 4 and 32, and no more than half the CPU count.
if [[ "$MODE" == "build" ]] && [[ -z "${BUILD_JOBS:-}" ]]; then
    MEM_GB=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
    CPU_COUNT=$(nproc 2>/dev/null || echo 10)
    HALF_CPUS=$(( (CPU_COUNT + 1) / 2 ))
    if [[ "$MEM_GB" -gt 0 ]]; then
        MAX_BY_MEM=$((MEM_GB / 5))
        [[ "$MAX_BY_MEM" -lt 4 ]] && MAX_BY_MEM=4
        [[ "$MAX_BY_MEM" -gt 32 ]] && MAX_BY_MEM=32
        BUILD_JOBS=$((HALF_CPUS < MAX_BY_MEM ? HALF_CPUS : MAX_BY_MEM))
        [[ "$BUILD_JOBS" -lt 4 ]] && BUILD_JOBS=4
    else
        BUILD_JOBS=10  # fallback if /proc/meminfo unavailable
    fi
    echo "Auto-detected build parallelism: -j${BUILD_JOBS} (CPUs: ${CPU_COUNT}, RAM: ${MEM_GB}GB, max by mem: ${MAX_BY_MEM:-N/A}, half CPUs: ${HALF_CPUS})"
fi
# Validate BUILD_JOBS (whether from env or auto-detection): must be a positive
# integer, no more than half CPUs and capped at 32.
if [[ "$MODE" == "build" ]]; then
    if ! [[ "${BUILD_JOBS:-1}" =~ ^[0-9]+$ ]] || [[ "${BUILD_JOBS:-1}" -le 0 ]]; then
        echo "WARNING: Invalid BUILD_JOBS='${BUILD_JOBS:-}'; clamping to 1."
        BUILD_JOBS=1
    else
        HALF_CPUS_VAL=$(( ($(nproc 2>/dev/null || echo 10) + 1) / 2 ))
        MAX_SAFE=$(( HALF_CPUS_VAL < 32 ? HALF_CPUS_VAL : 32 ))
        if [[ "${BUILD_JOBS}" -gt "$MAX_SAFE" ]]; then
            echo "WARNING: BUILD_JOBS=${BUILD_JOBS} exceeds safe limit ${MAX_SAFE} (min of half-CPUs, 32); clamping."
            BUILD_JOBS=$MAX_SAFE
        fi
    fi
fi

# --- cleanup (test mode only — build never starts a simulator) ---
SIM_PID=""
cleanup() {
    if [[ "$MODE" != "test" ]]; then
        return
    fi
    echo ""
    echo "=== Cleaning up simulator ==="
    if [[ -n "${SIM_PID:-}" ]]; then
        if sudo -n kill -0 "$SIM_PID" 2>/dev/null; then
            sudo -n kill -9 "$SIM_PID" 2>/dev/null || true
            echo "Simulator stopped (PID: $SIM_PID)."
        fi
        wait "$SIM_PID" 2>/dev/null || true
    fi
    # Safety net: kill anything still listening on the simulator port.
    if [[ -n "${SIM_PORT:-}" ]]; then
        kill_existing_on_port "$SIM_PORT" || true
    fi
    echo "Cleanup complete."
}
trap cleanup EXIT

# --- helper: kill any process listening on the target port ---
kill_existing_on_port() {
    local port="$1"
    local pids
    pids=$(sudo -n lsof -ti :"$port" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -z "$pids" ]]; then
        echo "Port $port is free."
        return 0
    fi

    echo "WARNING: Found existing process(es) on port $port: $pids"

    # Step 1: try graceful shutdown (SIGTERM)
    echo "$pids" | xargs -r sudo -n kill 2>/dev/null || true
    sleep 2

    # Step 2: check if still alive, escalate to SIGKILL
    pids=$(sudo -n lsof -ti :"$port" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "WARNING: Processes survived SIGTERM on port $port: $pids — escalating to SIGKILL"
        echo "$pids" | xargs -r sudo -n kill -9 2>/dev/null || true
        sleep 1
    fi

    # Step 3: final verification
    pids=$(sudo -n lsof -ti :"$port" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "ERROR: Unable to kill processes on port $port: $pids"
        return 1
    else
        echo "Port $port cleared."
    fi
}

# --- helper: wait for simulator to be ready (port listening) ---
wait_for_sim_ready() {
    local pid="$1" port="$2" timeout="$3"
    for i in $(seq 1 "$timeout"); do
        if ! sudo -n kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: Simulator process (PID $pid) died during startup!"
            return 1
        fi
        if sudo -n lsof -i :"$port" -sTCP:LISTEN &>/dev/null; then
            echo "Simulator is listening on port $port (after ${i}s)."
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Simulator process is alive but port $port is not listening after ${timeout}s!"
    return 1
}

# --- common: setup toolchain environment ---
setup_environment() {
    # Temporarily disable strict unbound-variable checking because
    # Intel setvars.sh / env.sh may reference undefined variables.
    echo ""
    echo "--- Setting up toolchain environment ---"
    set +u
    source "$SIM_DIR/env.sh" "$SIM_PORT"
    source /opt/intel/oneapi/setvars.sh
    set -u
    export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
    export IGC_ExtraOCLOptions="-cl-intel-512-GRF-per-thread"
    export CUTLASS_SYCL_PROFILING_ENABLED=ON
    export CC=icx
    export CXX=icpx
}

# ============================================================
# Mode dispatch
# ============================================================
INITIAL_PWD="$(pwd)"
WORKSPACE="${GITHUB_WORKSPACE:-$INITIAL_PWD}"

if [[ "$MODE" == "build" ]]; then
    echo ""
    echo "=== Setup & Build ==="
    setup_environment

    cd "$WORKSPACE"
    if [[ ! -f CMakeLists.txt ]]; then
        echo "ERROR: No CMakeLists.txt in '$WORKSPACE'. Not a valid repo root."
        exit 1
    fi
    mkdir -p build
    cd build

    echo ""
    echo "--- Configuring CMake ---"
    cmake .. -G Ninja \
        -DCUTLASS_ENABLE_SYCL=ON \
        -DDPCPP_SYCL_TARGET=intel_gpu_cri \
        -DCUTLASS_ENABLE_BENCHMARKS=ON \
        -DCUTLASS_SYCL_RUNNING_CI=ON \
        -DCUTLASS_SYCL_PROFILING_ENABLED=ON \
        -DCUTLASS_TEST_FOR_CRI=ON

    echo ""
    echo "--- Building (ninja -j${BUILD_JOBS}) ---"
    cmake --build . -j"${BUILD_JOBS}"

    echo ""
    echo "=== Build complete ==="
fi

if [[ "$MODE" == "test" ]]; then
    echo ""
    echo "=== Test: $TEST_SUITE ==="

    # Skip suites that have no tests configured yet.
    if [[ "$TEST_SUITE" == "benchmark" ]]; then
        echo "No benchmark tests configured yet. Skipping."
        exit 0
    fi

    setup_environment

    if [[ ! -d "$WORKSPACE/build" ]]; then
        echo "ERROR: Build directory '$WORKSPACE/build' not found. Run 'build' mode first."
        exit 1
    fi
    cd "$WORKSPACE/build"

    # --- ensure simulator is running (reuse if alive, start if not) ---
    if sudo -n lsof -i :"$SIM_PORT" -sTCP:LISTEN &>/dev/null; then
        echo "Simulator already listening on port $SIM_PORT — reusing."
    else
        echo "Simulator not running — starting fresh."
        kill_existing_on_port "$SIM_PORT" || true

        echo "--- Starting CRI simulator (port: $SIM_PORT, max retries: $SIM_MAX_RETRIES) ---"
        SIM_PID=""
        for attempt in $(seq 1 "$SIM_MAX_RETRIES"); do
            echo "  Attempt $attempt of $SIM_MAX_RETRIES"

            cd "$SIM_DIR"
            sudo -n ./runsim.sh "$SIM_PORT" > /dev/null 2>&1 &
            SIM_PID=$!
            echo "  Simulator PID: $SIM_PID"

            echo "  Waiting for simulator to initialize (up to ${SIM_STARTUP_WAIT}s)..."
            if wait_for_sim_ready "$SIM_PID" "$SIM_PORT" "$SIM_STARTUP_WAIT"; then
                break
            fi

            echo "WARNING: Simulator attempt $attempt failed. Check simulator logs on the runner if this persists."
            sudo -n kill -9 "$SIM_PID" 2>/dev/null || true
            wait "$SIM_PID" 2>/dev/null || true
            SIM_PID=""
            kill_existing_on_port "$SIM_PORT" || true
            sleep 2

            if [[ "$attempt" -eq "$SIM_MAX_RETRIES" ]]; then
                echo "ERROR: All $SIM_MAX_RETRIES simulator attempts failed!"
                exit 1
            fi
        done
        cd "$WORKSPACE/build"
    fi

    # --- run the requested test suite ---
    case "$TEST_SUITE" in
        examples)
            echo ""
            echo "--- Running example tests ---"
            ctest -V -R '^(ctest_examples_03_bmg_gemm_streamk|ctest_examples_04_bmg_grouped_gemm|ctest_examples_05_bmg_gemm_with_epilogue_relu|ctest_examples_06_bmg_prefill_attention_cachedkv_hdim64|ctest_examples_08_bmg_gemm_f8|ctest_examples_cute_tutorial_tiled_copy|ctest_examples_cute_tutorial_bmg|ctest_examples_12_xe35_block_scaled_gemm_e2m1|ctest_examples_13_xe35_block_scaled_grouped_gemm_e5m2|ctest_examples_06_xe_fmha_fwd_decode_mx_float_e4m3_t_hdim64|ctest_examples_06_xe_fmha_fwd_decode_mx_float_e2m1_t_hdim64|ctest_examples_14_xe_gdn_attention_bfloat16)$' --output-on-failure
            ;;
        ut)
            echo ""
            echo "--- Running unit tests ---"
            ctest -V -R '^(ctest_unit_flash_attention_decode_h128_xe|ctest_unit_cute_core|ctest_unit_flash_attention_prefill_fp8e4m3_fp32_fp32_h96_xe|ctest_unit_flash_attention_prefill_fp8e4m3_fp32_fp8e4m3_h96_xe|ctest_unit_gemm_device_tensorop_cooperative_xe|ctest_unit_gemm_device_tensorop_epilogue_fusion_xe|ctest_unit_gemm_device_mixed_input_tensorop_xe|ctest_unit_gemm_device_tensorop_xe_group_gemm|ctest_unit_gemm_device_mixed_dtype_tensorop_xe_group_gemm|ctest_unit_gdn_attention_chunkwise)$' --output-on-failure
            ;;
        # benchmark) — handled by early exit above; add ctest pattern here when ready.

    esac

    echo ""
    echo "=== $TEST_SUITE tests passed! ==="
fi
