#!/bin/bash
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
SOURCE_DIR="${BUILD_SOURCE_DIR:-${WORKSPACE}}"
BUILD_DIR="${BUILD_DIR:-${WORKSPACE}/build}"
CRI_DEVICE_SCRIPT="${SOURCE_DIR}/.github/scripts/cri_device.sh"
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
ONEAPI_SET_VARS_SCRIPT="${ONEAPI_SET_VARS_SCRIPT:-${ONEAPI_ROOT}/setvars.sh}"
NINJA_COMMAND="${NINJA_COMMAND:-ninja}"
BMG_SYCL_TARGET="${BMG_SYCL_TARGET:-intel_gpu_bmg_g21}"
PVC_SYCL_TARGET="${PVC_SYCL_TARGET:-intel_gpu_pvc}"
CRI_SYCL_TARGET="${CRI_SYCL_TARGET:-intel_gpu_cri}"
BMG_GRF="${BMG_GRF:-256}"
PVC_GRF="${PVC_GRF:-256}"
CRI_GRF="${CRI_GRF:-512}"
CI_UNIT_BUILD_TARGET="${CI_UNIT_BUILD_TARGET:-test/all}"
CI_EXAMPLE_BUILD_TARGET="${CI_EXAMPLE_BUILD_TARGET:-examples/all}"
CI_BENCHMARK_BUILD_TARGET="${CI_BENCHMARK_BUILD_TARGET:-benchmarks/all}"
CI_BUILD_PROFILE="${CI_BUILD_PROFILE:-full}"
CI_BUILD_TARGET_LIST="${CI_BUILD_TARGET_LIST:-}"
CI_RUNTIME_LIBRARY_PATHS="${CI_RUNTIME_LIBRARY_PATHS:-${BUILD_DIR}/benchmarks/02_flash_attention:${BUILD_DIR}/benchmarks/02_flash_attention/legacy/flash_attention_decode:${BUILD_DIR}/deps/oneMKL/lib}"
BMG_FLASH_ATTENTION_CMAKE_FILE="${BMG_FLASH_ATTENTION_CMAKE_FILE:-${SOURCE_DIR}/examples/06_bmg_flash_attention/CMakeLists.txt}"
ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
IGC_VISAOptions="${IGC_VISAOptions:--perfmodel}"
IGC_VectorAliasBBThreshold="${IGC_VectorAliasBBThreshold:-100000000000}"
CC="${CC:-icx}"
CXX="${CXX:-icpx}"
CUTLASS_ENABLE_SYCL="${CUTLASS_ENABLE_SYCL:-ON}"
CUTLASS_ENABLE_BENCHMARKS="${CUTLASS_ENABLE_BENCHMARKS:-ON}"
CUTLASS_SYCL_RUNNING_CI="${CUTLASS_SYCL_RUNNING_CI:-ON}"

source_oneapi() {
    set +u
    source "$ONEAPI_SET_VARS_SCRIPT" --force
    set -u
}

configure_platform() {
    local platform="$1" expected_target grf
    case "$platform" in
        bmg)
            expected_target="$BMG_SYCL_TARGET"
            grf="$BMG_GRF"
            ;;
        pvc)
            expected_target="$PVC_SYCL_TARGET"
            grf="$PVC_GRF"
            ;;
        cri)
            expected_target="$CRI_SYCL_TARGET"
            grf="$CRI_GRF"
            ;;
        *)
            echo "ERROR: Unsupported platform '$platform'. Use bmg, pvc, or cri."
            return 1
            ;;
    esac
    if [ -n "${SYCL_TARGET:-}" ] && [ "$SYCL_TARGET" != "$expected_target" ]; then
        echo "ERROR: Platform '$platform' requires SYCL_TARGET='$expected_target', got '$SYCL_TARGET'."
        return 1
    fi

    export CI_PLATFORM="$platform"
    export SYCL_TARGET="$expected_target"
    export IGC_ExtraOCLOptions="-cl-intel-${grf}-GRF-per-thread"
    export ONEAPI_DEVICE_SELECTOR CMAKE_BUILD_TYPE IGC_VISAOptions
    export IGC_VectorAliasBBThreshold CC CXX
    export CUTLASS_SYCL_PROFILING_ENABLED="${CUTLASS_SYCL_PROFILING_ENABLED:-ON}"

    echo "Platform: ${CI_PLATFORM}"
    echo "SYCL target: ${SYCL_TARGET}"
    echo "GRF configuration: ${IGC_ExtraOCLOptions}"
}

setup_environment() {
    source_oneapi
    configure_platform "$1"
    local ld_paths="$CI_RUNTIME_LIBRARY_PATHS" # need to check
    export LD_LIBRARY_PATH="${ld_paths}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [ -n "${GITHUB_ENV:-}" ]; then
        env >> "$GITHUB_ENV"
    fi
    which "$CXX"
    "$CXX" --version
    if command -v sycl-ls >/dev/null 2>&1; then
        sycl-ls
    fi
}

prepare_source_tree() {
    if [ "${DISABLE_FLASH_ATTENTION_LOCAL:-1}" = "1" ] && \
       [ "$CI_PLATFORM" = "bmg" ] && \
       [ -f "$BMG_FLASH_ATTENTION_CMAKE_FILE" ]; then
        : > "$BMG_FLASH_ATTENTION_CMAKE_FILE"
    fi
}

configure_cmake() {
    local -a extra_args=()
    configure_platform "$1"
    prepare_source_tree
    if ! command -v "$CXX" >/dev/null 2>&1; then
        source_oneapi
    fi
    if [ -n "${CI_CMAKE_EXTRA_FLAGS:-${CMAKE_EXTRA_FLAGS:-}}" ]; then
        read -r -a extra_args <<< "${CI_CMAKE_EXTRA_FLAGS:-${CMAKE_EXTRA_FLAGS:-}}"
    fi
    mkdir -p "$BUILD_DIR"
    cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DCUTLASS_ENABLE_SYCL="$CUTLASS_ENABLE_SYCL" \
        -DDPCPP_SYCL_TARGET="$SYCL_TARGET" \
        -DCUTLASS_ENABLE_BENCHMARKS="$CUTLASS_ENABLE_BENCHMARKS" \
        -DCUTLASS_SYCL_PROFILING_ENABLED="$CUTLASS_SYCL_PROFILING_ENABLED" \
        -DCUTLASS_SYCL_RUNNING_CI="$CUTLASS_SYCL_RUNNING_CI" \
        "${extra_args[@]}"
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "BUILD_DIR=${BUILD_DIR}" >> "$GITHUB_ENV"
    fi
}

build_target() {
    local jobs="${NINJA_JOBS:-$(nproc)}" target
    local -a targets=("$@")
    [ "${#targets[@]}" -gt 0 ] || {
        echo "ERROR: At least one build target is required."
        return 1
    }
    for target in "${targets[@]}"; do
        if [ "$target" = "$CI_EXAMPLE_BUILD_TARGET" ]; then
            jobs="${HEAVY_JOBS:-$jobs}"
            break
        fi
    done
    "$NINJA_COMMAND" -C "$BUILD_DIR" -j"$jobs" "${targets[@]}"
}

build_all() {
    if [ -n "$CI_BUILD_TARGET_LIST" ]; then
        build_target_list "$CI_BUILD_TARGET_LIST"
        return
    fi
    if [ "$CI_PLATFORM" = "cri" ] && [ "$CI_BUILD_PROFILE" = "tiny" ]; then
        build_cri_tiny
        return
    fi
    build_target "$CI_UNIT_BUILD_TARGET"
    build_target "$CI_EXAMPLE_BUILD_TARGET"
    build_target "$CI_BENCHMARK_BUILD_TARGET"
}

build_target_list() {
    local target_list="$1"
    local -a targets=()
    read -r -a targets <<< "$target_list"
    [ "${#targets[@]}" -gt 0 ] || {
        echo "ERROR: CI_BUILD_TARGET_LIST does not contain any targets."
        return 1
    }
    echo "Build target list: ${targets[*]}"
    build_target "${targets[@]}"
}

build_cri_tiny() {
    local target
    local -a wanted_targets=()
    local -a available_targets=()
    declare -A known_targets=()
    source "$CRI_DEVICE_SCRIPT"
    mapfile -t wanted_targets < <(cri_example_targets; cri_unit_targets)
    while IFS= read -r target; do
        [ -z "$target" ] || known_targets["${target%%:*}"]=1
    done < <("$NINJA_COMMAND" -C "$BUILD_DIR" -t targets all 2>/dev/null)
    for target in "${wanted_targets[@]}"; do
        if [ -n "${known_targets[$target]:-}" ]; then
            available_targets+=("$target")
        else
            echo "WARNING: CRI tiny target '$target' is unavailable and will be skipped."
        fi
    done
    [ "${#available_targets[@]}" -gt 0 ] || {
        echo "ERROR: No CRI tiny targets are available in ${BUILD_DIR}."
        return 1
    }
    build_target "${available_targets[@]}"
}

usage() {
    cat <<'EOF'
Usage: ci_build.sh <bmg|pvc|cri> <setup|configure|build|targets|all> [target ...]

Examples:
  ci_build.sh bmg setup
  ci_build.sh pvc configure
  ci_build.sh cri build test/all
    ci_build.sh bmg targets test/all examples/all
    CI_BUILD_TARGET_LIST="test/all examples/all" ci_build.sh bmg all
  ci_build.sh bmg all
EOF
}

main() {
    local platform="${1:-}" command="${2:-}"
    [ -n "$platform" ] && [ -n "$command" ] || { usage; return 1; }
    case "$command" in
        setup) setup_environment "$platform" ;;
        configure) configure_cmake "$platform" ;;
        build)
            configure_platform "$platform"
            build_target "${3:?Usage: $0 <platform> build <target>}"
            ;;
        targets)
            configure_platform "$platform"
            shift 2
            [ "$#" -gt 0 ] || { echo "ERROR: At least one build target is required."; return 1; }
            build_target_list "$*"
            ;;
        all)
            configure_cmake "$platform"
            build_all
            ;;
        *) echo "ERROR: Unknown command '$command'."; usage; return 1 ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
