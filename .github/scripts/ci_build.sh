#!/bin/bash
set -euo pipefail

BUILD_SOURCE_DIR="${BUILD_SOURCE_DIR:-$(pwd)}"
BUILD_OUTPUT_DIR="${BUILD_DIR:-${BUILD_SOURCE_DIR}/build}"
BUILD_CRI_DEVICE_SCRIPT="${BUILD_SOURCE_DIR}/.github/scripts/cri_device.sh"
BUILD_ONEAPI_ROOT="${BUILD_ONEAPI_ROOT:-/opt/intel/oneapi}"
BUILD_ONEAPI_SETUP_SCRIPT="${BUILD_ONEAPI_SETUP_SCRIPT:-${BUILD_ONEAPI_ROOT}/setvars.sh}"
BUILD_NINJA_COMMAND="${BUILD_NINJA_COMMAND:-ninja}"
BUILD_BMG_SYCL_TARGET="${BUILD_BMG_SYCL_TARGET:-intel_gpu_bmg_g21}"
BUILD_PVC_SYCL_TARGET="${BUILD_PVC_SYCL_TARGET:-intel_gpu_pvc}"
BUILD_CRI_SYCL_TARGET="${BUILD_CRI_SYCL_TARGET:-intel_gpu_cri}"
BUILD_BMG_GRF="${BUILD_BMG_GRF:-256}"
BUILD_PVC_GRF="${BUILD_PVC_GRF:-256}"
BUILD_CRI_GRF="${BUILD_CRI_GRF:-512}"
BUILD_UNIT_TARGET="${BUILD_UNIT_TARGET:-test/all}"
BUILD_EXAMPLE_TARGET="${BUILD_EXAMPLE_TARGET:-examples/all}"
BUILD_BENCHMARK_TARGET="${BUILD_BENCHMARK_TARGET:-benchmarks/all}"
BUILD_PROFILE="${BUILD_PROFILE:-full}"
BUILD_TARGET_LIST="${BUILD_TARGET_LIST:-}"
BUILD_CMAKE_EXTRA_FLAGS="${BUILD_CMAKE_EXTRA_FLAGS:-}"
BUILD_RUNTIME_LIBRARY_PATHS="${BUILD_RUNTIME_LIBRARY_PATHS:-${BUILD_OUTPUT_DIR}/benchmarks/02_flash_attention:${BUILD_OUTPUT_DIR}/benchmarks/02_flash_attention/legacy/flash_attention_decode:${BUILD_OUTPUT_DIR}/deps/oneMKL/lib}"
BUILD_FLASH_ATTENTION_CMAKE_FILE="${BUILD_FLASH_ATTENTION_CMAKE_FILE:-${BUILD_SOURCE_DIR}/examples/06_bmg_flash_attention/CMakeLists.txt}"
BUILD_DISABLE_FLASH_ATTENTION="${BUILD_DISABLE_FLASH_ATTENTION:-0}"
BUILD_REQUESTED_SYCL_TARGET="${BUILD_SYCL_TARGET:-}"
BUILD_DEVICE_SELECTOR="${BUILD_DEVICE_SELECTOR:-level_zero:gpu}"
BUILD_CMAKE_BUILD_TYPE="${BUILD_CMAKE_BUILD_TYPE:-Release}"
BUILD_IGC_VISA_OPTIONS="${BUILD_IGC_VISA_OPTIONS:--perfmodel}"
BUILD_IGC_VECTOR_ALIAS_BB_THRESHOLD="${BUILD_IGC_VECTOR_ALIAS_BB_THRESHOLD:-100000000000}"
BUILD_CC="${BUILD_CC:-icx}"
BUILD_CXX="${BUILD_CXX:-icpx}"
BUILD_CUTLASS_ENABLE_SYCL="${BUILD_CUTLASS_ENABLE_SYCL:-ON}"
BUILD_CUTLASS_ENABLE_BENCHMARKS="${BUILD_CUTLASS_ENABLE_BENCHMARKS:-ON}"
BUILD_CUTLASS_SYCL_PROFILING_ENABLED="${BUILD_CUTLASS_SYCL_PROFILING_ENABLED:-ON}"
BUILD_CUTLASS_SYCL_RUNNING_CI="${BUILD_CUTLASS_SYCL_RUNNING_CI:-ON}"
BUILD_NINJA_JOBS="${BUILD_NINJA_JOBS:-$(nproc)}"
BUILD_HEAVY_JOBS="${BUILD_HEAVY_JOBS:-${BUILD_NINJA_JOBS}}"

source_oneapi() {
    set +u
    source "$BUILD_ONEAPI_SETUP_SCRIPT" --force
    set -u
}

configure_platform() {
    local platform="$1" expected_target grf
    case "$platform" in
        bmg)
            expected_target="$BUILD_BMG_SYCL_TARGET"
            grf="$BUILD_BMG_GRF"
            ;;
        pvc)
            expected_target="$BUILD_PVC_SYCL_TARGET"
            grf="$BUILD_PVC_GRF"
            ;;
        cri)
            expected_target="$BUILD_CRI_SYCL_TARGET"
            grf="$BUILD_CRI_GRF"
            ;;
        *)
            echo "ERROR: Unsupported platform '$platform'. Use bmg, pvc, or cri."
            return 1
            ;;
    esac
    if [ -n "$BUILD_REQUESTED_SYCL_TARGET" ] && [ "$BUILD_REQUESTED_SYCL_TARGET" != "$expected_target" ]; then
        echo "ERROR: Platform '$platform' requires SYCL target '$expected_target', got '$BUILD_REQUESTED_SYCL_TARGET'."
        return 1
    fi

    BUILD_ACTIVE_PLATFORM="$platform"
    BUILD_ACTIVE_SYCL_TARGET="$expected_target"
    export BUILD_DIR="$BUILD_OUTPUT_DIR"
    export ONEAPI_ROOT="$BUILD_ONEAPI_ROOT"
    export SYCL_TARGET="$expected_target"
    export IGC_ExtraOCLOptions="-cl-intel-${grf}-GRF-per-thread"
    export ONEAPI_DEVICE_SELECTOR="$BUILD_DEVICE_SELECTOR"
    export IGC_VISAOptions="$BUILD_IGC_VISA_OPTIONS"
    export IGC_VectorAliasBBThreshold="$BUILD_IGC_VECTOR_ALIAS_BB_THRESHOLD"
    export CC="$BUILD_CC"
    export CXX="$BUILD_CXX"

    echo "Platform: ${BUILD_ACTIVE_PLATFORM}"
    echo "SYCL target: ${BUILD_ACTIVE_SYCL_TARGET}"
    echo "GRF configuration: ${IGC_ExtraOCLOptions}"
}

setup_environment() {
    source_oneapi
    configure_platform "$1"
    local runtime_library_paths="$BUILD_RUNTIME_LIBRARY_PATHS"
    export LD_LIBRARY_PATH="${runtime_library_paths}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [ -n "${GITHUB_ENV:-}" ]; then
        env >> "$GITHUB_ENV"
    fi
    command -v "$BUILD_CXX"
    "$BUILD_CXX" --version
    if command -v sycl-ls >/dev/null 2>&1; then
        sycl-ls
    fi
}

prepare_source_tree() {
    local build_disable_flash_attention="$BUILD_DISABLE_FLASH_ATTENTION"
    if [ "$build_disable_flash_attention" = "1" ] && \
       [ -f "$BUILD_FLASH_ATTENTION_CMAKE_FILE" ]; then
        : > "$BUILD_FLASH_ATTENTION_CMAKE_FILE"
    fi
}

configure_cmake() {
    local -a extra_args=()
    configure_platform "$1"
    prepare_source_tree
    if ! command -v "$BUILD_CXX" >/dev/null 2>&1; then
        source_oneapi
    fi
    if [ -n "${BUILD_CMAKE_EXTRA_FLAGS:-}" ]; then
        read -r -a extra_args <<< "$BUILD_CMAKE_EXTRA_FLAGS"
    fi
    mkdir -p "$BUILD_OUTPUT_DIR"
    cmake -S "$BUILD_SOURCE_DIR" -B "$BUILD_OUTPUT_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE="$BUILD_CMAKE_BUILD_TYPE" \
        -DCUTLASS_ENABLE_SYCL="$BUILD_CUTLASS_ENABLE_SYCL" \
        -DDPCPP_SYCL_TARGET="$BUILD_ACTIVE_SYCL_TARGET" \
        -DCUTLASS_ENABLE_BENCHMARKS="$BUILD_CUTLASS_ENABLE_BENCHMARKS" \
        -DCUTLASS_SYCL_PROFILING_ENABLED="$BUILD_CUTLASS_SYCL_PROFILING_ENABLED" \
        -DCUTLASS_SYCL_RUNNING_CI="$BUILD_CUTLASS_SYCL_RUNNING_CI" \
        "${extra_args[@]}"
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "BUILD_DIR=${BUILD_OUTPUT_DIR}" >> "$GITHUB_ENV"
    fi
}

build_target() {
    local jobs="$BUILD_NINJA_JOBS" target
    local -a targets=("$@")
    [ "${#targets[@]}" -gt 0 ] || {
        echo "ERROR: At least one build target is required."
        return 1
    }
    for target in "${targets[@]}"; do
        if [ "$target" = "$BUILD_EXAMPLE_TARGET" ]; then
            jobs="$BUILD_HEAVY_JOBS"
            break
        fi
    done
    "$BUILD_NINJA_COMMAND" -C "$BUILD_OUTPUT_DIR" -j"$jobs" "${targets[@]}"
}

build_all() {
    if [ -n "$BUILD_TARGET_LIST" ]; then
        build_target_list "$BUILD_TARGET_LIST"
        return
    fi
    if [ "$BUILD_ACTIVE_PLATFORM" = "cri" ] && [ "$BUILD_PROFILE" = "tiny" ]; then
        build_cri_tiny
        return
    fi
    build_target "$BUILD_UNIT_TARGET"
    build_target "$BUILD_EXAMPLE_TARGET"
    build_target "$BUILD_BENCHMARK_TARGET"
}

build_target_list() {
    local target_list="$1"
    local -a targets=()
    read -r -a targets <<< "$target_list"
    [ "${#targets[@]}" -gt 0 ] || {
        echo "ERROR: BUILD_TARGET_LIST does not contain any targets."
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
    source "$BUILD_CRI_DEVICE_SCRIPT"
    mapfile -t wanted_targets < <(cri_example_targets; cri_unit_targets)
    while IFS= read -r target; do
        [ -z "$target" ] || known_targets["${target%%:*}"]=1
    done < <("$BUILD_NINJA_COMMAND" -C "$BUILD_OUTPUT_DIR" -t targets all 2>/dev/null)
    for target in "${wanted_targets[@]}"; do
        if [ -n "${known_targets[$target]:-}" ]; then
            available_targets+=("$target")
        else
            echo "WARNING: CRI tiny target '$target' is unavailable and will be skipped."
        fi
    done
    [ "${#available_targets[@]}" -gt 0 ] || {
        echo "ERROR: No CRI tiny targets are available in ${BUILD_OUTPUT_DIR}."
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
    BUILD_TARGET_LIST="test/all examples/all" ci_build.sh bmg all
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
