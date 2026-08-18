#!/bin/bash
set -e

# ============================================================
# CUTLASS Docker Build Entrypoint
# Used by runners to offload compilation to a dedicated build server.
# The CI workflow auto-detects weak runners (low CPU/RAM) and routes
# them here.
#
# Exit codes (also referenced by CI workflow fallback logic):
#   0   = success
#   1   = general/unknown error (e.g. bad DOCKER_TARGET_GROUP value)
#   EXIT_COMPILE_ERROR (42)
#       = compilation error (cmake configure or ninja build failure)
#         CI workflow treats this as a code bug → no fallback
# ============================================================
initialize_config() {
    readonly EXIT_COMPILE_ERROR="${EXIT_COMPILE_ERROR:-42}"
    DOCKER_SOURCE_DIR="${DOCKER_SOURCE_DIR:-/workspace/source}"
    DOCKER_OUTPUT_DIR="${DOCKER_OUTPUT_DIR:-/workspace/output}"
    DOCKER_TMP_SOURCE_DIR="${DOCKER_TMP_SOURCE_DIR:-/workspace/tmp/cutlass-src}"
    DOCKER_BIN_DIR="${DOCKER_BIN_DIR:-/workspace/bin}"
    DOCKER_OCLOC_DIR="${DOCKER_OCLOC_DIR:-/workspace/tools/bmg-ocloc}"
    DOCKER_OCLOC_WRAPPER="${DOCKER_OCLOC_WRAPPER:-${DOCKER_OCLOC_DIR}/ocloc-wrapper}"

    BUILD_ONEAPI_ROOT="${DOCKER_ONEAPI_ROOT:-/opt/intel/oneapi}"
    BUILD_ONEAPI_SETUP_SCRIPT="${DOCKER_ONEAPI_SETUP_SCRIPT:-${BUILD_ONEAPI_ROOT}/setvars.sh}"
    BUILD_PLATFORM="${DOCKER_PLATFORM:?DOCKER_PLATFORM is required (bmg, pvc, or cri)}"
    BUILD_TARGET_GROUP="${DOCKER_TARGET_GROUP:-all}"
    BUILD_SOURCE_DIR="${DOCKER_SOURCE_DIR}"
    BUILD_DIR="${DOCKER_BUILD_DIR:-${DOCKER_OUTPUT_DIR}/build}"
    BUILD_PROFILE="${DOCKER_BUILD_PROFILE:-full}"
    BUILD_TARGET_LIST="${DOCKER_BUILD_TARGET_LIST:-}"
    BUILD_NINJA_COMMAND="${DOCKER_NINJA_COMMAND:-ninja}"
    BUILD_UNIT_TARGET="${DOCKER_UNIT_TARGET:-test/all}"
    BUILD_EXAMPLE_TARGET="${DOCKER_EXAMPLE_TARGET:-examples/all}"
    BUILD_BENCHMARK_TARGET="${DOCKER_BENCHMARK_TARGET:-benchmarks/all}"
    BUILD_CMAKE_EXTRA_FLAGS="${DOCKER_CMAKE_EXTRA_FLAGS:-}"
    BUILD_DISABLE_FLASH_ATTENTION="${DOCKER_DISABLE_FLASH_ATTENTION:-0}"
    BUILD_RUNTIME_LIBRARY_PATHS="${DOCKER_RUNTIME_LIBRARY_PATHS:-}"
    BUILD_FLASH_ATTENTION_CMAKE_FILE="${DOCKER_FLASH_ATTENTION_CMAKE_FILE:-}"
    BUILD_NINJA_JOBS="${DOCKER_NINJA_JOBS:-$(nproc)}"
    BUILD_HEAVY_JOBS="${DOCKER_HEAVY_JOBS:-${BUILD_NINJA_JOBS}}"
    BUILD_SYCL_TARGET="${DOCKER_SYCL_TARGET:-intel_gpu_bmg_g21}"
    BUILD_BMG_SYCL_TARGET="${DOCKER_BMG_SYCL_TARGET:-intel_gpu_bmg_g21}"
    BUILD_PVC_SYCL_TARGET="${DOCKER_PVC_SYCL_TARGET:-intel_gpu_pvc}"
    BUILD_CRI_SYCL_TARGET="${DOCKER_CRI_SYCL_TARGET:-intel_gpu_cri}"
    BUILD_BMG_GRF="${DOCKER_BMG_GRF:-256}"
    BUILD_PVC_GRF="${DOCKER_PVC_GRF:-256}"
    BUILD_CRI_GRF="${DOCKER_CRI_GRF:-512}"
    BUILD_DEVICE_SELECTOR="${DOCKER_DEVICE_SELECTOR:-level_zero:gpu}"
    BUILD_CMAKE_BUILD_TYPE="${DOCKER_CMAKE_BUILD_TYPE:-Release}"
    BUILD_IGC_VISA_OPTIONS="${DOCKER_IGC_VISA_OPTIONS:--perfmodel}"
    BUILD_IGC_VECTOR_ALIAS_BB_THRESHOLD="${DOCKER_IGC_VECTOR_ALIAS_BB_THRESHOLD:-100000000000}"
    BUILD_CC="${DOCKER_CC:-icx}"
    BUILD_CXX="${DOCKER_CXX:-icpx}"
    BUILD_CUTLASS_ENABLE_SYCL="${DOCKER_CUTLASS_ENABLE_SYCL:-ON}"
    BUILD_CUTLASS_ENABLE_BENCHMARKS="${DOCKER_CUTLASS_ENABLE_BENCHMARKS:-ON}"
    BUILD_CUTLASS_SYCL_PROFILING_ENABLED="${DOCKER_CUTLASS_SYCL_PROFILING_ENABLED:-ON}"
    BUILD_CUTLASS_SYCL_RUNNING_CI="${DOCKER_CUTLASS_SYCL_RUNNING_CI:-ON}"

    export BUILD_ONEAPI_ROOT BUILD_ONEAPI_SETUP_SCRIPT BUILD_SOURCE_DIR BUILD_DIR
    export BUILD_PROFILE BUILD_TARGET_LIST BUILD_NINJA_COMMAND
    export BUILD_UNIT_TARGET BUILD_EXAMPLE_TARGET BUILD_BENCHMARK_TARGET
    export BUILD_CMAKE_EXTRA_FLAGS BUILD_DISABLE_FLASH_ATTENTION
    export BUILD_RUNTIME_LIBRARY_PATHS BUILD_FLASH_ATTENTION_CMAKE_FILE
    export BUILD_NINJA_JOBS BUILD_HEAVY_JOBS BUILD_SYCL_TARGET
    export BUILD_BMG_SYCL_TARGET BUILD_PVC_SYCL_TARGET BUILD_CRI_SYCL_TARGET
    export BUILD_BMG_GRF BUILD_PVC_GRF BUILD_CRI_GRF
    export BUILD_DEVICE_SELECTOR BUILD_CMAKE_BUILD_TYPE
    export BUILD_IGC_VISA_OPTIONS BUILD_IGC_VECTOR_ALIAS_BB_THRESHOLD
    export BUILD_CC BUILD_CXX BUILD_CUTLASS_ENABLE_SYCL
    export BUILD_CUTLASS_ENABLE_BENCHMARKS BUILD_CUTLASS_SYCL_PROFILING_ENABLED
    export BUILD_CUTLASS_SYCL_RUNNING_CI
}

print_config() {
    echo "========================================"
    echo "CUTLASS Docker Build"
    echo "SYCL Target:       ${BUILD_SYCL_TARGET}"
    echo "Build Target Group:${BUILD_TARGET_GROUP}"
    echo "CMake Target List: ${BUILD_TARGET_LIST:-<default>}"
    echo "Source Dir:        ${BUILD_SOURCE_DIR}"
    echo "Output Dir:        ${DOCKER_OUTPUT_DIR}"
    echo "CMake Extra Flags: ${BUILD_CMAKE_EXTRA_FLAGS}"
    echo "========================================"
}

setup_environment() {
    echo "[1/4] Setting up oneAPI environment..."
    source "$BUILD_ONEAPI_SETUP_SCRIPT" --force

    if [ -x "$DOCKER_OCLOC_WRAPPER" ]; then
        ln -sf "$DOCKER_OCLOC_WRAPPER" "${DOCKER_BIN_DIR}/ocloc"
        export PATH="${DOCKER_BIN_DIR}:$PATH"
        echo "Using bmg-ocloc wrapper as ${DOCKER_BIN_DIR}/ocloc"
    fi

    export ONEAPI_ROOT="$BUILD_ONEAPI_ROOT"
    export ONEAPI_DEVICE_SELECTOR="$BUILD_DEVICE_SELECTOR"
    export IGC_VISAOptions="$BUILD_IGC_VISA_OPTIONS"
    export IGC_VectorAliasBBThreshold="$BUILD_IGC_VECTOR_ALIAS_BB_THRESHOLD"
    export CC="$BUILD_CC"
    export CXX="$BUILD_CXX"

    echo "Compiler: $(command -v "$BUILD_CXX")"
    "$BUILD_CXX" --version
}

prepare_source() {
    echo "[1.5/4] Copying source to writable location..."
    rm -rf "$DOCKER_TMP_SOURCE_DIR"
    cp -a "$BUILD_SOURCE_DIR" "$DOCKER_TMP_SOURCE_DIR"
    BUILD_SOURCE_DIR="$DOCKER_TMP_SOURCE_DIR"
}

run_ci_build() {
    bash "${BUILD_SOURCE_DIR}/.github/scripts/ci_build.sh" "$BUILD_PLATFORM" "$@" || return "$EXIT_COMPILE_ERROR"
}

run_build() {
    echo "[3/4] Building with -j${BUILD_NINJA_JOBS} (examples -j${BUILD_HEAVY_JOBS})..."
    case "$BUILD_TARGET_GROUP" in
        all)
            run_ci_build all
            ;;
        test)
            run_ci_build configure
            run_ci_build build test/all
            ;;
        examples)
            run_ci_build configure
            run_ci_build build examples/all
            ;;
        benchmarks)
            run_ci_build configure
            run_ci_build build benchmarks/all
            ;;
        *)
            echo "Unknown build target group: ${BUILD_TARGET_GROUP}"
            return 1
            ;;
    esac

    echo "[4/4] Build complete!"
    echo "Build output at: ${BUILD_DIR}"
}

fix_build_paths() {
    local host_build_dir="${DOCKER_HOST_BUILD_DIR:-}"
    local host_source_dir="${DOCKER_HOST_SOURCE_DIR:-}"
    if [ -n "$host_build_dir" ]; then
        echo "Fixing build paths: ${BUILD_DIR} -> ${host_build_dir}"
        find "$BUILD_DIR" \( -name 'DartConfiguration.tcl' -o -name 'CTestTestfile*.cmake' \) \
            -exec sed -i "s|${BUILD_DIR}|${host_build_dir}|g" {} +
        if [ -n "$host_source_dir" ]; then
            echo "Fixing source paths: ${BUILD_SOURCE_DIR} -> ${host_source_dir}"
            find "$BUILD_DIR" -name 'CTestTestfile*.cmake' \
                -exec sed -i "s|${BUILD_SOURCE_DIR}|${host_source_dir}|g" {} +
        fi
        mkdir -p "${BUILD_DIR}/Testing/Temporary"
    fi
}

fix_permissions() {
    echo "Fixing output permissions..."
    chmod -R u+rwX,go+rX "$BUILD_DIR"
}

main() {
    initialize_config
    print_config
    setup_environment
    prepare_source
    run_build
    fix_build_paths
    fix_permissions
    echo "========================================"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
