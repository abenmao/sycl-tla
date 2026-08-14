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
#   1   = general/unknown error (e.g. bad BUILD_TARGETS value)
#   EXIT_COMPILE_ERROR (42)
#       = compilation error (cmake configure or ninja build failure)
#         CI workflow treats this as a code bug → no fallback
# ============================================================
initialize_config() {
    readonly EXIT_COMPILE_ERROR="${EXIT_COMPILE_ERROR:-42}"
    ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
    ONEAPI_SET_VARS_SCRIPT="${ONEAPI_SET_VARS_SCRIPT:-${ONEAPI_ROOT}/setvars.sh}"
    CONTAINER_SOURCE_DIR="${CONTAINER_SOURCE_DIR:-/workspace/source}"
    CONTAINER_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR:-/workspace/output}"
    CONTAINER_TMP_SOURCE_DIR="${CONTAINER_TMP_SOURCE_DIR:-/workspace/tmp/cutlass-src}"
    CONTAINER_BIN_DIR="${CONTAINER_BIN_DIR:-/workspace/bin}"
    CONTAINER_OCLOC_DIR="${CONTAINER_OCLOC_DIR:-/workspace/tools/bmg-ocloc}"
    OCLOC_WRAPPER="${OCLOC_WRAPPER:-${CONTAINER_OCLOC_DIR}/ocloc-wrapper}"
    SYCL_TARGET="${SYCL_TARGET:-intel_gpu_bmg_g21}"
    CI_PLATFORM="${CI_PLATFORM:?CI_PLATFORM is required (bmg, pvc, or cri)}"
    BUILD_TARGETS="${BUILD_TARGETS:-all}"
    SOURCE_DIR="${CONTAINER_SOURCE_DIR}"
    OUTPUT_DIR="${CONTAINER_OUTPUT_DIR}"
    BUILD_DIR="${BUILD_DIR:-${OUTPUT_DIR}/build}"
    CMAKE_EXTRA_FLAGS="${CMAKE_EXTRA_FLAGS:-}"
    DISABLE_FLASH_ATTENTION="${DISABLE_FLASH_ATTENTION:-1}"
    NINJA_JOBS="${NINJA_JOBS:-$(nproc)}"
    HEAVY_JOBS="${HEAVY_JOBS:-${NINJA_JOBS}}"
    ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
    CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
    IGC_VISAOptions="${IGC_VISAOptions:--perfmodel}"
    IGC_VectorAliasBBThreshold="${IGC_VectorAliasBBThreshold:-100000000000}"
    CC="${CC:-icx}"
    CXX="${CXX:-icpx}"
    CUTLASS_SYCL_PROFILING_ENABLED="${CUTLASS_SYCL_PROFILING_ENABLED:-ON}"
}

print_config() {
    echo "========================================"
    echo "CUTLASS Docker Build"
    echo "SYCL Target:       ${SYCL_TARGET}"
    echo "Build Targets:     ${BUILD_TARGETS}"
    echo "CMake Target List: ${CI_BUILD_TARGET_LIST:-<default>}"
    echo "Source Dir:        ${SOURCE_DIR}"
    echo "Output Dir:        ${OUTPUT_DIR}"
    echo "CMake Extra Flags: ${CMAKE_EXTRA_FLAGS}"
    echo "========================================"
}

setup_environment() {
    echo "[1/4] Setting up oneAPI environment..."
    source "$ONEAPI_SET_VARS_SCRIPT" --force

    if [ -x "$OCLOC_WRAPPER" ]; then
        ln -sf "$OCLOC_WRAPPER" "${CONTAINER_BIN_DIR}/ocloc"
        export PATH="${CONTAINER_BIN_DIR}:$PATH"
        echo "Using bmg-ocloc wrapper as ${CONTAINER_BIN_DIR}/ocloc"
    fi

    export ONEAPI_ROOT ONEAPI_SET_VARS_SCRIPT
    export ONEAPI_DEVICE_SELECTOR CMAKE_BUILD_TYPE IGC_VISAOptions
    export IGC_VectorAliasBBThreshold CC CXX CUTLASS_SYCL_PROFILING_ENABLED
    export SYCL_TARGET

    echo "Compiler: $(which "$CXX")"
    "$CXX" --version
}

prepare_source() {
    echo "[1.5/4] Copying source to writable location..."
    rm -rf "$CONTAINER_TMP_SOURCE_DIR"
    cp -a "${SOURCE_DIR}" "$CONTAINER_TMP_SOURCE_DIR"
    SOURCE_DIR="$CONTAINER_TMP_SOURCE_DIR"

    export BUILD_SOURCE_DIR="${SOURCE_DIR}"
    export BUILD_DIR
    export CI_CMAKE_EXTRA_FLAGS="${CMAKE_EXTRA_FLAGS}"
    export DISABLE_FLASH_ATTENTION_LOCAL="${DISABLE_FLASH_ATTENTION}"
    export NINJA_JOBS HEAVY_JOBS
}

run_ci_build() {
    bash "${SOURCE_DIR}/.github/scripts/ci_build.sh" "${CI_PLATFORM}" "$@" || return "$EXIT_COMPILE_ERROR"
}

run_build() {
    echo "[3/4] Building with -j${NINJA_JOBS} (examples -j${HEAVY_JOBS})..."
    case "${BUILD_TARGETS}" in
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
            echo "Unknown build target: ${BUILD_TARGETS}"
            return 1
            ;;
    esac

    echo "[4/4] Build complete!"
    echo "Build output at: ${BUILD_DIR}"
}

fix_build_paths() {
    local remote_build_dir="${REMOTE_BUILD_DIR:-}"
    local remote_source_dir="${REMOTE_SOURCE_DIR:-}"
    if [ -n "$remote_build_dir" ]; then
        echo "Fixing build paths: ${BUILD_DIR} -> ${remote_build_dir}"
        find "$BUILD_DIR" \( -name 'DartConfiguration.tcl' -o -name 'CTestTestfile*.cmake' \) \
            -exec sed -i "s|${BUILD_DIR}|${remote_build_dir}|g" {} +
        if [ -n "$remote_source_dir" ]; then
            echo "Fixing source paths: ${SOURCE_DIR} -> ${remote_source_dir}"
            find "$BUILD_DIR" -name 'CTestTestfile*.cmake' \
                -exec sed -i "s|${SOURCE_DIR}|${remote_source_dir}|g" {} +
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
