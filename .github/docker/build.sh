#!/bin/bash
set -e

# ============================================================
# CUTLASS Docker Build Script
# Used by BMG runners to offload compilation to build server
# ============================================================

# Arguments
SYCL_TARGET="${SYCL_TARGET:-intel_gpu_bmg_g21}"
BUILD_TARGETS="${BUILD_TARGETS:-all}"  # all, test, examples, benchmarks
SOURCE_DIR="/workspace/source"
OUTPUT_DIR="/workspace/output"

echo "========================================"
CMAKE_EXTRA_FLAGS="${CMAKE_EXTRA_FLAGS:-}"

echo "CUTLASS Docker Build"
echo "SYCL Target:      ${SYCL_TARGET}"
echo "Build Targets:    ${BUILD_TARGETS}"
echo "Source Dir:        ${SOURCE_DIR}"
echo "Output Dir:        ${OUTPUT_DIR}"
echo "CMake Extra Flags: ${CMAKE_EXTRA_FLAGS}"
echo "========================================"

# Setup oneAPI environment
echo "[1/4] Setting up oneAPI environment..."
source /opt/intel/oneapi/setvars.sh --force

# Use bmg-ocloc wrapper if available (mounted at /workspace/tools/bmg-ocloc)
if [ -x /workspace/tools/bmg-ocloc/ocloc-wrapper ]; then
    ln -sf /workspace/tools/bmg-ocloc/ocloc-wrapper /workspace/bin/ocloc
    export PATH=/workspace/bin:$PATH
    echo "Using bmg-ocloc wrapper as /workspace/bin/ocloc"
fi

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export CMAKE_BUILD_TYPE=Release
export IGC_VISAOptions="-perfmodel"
export IGC_VectorAliasBBThreshold=100000000000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export CC=icx
export CXX=icpx
export CUTLASS_SYCL_PROFILING_ENABLED=ON

echo "Compiler: $(which $CXX)"
$CXX --version

# Copy source to a writable location so we can patch it
echo "[1.5/4] Copying source to writable location..."
cp -a "${SOURCE_DIR}" /workspace/tmp/cutlass-src
SOURCE_DIR="/workspace/tmp/cutlass-src"

# Disable bmg_flash_attention for BMG unless explicitly enabled via env var
# Default: disabled (OOM issue on runner). Set DISABLE_FLASH_ATTENTION=0 to enable.
DISABLE_FLASH_ATTENTION="${DISABLE_FLASH_ATTENTION:-1}"
if [ "${SYCL_TARGET}" = "intel_gpu_bmg_g21" ] && [ "${DISABLE_FLASH_ATTENTION}" != "0" ]; then
    if [ -f "${SOURCE_DIR}/examples/06_bmg_flash_attention/CMakeLists.txt" ]; then
        echo "Disabling 06_bmg_flash_attention for BMG target..."
        cat /dev/null > "${SOURCE_DIR}/examples/06_bmg_flash_attention/CMakeLists.txt"
    else
        echo "Warning: 06_bmg_flash_attention/CMakeLists.txt not found, skipping disable"
    fi
fi

# Configure CMake
echo "[2/4] Configuring CMake..."
mkdir -p "${OUTPUT_DIR}/build" && cd "${OUTPUT_DIR}/build"
cmake "${SOURCE_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
    -DCUTLASS_ENABLE_SYCL=ON \
    -DDPCPP_SYCL_TARGET=${SYCL_TARGET} \
    -DCUTLASS_ENABLE_BENCHMARKS=ON \
    -DCUTLASS_SYCL_RUNNING_CI=ON \
    -DCUTLASS_SYCL_PROFILING_ENABLED=ON \
    ${CMAKE_EXTRA_FLAGS}

NINJA_JOBS="${NINJA_JOBS:-$(nproc)}"

# Build
echo "[3/4] Building with -j${NINJA_JOBS}..."
case "${BUILD_TARGETS}" in
    all)
        echo "Building test/all..."
        ninja -j${NINJA_JOBS} test/all
        echo "Building examples/all..."
        ninja -j${NINJA_JOBS} examples/all
        echo "Building benchmarks/all..."
        ninja -j${NINJA_JOBS} benchmarks/all
        ;;
    test)
        ninja -k0 -j${NINJA_JOBS} test/all
        ;;
    examples)
        ninja -j${NINJA_JOBS} examples/all
        ;;
    benchmarks)
        ninja -j${NINJA_JOBS} benchmarks/all
        ;;
    *)
        echo "Unknown build target: ${BUILD_TARGETS}"
        exit 1
        ;;
esac

echo "[4/4] Build complete!"
echo "Build output at: ${OUTPUT_DIR}/build"

# Fix paths: Docker internal /output/build must be replaced with the actual
# path where the build output will be accessed on the test runner.
REMOTE_BUILD_DIR="${REMOTE_BUILD_DIR:-}"
REMOTE_SOURCE_DIR="${REMOTE_SOURCE_DIR:-}"
if [ -n "${REMOTE_BUILD_DIR}" ]; then
    echo "Fixing build paths: /workspace/output/build -> ${REMOTE_BUILD_DIR}"
    find "${OUTPUT_DIR}/build" \( -name 'DartConfiguration.tcl' -o -name 'CTestTestfile*.cmake' \) \
          -exec sed -i "s|/workspace/output/build|${REMOTE_BUILD_DIR}|g" {} +
    # Fix source paths for benchmarks: config files reference Docker source dir
    if [ -n "${REMOTE_SOURCE_DIR}" ]; then
        echo "Fixing source paths: /workspace/tmp/cutlass-src -> ${REMOTE_SOURCE_DIR}"
        find "${OUTPUT_DIR}/build" -name 'CTestTestfile*.cmake' \
          -exec sed -i "s|/workspace/tmp/cutlass-src|${REMOTE_SOURCE_DIR}|g" {} +
    fi
    mkdir -p "${OUTPUT_DIR}/build/Testing/Temporary"
fi

# Fix permissions: Docker runs as root, but the build server user needs
# read access to rsync build output to the runner. Owner (root) retains
# full access; group/other get read+execute only.
echo "Fixing output permissions..."
chmod -R u+rwX,go+rX "${OUTPUT_DIR}/build"

echo "========================================"
