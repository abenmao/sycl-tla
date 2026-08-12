# Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Config files for the CI.
if(CUTLASS_TEST_FOR_CRI AND SYCL_TARGET_INTEL_GPU_CRI)
  set(CONFIG_FILE_CRI_GROUPED --config_file=${CUTLASS_DIR}/benchmarks/config_files/01_grouped_gemm/cri/small.in)
  set(CONFIG_FILE_CRI_MOE --config_file=${CUTLASS_DIR}/benchmarks/config_files/01_grouped_gemm/cri/moe_small.in)
  set(CONFIG_FILE_MOE_TEST1_SMALL --config_file=${CUTLASS_DIR}/benchmarks/config_files/01_grouped_gemm/cri/test1/small.in)
else()
  set(CONFIG_FILE_BMG_GROUPED --config_file=${CUTLASS_DIR}/benchmarks/config_files/01_grouped_gemm/bmg/small.in)
endif()

cutlass_benchmark_add_suite(cutlass_benchmarks_grouped_gemm)

if(MKL_INCLUDE)
  include_directories(${MKL_INCLUDE})
endif()

if(CUTLASS_ENABLE_SYCL)

# The hand-written MoE grouped-GEMM kernel (and its low-precision block-scaled
# variants) currently target CRI / Xe3p only.
if(DEFINED SYCL_INTEL_TARGET AND SYCL_TARGET_INTEL_GPU_CRI)

# 8-11 tiles per dtype instead of the full ~39-54 keeps the device image small and the build fast.
# Set -DCUTLASS_MOE_ALL_SHAPES=ON to also compile every other valid tile
# (defines NON_DEFAULT_SHAPES), so pick_tile can pick the optimal tile for any
# problem shape, not just the DEFAULT ones — at a much larger build cost.
option(CUTLASS_MOE_ALL_SHAPES
       "Compile all valid MoE tiles, not just the DEFAULT-shape winners" OFF)
if(CUTLASS_MOE_ALL_SHAPES)
  set(MOE_ALL_SHAPES_DEF NON_DEFAULT_SHAPES)
else()
  set(MOE_ALL_SHAPES_DEF "")
endif()

# The MoE application's full-run diagnostics: per-launch GPU timing (the
# double-buffer launchers return real ms instead of 0.0) plus the
# reference-verify PASSED/FAILED prints and the tile-selection dumps. On by
# default; set -DFULL_RUN_TIMING_AND_VERIFY=OFF for a launch-only build with no
# timing, no reference GEMM and no prints.
option(FULL_RUN_TIMING_AND_VERIFY
       "Enable MoE full-run timing, verification and tile-select prints" ON)
if(FULL_RUN_TIMING_AND_VERIFY)
  set(MOE_FULL_RUN_DEF FULL_RUN_TIMING_AND_VERIFY)
else()
  set(MOE_FULL_RUN_DEF "")
endif()

# Diagnostic prints (field-transform tile-routing dump). Independent of
# FULL_RUN_TIMING_AND_VERIFY so prints can be toggled without changing the
# timing/verify build. On by default.
option(MOE_DEBUG_PRINT "Enable MoE diagnostic prints (tile-routing dump)" ON)
if(MOE_DEBUG_PRINT)
  set(MOE_DEBUG_PRINT_DEF MOE_DEBUG_PRINT)
else()
  set(MOE_DEBUG_PRINT_DEF "")
endif()

# ---------------------------------------------------------------------------
# VARIANT fields: <target_suffix>|<MOE_DTYPE_<TAG>_PART<n> define>
# The suffix names the binary cutlass_benchmarks_moe_gemm_<suffix>_sycl.
# These per-dtype split binaries exist for parallel compilation; real runs use the
# merged all-dtype binary (MOE_BENCH_ALL) below, which has every tile so the
# field-transform can route to double buffer. Only PARTs that still register tiles
# are listed: greedy (one per dtype) + the double-buffer tiles the field-transform
# actually selects (mxfp4 192x256 + 224x256, mxfp8 224x256). bf16/fp8_tensor DB and
# the emptied _p1 parts were removed.
set(MOE_BENCH_VARIANTS
    # -- greedy (one binary per dtype; each holds uniform + dynamic-M tiles) --
    "bf16|MOE_DTYPE_BF16"
    "mxfp8_e4m3|MOE_DTYPE_MXFP8_E4M3"
    "mxfp4_e2m1|MOE_DTYPE_MXFP4_E2M1"
    "fp8_tensor_e4m3|MOE_DTYPE_FP8_TENSOR_E4M3"
    # -- double-buffer (only the tiles the field-transform uses) --
    "mxfp8_e4m3_db|MOE_DTYPE_DOUBLE_BUFFER_MXFP8_E4M3"
    "mxfp4_e2m1_db|MOE_DTYPE_DOUBLE_BUFFER_MXFP4_E2M1")

foreach(VARIANT ${MOE_BENCH_VARIANTS})
  string(REPLACE "|" ";" VPARTS ${VARIANT})
  list(GET VPARTS 0 VSUFFIX)
  list(GET VPARTS 1 VDTYPE)

  cutlass_benchmark_add_executable(
      cutlass_benchmarks_moe_gemm_${VSUFFIX}_sycl
      main.cpp
      ${CUTLASS_DIR}/applications/moe_grouped_gemm/runner/moe_api.cpp
      SUITE cutlass_benchmarks_grouped_gemm
      INCLUDES
      ${CUTLASS_DIR}
      ${CUTLASS_DIR}/applications
      TEST_COMMAND_OPTIONS
      CONFIG_FILE_CRI_MOE
  )
  target_compile_definitions(cutlass_benchmarks_moe_gemm_${VSUFFIX}_sycl
      PRIVATE ${VDTYPE}
              CUTE_ENABLE_XE_BLOCK_2D_ASSERT ${MOE_ALL_SHAPES_DEF}
              ${MOE_FULL_RUN_DEF} ${MOE_DEBUG_PRINT_DEF})
  if(NOT DPCPP_SYCL_TARGET STREQUAL "spir64")
    target_link_options(cutlass_benchmarks_moe_gemm_${VSUFFIX}_sycl PRIVATE
        -Xs "-options \"-igc_opts 'VectorAliasBBThreshold=10000'\"")
  endif()
endforeach()

# ---------------------------------------------------------------------------
# MERGED all-dtype binary, built under the grouped_gemm name so the tooling
# calls it out of the box. -DMOE_BENCH_ALL enables every MOE_DTYPE_<TAG>.
cutlass_benchmark_add_executable(
    cutlass_benchmarks_grouped_gemm_sycl
    main.cpp
    ${CUTLASS_DIR}/applications/moe_grouped_gemm/runner/moe_api.cpp
    SUITE cutlass_benchmarks_grouped_gemm
    INCLUDES
    ${CUTLASS_DIR}
    ${CUTLASS_DIR}/applications
    TEST_COMMAND_OPTIONS
    CONFIG_FILE_CRI_GROUPED
    CONFIG_FILE_CRI_MOE
    CONFIG_FILE_MOE_TEST1_SMALL
)
target_compile_definitions(cutlass_benchmarks_grouped_gemm_sycl
    PRIVATE MOE_BENCH_ALL CUTE_ENABLE_XE_BLOCK_2D_ASSERT ${MOE_ALL_SHAPES_DEF}
            ${MOE_FULL_RUN_DEF} ${MOE_DEBUG_PRINT_DEF})
if(NOT DPCPP_SYCL_TARGET STREQUAL "spir64")
  target_link_options(cutlass_benchmarks_grouped_gemm_sycl PRIVATE
      -Xs "-options \"-igc_opts 'VectorAliasBBThreshold=10000'\"")
endif()

endif()

endif()
