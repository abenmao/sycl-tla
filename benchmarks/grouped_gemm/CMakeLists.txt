# Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
# Copyright (C) 2025 Intel Corporation, All rights reserved.
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

# Pass these configuration files for the CI
if(CUTLASS_TEST_FOR_CRI AND SYCL_TARGET_INTEL_GPU_CRI)
  set(CONFIG_FILE_CRI_GROUPED --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/input_small_grouped_gemm.in)
  set(CONFIG_FILE_CRI_MOE --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/input_small_moe_gemm.in)
  # ByteDance real-scenario LLM shapes from xpu-perf/micro_perf workloads
  set(CONFIG_FILE_BYTEDANCE --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/bytedance/input_small_grouped_gemm.in)
else()
  set(CONFIG_FILE_BMG_GROUPED --config_file=${CUTLASS_DIR}/benchmarks/device/bmg/input_files/input_small_grouped_gemm.in)
endif()

# MoE GEMM benchmark reuses the kernel/scheduler headers from example 12.
set(CUTLASS_MOE_EXAMPLE_DIR ${CUTLASS_DIR}/examples/12_xe20_moe_gemm_cute_interface)

cutlass_benchmark_add_suite(cutlass_benchmarks_grouped_gemm)

if(CUTLASS_ENABLE_SYCL)

# TODO: support grouped gemm benchmark on more platforms
if(DEFINED SYCL_INTEL_TARGET AND SYCL_TARGET_INTEL_GPU_CRI)

cutlass_benchmark_add_executable(
    cutlass_benchmarks_grouped_gemm_sycl
    main.cpp
    SUITE cutlass_benchmarks_grouped_gemm
    TEST_COMMAND_OPTIONS
    CONFIG_FILE_CRI_GROUPED
    CONFIG_FILE_CRI_MOE
    CONFIG_FILE_BYTEDANCE
)

target_include_directories(cutlass_benchmarks_grouped_gemm_sycl PRIVATE ${CUTLASS_MOE_EXAMPLE_DIR})

# Enable Xe-specific block2D alignment asserts in MXFP scale load (xe_common_blockscaled_mxfp.hpp)
target_compile_definitions(cutlass_benchmarks_grouped_gemm_sycl PRIVATE CUTE_ENABLE_XE_BLOCK_2D_ASSERT)

endif()

endif()
