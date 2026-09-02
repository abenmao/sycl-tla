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

# Dual GEMM benchmark pulls in headers from applications/dual_gemm.
set(CUTLASS_APPLICATIONS_DIR ${CUTLASS_DIR}/applications)

if(SYCLTLA_TARGET_XESIM AND SYCL_TARGET_INTEL_GPU_CRI)
  set(CONFIG_FILE_CRI_SMALL --config_file=${CUTLASS_DIR}/benchmarks/config_files/00_gemm/cri/small.in)
  # test1 real-scenario LLM shapes from xpu-perf/micro_perf workloads
  set(CONFIG_FILE_TEST1 --config_file=${CUTLASS_DIR}/benchmarks/config_files/00_gemm/cri/test1/small.in)
  set(CONFIG_FILE_GEMV --config_file=${CUTLASS_DIR}/benchmarks/config_files/00_gemm/cri/gemv_small.in)
  set(GEMM_CONFIG_ARGS CONFIG_FILE_CRI_SMALL CONFIG_FILE_TEST1 CONFIG_FILE_GEMV)
else()
  set(CONFIG_FILE_BMG_SMALL --config_file=${CUTLASS_DIR}/benchmarks/config_files/00_gemm/bmg/small.in)
  set(GEMM_CONFIG_ARGS CONFIG_FILE_BMG_SMALL)
endif()

cutlass_benchmark_add_suite(cutlass_benchmarks_gemm)

if(CUTLASS_ENABLE_SYCL)

if(SYCL_INTEL_TARGET)

# ---------------------------------------------------------------------------
# Per-file OBJECT libraries: each benchmark .cpp compiles exactly once.
# The combined executable links all of them; per-type executables link only
# their own OBJECT library — no duplicate compilation.
# main.cpp is also an OBJECT library so it compiles exactly once and is
# shared by the combined executable and all per-type executables.
# ---------------------------------------------------------------------------
set(_GEMM_BENCH_SRCS
    benchmarks_sycl_bf16.cpp
    benchmarks_sycl_fp16.cpp
    benchmarks_sycl_fp32.cpp
    benchmarks_sycl_tf32.cpp
    benchmarks_sycl_e4m3.cpp
    benchmarks_sycl_e4m3_block_scaled.cpp
    benchmarks_sycl_e5m2.cpp
    benchmarks_sycl_e5m2_block_scaled.cpp
    benchmarks_sycl_e2m1.cpp
    benchmarks_sycl_e2m1_block_scaled.cpp
)

# main.cpp OBJECT library — compiled once, linked into all executables.
add_library(_gemm_main_obj OBJECT main.cpp)
target_link_libraries(_gemm_main_obj
    PRIVATE CUTLASS cutlass_tools_util_includes benchmark::benchmark)
target_include_directories(_gemm_main_obj
    PRIVATE ${CUTLASS_APPLICATIONS_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(_gemm_main_obj
    PRIVATE CUTLASS_BENCHMARK_SYCL_TARGET="${DPCPP_SYCL_TARGET}")
if(TARGET onemkl_project)
 add_dependencies(_gemm_main_obj onemkl_project)
endif()
add_sycl_to_target(TARGET _gemm_main_obj)

# Per-file OBJECT libraries for each benchmark .cpp.
set(_ALL_BENCH_OBJ_LIBS)
foreach(_src IN LISTS _GEMM_BENCH_SRCS)
  string(REGEX REPLACE "^benchmarks_sycl_" "" _stem "${_src}")
  string(REGEX REPLACE "\\.cpp$" "" _stem "${_stem}")

  add_library(_gemm_obj_${_stem} OBJECT ${_src})
  target_link_libraries(_gemm_obj_${_stem}
      PRIVATE CUTLASS cutlass_tools_util_includes benchmark::benchmark)
  target_include_directories(_gemm_obj_${_stem}
      PRIVATE ${CUTLASS_APPLICATIONS_DIR})
  if(TARGET onemkl_project)
    add_dependencies(_gemm_obj_${_stem} onemkl_project)
  endif()
  add_sycl_to_target(TARGET _gemm_obj_${_stem})

  list(APPEND _ALL_BENCH_OBJ_LIBS _gemm_obj_${_stem})
endforeach()

# Combined executable (built by `ninja all`): shared main.o + all benchmark .o.
cutlass_benchmark_add_executable(
    cutlass_benchmarks_gemm_sycl
    $<TARGET_OBJECTS:_gemm_main_obj>
    SUITE cutlass_benchmarks_gemm
    TEST_COMMAND_OPTIONS
    ${GEMM_CONFIG_ARGS}
)
foreach(_lib IN LISTS _ALL_BENCH_OBJ_LIBS)
  target_link_libraries(cutlass_benchmarks_gemm_sycl PRIVATE ${_lib})
endforeach()

# ---------------------------------------------------------------------------
# Per-type executables (EXCLUDE_FROM_ALL — NOT built by `ninja all`).
# Usage: `ninja benchmarks_gemm_sycl_bf16` → produces a runnable binary
# containing only bf16 benchmarks.  Reuses main.o and the single type .o —
# zero recompilation if those objects already exist.
# ---------------------------------------------------------------------------
foreach(_src IN LISTS _GEMM_BENCH_SRCS)
  string(REGEX REPLACE "^benchmarks_sycl_" "" _stem "${_src}")
  string(REGEX REPLACE "\\.cpp$" "" _stem "${_stem}")

  add_executable(benchmarks_gemm_sycl_${_stem} EXCLUDE_FROM_ALL
      $<TARGET_OBJECTS:_gemm_main_obj>
      $<TARGET_OBJECTS:_gemm_obj_${_stem}>)
  target_link_libraries(benchmarks_gemm_sycl_${_stem}
      PRIVATE CUTLASS cutlass_tools_util_includes benchmark::benchmark)
  target_include_directories(benchmarks_gemm_sycl_${_stem}
      PRIVATE ${CUTLASS_APPLICATIONS_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
  add_sycl_to_target(TARGET benchmarks_gemm_sycl_${_stem})
endforeach()

else()
cutlass_benchmark_add_executable(
    cutlass_benchmarks_gemm_sycl
    main.cpp
    SUITE cutlass_benchmarks_gemm
)
endif()

# Use the SYCL target passed via -DDPCPP_SYCL_TARGET as the default benchmark
# name, so benchmark output is tagged with the target without needing --bm_name.
target_compile_definitions(cutlass_benchmarks_gemm_sycl
    PRIVATE CUTLASS_BENCHMARK_SYCL_TARGET="${DPCPP_SYCL_TARGET}")

else()

cutlass_benchmark_add_executable(
    cutlass_benchmarks_gemm_cuda
    main.cu
    SUITE cutlass_benchmarks_gemm
)

endif()