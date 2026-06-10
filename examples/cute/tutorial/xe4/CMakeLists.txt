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

set(AMMA_TARGET xe4_amma)
add_xe4_test(${AMMA_TARGET} amma_xe4.cpp)
target_include_directories(${AMMA_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${AMMA_TARGET} PUBLIC
	-Xsycl-target-backend=spir64_gen
	-device\ jgs-a0)

set(AMMA_ADMA_TARGET xe4_amma_adma)
add_xe4_test(${AMMA_ADMA_TARGET} amma_adma_xe4.cpp)
target_include_directories(${AMMA_ADMA_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${AMMA_ADMA_TARGET} PUBLIC
	-Xsycl-target-backend=spir64_gen
	-device\ jgs-a0)

# Drive the xe4_amma_adma tutorial executable over a matrix of input shapes
# and the supported transpose configuration so the kernel is exercised in CI
# rather than only via manual runs. The tutorial accepts:
#     <executable> M N K transA transB
# with defaults M=N=K=256, transA='T', transB='N'.
#
# Kernel constraints (must hold for every shape below):
#   * BLK_M = BLK_N = BLK_K = 128, no boundary handling, so M, N, K must each
#     be positive multiples of 128.
#   * K_PIPE_MAX = 3 pipeline stages are statically allocated in SLM. The
#     prologue may prefetch past the K extent when K < 3 * BLK_K, but the
#     MMA loop only consumes the actual k_tile range so any speculatively
#     loaded data is harmless. Therefore K only needs to be a positive
#     multiple of BLK_K = 128.
#   * The host dispatch in gemm() currently only implements transA='T' and
#     transB='N'; add cases here for additional layouts once they are
#     implemented in amma_adma_xe4.cpp.
function(_add_xe4_amma_adma_shape_test m n k)
  set(_test_name ctest_examples_${AMMA_ADMA_TARGET}_m${m}_n${n}_k${k}_TN)
  add_test(
    NAME ${_test_name}
    COMMAND $<TARGET_FILE:${AMMA_ADMA_TARGET}> ${m} ${n} ${k} T N
  )
  set_tests_properties(
    ${_test_name}
    PROPERTIES
      FAIL_REGULAR_EXPRESSION "FAILED;Test Failed"
      PASS_REGULAR_EXPRESSION "Verification: PASSED"
  )
endfunction()

# Small-K cases (K < K_PIPE_MAX * BLK_K = 384) to exercise the prologue
# over-prefetch path.
_add_xe4_amma_adma_shape_test(128 128 128)
_add_xe4_amma_adma_shape_test(256 256 128)
_add_xe4_amma_adma_shape_test(128 128 256)
_add_xe4_amma_adma_shape_test(256 256 256)
_add_xe4_amma_adma_shape_test(1024 1024 1024)

# Square problems at and above the pipeline depth.
_add_xe4_amma_adma_shape_test(128 128 384)
_add_xe4_amma_adma_shape_test(256 256 384)
_add_xe4_amma_adma_shape_test(256 256 512)
_add_xe4_amma_adma_shape_test(512 512 512)

# Rectangular M / N to exercise both M- and N-tile loop bounds.
_add_xe4_amma_adma_shape_test(128 256 128)
_add_xe4_amma_adma_shape_test(256 128 128)
_add_xe4_amma_adma_shape_test(128 256 384)
_add_xe4_amma_adma_shape_test(256 128 384)
_add_xe4_amma_adma_shape_test(128 512 384)
_add_xe4_amma_adma_shape_test(512 128 384)

# Varying K to vary K_TILE_MAX (1, 2, 3, 4, 6, 8 tiles).
_add_xe4_amma_adma_shape_test(256 256 768)
_add_xe4_amma_adma_shape_test(256 256 1024)

set(ADMA_LINEAR_COPY_TARGET xe4_adma_linear_copy)
add_xe4_test(${ADMA_LINEAR_COPY_TARGET} adma_linear_copy_xe4.cpp)
target_include_directories(${ADMA_LINEAR_COPY_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${ADMA_LINEAR_COPY_TARGET} PUBLIC
	-Xsycl-target-backend=spir64_gen
	-device\ jgs-a0)

set(ADMA_LINEAR_REDUCE_TARGET xe4_adma_linear_reduce)
add_xe4_test(${ADMA_LINEAR_REDUCE_TARGET} adma_linear_reduce_xe4.cpp)
target_include_directories(${ADMA_LINEAR_REDUCE_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${ADMA_LINEAR_REDUCE_TARGET} PUBLIC
    -Xsycl-target-backend=spir64_gen
        -device\ jgs-a0)

set(ADMA_CLUSTER_LINEAR_COPY_TARGET xe4_adma_cluster_linear_copy)
add_xe4_test(${ADMA_CLUSTER_LINEAR_COPY_TARGET} adma_cluster_linear_copy_xe4.cpp)
target_include_directories(${ADMA_CLUSTER_LINEAR_COPY_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${ADMA_CLUSTER_LINEAR_COPY_TARGET} PUBLIC
	-Xsycl-target-backend=spir64_gen
	-device\ jgs-a0)

set(ADMA_CLUSTER_LINEAR_REMOTE_COPY_TARGET xe4_adma_cluster_linear_remote_copy)
add_xe4_test(${ADMA_CLUSTER_LINEAR_REMOTE_COPY_TARGET} adma_cluster_linear_remote_copy_xe4.cpp)
target_include_directories(${ADMA_CLUSTER_LINEAR_REMOTE_COPY_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${ADMA_CLUSTER_LINEAR_REMOTE_COPY_TARGET} PUBLIC
	-Xsycl-target-backend=spir64_gen
	-device\ jgs-a0)

set(ADMA_ROW_COPY_TARGET xe4_adma_row_copy)
add_xe4_test(${ADMA_ROW_COPY_TARGET} adma_row_copy_xe4.cpp)
target_include_directories(${ADMA_ROW_COPY_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${ADMA_ROW_COPY_TARGET} PUBLIC
    -Xsycl-target-backend=spir64_gen
        -device\ jgs-a0)

set(ADMA_ROW_COPY_TILED_TARGET xe4_adma_row_copy_tiled)
add_xe4_test(${ADMA_ROW_COPY_TILED_TARGET} adma_row_copy_tiled_xe4.cpp)
target_include_directories(${ADMA_ROW_COPY_TILED_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${ADMA_ROW_COPY_TILED_TARGET} PUBLIC
    -Xsycl-target-backend=spir64_gen
        -device\ jgs-a0)

######################## EU_COPY_TARGET ########################################
# Check if the environment variable is specified
if(NOT DEFINED ENV{CUTLASS_DIR})
    # Set a default value if it's not found
    set(CUTLASS_DIR "../../")
    message(STATUS 
			"CUTLASS_DIR not found in environment, using default: ${CUTLASS_DIR}")
else()
    # Use the environment variable value
    set(CUTLASS_DIR $ENV{CUTLASS_DIR})
    message(STATUS "Using CUTLASS_DIR from environment: ${CUTLASS_DIR}")
endif()
set(EU_COPY_TARGET xe4_gemm_eu_copy_matrix_atoms)
add_xe4_test(${EU_COPY_TARGET} gemm_eu_copy_matrix_atoms.cpp)

target_include_directories(
	${EU_COPY_TARGET} PRIVATE 
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/examples/cute/tutorial/xe4/
) 
target_link_options(${EU_COPY_TARGET} PUBLIC
	-Xsycl-target-backend=spir64_gen
	-device\ jgs-a0)

set(TMMA_GEMM_TARGET xe4_gemm_tiled_tmma)
add_xe4_test(${TMMA_GEMM_TARGET} gemm_tiled_tmma.cpp)

target_include_directories(
	${TMMA_GEMM_TARGET} PRIVATE 
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/examples/cute/tutorial/xe4/
) 
target_link_options(${TMMA_GEMM_TARGET} PUBLIC
	-Xsycl-target-backend=spir64_gen
	-device\ jgs-a0)

add_executable(xe4_gemm_asymmetric gemm_epilogue_asymmetric_adma_amma.cpp)
add_test(
  NAME ctest_examples_test_xe4_gemm_asymmetric
  COMMAND $<TARGET_FILE:xe4_gemm_asymmetric>
)
add_dependencies(cutlass_examples xe4_gemm_asymmetric)

# Compile options  PRIVATE 
target_compile_options(xe4_gemm_asymmetric PRIVATE -fsycl -fsycl-targets=pisa)
target_link_libraries(xe4_gemm_asymmetric PRIVATE -lur_loader)
target_link_options(xe4_gemm_asymmetric PRIVATE -fsycl -fsycl-targets=pisa)
target_compile_definitions(xe4_gemm_asymmetric PRIVATE
    CUTLASS_ENABLE_SYCL=1
    SYCL_INTEL_TARGET=40
    SYCL_INTEL_XE4_TARGET
)
# Include dirs
target_include_directories(xe4_gemm_asymmetric PRIVATE
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/examples/cute/tutorial/xe4/
)

add_xe4_test(xe4_gemm_non_asymmetric gemm_epilogue_adma_amma.cpp)
# Compile options  PRIVATE 
target_compile_options(xe4_gemm_non_asymmetric PRIVATE -fsycl -fsycl-targets=pisa)
target_link_options(xe4_gemm_non_asymmetric PRIVATE -fsycl -fsycl-targets=pisa)
target_compile_definitions(xe4_gemm_non_asymmetric PRIVATE
    CUTLASS_ENABLE_SYCL=1
    SYCL_INTEL_TARGET=40
    SYCL_INTEL_XE4_TARGET
)
# Include dirs
target_include_directories(xe4_gemm_non_asymmetric PRIVATE
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/examples/cute/tutorial/xe4/
)
################################################################################


set(AMMA_ADMA_BLOCKSCALE_TARGET xe4_amma_adma_blockscaled_gemm)
add_xe4_test(${AMMA_ADMA_BLOCKSCALE_TARGET} amma_adma_xe4_blockscaled_gemm.cpp)
target_include_directories(${AMMA_ADMA_BLOCKSCALE_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${AMMA_ADMA_BLOCKSCALE_TARGET} PUBLIC
       -Xsycl-target-backend=spir64_gen
       -device\ jgs-a0)

set(AMMA_ADMA_CLUSTER_TARGET xe4_amma_adma_cluster)
add_xe4_test(${AMMA_ADMA_CLUSTER_TARGET} amma_adma_xe4_cluster.cpp)
target_include_directories(${AMMA_ADMA_CLUSTER_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${AMMA_ADMA_CLUSTER_TARGET} PUBLIC
	-Xsycl-target-backend=spir64_gen
	-device\ jgs-a0)

set(AMMA_ADMA_BLOCKSCALE_CLUSTER_TARGET xe4_amma_adma_blockscaled_gemm_cluster)
add_xe4_test(${AMMA_ADMA_BLOCKSCALE_CLUSTER_TARGET} amma_adma_xe4_blockscaled_gemm_cluster.cpp)
target_include_directories(${AMMA_ADMA_BLOCKSCALE_CLUSTER_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${AMMA_ADMA_BLOCKSCALE_CLUSTER_TARGET} PUBLIC
       -Xsycl-target-backend=spir64_gen
       -device\ jgs-a0)
