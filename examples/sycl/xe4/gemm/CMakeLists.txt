# Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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


set(TARGET_NAME xe4_gemm)
add_xe4_test(${TARGET_NAME} gemm.cpp)
target_compile_definitions(${TARGET_NAME} PRIVATE AMMA_GENERATED)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${TARGET_NAME} PUBLIC "-Xs;\"-xe-set-abarrier-arrive-lmc -disable-branch-fold\"")

set(GEMM_CLUSTER_TARGET xe4_gemm_cluster)
add_xe4_test(${GEMM_CLUSTER_TARGET} gemm_cluster.cpp)
target_compile_definitions(${GEMM_CLUSTER_TARGET} PRIVATE AMMA_GENERATED)
target_include_directories(${GEMM_CLUSTER_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(${GEMM_CLUSTER_TARGET} PUBLIC "-Xs;\"-xe-set-abarrier-arrive-lmc -disable-branch-fold\"")

generate_amma_header(
  ${TARGET_NAME}_helper
  "--mma_type" "mma"
  "--shape" "128x128x128;256x256x128;256x512x128"
  "--dtype" "f32_f32_f16_f16;f16_f32_f16_f16;f32_f32_bf8_bf8;bf16_f32_bf8_bf8"
)
add_dependencies(${TARGET_NAME} ${TARGET_NAME}_helper)
add_dependencies(${GEMM_CLUSTER_TARGET} ${TARGET_NAME}_helper)
