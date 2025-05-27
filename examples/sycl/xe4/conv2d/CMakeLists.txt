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

find_package(DNNL REQUIRED)

add_xe4_test(xe4_conv2d conv2d.cpp)
target_link_libraries(xe4_conv2d PRIVATE DNNL::dnnl)
target_include_directories(xe4_conv2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(xe4_conv2d PRIVATE AMMA_GENERATED)
target_compile_definitions(xe4_conv2d PRIVATE -DTEST_PERF_NORMAL)
target_link_options(xe4_conv2d PUBLIC "-Xs;\"-xe-set-abarrier-arrive-lmc -disable-branch-fold -xe-enable-phi-uniform -misched-prera-direction=topdown\"")

add_xe4_test(xe4_conv2d_dgrad conv2d_dgrad.cpp)
target_link_libraries(xe4_conv2d_dgrad PRIVATE DNNL::dnnl)
target_include_directories(xe4_conv2d_dgrad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(xe4_conv2d_dgrad PRIVATE AMMA_GENERATED)
target_compile_definitions(xe4_conv2d_dgrad PRIVATE -DTEST_WITH_PAD_WITH_DILATION)
target_link_options(xe4_conv2d_dgrad PUBLIC "-Xs;\"-xe-set-abarrier-arrive-lmc -disable-branch-fold -xe-enable-phi-uniform\"")

add_xe4_test(xe4_conv2d_wgrad conv2d_wgrad.cpp)
target_link_libraries(xe4_conv2d_wgrad PRIVATE DNNL::dnnl)
target_include_directories(xe4_conv2d_wgrad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(xe4_conv2d_wgrad PRIVATE AMMA_GENERATED)
target_compile_definitions(xe4_conv2d_wgrad PRIVATE -DTEST_WITH_PAD_WITH_STRIDE_WITH_DILATION)
target_link_options(xe4_conv2d_wgrad PUBLIC "-Xs;\"-xe-set-abarrier-arrive-lmc -disable-branch-fold -xe-enable-phi-uniform\"")

add_xe4_test(xe4_conv2d_int8 conv2d_int8.cpp)
target_link_libraries(xe4_conv2d_int8 PRIVATE DNNL::dnnl)
target_include_directories(xe4_conv2d_int8 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(xe4_conv2d_int8 PRIVATE AMMA_GENERATED)
target_compile_definitions(xe4_conv2d_int8 PRIVATE -DTEST_SMALL_WITH_PAD_WITH_STRIDE)
target_link_options(xe4_conv2d_int8 PUBLIC "-Xs;\"-xe-set-abarrier-arrive-lmc -disable-branch-fold -xe-enable-phi-uniform\"")

generate_amma_header(
  xe4_conv2d_helper
  "--mma_type" "mma"
  "--shape" "128x128x128;256x512x128;256x128x128;256x256x128;128x512x128"
  "--dtype" "f32_f32_f16_f16;f16_f32_f16_f16;s32_s32_s8_s8"
)
add_dependencies(xe4_conv2d xe4_conv2d_helper)
add_dependencies(xe4_conv2d_dgrad xe4_conv2d_helper)
add_dependencies(xe4_conv2d_wgrad xe4_conv2d_helper)
add_dependencies(xe4_conv2d_int8 xe4_conv2d_helper)
