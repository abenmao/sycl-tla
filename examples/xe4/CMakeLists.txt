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

set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/utils)
set(MMA_GENERATOR ${CMAKE_CURRENT_SOURCE_DIR}/utils/generator/gen_mma.py)

function(generate_amma_header TARGET_NAME)
  set(AMMA_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/generated_headers/xe4_async_gmma.hpp)
  set(GEN_ARGS ${ARGN} --output ${AMMA_HEADER})

  add_custom_command(
    OUTPUT ${AMMA_HEADER}
    COMMAND ${Python3_EXECUTABLE} ${MMA_GENERATOR} ${GEN_ARGS}
    DEPENDS ${MMA_GENERATOR}
    COMMENT "Generating header file with ${TARGET_NAME}")

  add_custom_target(${TARGET_NAME} ALL DEPENDS ${AMMA_HEADER})
endfunction()

function(add_xe4_test TARGET_NAME SRCS)
  cutlass_example_add_executable(${TARGET_NAME} ${SRCS} ${ARGN})
  target_include_directories(${TARGET_NAME} PRIVATE ${INCLUDE_DIRS})
  target_link_libraries(${TARGET_NAME} PRIVATE gtest gtest_main)
  target_link_libraries(${TARGET_NAME} PRIVATE -lur_loader)
  set_tests_properties(ctest_examples_${TARGET_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION "Test Failed")
endfunction()

add_subdirectory(gemm)
add_subdirectory(fmha3)
add_subdirectory(fmha4)
# add_subdirectory(conv2d)
add_subdirectory(perf)