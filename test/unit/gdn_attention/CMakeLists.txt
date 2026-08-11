# Copyright (C) 2026 Intel Corporation, All rights reserved.
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

# The GDN kernels, launcher, and shared GdnRunner are header-only under
# applications/gdn_attention/ (the launcher is an inline template instantiated
# in the test_gdn_chunkwise_*.cpp files). The CUTLASS_APPLICATIONS_DIR include
# path is added automatically by cutlass_test_unit_add_executable, so the
# runner is reached as "gdn_attention/gdn_runner.hpp" with no extra include dir.
set(CUTLASS_APPLICATIONS_DIR ${CUTLASS_DIR}/applications)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gdn_attention_chunkwise_shapes
  test_gdn_chunkwise_shapes.cpp
  EXTRA_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_gdn_attention_chunkwise_variants
  test_gdn_chunkwise_variants.cpp
  EXTRA_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(
  cutlass_test_unit_gdn_attention
  DEPENDS
  cutlass_test_unit_gdn_attention_chunkwise_shapes
  cutlass_test_unit_gdn_attention_chunkwise_variants
)

add_custom_target(
  test_unit_gdn_attention
  DEPENDS
  test_unit_gdn_attention_chunkwise_shapes
  test_unit_gdn_attention_chunkwise_variants
)
