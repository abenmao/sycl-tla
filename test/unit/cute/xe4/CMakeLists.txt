# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025 INTEL CORPORATION. All rights reserved.
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

add_custom_target(
  cutlass_test_unit_cute_xe4
  DEPENDS
  cutlass_test_unit_cute_xe4_reduce_exp
  cutlass_test_unit_cute_xe4_adma_load_store
  cutlass_test_unit_cute_xe4_ldsm_load
  cutlass_test_unit_cute_xe4_ldsm_adma
  cutlass_test_unit_cute_xe4_ldsm_store
  cutlass_test_unit_cute_xe4_ldsm_uaov
  cutlass_test_unit_cute_xe4_ldsm_adma_aov
  cutlass_test_unit_cute_xe4_ldsm_adma_uaov
  cutlass_test_unit_cute_xe4_ldsm_red
  cutlass_test_unit_cute_xe4_eu_slm_copy
  cutlass_test_unit_cute_xe4_adma_prefetch
  cutlass_test_unit_cute_xe4_adma_reduce
  cutlass_test_unit_cute_xe4_ldsm_epilogue
  cutlass_test_unit_cute_xe4_ldsm_epilogue_mma
)

add_custom_target(
  test_unit_cute_xe4
  DEPENDS
  test_unit_cute_xe4_reduce_exp
  test_unit_cute_xe4_adma_load_store
  test_unit_cute_xe4_ldsm_load
  test_unit_cute_xe4_ldsm_adma
  test_unit_cute_xe4_ldsm_store
  test_unit_cute_xe4_ldsm_uaov
  test_unit_cute_xe4_ldsm_adma_aov
  test_unit_cute_xe4_ldsm_adma_uaov
  test_unit_cute_xe4_ldsm_red
  test_unit_cute_xe4_adma_prefetch
  test_unit_cute_xe4_adma_reduce
  test_unit_cute_xe4_ldsm_epilogue
  test_unit_cute_xe4_ldsm_epilogue_mma
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_reduce_exp
  tensor_algs.cpp
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_adma_load_store
  adma_load_store.cpp
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_ldsm_load
  ldsm_load.cpp
)
cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_ldsm_adma
  ldsm_adma.cpp
)
cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_ldsm_store
  ldsm_store.cpp
)
cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_ldsm_uaov
  ldsm_uaov.cpp
)
cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_ldsm_adma_aov
  ldsm_adma_aov.cpp
)
cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_ldsm_adma_uaov
  ldsm_adma_uaov.cpp
)
cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_ldsm_red
  ldsm_red.cpp
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_eu_slm_copy
  slm_copy_test.cpp
)
# ADMA prefetch unit tests — validates async_tensor_prefetch + load roundtrip
cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_adma_prefetch
  adma_prefetch.cpp
)
# ADMA store-reduce unit tests — validates async_tensor_fred/ired (SLM → gmem reduction)
# Note: These tests require real hardware; TBX simulator does not support
# send.dma.tensor.reduce_l2g instruction variant.
cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_adma_reduce
  adma_reduce.cpp
)
cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_ldsm_epilogue
  ldsm_epilogue.cpp
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_ldsm_epilogue_mma
  ldsm_epilogue_mma.cpp
)
