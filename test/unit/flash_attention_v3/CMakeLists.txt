# Copyright (c) 2025 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

set(TEST_EXES "")
set(TEST_RUNS "")

cutlass_test_unit_add_executable(
  cutlass_test_unit_flash_attention_v3_xe4_fmha_fwd
  xe4_fmha_fwd_test.cpp
)
target_include_directories(
  cutlass_test_unit_flash_attention_v3_xe4_fmha_fwd
  PRIVATE
  ${CMAKE_SOURCE_DIR}/examples/xe4/fmha
  ${CMAKE_SOURCE_DIR}/examples/common
)
list(APPEND TEST_EXES cutlass_test_unit_flash_attention_v3_xe4_fmha_fwd)
string(REGEX REPLACE cutlass_ "" xe4_fmha_fwd_test_run cutlass_test_unit_flash_attention_v3_xe4_fmha_fwd)
list(APPEND TEST_RUNS ${xe4_fmha_fwd_test_run})

add_custom_target(
  cutlass_test_unit_flash_attention_v3
  DEPENDS
  ${TEST_EXES}
)

add_custom_target(
  test_unit_flash_attention_v3
  DEPENDS
  ${TEST_RUNS}
)
