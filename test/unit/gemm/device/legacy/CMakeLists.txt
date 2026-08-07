# Copyright (c) 2025 INTEL CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if(CUTLASS_ENABLE_SYCL)
  cutlass_test_unit_add_executable(
    cutlass_test_unit_gemm_device_tensorop_xe_legacy
    xe_gemm_universal_fp8_fp8_fp32_tensor_op_f32.cpp
    EXTRA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

  # cutlass_test_unit_gemm_device_tensorop_cooperative_xe_legacy:
  # All cooperative legacy test sources are disabled:
  # - xe_gemm_fp16_fp16_f32_ptr_array_cooperative.cpp uses IntelXeXMX16Group (no CollectiveBuilder support)
  # - xe_gemm_s8_s8_s32_tensor_op_s32_cooperative.cpp (TODO: fix, same as parent)
  # - xe_gemm_tf32_tf32_fp32_tensor_op_fp32_cooperative.cpp (TODO: fix, same as parent)

  cutlass_test_unit_add_executable(
    cutlass_test_unit_gemm_device_tensorop_epilogue_fusion_xe_legacy
    xe_gemm_bf16_bf16_fp32_tensor_op_fp32_evt.cpp
    EXTRA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

  add_custom_target(
    cutlass_test_unit_gemm_device_legacy
    DEPENDS
    cutlass_test_unit_gemm_device_tensorop_xe_legacy
    cutlass_test_unit_gemm_device_tensorop_epilogue_fusion_xe_legacy
  )
endif()
