add_custom_target(
  cutlass_test_unit_cute_xe4
  DEPENDS
  cutlass_test_unit_cute_xe4_tma_load
  cutlass_test_unit_cute_xe4_linear_load
  cutlass_test_unit_cute_xe4_linear_store
)

add_custom_target(
  test_unit_cute_xe4
  DEPENDS
  test_unit_cute_xe4_tma_load
)

cutlass_test_unit_add_executable(
  cutlass_test_unit_cute_xe4_tma_load
  tma_load.cpp
)
