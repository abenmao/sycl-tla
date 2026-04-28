# Include Google Test using CUTLASS cmake integration
include(${CMAKE_SOURCE_DIR}/cmake/googletest.cmake)

# Ensure googletest targets are available
if(NOT TARGET gtest)
    message(FATAL_ERROR "GoogleTest targets not found after including googletest.cmake")
endif()

add_xe4_test(gemm_perf gemm/gemm_test.cpp)
add_xe4_test(xe4_gemm_cluster_perf gemm_cluster/gemm_cluster_test.cpp)
add_xe4_test(xe4_gemm_cluster_simple_perf gemm_cluster_simple/gemm_cluster_simple_perf.cpp)

# Include directories for headers
target_include_directories(gemm_perf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/testFixture
)

target_include_directories(xe4_gemm_cluster_perf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm_cluster
    ${CMAKE_CURRENT_SOURCE_DIR}/testFixture
)

target_include_directories(xe4_gemm_cluster_simple_perf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm_cluster_simple
    ${CMAKE_CURRENT_SOURCE_DIR}/testFixture
)

# Link Google Test libraries
target_link_libraries(gemm_perf PRIVATE
    gtest
    gtest_main
)

target_link_libraries(xe4_gemm_cluster_perf PRIVATE
    gtest
    gtest_main
)

# Set target link options for XE4 architecture
target_link_options(gemm_perf PUBLIC
    -Xsycl-target-backend=spir64_gen
    -device\ jgs-a0
)

target_link_options(xe4_gemm_cluster_perf PUBLIC
    -Xsycl-target-backend=spir64_gen
    -device\ jgs-a0
)

# FMHA3 source directory — kernel/ and collective/ headers are reused from here
set(FMHA3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../fmha3)

# fmha_perf: Flash Attention perf tests using kernel/collective from fmha3
cutlass_example_add_executable(fmha_perf flash_attention/fmha_test.cpp)
target_include_directories(fmha_perf PRIVATE ${INCLUDE_DIRS})
target_compile_options(fmha_perf PUBLIC -fsycl-targets=intel_gpu_jgs)
target_link_options(fmha_perf PUBLIC -fsycl-targets=intel_gpu_jgs)
target_link_options(fmha_perf PUBLIC
    -Xs "-options -ze-intel-xe-features=+set-abarrier-arrive-lmc,+disable-misched,+total-grf-num-96,+null-dst"
)
target_link_libraries(fmha_perf PRIVATE gtest gtest_main)
target_include_directories(fmha_perf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/flash_attention
    ${CMAKE_CURRENT_SOURCE_DIR}/testFixture
    ${FMHA3_DIR}
)
