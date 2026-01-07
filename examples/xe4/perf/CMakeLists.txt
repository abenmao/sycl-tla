# Include Google Test using CUTLASS cmake integration
include(${CMAKE_SOURCE_DIR}/cmake/googletest.cmake)

# Ensure googletest targets are available
if(NOT TARGET gtest)
    message(FATAL_ERROR "GoogleTest targets not found after including googletest.cmake")
endif()

# Define the target name for the performance test executable
set(PERF_TARGET_NAME gemm_perf)

# Collect all source files
set(PERF_SOURCES
    gemm/gemm_test.cpp
)

# Create the executable using the existing xe4 test infrastructure
add_xe4_test(${PERF_TARGET_NAME} ${PERF_SOURCES})

# Include directories for headers
target_include_directories(${PERF_TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/testFixture
)

# Link Google Test libraries
target_link_libraries(${PERF_TARGET_NAME} PRIVATE
    gtest
    gtest_main
)

# Set target link options for XE4 architecture
target_link_options(${PERF_TARGET_NAME} PUBLIC
    -Xsycl-target-backend=spir64_gen
    -device\ jgs-a0
)
