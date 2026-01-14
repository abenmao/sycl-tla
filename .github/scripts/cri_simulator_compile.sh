#!/bin/bash

# --- launch simulator ---
tmux new-session -d -s session_simulator -n "Simulator"
tmux send-keys -t session_simulator:0 \
    "echo 'Starting simulator...'; \
     cd /opt/intel/crisim/ && \
     sudo -S ./runsim.sh 6117" Enter

tmux new-session -d -s session_compile -n "Compile"

tmux send-keys -t session_compile:0 \
    "mkdir -p build && \
     cd build" Enter

# --- waiting for the simulator ---
tmux send-keys -t session_compile:0 "sleep 5" Enter

# --- setup environment ---
tmux send-keys -t session_compile:0 \
    "source /opt/intel/crisim/env.sh 6117 && \
     source /opt/intel/oneapi/setvars.sh && \
     export ONEAPI_DEVICE_SELECTOR=level_zero:gpu && \
     export CMAKE_BUILD_TYPE=Release && \
     export IGC_ExtraOCLOptions=\"-cl-intel-512-GRF-per-thread\" && \
     export CUTLASS_SYCL_PROFILING_ENABLED=ON && \
     export CC=icx && \
     export CXX=icpx" Enter

# --- config cmake ---
tmux send-keys -t session_compile:0 \
    "cmake .. -G Ninja \
        -DCUTLASS_ENABLE_SYCL=ON \
        -DDPCPP_SYCL_TARGET=intel_gpu_cri \
        -DCUTLASS_ENABLE_BENCHMARKS=ON \
        -DCUTLASS_SYCL_RUNNING_CI=ON \
        -DCUTLASS_SYCL_PROFILING_ENABLED=ON \
        -DCUTLASS_TEST_FOR_CRI=ON \
        > cmake_build.log 2>&1; \
     CMAKE_EXIT_CODE=\$?; \
     echo \$CMAKE_EXIT_CODE > test_exit_code.log; \
     if [ \$CMAKE_EXIT_CODE -ne 0 ]; then \
         touch finish_compilation_job.log; \
         exit \$CMAKE_EXIT_CODE; \
     fi" Enter

# --- build library ---
tmux send-keys -t session_compile:0 \
    "cmake --build . -j10 > build.log 2>&1; \
     BUILD_EXIT_CODE=\$?; \
     echo \$BUILD_EXIT_CODE > test_exit_code.log; \
     if [ \$BUILD_EXIT_CODE -ne 0 ]; then \
         touch finish_compilation_job.log; \
         exit \$BUILD_EXIT_CODE; \
     fi" Enter

# --- test example ---
tmux send-keys -t session_compile:0 \
    "ctest -V -R '^(ctest_examples_03_bmg_gemm_streamk|ctest_examples_04_bmg_grouped_gemm|ctest_examples_05_bmg_gemm_with_epilogue_relu|ctest_examples_06_bmg_prefill_attention_cachedkv_hdim64|ctest_examples_08_bmg_gemm_f8|ctest_examples_cute_tutorial_tiled_copy|ctest_examples_cute_tutorial_bmg|ctest_examples_12_xe35_block_scaled_gemm_e2m1|ctest_examples_13_xe35_block_scaled_grouped_gemm_e5m2|ctest_examples_06_xe_fmha_fwd_decode_mx_float_e4m3_t_hdim64|ctest_examples_06_xe_fmha_fwd_decode_mx_float_e2m1_t_hdim64)$' --output-on-failure > test_example.log 2>&1; \
     TEST_EXAMPLE_EXIT_CODE=\$?; \
     echo \$TEST_EXAMPLE_EXIT_CODE > test_exit_code.log; \
     if [ \$TEST_EXAMPLE_EXIT_CODE -ne 0 ]; then \
         touch finish_compilation_job.log; \
         exit \$TEST_EXAMPLE_EXIT_CODE; \
     fi" Enter

# --- test UT ---
tmux send-keys -t session_compile:0 \
    "ctest -V -R '^(ctest_unit_flash_attention_decode_h128_xe|ctest_unit_cute_core|ctest_unit_flash_attention_prefill_fp8e4m3_fp32_fp32_h96_xe|ctest_unit_flash_attention_prefill_fp8e4m3_fp32_fp8e4m3_h96_xe|ctest_unit_gemm_device_tensorop_cooperative_xe|ctest_unit_gemm_device_tensorop_epilogue_fusion_xe|ctest_unit_gemm_device_mixed_input_tensorop_xe|ctest_unit_gemm_device_tensorop_xe_group_gemm|ctest_unit_gemm_device_mixed_dtype_tensorop_xe_group_gemm)$' --output-on-failure > test_ut.log 2>&1; \
     TEST_UT_EXIT_CODE=\$?; \
     echo \$TEST_UT_EXIT_CODE > test_exit_code.log; \
     touch finish_compilation_job.log; \
     if [ \$TEST_UT_EXIT_CODE -ne 0 ]; then \
         exit \$TEST_UT_EXIT_CODE; \
     fi" Enter