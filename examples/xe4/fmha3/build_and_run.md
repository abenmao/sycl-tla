## Build

Build the kernel

```shell
export WORK_ROOT=$HOME/trees/jgs
source ${WORK_ROOT}/DPCPP/setvars.sh

export IGC_DumpToCustomDir=./igc_dump
export IGC_ShaderDumpEnable=1
export IGC_LLVMOptions="-swsb-allocation -dstcache=3 -xe-enable-phi-uniform -xe-enable-texp-overlap"

mkdir build && cd build

cmake .. -GNinja \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCUTLASS_ENABLE_SYCL=ON \
    -DCUTLASS_ENABLE_BENCHMARKS=OFF\
    -DCUTLASS_ENABLE_GTEST_UNIT_TESTS=OFF\
    -DSYCL_INTEL_TARGET=40 \
    -DDPCPP_SYCL_TARGET=intel_gpu_jgs \
    -DCMAKE_CXX_FLAGS="-w"
ninja test_sdpa_bfloat16
```

## Run

Run coral simulator

```shell
export WORK_ROOT=$HOME/trees/jgs
source ${WORK_ROOT}/applications.simulators.gpu.soc.coral-cicd/scripts/functions.sh
coral_update_deps_env
run_coral_sim -r -m umd -device_cfg 1tx1x1x4 -xesim_cfg "-cb_cfg device_info_json_enable true -msglevel verbose -dtrace -cb_cfg stats_enable true tg_interval 100 tg_cfg_filename TG_ArchTarget_xe4.txt -cb_cfg dump_stats_on_eu_thread_busy true staging_dump_stats_on_swtag true"
```

Open another terminal and run kernel

```shell
export WORK_ROOT=$HOME/trees/jgs
source ${WORK_ROOT}/applications.simulators.gpu.soc.coral-cicd/scripts/functions.sh
coral_update_deps_env
umd_driver_env_variables_export
export TbxPort=1234

source ${WORK_ROOT}/DPCPP/setvars.sh

../examples/xe4/fmha/run_tests.sh
```

The profiling and debug info will be dumped in $COBALT_LIBRARY_PATH/fulsim_logs_<timestamp>
