
## Build and Run xe4 bgemm test

### With /root/dpcpp

```bash
cmake --preset xe4
cmake --build --preset xe4
ctest --preset xesim -R bgemm$ -V
ctest --preset xesim -R conv2d$ -V
ctest --preset xesim -R conv2d_dgrad$ -V
```

### With oneAPI's dpcpp

```bash
source ../drivers.gpu.compute.workloads/simt_workloads/scripts/setvars.sh
cmake .. -DCUTLASS_ENABLE_SYCL=ON -DCUTLASS_ENABLE_BENCHMARKS=OFF -DCUTLASS_ENABLE_GTEST_UNIT_TESTS=OFF -DDPCPP_SYCL_TARGET=intel_gpu_xe4
make -j
```
