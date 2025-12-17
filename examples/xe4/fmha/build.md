```shell
export IGC_LLVMOptions="-swsb-allocation -dstcache=3 -xe-enable-phi-uniform -xe-enable-texp-overlap"
mkdir build && cd build
cmake .. -GNinja \
    -DCUTLASS_ENABLE_SYCL=ON \
    -DCUTLASS_ENABLE_BENCHMARKS=OFF\
    -DCUTLASS_ENABLE_GTEST_UNIT_TESTS=OFF\
    -DSYCL_INTEL_TARGET=40 \
    -DDPCPP_SYCL_TARGET=intel_gpu_jgs \
    -DCMAKE_CXX_FLAGS="-w"
ninja xe4_fmha_fwd
```