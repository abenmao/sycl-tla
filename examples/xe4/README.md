
## Build and Run xe4 bgemm test

### With /root/dpcpp

```bash
cmake --preset xe4
cmake --build --preset xe4
ctest --preset xesim -R gemm$ -V
ctest --preset xesim -R conv2d$ -V
ctest --preset xesim -R conv2d_dgrad$ -V
```
