# Benchmarks

```
cd cutlass-fork/build/
```

## Compiling GEMM benchmarks with CUDA backend
```
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=OFF -DDPCPP_SYCL_TARGET=nvptx64_nvidia_cuda -DDPCPP_SYCL_ARCH=sm_80 -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja cutlass_benchmarks_gemm_cuda
./benchmarks/gemm/cutlass_benchmarks_gemm_cuda --config_file=../benchmarks/device/ampere/input_files/input_gemm.in
```

## Compiling and Running GEMM benchmarks with default configurations with CUDA backend
```
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=OFF -DDPCPP_SYCL_TARGET=nvptx64_nvidia_cuda -DDPCPP_SYCL_ARCH=sm_80 -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja benchmarks_gemm_cuda
```

## Compiling GEMM benchmarks with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja cutlass_benchmarks_gemm_sycl
./benchmarks/gemm/cutlass_benchmarks_gemm --config_file=../benchmarks/device/pvc/input_files/input_gemm.in
```

## Compiling and Running GEMM benchmarks with default configurations with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja benchmarks_gemm_sycl
```

## Compiling Flash Attention v2 benchmarks with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja cutlass_benchmarks_flash_attention
./benchmarks/flash_attention/flash_attention_prefill/cutlass_benchmarks_flash_attention_prefill_xe --config_file=../benchmarks/device/bmg/input_files/input_sglang_flash_attention_prefill_extend_nokvcache.in
./benchmarks/flash_attention/flash_attention_prefill_cachedKV/cutlass_benchmarks_flash_attention_prefill_cachedkv_xe --config_file=../benchmarks/device/bmg/input_files/input_sglang_flash_attention_prefill_extend_kvcache.in
./benchmarks/flash_attention/flash_attention_decode/cutlass_benchmarks_flash_attention_decode_xe --config_file=../benchmarks/device/bmg/input_files/input_sglang_flash_attention_decode_kvcache.in
```

## Compiling and Running Flash Attention v2 benchmarks with default configurations with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja benchmarks_flash_attention
```

## Compiling GDN Attention benchmarks with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_bmg_g21 | intel_gpu_cri
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja cutlass_benchmarks_gdn_xe
# Pick the config matching your target. CRI uses shorter sequence lengths
# (256-4096) to stay within the simulator timeout; BMG sweeps longer ones (up to 128000).
./benchmarks/gdn/cutlass_benchmarks_gdn_xe --config_file=../benchmarks/device/bmg/input_files/input_gdn_bf16.in   # intel_gpu_bmg_g21
./benchmarks/gdn/cutlass_benchmarks_gdn_xe --config_file=../benchmarks/device/cri/input_files/input_gdn_bf16.in   # intel_gpu_cri
```

## Compiling and Running GDN Attention benchmarks with default configurations with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_bmg_g21 | intel_gpu_cri
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja benchmarks_gdn
```

## Compiling and Running all benchmarks with default configurations with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja benchmarks
```