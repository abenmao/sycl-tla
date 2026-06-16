![ALT](https://raw.githubusercontent.com/intel/sycl-tla/main/media/images/gemm-hierarchy-with-epilogue-no-labels.png "Complete CUDA GEMM decomposition")
![ALT](https://raw.githubusercontent.com/intel/sycl-tla/main/media/images/gemm-hierarchy-with-epilogue-no-labels.png "Complete CUDA GEMM decomposition")

# SYCL\* Templates for Linear Algebra (SYCL\*TLA)

**This repository is forked from the NVIDIA CUTLASS repository and extends CUTLASS and CuTe API support to Intel GPUs through SYCL enablement.**
*This project was previously referred to as CUTLASS-SYCL, you may see references to CUTLASS-SYCL in the code and documentation.*
*For SYCL support instructions, refer to the [SYCL build documentation](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/build/building_with_sycl_support.md)*
*For SYCL support instructions, refer to the [SYCL build documentation](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/build/building_with_sycl_support.md)*

*SYCL is a trademark of the Khronos Group Inc, Other names and brands may be claimed as the property of others.*
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/intel/sycl-tla/badge)](https://scorecard.dev/viewer/?uri=github.com/intel/sycl-tla)

SYCL\*TLA is a modular, header‑only C++ template framework for high‑performance 
GEMM, and fused epilogue kernels. It applies hierarchical tiling, composable policy 
abstractions, and efficient data‑movement primitives to build flexible, reusable 
building blocks for dense linear algebra. The SYCL implementation brings those 
optimizations to Intel GPUs with tuned kernels for modern execution units and memory 
hierarchies. It adds mixed‑precision and epilogue fusion pathways designed to 
simplify integrating advanced quantization and post‑processing into custom pipelines.

To support a wide variety of applications, SYCL\*TLA provides extensive
support for mixed-precision computations on Intel hardware, providing
specialized data-movement and multiply-accumulate abstractions for FP64, FP32,
FP16, BF16, 8b floating point types (E5M2 and E4M3 for FP8), narrow integer
types (4 and 8b signed and unsigned integers with support for zero-point
quantization), and mixed-precision operations with tensor-wise, channel-wise,
and group-wise quantization support. SYCL\*TLA demonstrates optimal matrix
multiply operations targeting Intel's programmable, high-throughput execution
units implemented in Intel Data Center GPU Max/Flex Series (Intel Xe
architecture, codename: Ponte-Vecchio) and Intel Arc B580 GPUs.

See the [Quick Start Guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md) to get started quickly.
See the [Quick Start Guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md) to get started quickly.

See the [functionality docs](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/functionality.md) for a more comprehensive
See the [functionality docs](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/functionality.md) for a more comprehensive
list of kernel level features, data types, instructions, and minimum supported by CUTLASS on each GPU
architecture.

This project fast follows NVIDIA CUTLASS releases to ensure parity of APIs and features.

Base NVIDIA CUTLASS Versions for SYCL*TLA releases:
| SYCL*TLA | NVIDIA CUTLASS |
|-----------------|----------|
|0.1| 3.9|
|0.2 | 3.9.2 |
|0.3 | 3.9.2 |
|0.5 | 4.2.0 |
|0.5-cri | 4.2.0 |
|0.6 | 4.2.0 |
|0.6-cri | 4.2.0 |
|0.6-jgs | 4.2.1 |
|0.7 | 4.2.1 |
|0.7-cri | 4.2.1 |
|0.7-jgs | 4.2.1 |
|0.8 | 4.2.1 |
|0.8-cri | 4.2.1 |
|0.8-jgs | 4.2.1 |
|0.9 | 4.2.1 |
|0.9-cri | 4.2.1 |
|0.9-jgs | 4.2.1 |
|0.9.1 | 4.2.1 |
|0.9.1-cri | 4.2.1 |
|0.9.1-jgs | 4.2.1 |

# What's New in SYCL*TLA [0.9.1-jgs](https://github.com/intel-innersource/libraries.ai.cutlass.internal/releases/tag/v0.9.1-jgs)

### Architecture & APIs (XE4)
  - Add XE4 StreamK tile scheduler support ([#502](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/502))
  - Add XE4 async reduction examples ([#512](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/512))
  - Add FMHA4 Load/Store Matrix APIs for epilogue flows ([#534](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/534))
  - Revise XE4 `abarrier` APIs ([#501](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/501))
  - Enable tensor descriptor update for XE4 copy path ([#563](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/563))
  - Add XE4 ADMA row-copy CuTe atoms and tiled-mode tutorial (GMEM <-> SLM) ([#566](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/566))
  - Enhance row-copy API naming and modes (drop `_COLLECTIVE`, add prefetch/completion modes) ([#564](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/564), [#582](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/582))

### GEMM & Flash Attention
  - Enable direct `cute::gemm` for XE4 TMMA using specialized `mma_unpack`; update TMMA example path and tutorials ([#546](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/546))
  - Add Xe4 ColBroadcast and ScaledMM epilogue fusion support ([#521](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/521))
  - Update block-scaled GEMM tutorial to decouple data and scale-factor loading ([#571](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/571))

### Validation & Tooling
  - Add new unit tests for tensor pipe quantize/dequantize APIs with and without MX scaling ([#569](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/569))
  - Refactor XE4 GEMM test configs into compact inline-documented format ([#488](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/488))
  - Use AOT build for PISA target ([#532](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/532))
  - Add `public-main` CI lane and improve branch autosync/secrets workflow ([#596](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/596), [#511](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/511), [#567](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/567))

### Bug Fixes
  - Fix GEMM ADMA+AMMA race-condition hang in example path ([#558](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/558))
  - Fix XE4 ADMA alignment validation for device builds and add runtime Core Matrix alignment validation after ADMA cluster truncation ([#605](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/605), [#581](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/581))
  - Fix scale-factor errors caused by new xesim behavior ([#585](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/585))
  - Fix XE4 LDSM examples that failed in release validation ([#654](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/654))
  - Fix XE4 ADMA linear prefetch/reduction API and tutorial issues (`async_linear_copy` and ADMA copy-path updates) ([#505](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/505))
  - Fix XE4 LDSM copy-path issues and ADMA reduce inline-assembly/test coverage issues ([#499](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/499), [#500](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/500))
  - Resolve XE4 build warnings due to C++20 extensions ([#640](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/640))

# What's New in SYCL*TLA [0.9.1-cri](https://github.com/intel-innersource/libraries.ai.cutlass.internal/releases/tag/v0.9.1-cri)

## [SYCL*TLA 0.9.1-cri](https://github.com/intel-innersource/libraries.ai.cutlass.internal/releases/tag/v0.9.1-cri) (2026-06-12)
### Enhancements (Notes: all tests are based on the CRI simulator)
  - **Flash Attention Performance Optimizations**
    - Add FA BF16 output support ([#620](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/620))
    - Add per tensor scale support to FA FP8 ([#627](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/627))
    - Optimize FP8 FA QK tile depth ([#629](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/629))
    - Optimize softmax: defer row-sum hreduce, overlap rescale with PV GEMM, fuse epilogue rescale+reorder ([#540](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/540))
    - Preload Q once into registers to eliminate per-K-tile Q copy/reorder ([#538](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/538))
    - Optimize GQA/batch divmod overhead and enlarge tile shape ([#542](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/542))
    - Reorder mainloop ops to help compiler avoid spurious register reuse and movs ([#541](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/541))
    - Hoist causal-only and var-len-only setup out of common path ([#543](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/543))
    - Add down conversion of tSrS ([#537](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/537))
    - Optimize reduce: ILP-friendly accumulation, fused 4x16 reduction, strided-mov hreduce ([#531](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/531))
    - Split CachedKV kernels into separate binary ([#516](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/516))
  - **Block Scaled GEMM Enhancements**
    - Block Scaled GEMM epilogue support ([#587](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/587))
    - Improve MXFP4/8 performance when M is unaligned ([#570](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/570))
  - **CuTe / Copy API Improvements**
    - Add multi-payload Block2D copy API and enlarge block2d load width ([#533](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/533))
    - Support null-src0 DPAS to elide redundant accumulator initialization and reduce register read pressure ([#530](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/530))
    - Add bdpas src0=null support in CuTe ([#597](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/597))
    - Use uc.wb for CRI block 2D store ([#528](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/528))
  - **Benchmarks**
    - Support GEMM Benchmark BF16 Accumulator & Destination ([#621](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/621))
    - Add Flash Attention benchmark for FP8 and BF16 ([#539](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/539))
    - Add GEMV config and input file for GEMM benchmark ([#591](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/591))
    - Explicitly specify KernelXeCooperative schedule in the benchmark ([#580](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/580))
    - Align BlockScalingGemm's CollectiveEpilogue with example in benchmark ([#520](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/520))
    - Align the W8A8 / FP16-MMA fast-path variants of the 08_bmg_gemm_f8 kernel pipeline and benchmark
  - **Others**
    - Sync upstream v0.9.1 release ([#631](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/631))
    - Add real shape test cases ([#515](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/515))
    - Remove code unrelated to CRI from the CRI development branch to support upstreaming CRI code to the public repo ([#574](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/574))
    - Update docs ([#482](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/482))

### Bug Fixes
  - **Flash Attention**
    - Fix persistent decode partition overflow with dynamic max_num_partitions ([#536](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/536))
    - Fix split barrier for persistent decode split-K synchronization ([#476](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/476))
    - Fix k_blocks computation when kv_cache is enabled ([#485](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/485))
    - Fix XE TopK softmax epilogue ([#568](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/568))
  - **GEMM / Epilogue**
    - Fix DispatchPolicy alias in Xe Standard epilogues ([#584](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/584))
    - Fix MXFP4/MXFP8 M-unaligned block scale fallback ([#549](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/549))
    - Fix --g=0 causing division-by-zero crash in grouped gemm mixed dtype ([#514](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/514))
  - **Build / Compilation**
    - Fix SYCL_INTEL_TARGET macro redefinition in cutlass.h ([#552](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/552))
    - Add noexcept for __spirv_ConvertFToBF16INTEL declaration ([#614](https://github.com/intel-innersource/libraries.ai.cutlass.internal/pull/614))

**See the [CHANGELOG](CHANGELOG-SYCL.md) for details of all past releases and updates.**

# CuTe

SYCL\*TLA supports the newly introduced core library, CuTe, to describe and manipulate tensors of threads and data.
CuTe in SYCL\*TLA is a collection of C++ SYCL template abstractions for
defining and operating on hierarchically multidimensional layouts of threads and data.
CuTe provides `Layout` and `Tensor` objects that compactly package the type,
shape, memory space, and layout of data, while performing the complicated indexing for the user.
This lets programmers focus on the logical descriptions of their algorithms while
CuTe does the mechanical bookkeeping for them. With these tools, we can quickly design,
implement, and modify all dense linear algebra operations.

The core abstractions of CuTe are hierarchically multidimensional layouts
which can be composed with data arrays to represent tensors.
The representation of layouts is powerful enough to represent nearly
everything we need to implement efficient dense linear algebra.
Layouts can also be combined and manipulated via functional composition, on which we build a large set of common operations such as tiling and partitioning.

SYCL\*TLA and beyond adopts CuTe throughout the GEMM hierarchy in its templates.
This greatly simplifies the design and improves code composability and readability.
More documentation specific to CuTe can be found in its
[dedicated documentation directory](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/cute/00_quickstart.md).
[dedicated documentation directory](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/cute/00_quickstart.md).

# Compatibility

Minimum requirements:

- Architecture: Intel Data Center GPU Max Series (codename: Ponte-Vecchio)
- Compiler: Must support at least C++17
- DPC++ Compiler Version: oneAPI 2025.1 and onwards
- Intel Compute Runtime and Graphics Compiler: 
  - For Intel Data Center GPU Max Series: Runtime from [LTS driver installation guide](https://dgpu-docs.intel.com/driver/installation-lts2.html), IGC [v2.27.10](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.27.10) or later

## Hardware Support

SYCL*TLA runs successfully on the following Intel GPUs.

|**GPU**|**Intel GPU Architecture**
|---|---|
|Intel Data Center GPU Max Series            |Xe-HPC|
|Intel Arc GPU B580 Graphics                       |Xe2|
|Intel Data Center GPU Crescent Island             |Xe3p|

## Validated Software Configurations

We are regularly testing following setup in CI.

|**Platform**|**Operating System** | **DPC++ Compiler** | **G++** | **Intel Compute Runtime** |**Intel Graphics Compiler** |
|-----------------|----------|-----------------|--------|---------------------|-----------------------|
|Xe-HPC| Ubuntu 24.04 |2025.3+ |G++13  | 25.48 | 2.24 |
|Xe2| Ubuntu 25.04 |2025.3+  |G++13  | 26.01 | 2.27 |
|Xe3p| Ubuntu 25.04 |2025.3+  |G++13  | 26.01 | 2.27 |


## Target Architecture

The target architecture information is passed on to SYCL*TLA via the cmake flag
`DPCPP_SYCL_TARGET`. 

```
cmake .. -DDPCPP_SYCL_TARGET="intel_gpu_pvc"
```
Or
Or

```
cmake .. -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g21" 
```
Or

```
cmake .. -DDPCPP_SYCL_TARGET="intel_gpu_cri"
```


Or

```
cmake .. -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g31" 
```

> Note: `-DDPCPP_SYCL_TARGET="bmg"` will compile for both `intel_gpu_bmg_g21`, `intel_gpu_bmg_g31` targets.

Please refer to the [functionality documentation](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/functionality.md)
Or

```
cmake .. -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g31" 
```

> Note: `-DDPCPP_SYCL_TARGET="bmg"` will compile for both `intel_gpu_bmg_g21`, `intel_gpu_bmg_g31` targets.

Please refer to the [functionality documentation](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/functionality.md)
for details on which kernels require which target architectures.

# Documentation

CUTLASS is described in the following documents and the accompanying
[Doxygen documentation](https://nvidia.github.io/cutlass).

- [Quick Start Guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md) - basics of building and running CUTLASS
- [Functionality](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/functionality.md) - summarizes functionality available in CUTLASS
- [Efficient GEMM in CUDA](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/efficient_gemm.md) - describes how GEMM kernels may be implemented efficiently in CUDA
- [CUTLASS 3.x Design](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/cutlass_3x_design.md) - describes the CUTLASS 3.x design, its benefits, and how CuTe enables us to write much more composable components
- [GEMM API 3.x](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/gemm_api_3x.md) - describes the CUTLASS 3.x GEMM model and C++ template concepts
- [Implicit GEMM Convolution](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/implicit_gemm_convolution.md) - describes 2-D and 3-D convolution in CUTLASS
- [Code Organization](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/code_organization.md) - describes the organization and contents of the CUTLASS project
- [Terminology](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/terminology.md) - describes terms used in the code
- [Programming Guidelines](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/programming_guidelines.md) - guidelines for writing efficient modern CUDA C++
- [Fundamental types](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/fundamental_types.md) - describes basic C++ classes used in CUTLASS to represent numeric quantities and arrays
- [Layouts](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/layout.md) - describes layouts of matrices and tensors in memory
- [Tile Iterators](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/tile_iterator_concept.md) - describes C++ concepts for iterating over tiles of matrices in memory
- [CUTLASS Utilities](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/utilities.md) - additional templates used to facilitate rapid development
- [Xe4 Feature Implementation Status](./media/docs/cpp/xe4_features_apis.md) - Xe4 CuTe atom and API implementation status for Intel Xe4 GPUs

  | # | Feature | Summary |
  |---|---|---|
  | 1 | [MMA (AMMA/TMM)](./media/docs/cpp/xe4_features_apis.md#1-mma-ammatmm--compute-apis) | AMMA async compute atoms and TMM synchronous tensor-matrix atoms with barrier tracking and cluster multicast variants |
  | 2 | [Data Types for MMA](./media/docs/cpp/xe4_features_apis.md#2-data-types-for-mma) | Standard and block-scaled MMA data types (TF32, BF16, FP16, FP8, FP4, MXFP4/8) |
  | 3 | [Async DMA (ADMA)](./media/docs/cpp/xe4_features_apis.md#3-async-dma-adma--data-movement) | Tensor + linear ADMA load/store/prefetch/reduce and multicast flows |
  | 4 | [LDSM/STSM](./media/docs/cpp/xe4_features_apis.md#4-slm--register-ldsmstsm--eu-access-to-core-matrix) | Load/store matrix atoms for SLM ↔ register data movement |
  | 5 | [A-Barriers](./media/docs/cpp/xe4_features_apis.md#5-addressable-barriers-a-barriers) | Addressable barrier init, arrive, wait, and transaction APIs |
  | 6 | [Cluster APIs](./media/docs/cpp/xe4_features_apis.md#6-cluster-apis) | Cluster synchronization, relaxed barriers, and leader election |
  | 7 | [Tile Scheduler](./media/docs/cpp/xe4_features_apis.md#7-tile-scheduler) | Static and dynamic persistent tile schedulers with CLC support |
  | 8 | [Collective Builder & Block-Scaled GEMM](./media/docs/cpp/xe4_features_apis.md#8-collective-builder--block-scaled-gemm-support) | Collective MMA builders for standard and block-scaled GEMM |
  | 9 | [Asymmetric Register Allocation](./media/docs/cpp/xe4_features_apis.md#9-asymmetric-register-allocation) | Control/worker sub-group register partitioning for epilogue |
  | 10 | [EU Copy Atoms](./media/docs/cpp/xe4_features_apis.md#10-eu-copy-atoms) | EU-based copy atoms for GMEM/SLM/register data movement |
  | 11 | [Tensor Pipe Quantize/Downconvert](./media/docs/cpp/xe4_features_apis.md#11-tensor-pipe-quantizedownconvert) | Register-level tensor-pipe quantize/downconvert APIs and XE4 tests |

# Resources


# Building SYCL*TLA

SYCL*TLA is a header-only template library and does not need to be built to be used by other
projects. Client applications should target SYCL*TLA's `include/` directory in their include
paths.

SYCL*TLA unit tests, examples, and utilities can be built with CMake.
The minimum version of CMake is given in the [Quickstart guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md).
The minimum version of CMake is given in the [Quickstart guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md).
Make sure you have Intel oneAPI DPC++ compiler installed and the environment is properly set up.

```bash
$ source /opt/intel/oneapi/setvars.sh
```

Create a build directory within the SYCL*TLA project, then run CMake. You need to specify
the target Intel GPU architecture using the `DPCPP_SYCL_TARGET` flag.
For Intel Data Center GPU Max Series (Ponte Vecchio), use `intel_gpu_pvc`.
For Intel Arc GPU B580 Graphics, use `intel_gpu_bmg_g21`.
For Intel Data Center GPU Crescent Island, use `intel_gpu_cri`.
For Intel Arc GPU Battlemage (G31), use `intel_gpu_bmg_g31`.
For Intel Arc GPU Battlemage (G31), use `intel_gpu_bmg_g31`.

```bash
$ mkdir build && cd build

$ CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET="intel_gpu_pvc"     # compiles for Intel Data Center GPU Max Series
```

Or for Intel Arc GPU B580 Graphics:

```bash
$  CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g21" # compiles for Intel Arc GPU B580 Graphics
```

Or for Intel Data Center GPU Crescent Island:

```bash
$  CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET="intel_gpu_cri" # compiles for Intel Data Center GPU Crescent Island
```

Or for Intel Arc GPU Battlemage (G31):

```bash
$  CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g31" # compiles for Intel Arc GPU Battlemage (G31)
```

Or for Intel Arc GPU Battlemage (G31):

```bash
$  CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g31" # compiles for Intel Arc GPU Battlemage (G31)
```

To compile with G++ as host compiler, add the flag `-DDPCPP_HOST_COMPILER=g++-13` to the cmake command. Please note that the build system must be able to find `g++-13` in your PATH.

```bash
$  CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_HOST_COMPILER=g++-13 -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g21" # compiles for Intel Arc GPU B580 Graphics with G++ as host compiler
```

From the `build/` directory, compile and run the SYCL*TLA unit tests by building the target `test_unit` with make.

The unit tests are organized as several binaries mirroring the top-level namespaces of SYCL*TLA,
and they may be executed in parallel via Ninja's `-j` command line argument.

```bash
$ ninja test_unit -j$(nproc)
...
...
...
[----------] Global test environment tear-down
[==========] XXX tests from YY test cases ran. (ZZZZ ms total)
[  PASSED  ] XXX tests.
```

All tests should pass on supported Intel GPU platforms, though the exact number of tests may vary over time.


# Project Structure

SYCL*TLA is arranged as a header-only library along with Utilities, Tools, Examples, and unit tests.

A detailed explanation of the source code organization may be found in the
[SYCL*TLA documentation](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/code_organization.md), but several main components are summarized below.
[SYCL*TLA documentation](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/code_organization.md), but several main components are summarized below.

## SYCL*TLA

```
include/                     # client applications should target this directory in their build's include paths

  cutlass/                   # SYCL Templates for Linear Algebra Subroutines and Solvers - headers only

    arch/                    # direct exposure of Intel GPU architecture features (including instruction-level GEMMs)

    conv/                    # code specialized for convolution on Intel GPUs

    epilogue/                # code specialized for the epilogue of gemm/convolution using SYCL

    gemm/                    # code specialized for general matrix product computations with SYCL

    layout/                  # layout definitions for matrices, tensors, and other mathematical objects in memory

    platform/                # SYCL-capable Standard Library components for Intel GPUs

    reduction/               # bandwidth-limited reduction kernels optimized for Intel GPU architectures

    thread/                  # SYCL workgroup and subgroup code for Intel GPU execution units
    
    transform/               # code specialized for layout, type, and domain transformations using SYCL

    *                        # core vocabulary types, containers, and basic numeric operations

  cute/                      # CuTe Layout, layout algebra, MMA/Copy atoms, tiled MMA/Copy for SYCL

    algorithm/               # Definitions of core operations such as copy, gemm, and operations on cute::tuples

    arch/                    # Intel GPU architecture wrapper structs for copy and math instructions

    atom/                    # Meta-information for Intel GPU operators and SYCL kernels

      mma_atom.hpp           # cute::Mma_Atom and cute::TiledMma for Intel GPU architectures

      copy_atom.hpp          # cute::Copy_Atom and cute::TiledCopy optimized for SYCL

      *xe*.hpp               # Intel Xe architecture specific meta-information for copy and math operations

    *                        # Core library types such as Shape, Stride, Layout, Tensor, and associated operations

```

### SYCL*TLA Examples

[SYCL*TLA examples](https://github.com/intel/sycl-tla/tree/main/examples) apply SYCL*TLA templates to implement basic computations.
[SYCL*TLA examples](https://github.com/intel/sycl-tla/tree/main/examples) apply SYCL*TLA templates to implement basic computations.

### Tools

```
tools/
  library/                   # SYCL*TLA Instance Library - contains instantiations of all supported SYCL*TLA templates
    include/
      cutlass/
        library/

  profiler/                  # Profiler                 - SYCL support not yet available
                             #                            (command-line utility for executing operations)
  
  util/                      # Utilities               - contains numerous helper classes for
    include/                 #                            managing tensors in Intel GPU device memory, reference
      cutlass/               #                            implementations for SYCL GEMM, random initialization
        util/                #                            of tensors, and I/O for Intel GPU environments.
```

### Test

The `test/unit/` directory consist of unit tests implemented with Google Test that demonstrate
basic usage of Core API components and complete tests of the CUTLASS GEMM computations.

Instructions for building and running the Unit tests are described in the [Quickstart guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md).
Instructions for building and running the Unit tests are described in the [Quickstart guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md).

# About

SYCL*TLA is released by INTEL Corporation as Open Source software under the
[3-clause "New" BSD license](https://github.com/intel/sycl-tla/blob/main/LICENSE.txt).
[3-clause "New" BSD license](https://github.com/intel/sycl-tla/blob/main/LICENSE.txt).

# Contributors

The official list of SYCL*TLA developers and contributors is available here: [CONTRIBUTORS](https://github.com/intel/sycl-tla/blob/main/CONTRIBUTORS.md).
The official list of SYCL*TLA developers and contributors is available here: [CONTRIBUTORS](https://github.com/intel/sycl-tla/blob/main/CONTRIBUTORS.md).

# Contributing

## Pull Request Templates

We provide concise PR templates to streamline documentation:

### Quick Start

**GitHub CLI:**
```bash
gh pr create --template .github/PULL_REQUEST_TEMPLATE/bug_fix.md
gh pr create --template .github/PULL_REQUEST_TEMPLATE/performance.md
gh pr create --template .github/PULL_REQUEST_TEMPLATE/feature.md
gh pr create --template .github/PULL_REQUEST_TEMPLATE/refactoring.md
```

**GitHub Web:** Add `?template=<name>.md` to PR URL (e.g., `?template=bug_fix.md`)

### Which Template?

- 🐛 **Bug fixes** → `bug_fix.md` - Root cause + verification
- ⚡ **Performance** → `performance.md` - Profiling data + benchmarks
- ✨ **Features** → `feature.md` - API design + examples
- 🔨 **Refactoring** → `refactoring.md` - Refactored/Redesigned code
- 📝 **Mixed/Other** → Default template

See [`.github/PULL_REQUEST_TEMPLATE`](https://github.com/intel/sycl-tla/tree/main/.github/PULL_REQUEST_TEMPLATE) for details.
See [`.github/PULL_REQUEST_TEMPLATE`](https://github.com/intel/sycl-tla/tree/main/.github/PULL_REQUEST_TEMPLATE) for details.

# Copyright

Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2025 Intel Corporation. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
