# CUTLASS XE4 Performance Test Framework

This directory contains a performance testing framework based on Google Test (gtest) for CUTLASS XE4 examples.

## Performance Test Runner

The `run_perf_tests.py` script automates execution of performance tests, collects metrics from simulator output, and generates CSV reports with optional baseline comparison for regression detection.

### Basic Usage
```bash
cd /path/to/repo/
cd build

# Run all tests and generate CSV report
python ../examples/xe4/perf/run_perf_tests.py \
  --perf_binary ./examples/xe4/perf/gemm_perf \
  --simulator_dir /path/to/dir/aubload/started \
  --output_dir ./results \
  --test_all
```

### Performance Regression Testing
```bash
# Run tests with baseline comparison (auto-detects regressions > 10%)
python ../examples/xe4/perf/run_perf_tests.py \
  --perf_binary ./examples/xe4/perf/gemm_perf \
  --simulator_dir /path/to/dir/aubload/started \
  --output_dir ./results \
  --test_all \
  --baseline_csv ./references/baseline_perf.csv

# Disable comparison even if baseline is provided
python ../examples/xe4/perf/run_perf_tests.py \
  --perf_binary ./examples/xe4/perf/gemm_perf \
  --simulator_dir /path/to/dir/aubload/started \
  --output_dir ./results \
  --test_all \
  --baseline_csv ./references/baseline_perf.csv \
  --disable_compare_perf
```

### Additional Options
```bash
# List available tests without running them
python ../examples/xe4/perf/run_perf_tests.py \
  --perf_binary ./examples/xe4/perf/gemm_perf \
  --list_tests

# Run a specific test by name
python ../examples/xe4/perf/run_perf_tests.py \
  --perf_binary ./examples/xe4/perf/gemm_perf \
  --simulator_dir /path/to/dir/aubload/started \
  --output_dir ./results \
  -t "TestSuite.TestName"

# Run only tests matching a pattern (e.g., L2 or L3)
python ../examples/xe4/perf/run_perf_tests.py \
  --perf_binary ./examples/xe4/perf/gemm_perf \
  --simulator_dir /path/to/dir/aubload/started \
  --output_dir ./results \
  --test_all \
  --test_type L2
```

### Output Files
- **perf_test_results.csv**: Performance metrics for all tests
- **regression.csv**: Detailed comparison report (when baseline is provided)
- **XML Reports**: Individual test result files in output directory

## Directory Structure

```
perf
├── CMakeLists.txt
├── compare_perf.py
├── gemm
│   ├── gemm_params.hpp
│   └── gemm_test.cpp
├── gemm_cluster
│   ├── gemm_cluster_params.hpp
│   └── gemm_cluster_test.cpp
├── README.md
├── references
│   └── baseline_perf.csv
├── requirements.txt
├── run_perf_tests.py
└── testFixture
    └── testFixture.hpp
```

## Components

### TestParamInfo Class
The `TestParamInfo` class is used to hold parameters for gtest parametric `TEST_P` tests:
- `testName`: String identifier for the test
- `testParamsPtr`: Void pointer to test-specific parameter structure

### Test Fixtures
- `PerfTestFixture`: Base test fixture providing common functionality
- `ArithmeticTestFixture`: Specialized fixture for arithmetic operation tests

### Test Parameters
- `GemmOpPerfTestParams`: Structure for multiplication test parameters
- `GemmPerfParams`: Structure for GEMM performance test parameters (for future extension)
- `TestParametersManager`: Manages and provides test parameter collections

## Building

The performance tests are integrated into the existing CUTLASS build system. To build:

```bash
# From the CUTLASS root directory
cmake -S . -B build -GNinja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DCUTLASS_ENABLE_BENCHMARKS=OFF \
  -DCUTLASS_ENABLE_GTEST_UNIT_TESTS=OFF \
  -DDPCPP_SYCL_TARGET=intel_gpu_jgs \
  -DCMAKE_CXX_COMPILER=/home/rajprinc/Cutlass_setup/setup/DPCPP/compiler/latest/bin/icpx \
  -DCMAKE_C_COMPILER=/home/rajprinc/Cutlass_setup/setup/DPCPP/compiler/latest/bin/icx \
  -DCMAKE_CXX_FLAGS="-fsycl -I/home/rajprinc/Cutlass_setup/setup/oneMKL_inst/mkl/2025.2/include/" \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE \
  -DCMAKE_PREFIX_PATH=/home/rajprinc/Cutlass_setup/setup/oneMKL_inst/mkl/2025.2/lib/cmake \
  -DDNNL_DIR=/home/rajprinc/Cutlass_setup/setup/oneDNN_inst/lib/cmake/dnnl


ninja gemm_perf
```

## Running Tests

### Run all performance tests:
```bash
./examples/sycl/xe4/perf_test/gemm_perf
```

### Run with verbose output:
```bash
make run_perf_tests_verbose
```

### Run specific test pattern:
```bash
./examples/sycl/xe4/perf_test/gemm_perf --gtest_filter="*GemmOpPerfTest*"
```

## Current Test Cases

The framework currently includes arithmetic multiplication tests with various scenarios:
1. Simple positive multiplication
2. Negative and positive multiplication
3. Multiplication by zero
4. Decimal multiplication
5. Both negative multiplication
6. Large and small number multiplication
7. Mathematical constants multiplication

## Extending the Framework

To add new test types:

1. **Define parameter structure** in `header/sample.hpp`:
```cpp
struct YourTestParams {
    // Your test parameters
};
```

2. **Create test fixture** in `testFixture/testFixture.hpp`:
```cpp
class YourTestFixture : public PerfTestFixture {
public:
    void runTest() override;
    void displayTestInfo() override;
};
```

3. **Implement fixture** in `testFixture/testFixture.cpp`

4. **Add test cases** in `src/sample.cpp` using `TEST_P` macro

5. **Update CMakeLists.txt** if needed for additional dependencies

## Integration with CUTLASS Build System

The framework uses the existing `add_xe4_test()` function which:
- Creates executable with SYCL compilation
- Sets up include directories
- Configures target for XE4 architecture
- Integrates with CTest for automated testing

## Dependencies

- Google Test (GTest)
- SYCL compiler support
- CUTLASS library
- XE4 target support
