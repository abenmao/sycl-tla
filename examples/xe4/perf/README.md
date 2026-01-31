# CUTLASS XE4 Performance Test Framework

This directory contains a performance testing framework based on Google Test (gtest) for CUTLASS XE4 examples.

Use the run_perf_tests.py file to generate the CSV sample usage looks like
```bash
cd /path/to/repo/
cd build
python ../examples/xe4/perf/run_perf_tests.py \
  --perf_binary ./examples/xe4/perf/gemm_perf \
  --simulator_dir /path/to/dir/aubload/started \
  --output_dir ./results \
  --test_all
```

## Directory Structure

```
perf_test/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── header/
│   └── sample.hpp              # Test parameter structures and management
├── src/
│   └── sample.cpp              # Main test implementation and entry point
└── testFixture/
    ├── testFixture.hpp         # Test fixture class definitions
    └── testFixture.cpp         # Test fixture implementations
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
mkdir build && cd build
cmake ..
make gemm_perf
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
