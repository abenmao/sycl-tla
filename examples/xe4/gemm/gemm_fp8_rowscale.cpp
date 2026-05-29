#include "gemm_fp8_rowscale.hpp"
#include <gtest/gtest.h>
#include <iostream>

template <class ElementD_>
struct FP8_ROWSCALE_CONFIG {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementD = ElementD_;
  using ElementAccumulator = float;
  using ElementScale = fp16;
  using ElementBias = ElementD_;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using CtaTileShape_MNK = Shape<_256, _256, _128>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  static constexpr int StagesA = 2;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 768, 256, 1};
};

struct FP8_ROWSCALE_FP16 : public FP8_ROWSCALE_CONFIG<fp16> {};
struct FP8_ROWSCALE_BF16 : public FP8_ROWSCALE_CONFIG<bf16> {};
struct FP8_ROWSCALE_FP32 : public FP8_ROWSCALE_CONFIG<float> {};

template <typename T>
class GemmFp8RowscaleTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(GemmFp8RowscaleTest);

TYPED_TEST_P(GemmFp8RowscaleTest, simple_run) {
  run_gemm_fp8_rowscale<TypeParam>();
}

REGISTER_TYPED_TEST_SUITE_P(GemmFp8RowscaleTest, simple_run);

using GemmFp8RowscaleTests = ::testing::Types<
    FP8_ROWSCALE_FP16,
    FP8_ROWSCALE_BF16,
    FP8_ROWSCALE_FP32>;

INSTANTIATE_TYPED_TEST_SUITE_P(GemmFp8Rowscale, GemmFp8RowscaleTest, GemmFp8RowscaleTests);

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int run_tests = RUN_ALL_TESTS();
  if (!::testing::UnitTest::GetInstance()->test_to_run_count()) {
    std::cout << "No tests were run.\n";
    return 1;
  }
  return run_tests;
}
