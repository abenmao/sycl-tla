#include "gemm_static.hpp"
#include <gtest/gtest.h>

struct GEMM_TEST_CONFIG {
  using ElementA = fp16;
  using ElementB = fp16;
  using ElementC = fp16;
  using ElementD = fp16;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using CtaTileShape_MNK = Shape<_256,_256,_128>;
  using CtaNum_MN = Shape<_2, _1>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  static constexpr int StagesA = 2;
  static constexpr auto activation_type = ActivationType::SiLu;
  static constexpr auto operationC_type = OperationCType::Mul;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 768, 384, 1};
};

struct GEMM_ROW_ROW : public GEMM_TEST_CONFIG {
};

struct GEMM_ROW_ROW_BiasAdd : public GEMM_ROW_ROW {
  using ElementC = void;
  using CtaNum_MN = Shape<_1, _1>;
  static constexpr auto activation_type = ActivationType::None;
  static constexpr auto operationC_type = OperationCType::BiasAdd;
};

struct GEMM_ROW_ROW_PERF : public GEMM_TEST_CONFIG {
  using CtaNum_MN = Shape<_2, _2>;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {2048, 2048, 4096, 1};
};

struct GEMM_COL_ROW : public GEMM_TEST_CONFIG {
  using LayoutA = cutlass::layout::ColumnMajor;
};

struct GEMM_ROW_COL : public GEMM_TEST_CONFIG {
  using LayoutB = cutlass::layout::ColumnMajor;
};

struct GEMM_COL_COL : public GEMM_TEST_CONFIG {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
};

struct BATCH_GEMM_ROW_ROW : public GEMM_ROW_ROW {
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {1024, 1024, 1024, 4};
};

struct GEMM_ROW_ROW_ResidualAddC : public GEMM_ROW_ROW {
  static constexpr auto activation_type = ActivationType::None;
  static constexpr auto operationC_type = OperationCType::Add;
};

struct BF8_GEMM_ROW_ROW : public GEMM_TEST_CONFIG {
  using ElementA = bf8;
  using ElementB = bf8;
  using ElementD = bf8;
  static constexpr auto activation_type = ActivationType::None;
};

template <typename T>
class GemmTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(GemmTest);

TYPED_TEST_P(GemmTest, simple_run) {
  run_gemm<TypeParam>();
}

REGISTER_TYPED_TEST_SUITE_P(GemmTest, simple_run);
using GemmTests = ::testing::Types<GEMM_ROW_ROW>;
INSTANTIATE_TYPED_TEST_SUITE_P(GemmStatic, GemmTest, GemmTests);

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int run_tests = RUN_ALL_TESTS();
  if (!::testing::UnitTest::GetInstance()->test_to_run_count()) {
    std::cout << "No tests were run.\n";
    return 1;
  }
  return run_tests;
}