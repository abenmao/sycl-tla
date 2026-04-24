#include "gemm.hpp"
#include <gtest/gtest.h>
#include <iostream>

template <bool DynamicPersistent>
struct GEMM_BASE_CONFIG {
  using ElementA = fp16;
  using ElementB = fp16;
  using ElementC = fp16;
  using ElementD = fp16;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using CtaTileShape_MNK = Shape<_256, _256, _128>;
  using CtaNum_MN = Shape<_2, _1>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  static constexpr int StagesA = 2;
  static constexpr bool isDynamicPersistent = DynamicPersistent;
  static constexpr auto activation_type = ActivationType::SiLu;
  static constexpr auto operationC_type = OperationCType::Mul;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 768, 384, 1};
};

template <bool DynamicPersistent>
struct GEMM_CLUSTER_BASE_CONFIG {
  using ElementA = fp16;
  using ElementB = fp16;
  using ElementC = fp16;
  using ElementD = fp16;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using CtaTileShape_MNK = Shape<_256, _256, _128>;
  using CtaNum_MN = Shape<_1, _1>;
  using ClusterShape_MNK = Shape<_2, _1, _1>;

  static constexpr int StagesA = 2;
  static constexpr bool isDynamicPersistent = DynamicPersistent;
  static constexpr auto activation_type = ActivationType::SiLu;
  static constexpr auto operationC_type = OperationCType::Mul;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {1024, 1024, 1024, 1};
};

struct GEMM_STATIC_ROW_ROW : public GEMM_BASE_CONFIG<false> {};

struct GEMM_DYNAMIC_ROW_ROW : public GEMM_BASE_CONFIG<true> {};

struct GEMM_CLUSTER_STATIC_ROW_ROW_VOID_C : public GEMM_CLUSTER_BASE_CONFIG<false> {
  using ElementC = void;
  static constexpr auto operationC_type = OperationCType::None;
};

struct GEMM_CLUSTER_DYNAMIC_ROW_ROW_VOID_C : public GEMM_CLUSTER_BASE_CONFIG<true> {
  using ElementC = void;
  static constexpr auto operationC_type = OperationCType::None;
};

template <typename T>
class GemmTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(GemmTest);

TYPED_TEST_P(GemmTest, simple_run) {
  run_gemm<TypeParam>();
}

REGISTER_TYPED_TEST_SUITE_P(GemmTest, simple_run);

using GemmTests = ::testing::Types<
    GEMM_STATIC_ROW_ROW,
    GEMM_DYNAMIC_ROW_ROW,
    GEMM_CLUSTER_STATIC_ROW_ROW_VOID_C,
    GEMM_CLUSTER_DYNAMIC_ROW_ROW_VOID_C>;

INSTANTIATE_TYPED_TEST_SUITE_P(Gemm, GemmTest, GemmTests);

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int run_tests = RUN_ALL_TESTS();
  if (!::testing::UnitTest::GetInstance()->test_to_run_count()) {
    std::cout << "No tests were run.\n";
    return 1;
  }
  return run_tests;
}