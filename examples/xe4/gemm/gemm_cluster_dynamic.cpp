#include "gemm_dynamic.hpp"

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
  using CtaNum_MN = Shape<_1, _1>;
  using ClusterShape_MNK = Shape<_2, _1, _1>;

  static constexpr int StagesA = 2;
  static constexpr auto activation_type = ActivationType::SiLu;
  static constexpr auto operationC_type = OperationCType::Mul;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {1024, 1024, 1024, 1};
};

struct GEMM_ROW_ROW : public GEMM_TEST_CONFIG {
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

struct GEMM_ROW_ROW_VOID_C : public GEMM_ROW_ROW {
  using ElementC = void;
  static constexpr auto operationC_type = OperationCType::None;
};

struct BF8_GEMM_ROW_ROW_VOID_C : public GEMM_ROW_ROW_VOID_C {
  using ElementA = bf8;
  using ElementB = bf8;
  using ElementD = bf8;
  static constexpr auto activation_type = ActivationType::None;
};

int main()
{
  run_gemm<GEMM_ROW_ROW_VOID_C>();
  return 0;
}