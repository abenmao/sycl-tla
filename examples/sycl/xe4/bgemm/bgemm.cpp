#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "cutlass/layout/matrix.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/arch/mma_xe4.hpp"
#include "cute/arch/copy_xe4_dma.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "validation.hpp"

using namespace cute;
using namespace sycl;
using namespace cute::xe4;
using namespace cutlass::gemm;
using namespace cutlass::epilogue;
using namespace cutlass::gemm::collective;
using namespace cutlass::epilogue::collective;
using namespace cutlass::epilogue::collective::detail;
using namespace cutlass::epilogue::thread;

struct BGEMM_TEST_CONFIG {
  using dtypeA = bf16;
  using dtypeB = bf16;
  using dtypeAcc = float;
  using dtypeC = bf16;
  static constexpr uint32_t wg_m = 256;
  static constexpr uint32_t wg_n = 512;
  static constexpr uint32_t wg_k = 128;
  static constexpr uint32_t stage = 3;
  static constexpr uint32_t num_xecore_x = 1;
  static constexpr uint32_t num_xecore_y = 2;
};

struct BGEMM_ROW_ROW : public BGEMM_TEST_CONFIG {
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

struct BGEMM_COL_ROW : public BGEMM_TEST_CONFIG {
  static constexpr mem_layout layout_a = mem_layout::col_major;
  static constexpr mem_layout layout_b = mem_layout::row_major;
};

struct BGEMM_ROW_COL : public BGEMM_TEST_CONFIG {
  static constexpr mem_layout layout_a = mem_layout::row_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
};

struct BGEMM_COL_COL : public BGEMM_TEST_CONFIG {
  static constexpr mem_layout layout_a = mem_layout::col_major;
  static constexpr mem_layout layout_b = mem_layout::col_major;
};

#define ENABLE_EPILOGUE_RELU

template<typename test>
void run_test(bool is_persistent_mode = false)
{
  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  int mat_m = 1024;
  int mat_n = 1024;
  int mat_k = 256;
  int mat_l = 1;

  using dtypeA = typename test::dtypeA;
  using dtypeB = typename test::dtypeB;
  using dtypeAcc = typename test::dtypeAcc;
  using dtypeC = typename test::dtypeC;

  constexpr int wg_m = test::wg_m;
  constexpr int wg_n = test::wg_n;
  constexpr int wg_k = test::wg_k;
  constexpr int stage = test::stage;
  constexpr mem_layout layout_a = test::layout_a;
  constexpr mem_layout layout_b = test::layout_b;

  assert(((mat_k + wg_k - 1) / wg_k) > 1);

  uint32_t sizeA = mat_m * mat_k;
  uint32_t sizeB = mat_n * mat_k;
  uint32_t sizeC = mat_m * mat_n;

  static constexpr auto tnspA = (layout_a == mem_layout::row_major) ? xe4::GMMA::Major::K : xe4::GMMA::Major::MN;
  static constexpr auto tnspB = (layout_b == mem_layout::row_major) ? xe4::GMMA::Major::MN : xe4::GMMA::Major::K;

  auto A_s = malloc_shared<dtypeA>(sizeA, q);
  std::generate_n(A_s, sizeA, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

  auto B_s = malloc_shared<dtypeB>(sizeB, q);
  std::generate_n(B_s, sizeB, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

  auto C_s = malloc_shared<dtypeC>(sizeC, q);
  std::fill_n(C_s, sizeC, dtypeC(0));

  constexpr uint32_t SubGroupSize = 32;
  constexpr uint32_t NumControlSubGroup = 4;
  constexpr uint32_t NumPostOpSubGroup = 16;
  range<3> local_range(1, NumControlSubGroup + NumPostOpSubGroup, SubGroupSize);
  range<3> group_range(1, ceil_div(mat_m, wg_m), ceil_div(mat_n, wg_n));
  if (is_persistent_mode) {
    group_range[1] = min(group_range[1], test::num_xecore_y);
    group_range[2] = min(group_range[2], test::num_xecore_x);
  }
  nd_range<3> Range(group_range * local_range, local_range);

  std::cout << "IsPersistentMode: " << is_persistent_mode << std::endl;
  std::cout << "ProblemShape: (" << mat_m << ", " << mat_n << ", " << mat_k << ")\n";
  std::cout << "TileShape: (" << wg_m << ", " << wg_n << ", " << wg_k << ")\n";
  std::cout << "Group range: {" << group_range[0] << ", " << group_range[1] << ", " << group_range[2] << "} \n";
  std::cout << "ProblemShape: (" << ceil_div(mat_m, wg_m) << ", " << ceil_div(mat_n, wg_n) << ")\n";

  using LayoutA = std::conditional_t<layout_a == mem_layout::row_major, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
  using LayoutB = std::conditional_t<layout_b == mem_layout::row_major, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
  using LayoutC = cutlass::layout::RowMajor;

  using TileShape = Shape<Int<wg_m>, Int<wg_n>, Int<wg_k>>;
  using ClusterShape = Shape<_1, _1, _1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Xe4, cutlass::arch::OpClassTensorOp,
    dtypeA, LayoutA, 16,
    dtypeB, LayoutB, 16,
    tuple<dtypeAcc, dtypeC>,
    TileShape, ClusterShape, cutlass::gemm::collective::StageCount<stage>,
    cutlass::gemm::KernelTmaWarpSpecializedXe4<stage, 1>>::CollectiveOp;

#ifdef ENABLE_EPILOGUE_RELU
  static constexpr int FragmentSize = 2;
  using FusionOp = fusion::LinCombEltAct<thread::ReLu, dtypeC, dtypeC, void>;  // LinCombEltAct<ElementOutput, ElementCompute, ElementSource>
  using FusionCallbacks = fusion::FusionCallbacks<
    Sm90TmaWarpSpecialized<1, 1, FragmentSize, false, false>,
    FusionOp, Shape<Int<wg_m>, Int<wg_n>, _1>, Shape<_2,_1>
  >;
#else
  static constexpr int FragmentSize = 1;
  using FusionCallbacks = fusion::Sm90EVT<fusion::Sm90AccFetch>;
#endif

  using SmemLayoutAtomC = Layout<Shape<Int<wg_m>, Int<wg_n>>, Stride<Int<wg_n>, _1>>;

  using CollectiveEpilogue = CollectiveEpilogue<
    Xe4DmaWarpSpecialized<FragmentSize, NumPostOpSubGroup, SubGroupSize>,
    dtypeC, cutlass::detail::TagToStrideC_t<LayoutC>, SmemLayoutAtomC, Shape<Int<wg_m>, Int<wg_n>>, FusionCallbacks
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void
  >;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;

  q.parallel_for<test>(Range, [=](nd_item<3> item) {
    auto args = typename GemmKernel::Arguments {
      {SubGroupSize, NumControlSubGroup, NumPostOpSubGroup},
      make_shape(mat_m, mat_n, mat_k, mat_l),
      {
        A_s, cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(mat_m, mat_k, mat_l)),
        B_s, cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(mat_n, mat_k, mat_l)),
      },
      {
        typename FusionCallbacks::Arguments{},
        C_s, cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(mat_m, mat_n, mat_l)),
      }
    };

    GemmKernel kernel;
    auto params = kernel.to_underlying_arguments(args, nullptr);
    kernel(params, item);
   }).wait();

  uint32_t err_cnt = validate_gemm_result(A_s, B_s, C_s, mat_m, mat_n, mat_k, layout_a, layout_b, ReluOp{});
  if (err_cnt > 0) {
    std::runtime_error("Test Failed!");
  } else {
    std::cout << "Test Pass!" << std::endl;
  }
}

int main()
{
  bool is_persistent_mode = true;
  run_test<BGEMM_COL_ROW>(is_persistent_mode);
  return 0;
}
