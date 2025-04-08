#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "cutlass/layout/matrix.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/arch/mma_xe4.hpp"
#include "cute/arch/copy_xe4_dma.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "validation.hpp"

using namespace cute;
using namespace sycl;
using namespace cute::xe4;
using namespace cutlass;
using namespace cutlass::gemm;
using namespace cutlass::gemm::collective;
using namespace cutlass::epilogue::collective;
using namespace cutlass::epilogue::collective::detail;
using namespace cutlass::epilogue::thread;

struct BGEMM_TEST_CONFIG {
  using ElementA = fp16;
  using ElementB = fp16;
  using ElementC = fp16;
  using ElementAccumulator = float;
  using CtaTileShape_MNK = Shape<_256,_512,_128>;
  static constexpr int StagesA = 3;
  static constexpr int StagesC = 1;
  static constexpr int FragmentSize = 2;
  static constexpr int num_xecore_x = 1;
  static constexpr int num_xecore_y = 2;
};

struct BGEMM_ROW_ROW : public BGEMM_TEST_CONFIG {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
};

struct BGEMM_COL_ROW : public BGEMM_TEST_CONFIG {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
};

struct BGEMM_ROW_COL : public BGEMM_TEST_CONFIG {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
};

struct BGEMM_COL_COL : public BGEMM_TEST_CONFIG {
  using LayoutA = cutlass::layout::ColumnMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
};

template<typename Config>
void run_test(bool is_persistent_mode = false)
{
  constexpr int NumControlWarps = 4;
  constexpr int NumEpilogueWarps = 16;

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /// GEMM kernel configurations
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // A matrix configuration
  using         ElementA    = typename Config::ElementA;                      // Element type for A matrix operand
  using         LayoutA     = typename Config::LayoutA;                       // Layout type for A matrix operand
  constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         ElementB    = typename Config::ElementB;                      // Element type for B matrix operand
  using         LayoutB     = typename Config::LayoutB;                       // Layout type for B matrix operand
  constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         ElementC    = typename Config::ElementC;                      // Element type for C and D matrix operands
  using         LayoutC     = typename Config::LayoutC;                       // Layout type for C and D matrix operands
  constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

  // Kernel functional config
  using ElementAccumulator  = typename Config::ElementAccumulator;            // Element type for internal accumulation
  using ArchTag             = cutlass::arch::Xe4;                             // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
  using TileShape           = typename Config::CtaTileShape_MNK;              // Threadblock-level tile size
  using ClusterShape        = Shape<_1, _1, _1>;                              // Shape of the threadblocks in a cluster

  // Build the epilogue
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementC, LayoutC, AlignmentC,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  // Build the mainloop
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      tuple<ElementAccumulator, ElementC>,
      TileShape, ClusterShape, cutlass::gemm::collective::StageCount<Config::StagesA>,
      cutlass::gemm::KernelTmaWarpSpecializedXe4<Config::StagesA, 1>
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;

  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  int mat_m = 1024;
  int mat_n = 1024;
  int mat_k = 256;
  int mat_l = 1;

  auto problem_shape_mnkl = make_shape(mat_m, mat_n, mat_k, mat_l);

  uint32_t sizeA = size(select<0,2,3>(problem_shape_mnkl));
  uint32_t sizeB = size(select<1,2,3>(problem_shape_mnkl));
  uint32_t sizeC = size(select<0,1,3>(problem_shape_mnkl));

  auto A_s = malloc_shared<ElementA>(sizeA, q);
  std::generate_n(A_s, sizeA, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

  auto B_s = malloc_shared<ElementB>(sizeB, q);
  std::generate_n(B_s, sizeB, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

  auto C_s = malloc_shared<ElementC>(sizeC, q);
  std::fill_n(C_s, sizeC, ElementC(0));

  auto num_groups = ceil_div(problem_shape_mnkl, TileShape{});
  range<3> local_range(1, NumControlWarps + NumEpilogueWarps, NumThreadsPerWarp);
  range<3> group_range(1, get<0>(num_groups), get<1>(num_groups));
  if (is_persistent_mode) {
    group_range[1] = min(group_range[1], Config::num_xecore_y);
    group_range[2] = min(group_range[2], Config::num_xecore_x);
  }
  nd_range<3> Range(group_range * local_range, local_range);

  std::cout << "IsPersistentMode: " << is_persistent_mode << std::endl;
  print("ProblemShape_MNKL: "); print(problem_shape_mnkl); print("\n");
  print("TileShape_MNK: "); print(TileShape{}); print("\n");
  print("ceil_div(ProblemShape,TileShape): "); print(num_groups); print("\n");
  std::cout << "Group range: {" << group_range[0] << ", " << group_range[1] << ", " << group_range[2] << "} \n";

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, select<0,2,3>(problem_shape_mnkl));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, select<1,2,3>(problem_shape_mnkl));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, select<0,1,3>(problem_shape_mnkl));

  q.parallel_for<Config>(Range, [=](nd_item<3> item) {
    auto args = typename Gemm::GemmKernel::Arguments {
      problem_shape_mnkl,
      { A_s, stride_A, B_s, stride_B },
      { C_s, stride_C }
    };

    GemmKernel kernel;
    auto params = kernel.to_underlying_arguments(args, nullptr);
    kernel(params);
   }).wait();

  auto as_mem_layout = [](auto layout) {
    if constexpr (std::is_same_v<decltype(layout), cutlass::layout::RowMajor>) {
      return mem_layout::row_major;
    } else if constexpr (std::is_same_v<decltype(layout), cutlass::layout::ColumnMajor>) {
      return mem_layout::col_major;
    } else {
      static_assert(false, "Unsupported layout");
    }
  };

  uint32_t err_cnt = validate_gemm_result(A_s, B_s, C_s, mat_m, mat_n, mat_k, as_mem_layout(LayoutA{}), as_mem_layout(LayoutB{}), ReluOp{});
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
