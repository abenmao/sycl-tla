#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <sycl/sycl.hpp>

#include "validation.hpp"

using namespace cute;
using namespace sycl;

struct BGEMM_TEST_CONFIG {
  using ElementA = fp16;
  using ElementB = fp16;
  using ElementC = fp16;
  using ElementD = fp16;
  using ElementAccumulator = float;
  using CtaTileShape_MNK = Shape<_256,_256,_128>;
  using CtaNum_MN = Shape<_2, _1>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  static constexpr int StagesA = 3;
  static constexpr int StagesC = 2;
  static constexpr int FragmentSize = 2;
  static constexpr int num_xecore_x = 1;
  static constexpr int num_xecore_y = 2;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {1024, 1024, 1024, 1};
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

struct BGEMM_ROW_ROW_VOID_C : public BGEMM_ROW_ROW {
  using ElementC = void;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
};

// Define a macro to extract memory info (offset and raw size)
#define GET_MEM_INFO(cls, path) \
    std::make_tuple( \
        (size_t)&(((cls*)0)->path), \
        sizeof(((cls*)0)->path) \
    )

// Add function to get shared memory information
template<typename GemmKernel>
std::string get_shared_memory_info() {
  using TensorStorage = typename GemmKernel::TensorStorage;

  std::ostringstream oss;

  // Lambda to convert bytes to KB with proper formatting
  auto bytes2kb = [](size_t bytes) -> std::string {
    double kb = bytes / 1024.0;
    std::ostringstream format;
    if (bytes < 1024) {
      // No decimal point for integer byte values
      format << bytes << " Bytes";
    } else if (bytes >= 1024 * 1024) {
      format << std::fixed << std::setprecision(2) << kb / 1024.0 << " MB";
    } else if (bytes % 1024 == 0) {
      // No decimal point for integer KB values
      format << static_cast<int>(kb) << " KB";
    } else {
      // Two decimal places for non-integer KB values
      format << std::fixed << std::setprecision(2) << kb << " KB";
    }
    return format.str();
  };

  // Extract memory information using the macro
  auto [offsetA, sizeA] = GET_MEM_INFO(TensorStorage, mainloop.smem_A);
  auto [offsetB, sizeB] = GET_MEM_INFO(TensorStorage, mainloop.smem_B);
  auto [offsetAcc, sizeAcc] = GET_MEM_INFO(TensorStorage, mainloop.smem_Acc);
  auto [offsetC, sizeC] = GET_MEM_INFO(TensorStorage, epilogue.collective.smem_C);
  auto [offsetD, sizeD] = GET_MEM_INFO(TensorStorage, epilogue.collective.smem_D);

  // Calculate the total size of TensorStorage
  size_t total_size = sizeof(TensorStorage);

  // Format output information
  oss << "Share Memory Allocation (Total: " << bytes2kb(total_size) << ")" << std::endl;
  oss << "- Mainloop" << std::endl;
  oss << "    A: size=" << bytes2kb(sizeA) << ", offset=" << bytes2kb(offsetA) << std::endl;
  oss << "    B: size=" << bytes2kb(sizeB) << ", offset=" << bytes2kb(offsetB) << std::endl;
  oss << "    Acc: size=" << bytes2kb(sizeAcc) << ", offset=" << bytes2kb(offsetAcc) << std::endl;
  oss << "- Epilogue" << std::endl;
  oss << "    C: size=" << bytes2kb(sizeC) << ", offset=" << bytes2kb(offsetC) << std::endl;
  oss << "    D: size=" << bytes2kb(sizeD) << ", offset=" << bytes2kb(offsetD) << std::endl;
  return oss.str();
}

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
  constexpr int AlignmentA  = 512;                                            // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         ElementB    = typename Config::ElementB;                      // Element type for B matrix operand
  using         LayoutB     = typename Config::LayoutB;                       // Layout type for B matrix operand
  constexpr int AlignmentB  = 512;                                            // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         ElementC    = typename Config::ElementC;                      // Element type for C matrix operands
  using         ElementD    = typename Config::ElementD;                      // Element type for D matrix operands
  using         LayoutC     = typename Config::LayoutC;                       // Layout type for C and D matrix operands
  constexpr int AlignmentC  = 512;                                            // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

  // Kernel functional config
  using ElementAccumulator  = typename Config::ElementAccumulator;            // Element type for internal accumulation
  using ArchTag             = cutlass::arch::Xe4;                             // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
  using TileShape           = typename Config::CtaTileShape_MNK;              // Threadblock-level tile size
  using ClusterShape        = typename Config::ClusterShape_MNK;              // Shape of the threadblocks in a cluster

  using ElementEpilogueCompute = float;
  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using EpilogueOperation = cute::conditional_t<
    cute::is_void_v<ElementC>,
    cutlass::epilogue::fusion::EltAct<cutlass::epilogue::thread::SiLu, ElementD, ElementEpilogueCompute>,
    cutlass::epilogue::fusion::EltActMul<cutlass::epilogue::thread::SiLu, ElementD, ElementEpilogueCompute>
  >;

  // Build the epilogue
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutC, AlignmentC,
      EpilogueScheduleType,
      EpilogueOperation
    >::CollectiveOp;

  // Build the mainloop
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      tuple<ElementAccumulator, ElementD>,
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
  using StrideD = typename GemmKernel::StrideD;

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /// GEMM setup and evaluation
  /////////////////////////////////////////////////////////////////////////////////////////////////

  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  auto problem_shape_mnkl = typename GemmKernel::ProblemShape {};
  cute::fill_int_tuple_from(problem_shape_mnkl, Config::ProblemShape_MNKL);

  uint32_t sizeA = size(select<0,2,3>(problem_shape_mnkl));
  uint32_t sizeB = size(select<1,2,3>(problem_shape_mnkl));
  uint32_t sizeC = size(select<0,1,3>(problem_shape_mnkl));

  auto A_s = malloc_shared<ElementA>(sizeA, q);
  std::generate_n(A_s, sizeA, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

  auto B_s = malloc_shared<ElementB>(sizeB, q);
  std::generate_n(B_s, sizeB, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

  ElementC* C_s = nullptr;
  if constexpr (!cute::is_void_v<ElementC>) {
    C_s = malloc_shared<ElementC>(sizeC, q);
    std::generate_n(C_s, sizeC, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });
  }

  auto D_s = malloc_shared<ElementD>(sizeC, q);
  std::fill_n(D_s, sizeC, ElementD(0));

  auto num_groups = ceil_div(problem_shape_mnkl, TileShape {});
  range<3> local_range(1, NumControlWarps + NumEpilogueWarps, cutlass::NumThreadsPerWarp);
  range<3> group_range(1, get<0>(num_groups), get<1>(num_groups));
  if (is_persistent_mode) {
    auto [cta_num_y, cta_num_x] = typename Config::CtaNum_MN {};
    auto [cluster_size_y, cluster_size_x, _] = typename Config::ClusterShape_MNK {};
    group_range[1] = min(group_range[1], cta_num_y * cluster_size_y);
    group_range[2] = min(group_range[2], cta_num_x * cluster_size_x);
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
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, select<0,1,3>(problem_shape_mnkl));

  auto smem_info = get_shared_memory_info<GemmKernel>();
  std::cout << smem_info << std::endl;

  q.parallel_for<Config>(Range, [=](nd_item<3> item) {
    auto args = typename Gemm::GemmKernel::Arguments {
      problem_shape_mnkl,
      { A_s, stride_A, B_s, stride_B },
      { C_s, stride_C, D_s, stride_D }
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

  auto silu_mul_op = [&](auto&& vec) {
    constexpr auto one = ElementEpilogueCompute(1.0);

    auto sigmod = [&](const auto &x) {
      return one / (one + sycl::exp(-x));
    };

    for (int i = 0; i < vec.size(); ++i) {
      auto val = ElementEpilogueCompute(vec[i]);
      auto result = val * sigmod(val);

      if constexpr (cute::is_void_v<ElementC>) {
        vec[i] = val * sigmod(val);
      } else {
        vec[i] = val * sigmod(val) * ElementEpilogueCompute(C_s[i]);
      }
    }

    return vec;
  };

  auto [mat_m, mat_n, mat_k, _] = problem_shape_mnkl;
  uint32_t err_cnt = validate_gemm_result(A_s, B_s, D_s, mat_m, mat_n, mat_k, as_mem_layout(LayoutA{}), as_mem_layout(LayoutB{}), NoOp{}, silu_mul_op);

  if (err_cnt > 0) {
    std::cout << smem_info << std::endl;
    std::cerr << "Test Failed!" << std::endl;
    exit(1);
  }

  std::cout << "Test Pass!" << std::endl;
}

int main()
{
  bool is_persistent_mode = true;
  run_test<BGEMM_ROW_ROW>(is_persistent_mode);
  return 0;
}