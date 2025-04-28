#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <sycl/sycl.hpp>

#include "validation.hpp"

using namespace cute;
using namespace sycl;

enum class ActivationType {
  SiLu,
  None
};

enum class OperationCType {
  Mul,
  Add,
  None
};

template <ActivationType activation_type, OperationCType operationC_type, class ElementOutput, class ElementCompute>
struct EpilogueOperationSelector {
  static_assert(false, "Unsupported activation type or operationC type");
};

template <class ElementOutput, class ElementCompute>
struct EpilogueOperationSelector<ActivationType::SiLu, OperationCType::Mul, ElementOutput, ElementCompute> {
  using type = cutlass::epilogue::fusion::EltActMul<cutlass::epilogue::thread::SiLu, ElementOutput, ElementCompute>;
};

template <class ElementOutput, class ElementCompute>
struct EpilogueOperationSelector<ActivationType::SiLu, OperationCType::Add, ElementOutput, ElementCompute> {
  using type = cutlass::epilogue::fusion::EltActAdd<cutlass::epilogue::thread::SiLu, ElementOutput, ElementCompute>;
};

template <class ElementOutput, class ElementCompute>
struct EpilogueOperationSelector<ActivationType::SiLu, OperationCType::None, ElementOutput, ElementCompute> {
  using type = cutlass::epilogue::fusion::EltAct<cutlass::epilogue::thread::SiLu, ElementOutput, ElementCompute>;
};

template <class ElementOutput>
struct ImmediateTypeSelector {
  static constexpr bool is_fp_postop = is_floating_t<ElementOutput>::value && (sizeof_bits_v<ElementOutput> < 16);
  using type = cute::conditional_t<is_fp_postop, bf16, ElementOutput>;
};

template <class ElementOutput>
using immediate_type = typename ImmediateTypeSelector<ElementOutput>::type;

template <class ElementOutput, class ElementCompute>
struct EpilogueOperationSelector<ActivationType::None, OperationCType::Mul, ElementOutput, ElementCompute> {
  using type = cutlass::epilogue::fusion::EltActMul<cutlass::epilogue::thread::Identity, ElementOutput, ElementCompute, immediate_type<ElementOutput>>;
};

template <class ElementOutput, class ElementCompute>
struct EpilogueOperationSelector<ActivationType::None, OperationCType::Add, ElementOutput, ElementCompute> {
  using type = cutlass::epilogue::fusion::EltActAdd<cutlass::epilogue::thread::Identity, ElementOutput, ElementCompute, immediate_type<ElementOutput>>;
};

template <class ElementOutput, class ElementCompute>
struct EpilogueOperationSelector<ActivationType::None, OperationCType::None, ElementOutput, ElementCompute> {
  using type = cutlass::epilogue::fusion::EltAct<cutlass::epilogue::thread::Identity, ElementOutput, immediate_type<ElementOutput>>;
};

template <ActivationType activation_type, OperationCType operationC_type, class ElementOutput, class ElementCompute>
using select_epilogue_operation = typename EpilogueOperationSelector<activation_type, operationC_type, ElementOutput, ElementCompute>::type;

// Define a macro to extract memory info (offset and raw size)
#define GET_MEM_INFO(cls, path) std::make_tuple((size_t) & (((cls *)0)->path), sizeof(((cls *)0)->path))

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
void run_gemm()
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

  constexpr auto activation_type = Config::activation_type;
  constexpr auto operationC_type = Config::operationC_type;

  if constexpr (cute::is_void_v<ElementC> && operationC_type != OperationCType::None) {
    static_assert(false, "OperationC is not supported with void C");
  }

  using ElementEpilogueCompute = float;
  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using EpilogueOperation = select_epilogue_operation<activation_type, operationC_type, ElementD, ElementEpilogueCompute>;

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

  auto [cluster_size_y, cluster_size_x, cluster_size_z] = ClusterShape{};

  if constexpr (size(ClusterShape{}) > 1) {
    std::stringstream ss;
    ss << int(cluster_size_x) << "x" << int(cluster_size_y) << "x" << int(cluster_size_z);
    std::string cluster_str = ss.str();
    std::cout << "Cluster size: " << cluster_str << "\n";
    setenv("XE4_CLUSTER_SIZE", cluster_str.c_str(), 1);
  }

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

  constexpr bool is_persistent = Config::is_persistent;
  if constexpr (is_persistent) {
    auto [cta_num_y, cta_num_x] = typename Config::CtaNum_MN {};
    group_range[1] = min(group_range[1], cta_num_y * cluster_size_y);
    group_range[2] = min(group_range[2], cta_num_x * cluster_size_x);
  }
  nd_range<3> Range(group_range * local_range, local_range);

  std::cout << "IsPersistentMode: " << is_persistent << std::endl;
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

  auto ptr_A = A_s;
  auto ptr_B = B_s;
  auto ptr_C = C_s;
  auto ptr_D = D_s;
  auto [mat_m, mat_n, mat_k, mat_l] = problem_shape_mnkl;

  for (int mat_i = 0; mat_i < mat_l; ++mat_i) {
    auto post_op = [&](auto&& vec) {
      std::vector<ElementD> result(vec.size());

      constexpr auto one = ElementEpilogueCompute(1.0);

      auto sigmod = [&](const auto &x) {
        return one / (one + sycl::exp(-x));
      };

      auto silu = [&](const auto &x) {
        auto tmp = ElementEpilogueCompute(x);
        return tmp * sigmod(tmp);
      };

      for (int i = 0; i < vec.size(); ++i) {
        if constexpr (cute::is_void_v<ElementC>) {
          if (activation_type == ActivationType::None) {
            result[i] = vec[i];
          } else {
            result[i] = silu(vec[i]);
          }
        } else {
          auto value = ElementEpilogueCompute(vec[i]);
          auto valueC = ElementEpilogueCompute(ptr_C[i]);

          if (activation_type == ActivationType::SiLu) {
            value = silu(value);
          }

          if (operationC_type == OperationCType::Mul) {
            value *= valueC;
          } else if (operationC_type == OperationCType::Add) {
            value += valueC;
          }

          result[i] = ElementD(value);
        }
      }

      return result;
    };

    uint32_t err_cnt = validate_gemm_result(ptr_A, ptr_B, ptr_D, mat_m, mat_n, mat_k, as_mem_layout(LayoutA{}), as_mem_layout(LayoutB{}), NoOp{}, post_op);
    if (err_cnt > 0) {
      std::cout << smem_info << std::endl;
      std::cerr << "Test Failed at " << mat_i << "th batch, error count: " << err_cnt << std::endl;
      exit(1);
    }

    ptr_A += mat_m * mat_k;
    ptr_B += mat_k * mat_n;
    ptr_D += mat_m * mat_n;

    if constexpr (!cute::is_void_v<ElementC>) {
      ptr_C += mat_m * mat_n;;
    }
  }

  std::cout << "Test Pass!" << std::endl;
}
