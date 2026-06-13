#pragma once

#include <algorithm>
#include <cstring>
#include <sstream>
#include <vector>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/cluster_launch.hpp>
#include <sycl/sycl.hpp>

#include "cutlass/util/device_memory.h"

#include "validation.hpp"

// ---------------------------------------------------------------------------
// Epilogue type selectors — self-contained copy so gemm_streamk.hpp does not
// need to include gemm.hpp (which pulls in legacy CUDA StreamK headers that
// are ambiguous under SYCL's sycl::max / cute::max overload set).
// ---------------------------------------------------------------------------
enum class ActivationType { SiLu, None };
enum class OperationCType { Mul, Add, BiasAdd, None };

template <ActivationType, OperationCType, class O, class C>
struct StreamKEpilogueOperationSelector {
  static_assert(sizeof(O) < 0, "Unsupported activation/operationC combination");
};
template <class O, class C>
struct StreamKEpilogueOperationSelector<ActivationType::SiLu, OperationCType::Mul, O, C> {
  using type = cutlass::epilogue::fusion::EltActMul<cutlass::epilogue::thread::SiLu, O, C>;
};
template <class O, class C>
struct StreamKEpilogueOperationSelector<ActivationType::SiLu, OperationCType::Add, O, C> {
  using type = cutlass::epilogue::fusion::EltActAdd<cutlass::epilogue::thread::SiLu, O, C>;
};
template <class O, class C>
struct StreamKEpilogueOperationSelector<ActivationType::SiLu, OperationCType::None, O, C> {
  using type = cutlass::epilogue::fusion::EltAct<cutlass::epilogue::thread::SiLu, O, C>;
};
template <class O, class C>
struct StreamKEpilogueOperationSelector<ActivationType::None, OperationCType::Mul, O, C> {
  using type = cutlass::epilogue::fusion::EltActMul<cutlass::epilogue::thread::Identity, O, C, O>;
};
template <class O, class C>
struct StreamKEpilogueOperationSelector<ActivationType::None, OperationCType::Add, O, C> {
  using type = cutlass::epilogue::fusion::EltActAdd<cutlass::epilogue::thread::Identity, O, C, O>;
};
template <class O, class C>
struct StreamKEpilogueOperationSelector<ActivationType::None, OperationCType::None, O, C> {
  using type = cutlass::epilogue::fusion::EltAct<cutlass::epilogue::thread::Identity, O, O>;
};
template <ActivationType A, OperationCType Op, class O, class C>
using select_streamk_epilogue_operation = typename StreamKEpilogueOperationSelector<A, Op, O, C>::type;
// ---------------------------------------------------------------------------

using namespace cute;
using namespace sycl;

template<typename Config>
void run_gemm_streamk()
{
  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementC = typename Config::ElementC;
  using ElementD = typename Config::ElementD;
  using ElementAccumulator = typename Config::ElementAccumulator;
  using LayoutA = typename Config::LayoutA;
  using LayoutB = typename Config::LayoutB;
  using LayoutC = typename Config::LayoutC;
  using TileShape = typename Config::CtaTileShape_MNK;
  using ClusterShape = typename Config::ClusterShape_MNK;

  constexpr auto activation_type = Config::activation_type;
  constexpr auto operationC_type = Config::operationC_type;

  if constexpr (cute::is_void_v<ElementC> && operationC_type < OperationCType::BiasAdd) {
    static_assert(sizeof(cute::C<operationC_type>) < 0,
      "OperationC Mul/Add requires a non-void ElementC; use OperationCType::None for void-C configs");
  }

  constexpr int AlignmentA = 512;
  constexpr int AlignmentB = 512;
  constexpr int AlignmentC = 512;

  using ArchTag = cutlass::arch::Xe4;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using ElementEpilogueCompute = float;
  using EpilogueOperation = select_streamk_epilogue_operation<activation_type, operationC_type, ElementD, ElementEpilogueCompute>;

  // SK-reduce epilogue: uses Xe4AdmaSkReduceBuilderImpl which wires in
  // CopyOpS2GReduce = XE4_ADMA_STORE_REDUCE<ElementImm, RedOp::Add>.
  // Non-final splits fire async_tensor_fred (smem_Imm -> gmem D workspace);
  // final split polls the per-tile counter, G2S loads back, then runs full epilogue.
  using CollectiveEpilogue =
    typename cutlass::epilogue::collective::detail::Xe4AdmaSkReduceBuilderImpl<
      OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutC, AlignmentC,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueOperation
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      tuple<ElementAccumulator, ElementD>,
      TileShape, ClusterShape, cutlass::gemm::collective::StageCount<Config::StagesA>,
      cutlass::gemm::KernelTmaWarpSpecializedXe4<Config::StagesA, 1>
    >::CollectiveOp;

  using TileSchedulerType = cutlass::gemm::StreamKScheduler;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileSchedulerType
  >;

  using TileScheduler = typename GemmKernel::TileScheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;

  using FusionCallbacks = typename CollectiveEpilogue::FusionCallbacks;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

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

  using ElementBias = float;
  ElementBias* Bias_s = nullptr;
  if constexpr (operationC_type == OperationCType::BiasAdd) {
    uint32_t sizeBias = size(select<1,3>(problem_shape_mnkl));
    Bias_s = malloc_shared<ElementBias>(sizeBias, q);
    std::generate_n(Bias_s, sizeBias, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });
  }

  auto [cluster_size_x, cluster_size_y, cluster_size_z] = ClusterShape{};
  sycl::range<3> cluster_size(cluster_size_z, cluster_size_y, cluster_size_x);

  print("ProblemShape_MNKL: "); print(problem_shape_mnkl); print("\n");
  print("TileShape_MNK: "); print(TileShape{}); print("\n");
  std::cout << "Decomposition: " << Config::decomposition_name << "\n";

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, select<0,2,3>(problem_shape_mnkl));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, select<1,2,3>(problem_shape_mnkl));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, select<0,1,3>(problem_shape_mnkl));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, select<0,1,3>(problem_shape_mnkl));

  auto callbacks_args = [&]() {
    if constexpr (operationC_type == OperationCType::BiasAdd) {
      auto stride_Bias = make_stride(_0{}, _1{}, static_cast<int64_t>(get<1>(problem_shape_mnkl)));
      return typename FusionCallbacks::Arguments{ Bias_s, stride_Bias };
    } else {
      return typename FusionCallbacks::Arguments{};
    }
  }();

  int device_id = 0;
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  cutlass::KernelHardwareInfo kernel_hw_info{device_id, sm_count, 0};

  TileSchedulerArguments scheduler_args = Config::template get_scheduler_args<TileSchedulerArguments>();

  auto args = typename GemmKernel::Arguments {
    problem_shape_mnkl,
    { A_s, stride_A, B_s, stride_B },
    { callbacks_args, C_s, stride_C, D_s, stride_D },
    scheduler_args
  };

  // Workspace holds: per-tile int32 atomic counters used by the SK freduce fixup.
  // EpiLoad increments the counter after each non-final split's fred completes;
  // the final split polls the counter until all non-final freds are visible.
  size_t workspace_size = TileScheduler::template get_workspace_size<
      decltype(problem_shape_mnkl), ElementAccumulator>(
      scheduler_args, problem_shape_mnkl, kernel_hw_info);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    workspace = malloc_device<uint8_t>(workspace_size, q);
    q.memset(workspace, 0, workspace_size).wait();
    std::cout << "Workspace size: " << workspace_size << " bytes\n";
  }

  GemmKernel kernel;
  auto params = kernel.to_underlying_arguments(args, kernel_hw_info, workspace);

  dim3 const grid = GemmKernel::get_grid_shape(params);
  dim3 const block = GemmKernel::get_block_shape();
  uint32_t const k_tiles_per_output_tile = (get<2>(problem_shape_mnkl) + get<2>(TileShape{}) - 1) / get<2>(TileShape{});

  std::cout << "Grid: (" << grid.x << ", " << grid.y << ", " << grid.z << ")\n";
  std::cout << "Block: (" << block.x << ", " << block.y << ", " << block.z << ")\n";

  range<3> group_range(grid.z, grid.y, grid.x);
  range<3> local_range(block.z, block.y, block.x);

  int smem_size = 0;
  cutlass::SyclClusterLaunchParams launch_params = {group_range, local_range, cluster_size, smem_size, q};

  cutlass::launch_kernel_on_cluster(
    launch_params,
    kernel,
    params
  ).wait();

  auto as_mem_layout = [](auto layout) {
    if constexpr (std::is_same_v<decltype(layout), cutlass::layout::RowMajor>) {
      return mem_layout::row_major;
    } else {
      return mem_layout::col_major;
    }
  };

  auto [mat_m, mat_n, mat_k, mat_l] = problem_shape_mnkl;

  auto ptr_A = A_s;
  auto ptr_B = B_s;
  auto ptr_C = C_s;
  auto ptr_D = D_s;
  auto ptr_Bias = Bias_s;

  for (int mat_i = 0; mat_i < mat_l; ++mat_i) {
    auto post_op = [&](auto&& vec) {
      std::vector<ElementD> result(vec.size());
      constexpr auto one = ElementEpilogueCompute(1.0);
      auto sigmod = [&](const auto& x) { return one / (one + sycl::exp(-x)); };
      auto silu   = [&](const auto& x) { auto t = ElementEpilogueCompute(x); return t * sigmod(t); };
      for (int i = 0; i < (int)vec.size(); ++i) {
        auto value = ElementEpilogueCompute(vec[i]);
        if constexpr (activation_type == ActivationType::SiLu) value = silu(value);
        if constexpr (cute::is_void_v<ElementC>) {
          if constexpr (operationC_type == OperationCType::BiasAdd)
            value += ElementEpilogueCompute(ptr_Bias[i % mat_n]);
        } else {
          auto valueC = ElementEpilogueCompute(ptr_C[i]);
          if constexpr (operationC_type == OperationCType::Mul) value *= valueC;
          else if constexpr (operationC_type == OperationCType::Add) value += valueC;
        }
        result[i] = ElementD(value);
      }
      return result;
    };

    uint32_t err_cnt = validate_gemm_result(ptr_A, ptr_B, ptr_D, mat_m, mat_n, mat_k,
                                             as_mem_layout(LayoutA{}), as_mem_layout(LayoutB{}),
                                             NoOp{}, post_op);
    if (err_cnt > 0) {
      std::cerr << "Test FAILED at batch " << mat_i << ", error count: " << err_cnt << std::endl;
      exit(1);
    }

    ptr_A += mat_m * mat_k;
    ptr_B += mat_k * mat_n;
    ptr_D += mat_m * mat_n;
    if constexpr (!cute::is_void_v<ElementC>)                 ptr_C    += mat_m * mat_n;
    if constexpr (operationC_type == OperationCType::BiasAdd)  ptr_Bias += mat_n;
  }

  std::cout << "Test PASSED!" << std::endl;

  if (workspace) free(workspace, q);
  free(A_s, q);
  free(B_s, q);
  if (C_s) free(C_s, q);
  if (Bias_s) free(Bias_s, q);
  free(D_s, q);
}
