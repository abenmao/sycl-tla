#pragma once

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/float8.h>
#include <sycl/sycl.hpp>

#include "validation.hpp"

using namespace cute;
using namespace sycl;

// D[m,n] = scale_a[m] * scale_b[n] * acc[m,n] + bias[m]

template<typename Config>
void run_gemm_fp8_rowscale()
{
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /// GEMM kernel configurations
  /////////////////////////////////////////////////////////////////////////////////////////////////

  using         ElementA    = typename Config::ElementA;
  using         LayoutA     = typename Config::LayoutA;
  constexpr int AlignmentA  = 512;

  using         ElementB    = typename Config::ElementB;
  using         LayoutB     = typename Config::LayoutB;
  constexpr int AlignmentB  = 512;

  using         ElementC    = void;
  using         ElementD    = typename Config::ElementD;
  using         LayoutC     = typename Config::LayoutC;
  constexpr int AlignmentC  = 512;

  using ElementAccumulator  = typename Config::ElementAccumulator;
  using ElementCompute      = float;
  using ElementScale        = typename Config::ElementScale;
  using ElementBias         = typename Config::ElementBias;
  using ArchTag             = cutlass::arch::Xe4;
  using OperatorClass       = cutlass::arch::OpClassTensorOp;
  using TileShape           = typename Config::CtaTileShape_MNK;
  using ClusterShape        = typename Config::ClusterShape_MNK;

  using EpilogueOperation = cutlass::epilogue::fusion::ScaledMM<
    ElementD, ElementCompute, ElementScale, ElementBias>;

  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;

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
    cutlass::gemm::StaticPersistentScheduler
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

  auto [mat_m, mat_n, mat_k, mat_l] = problem_shape_mnkl;

  uint32_t sizeA = size(select<0,2,3>(problem_shape_mnkl));
  uint32_t sizeB = size(select<1,2,3>(problem_shape_mnkl));
  uint32_t sizeD = size(select<0,1,3>(problem_shape_mnkl));
  uint32_t sizeScaleA = mat_m * mat_l;
  uint32_t sizeScaleB = mat_n * mat_l;
  uint32_t sizeBias   = mat_m * mat_l;

  auto A_s = malloc_shared<ElementA>(sizeA, q);
  std::generate_n(A_s, sizeA, [=]() {
    return static_cast<ElementA>(0.5f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  });

  auto B_s = malloc_shared<ElementB>(sizeB, q);
  std::generate_n(B_s, sizeB, [=]() {
    return static_cast<ElementB>(0.5f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  });

  auto D_s = malloc_shared<ElementD>(sizeD, q);
  std::fill_n(D_s, sizeD, ElementD(0));

  auto scale_a_s = malloc_shared<ElementScale>(sizeScaleA, q);
  std::generate_n(scale_a_s, sizeScaleA, [=]() {
    return static_cast<ElementScale>(0.01f + 2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  });

  auto scale_b_s = malloc_shared<ElementScale>(sizeScaleB, q);
  std::generate_n(scale_b_s, sizeScaleB, [=]() {
    return static_cast<ElementScale>(0.01f + 2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  });

  auto bias_s = malloc_shared<ElementBias>(sizeBias, q);
  std::generate_n(bias_s, sizeBias, [=]() {
    return static_cast<ElementBias>(0.5f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  });

  auto [cluster_size_x, cluster_size_y, cluster_size_z] = ClusterShape{};
  sycl::range<3> cluster_size(cluster_size_z, cluster_size_y, cluster_size_x);

  print("ProblemShape_MNKL: "); print(problem_shape_mnkl); print("\n");
  print("TileShape_MNK: "); print(TileShape{}); print("\n");

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, select<0,2,3>(problem_shape_mnkl));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, select<1,2,3>(problem_shape_mnkl));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, select<0,1,3>(problem_shape_mnkl));
  StrideC stride_C;

  auto stride_scale_a = make_stride(_1{}, _0{}, static_cast<int64_t>(mat_m));
  auto stride_scale_b = make_stride(_0{}, _1{}, static_cast<int64_t>(mat_n));
  auto stride_bias    = make_stride(_1{}, _0{}, static_cast<int64_t>(mat_m));

  using FusionCallbacks = typename CollectiveEpilogue::FusionCallbacks;

  auto callbacks_args = typename FusionCallbacks::Arguments {
    scale_a_s, stride_scale_a,
    scale_b_s, stride_scale_b,
    bias_s, stride_bias
  };

  int device_id = 0;
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  cutlass::KernelHardwareInfo kernel_hw_info{device_id, sm_count, 0};

  auto args = typename Gemm::GemmKernel::Arguments {
    problem_shape_mnkl,
    { A_s, stride_A, B_s, stride_B },
    { callbacks_args, nullptr, stride_C, D_s, stride_D }
  };

  GemmKernel kernel;
  auto params = kernel.to_underlying_arguments(args, kernel_hw_info, nullptr);

  dim3 const grid = GemmKernel::get_grid_shape(params);
  dim3 const block = GemmKernel::get_block_shape();

  range<3> group_range(grid.z, grid.y, grid.x);
  range<3> local_range(block.z, block.y, block.x);

  int smem_size = 0;
  cutlass::SyclClusterLaunchParams launch_params = {group_range, local_range, cluster_size, smem_size, q};

  cutlass::launch_kernel_on_cluster(
    launch_params,
    kernel,
    params
  ).wait();

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /// Validation
  /////////////////////////////////////////////////////////////////////////////////////////////////

  auto ptr_A = A_s;
  auto ptr_B = B_s;
  auto ptr_D = D_s;
  auto ptr_scale_a = scale_a_s;
  auto ptr_scale_b = scale_b_s;
  auto ptr_bias = bias_s;

  auto as_mem_layout = [](auto layout) {
    if constexpr (std::is_same_v<decltype(layout), cutlass::layout::RowMajor>) {
      return mem_layout::row_major;
    } else if constexpr (std::is_same_v<decltype(layout), cutlass::layout::ColumnMajor>) {
      return mem_layout::col_major;
    } else {
      static_assert(false, "Unsupported layout");
    }
  };

  int n_dim = mat_n;

  for (int mat_i = 0; mat_i < mat_l; ++mat_i) {
    auto post_op = [&](auto&& acc_vec) {
      std::vector<ElementD> result(acc_vec.size());
      for (int i = 0; i < static_cast<int>(acc_vec.size()); ++i) {
        int row = i / n_dim;
        int col = i % n_dim;
        float sa = static_cast<float>(ptr_scale_a[row]);
        float sb = static_cast<float>(ptr_scale_b[col]);
        float b  = static_cast<float>(ptr_bias[row]);
        float val = sa * sb * static_cast<float>(acc_vec[i]) + b;
        result[i] = static_cast<ElementD>(val);
      }
      return result;
    };

    uint32_t err_cnt = validate_gemm_result(ptr_A, ptr_B, ptr_D, mat_m, mat_n, mat_k,
      as_mem_layout(LayoutA{}), as_mem_layout(LayoutB{}), NoOp{}, post_op);
    if (err_cnt > 0) {
      std::cerr << "Test Failed at " << mat_i << "th batch, error count: " << err_cnt << std::endl;
      exit(1);
    }

    ptr_A += mat_m * mat_k;
    ptr_B += mat_k * mat_n;
    ptr_D += mat_m * mat_n;
    ptr_scale_a += mat_m;
    ptr_scale_b += mat_n;
    ptr_bias += mat_m;
  }

  std::cout << "Test Pass!" << std::endl;

  sycl::free(A_s, q);
  sycl::free(B_s, q);
  sycl::free(D_s, q);
  sycl::free(scale_a_s, q);
  sycl::free(scale_b_s, q);
  sycl::free(bias_s, q);
}
