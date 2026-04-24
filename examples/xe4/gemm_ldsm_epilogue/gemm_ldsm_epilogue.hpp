/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// GEMM example with LDSM-based epilogue using descriptor-based XE4_LOAD_MATRIX / XE4_STORE_MATRIX
// for S2R and R2S operations instead of legacy XE4_LDSM / XE4_STSM vector copies.
// Uses Xe4LdsmAdmaBuilderImpl (Module 3) to configure the epilogue collective.
// See MODULE_DESIGN.md - Module 5: GEMM Example with LDSM Epilogue

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/cluster_launch.hpp>
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
  BiasAdd,
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
struct EpilogueOperationSelector<ActivationType::None, OperationCType::BiasAdd, ElementOutput, ElementCompute> {
  using type = cutlass::epilogue::fusion::PerColBias<ElementOutput, ElementCompute, float>;
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
      format << bytes << " Bytes";
    } else if (bytes >= 1024 * 1024) {
      format << std::fixed << std::setprecision(2) << kb / 1024.0 << " MB";
    } else if (bytes % 1024 == 0) {
      format << static_cast<int>(kb) << " KB";
    } else {
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
void run_gemm_ldsm_epilogue()
{
  constexpr int NumControlWarps = 4;
  constexpr int NumEpilogueWarps = 16;

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /// GEMM kernel configurations
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // A matrix configuration
  using         ElementA    = typename Config::ElementA;
  using         LayoutA     = typename Config::LayoutA;
  constexpr int AlignmentA  = 512;

  // B matrix configuration
  using         ElementB    = typename Config::ElementB;
  using         LayoutB     = typename Config::LayoutB;
  constexpr int AlignmentB  = 512;

  // C/D matrix configuration
  using         ElementC    = typename Config::ElementC;
  using         ElementD    = typename Config::ElementD;
  using         LayoutC     = typename Config::LayoutC;
  constexpr int AlignmentC  = 512;

  // Kernel functional config
  using ElementAccumulator  = typename Config::ElementAccumulator;
  using ArchTag             = cutlass::arch::Xe4;
  using OperatorClass       = cutlass::arch::OpClassTensorOp;
  using TileShape           = typename Config::CtaTileShape_MNK;
  using ClusterShape        = typename Config::ClusterShape_MNK;

  constexpr auto activation_type = Config::activation_type;
  constexpr auto operationC_type = Config::operationC_type;

  if constexpr (cute::is_void_v<ElementC> && operationC_type < OperationCType::BiasAdd) {
    static_assert(sizeof(cute::C<operationC_type>) < 0, "OperationC is not supported with void C");
  }

  using ElementEpilogueCompute = float;
  using EpilogueOperation = select_epilogue_operation<activation_type, operationC_type, ElementD, ElementEpilogueCompute>;

  // Build the epilogue using LDSM-based builder (Module 3)
  // Uses XE4_LOAD_MATRIX / XE4_STORE_MATRIX for S2R / R2S
  // instead of legacy XE4_LDSM / XE4_STSM vector copies
  using CollectiveEpilogue = typename cutlass::epilogue::collective::detail::Xe4LdsmAdmaBuilderImpl<
      OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutC, AlignmentC,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueOperation
    >::CollectiveOp;

  // Build the mainloop (identical to standard GEMM)
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      tuple<ElementAccumulator, ElementD>,
      TileShape, ClusterShape, cutlass::gemm::collective::StageCount<Config::StagesA>,
      cutlass::gemm::KernelTmaWarpSpecializedXe4<Config::StagesA, 1>
    >::CollectiveOp;

  // TODO: Fix hang issue with dynamic scheduler.
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

  auto [cluster_size_x, cluster_size_y, cluster_size_z] = ClusterShape{};
  sycl::range<3> cluster_size(cluster_size_z, cluster_size_y, cluster_size_x);

  auto num_groups = ceil_div(problem_shape_mnkl, TileShape {});

  std::cout << "LDSM Epilogue GEMM" << std::endl;
  print("ProblemShape_MNKL: "); print(problem_shape_mnkl); print("\n");
  print("TileShape_MNK: "); print(TileShape{}); print("\n");
  print("ceil_div(ProblemShape,TileShape): "); print(num_groups); print("\n");

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, select<0,2,3>(problem_shape_mnkl));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, select<1,2,3>(problem_shape_mnkl));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, select<0,1,3>(problem_shape_mnkl));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, select<0,1,3>(problem_shape_mnkl));

  auto smem_info = get_shared_memory_info<GemmKernel>();
  std::cout << smem_info << std::endl;

  using FusionCallbacks = typename CollectiveEpilogue::FusionCallbacks;
  auto callbacks_args = typename FusionCallbacks::Arguments {};

  auto args = typename Gemm::GemmKernel::Arguments {
    problem_shape_mnkl,
    { A_s, stride_A, B_s, stride_B },
    { callbacks_args, C_s, stride_C, D_s, stride_D }
  };

  GemmKernel kernel;
  auto params = kernel.to_underlying_arguments(args, nullptr);

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

  auto as_mem_layout = [](auto layout) {
    if constexpr (std::is_same_v<decltype(layout), cutlass::layout::RowMajor>) {
      return mem_layout::row_major;
    } else if constexpr (std::is_same_v<decltype(layout), cutlass::layout::ColumnMajor>) {
      return mem_layout::col_major;
    } else {
      static_assert(false, "Unsupported layout");
    }
  };

  auto [mat_m, mat_n, mat_k, mat_l] = problem_shape_mnkl;

  auto ptr_A = A_s;
  auto ptr_B = B_s;
  auto ptr_C = C_s;
  auto ptr_D = D_s;

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
        auto value = ElementEpilogueCompute(vec[i]);

        if (activation_type == ActivationType::SiLu) {
          value = silu(value);
        }

        if constexpr (!cute::is_void_v<ElementC>) {
          auto valueC = ElementEpilogueCompute(ptr_C[i]);

          if (operationC_type == OperationCType::Mul) {
            value *= valueC;
          } else if (operationC_type == OperationCType::Add) {
            value += valueC;
          }
        }

        result[i] = ElementD(value);
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
      ptr_C += mat_m * mat_n;
    }
  }

  std::cout << "Test Pass!" << std::endl;
}
