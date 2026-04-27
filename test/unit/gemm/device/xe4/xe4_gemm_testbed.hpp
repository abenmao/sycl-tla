#pragma once

/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation, All rights reserved.
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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/arch/xe4_util.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/float_subbyte.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <sycl/sycl.hpp>

// Adding temporary include for validation utilities until we refactor the testbed to be shared between unit tests and examples.
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

namespace cutlass {
namespace test {
namespace gemm {
namespace xe4 {

namespace detail {

template <class GemmKernel, class Config>
class TestbedImpl {
 public:
  using ElementA = typename Config::ElementA;
  using ElementB = typename Config::ElementB;
  using ElementC = typename Config::ElementC;
  using ElementD = typename Config::ElementD;
  using ElementAccumulator = typename Config::ElementAccumulator;
  using LayoutA = typename Config::LayoutA;
  using LayoutB = typename Config::LayoutB;
  using LayoutC = typename Config::LayoutC;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  using ProblemShape = typename GemmKernel::ProblemShape;

  bool run(ProblemShape problem_shape) {
    if (!sufficient()) {
      std::cout << "Test failed due to insufficient device." << std::endl;
      return false;
    }

    std::cout << "Running on " << queue_.get_device().get_info<info::device::name>() << "\n";
    initialize(problem_shape);
    launch(problem_shape);
    queue_.wait();

    bool passed = verify(problem_shape);
    if (!passed) {
      std::cout << "Error : Failed \n";
    }

    return passed;
  }

  ~TestbedImpl() {
    release();
  }

 private:
  using ElementBias = float;
  using ElementEpilogueCompute = float;
  using StrideBias = decltype(make_stride(_0{}, _1{}, static_cast<int64_t>(0)));

  static constexpr int NumControlWarps = 4;
  static constexpr int NumEpilogueWarps = 16;
  static constexpr int AlignmentA = 512;
  static constexpr int AlignmentB = 512;
  static constexpr int AlignmentC = 512;

  sycl::queue queue_;

  ElementA* A_ = nullptr;
  ElementB* B_ = nullptr;
  ElementC* C_ = nullptr;
  ElementD* D_ = nullptr;
  ElementBias* Bias_ = nullptr;

  uint32_t sizeA_ = 0;
  uint32_t sizeB_ = 0;
  uint32_t sizeC_ = 0;
  uint32_t sizeBias_ = 0;

  StrideA stride_A_{};
  StrideB stride_B_{};
  StrideC stride_C_{};
  StrideD stride_D_{};
  StrideBias stride_Bias_{};

  std::string smem_info_{};

  bool sufficient() {
    return true;
  }

  void release() {
    if (A_) {
      sycl::free(A_, queue_);
      A_ = nullptr;
    }
    if (B_) {
      sycl::free(B_, queue_);
      B_ = nullptr;
    }
    if (C_) {
      sycl::free(C_, queue_);
      C_ = nullptr;
    }
    if (D_) {
      sycl::free(D_, queue_);
      D_ = nullptr;
    }
    if (Bias_) {
      sycl::free(Bias_, queue_);
      Bias_ = nullptr;
    }
  }

  void initialize(ProblemShape problem_shape) {
    release();

    sizeA_ = size(select<0, 2, 3>(problem_shape));
    sizeB_ = size(select<1, 2, 3>(problem_shape));
    sizeC_ = size(select<0, 1, 3>(problem_shape));

    A_ = malloc_shared<ElementA>(sizeA_, queue_);
    B_ = malloc_shared<ElementB>(sizeB_, queue_);
    D_ = malloc_shared<ElementD>(sizeC_, queue_);

    std::generate_n(A_, sizeA_, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });
    std::generate_n(B_, sizeB_, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });
    std::fill_n(D_, sizeC_, ElementD(0));

    if constexpr (!cute::is_void_v<ElementC>) {
      C_ = malloc_shared<ElementC>(sizeC_, queue_);
      std::generate_n(C_, sizeC_, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });
    }

    if constexpr (Config::operationC_type == OperationCType::BiasAdd) {
      sizeBias_ = size(select<1, 3>(problem_shape));
      Bias_ = malloc_shared<ElementBias>(sizeBias_, queue_);
      std::generate_n(Bias_, sizeBias_, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });
    }

    stride_A_ = cutlass::make_cute_packed_stride(StrideA{}, select<0, 2, 3>(problem_shape));
    stride_B_ = cutlass::make_cute_packed_stride(StrideB{}, select<1, 2, 3>(problem_shape));
    stride_C_ = cutlass::make_cute_packed_stride(StrideC{}, select<0, 1, 3>(problem_shape));
    stride_D_ = cutlass::make_cute_packed_stride(StrideD{}, select<0, 1, 3>(problem_shape));

    auto [mat_m, mat_n, mat_k, mat_l] = problem_shape;
    stride_Bias_ = make_stride(_0{}, _1{}, static_cast<int64_t>(mat_n));
  }

  void launch(ProblemShape problem_shape) {
    using ArchTag = cutlass::arch::Xe4;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using TileShape = typename Config::CtaTileShape_MNK;
    using ClusterShape = typename Config::ClusterShape_MNK;

    constexpr auto activation_type = Config::activation_type;
    constexpr auto operationC_type = Config::operationC_type;

    if constexpr (cute::is_void_v<ElementC> && operationC_type < OperationCType::BiasAdd) {
      static_assert(sizeof(cute::C<operationC_type>) < 0, "OperationC is not supported with void C");
    }

    using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using EpilogueOperation = select_epilogue_operation<activation_type, operationC_type, ElementD, ElementEpilogueCompute>;

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

    using LocalGemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void
    >;

    using LocalGemm = cutlass::gemm::device::GemmUniversalAdapter<LocalGemmKernel>;

    auto [cluster_size_x, cluster_size_y, cluster_size_z] = ClusterShape{};
    sycl::range<3> cluster_size(cluster_size_z, cluster_size_y, cluster_size_x);

    auto num_groups = ceil_div(problem_shape, TileShape {});
    
    print("ProblemShape_MNKL: "); print(problem_shape); print("\n");
    print("TileShape_MNK: "); print(TileShape{}); print("\n");
    print("ceil_div(ProblemShape,TileShape): "); print(num_groups); print("\n");

    using FusionCallbacks = typename CollectiveEpilogue::FusionCallbacks;
    auto callbacks_args = [&]() {
      if constexpr (operationC_type == OperationCType::BiasAdd) {
        return typename FusionCallbacks::Arguments { Bias_, stride_Bias_ };
      } else {
        return typename FusionCallbacks::Arguments {};
      }
    }();

    auto args = typename LocalGemm::GemmKernel::Arguments {
      problem_shape,
      { A_, stride_A_, B_, stride_B_ },
      { callbacks_args, C_, stride_C_, D_, stride_D_ }
    };

    LocalGemmKernel kernel;
    auto params = kernel.to_underlying_arguments(args, nullptr);

    dim3 const grid = LocalGemmKernel::get_grid_shape(params);
    dim3 const block = LocalGemmKernel::get_block_shape();

    range<3> group_range(grid.z, grid.y, grid.x);
    range<3> local_range(block.z, block.y, block.x);

    int smem_size = 0;
    cutlass::SyclClusterLaunchParams launch_params = {group_range, local_range, cluster_size, smem_size, queue_};

    cutlass::launch_kernel_on_cluster(
      launch_params,
      kernel,
      params
    ).wait();
  }

  bool verify(ProblemShape problem_shape) {
    auto ptr_A = A_;
    auto ptr_B = B_;
    auto ptr_C = C_;
    auto ptr_D = D_;
    auto ptr_Bias = Bias_;

    auto [mat_m, mat_n, mat_k, mat_l] = problem_shape;
    constexpr ElementEpilogueCompute epsilon = ElementEpilogueCompute(0.05f);
    constexpr ElementEpilogueCompute nonzero_floor = ElementEpilogueCompute(0.05f);

    for (int mat_i = 0; mat_i < mat_l; ++mat_i) {
      std::vector<ElementD> gold(mat_m * mat_n);
      for (int row = 0; row < mat_m; ++row) {
        for (int col = 0; col < mat_n; ++col) {
          ElementEpilogueCompute acc = ElementEpilogueCompute(0);
          for (int kk = 0; kk < mat_k; ++kk) {
            ElementEpilogueCompute a = ElementEpilogueCompute(load_matrix_element<LayoutA>(ptr_A, row, kk, mat_m, mat_k));
            ElementEpilogueCompute b = ElementEpilogueCompute(load_matrix_element<LayoutB>(ptr_B, kk, col, mat_k, mat_n));
            acc += a * b;
          }

          ElementEpilogueCompute value = acc;
          if (Config::activation_type == ActivationType::SiLu) {
            value = silu(value);
          }

          if constexpr (cute::is_void_v<ElementC>) {
            if (Config::operationC_type == OperationCType::BiasAdd) {
              value += ElementEpilogueCompute(ptr_Bias[col]);
            }
          } else {
            ElementEpilogueCompute valueC = ElementEpilogueCompute(load_matrix_element<LayoutC>(ptr_C, row, col, mat_m, mat_n));
            if (Config::operationC_type == OperationCType::Mul) {
              value *= valueC;
            } else if (Config::operationC_type == OperationCType::Add) {
              value += valueC;
            } else if (Config::operationC_type == OperationCType::BiasAdd) {
              assert(sizeof(C<Config::operationC_type>) < 0);
            }
          }

          gold[row * mat_n + col] = ElementD(value);
        }
      }

      if (!compare_vectors(gold.data(), ptr_D, gold.size(), epsilon, nonzero_floor)) {
        std::cout << smem_info_ << std::endl;
        std::cerr << "Test Failed at " << mat_i << "th batch." << std::endl;
        return false;
      }

      ptr_A += mat_m * mat_k;
      ptr_B += mat_k * mat_n;
      ptr_D += mat_m * mat_n;

      if constexpr (!cute::is_void_v<ElementC>) {
        ptr_C += mat_m * mat_n;
      }
      if constexpr (Config::operationC_type == OperationCType::BiasAdd) {
        ptr_Bias += mat_n;
      }
    }

    std::cout << "Test Pass!" << std::endl;
    return true;
  }

  template <typename Layout, typename Element>
  Element load_matrix_element(const Element* ptr, int row, int col, int rows, int cols) const {
    static_assert(
      std::is_same_v<Layout, cutlass::layout::RowMajor> ||
      std::is_same_v<Layout, cutlass::layout::ColumnMajor>,
      "Unsupported layout");
    if constexpr (std::is_same_v<Layout, cutlass::layout::RowMajor>) {
      return ptr[row * cols + col];
    } else {
      return ptr[col * rows + row];
    }
  }

  ElementEpilogueCompute silu(ElementEpilogueCompute x) const {
    ElementEpilogueCompute one = ElementEpilogueCompute(1);
    return x * (one / (one + sycl::exp(-x)));
  }

  bool relatively_equal(ElementEpilogueCompute a, ElementEpilogueCompute b,
                        ElementEpilogueCompute epsilon,
                        ElementEpilogueCompute nonzero_floor) const {
    ElementEpilogueCompute abs_a = std::abs(a);
    ElementEpilogueCompute abs_b = std::abs(b);
    ElementEpilogueCompute diff = std::abs(a - b);
    ElementEpilogueCompute zero = ElementEpilogueCompute(0);

    if (a == b) {
      return true;
    }
    if (a == zero || b == zero || (abs_a + abs_b) < nonzero_floor) {
      return diff < epsilon * nonzero_floor;
    }
    return diff < epsilon * (abs_a + abs_b);
  }

  bool compare_vectors(const ElementD* gold, const ElementD* out, size_t count,
                       ElementEpilogueCompute epsilon,
                       ElementEpilogueCompute nonzero_floor) const {
    size_t mismatch = 0;
    for (size_t i = 0; i < count; ++i) {
      ElementEpilogueCompute g = ElementEpilogueCompute(gold[i]);
      ElementEpilogueCompute o = ElementEpilogueCompute(out[i]);
      if (!relatively_equal(g, o, epsilon, nonzero_floor)) {
        if (mismatch < 10) {
          std::cout << "mismatch idx=" << i << " gold=" << g << " out=" << o << std::endl;
        }
        ++mismatch;
      }
    }

    if (mismatch > 0) {
      std::cout << "Failed: " << mismatch << "/" << count << " mismatch" << std::endl;
      return false;
    }
    return true;
  }
};

} // namespace detail

template<typename Config>
void run_gemm()
{
  /////////////////////////////////////////////////////////////////////////////////////////////////
  /// GEMM kernel configurations
  /////////////////////////////////////////////////////////////////////////////////////////////////

  using ElementA = typename Config::ElementA;
  using LayoutA = typename Config::LayoutA;
  using ElementB = typename Config::ElementB;
  using LayoutB = typename Config::LayoutB;
  using ElementC = typename Config::ElementC;
  using ElementD = typename Config::ElementD;
  using LayoutC = typename Config::LayoutC;
  using ElementAccumulator = typename Config::ElementAccumulator;

  using ArchTag = cutlass::arch::Xe4;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = typename Config::CtaTileShape_MNK;
  using ClusterShape = typename Config::ClusterShape_MNK;

  constexpr int AlignmentA = 512;
  constexpr int AlignmentB = 512;
  constexpr int AlignmentC = 512;

  constexpr auto activation_type = Config::activation_type;
  constexpr auto operationC_type = Config::operationC_type;

  if constexpr (cute::is_void_v<ElementC> && operationC_type < OperationCType::BiasAdd) {
    static_assert(sizeof(cute::C<operationC_type>) < 0, "OperationC is not supported with void C");
  }

  using ElementEpilogueCompute = float;
  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using EpilogueOperation = select_epilogue_operation<activation_type, operationC_type, ElementD, ElementEpilogueCompute>;

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
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void
  >;

  auto problem_shape_mnkl = typename GemmKernel::ProblemShape {};
  cute::fill_int_tuple_from(problem_shape_mnkl, Config::ProblemShape_MNKL);

  cutlass::test::gemm::xe4::detail::TestbedImpl<GemmKernel, Config> testbed;
  bool passed = testbed.run(problem_shape_mnkl);
  if (!passed) {
    exit(1);
  }
}

template <typename Tensor>
inline void blockscaled_random_fill_fp4(Tensor& X, uint64_t seed) {
  using T = typename Tensor::element_type;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  for (int i = 0; i < size(X); ++i) {
    X(i) = T(dist(rng));
  }
}

template <typename Tensor>
inline void blockscaled_random_fill_sf(Tensor& X, uint64_t seed) {
  using T = typename Tensor::element_type;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(1.0f, 4.0f);
  for (int i = 0; i < size(X); ++i) {
    X(i) = T(dist(rng));
  }
}

template <typename Tensor>
inline void blockscaled_zero_fill(Tensor& X) {
  using T = typename Tensor::element_type;
  for (int i = 0; i < size(X); ++i) {
    X(i) = T(0);
  }
}

template <typename InTensor>
inline void blockscaled_subbyte_pack(InTensor& X) {
  using T = typename InTensor::element_type;
  if constexpr (sizeof_bits_v<T> % 8 != 0) {
    static_assert(sizeof_bits_v<T> == 4, "Unsupported sub-byte data size");

    auto ptr = recast_ptr<uint8_t>(&*X.data());
    auto bytes = X.size();

    for (size_t i = 0; i < bytes / 2; i++) {
      ptr[i] = ptr[2 * i] | (ptr[2 * i + 1] << 4);
    }
    if (bytes & 1) {
      ptr[bytes >> 1] = ptr[bytes - 1];
    }
  }
}

template <class Config>
class BlockscaledTestbed {
 public:
  using ElementA  = typename Config::ElementA;
  using ElementB  = typename Config::ElementB;
  using ElementSF = typename Config::ElementSF;
  using ElementD  = typename Config::ElementD;
  using ElementAccumulator = typename Config::ElementAccumulator;
  using LayoutA = typename Config::LayoutA;
  using LayoutB = typename Config::LayoutB;
  using LayoutC = cutlass::layout::RowMajor;

  static constexpr int SFVecSize = Config::SFVecSize;
  using ElementPairA = cute::tuple<ElementA, ElementSF, cute::Int<SFVecSize>>;
  using ElementPairB = cute::tuple<ElementB, ElementSF, cute::Int<SFVecSize>>;

  using ArchTag = cutlass::arch::Xe4;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using TileShape = typename Config::CtaTileShape_MNK;
  using ClusterShape = typename Config::ClusterShape_MNK;

  static constexpr int AlignmentA = 512;
  static constexpr int AlignmentB = 512;
  static constexpr int AlignmentC = 512;

  using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using EpilogueOperation = cutlass::epilogue::fusion::EltAct<
      cutlass::epilogue::thread::Identity, ElementD, ElementD>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      void, LayoutC, AlignmentC,
      ElementD, LayoutC, AlignmentC,
      EpilogueScheduleType,
      EpilogueOperation
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementPairA, LayoutA, AlignmentA,
      ElementPairB, LayoutB, AlignmentB,
      cute::tuple<ElementAccumulator, ElementD>,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<Config::PipelineStages>,
      cutlass::gemm::KernelTmaWarpSpecializedXe4<Config::PipelineStages, 1>
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void
  >;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using ProblemShape = typename GemmKernel::ProblemShape;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  bool run(ProblemShape problem_shape) {
    initialize(problem_shape);
    launch(problem_shape);
    queue_.wait();
    bool passed = validate(problem_shape);
    release();
    return passed;
  }

 private:
  sycl::queue queue_{};
  std::string smem_info_{};

  ElementA* A_ = nullptr;
  ElementB* B_ = nullptr;
  ElementSF* SFA_ = nullptr;
  ElementSF* SFB_ = nullptr;
  ElementD* D_ = nullptr;
  ElementA* A_ref_ = nullptr;
  ElementB* B_ref_ = nullptr;

  uint32_t sizeA_ = 0;
  uint32_t sizeB_ = 0;
  uint32_t sizeD_ = 0;
  uint32_t sizeSFA_ = 0;
  uint32_t sizeSFB_ = 0;

  StrideA stride_A_{};
  StrideB stride_B_{};
  StrideC stride_C_{};
  StrideD stride_D_{};

  void release() {
    if (A_) { sycl::free(A_, queue_); A_ = nullptr; }
    if (B_) { sycl::free(B_, queue_); B_ = nullptr; }
    if (SFA_) { sycl::free(SFA_, queue_); SFA_ = nullptr; }
    if (SFB_) { sycl::free(SFB_, queue_); SFB_ = nullptr; }
    if (D_) { sycl::free(D_, queue_); D_ = nullptr; }
    if (A_ref_) { sycl::free(A_ref_, queue_); A_ref_ = nullptr; }
    if (B_ref_) { sycl::free(B_ref_, queue_); B_ref_ = nullptr; }
  }

  void initialize(ProblemShape problem_shape) {
    release();

    auto [mat_m, mat_n, mat_k, mat_l] = problem_shape;
    sizeA_ = size(select<0,2,3>(problem_shape));
    sizeB_ = size(select<1,2,3>(problem_shape));
    sizeD_ = size(select<0,1,3>(problem_shape));

    int sf_k = mat_k / SFVecSize;
    sizeSFA_ = mat_m * sf_k * mat_l;
    sizeSFB_ = mat_n * sf_k * mat_l;

    A_ = sycl::malloc_shared<ElementA>(sizeA_, queue_);
    B_ = sycl::malloc_shared<ElementB>(sizeB_, queue_);
    D_ = sycl::malloc_shared<ElementD>(sizeD_, queue_);
    SFA_ = sycl::malloc_shared<ElementSF>(sizeSFA_, queue_);
    SFB_ = sycl::malloc_shared<ElementSF>(sizeSFB_, queue_);

    auto A_ten = make_tensor(make_gmem_ptr(A_), make_layout(make_shape(mat_m, mat_k)));
    auto B_ten = make_tensor(make_gmem_ptr(B_), make_layout(make_shape(mat_n, mat_k)));

    constexpr uint64_t seed_base = 42;
    blockscaled_random_fill_fp4(A_ten, seed_base + 2022);
    blockscaled_random_fill_fp4(B_ten, seed_base + 2021);

    A_ref_ = sycl::malloc_shared<ElementA>(sizeA_, queue_);
    B_ref_ = sycl::malloc_shared<ElementB>(sizeB_, queue_);
    std::memcpy(A_ref_, A_, sizeA_ * sizeof(ElementA));
    std::memcpy(B_ref_, B_, sizeB_ * sizeof(ElementB));

    blockscaled_subbyte_pack(A_ten);
    blockscaled_subbyte_pack(B_ten);

    auto SFA_ten = make_tensor(make_gmem_ptr(SFA_), make_layout(make_shape(mat_m, sf_k)));
    auto SFB_ten = make_tensor(make_gmem_ptr(SFB_), make_layout(make_shape(mat_n, sf_k)));
    blockscaled_random_fill_sf(SFA_ten, seed_base + 2024);
    blockscaled_random_fill_sf(SFB_ten, seed_base + 2025);

    auto D_ten = make_tensor(make_gmem_ptr(D_), make_layout(make_shape(mat_m, mat_n)));
    blockscaled_zero_fill(D_ten);

    stride_A_ = cutlass::make_cute_packed_stride(StrideA{}, select<0,2,3>(problem_shape));
    stride_B_ = cutlass::make_cute_packed_stride(StrideB{}, select<1,2,3>(problem_shape));
    stride_C_ = cutlass::make_cute_packed_stride(StrideC{}, select<0,1,3>(problem_shape));
    stride_D_ = cutlass::make_cute_packed_stride(StrideD{}, select<0,1,3>(problem_shape));
  }

  void launch(ProblemShape problem_shape) {
    auto [cluster_size_x, cluster_size_y, cluster_size_z] = ClusterShape{};
    sycl::range<3> cluster_size(cluster_size_z, cluster_size_y, cluster_size_x);
    auto num_groups = ceil_div(problem_shape, TileShape{});

    auto args = typename Gemm::GemmKernel::Arguments {
      problem_shape,
      { A_, stride_A_, B_, stride_B_, SFA_, SFB_ },
      { typename CollectiveEpilogue::FusionCallbacks::Arguments{}, nullptr, stride_C_, D_, stride_D_ }
    };

    GemmKernel kernel;
    auto params = kernel.to_underlying_arguments(args, nullptr);

    dim3 const grid = GemmKernel::get_grid_shape(params);
    dim3 const block = GemmKernel::get_block_shape();

    range<3> group_range(grid.z, grid.y, grid.x);
    range<3> local_range(block.z, block.y, block.x);

    int smem_size = 0;
    cutlass::SyclClusterLaunchParams launch_params = {group_range, local_range, cluster_size, smem_size, queue_};
    cutlass::launch_kernel_on_cluster(launch_params, kernel, params).wait();
  }

  bool validate(ProblemShape problem_shape) {
    auto [mat_m, mat_n, mat_k, mat_l] = problem_shape;
    constexpr auto layout_a = std::is_same_v<LayoutA, cutlass::layout::RowMajor>
        ? mem_layout::row_major : mem_layout::col_major;
    constexpr auto layout_b = std::is_same_v<LayoutB, cutlass::layout::RowMajor>
        ? mem_layout::row_major : mem_layout::col_major;
    int sf_k = mat_k / SFVecSize;

    for (int mat_i = 0; mat_i < mat_l; ++mat_i) {
      ElementA* ptr_A = A_ref_ + mat_i * mat_m * mat_k;
      ElementB* ptr_B = B_ref_ + mat_i * mat_n * mat_k;
      ElementD* ptr_D = D_ + mat_i * mat_m * mat_n;
      ElementSF* ptr_SFA = SFA_ + mat_i * mat_m * sf_k;
      ElementSF* ptr_SFB = SFB_ + mat_i * mat_n * sf_k;

      uint32_t err_cnt = validate_mxfp_gemm_result<ElementA, ElementB, ElementD, ElementSF>(
        ptr_A, ptr_B, ptr_D,
        mat_m, mat_n, mat_k,
        true, true,
        ptr_SFA, ptr_SFB,
        layout_a,
        layout_b,
        false,
        tolerance<ElementD>{},
        SFVecSize);

      if (err_cnt > 0) {
        std::cerr << "Test FAILED at batch " << mat_i << ", error count: " << err_cnt << std::endl;
        return false;
      }
    }
    return true;
  }

};

template <typename Config>
bool run_blockscaled_gemm()
{
  using ProblemShape = typename BlockscaledTestbed<Config>::ProblemShape;
  auto problem_shape_mnkl = ProblemShape{};
  cute::fill_int_tuple_from(problem_shape_mnkl, Config::ProblemShape_MNKL);
  BlockscaledTestbed<Config> testbed;
  return testbed.run(problem_shape_mnkl);
}

} // namespace xe4
} // namespace gemm
} // namespace test
} // namespace cutlass
