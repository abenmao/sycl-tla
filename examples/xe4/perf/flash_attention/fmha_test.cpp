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

#include <cmath>
#include <algorithm>
#include <vector>

// FMHA kernel infrastructure from fmha3 (resolved via CMake include path)
#include <cute/arch/mma_xe4.hpp>
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "sycl_common.hpp"
#include "cute/util/compat.hpp"
#include "kernel/xe4_tile_scheduler.hpp"
#include "kernel/xe4_fmha_fwd_kernel.hpp"
#include "collective/xe4_fmha_fwd_epilogue.hpp"
#include "collective/xe4_fmha_fwd_softmax_epilogue.hpp"

// Test framework
#include "fmha_params.hpp"
#include "testFixture.hpp"

using ::testing::TestWithParam;
using namespace cute;

// ============================================================================
// Options: runtime parameters for FMHA kernel launch
// ============================================================================
struct Options {
  int batch = 1;
  int num_heads = 1;
  int seq_len_qo = 512;
  int seq_len_kv = 512;
  int head_size_qk = 128;
  int head_size_vo = 128;
  int iterations = 0;
  int warmup_iterations = 0;
  float softmax_scale;
};

// ============================================================================
// SYCL kernel name tag — templated for unique mangled name per kernel config
// ============================================================================
template <class FMHAKernelType>
class FlashAttentionKernel;

using LayoutQ = cutlass::layout::RowMajor;
using LayoutK = cutlass::layout::ColumnMajor;
using LayoutV = cutlass::layout::RowMajor;
using LayoutO = cutlass::layout::RowMajor;

// ============================================================================
// ExampleRunner: allocates data, launches kernel, verifies output
// ============================================================================
template <class GemmKernel> 
struct ExampleRunner {
  
  using ElementQ = typename GemmKernel::ElementQ;
  using ElementK = typename GemmKernel::ElementK;
  using ElementV = typename GemmKernel::ElementV;
  using ElementAccum = typename GemmKernel::ElementAccum;
  using ElementOutput = typename GemmKernel::ElementOutput;

  using StrideQ = typename GemmKernel::StrideQ;
  using StrideK = typename GemmKernel::StrideK;
  using StrideV = typename GemmKernel::StrideV;
  using StrideO = typename GemmKernel::StrideO;

  using ProblemShape = typename GemmKernel::ProblemShape;

  cutlass::DeviceAllocation<ElementQ> block_Q;
  cutlass::DeviceAllocation<ElementK> block_K;
  cutlass::DeviceAllocation<ElementV> block_V;
  cutlass::DeviceAllocation<ElementOutput> block_O;
  cutlass::DeviceAllocation<ElementOutput> block_ref_O;

  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;

  sycl::queue q;

  template <typename T>
  bool relatively_equal(T a, T b, T epsilon, T nonzero_floor) {
    T abs_A = std::abs(a);
    T abs_B = std::abs(b);
    T diff = std::abs(a - b);
    T zero = T(0);

    if (a == b) {
      return true;
    }
    else if (a == zero || b == zero || (abs_A + abs_B) < nonzero_floor) {
      return diff < epsilon * nonzero_floor;
    }
    return diff < epsilon * (abs_A + abs_B);
  }

  template <typename Element>
  bool TensorCompareRelativelyEqual(
    const Element* ptr_A,
    const Element* ptr_B,
    size_t num_elements,
    Element epsilon,
    Element nonzero_floor) {

    std::vector<Element> host_A(num_elements);
    std::vector<Element> host_B(num_elements);

    compat::memcpy(host_A.data(), ptr_A, num_elements * sizeof(Element));
    compat::memcpy(host_B.data(), ptr_B, num_elements * sizeof(Element));
    
    compat::wait();

    size_t total_mismatch = 0;
    for (size_t i = 0; i < num_elements; ++i) {
      bool ok = relatively_equal(host_A[i], host_B[i], epsilon, nonzero_floor);
      if (!ok) {
        std::cout << "ref: " << host_A[i] << ", res: " << host_B[i] << ", diff abs: " << std::abs(host_A[i] - host_B[i]) << ", rate: " << host_A[i] / host_B[i] <<std::endl;
        total_mismatch++;
      }
    }
    if (total_mismatch > 0) {
      std::cout << "Failed: " << total_mismatch << "/" << num_elements << " mismatch";
      return false;
    }
    return true;
  }

  bool verify(ProblemShape problem_shape) {

    auto [batch, num_heads, head_size_qk, head_size_vo] = cute::select<0, 1, 4, 5>(problem_shape);
    int seq_len_qo = get<2>(problem_shape);
    int seq_len_kv = get<3>(problem_shape);

    int offset_q = 0;
    int offset_k = 0;
    int offset_v = 0;
    int offset_o = 0;

    std::vector<ElementOutput> host_ref_all(batch * num_heads * seq_len_qo * head_size_vo);
    int ref_offset = 0;

    for (int b = 0; b < batch; b++) {
      for (int h = 0; h < num_heads; h++) {
        std::vector<ElementAccum> host_S(seq_len_qo * seq_len_kv);
        std::vector<ElementV> host_P(seq_len_qo * seq_len_kv);
        
        std::vector<ElementQ> host_Q_data(seq_len_qo * head_size_qk);
        std::vector<ElementK> host_K_data(seq_len_kv * head_size_qk);
        std::vector<ElementV> host_V_data(seq_len_kv * head_size_vo);
        std::vector<ElementOutput> host_O_ref_data(seq_len_qo * head_size_vo);
        
        compat::memcpy(host_Q_data.data(), block_Q.get() + offset_q, host_Q_data.size() * sizeof(ElementQ));
        compat::memcpy(host_K_data.data(), block_K.get() + offset_k, host_K_data.size() * sizeof(ElementK));
        compat::memcpy(host_V_data.data(), block_V.get() + offset_v, host_V_data.size() * sizeof(ElementV));
        compat::wait();

        cutlass::TensorRef ref_Q(host_Q_data.data(), LayoutQ::packed({seq_len_qo, head_size_qk}));
        cutlass::TensorRef ref_K(host_K_data.data(), LayoutK::packed({head_size_qk, seq_len_kv}));
        cutlass::TensorRef ref_V(host_V_data.data(), LayoutV::packed({seq_len_kv, head_size_vo}));
        cutlass::TensorRef ref_S(host_S.data(), LayoutQ::packed({seq_len_qo, seq_len_kv}));
        cutlass::TensorRef ref_P(host_P.data(), LayoutQ::packed({seq_len_qo, seq_len_kv}));
        cutlass::TensorRef ref_O(host_O_ref_data.data(), LayoutO::packed({seq_len_qo, head_size_vo}));

        // S = Q * K^T
        cutlass::reference::host::GemmComplex(
          {seq_len_qo, seq_len_kv, head_size_qk},
          1.f,
          ref_Q, cutlass::ComplexTransform::kNone,
          ref_K, cutlass::ComplexTransform::kNone,
          0.f, ref_S, ref_S, ElementAccum(0),
          1, seq_len_qo * head_size_qk, seq_len_kv * head_size_qk,
          seq_len_qo * seq_len_kv, seq_len_qo * seq_len_kv
        );
        
        // Row-wise max of S
        std::vector<ElementAccum> max_vec(seq_len_qo, -INFINITY);
        for (int row = 0; row < seq_len_qo; row++) {
          int idx = row * seq_len_kv;
          max_vec[row] = host_S[idx++];
          for (int col = 1; col < seq_len_kv; col++, idx++) {
            if (max_vec[row] < host_S[idx]) {
              max_vec[row] = host_S[idx];
            }
          }
        }

        // exp(S - max) / sqrt(head_size)
        for (int row = 0; row < seq_len_qo; row++) {
          int idx = row * seq_len_kv;
          for (int col = 0; col < seq_len_kv; col++, idx++) {
            host_S[idx] = expf((host_S[idx] - max_vec[row]) / std::sqrt(static_cast<ElementAccum>((head_size_qk))));
          }
        }

        // Row-wise sum and normalize (softmax)
        std::vector<ElementAccum> sum_vec(seq_len_qo, ElementAccum{0});
        for (int row = 0; row < seq_len_qo; row++) {
          int idx = row * seq_len_kv;
          for (int col = 0; col < seq_len_kv; col++, idx++) {
            sum_vec[row] += host_S[idx];
          }
          idx = row * seq_len_kv;
          for (int col = 0; col < seq_len_kv; col++, idx++) {
            host_S[idx] /= sum_vec[row];
          }
        }

        std::vector<ElementV> host_P_converted(host_S.size());
        for (size_t p = 0; p < host_P_converted.size(); p++) {
          host_P_converted[p] = static_cast<ElementV>(host_S[p]);
          host_P[p] = host_P_converted[p];
        }

        // O = P * V
        cutlass::reference::host::GemmComplex(
          {seq_len_qo, head_size_vo, seq_len_kv},
          1.f,
          ref_P, cutlass::ComplexTransform::kNone,
          ref_V, cutlass::ComplexTransform::kNone,
          0.f, ref_O, ref_O, ElementAccum(0),
          1, seq_len_qo * seq_len_kv, seq_len_kv * head_size_vo,
          seq_len_qo * head_size_vo, seq_len_qo * head_size_vo
        );

        std::copy(host_O_ref_data.begin(), host_O_ref_data.end(), 
                  host_ref_all.begin() + ref_offset);
        ref_offset += seq_len_qo * head_size_vo;

        offset_q += seq_len_qo * head_size_qk;
        offset_k += seq_len_kv * head_size_qk;
        offset_v += seq_len_kv * head_size_vo;
        offset_o += seq_len_qo * head_size_vo;
      }
    }
    
    size_t total_output_size = block_O.size();
    std::vector<ElementOutput> host_device_output(total_output_size);
    compat::memcpy(host_device_output.data(), block_O.get(), total_output_size * sizeof(ElementOutput));
    compat::wait();
    
    return TensorCompareRelativelyEqual<ElementOutput>(host_ref_all.data(), host_device_output.data(), host_device_output.size(), 0.05, 0.05);
  }

  ProblemShape initialize(const Options &options) {
    auto problem_shape = cute::make_tuple(
      options.batch, options.num_heads,
      options.seq_len_qo, options.seq_len_kv,
      options.head_size_qk, options.head_size_vo);

    auto [batch, num_heads, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo] = problem_shape;

    stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, cute::make_shape(seq_len_qo, head_size_qk, batch * num_heads));
    stride_K = cutlass::make_cute_packed_stride(StrideK{}, cute::make_shape(seq_len_kv, head_size_qk, batch * num_heads));
    stride_V = cutlass::make_cute_packed_stride(StrideV{}, cute::make_shape(head_size_vo, seq_len_kv, batch * num_heads));
    stride_O = cutlass::make_cute_packed_stride(StrideO{}, cute::make_shape(seq_len_qo, head_size_vo, batch * num_heads));

    size_t size_Q = static_cast<std::size_t>(batch) * num_heads * seq_len_qo * head_size_qk;
    size_t size_K = static_cast<std::size_t>(batch) * num_heads * seq_len_kv * head_size_qk;
    size_t size_V = static_cast<std::size_t>(batch) * num_heads * seq_len_kv * head_size_vo;
    size_t size_O = static_cast<std::size_t>(batch) * num_heads * seq_len_qo * head_size_vo;
    
    block_Q.reset(size_Q);
    block_K.reset(size_K);
    block_V.reset(size_V);
    block_O.reset(size_O);
    block_ref_O.reset(size_O);
    
    std::vector<ElementQ> host_Q(size_Q);
    std::vector<ElementK> host_K(size_K);
    std::vector<ElementV> host_V(size_V);
    
    uint64_t seed = 2025;
    cutlass::reference::host::BlockFillRandomUniform(host_Q.data(), host_Q.size(), seed, ElementQ(1), ElementQ(-1), 0);
    cutlass::reference::host::BlockFillRandomUniform(host_K.data(), host_K.size(), seed + 1, ElementK(1), ElementK(-1), 0);
    cutlass::reference::host::BlockFillRandomUniform(host_V.data(), host_V.size(), seed + 2, ElementV(1), ElementV(-1), 0);
    
    compat::memcpy(block_Q.get(), host_Q.data(), size_Q * sizeof(ElementQ));
    compat::memcpy(block_K.get(), host_K.data(), size_K * sizeof(ElementK));
    compat::memcpy(block_V.get(), host_V.data(), size_V * sizeof(ElementV));
    
    return problem_shape;
  }
  
  void run(typename GemmKernel::Params params) {
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    
    dim3 const block = GemmKernel::get_block_shape();
    dim3 const grid = GemmKernel::get_grid_shape(params);

    std::cout << "block: " << block.x << ", " << block.y << ", " << block.z << std::endl;
    std::cout << "grid: " << grid.x << ", " << grid.y << ", " << grid.z << std::endl;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    GemmKernel kernel;

    auto* q_ptr = block_Q.get();
    auto* k_ptr = block_K.get();
    auto* v_ptr = block_V.get();
    auto* o_ptr = block_O.get();

    q.parallel_for<FlashAttentionKernel<GemmKernel>>(sycl::nd_range<3>{sycl_grid * sycl_block, sycl_block},
      [=](sycl::nd_item<3> item) {

        auto params_workaround = params;
        params_workaround.mainloop.tma_load_Q.cache_.set_gmem_ptr(q_ptr);
        params_workaround.mainloop.tma_load_K.cache_.set_gmem_ptr(k_ptr);
        params_workaround.mainloop.tma_load_V.cache_.set_gmem_ptr(v_ptr);
        params_workaround.epilogue.tma_store_O.cache_.set_gmem_ptr(o_ptr);

        kernel(params_workaround);
      }).wait();
  }

  cutlass::Status run(const Options &options) {
    ProblemShape problem_shape = initialize(options);

    typename GemmKernel::Arguments arguments{
      problem_shape,
      {block_Q.get(), stride_Q, block_K.get(), stride_K, block_V.get(), stride_V},
      {options.softmax_scale},
      {block_O.get(), stride_O}
    };

    auto params = GemmKernel::to_underlying_arguments(arguments);
    
    run(params);
    q.wait();
    
    bool passed = verify(problem_shape);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

    if (!passed) {
      return cutlass::Status::kErrorInternal;
    }

    if (options.iterations > 0) {
      GPU_Clock timer;

      if (options.warmup_iterations > 0) {
        std::cout << "Warm up " << options.warmup_iterations << " iterations..." << std::endl;
        for (int i = 0; i < options.warmup_iterations; ++i) {
          run(params);
        }
        compat::wait();
      }

      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        run(params);
      }
      compat::wait();

      double avg_time = timer.seconds() / options.iterations;
      std::cout << "\tBatch: " << options.batch
                << "\tSeqLenQO: " << options.seq_len_qo
                << "\tSeqLenKV: " << options.seq_len_kv
                << "\tHeadSizeQK: " << options.head_size_qk
                << "\tHeadSizeVO: " << options.head_size_vo
                << "\tIterations: " << options.iterations;

      printf("\nPerformance:   %6.4f  ms\n\n", avg_time * 1000);
    }
    return cutlass::Status::kSuccess;
  }
};

// ============================================================================
// FMHAConfig: builds kernel types from tile/softmax/type parameters
// ============================================================================
template <typename TileShape, 
          int NumSoftmaxWarps = 16, int NumThreadPerRow = 16, 
          int SoftmaxUnroll = 2, int SoftmaxNumStage = 2, 
          bool IsPersistent = false> 
struct FMHAConfig {

  template <typename ProblemConfig>
  static int run(const Options &options) {
    using ProblemShape = cute::tuple<int, int, int, int, int, int>;
    using ClusterShape = Shape<_1, _1, _1>;

    using TileShapeQK_MNK = decltype(select<0, 2, 3>(TileShape{}));
    using TileShapePV_MNK = decltype(select<0, 1, 2>(TileShape{}));

    using ElementInputQ = typename ProblemConfig::ElementInputQ;
    using ElementInputKV = typename ProblemConfig::ElementInputKV;
    using ElementS = typename ProblemConfig::ElementS;
    using ElementP = typename ProblemConfig::ElementP;
    using ElementAccumulator = typename ProblemConfig::ElementAccumulator;
    using ElementOutput = typename ProblemConfig::ElementOutput;

    constexpr auto majorQ = cute::AMMA::Major::K;
    constexpr auto majorK = cute::AMMA::Major::K;
    constexpr auto majorP = cute::AMMA::Major::K;
    constexpr auto majorV = cute::AMMA::Major::MN;

    constexpr int PipelineStages = 2;
    constexpr int PipelineStagesQ = 2;

    using SmemLayoutAtomQ = decltype(make_layout(
      cute::select<0, 2>(TileShapeQK_MNK{}), GenRowMajor{}));
    using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQ{},
      make_shape(shape<0>(TileShapeQK_MNK{}), shape<2>(TileShapeQK_MNK{}), Int<PipelineStagesQ>{})));

    using SmemLayoutAtomK = decltype(make_layout(
      cute::select<1, 2>(TileShapeQK_MNK{}), GenRowMajor{}));
    using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomK{},
      make_shape(shape<1>(TileShapeQK_MNK{}), shape<2>(TileShapeQK_MNK{}), Int<PipelineStages>{})));

    using SmemLayoutAtomV = decltype(make_layout(
      cute::select<1, 2>(TileShapePV_MNK{}), GenColMajor{}));
    using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomV{},
      make_shape(shape<1>(TileShapePV_MNK{}), shape<2>(TileShapePV_MNK{}), Int<PipelineStages>{})));

    using SmemLayoutOutputAccum = decltype(make_layout(
      cute::select<0, 1>(TileShapePV_MNK{}), GenRowMajor{}));
    using SmemLayoutOutput = decltype(make_layout(
      cute::select<0, 1>(TileShapePV_MNK{}), GenRowMajor{}));

    using TMACopyAtomQ = cute::xe4::ASYNC_TENSOR_LOAD<slm_matrix_type::type1, size<2>(TileShapeQK_MNK{})>;
    using TMACopyAtomK = cute::xe4::ASYNC_TENSOR_LOAD<slm_matrix_type::type1, size<2>(TileShapeQK_MNK{})>;
    using TMACopyAtomV = cute::xe4::ASYNC_TENSOR_LOAD<slm_matrix_type::type1, size<1>(TileShapePV_MNK{})>;
    using TMACopyAtomO = cute::xe4::ASYNC_TENSOR_STORE<slm_matrix_type::type1, size<1>(TileShapePV_MNK{})>;

    using TiledMmaQK = decltype(cute::make_tiled_mma(
      cute::AMMA::ss_op_selector<
      ElementS, ElementInputQ, ElementInputKV, ElementS,
      TileShapeQK_MNK, ClusterShape, majorQ, majorK>()));

    using TiledMmaPV = decltype(cute::make_tiled_mma(
      cute::AMMA::ss_op_selector<
      ElementAccumulator, ElementInputKV, ElementInputKV, ElementAccumulator,
      TileShapePV_MNK, ClusterShape, majorP, majorV>()));

    using CollectiveMainloop = cutlass::flash_attention::collective::CollectiveMmaAttention<
      ProblemShape, TileShape,
      ElementInputQ, ElementInputKV, ElementInputKV,
      ElementS, ElementP, ElementAccumulator, ElementOutput,
      cutlass::gemm::TagToStrideA_t<LayoutQ>,
      cutlass::gemm::TagToStrideB_t<LayoutK>,
      cutlass::gemm::TagToStrideB_t<LayoutV>,
      TiledMmaQK, TiledMmaPV,
      SmemLayoutQ, SmemLayoutK, SmemLayoutV,
      SmemLayoutOutputAccum, SmemLayoutOutput,
      TMACopyAtomQ, TMACopyAtomK, TMACopyAtomV,
      PipelineStages, PipelineStagesQ>;

    using CollectiveSoftmaxEpilogue = cutlass::flash_attention::collective::CollectiveSoftmaxEpilogue<
      TileShape, ElementAccumulator, ElementOutput, ElementS, ElementP,
      NumSoftmaxWarps, NumThreadPerRow, SoftmaxUnroll, SoftmaxNumStage>;

    using CollectiveEpilogue = cutlass::flash_attention::collective::CollectiveEpilogueAttention<
      ProblemShape, TileShape, ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutO>, SmemLayoutOutput, TMACopyAtomO>;

    using TileScheduler = typename std::conditional<
      IsPersistent, 
      cutlass::flash_attention::kernel::XeFlashPersistentTileScheduler,
      cutlass::flash_attention::kernel::XeFlashIndividualTileScheduler>::type;

    using GemmKernel = cutlass::flash_attention::kernel::GemmUniversalAttention<
      ProblemShape, CollectiveMainloop, CollectiveSoftmaxEpilogue,
      CollectiveEpilogue, TileScheduler>;
    
    ExampleRunner<GemmKernel> runner;
    auto status = runner.run(options);
    return (status == cutlass::Status::kSuccess) ? 0 : -1;
  }
};

// Global parameter list for FMHA operations
std::deque<TestParamInfo> FMHAOpParamsList;

// ProblemConfig at file scope (external linkage) for proper SYCL kernel name generation
template <typename ElementQ_, typename ElementKV_, typename ElementS_,
          typename ElementO_, typename ElementAccumulator_>
struct FMHAProblemConfig {
    using ElementInputQ = ElementQ_;
    using ElementInputKV = ElementKV_;
    using ElementS = ElementS_;
    using ElementP = ElementKV_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementOutput = ElementO_;
};

// Each unique tile + softmax + type configuration is an explicit function to avoid
// compiling all 11 tile configs × 5 type combos = 55 SYCL kernels into one binary.
// The SYCL AOT compiler can fail or produce invalid device code when too many
// kernel types with complex mangled names are compiled in a single translation unit.
//
// DISPATCH_FMHA_TYPED_CONFIG: dispatches a single (tile, softmax, type) combo.
#define DISPATCH_FMHA_TYPED_CONFIG(tile_q_val, tile_v_val, tile_kv_val, tile_qk_val, \
                                   sw_val, tpr_val, su_val, sns_val, persist_val,     \
                                   ElemQ, ElemKV, ElemS, ElemO, ElemAcc)               \
    if (tile_q == tile_q_val && tile_v == tile_v_val &&                                \
        tile_kv == tile_kv_val && tile_qk == tile_qk_val &&                            \
        num_sw == sw_val && num_tpr == tpr_val &&                                      \
        softmax_u == su_val && softmax_ns == sns_val &&                                \
        is_persist == persist_val) {                                                   \
        using TileShape = Shape<Int<tile_q_val>, Int<tile_v_val>,                      \
                                Int<tile_kv_val>, Int<tile_qk_val>>;                   \
        using Config = FMHAConfig<TileShape, sw_val, tpr_val,                          \
                                  su_val, sns_val, persist_val>;                       \
        using PC = FMHAProblemConfig<ElemQ, ElemKV, ElemS, ElemO, ElemAcc>;             \
        return Config::template run<PC>(options);                                      \
    }

// Dispatch function for FP16 all
// NOTE: Only include configs that are actually used in fmha_test_params to minimize
// the number of SYCL kernels compiled into the binary. Each config compiles a unique
// kernel; too many can cause AOT compilation issues with dual fsycl-targets.
static int dispatchFMHA_fp16(const Options& options,
                             int tile_q, int tile_v, int tile_kv, int tile_qk,
                             int num_sw, int num_tpr, int softmax_u, int softmax_ns,
                             bool is_persist) {
    // Non-persistent: tile 128/128/512/128, softmax 16/16/2/2 (L0 test 0)
    DISPATCH_FMHA_TYPED_CONFIG(128, 128, 512, 128, 16, 16, 2, 2, false, fp16, fp16, fp16, fp16, float)
    // Persistent: tile 128/128/512/128, softmax 16/16/1/1 (L0 test 1)
    DISPATCH_FMHA_TYPED_CONFIG(128, 128, 512, 128, 16, 16, 1, 1, true,  fp16, fp16, fp16, fp16, float)
    return -1; // no config matched
}

// Dispatch functions for other type combos — currently no enabled tests use these.
// Add DISPATCH_FMHA_TYPED_CONFIG entries here when enabling tests with these types.
static int dispatchFMHA_fp16_floatS(const Options& /*options*/,
                                    int /*tile_q*/, int /*tile_v*/, int /*tile_kv*/, int /*tile_qk*/,
                                    int /*num_sw*/, int /*num_tpr*/, int /*softmax_u*/, int /*softmax_ns*/,
                                    bool /*is_persist*/) {
    return -1;
}

static int dispatchFMHA_fp16_floatO(const Options& /*options*/,
                                    int /*tile_q*/, int /*tile_v*/, int /*tile_kv*/, int /*tile_qk*/,
                                    int /*num_sw*/, int /*num_tpr*/, int /*softmax_u*/, int /*softmax_ns*/,
                                    bool /*is_persist*/) {
    return -1;
}

static int dispatchFMHA_bf16(const Options& /*options*/,
                             int /*tile_q*/, int /*tile_v*/, int /*tile_kv*/, int /*tile_qk*/,
                             int /*num_sw*/, int /*num_tpr*/, int /*softmax_u*/, int /*softmax_ns*/,
                             bool /*is_persist*/) {
    return -1;
}

static int dispatchFMHA_bf16_floatS(const Options& /*options*/,
                                    int /*tile_q*/, int /*tile_v*/, int /*tile_kv*/, int /*tile_qk*/,
                                    int /*num_sw*/, int /*num_tpr*/, int /*softmax_u*/, int /*softmax_ns*/,
                                    bool /*is_persist*/) {
    return -1;
}

#undef DISPATCH_FMHA_TYPED_CONFIG

/*
 * FMHA Operator Test Fixture
 * Uses FMHAConfig and ExampleRunner from xe4_fmha_fwd_runner.hpp
 */
class FMHAOperator : public testFixture {
public:
    void runTest() override {
        auto* params = getTestParams<FMHATestParams>();
        if (params == nullptr) {
            FAIL() << "Invalid test parameters";
            return;
        }

        // Check if test is enabled
        if (params->state != "enabled") {
            GTEST_SKIP() << "Test is disabled. Skipping execution";
            return;
        }

        // Print test configuration
        std::cout << "\n========== FMHA Test Configuration ==========\n";
        std::cout << "Problem Shape:\n";
        std::cout << "  Batch=" << params->batch 
                  << ", NumHeads=" << params->num_heads
                  << ", SeqLenQO=" << params->seq_len_qo 
                  << ", SeqLenKV=" << params->seq_len_kv
                  << ", HeadSizeQK=" << params->head_size_qk 
                  << ", HeadSizeVO=" << params->head_size_vo << "\n";
        std::cout << "Tile Shape:\n";
        std::cout << "  Q_blk=" << params->tile_q_blk
                  << ", V_head_dim=" << params->tile_v_head_dim
                  << ", KV_blk=" << params->tile_kv_blk
                  << ", QK_head_dim=" << params->tile_qk_head_dim << "\n";
        std::cout << "Softmax Config:\n";
        std::cout << "  Warps=" << params->num_softmax_warps
                  << ", ThreadsPerRow=" << params->num_thread_per_row
                  << ", Unroll=" << params->softmax_unroll
                  << ", Stages=" << params->softmax_num_stage << "\n";
        std::cout << "Kernel Mode: " << (params->is_persistent ? "Persistent" : "Non-Persistent") << "\n";
        std::cout << "Data Types:\n";
        std::cout << "  Q=" << params->dtype_q 
                  << ", KV=" << params->dtype_kv
                  << ", S=" << params->dtype_s 
                  << ", O=" << params->dtype_o 
                  << ", Acc=" << params->dtype_acc << "\n";
        std::cout << "=============================================\n";

        // Create Options from test parameters
        Options options;
        options.batch = params->batch;
        options.num_heads = params->num_heads;
        options.seq_len_qo = params->seq_len_qo;
        options.seq_len_kv = params->seq_len_kv;
        options.head_size_qk = params->head_size_qk;
        options.head_size_vo = params->head_size_vo;
        options.iterations = 1;
        if (params->softmax_scale == 0.0f) {
            options.softmax_scale = 1.0f / std::sqrt(static_cast<float>(params->head_size_qk));
        } else {
            options.softmax_scale = params->softmax_scale;
        }

        int tile_q = params->tile_q_blk;
        int tile_v = params->tile_v_head_dim;
        int tile_kv = params->tile_kv_blk;
        int tile_qk = params->tile_qk_head_dim;
        int num_sw = params->num_softmax_warps;
        int num_tpr = params->num_thread_per_row;
        int softmax_u = params->softmax_unroll;
        int softmax_ns = params->softmax_num_stage;
        bool is_persist = params->is_persistent;

        // Timing measurement
        auto start = std::chrono::high_resolution_clock::now();

        // Dispatch to the correct type-specific dispatch function.
        // Each dispatch function is a separate (non-template) function, so only
        // the kernel types for that element type combo get compiled.
        int result = -1;
        if (params->dtype_q == "fp16" && params->dtype_kv == "fp16" &&
            params->dtype_s == "fp16" && params->dtype_o == "fp16" &&
            params->dtype_acc == "float") {
            result = dispatchFMHA_fp16(options, tile_q, tile_v, tile_kv, tile_qk,
                                      num_sw, num_tpr, softmax_u, softmax_ns, is_persist);
        } else if (params->dtype_q == "fp16" && params->dtype_kv == "fp16" &&
                   params->dtype_s == "float" && params->dtype_o == "fp16" &&
                   params->dtype_acc == "float") {
            result = dispatchFMHA_fp16_floatS(options, tile_q, tile_v, tile_kv, tile_qk,
                                             num_sw, num_tpr, softmax_u, softmax_ns, is_persist);
        } else if (params->dtype_q == "fp16" && params->dtype_kv == "fp16" &&
                   params->dtype_s == "fp16" && params->dtype_o == "float" &&
                   params->dtype_acc == "float") {
            result = dispatchFMHA_fp16_floatO(options, tile_q, tile_v, tile_kv, tile_qk,
                                             num_sw, num_tpr, softmax_u, softmax_ns, is_persist);
        } else if (params->dtype_q == "bf16" && params->dtype_kv == "bf16" &&
                   params->dtype_s == "bf16" && params->dtype_o == "bf16" &&
                   params->dtype_acc == "float") {
            result = dispatchFMHA_bf16(options, tile_q, tile_v, tile_kv, tile_qk,
                                      num_sw, num_tpr, softmax_u, softmax_ns, is_persist);
        } else if (params->dtype_q == "bf16" && params->dtype_kv == "bf16" &&
                   params->dtype_s == "float" && params->dtype_o == "bf16" &&
                   params->dtype_acc == "float") {
            result = dispatchFMHA_bf16_floatS(options, tile_q, tile_v, tile_kv, tile_qk,
                                             num_sw, num_tpr, softmax_u, softmax_ns, is_persist);
        } else {
            FAIL() << "Unsupported data type combination: "
                   << "Q=" << params->dtype_q << " KV=" << params->dtype_kv
                   << " S=" << params->dtype_s << " O=" << params->dtype_o
                   << " Acc=" << params->dtype_acc;
            return;
        }

        if (result == -1) {
            std::cout << "Unsupported Tile/Softmax configuration:\n"
                      << "  Tile(Q=" << tile_q << ", V=" << tile_v
                      << ", KV=" << tile_kv << ", QK=" << tile_qk << ")\n"
                      << "  Softmax(Warps=" << num_sw << ", ThreadsPerRow=" << num_tpr
                      << ", Unroll=" << softmax_u << ", Stages=" << softmax_ns << ")\n"
                      << "  Persistent=" << is_persist << std::endl;
            FAIL() << "Configuration not supported";
            return;
        }
        if (result != 0) {
            FAIL() << "FMHA kernel execution failed";
            return;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Total execution time: " << duration.count() << " microseconds" << std::endl;
    }
};

/*
 * Function to create FMHA operation parameters list
 */
bool createFMHAOpParamsList(std::deque<TestParamInfo>& FMHAOpParamsList,
                            struct FMHATestParams* ptestParam,
                            int n) {
    if (FMHAOpParamsList.size() != 0)
        return true;

    for (int i = 0; i < n; i++) {
        std::string testname =
            "fmha" + std::string("_") +
            std::string(ptestParam[i].test_level) + std::string("_") +
            std::string(ptestParam[i].state) + std::string("_") +
            std::string("B_") + std::to_string(ptestParam[i].batch) + std::string("_") +
            std::string("H_") + std::to_string(ptestParam[i].num_heads) + std::string("_") +
            std::string("Tq_") + std::to_string(ptestParam[i].seq_len_qo) + std::string("_") +
            std::string("Tkv_") + std::to_string(ptestParam[i].seq_len_kv) + std::string("_") +
            std::string("Dqk_") + std::to_string(ptestParam[i].head_size_qk) + std::string("_") +
            std::string("Dvo_") + std::to_string(ptestParam[i].head_size_vo) + std::string("_") +
            std::string("dtypeQ_") + ptestParam[i].dtype_q + std::string("_") +
            std::string("dtypeKV_") + ptestParam[i].dtype_kv + std::string("_") +
            std::string("dtypeS_") + ptestParam[i].dtype_s + std::string("_") +
            std::string("dtypeO_") + ptestParam[i].dtype_o + std::string("_") +
            std::string("dtypeAcc_") + ptestParam[i].dtype_acc + std::string("_") +
            std::string("tileQ_") + std::to_string(ptestParam[i].tile_q_blk) + std::string("_") +
            std::string("tileV_") + std::to_string(ptestParam[i].tile_v_head_dim) + std::string("_") +
            std::string("tileKV_") + std::to_string(ptestParam[i].tile_kv_blk) + std::string("_") +
            std::string("tileQK_") + std::to_string(ptestParam[i].tile_qk_head_dim) + std::string("_") +
            std::string("swWarps_") + std::to_string(ptestParam[i].num_softmax_warps) + std::string("_") +
            std::string("swTPR_") + std::to_string(ptestParam[i].num_thread_per_row) + std::string("_") +
            std::string("swUnroll_") + std::to_string(ptestParam[i].softmax_unroll) + std::string("_") +
            std::string("swStages_") + std::to_string(ptestParam[i].softmax_num_stage) + std::string("_") +
            std::string("isPersistent_") + (ptestParam[i].is_persistent ? "true" : "false") + std::string("_") +
            std::string("causalMask_") + (ptestParam[i].causal_mask ? "true" : "false");

        // Add test to parameter list
        TestParamInfo::addTestToParamList(FMHAOpParamsList, testname, (void*)&ptestParam[i]);
    }
    return true;
}

// Custom name generator that includes the test name as the parameter name
struct FMHATestNameGenerator {
    std::string operator()(const ::testing::TestParamInfo<TestParamInfo>& info) const {
        return info.param.testName;
    }
};

TEST_P(FMHAOperator, FMHAOpPerfTest) {
    runTest();
}

// Instantiate the test case with custom name generator to show full test names
INSTANTIATE_TEST_SUITE_P(FMHAPerfTests, FMHAOperator,
                         ::testing::ValuesIn(true == createFMHAOpParamsList(FMHAOpParamsList,
                                                                            fmha_test_params,
                                                                            sizeof(fmha_test_params) / sizeof(FMHATestParams))
                                                 ? FMHAOpParamsList
                                                 : FMHAOpParamsList),
                         FMHATestNameGenerator());
