/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation, All rights reserved.
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
#pragma once

#include <cute/arch/mma_xe4.hpp>
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "sycl_common.hpp"
#include "cute/util/compat.hpp"
#include "kernel/xe4_tile_scheduler.hpp"
#include "kernel/xe4_fmha_fwd_kernel.hpp"
#include "collective/xe4_fmha_fwd_epilogue.hpp"
#include "collective/xe4_fmha_fwd_softmax_epilogue.hpp"

using namespace cute;

struct Options {
  bool help = false;
  bool error = false;
  int batch = 1;
  int num_heads = 1;
  int seq_len_qo = 512;
  int seq_len_kv = 512;
  int head_size_qk = 128;
  int head_size_vo = 128;
  int iterations = 100;
  float softmax_scale;

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("batch", batch, 1);
    cmd.get_cmd_line_argument("num_heads", num_heads, 1);
    cmd.get_cmd_line_argument("seq_len_qo", seq_len_qo, 512);
    cmd.get_cmd_line_argument("seq_len_kv", seq_len_kv, seq_len_qo);
    cmd.get_cmd_line_argument("head_size_vo", head_size_vo, 128);
    cmd.get_cmd_line_argument("head_size_qk", head_size_qk, head_size_vo);
    cmd.get_cmd_line_argument("iterations", iterations, 100);

    softmax_scale = 1 / sqrt(static_cast<float>(head_size_qk));
  }

  std::ostream &print_usage(std::ostream &out) const {
    out << "Xe4 FlashAttention3 Example\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement\n\n"
        << "  --batch=<int>               Sets the Batch Size of the Multi-Head Self Attention module\n"
        << "  --num_heads=<int> Sets the Number of Attention Heads of the Multi-Head Self Attention module\n"
        << "  --seq_len_qo=<int>          Sets the Sequence length of the Query input in Multi-Head Self Attention module\n"
        << "  --seq_len_kv=<int>          Sets the Sequence length of the Key-Value pair in Multi-Head Self Attention module\n"
        << "  --head_size_qk=<int>        Sets the Attention Head dimension of the 1st Matrix Multiplication in Multi-Head Self Attention module\n"
        << "  --head_size_vo=<int>        Sets the Attention Head dimension of the 2nd Matrix Multiplication in Multi-Head Self Attention module\n"
        << "  --iterations=<int>          Iterations\n\n";
    return out;
  }
};

class FlashAttentionKernel; 

using LayoutQ = cutlass::layout::RowMajor;
using LayoutK = cutlass::layout::ColumnMajor;
using LayoutV = cutlass::layout::RowMajor;
using LayoutO = cutlass::layout::RowMajor;

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
        std::cout << "ref: " << host_A[i] << ", res: " << host_B[i] << ", diff abs: " << std::abs(host_A[i] - host_B[i]) << std::endl;
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

    for (int b = 0; b < batch; b++) {
      for (int h = 0; h < num_heads; h++) {
        cutlass::DeviceAllocation<ElementAccum> block_S;
        cutlass::DeviceAllocation<ElementV> block_P;

        block_S.reset(seq_len_qo * seq_len_kv);
        block_P.reset(seq_len_qo * seq_len_kv);

        cutlass::TensorRef ref_Q(block_Q.get() + offset_q, LayoutQ::packed({seq_len_qo, head_size_qk}));
        cutlass::TensorRef ref_K(block_K.get() + offset_k, LayoutK::packed({head_size_qk, seq_len_kv}));
        cutlass::TensorRef ref_V(block_V.get() + offset_v, LayoutV::packed({seq_len_kv, head_size_vo}));
        cutlass::TensorRef ref_S(block_S.get(), LayoutQ::packed({seq_len_qo, seq_len_kv}));
        cutlass::TensorRef ref_P(block_P.get(), LayoutQ::packed({seq_len_qo, seq_len_kv}));
        cutlass::TensorRef ref_O(block_ref_O.get() + offset_o, LayoutO::packed({seq_len_qo, head_size_vo}));

        // Compute S = Q * K^T
        cutlass::reference::host::GemmComplex(
          {seq_len_qo, seq_len_kv, head_size_qk},
          1.f,
          ref_Q,
          cutlass::ComplexTransform::kNone,
          ref_K,
          cutlass::ComplexTransform::kNone,
          0.f,
          ref_S,
          ref_S,
          ElementAccum(0),
          1,                         // batch_count
          seq_len_qo * head_size_qk, // batch_stride_Q
          seq_len_kv * head_size_qk, // batch_stride_K  
          seq_len_qo * seq_len_kv,   // batch_stride_S
          seq_len_qo * seq_len_kv    // batch_stride_S
        );

        std::vector<ElementAccum> host_S(block_S.size());
        compat::memcpy<ElementAccum>(host_S.data(), block_S.get(), host_S.size());
        compat::wait();
        
        block_S.reset();
        
        // compute max element per row of S
        std::vector<ElementAccum> max_vec(seq_len_qo, -INFINITY);
        for (int row = 0; row < seq_len_qo; row++) {
          int idx = row * seq_len_kv;
          int max_idx = row;
          max_vec[max_idx] = host_S[idx++];
          for (int col = 1; col < seq_len_kv; col++, idx++) {
            if (max_vec[max_idx] < host_S[idx]) {
              max_vec[max_idx] = host_S[idx];
            }
          }
        }

        // compute exp of S
        for (int row = 0; row < seq_len_qo; row++) {
          int idx = row * seq_len_kv;
          int max_idx = row;
          for (int col = 0; col < seq_len_kv; col++, idx++) {
            host_S[idx] = expf((host_S[idx] - max_vec[max_idx]) / sqrt(static_cast<ElementAccum>((head_size_qk))));
          }
        }

        // compute sum per row of S
        std::vector<ElementAccum> sum_vec(seq_len_qo, ElementAccum{0});
        for (int row = 0; row < seq_len_qo; row++) {
          int idx = row * seq_len_kv;
          int sum_idx = row;
          for (int col = 0; col < seq_len_kv; col++, idx++) {
            sum_vec[sum_idx] += host_S[idx];
          }

          // scale each row with the sum to compute softmax
          idx = row * seq_len_kv;
          sum_idx = row;
          for (int col = 0; col < seq_len_kv; col++, idx++) {
            host_S[idx] /= sum_vec[sum_idx];
          }
        }

        std::vector<ElementV> host_P(host_S.size());
        for (int p = 0; p < host_P.size(); p++) {
          host_P[p] = static_cast<ElementV>(host_S[p]);
        }

        compat::memcpy<ElementV>(block_P.get(), host_P.data(), host_P.size());
        compat::wait();

        // Compute O = P * V^T
        cutlass::reference::host::GemmComplex(
          {seq_len_qo, head_size_vo, seq_len_kv},
          1.f,
          ref_P,
          cutlass::ComplexTransform::kNone,
          ref_V,
          cutlass::ComplexTransform::kNone,
          0.f,
          ref_O,
          ref_O,
          ElementAccum(0),
          1,                         // batch_count
          seq_len_qo * seq_len_kv,   // batch_stride_P
          seq_len_kv * head_size_vo, // batch_stride_V
          seq_len_qo * head_size_vo, // batch_stride_O
          seq_len_qo * head_size_vo  // batch_stride_O
        );

        block_P.reset();

        offset_q += seq_len_qo * head_size_qk;
        offset_k += seq_len_kv * head_size_qk;
        offset_v += seq_len_kv * head_size_vo;
        offset_o += seq_len_qo * head_size_vo;
      }
    }

    compat::wait();
    
    bool passed = TensorCompareRelativelyEqual<ElementOutput>(block_ref_O.get(), block_O.get(), block_O.size(), 0.05, 0.05);

    return passed;
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

    block_Q.reset(static_cast<std::size_t>(batch) * num_heads * seq_len_qo * head_size_qk);
    block_K.reset(static_cast<std::size_t>(batch) * num_heads * seq_len_kv * head_size_qk);
    block_V.reset(static_cast<std::size_t>(batch) * num_heads * seq_len_kv * head_size_vo);
    block_O.reset(static_cast<std::size_t>(batch) * num_heads * seq_len_qo * head_size_vo);
    block_ref_O.reset(static_cast<std::size_t>(batch) * num_heads * seq_len_qo * head_size_vo);
    
    uint64_t seed = 2025;
    cutlass::reference::host::BlockFillRandomUniform(block_Q.get(), block_Q.size(), seed, ElementQ(1), ElementQ(-1), 0);
    cutlass::reference::host::BlockFillRandomUniform(block_K.get(), block_K.size(), seed + 1, ElementK(1), ElementK(-1), 0);
    cutlass::reference::host::BlockFillRandomUniform(block_V.get(), block_V.size(), seed + 2, ElementV(1), ElementV(-1), 0);
    
    return problem_shape;
  }
  
  // GemmUniversalAdapter  doesn't support attention,
  // so a secondary 'run' is used to launch the kernel
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

    q.parallel_for<FlashAttentionKernel>(sycl::nd_range<3>{sycl_grid * sycl_block, sycl_block},
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

    return cutlass::Status::kSuccess;
  }
};

template <typename TileShape> 
struct FMHAConfig {

  template <typename ProblemConfig>
  static int run(const Options &options) {
    using ProblemShape = cute::tuple<int, int, int, int, int, int>;
    using ClusterShape = Shape<_1, _1, _1>;

    using TileShapeQK_MNK = decltype(select<0, 2, 3>(TileShape{}));
    using TileShapePV_MNK = decltype(select<0, 1, 2>(TileShape{}));

    using ElementInputQ = typename ProblemConfig::ElementInputQ;       // dtype of Q
    using ElementInputKV = typename ProblemConfig::ElementInputKV;      // dtype of K and V
    using ElementS = typename ProblemConfig::ElementS;                // dtype of S
    using ElementP = typename ProblemConfig::ElementP;                // dtype of P
    using ElementAccumulator = typename ProblemConfig::ElementAccumulator; // dtype of QK accum and PV accum
    using ElementOutput = typename ProblemConfig::ElementOutput;      // dtype of output

    constexpr auto majorQ = cute::AMMA::Major::K;
    constexpr auto majorK = cute::AMMA::Major::K;
    constexpr auto majorP = cute::AMMA::Major::K;
    constexpr auto majorV = cute::AMMA::Major::MN;

    constexpr int PipelineStages = 2;

    using SmemLayoutQ = decltype(make_layout(
      cute::select<0, 2>(TileShapeQK_MNK{}), GenRowMajor{}));

    using SmemLayoutAtomK = decltype(make_layout(
      cute::select<1, 2>(TileShapeQK_MNK{}), GenRowMajor{}));

    using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomK{},
      make_shape(shape<1>(TileShapeQK_MNK{}),
                 shape<2>(TileShapeQK_MNK{}),
                 Int<PipelineStages>{})));

    using SmemLayoutAtomV = decltype(make_layout(
      cute::select<1, 2>(TileShapePV_MNK{}), GenColMajor{}));
 
    using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomV{},
      make_shape(shape<1>(TileShapePV_MNK{}), 
                 shape<2>(TileShapePV_MNK{}), 
                 Int<PipelineStages>{})));

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
      ElementS /*D dtype*/,
      ElementInputQ /*A dtype*/,
      ElementInputKV /*B dtype*/,
      ElementS /*C dtype*/,
      TileShapeQK_MNK,
      ClusterShape,
      majorQ,
      majorK>()));

    // As we will accumulate O for num kv tile times, to ensure the accuracy, we
    // use ElementAccumulator dtype for PV mma C/D tensors.
    using TiledMmaPV = decltype(cute::make_tiled_mma(
      cute::AMMA::ss_op_selector<
      ElementAccumulator /*D dtype*/,
      ElementInputKV /*A dtype*/,
      ElementInputKV /*B dtype*/,
      ElementAccumulator /*C dtype*/,
      TileShapePV_MNK,
      ClusterShape,
      majorP,
      majorV>()));

    using CollectiveMainloop = cutlass::flash_attention::collective::CollectiveMmaAttention<
      ProblemShape, TileShape,
      ElementInputQ, ElementInputKV, ElementInputKV,
      ElementS, ElementP,
      ElementAccumulator, ElementOutput,
      cutlass::gemm::TagToStrideA_t<LayoutQ>,
      cutlass::gemm::TagToStrideB_t<LayoutK>,
      cutlass::gemm::TagToStrideB_t<LayoutV>,
      TiledMmaQK, TiledMmaPV,
      SmemLayoutQ, SmemLayoutK, SmemLayoutV,
      SmemLayoutOutputAccum, SmemLayoutOutput,
      TMACopyAtomQ, TMACopyAtomK, TMACopyAtomV>;

    using CollectiveSoftmaxEpilogue = cutlass::flash_attention::collective::CollectiveSoftmaxEpilogue<
      TileShape,
      ElementAccumulator,
      ElementOutput,
      ElementS,
      ElementP>;

    using CollectiveEpilogue = cutlass::flash_attention::collective::CollectiveEpilogueAttention<
      ProblemShape, TileShape,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutO>,
      SmemLayoutOutput,
      TMACopyAtomO>;

    using TileScheduler = cutlass::flash_attention::kernel::XeFlashIndividualTileScheduler;

    using GemmKernel = cutlass::flash_attention::kernel::GemmUniversalAttention<
      ProblemShape, 
      CollectiveMainloop,
      CollectiveSoftmaxEpilogue,
      CollectiveEpilogue, 
      TileScheduler>;
    
    ExampleRunner<GemmKernel> runner;
    
    runner.run(options);
    
    return 0;    
  }

};
