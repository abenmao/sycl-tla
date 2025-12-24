/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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
#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/xe_fhma_fwd_kernel.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include <cute/tensor.hpp>
#include <random>

#include "helper.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

using namespace cute;

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool is_causal;
  bool varlen = false;
  std::string scheduler;

  int batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo, iterations, verify;
  float softmax_scale;

  Options()
      : help(false), error(false), is_causal(false), varlen(false), batch(32), num_heads_q(16), num_heads_kv(16), seq_len_qo(512), head_size_qk(128),
        seq_len_kv(512), head_size_vo(128), iterations(100), softmax_scale(1.f), scheduler("Individual") {}

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    if (cmd.check_cmd_line_flag("is_causal")) {
      is_causal = true;
    }

    if (cmd.check_cmd_line_flag("varlen")) {
      varlen = true;
    }

    cmd.get_cmd_line_argument("scheduler", scheduler, std::string("Individual"));

    cmd.get_cmd_line_argument("batch", batch, 32);
    cmd.get_cmd_line_argument("num_heads_q", num_heads_q, 16);
    cmd.get_cmd_line_argument("num_heads_kv", num_heads_kv, num_heads_q);
    cmd.get_cmd_line_argument("seq_len_kv", seq_len_kv, 512);
#ifdef DECODE
    cmd.get_cmd_line_argument("seq_len_qo", seq_len_qo, 1);
#else
    cmd.get_cmd_line_argument("seq_len_qo", seq_len_qo, seq_len_kv);
#endif
    cmd.get_cmd_line_argument("head_size_vo", head_size_vo, HEAD_DIM);
    cmd.get_cmd_line_argument("head_size_qk", head_size_qk, head_size_vo);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);    

    softmax_scale = 1 / sqrt(static_cast<float>(head_size_qk));
  }

  /// Prints the usage statement.
  std::ostream &print_usage(std::ostream &out) const {

    out << "Xe FMHA Example\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement\n\n"
        << "  --is_causal                 Apply Causal Mask to the output of first Matmul\n"
        << "  --varlen                    Enable variable sequence length\n"
        << "  --scheduler=\"Value\"       Choose between Individual or Persistent Scheduler\n"
        << "  --batch=<int>               Sets the Batch Size of the Multi-Head Self Attention module\n"
        << "  --num_heads_q=<int>         Sets the Number of Attention Heads for Key-Value pair the Multi-Head Self Attention module\n"
        << "  --num_heads_kv=<int>        Sets the Number of Attention Heads for Query input in the Multi-Head Self Attention module\n"
        << "  --seq_len_qo=<int>          Sets the Sequence length of the Query input in Multi-Head Self Attention module\n"
        << "  --seq_len_kv=<int>          Sets the Sequence length of the Key-Value pair in Multi-Head Self Attention module\n"
        << "  --head_size_qk=<int>        Sets the Attention Head dimension of the 1st Matrix Multiplication in Multi-Head Self Attention module\n"
        << "  --head_size_vo=<int>        Sets the Attention Head dimension of the 2nd Matrix Multiplication in Multi-Head Self Attention module\n"
        << "  --iterations=<int>          Iterations\n\n"
        << "  --verify=<int>              Specify whether to verify.\n\n";
    return out;
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers

template <typename SrcT, typename DstT> class ConvertTensorKernelTag{};

template <typename SrcT, typename DstT>
void convert_tensor(const SrcT* d_src, DstT* d_dst, size_t size) {
  using Tag = ConvertTensorKernelTag<SrcT, DstT>;
  compat::get_default_queue().parallel_for<Tag>(size, [=](auto indx) {
    d_dst[indx] = static_cast<DstT>(d_src[indx]);
  }).wait();
}

template <typename InT> inline auto in_memory(cutlass::DeviceAllocation<InT>& in) {
  using OutT = cute::conditional_t<(sizeof_bits_v<InT> <= 8), half_t, InT>;
  if constexpr (!is_same_v<InT, OutT>) {
    cutlass::DeviceAllocation<OutT> out(in.size());
    convert_tensor<InT, OutT>(in.get(), out.get(), in.size());
    return out;
  } else {
    return in;
  };
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// 3 input matrices: (K)eys, (Q)ueries and (V)alues.
using LayoutQ = cutlass::layout::RowMajor;
using LayoutK = cutlass::layout::ColumnMajor;
using LayoutV = cutlass::layout::RowMajor;
using LayoutO = cutlass::layout::RowMajor;


static constexpr int GROUP_SIZE = 32;

template <class FMHAKernel, bool isVarLen = false> struct ExampleRunner {

  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;
  using StrideScaleQ = typename FMHAKernel::StrideScaleQ;
  using StrideScaleK = typename FMHAKernel::StrideScaleK;
  using StrideScaleV = typename FMHAKernel::StrideScaleV;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;
  using ElementScale = typename FMHAKernel::ElementScale;
  using ElementQKMMAVerify = cute::conditional_t<(sizeof_bits_v<ElementQ> <= 8), half_t, ElementQ>;
  using ElementPVMMAVerify = cute::conditional_t<(sizeof_bits_v<ElementV> >= 16), ElementV, ElementQKMMAVerify>;
  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;
  static constexpr bool UseScale = FMHAKernel::UseScale;
  static constexpr bool FP4Input = sizeof_bits_v<ElementQ> < 8;
  static constexpr bool F8kvF16mma = CollectiveMainloop::F8kvF16mma;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;

  StrideScaleQ stride_SQ;
  StrideScaleK stride_SK;
  StrideScaleV stride_SV;

  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementQ> block_Q;
  cutlass::DeviceAllocation<ElementK> block_K;
  cutlass::DeviceAllocation<ElementV> block_V;
  cutlass::DeviceAllocation<ElementO> block_O;
  cutlass::DeviceAllocation<ElementO> block_ref_O;

  std::vector<int> cumulative_seqlen_q;
  std::vector<int> cumulative_seqlen_kv;
  cutlass::DeviceAllocation<int> device_cumulative_seqlen_q;
  cutlass::DeviceAllocation<int> device_cumulative_seqlen_kv;

  cutlass::DeviceAllocation<ElementQKMMAVerify> block_Q_dq; // Dequantized copy of Q for validation
  cutlass::DeviceAllocation<ElementQKMMAVerify> block_K_dq; // Dequantized copy of K for validation
  cutlass::DeviceAllocation<ElementPVMMAVerify> block_V_dq; // Dequantized copy of V for validation
  cutlass::DeviceAllocation<ElementScale> block_scaleQ;
  cutlass::DeviceAllocation<ElementScale> block_scaleK;
  cutlass::DeviceAllocation<ElementScale> block_scaleV;
  std::vector<int> cumulative_scale_q;
  std::vector<int> cumulative_scale_kv;
  cutlass::DeviceAllocation<int> device_cumulative_scale_q;
  cutlass::DeviceAllocation<int> device_cumulative_scale_kv;

  ElementScale scale_k;
  ElementScale scale_v;

  //
  // Methods
  //

  template<class ProblemShape>
  auto initialize_varlen(const ProblemShape& problem_size) {
    int num_batches = get<0>(problem_size);

    // generate Q as --b times
    //    gaussian (--Q, --Q / 2) sampled positive
    //    track cumulative 
    std::mt19937 rng(0x202305151552ull);
    std::normal_distribution<double> dist_q(get<3>(problem_size), get<3>(problem_size) / 2);
    std::normal_distribution<double> dist_kv(get<4>(problem_size), get<4>(problem_size) / 2);

    // Use Cacheline Size to calculate alignment
    constexpr int cacheline_bytes = 64;
    constexpr int AlignmentQ = cacheline_bytes / sizeof(ElementQ);    // Alignment of Q matrix in units of elements
    constexpr int AlignmentKV = cacheline_bytes / sizeof(ElementK);   // Alignment of Kand V matrix in units of elements

    auto generate_positive_int = [](auto& dist, auto& gen) {
      int result = 0;
      do {
        result = static_cast<int>(dist(gen));
      } while (result <= 0);
      return result;
    };

    cumulative_seqlen_q = {0};
    cumulative_seqlen_kv = {0};

    int total_seqlen_q = 0;
    int total_seqlen_kv = 0;
    int max_seqlen_q = 0;
    int max_seqlen_kv = 0;

    if constexpr (UseScale) {
      cumulative_scale_q = {0};
      cumulative_scale_kv = {0};
    }


    for (int i = 0; i < num_batches; i++) {
      int seqlen_q = cutlass::round_up(generate_positive_int(dist_q, rng), AlignmentQ);
      int seqlen_kv = cutlass::round_up(generate_positive_int(dist_kv, rng), AlignmentKV);

      total_seqlen_q += seqlen_q;
      total_seqlen_kv += seqlen_kv;

      max_seqlen_q = std::max(max_seqlen_q, seqlen_q);
      max_seqlen_kv = std::max(max_seqlen_kv, seqlen_kv);

      cumulative_seqlen_q.push_back(cumulative_seqlen_q.back() + seqlen_q);
      cumulative_seqlen_kv.push_back(cumulative_seqlen_kv.back() + seqlen_kv);
      if constexpr (UseScale) {
        int scale_len_q = cute::ceil_div(seqlen_q, GROUP_SIZE);
        int scale_len_kv = cute::ceil_div(seqlen_kv, GROUP_SIZE);
        cumulative_scale_q.push_back(cumulative_scale_q.back() + scale_len_q);
        cumulative_scale_kv.push_back(cumulative_scale_kv.back() + scale_len_kv);
      }
    }

    ProblemShape problem_size_for_init = problem_size;
    get<0>(problem_size_for_init) = 1;
    get<3>(problem_size_for_init) = total_seqlen_q;
    get<4>(problem_size_for_init) = total_seqlen_kv;

    ProblemShapeType problem_size_for_launch;
    problem_size_for_launch.batch = get<0>(problem_size);
    problem_size_for_launch.num_heads_q = get<1>(problem_size);
    problem_size_for_launch.num_heads_kv = get<2>(problem_size);
    problem_size_for_launch.seq_len_qo = cutlass::fmha::collective::VariableLength{max_seqlen_q};
    problem_size_for_launch.seq_len_kv = cutlass::fmha::collective::VariableLength{max_seqlen_kv};
    problem_size_for_launch.head_size_qk = get<5>(problem_size);
    problem_size_for_launch.head_size_vo = get<6>(problem_size);

    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }

  bool verify(ProblemShapeType shape, bool is_causal) {

    if constexpr (isVarLen) {
      int max_seq_len_q = shape.seq_len_qo;
      int max_seq_len_kv = shape.seq_len_kv;
      shape.seq_len_qo = cutlass::fmha::collective::VariableLength{max_seq_len_q, cumulative_seqlen_q.data()};
      shape.seq_len_kv = cutlass::fmha::collective::VariableLength{max_seq_len_kv, cumulative_seqlen_kv.data()};
      if constexpr (UseScale) {
        shape.seq_len_qo.cumulative_scale_length = cumulative_scale_q.data();
        shape.seq_len_kv.cumulative_scale_length = cumulative_scale_kv.data();
      }
    }

    auto batch = shape.batch;
    auto num_heads_q = shape.num_heads_q;
    auto num_heads_kv = shape.num_heads_kv;
    auto head_size_qk = shape.head_size_qk;
    auto head_size_vo = shape.head_size_vo;
    int seq_len_qo, seq_len_kv;

    auto block_Q_ = UseScale ? block_Q_dq : in_memory(block_Q);
    auto block_K_ = (UseScale || F8kvF16mma) ? block_K_dq : in_memory(block_K);
    auto block_V_ = ((UseScale && !FP4Input) || F8kvF16mma) ? block_V_dq : in_memory(block_V);
    using ElementV_ = std::conditional_t<UseScale && !FP4Input, 
                                    ElementPVMMAVerify,
                                    std::remove_pointer_t<decltype(block_V_.get())>>;

    int offset_q = 0;
    int offset_k = 0;
    int offset_v = 0;
    int offset_o = 0;

    // loop over the batch dimension to compute the output
    // to avoid the risk of running out of device memory
    int q_group_size = num_heads_q/num_heads_kv;
    for (int b = 0; b < batch; b++) {
      if constexpr (isVarLen) {
        auto logical_seq_shape = cutlass::fmha::collective::apply_variable_length(make_shape(shape.seq_len_qo, shape.seq_len_kv), b);
        seq_len_qo = get<0>(logical_seq_shape);
        seq_len_kv = get<1>(logical_seq_shape);
      } else {
        seq_len_qo = shape.seq_len_qo;
        seq_len_kv = shape.seq_len_kv;
      }

      int kv_group_update=1;
      for (int h = 0; h < num_heads_q; h++) {
        cutlass::DeviceAllocation<ElementS> block_S;
        block_S.reset(seq_len_qo * seq_len_kv);

        cutlass::TensorRef ref_Q(block_Q_.get() + offset_q, LayoutQ::packed({seq_len_qo, head_size_qk}));
        cutlass::TensorRef ref_K(block_K_.get() + offset_k, LayoutK::packed({head_size_qk, seq_len_kv}));
        cutlass::TensorRef ref_V(block_V_.get() + offset_v, LayoutV::packed({seq_len_kv, head_size_vo}));
        cutlass::TensorRef ref_S(block_S.get(), LayoutQ::packed({seq_len_qo, seq_len_kv}));

        cutlass::reference::device::GemmComplex({seq_len_qo, seq_len_kv, head_size_qk}, 1.f, ref_Q,
                                                cutlass::ComplexTransform::kNone, ref_K, cutlass::ComplexTransform::kNone,
                                                0.f, ref_S, ref_S, ElementS(0),
                                                1,                   // batch_count
                                                seq_len_qo * head_size_qk, // batch_stride_Q
                                                seq_len_kv * head_size_qk, // batch_stride_K
                                                seq_len_qo * seq_len_kv,   // batch_stride_S
                                                seq_len_qo * seq_len_kv    // batch_stride_S
        );

        compat::wait();

        std::vector<ElementS> host_S(block_S.size());
        compat::memcpy<ElementS>(host_S.data(), block_S.get(), host_S.size());

        // delete this memory as it is no longer needed
        block_S.reset();
        auto offset = cute::min(seq_len_qo, seq_len_kv);
        auto discard_seq_coord = seq_len_qo - offset;
        auto full_tile_offset = seq_len_kv - offset;
        if (is_causal) {
          // apply mask to S
          for (int row = 0; row < seq_len_qo; row++) {
            for (int col = 0; col < seq_len_kv; col++) {
              if ((col - full_tile_offset) > (row - discard_seq_coord))
                host_S[col + row * seq_len_kv] = ElementS{-INFINITY};
            }
          }
        }

        // compute max element per row of S
        std::vector<ElementS> max_vec(seq_len_qo, ElementS{-INFINITY});
        for (int row = 0; row < seq_len_qo; row++) {
          int idx = row * seq_len_kv;
          int max_idx = row;
          max_vec[max_idx] = host_S[idx++];
          for (int col = 1; col < seq_len_kv; col++, idx++) {
            if (max_vec[max_idx] < host_S[idx])
              max_vec[max_idx] = host_S[idx];
          }
        }

        // compute exp of S
        for (int row = 0; row < seq_len_qo; row++) {
          int idx = row * seq_len_kv;
          int max_idx = row;
          for (int col = 0; col < seq_len_kv; col++, idx++) {
            /* FIXME: use softmax_scale instead of assuming its value here */
            host_S[idx] = expf((host_S[idx] - max_vec[max_idx]) / sqrt(static_cast<ElementS>((head_size_qk))));
          }
        }

        // compute sum per row of S
        std::vector<ElementS> sum_vec(seq_len_qo, ElementS{0});
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
            if(is_causal && row < discard_seq_coord) {
              host_S[idx] = 0;
            } else {
              host_S[idx] /= sum_vec[sum_idx];
            }
          }
        }

        std::vector<ElementV_> host_P(host_S.size());
        for (int p = 0; p < host_P.size(); p++)
          host_P[p] = static_cast<ElementV_>(host_S[p]);

        cutlass::DeviceAllocation<ElementV_> block_P;
        block_P.reset(host_P.size());

        compat::memcpy<ElementV_>(block_P.get(), host_P.data(), host_P.size());

        cutlass::TensorRef ref_P(block_P.get(), LayoutQ::packed({seq_len_qo, seq_len_kv}));

        cutlass::DeviceAllocation<ElementS> block_acc;
        block_acc.reset(seq_len_qo * head_size_vo);
        cutlass::TensorRef ref_acc(block_acc.get(), LayoutO::packed({seq_len_qo, head_size_vo}));

        cutlass::reference::device::GemmComplex({seq_len_qo, head_size_vo, seq_len_kv}, ElementS{1}, ref_P,
                                                cutlass::ComplexTransform::kNone, ref_V, cutlass::ComplexTransform::kNone,
                                                ElementS{0}, ref_acc, ref_acc, ElementS{0},
                                                1,                   // batch_count
                                                seq_len_qo * seq_len_kv,   // batch_stride_P
                                                seq_len_kv * head_size_vo, // batch_stride_V
                                                seq_len_qo * head_size_vo, // batch_stride_O
                                                seq_len_qo * head_size_vo  // batch_stride_O
        );

        compat::wait();
        // delete this memory as it is no longer needed
        block_P.reset();

        std::vector<ElementS> vec_acc(block_acc.size());
        compat::memcpy<ElementS>(vec_acc.data(), block_acc.get(), vec_acc.size());

        // delete this memory as it is no longer needed
        block_acc.reset();
        std::vector<ElementO> vec_out(vec_acc.size());
        for(int i = 0; i < vec_out.size(); i++) {
          vec_out[i] = static_cast<ElementO>(vec_acc[i]);
        }
        compat::memcpy<ElementO>(block_ref_O.get() + offset_o, vec_out.data(), vec_out.size());

        offset_q += seq_len_qo * head_size_qk;
        if(kv_group_update % q_group_size==0) {
          offset_k += seq_len_kv * head_size_qk;
          offset_v += seq_len_kv * head_size_vo;
        }
        kv_group_update++;
        offset_o += seq_len_qo * head_size_vo;
      }
    }

    compat::wait();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(block_ref_O.get(), block_O.get(),
                                                                          block_O.size(), ElementO{0.05}, ElementO{0.05});

    return passed;
  }
  template <class Element>
  bool initialize_scale(
    cutlass::DeviceAllocation<Element>& block,
    Options const& options, bool const_scale = false) {
    const float elt_max_f = float(cutlass::platform::numeric_limits<Element>::max());
    // Need to fix max_dequant_val and min_dequant_val?
    const float max_dequant_val = elt_max_f * 0.25f;
    const float min_dequant_val = 0.5f;
    const float scale_max = max_dequant_val / elt_max_f;
    const float scale_min = min_dequant_val / elt_max_f;
    if (const_scale) {
      std::vector<Element> host(block.size(), Element(1));
      cutlass::device_memory::copy_to_device(block.get(), host.data(), host.size());
      compat::wait();
    } else {
      cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, Element(scale_max), Element(scale_min));
    }
    return true;
  }

  template <
  class DstElement,
  class SrcElement,
  class Layout,
  class ElementScale,
  class ScaleLayout>
  static void apply_scale(DstElement* dq_buffer,
                       SrcElement const* q_buffer,
                       Layout const operand_layout,
                       ElementScale const* scale_buffer,
                       ScaleLayout const scale_layout) {
    std::vector<uint8_t> dst(size(operand_layout) * sizeof_bits_v<DstElement> / 8, 0);
    cutlass::device_memory::copy_to_host(dst.data(), (uint8_t*)dq_buffer, dst.size());

    std::vector<uint8_t> src(size(operand_layout) * sizeof_bits_v<SrcElement> / 8, 0);
    cutlass::device_memory::copy_to_host(src.data(), (uint8_t*)q_buffer, src.size());

    std::vector<uint8_t> scale(size(scale_layout) * sizeof_bits_v<ElementScale> / 8, 0);
    cutlass::device_memory::copy_to_host(scale.data(), (uint8_t*)scale_buffer, scale.size());

    compat::wait();

    static_assert(sizeof_bits_v<DstElement> >= 8);

    auto dst_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<DstElement*>(dst.data())), operand_layout);

    auto src_tensor = [&]() {
      if constexpr (sizeof_bits_v<SrcElement> < 8) {
        return make_tensor(cute::subbyte_iterator<const SrcElement>(src.data()), operand_layout);
      } else {
        return make_tensor(make_gmem_ptr(reinterpret_cast<SrcElement const *>(src.data())), operand_layout);
      }
    }();

    auto scale_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<ElementScale const *>(scale.data())), scale_layout);

    auto dim0 = size<0>(src_tensor);
    auto dim1 = size<1>(src_tensor);
    auto dim2 = size<2>(src_tensor);
    auto dim3 = size<3>(src_tensor);

    using ret_type = float;

    for (int b = 0; b < dim3; b++) {
      for (int h = 0; h < dim2; h++) {
        for (int k = 0; k < dim1; k++) {
          for (int mn = 0; mn < dim0; mn++) {
            auto src_data = [&]() {
              if constexpr (sizeof_bits_v<SrcElement> >= 8) {
                return  (ret_type)(src_tensor(mn, k, h, b));
              } else {
                return (ret_type)(src_tensor(mn, k, h, b).get());
              }
            }();

            auto scale_data = (ret_type)(scale_tensor(mn, k / 32, h, b));

            dst_tensor(mn, k, h, b) = static_cast<DstElement>((src_data) * scale_data);
          }
        }
      }
    }

    cutlass::device_memory::copy_to_device(dq_buffer, (DstElement*)(raw_pointer_cast(dst_tensor.data())), dst_tensor.size());
    compat::wait();
  }

  template <typename LowpT, typename DeqT>
  void apply_dequantization(const cutlass::DeviceAllocation<LowpT>& lowp, cutlass::DeviceAllocation<DeqT>& deq, float& scale) {
    const float lowp_max = float(cutlass::platform::numeric_limits<LowpT>::max());
    const float highp_max = float(cutlass::platform::numeric_limits<DeqT>::max());
    auto s = highp_max / lowp_max;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, s/2.0f);
    scale = dis(gen);

    auto deq_buff = std::vector<DeqT>(deq.size());
    compat::memcpy<DeqT>(deq_buff.data(), deq.get(), deq.size());
    compat::wait();

    for (auto i = 0; i < deq.size(); ++i) {
      deq_buff[i] = static_cast<DeqT>(scale * static_cast<float>(deq_buff[i]));
    }

    compat::memcpy<DeqT>(deq.get(), deq_buff.data(), deq.size());
    compat::wait();
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  ProblemShapeType initialize(const Options &options) {
    auto problem_shape_in = cute::make_tuple(options.batch, options.num_heads_q, options.num_heads_kv, options.seq_len_qo, options.seq_len_kv, options.head_size_qk, options.head_size_vo);
    ProblemShapeType shape;

    decltype(problem_shape_in) problem_size;

    if constexpr (isVarLen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(problem_shape_in);
      problem_size = problem_shape_init;
      shape = problem_shape_launch;
    } else {
      problem_size = problem_shape_in;
      shape.batch        = options.batch;
      shape.num_heads_q  = options.num_heads_q;
      shape.num_heads_kv = options.num_heads_kv;
      shape.seq_len_qo   = options.seq_len_qo;
      shape.seq_len_kv   = options.seq_len_kv;
      shape.head_size_qk = options.head_size_qk;
      shape.head_size_vo = options.head_size_vo;
    }

    auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo] = problem_size;
    auto shape_Q = cute::make_shape(seq_len_qo, head_size_qk, num_heads_q,  batch);
    auto shape_K = cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch);
    auto shape_V = cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch);
    auto shape_O = cute::make_shape(seq_len_qo, head_size_vo, num_heads_q,  batch);

    stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, shape_Q);
    stride_K = cutlass::make_cute_packed_stride(StrideK{}, shape_K);
    stride_V = cutlass::make_cute_packed_stride(StrideV{}, shape_V);
    stride_O = cutlass::make_cute_packed_stride(StrideO{}, shape_O);
    stride_SQ = StrideScaleQ{};
    stride_SK = StrideScaleK{};
    stride_SV = StrideScaleV{};

    block_Q.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo * head_size_qk);
    block_K.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv * head_size_qk);
    block_V.reset(static_cast<std::size_t>(batch) * num_heads_kv * seq_len_kv * head_size_vo);
    block_O.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo * head_size_vo);
    block_ref_O.reset(static_cast<std::size_t>(batch) * num_heads_q * seq_len_qo * head_size_vo);

    initialize_block(block_Q, seed + 2023);
    initialize_block(block_K, seed + 2022);
    initialize_block(block_V, seed + 2021);
    
    if (!cumulative_seqlen_q.empty()) {
      device_cumulative_seqlen_q.reset(cumulative_seqlen_q.size());
      device_cumulative_seqlen_q.copy_from_host(cumulative_seqlen_q.data(), cumulative_seqlen_q.size());
    }

    if (!cumulative_seqlen_kv.empty()) {
      device_cumulative_seqlen_kv.reset(cumulative_seqlen_kv.size());
      device_cumulative_seqlen_kv.copy_from_host(cumulative_seqlen_kv.data(), cumulative_seqlen_kv.size());
    }
    if constexpr (isVarLen) {
      shape.seq_len_qo.cumulative_length = device_cumulative_seqlen_q.get();
      shape.seq_len_kv.cumulative_length = device_cumulative_seqlen_kv.get();
      if constexpr (UseScale) {
        if (!cumulative_scale_q.empty()) {
          device_cumulative_scale_q.reset(cumulative_scale_q.size());
          device_cumulative_scale_q.copy_from_host(cumulative_scale_q.data(), cumulative_scale_q.size());
        }
        if (!cumulative_scale_kv.empty()) {
          device_cumulative_scale_kv.reset(cumulative_scale_kv.size());
          device_cumulative_scale_kv.copy_from_host(cumulative_scale_kv.data(), cumulative_scale_kv.size());
        }
        shape.seq_len_qo.cumulative_scale_length = device_cumulative_scale_q.get();
        shape.seq_len_kv.cumulative_scale_length = device_cumulative_scale_kv.get();
      }
    }

    block_Q_dq.reset(block_Q.size());
    block_K_dq.reset(block_K.size());
    block_V_dq.reset(block_V.size());

    convert_dtype<ElementQ, ElementQKMMAVerify, ExampleRunner>(block_Q, block_Q_dq);
    convert_dtype<ElementK, ElementQKMMAVerify, ExampleRunner>(block_K, block_K_dq);
    convert_dtype<ElementV, ElementPVMMAVerify, ExampleRunner>(block_V, block_V_dq);

    if constexpr (F8kvF16mma) {
      apply_dequantization(block_K, block_K_dq, scale_k);
      apply_dequantization(block_V, block_V_dq, scale_v);
    } else if constexpr (UseScale) {
      auto scale_q = cute::ceil_div(head_size_qk, GROUP_SIZE);
      auto scale_k = cute::ceil_div(head_size_qk, GROUP_SIZE);
      int scale_v = cute::ceil_div(seq_len_kv, GROUP_SIZE);
      if constexpr (isVarLen && UseScale) { scale_v = cumulative_scale_kv.back(); }

      auto shape_scale_Q = cute::make_shape(seq_len_qo, scale_q, num_heads_q, batch);
      auto shape_scale_K = cute::make_shape(seq_len_kv, scale_k, num_heads_kv, batch);
      auto shape_scale_V = cute::make_shape(head_size_vo, scale_v, num_heads_kv, batch);

      stride_SQ = cutlass::make_cute_packed_stride(StrideScaleQ{}, shape_scale_Q); 
      stride_SK = cutlass::make_cute_packed_stride(StrideScaleK{}, shape_scale_K);
      stride_SV = cutlass::make_cute_packed_stride(StrideScaleV{}, shape_scale_V);

      block_scaleQ.reset(cute::size(shape_scale_Q));
      block_scaleK.reset(cute::size(shape_scale_K));
      block_scaleV.reset(cute::size(shape_scale_V));

      initialize_scale(block_scaleQ, options);
      initialize_scale(block_scaleK, options);
      initialize_scale(block_scaleV, options);

      auto layout_Q = cute::make_layout(shape_Q, stride_Q);
      auto layout_K = cute::make_layout(shape_K, stride_K);
      auto layout_V = cute::make_layout(shape_V, stride_V);

      auto layout_scale_Q = cute::make_layout(shape_scale_Q, stride_SQ);
      auto layout_scale_K = cute::make_layout(shape_scale_K, stride_SK);
      auto layout_scale_V = cute::make_layout(shape_scale_V, stride_SV);

      apply_scale<ElementQKMMAVerify, ElementQ>(block_Q_dq.get(), block_Q.get(), layout_Q, block_scaleQ.get(), layout_scale_Q);
      apply_scale<ElementQKMMAVerify, ElementK>(block_K_dq.get(), block_K.get(), layout_K, block_scaleK.get(), layout_scale_K);
      apply_scale<ElementPVMMAVerify, ElementV>(block_V_dq.get(), block_V.get(), layout_V, block_scaleV.get(), layout_scale_V);
    }

    return shape;
  }

  // Note that the GemmUniversalAdapter currently doesn't support flash attention, which is why this
  // secondary `run` function is required to launch the kernel.
  static void run(typename FMHAKernel::Params params)
  {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group scratch memory extension
    compat::experimental::launch_properties launch_props {
      syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
      syclex::sub_group_size<cute::intel::sg_size>
    };
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, params);

    EventManager::getInstance().addEvent(event);
  }

  cutlass::Status run(const Options &options, const cutlass::KernelHardwareInfo &hw_info) {

    ProblemShapeType shape = initialize(options);

    typename FMHAKernel::Arguments arguments{
      {
        shape,
        block_Q.get(), stride_Q,
        block_K.get(), stride_K,
        block_V.get(), stride_V,
        block_O.get(), stride_O,
        block_scaleQ.get(), stride_SQ,
        block_scaleK.get(), stride_SK,
        block_scaleV.get(), stride_SV,
        scale_k, scale_v,
        GROUP_SIZE
      },
      {options.softmax_scale},
      {},
      hw_info
    };

    // Define device-global scratch memory
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (!FMHAKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << options.batch << 'x' << options.num_heads_q << 'x' <<
        options.seq_len_qo << 'x' << options.seq_len_kv << 'x' << options.head_size_qk << 'x'  << options.head_size_vo
        << (options.is_causal ? "xCausal" : "xNonCausal") << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    CUTLASS_CHECK(FMHAKernel::initialize_workspace(arguments, workspace.get()));

    // Convert host-side arguments to device-side arguments to be passed to the kernel
    auto params = FMHAKernel::to_underlying_arguments(arguments, workspace.get());

    // Run the GEMM
    run(params);

    compat::wait();

    if (options.verify != 0) {
      // Verify that the result is correct
      bool passed = verify(shape, options.is_causal);
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

      if (!passed) {
        return cutlass::Status::kErrorInternal;
      }
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }
    
    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        run(params);
      }
      compat::wait();
    // when seq_len_qo is not equal to seq_len_kv we use bottom up approach for the masking.
      // Following changes will adjust the effective_seq_len_kv when masking applied for such cases
      auto offset = cute::min(options.seq_len_qo, options.seq_len_kv);
      auto discard_seq_coord = options.seq_len_qo - offset;
      auto full_tile_offset = options.seq_len_kv - offset;
      // offset + 1 is going to be ceil_div
      auto effective_seq_len_kv = options.is_causal ? full_tile_offset + ((offset + 1) / 2.0): options.seq_len_kv;
      auto effective_seq_len_qo = options.is_causal ? options.seq_len_qo - discard_seq_coord : options.seq_len_qo;
      double cute_time = timer.seconds() / options.iterations;
      double flops_qk = 2.0 * options.batch * options.num_heads_q * effective_seq_len_qo * effective_seq_len_kv * options.head_size_qk;
      double flops_pv = 2.0 *  options.batch * options.num_heads_q * effective_seq_len_qo * options.head_size_vo * effective_seq_len_kv;
      double tflops = ((flops_qk + flops_pv) * 1e-12) / cute_time;
      double gbps_qk =  options.batch * (sizeof(ElementQ) * options.num_heads_q * effective_seq_len_qo * options.head_size_qk +
                                         sizeof(ElementK) * options.num_heads_kv * effective_seq_len_kv * options.head_size_qk);
      double gbps_pv = sizeof(ElementV) * options.batch * options.num_heads_kv * effective_seq_len_kv * options.head_size_vo +
                     sizeof(ElementO) * options.batch * options.num_heads_q * effective_seq_len_qo * options.head_size_vo;
      double gbps = ((gbps_qk + gbps_pv)  * 1e-9) / (cute_time);
      std::cout << "Batch: " << options.batch << "\tNumHeads_q: " << options.num_heads_q  << "\tNumHeads_kv: " << options.num_heads_kv  << "\tSeq Length QO: " << options.seq_len_qo
                << "\tSeq Length KV: " << options.seq_len_kv << "\tHead Size QK: " << options.head_size_qk << "\tHead Size VO: " << options.head_size_vo
                << "\tCausal Mask: " << (options.is_causal ? "true" : "false") << "\tVariable Sequence Length: " << (options.varlen ? "true" : "false")
                << "\t Scheduler: " << options.scheduler;
      printf("\nPerformance:   %4.3f  GB/s,    %4.3f  TFlop/s,   %6.4f  ms\n\n", gbps, tflops, cute_time * 1000);
    }

    return cutlass::Status::kSuccess;
  }
};

template <bool Causal,
          bool UseScale,
          typename TileShapeQK,
          typename TileShapePV,
          typename TileShapeOutput,
          typename SubgroupLayoutQK,
          typename SubgroupLayoutPV_,      /* void -> default */
          int PipelineStages,
          typename ElementQ = bfloat16_t,
          typename ElementK = bfloat16_t,
          typename ElementV = bfloat16_t,
          typename ElementScale = float,
          typename ElementO = float,
          typename MMAOperation_ = void,    /* void -> default */
          typename StrideQ = Stride<int, _1, int, int>,
          typename StrideK = Stride<int, _1, int, int>,
          typename StrideV = Stride<_1, int, int, int>,
          typename StrideO = Stride<int, _1, int, int>,
          typename StrideScaleQ = Stride<_1, int, int, int>,
          typename StrideScaleK = Stride<_1, int, int, int>,
          typename StrideScaleV = Stride<_1, int, int, int>,
          typename GmemTiledCopyQ = void,   /* void -> default block 2D */
          typename GmemTiledCopyK = void,
          typename GmemTiledCopyV = void,
          typename GmemTiledCopyO = void>
struct FMHAConfig {

  static constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();

  template <typename T>
  static constexpr bool is_f8_v = cute::is_any_of_v<T, cute::float_e5m2_t, cute::float_e4m3_t>;
  template <typename T>
  static constexpr bool is_f16_v = cute::is_any_of_v<T, cute::half_t, cute::bfloat16_t>;
  static constexpr bool F8kvF16mma = is_f16_v<ElementQ> && is_f8_v<ElementK> && cute::is_same_v<ElementScale, float> && !UseScale;

#if !(defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
  using DefaultMMA = typename cute::conditional_t<
      cute::is_same_v<ElementQ, cutlass::float_e5m2_t> || cute::is_same_v<ElementQ, cutlass::float_e4m3_t>,
      XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, half_t>,
      XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>
  >;
  using MMAOperationPV = DefaultMMA;
#else
  using DefaultDpasOp = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>;
  using DefaultBdpasOp = XE_BDPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>;
  using DefaultMMA = cute::conditional_t<UseScale, DefaultBdpasOp, DefaultDpasOp>;
  using MMAOperationPV = typename cute::conditional_t<
      cute::is_same_v<ElementV, cutlass::float_e5m2_t> || cute::is_same_v<ElementV, cutlass::float_e4m3_t>,
      DefaultMMA,
      XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementV>
  >;
#endif

  using MMAOperation = cute::conditional_t<is_void_v<MMAOperation_>,
                                           DefaultMMA,
                                           MMAOperation_>;
  
  using SubgroupLayoutPV = cute::conditional_t<is_void_v<SubgroupLayoutPV_>,
                                               decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})),
                                               SubgroupLayoutPV_>;

  template <bool isVarLen, class Scheduler>
  static int run(const Options &options) {
    //
    // Run examples
    //

    // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
    // information is used by the underlying kernel.
    cutlass::KernelHardwareInfo hw_info;

    using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;

    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperationPV>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

    static_assert(get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
        "Output tile and P*V tile have different sizes in Q dimension");
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(make_gmem_ptr(&val),
                         make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ = decltype(make_dummy_tensor(ElementQ{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementK{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementV{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));
    using TensorScaleQ = decltype(make_dummy_tensor(ElementScale{}, StrideScaleQ{}));
    using TensorScaleK = decltype(make_dummy_tensor(ElementScale{}, StrideScaleK{}));
    using TensorScaleV = decltype(make_dummy_tensor(ElementScale{}, StrideScaleV{}));
    // Mainloop
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy, Causal, UseScale, F8kvF16mma,
        TiledMMAQK, TiledMMAPV, VTiles,
        TensorQ, TensorK, TensorV,
        TensorScaleQ, TensorScaleK, TensorScaleV,
        GmemTiledCopyQ, GmemTiledCopyK, GmemTiledCopyV
    >;

    // Epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        GmemTiledCopyO
    >;

    using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
        ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler
    >;

    ExampleRunner<FMHAKernel, isVarLen> runner;

    CUTLASS_CHECK(runner.run(options, hw_info));
    return 0;
  }

  static int run(const Options &options) {
    if (options.varlen) {
      return run<true, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else {
      return run<false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    }
  }
};
