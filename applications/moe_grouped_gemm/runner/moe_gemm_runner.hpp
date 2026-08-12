/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Shared config-driven MoE grouped-GEMM runner: device kernel launchers
           + verification. Source-only (no deps on benchmarks/ or examples/).
           Kernel-free shared declarations come from moe_types.hpp.

    The device kernel is only instantiated by the consumer's device-codegen TU
    (moe_api.cpp), keeping AOT device codegen localized there.
*/

#pragma once

#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include "cutlass/cutlass.h"

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/initialize_block.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/sycl_tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/relatively_equal.h"
#include "cutlass/util/sycl_event_manager.hpp"

#include "moe_grouped_gemm/runner/moe_types.hpp"

#include "moe_grouped_gemm/kernel/xe_moe_gemm_greedy.hpp"
#include "moe_grouped_gemm/kernel/xe_moe_tile_scheduler.hpp"
#include "moe_grouped_gemm/kernel/xe_moe_gemm_double_buffer.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace cutlass::moe {

using namespace cute;
using namespace MoE;

using ElementAccumulator = float;

///////////////////////////////////////////////////////////////////////////////////////////////////

struct VerificationHelper {

  int m = 0, n = 0, k = 0, groups;
  int *num_rows_per_expert = nullptr;
  std::vector<typename MoE::ProblemShape::UnderlyingProblemShape>
      problem_sizes_host;

  VerificationHelper() = default;

  void parse(const int num_experts, const int *num_tokens_per_expert_host,
             int moe_n, int moe_k,
             const int *num_tokens_per_expert_device = nullptr) {
    m = 0; // reset so a reused helper doesn't accumulate stale rows
    n = moe_n;
    k = moe_k;
    groups = num_experts;
    num_rows_per_expert = const_cast<int *>(num_tokens_per_expert_device);
    assert(groups > 0);
    problem_sizes_host.clear();
    problem_sizes_host.reserve(groups);
    for (int i = 0; i < groups; i++) {
      problem_sizes_host.push_back({num_tokens_per_expert_host[i], n, k});
      m += num_tokens_per_expert_host[i];
    }
  }

 struct Tolerance {
    float rtol;
    float atol;
  };

  static Tolerance tolerance_for(int k, int input_bits) {
    const float log2_k = std::log2(static_cast<float>(std::max(k, 1)));
    const bool subbyte_input = input_bits < 8;
    const float base = subbyte_input ? 8e-3f : 4e-3f;
    const float atol = subbyte_input ? 1e-2f : 1e-4f;
    return {base * (1.0f + 0.1f * log2_k), atol};
  }

  // BF16 verify via host reference GEMM (no scaling); per-expert relative tol.
  template <class ElementA, class ElementB, class ElementD,
            class = std::enable_if_t<
                is_any_of_v<ElementA, cute::bfloat16_t, cute::half_t> &&
                is_any_of_v<ElementB, cute::bfloat16_t, cute::half_t> &&
                is_any_of_v<ElementD, cute::bfloat16_t, cute::half_t>>>
  bool verify(const ElementA *activations, const ElementB *weights,
              ElementD *outputs) {
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    sycl::queue Q = compat::get_default_queue();

    const int64_t A_elems = int64_t(m) * k;
    const int64_t B_elems = int64_t(groups) * n * k;
    const int64_t D_elems = int64_t(m) * n;
    std::vector<ElementA> h_A(A_elems);
    std::vector<ElementB> h_B(B_elems);
    std::vector<ElementD> h_D(D_elems);
    Q.memcpy(h_A.data(), activations, A_elems * sizeof(ElementA)).wait();
    Q.memcpy(h_B.data(), weights, B_elems * sizeof(ElementB)).wait();
    Q.memcpy(h_D.data(), outputs, D_elems * sizeof(ElementD)).wait();

    const auto tol = tolerance_for(k, cute::sizeof_bits_v<ElementA>);
    const float rtol = tol.rtol;
    const float atol = tol.atol;

    // Reference mirrors the device numerics: native input type, FP32
    // accumulation (like DPAS), round-to-ElementD on store like the epilogue.
    // Only accumulation ORDER differs, which tolerance_for() covers.
    std::vector<ElementD> h_C(D_elems, ElementD(0.f)); // beta=0, unused C
    std::vector<ElementD> h_ref_D;
    bool passed = true;
    int cumM = 0;
    for (int g = 0; g < groups; g++) {
      int Mg = cute::get<0>(problem_sizes_host[g]);
      const int64_t a_off = int64_t(cumM) * k;
      const int64_t b_off = int64_t(g) * n * k;
      const int64_t d_off = int64_t(cumM) * n;
      h_ref_D.assign(int64_t(Mg) * n, ElementD(0.f));

      cutlass::TensorRef<ElementA, LayoutA> ref_A(h_A.data() + a_off, LayoutA::packed({Mg, k}));
      cutlass::TensorRef<ElementB, LayoutB> ref_B(h_B.data() + b_off, LayoutB::packed({k, n}));
      cutlass::TensorRef<ElementD, LayoutD> ref_C(h_C.data() + d_off, LayoutD::packed({Mg, n}));
      cutlass::TensorRef<ElementD, LayoutD> ref_Dt(h_ref_D.data(),    LayoutD::packed({Mg, n}));

      cutlass::reference::host::compute_gemm<
          ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD,
          ElementAccumulator, ElementAccumulator>(
          {Mg, n, k}, ElementAccumulator(1.f), ref_A, ref_B,
          ElementAccumulator(0.f), ref_C, ref_Dt, ElementAccumulator(0.f));

      int mismatch_count = 0;
      for (int64_t idx = 0; idx < int64_t(Mg) * n; idx++) {
        float got = float(h_D[d_off + idx]);
        float ref = float(h_ref_D[idx]);
        if (!cutlass::relatively_equal(ref, got, rtol, atol)) {
          passed = false;
          if (mismatch_count < 10)
            std::cerr << "  mismatch expert=" << g << " idx=" << idx
                      << " got=" << got << " ref=" << ref
                      << " abs_error=" << std::abs(got - ref)
                      << " (rtol=" << rtol
                      << ", atol=" << atol << ")\n";
          if (++mismatch_count >= 100) {
            std::cerr << "  Stopping after 100 mismatches..." << std::endl;
            break;
          }
        }
      }
      if (g == 0) {
        std::cerr << "  [expert 0 first 10 elements]  got vs ref:" << std::endl;
        for (int64_t i = 0; i < std::min<int64_t>(10, int64_t(Mg) * n); i++)
          std::cerr << "    [" << i << "]  got=" << float(h_D[d_off + i])
                    << "  ref=" << float(h_ref_D[i]) << "\n";
      }
      cumM += Mg;
    }
    std::cerr << "\n=== Verification Summary ===" << std::endl
              << "  Result: " << (passed ? "PASSED" : "FAILED") << std::endl
              << "============================" << std::endl;
    return passed;
  }

  // Verify block/tensor scaled low-precision: dequant A/B to FP32 on host,
  // per-expert host reference GEMM, relative-tol compare.
  //
  // Scales read from the UNPADDED host grids (not the packed surface), keeping
  // verify independent of pack_moe_scales.
    template <bool IsTensor, bool BColMajor, class ScaleQ, class ElementA,
            class ElementScaleIn, class ElementD>
  bool verify_scaled(sycl::queue &Q, const ElementA *d_A, const ElementA *d_B,
                     const ElementD *d_D,
                     const ElementScaleIn *h_per_token_scale,
                     const ElementScaleIn *h_experts_scale,
                     int GroupN, int GroupK) {
    const int scale_k_logical = IsTensor ? 1 : (k + GroupK - 1) / GroupK;

    constexpr int kBitsPerA = cute::sizeof_bits_v<ElementA>;
    constexpr bool kSubbyte = (kBitsPerA < 8);

    int64_t A_elems = int64_t(m) * k;
    int64_t B_elems = int64_t(groups) * n * k;

    // Sub-byte types are copied as raw packed bytes and read via subbyte_iterator.
    std::vector<uint8_t> h_A_raw;
    std::vector<ElementA> h_A_full;
    std::vector<uint8_t> h_B_raw;
    std::vector<ElementA> h_B_full;
    if constexpr (kSubbyte) {
      constexpr int kElemsPerByte = 8 / kBitsPerA;
      h_A_raw.resize((A_elems + kElemsPerByte - 1) / kElemsPerByte);
      h_B_raw.resize((B_elems + kElemsPerByte - 1) / kElemsPerByte);
      Q.memcpy(h_A_raw.data(), (const uint8_t *)d_A, h_A_raw.size()).wait();
      Q.memcpy(h_B_raw.data(), (const uint8_t *)d_B, h_B_raw.size()).wait();
    } else {
      h_A_full.resize(A_elems);
      h_B_full.resize(B_elems);
      Q.memcpy(h_A_full.data(), d_A, A_elems * sizeof(ElementA)).wait();
      Q.memcpy(h_B_full.data(), d_B, B_elems * sizeof(ElementA)).wait();
    }

    auto get_A = [&](int64_t idx) -> float {
      if constexpr (kSubbyte) {
        cute::subbyte_iterator<const ElementA> it(h_A_raw.data());
        auto ref = it[idx];
        return float(ElementA(ref));
      } else {
        return float(h_A_full[idx]);
      }
    };
    auto get_B = [&](int64_t idx) -> float {
      if constexpr (kSubbyte) {
        cute::subbyte_iterator<const ElementA> it(h_B_raw.data());
        auto ref = it[idx];
        return float(ElementA(ref));
      } else {
        return float(h_B_full[idx]);
      }
    };

    auto scaleA = [&](int64_t global_tok, int kb) -> float {
      const int src_k = IsTensor ? 0 : kb;
      const int64_t idx = global_tok * scale_k_logical + src_k;
      return float(ScaleQ(float(h_per_token_scale[idx])));
    };
    auto scaleB = [&](int g, int col, int kb) -> float {
      const int64_t idx =
          IsTensor ? int64_t(g)
                   : (int64_t(g) * n + col) * scale_k_logical + kb;
      return float(ScaleQ(float(h_experts_scale[idx])));
    };

    // Dequantize into FP32 host buffers (same layout as the quantized inputs).
    std::vector<float> h_A_dq(A_elems);
    std::vector<float> h_B_dq(B_elems);
    int cumM = 0;
    for (int g = 0; g < groups; g++) {
      int Mg = cute::get<0>(problem_sizes_host[g]);
      for (int row = 0; row < Mg; row++) {
        const int64_t global_tok = cumM + row;
        for (int kk = 0; kk < k; kk++) {
          int kb = kk / GroupK;
          h_A_dq[global_tok * k + kk] =
              get_A(global_tok * k + kk) * scaleA(global_tok, kb);
        }
      }
      for (int kk = 0; kk < k; kk++) {
        int kb = kk / GroupK;
        for (int col = 0; col < n; col++) {
          const float sb = scaleB(g, col, kb);
          // B element (kk, col) within the expert's flat buffer:
          //   RowMajor B: kk*n + col;  ColumnMajor B: col*k + kk
          const int64_t b_elem = BColMajor ? (int64_t(col) * k + kk)
                                           : (int64_t(kk) * n + col);
          const int64_t b_idx = int64_t(g) * n * k + b_elem;
          h_B_dq[b_idx] = get_B(b_idx) * sb;
        }
      }
      cumM += Mg;
    }

    using LayoutA = cutlass::layout::RowMajor;
    // B reference layout mirrors the dequant fill above.
    using LayoutB = cute::conditional_t<BColMajor, cutlass::layout::ColumnMajor,
                                        cutlass::layout::RowMajor>;
    using LayoutD = cutlass::layout::RowMajor;

    std::vector<ElementD> h_D_raw(int64_t(m) * n);
    Q.memcpy(h_D_raw.data(), d_D, int64_t(m) * n * sizeof(ElementD)).wait();

    const auto tol = tolerance_for(k, kBitsPerA);
    const float rtol = tol.rtol;
    const float atol = tol.atol;

    // A/B stay FP32: the dequantized value (quant*scale) isn't representable in
    // ElementA, and BDPAS applies the scale in FP32 too. D is ElementD to model
    // the epilogue's round-to-output.
    std::vector<ElementD> h_C(int64_t(m) * n, ElementD(0.f)); // beta=0, unused C
    std::vector<ElementD> h_ref_D;
    bool passed = true;
    cumM = 0;
    for (int g = 0; g < groups; g++) {
      int Mg = cute::get<0>(problem_sizes_host[g]);
      const int64_t dq_off = int64_t(cumM) * k;
      const int64_t b_off  = int64_t(g) * n * k;
      const int64_t d_off  = int64_t(cumM) * n;
      h_ref_D.assign(int64_t(Mg) * n, ElementD(0.f));

      cutlass::TensorRef<float, LayoutA> ref_A(h_A_dq.data() + dq_off, LayoutA::packed({Mg, k}));
      cutlass::TensorRef<float, LayoutB> ref_B(h_B_dq.data() + b_off,  LayoutB::packed({k, n}));
      cutlass::TensorRef<ElementD, LayoutD> ref_C(h_C.data() + d_off,  LayoutD::packed({Mg, n}));
      cutlass::TensorRef<ElementD, LayoutD> ref_Dt(h_ref_D.data(),     LayoutD::packed({Mg, n}));

      cutlass::reference::host::compute_gemm<
          float, LayoutA, float, LayoutB, ElementD, LayoutD,
          ElementAccumulator, ElementAccumulator>(
          {Mg, n, k}, ElementAccumulator(1.f), ref_A, ref_B,
          ElementAccumulator(0.f), ref_C, ref_Dt, ElementAccumulator(0.f));

      int mismatch_count = 0;
      for (int64_t idx = 0; idx < int64_t(Mg) * n; idx++) {
        float got = float(h_D_raw[d_off + idx]);
        float ref = float(h_ref_D[idx]);
        if (!cutlass::relatively_equal(ref, got, rtol, atol)) {
          passed = false;
          if (mismatch_count < 10)
            std::cerr << "  mismatch expert=" << g << " idx=" << idx
                      << " got=" << got << " ref=" << ref
                      << " abs_error=" << std::abs(got - ref)
                      << " (rtol=" << rtol
                      << ", atol=" << atol << ")\n";
          if (++mismatch_count >= 100) {
            std::cerr << "  Stopping after 100 mismatches..." << std::endl;
            break;
          }
        }
      }
      if (g == 0) {
        std::cerr << "  [expert 0 first 10 elements]  got vs ref:" << std::endl;
        for (int64_t i = 0; i < std::min<int64_t>(10, int64_t(Mg) * n); i++)
          std::cerr << "    [" << i << "]  got=" << float(h_D_raw[d_off + i])
                    << "  ref=" << float(h_ref_D[i]) << std::endl;
      }
      cumM += Mg;
    }
    std::cerr << "\n=== Verification Summary ===" << std::endl
              << "  Result: " << (passed ? "PASSED" : "FAILED") << std::endl
              << "============================" << std::endl;
    return passed;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Per-target workgroup tile from a Config. TileShape{Cri,Bmg} are dependent
// names, so this may precede the config definitions.
template <class Config>
using MoETileShape =
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
    typename Config::TileShapeCri;
#else
    typename Config::TileShapeBmg;
#endif

namespace detail {
// SGLayout selection: Config::SGLayout if declared, else Def.
template <class C, class Def, class = void> struct ConfigSGLayoutT { using type = Def; };
template <class C, class Def>
struct ConfigSGLayoutT<C, Def, cute::void_t<typename C::SGLayout>> {
  using type = typename C::SGLayout;
};
template <class C, class Def> using ConfigSGLayout = typename ConfigSGLayoutT<C, Def>::type;
} // namespace detail

template <class Config, class TA, class TB>
auto choose_tiled_mma(TA *A, TB *B) {
  using TA_non_CV = cutlass::platform::remove_cv_t<TA>;
  using TB_non_CV = cutlass::platform::remove_cv_t<TB>;

  // Subgroup tiling, n-major, per hardware target. Per-SG N (BLK_N/SG_N) must
  // stay >= the GroupN scale-broadcast width in the block-scale mainloop, else
  // the broadcast breaks (e.g. 4x8 layout -> per-SG N=16).
#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
  using DefaultSGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
#else
  using DefaultSGLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
#endif
  using SGLayout = detail::ConfigSGLayout<Config, DefaultSGLayout>;

  using WGTile = MoETileShape<Config>;
  if constexpr (Config::scale_kind == ScaleKind::Plain) {
    auto op = XE_DPAS_TT<8, float, TA_non_CV, TB_non_CV>{};
    using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                        SGLayout>::TiledMMA;
    return MMA{};
  } else {
    // Block-scaled DPAS. WGTile K must be a multiple of the DPAS atom K
    // (32 for 8-bit, 64 for 4-bit).
    auto op = XE_BDPAS_TT<8, float, TA_non_CV>{};
    using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                        SGLayout>::TiledMMA;
    return MMA{};
  }
}

// Unique sycl kernel name. Trailing Config is required: A/B/D + layouts +
// TileShape aren't unique alone — configs can share them yet differ in scaling
// (fp8-tensor vs mxfp8-e4m3 are both e4m3->bf16 at 256x256x64), and one binary
// compiles more than one config.
template <typename, typename, typename, typename, typename, typename, typename>
class GemmCuteName;

///////////////////////////////////////////////////////////////////////////////////////////////////
// GREEDY launch (plain bf16). Per-expert tile size chosen ON-DEVICE by the
// greedy split; LARGE/SMALL/TINY variants share one WG thread count so a single
// nd_range hosts all three. Grid = sm_count persistent workgroups.
// Config::is_dynamic_m selects uniform-M vs dynamic-M instantiation.
template <class Config, typename ElementA, typename ElementScaleIn, typename ElementD>
double moe_launch_timed_greedy(
    const VendorTensorMapping<ElementA, ElementScaleIn, ElementD> &tm) {
  using LayoutA = typename Config::LayoutA;
  using LayoutB = typename Config::LayoutB;
  using LayoutD = typename Config::LayoutD;
  using SubgroupLayout = typename Config::SGLayout;
  using LargeTile = typename Config::LargeTile;
  using SmallTile = typename Config::SmallTile;
  using TinyTile  = typename Config::TinyTile;
  using TinySGLayout = typename Config::TinySGLayout;

  using DpasOp = XE_DPAS_TT<8, float, ElementA, ElementA>;
  using MmaLarge = typename TiledMMAHelper<
      MMA_Atom<DpasOp>, Layout<LargeTile>, SubgroupLayout>::TiledMMA;
  using MmaSmall = typename TiledMMAHelper<
      MMA_Atom<DpasOp>, Layout<SmallTile>, SubgroupLayout>::TiledMMA;
  using MmaTiny = typename TiledMMAHelper<
      MMA_Atom<DpasOp>, Layout<TinyTile>, TinySGLayout>::TiledMMA;
  static_assert(size(MmaLarge{}) == size(MmaSmall{}) &&
                size(MmaLarge{}) == size(MmaTiny{}),
                "greedy tile variants must share one workgroup thread count");

  const ElementA *activations = tm.scatter_tokens;
  const ElementA *weights     = tm.experts_weight;
  ElementD       *outputs     = tm.y;
  const int32_t *num_rows_per_expert_device = tm.experts_token_count_device;
  const int gemm_n = tm.N, gemm_k = tm.K, num_experts = tm.num_experts;
  // Uniform per-expert M, computed host-side and passed as a scalar so the
  // uniform-M kernel path needn't read it back from the device counts array.
  const int32_t uniform_m =
      (num_experts > 0 && tm.experts_token_count) ? tm.experts_token_count[0] : 0;

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  const size_t threads_per_workgroup = size(MmaLarge{});
  sycl::range<3> local_range  = {1, 1, threads_per_workgroup};
  sycl::range<3> group_range  = {1, 1, static_cast<size_t>(sm_count)};
  sycl::range<3> global_range = {local_range[0] * group_range[0],
                                 local_range[1] * group_range[1],
                                 local_range[2] * group_range[2]};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<16>,
#if (defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
                                  intelex::grf_size<512>
#else
                                  intelex::grf_size<256>
#endif
  };
  sycl::queue Q = compat::get_default_queue();
  GPU_Clock timer;
  timer.start();
  auto event = Q.parallel_for<
      GemmCuteName<ElementA, ElementA, ElementD, LayoutA, LayoutB, LargeTile, Config>>(
      sycl::nd_range<3>(global_range, local_range), kernel_props, [=](auto) {
        MoE::MoEGEMMGreedy<Config::is_dynamic_m, void, void, void,
                           LayoutA, LayoutB, LayoutD,
                           MmaLarge, MmaSmall, MmaTiny>(
            activations, weights, outputs, num_rows_per_expert_device,
            num_experts, gemm_n, gemm_k, uniform_m);
      });
  EventManager::getInstance().addEvent(event);
  Q.wait_and_throw();
  return double(timer.seconds() * 1000);
}

// GREEDY scaled launch (fp8, mxfp8, mxfp4): builds BDPAS MMAs (incl. TINY) and
// passes scale pointers to MoEGEMMGreedyScaled. Config::is_dynamic_m selects mode.
template <class Config, typename ElementA, typename ElementB, typename ElementS,
          typename ElementD>
double moe_launch_timed_greedy_scaled(
    const ElementA *activations, const ElementB *weights,
    const ElementS *scalesA, const ElementS *scalesB, ElementD *outputs,
    const int gemm_n, const int gemm_k,
    const int32_t *num_rows_per_expert_device, const int num_experts,
    const int32_t uniform_m) {
  static_assert(Config::scale_kind != ScaleKind::Plain,
                "use moe_launch_timed_greedy for plain BF16");
  using LayoutA = typename Config::LayoutA;
  using LayoutB = typename Config::LayoutB;
  using LayoutD = typename Config::LayoutD;
  using SubgroupLayout = typename Config::SGLayout;
  using LargeTile = typename Config::LargeTile;
  using SmallTile = typename Config::SmallTile;
  using TinyTile  = typename Config::TinyTile;
  using TinySGLayout = typename Config::TinySGLayout;

  using DpasOp = XE_BDPAS_TT<8, float, cutlass::platform::remove_cv_t<ElementA>>;
  using MmaLarge = typename TiledMMAHelper<
      MMA_Atom<DpasOp>, Layout<LargeTile>, SubgroupLayout>::TiledMMA;
  using MmaSmall = typename TiledMMAHelper<
      MMA_Atom<DpasOp>, Layout<SmallTile>, SubgroupLayout>::TiledMMA;
  using MmaTiny = typename TiledMMAHelper<
      MMA_Atom<DpasOp>, Layout<TinyTile>, TinySGLayout>::TiledMMA;
  static_assert(size(MmaLarge{}) == size(MmaSmall{}) &&
                size(MmaLarge{}) == size(MmaTiny{}),
                "greedy tile variants must share one workgroup thread count");

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  const size_t threads_per_workgroup = size(MmaLarge{});
  sycl::range<3> local_range  = {1, 1, threads_per_workgroup};
  sycl::range<3> group_range  = {1, 1, static_cast<size_t>(sm_count)};
  sycl::range<3> global_range = {local_range[0] * group_range[0],
                                 local_range[1] * group_range[1],
                                 local_range[2] * group_range[2]};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{syclex::sub_group_size<16>,
#if (defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
                                  intelex::grf_size<512>
#else
                                  intelex::grf_size<256>
#endif
  };
  sycl::queue Q = compat::get_default_queue();
  GPU_Clock timer;
  timer.start();
  auto event = Q.parallel_for<
      GemmCuteName<ElementA, ElementB, ElementD, LayoutA, LayoutB, LargeTile, Config>>(
      sycl::nd_range<3>(global_range, local_range), kernel_props, [=](auto) {
        MoE::MoEGEMMGreedyScaled<Config::is_dynamic_m, void, void, void,
                                 LayoutA, LayoutB, LayoutD,
                                 Config::group_n, Config::group_k,
                                 MmaLarge, MmaSmall, MmaTiny,
                                 ElementA, ElementB, ElementS, ElementD>(
            activations, weights, scalesA, scalesB, outputs,
            num_rows_per_expert_device, num_experts, gemm_n, gemm_k,
            Config::group_n, Config::group_k, uniform_m);
      });
  EventManager::getInstance().addEvent(event);
  Q.wait_and_throw();
  return double(timer.seconds() * 1000);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// SG tile-sweep configs, used only by the device-codegen TU (moe_api.cpp).
///////////////////////////////////////////////////////////////////////////////////////////////////


using SG_8x4 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
using SG_8x2 = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
// SG_4x8 comes from moe_types.hpp (greedy configs); do not redefine here.
// (Single-buffer LowpConfigSG / Bf16ConfigSG / MxFp4ConfigSG removed: greedy
//  replaced that path. The double-buffer *DoubleBufferConfigSG templates below
//  are still used.)

using MoeTile_256_256_32 = Shape<_256, _256, _32>;
using MoeTile_256_256_64 = Shape<_256, _256, _64>;
using MoeTile_256_256_128 = Shape<_256, _256, _128>;
// (MoeTile_*_512_* aliases come from moe_types.hpp; not redefined here.)
using MoeTile_384_320_64 = Shape<cute::Int<384>, cute::Int<320>, _64>;
using MoeTile_384_320_128 = Shape<cute::Int<384>, cute::Int<320>, _128>;
// Double-buffer tile shapes.
using MoeTile_192_256_64  = Shape<cute::Int<192>, _256, _64>;
using MoeTile_192_256_128 = Shape<cute::Int<192>, _256, _128>;
using MoeTile_224_256_32  = Shape<cute::Int<224>, _256, _32>;
using MoeTile_224_256_64  = Shape<cute::Int<224>, _256, _64>;
using MoeTile_224_256_128 = Shape<cute::Int<224>, _256, _128>;
using MoeTile_288_256_64  = Shape<cute::Int<288>, _256, _64>;
using MoeTile_288_256_128 = Shape<cute::Int<288>, _256, _128>;

// (Single-buffer SG-sweep configs removed: greedy replaced that path.)

using MoeTile_192_256_32 = Shape<cute::Int<192>, cute::Int<256>, cute::Int<32>>;
using MoeTile_192_384_32 = Shape<cute::Int<192>, cute::Int<384>, cute::Int<32>>;
using MoeTile_192_384_64 = Shape<cute::Int<192>, cute::Int<384>, cute::Int<64>>;
using MoeTile_192_384_128 = Shape<cute::Int<192>, cute::Int<384>, cute::Int<128>>;
using MoeTile_288_256_32 = Shape<cute::Int<288>, cute::Int<256>, cute::Int<32>>;
using MoeTile_320_192_32 = Shape<cute::Int<320>, cute::Int<192>, cute::Int<32>>;
using MoeTile_320_192_64 = Shape<cute::Int<320>, cute::Int<192>, cute::Int<64>>;
using MoeTile_320_192_128 = Shape<cute::Int<320>, cute::Int<192>, cute::Int<128>>;
using MoeTile_384_192_32 = Shape<cute::Int<384>, cute::Int<192>, cute::Int<32>>;

template <class TTileCri, class TSG>
struct Bf16DoubleBufferConfigSG {
  using Element = cutlass::bfloat16_t;
  using ElementScale = void;
  using ElementOutput = cutlass::bfloat16_t;
  using TileShapeCri = TTileCri;
  using TileShapeBmg = TTileCri;
  using SGLayout = TSG;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int group_k = 0;
  static constexpr int group_n = 0;
  static constexpr ScaleKind scale_kind = ScaleKind::Plain;
  static constexpr bool uniform_m = true;
};

// Uniform-M launch from the client's mapping. uniform_m is the one argument not
// taken from it: the mapping holds per-expert counts, and collapsing them to a
// single M is the client's call (see MoEBenchmarkRunner::run).
template <class Config, typename ElementA, typename ElementScaleIn,
          typename ElementD>
double moe_launch_timed_double_buffer(
    const VendorTensorMapping<ElementA, ElementScaleIn, ElementD> &tm,
    const int uniform_m) {
  static_assert(Config::scale_kind == ScaleKind::Plain,
                "moe_launch_timed_double_buffer only supports plain BF16");

  const ElementA *activations = tm.scatter_tokens;
  const ElementA *weights     = tm.experts_weight;
  ElementD       *outputs     = tm.y;
  const int gemm_n = tm.N, gemm_k = tm.K;
  const int num_experts = tm.num_experts;

  using LayoutA = typename Config::LayoutA;
  using LayoutB = typename Config::LayoutB;
  using LayoutD = typename Config::LayoutD;

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info{0, sm_count};

  auto dummy_problem_shape = cute::Shape<int, int, int>{1, gemm_k, gemm_n};
  auto dummy_group_problem_shape =
      cutlass::gemm::GroupProblemShape<Shape<int, int, int>>{
          1, &dummy_problem_shape, nullptr};

  using TileShape = MoETileShape<Config>;
  using ClusterShape = Shape<_1, _1, _1>;

  // The uniform kernel only reads raster_order_ from the params; grid is a flat
  // sm_count x 1 x 1, so no tile-count math needed.
  auto scheduler_params =
      PersistentTileSchedulerXeMoE<ProblemShape>::to_underlying_arguments(
          dummy_group_problem_shape, TileShape{}, ClusterShape{}, hw_info,
          PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{
              1, RasterOrderOptions::AlongN});

  auto mma = choose_tiled_mma<Config>(activations, weights);
  auto MaxThreadsPerWorkgroup = size(mma);
  dim3 local_range{MaxThreadsPerWorkgroup, 1, 1};

  sycl::range<3> local  = {1, 1, local_range.x};
  sycl::range<3> groups = {1, 1, static_cast<size_t>(sm_count)};
  sycl::range<3> global = {local[0] * groups[0], local[1] * groups[1],
                           local[2] * groups[2]};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>,
#if (defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
                                  intelex::grf_size<512>
#else
                                  intelex::grf_size<256>
#endif
  };
  sycl::queue Q = compat::get_default_queue();
#ifdef FULL_RUN_TIMING_AND_VERIFY
  GPU_Clock timer;
  timer.start();
#endif

  auto event = Q.parallel_for<
      GemmCuteName<ElementA, ElementA, ElementD, LayoutA, LayoutB, TileShape,
                   Config>>(
      sycl::nd_range<3>(global, local), kernel_props, [=](auto) {
        MoE::MoEGEMMDoubleBuffer<void, void, void,
                                 LayoutA, LayoutB, LayoutD,
                                 decltype(mma), ElementA, ElementA, void, ElementD>(
            activations, weights, static_cast<const void *>(nullptr),
            outputs, mma, uniform_m, num_experts, gemm_n, gemm_k,
            scheduler_params);
      });
  EventManager::getInstance().addEvent(event);
  Q.wait_and_throw();
#ifdef FULL_RUN_TIMING_AND_VERIFY
  return double(timer.seconds() * 1000);
#else
  return 0.0;
#endif
}

using Bf16DoubleBuffer_224_256_32 = Bf16DoubleBufferConfigSG<MoeTile_224_256_32, SG_4x8>;
using Bf16DoubleBuffer_256_256_32 = Bf16DoubleBufferConfigSG<MoeTile_256_256_32, SG_8x4>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Scaled uniform-M double-buffer configs + launcher.
// Mirrors LowpConfigSG but adds uniform_m = true so benchmark wiring can
// distinguish it from the variable-M path.
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class TElement, class TScale, int GroupK, int GroupN, ScaleKind Kind,
          class TTileCri, class TSG, class TOut = cutlass::bfloat16_t>
struct ScaledDoubleBufferConfigSG {
  using Element = TElement;
  using ElementScale = TScale;
  using ElementOutput = TOut;
  using TileShapeCri = TTileCri;
  using TileShapeBmg = TTileCri;
  using SGLayout = TSG;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int group_k = GroupK;
  static constexpr int group_n = GroupN;
  static constexpr ScaleKind scale_kind = Kind;
  static constexpr bool uniform_m = true;

  // SG_N must be divisible by 16, else silent incorrect D writes for experts 1+.
  static constexpr int SG_NUMS_N = get<1>(TSG{}.shape());
  static constexpr int BLK_N    = get<1>(TTileCri{});
  static constexpr int SG_N     = BLK_N / SG_NUMS_N;
  static_assert(SG_N % 16 == 0,
      "Double-buffer D store requires SG_N (BLK_N/NUMS_N) divisible by 16.");
};

// MxFp4 double-buffer wrapper: ColumnMajor B, like MxFp4ConfigSG.
template <class TElement, class TScale, int GroupK, int GroupN, ScaleKind Kind,
          class TTileCri, class TSG, class TOut = cutlass::bfloat16_t>
struct MxFp4ScaledDoubleBufferConfigSG
    : ScaledDoubleBufferConfigSG<TElement, TScale, GroupK, GroupN, Kind, TTileCri, TSG, TOut> {
  using LayoutB = cutlass::layout::ColumnMajor;
};

// Scaled uniform-M launch from the client's mapping; uniform_m as above. Scale
// block sizes come from the Config, so runtime values can't disagree with the
// template arguments the kernel is instantiated with.
template <class Config, typename ElementA, typename ElementScaleIn,
          typename ElementD>
double moe_launch_timed_double_buffer_scaled(
    const VendorTensorMapping<ElementA, ElementScaleIn, ElementD> &tm,
    const int uniform_m) {
  static_assert(Config::scale_kind != ScaleKind::Plain,
                "use moe_launch_timed_double_buffer for plain BF16");

  using ElementS = ElementScaleIn;
  const ElementA *activations = tm.scatter_tokens;
  const ElementA *weights     = tm.experts_weight;
  const ElementS *scalesA     = tm.packed_scale_a;
  const ElementS *scalesB     = tm.packed_scale_b;
  ElementD       *outputs     = tm.y;
  const int gemm_n = tm.N, gemm_k = tm.K;
  const int num_experts = tm.num_experts;
  constexpr int group_n = Config::group_n;
  constexpr int group_k = Config::group_k;

  using LayoutA = typename Config::LayoutA;
  using LayoutB = typename Config::LayoutB;
  using LayoutD = typename Config::LayoutD;

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  cutlass::KernelHardwareInfo hw_info{0, sm_count};

  auto dummy_problem_shape = cute::Shape<int, int, int>{1, gemm_k, gemm_n};
  auto dummy_group_problem_shape =
      cutlass::gemm::GroupProblemShape<Shape<int, int, int>>{
          1, &dummy_problem_shape, nullptr};

  using TileShape    = MoETileShape<Config>;
  using ClusterShape = Shape<_1, _1, _1>;

  auto scheduler_params =
      PersistentTileSchedulerXeMoE<ProblemShape>::to_underlying_arguments(
          dummy_group_problem_shape, TileShape{}, ClusterShape{}, hw_info,
          PersistentTileSchedulerXeMoE<ProblemShape>::Arguments{
              1, RasterOrderOptions::AlongN});

  auto mma = choose_tiled_mma<Config>(activations, weights);
  auto MaxThreadsPerWorkgroup = size(mma);

  sycl::range<3> local  = {1, 1, static_cast<size_t>(MaxThreadsPerWorkgroup)};
  sycl::range<3> groups = {1, 1, static_cast<size_t>(sm_count)};
  sycl::range<3> global = {local[0] * groups[0], local[1] * groups[1],
                           local[2] * groups[2]};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{syclex::sub_group_size<16>,
#if (defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
                                  intelex::grf_size<512>
#else
                                  intelex::grf_size<256>
#endif
  };
  sycl::queue Q = compat::get_default_queue();
#ifdef FULL_RUN_TIMING_AND_VERIFY
  GPU_Clock timer;
  timer.start();
#endif
  auto event = Q.parallel_for<
      GemmCuteName<ElementA, ElementA, ElementD, LayoutA, LayoutB, TileShape,
                   Config>>(
      sycl::nd_range<3>(global, local), kernel_props, [=](auto) {
        MoE::MoEGEMMDoubleBufferScaled<void, void, void,
                                       LayoutA, LayoutB, LayoutD,
                                       Config::group_n, Config::group_k,
                                       decltype(mma),
                                       ElementA, ElementA, ElementS, ElementD>(
            activations, weights, scalesA, scalesB,
            outputs, mma, uniform_m, num_experts, gemm_n, gemm_k,
            group_n, group_k, scheduler_params);
      });
  EventManager::getInstance().addEvent(event);
  Q.wait_and_throw();
#ifdef FULL_RUN_TIMING_AND_VERIFY
  return double(timer.seconds() * 1000);
#else
  return 0.0;
#endif
}

// Scaled double-buffer configs, one per dtype.
using MxFp4DoubleBuffer_256_256_128 = MxFp4ScaledDoubleBufferConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_256_128, SG_8x4>;
using Fp8TensorDoubleBuffer_256_256_64 = ScaledDoubleBufferConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 0, 0, ScaleKind::Tensor, MoeTile_256_256_64, SG_8x4>;
using MxFp4DoubleBuffer_192_256_128 = MxFp4ScaledDoubleBufferConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_192_256_128, SG_4x8>;
using MxFp4DoubleBuffer_224_256_128 = MxFp4ScaledDoubleBufferConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_224_256_128, SG_4x8>;
using MxFp4DoubleBuffer_288_256_128 = MxFp4ScaledDoubleBufferConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_288_256_128, SG_4x8>;
using MoeTile_224_512_64 = Shape<cute::Int<224>, cute::Int<512>, cute::Int<64>>;
using MoeTile_320_384_64 = Shape<cute::Int<320>, cute::Int<384>, cute::Int<64>>;
using MoeTile_384_256_64 = Shape<cute::Int<384>, cute::Int<256>, cute::Int<64>>;
using MoeTile_384_320_64 = Shape<cute::Int<384>, cute::Int<320>, cute::Int<64>>;
// (MoeTile_192_512_128 / _192_512_32 / _256_512_32 come from moe_types.hpp.)
using MoeTile_224_512_128 = Shape<cute::Int<224>, cute::Int<512>, cute::Int<128>>;
using MoeTile_320_384_128 = Shape<cute::Int<320>, cute::Int<384>, cute::Int<128>>;
using MoeTile_384_256_128 = Shape<cute::Int<384>, cute::Int<256>, cute::Int<128>>;
using MoeTile_384_320_128 = Shape<cute::Int<384>, cute::Int<320>, cute::Int<128>>;
using MoeTile_224_512_32 = Shape<cute::Int<224>, cute::Int<512>, cute::Int<32>>;
using MoeTile_320_384_32 = Shape<cute::Int<320>, cute::Int<384>, cute::Int<32>>;
using MoeTile_384_256_32 = Shape<cute::Int<384>, cute::Int<256>, cute::Int<32>>;
using MoeTile_384_320_32 = Shape<cute::Int<384>, cute::Int<320>, cute::Int<32>>;
// (Single-buffer SG-sweep configs removed: greedy replaced that path.)
using Fp8TensorDoubleBuffer_192_256_64 = ScaledDoubleBufferConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 0, 0, ScaleKind::Tensor, MoeTile_192_256_64, SG_4x8>;
using MxFp8DoubleBuffer_192_256_64 = ScaledDoubleBufferConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_192_256_64, SG_4x8>;
using Fp8TensorDoubleBuffer_192_384_64 = ScaledDoubleBufferConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 0, 0, ScaleKind::Tensor, MoeTile_192_384_64, SG_4x8>;
using MxFp8DoubleBuffer_192_384_64 = ScaledDoubleBufferConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_192_384_64, SG_4x8>;
using Fp8TensorDoubleBuffer_224_256_64 = ScaledDoubleBufferConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 0, 0, ScaleKind::Tensor, MoeTile_224_256_64, SG_4x8>;
using MxFp8DoubleBuffer_224_256_64 = ScaledDoubleBufferConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_224_256_64, SG_4x8>;
using MxFp8DoubleBuffer_256_256_64 = ScaledDoubleBufferConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_256_256_64, SG_8x4>;
using Fp8TensorDoubleBuffer_288_256_64 = ScaledDoubleBufferConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 0, 0, ScaleKind::Tensor, MoeTile_288_256_64, SG_4x8>;
using MxFp8DoubleBuffer_288_256_64 = ScaledDoubleBufferConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_288_256_64, SG_4x8>;
using Fp8TensorDoubleBuffer_320_192_64 = ScaledDoubleBufferConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 0, 0, ScaleKind::Tensor, MoeTile_320_192_64, SG_8x4>;
using MxFp8DoubleBuffer_320_192_64 = ScaledDoubleBufferConfigSG<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_320_192_64, SG_8x4>;
using MxFp4DoubleBuffer_192_384_128 = MxFp4ScaledDoubleBufferConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_192_384_128, SG_4x8>;
using MxFp4DoubleBuffer_320_192_128 = MxFp4ScaledDoubleBufferConfigSG<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32, 1, ScaleKind::Block, MoeTile_320_192_128, SG_8x4>;
using Bf16DoubleBuffer_192_256_32 = Bf16DoubleBufferConfigSG<MoeTile_192_256_32, SG_4x8>;
using Bf16DoubleBuffer_192_384_32 = Bf16DoubleBufferConfigSG<MoeTile_192_384_32, SG_4x8>;
using Bf16DoubleBuffer_288_256_32 = Bf16DoubleBufferConfigSG<MoeTile_288_256_32, SG_4x8>;
using Bf16DoubleBuffer_320_192_32 = Bf16DoubleBufferConfigSG<MoeTile_320_192_32, SG_8x4>;
using Bf16DoubleBuffer_384_192_32 = Bf16DoubleBufferConfigSG<MoeTile_384_192_32, SG_8x4>;


} // namespace cutlass::moe
