/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation. All rights reserved.
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

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/float_subbyte.h>
#include <cutlass/util/command_line.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "validation.hpp"
#include "../../common/sycl_cute_common.hpp"

using namespace cute;
using namespace sycl;

// Type-name helpers used in config names and banners.
template <class T> constexpr const char* type_name() {
  if constexpr (std::is_same_v<T, cutlass::float_e2m1_t>)       return "fp4";
  if constexpr (std::is_same_v<T, cutlass::float_e4m3_t>)       return "fp8";
  if constexpr (std::is_same_v<T, float>)                        return "fp32";
  if constexpr (std::is_same_v<T, sycl::half>)                   return "fp16";
  if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) return "bf16";
  return "unknown";
}
template <class T> constexpr const char* sf_tag() {
  if constexpr (std::is_same_v<T, cutlass::float_ue4m3_t>) return "ue4m3";
  if constexpr (std::is_same_v<T, cutlass::float_ue5m3_t>) return "ue5m3";
  return "ue8m0";
}

// Runtime override for problem shape via command line: --m=<int> --n=<int> --k=<int> [--l=<int>]
// Dimensions not specified on the command line retain the per-config default.
//
// Note: print_usage() is intentionally omitted — each executable defines its own
// to list program-specific configs and examples.
struct Options {
  static inline std::optional<int> m, n, k, l;
  static inline std::vector<std::string> configs;
  static inline bool help = false;
  static inline bool run_all = false;
  static inline std::optional<bool> coop_sf;

  // Defined in the .cpp file — parses general flags
  // (--help, --run-all, --m/--n/--k/--l, --coop_sf) and blockscaled operand shorthand
  // (--A/--sfA/--blockA, --B/--sfB/--blockB, --D).
  static void parse(int argc, char** argv);

  template <typename Config>
  static cute::array<int, 4> get_problem_shape() {
    auto s = Config::ProblemShape_MNKL;
    if (m) s[0] = *m;
    if (n) s[1] = *n;
    if (k) s[2] = *k;
    if (l) s[3] = *l;
    return s;
  }
};

// Wrapper that overrides a Config's EnableCooperativeSF at compile time.
// Used to dispatch --coop_sf runtime flag to the correct template instantiation.
// When cooperative SF is enabled, also overrides CtaTileShape_MNK with a larger TileK
// to satisfy the cm_8x32B constraint (VS=16 → TileK=256, VS=32 → TileK=512).
template <typename BaseConfig, bool CoopSFOverride>
struct ConfigWithCoopSF : BaseConfig {
  static constexpr bool EnableCooperativeSF = CoopSFOverride;
};

template <typename BaseConfig>
struct ConfigWithCoopSF<BaseConfig, true> : BaseConfig {
  static constexpr bool EnableCooperativeSF = true;
  static constexpr int CoopTileK = (BaseConfig::SFVecSizeA == 16) ? 256 : 512;
  using CtaTileShape_MNK = Shape<_128, _256, cute::Int<CoopTileK>>;

  // Validate that the cooperative TileK satisfies the cm_8x32B constraint.
  // After ADMA box truncation by max(ClusterM, ClusterN), each CTA's SF K-dimension
  // must be a multiple of 8 rows. The hardware's cm_8x32B core-matrix always writes
  // 8 rows per unit; if a CTA's SF row count is not a multiple of 8 (e.g. 12), the
  // last core-matrix write overflows into a peer CTA's SF region within the same
  // pipeline stage, causing intra-stage data corruption that padding cannot fix
  // (padding only prevents inter-stage overflow between pipeline stages).
  static constexpr int MaxClusterDim_ =
      (static_cast<int>(cute::size<0>(typename BaseConfig::ClusterShape_MNK{})) >
       static_cast<int>(cute::size<1>(typename BaseConfig::ClusterShape_MNK{})))
      ? static_cast<int>(cute::size<0>(typename BaseConfig::ClusterShape_MNK{}))
      : static_cast<int>(cute::size<1>(typename BaseConfig::ClusterShape_MNK{}));
  static_assert((CoopTileK / BaseConfig::SFVecSizeA / MaxClusterDim_ >= 8) &&
      (CoopTileK / BaseConfig::SFVecSizeA / MaxClusterDim_) % 8 == 0,
      "Cooperative SF loading requires (CoopTileK / SFVecSize / max(ClusterM, ClusterN)) to be "
      "a multiple of 8 (>= 8). The ADMA 2D-block-copy writes in cm_8x32B core-matrix units "
      "(8 rows). If the per-CTA SF row count is not a multiple of 8 (e.g. 12), the last "
      "core-matrix write overflows into a peer CTA's SF region within the same pipeline "
      "stage, causing intra-stage collision that padding cannot fix. "
      "Increase CoopTileK or reduce cluster dimensions.");
};

// Shared memory info utility for block-scaled mainloop (extends the regular version with SF buffers).
#define GET_MEM_INFO(cls, path) std::make_tuple((size_t) & (((cls *)0)->path), sizeof(((cls *)0)->path))

template<typename GemmKernel>
std::string get_shared_memory_info_blockscaled() {
  using TensorStorage = typename GemmKernel::TensorStorage;
  using SharedStorage = typename GemmKernel::SharedStorage;

  std::ostringstream oss;

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

  // Mainloop buffers
  auto [offsetA, sizeA]     = GET_MEM_INFO(TensorStorage, mainloop.smem_A);
  auto [offsetB, sizeB]     = GET_MEM_INFO(TensorStorage, mainloop.smem_B);
  auto [offsetAcc, sizeAcc] = GET_MEM_INFO(TensorStorage, mainloop.smem_Acc);
  auto [offsetSFA, sizeSFA] = GET_MEM_INFO(TensorStorage, mainloop.smem_SFA);
  auto [offsetSFB, sizeSFB] = GET_MEM_INFO(TensorStorage, mainloop.smem_SFB);

  // Epilogue buffers
  auto [offsetEpiC, sizeEpiC]   = GET_MEM_INFO(TensorStorage, epilogue.collective.smem_C);
  auto [offsetEpiD, sizeEpiD]   = GET_MEM_INFO(TensorStorage, epilogue.collective.smem_D);

  // CLC responses
  auto [offsetCLC, sizeCLC]     = GET_MEM_INFO(TensorStorage, clc_response);

  // Section sizes
  size_t mainloop_size  = sizeof(typename TensorStorage::MainloopTensorStorage);
  size_t epilogue_size  = sizeof(typename TensorStorage::EpilogueTensorStorage);
  size_t tensor_size    = sizeof(TensorStorage);
  size_t pipeline_size  = sizeof(typename SharedStorage::PipelineStorage);

  oss << "SLM Allocation (SLM total: " << bytes2kb(tensor_size) << ")" << std::endl;
  oss << "- Mainloop (" << bytes2kb(mainloop_size) << ")" << std::endl;
  oss << "    A:   size=" << bytes2kb(sizeA) << ", offset=" << bytes2kb(offsetA) << std::endl;
  oss << "    B:   size=" << bytes2kb(sizeB) << ", offset=" << bytes2kb(offsetB) << std::endl;
  oss << "    Acc: size=" << bytes2kb(sizeAcc) << ", offset=" << bytes2kb(offsetAcc) << std::endl;
  oss << "    SFA: size=" << bytes2kb(sizeSFA) << ", offset=" << bytes2kb(offsetSFA) << std::endl;
  oss << "    SFB: size=" << bytes2kb(sizeSFB) << ", offset=" << bytes2kb(offsetSFB) << std::endl;
  oss << "- Epilogue (" << bytes2kb(epilogue_size) << ")" << std::endl;
  oss << "    C:   size=" << bytes2kb(sizeEpiC) << ", offset=" << bytes2kb(offsetEpiC) << std::endl;
  oss << "    D:   size=" << bytes2kb(sizeEpiD) << ", offset=" << bytes2kb(offsetEpiD) << std::endl;
  oss << "- CLC responses (" << bytes2kb(sizeCLC) << ", offset=" << bytes2kb(offsetCLC) << ")" << std::endl;
  oss << "- Pipelines (" << bytes2kb(pipeline_size) << ", separate abar space)" << std::endl;
  return oss.str();
}

#undef GET_MEM_INFO

template<typename Config>
bool run_gemm_blockscaled(sycl::queue& q)
{
  using ElementA          = typename Config::ElementA;
  using ElementB          = typename Config::ElementB;
  using ElementSFA        = typename Config::ElementSFA;
  using ElementSFB        = typename Config::ElementSFB;
  using ElementD          = typename Config::ElementD;
  using ElementAccumulator = typename Config::ElementAccumulator;
  using LayoutA           = typename Config::LayoutA;
  using LayoutB           = typename Config::LayoutB;
  using LayoutC           = cutlass::layout::RowMajor;
  using TileShape         = typename Config::CtaTileShape_MNK;
  using ClusterShape      = typename Config::ClusterShape_MNK;
  static constexpr int SFVecSizeA = Config::SFVecSizeA;
  static constexpr int SFVecSizeB = Config::SFVecSizeB;
  constexpr int AlignmentA = 512, AlignmentB = 512, AlignmentC = 512;

  static constexpr bool CoopSF = Config::EnableCooperativeSF;
  using ElementTupleA = cute::tuple<ElementA, ElementSFA, cute::Int<SFVecSizeA>, cute::Int<CoopSF ? 1 : 0>>;
  using ElementTupleB = cute::tuple<ElementB, ElementSFB, cute::Int<SFVecSizeB>, cute::Int<CoopSF ? 1 : 0>>;

  using ArchTag       = cutlass::arch::Xe4;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  // Identity epilogue: D = Acc, no source C.
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

  // --- Build Mainloop ---
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementTupleA, LayoutA, AlignmentA,
      ElementTupleB, LayoutB, AlignmentB,
      cute::tuple<ElementAccumulator, ElementD>,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<Config::PipelineStages>,
      cutlass::gemm::KernelTmaWarpSpecializedXe4<3, 1>
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

  std::cout << "Running on " << q.get_device().get_info<info::device::name>() << "\n";

  auto problem_shape_mnkl = typename GemmKernel::ProblemShape {};
  cute::fill_int_tuple_from(problem_shape_mnkl, Options::get_problem_shape<Config>());
  auto [mat_m, mat_n, mat_k, mat_l] = problem_shape_mnkl;

  // Runtime constraint checks
  constexpr int TileK = cute::size<2>(TileShape{});
  if (mat_l != 1) { std::cerr << "Error: batched GEMM (L=" << mat_l << ") not supported.\n"; return false; }
  if (mat_k % TileK    != 0) { std::cerr << "Error: K=" << mat_k << " not multiple of TileK=" << TileK << ".\n"; return false; }
  if (mat_k % SFVecSizeA != 0) { std::cerr << "Error: K not multiple of SFVecSizeA=" << SFVecSizeA << ".\n"; return false; }
  if (mat_k % SFVecSizeB != 0) { std::cerr << "Error: K not multiple of SFVecSizeB=" << SFVecSizeB << ".\n"; return false; }

  // Sizes
  uint32_t sizeA = size(select<0,2,3>(problem_shape_mnkl));
  uint32_t sizeB = size(select<1,2,3>(problem_shape_mnkl));
  uint32_t sizeD = size(select<0,1,3>(problem_shape_mnkl));
  int sf_kA = mat_k / SFVecSizeA, sf_kB = mat_k / SFVecSizeB;
  uint32_t sizeSFA = mat_m * sf_kA * mat_l;
  uint32_t sizeSFB = mat_n * sf_kB * mat_l;

  // Allocate and fill A/B (keep unpacked copies for validation)
  auto A_s   = sycl::malloc_shared<ElementA>(sizeA, q);
  auto B_s   = sycl::malloc_shared<ElementB>(sizeB, q);
  auto A_ref = sycl::malloc_shared<ElementA>(sizeA, q);
  auto B_ref = sycl::malloc_shared<ElementB>(sizeB, q);
  auto A_ten = make_tensor(make_gmem_ptr(A_s), make_layout(make_shape(mat_m, mat_k)));
  auto B_ten = make_tensor(make_gmem_ptr(B_s), make_layout(make_shape(mat_n, mat_k)));
  constexpr uint64_t seed_base = 42;
  random_fill_data(A_ten, seed_base + 2022);
  random_fill_data(B_ten, seed_base + 2021);
  std::memcpy(A_ref, A_s, sizeA * sizeof(ElementA));
  std::memcpy(B_ref, B_s, sizeB * sizeof(ElementB));
  if constexpr (cute::sizeof_bits_v<ElementA> < 8) subbyte_pack(A_ten);
  if constexpr (cute::sizeof_bits_v<ElementB> < 8) subbyte_pack(B_ten);

  // Scale factor allocation:
  // - MX operands (fp4/fp8): random values for actual block-scaling
  // - Plain operands (bf16/fp16): fill with 1.0 (hardware identity value in UE8M0)
  constexpr bool is_mx_a = std::is_same_v<ElementA, cutlass::float_e2m1_t> ||
                            std::is_same_v<ElementA, cutlass::float_e4m3_t>;
  constexpr bool is_mx_b = std::is_same_v<ElementB, cutlass::float_e2m1_t> ||
                            std::is_same_v<ElementB, cutlass::float_e4m3_t>;
  
  ElementSFA* SFA_s = sycl::malloc_shared<ElementSFA>(sizeSFA, q);
  auto SFA_ten = make_tensor(make_gmem_ptr(SFA_s), make_layout(make_shape(mat_m, sf_kA)));
  if constexpr (is_mx_a) {
    random_fill_sf(SFA_ten, seed_base + 2024);
  } else {
    for (int i = 0; i < size(SFA_ten); ++i) SFA_ten(i) = ElementSFA(1.0f);  // identity
  }
  
  ElementSFB* SFB_s = sycl::malloc_shared<ElementSFB>(sizeSFB, q);
  auto SFB_ten = make_tensor(make_gmem_ptr(SFB_s), make_layout(make_shape(mat_n, sf_kB)));
  if constexpr (is_mx_b) {
    random_fill_sf(SFB_ten, seed_base + 2025);
  } else {
    for (int i = 0; i < size(SFB_ten); ++i) SFB_ten(i) = ElementSFB(1.0f);  // identity
  }

  // Allocate output
  auto D_s = sycl::malloc_shared<ElementD>(sizeD, q);
  std::fill_n(D_s, sizeD, ElementD(0));

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, select<0,2,3>(problem_shape_mnkl));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, select<1,2,3>(problem_shape_mnkl));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, select<0,1,3>(problem_shape_mnkl));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, select<0,1,3>(problem_shape_mnkl));

  auto [cluster_size_x, cluster_size_y, cluster_size_z] = ClusterShape{};
  sycl::range<3> cluster_size(cluster_size_z, cluster_size_y, cluster_size_x);

  print("ProblemShape_MNKL: "); print(problem_shape_mnkl); print("\n");
  print("TileShape_MNK: ");     print(TileShape{});        print("\n");
  std::cout << get_shared_memory_info_blockscaled<GemmKernel>();
  std::cout << "Coop SF load: " << (Config::EnableCooperativeSF ? "enabled" : "disabled") << "\n";

  auto args = typename Gemm::GemmKernel::Arguments {
    problem_shape_mnkl,
    { A_s, stride_A, B_s, stride_B, SFA_s, SFB_s },
    { typename CollectiveEpilogue::FusionCallbacks::Arguments{}, nullptr, stride_C, D_s, stride_D }
  };

  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  GemmKernel kernel;
  auto params = kernel.to_underlying_arguments(args, {0, sm_count, 0}, nullptr);
  dim3 grid = GemmKernel::get_grid_shape(params);
  dim3 block = GemmKernel::get_block_shape();
  cutlass::SyclClusterLaunchParams launch_params = {
    range<3>(grid.z, grid.y, grid.x), range<3>(block.z, block.y, block.x), cluster_size, 0, q};
  cutlass::launch_kernel_on_cluster(launch_params, kernel, params).wait();

  // Validate
  constexpr auto layout_a = std::is_same_v<LayoutA, cutlass::layout::RowMajor> ? mem_layout::row_major : mem_layout::col_major;
  constexpr auto layout_b = std::is_same_v<LayoutB, cutlass::layout::RowMajor> ? mem_layout::row_major : mem_layout::col_major;
  for (int mat_i = 0; mat_i < mat_l; ++mat_i) {
    ElementSFA* sfa_batch = SFA_s + mat_i * mat_m * sf_kA;
    ElementSFB* sfb_batch = SFB_s + mat_i * mat_n * sf_kB;
    uint32_t err_cnt = validate_mxfp_gemm_result<ElementA, ElementB, ElementD, ElementSFA, ElementSFB, float>(
      A_ref + mat_i * mat_m * mat_k, B_ref + mat_i * mat_n * mat_k, D_s + mat_i * mat_m * mat_n,
      mat_m, mat_n, mat_k,
      is_mx_a, is_mx_b, sfa_batch, sfb_batch,
      layout_a, layout_b, false, tolerance<ElementD>{}, SFVecSizeA, SFVecSizeB);
    if (err_cnt > 0) {
      std::cerr << "FAILED at batch " << mat_i << ", errors: " << err_cnt << "\n";
      return false;
    }
  }
  std::cout << "Passed.\n";

  sycl::free(A_s, q); sycl::free(B_s, q); sycl::free(A_ref, q); sycl::free(B_ref, q);
  sycl::free(SFA_s, q);
  sycl::free(SFB_s, q);
  sycl::free(D_s, q);
  return true;
}

template <typename Config>
bool run_if_selected(const std::vector<std::string>& configs, sycl::queue& q) {
  if (configs.empty() && !Config::run_by_default && !Options::run_all) return true;
  if (!configs.empty() &&
      std::find(configs.begin(), configs.end(), Config::Name) == configs.end()) return true;
  std::cout << "\n=== A="    << type_name<typename Config::ElementA>()
            << "  sfA="      << sf_tag<typename Config::ElementSFA>()
            << "  blockA="   << Config::SFVecSizeA
            << "  |  B="     << type_name<typename Config::ElementB>()
            << "  sfB="      << sf_tag<typename Config::ElementSFB>()
            << "  blockB="   << Config::SFVecSizeB
            << "  |  D="     << type_name<typename Config::ElementD>()
            << " ===\n";
  // If --coop_sf was specified on the command line, override the config's default,
  // but only for cluster configs (cluster size > 1). Non-cluster configs ignore it.
  constexpr int cluster_size = cute::size(typename Config::ClusterShape_MNK{});
  if constexpr (cluster_size > 1) {
    if (Options::coop_sf.has_value()) {
      if (*Options::coop_sf) {
        if constexpr (std::is_same_v<typename Config::ElementA, cutlass::float_e4m3_t>) {
          std::cerr << "Warning: --coop_sf=1 is not supported for MXFP8 config '"
                    << Config::Name << "' (max atom K=256). Skipping.\n";
          return true;
        } else {
          using CoopConfig = ConfigWithCoopSF<Config, true>;
          return run_gemm_blockscaled<CoopConfig>(q);
        }
      } else {
        return run_gemm_blockscaled<ConfigWithCoopSF<Config, false>>(q);
      }
    }
  }
  return run_gemm_blockscaled<Config>(q);
}

template <typename... Configs>
void warn_unknown_configs(const std::vector<std::string>& user_configs) {
  if (user_configs.empty()) return;
  const std::vector<std::string> known = { Configs::Name... };
  for (const auto& c : user_configs)
    if (std::find(known.begin(), known.end(), c) == known.end())
      std::cerr << "Warning: unknown config '" << c << "' (run --help for available configs)\n";
}

template <typename... Configs>
bool run_configs(const std::vector<std::string>& user_configs, sycl::queue& q) {
  bool pass = (run_if_selected<Configs>(user_configs, q) & ...);
  warn_unknown_configs<Configs...>(user_configs);
  return pass;
}
