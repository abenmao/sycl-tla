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

// Runtime override for problem shape via command line: --m=<int> --n=<int> --k=<int> [--l=<int>]
// Dimensions not specified on the command line retain the per-config default.
// Select config via --config.
//
// Note: print_usage() is intentionally omitted — each executable defines its own
// to list program-specific configs and examples.
struct Options {
  static inline std::optional<int> m, n, k, l;
  static inline std::vector<std::string> configs;
  static inline bool help = false;

  static void parse(int argc, char** argv) {
    cutlass::CommandLine cmd(argc, const_cast<char const**>(argv));
    help = cmd.check_cmd_line_flag("help");
    int val;
    if (cmd.check_cmd_line_flag("m")) { cmd.get_cmd_line_argument("m", val); m = val; }
    if (cmd.check_cmd_line_flag("n")) { cmd.get_cmd_line_argument("n", val); n = val; }
    if (cmd.check_cmd_line_flag("k")) { cmd.get_cmd_line_argument("k", val); k = val; }
    if (cmd.check_cmd_line_flag("l")) { cmd.get_cmd_line_argument("l", val); l = val; }
    std::string config_str;
    cmd.get_cmd_line_argument("config", config_str);
    if (!config_str.empty()) {
      std::istringstream iss(config_str);
      std::string token;
      while (std::getline(iss, token, ',')) {
        if (!token.empty()) configs.push_back(token);
      }
    }
  }

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
  // Block-scaled data types: sub-byte or byte data + scale factors
  using ElementA  = typename Config::ElementA;
  using ElementB  = typename Config::ElementB;
  using ElementSF = typename Config::ElementSF;
  static constexpr int SFVecSize = Config::SFVecSize;

  // ElementTupleA/B bundle data + SF types for CollectiveBuilder.
  using ElementTupleA = cute::tuple<ElementA, ElementSF, cute::Int<SFVecSize>>;
  using ElementTupleB = cute::tuple<ElementB, ElementSF, cute::Int<SFVecSize>>;

  // Layouts
  using LayoutA = typename Config::LayoutA;
  using LayoutB = typename Config::LayoutB;
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 512;
  constexpr int AlignmentB = 512;
  constexpr int AlignmentC = 512;

  // Accumulator / output types
  using ElementD = typename Config::ElementD;
  using ElementAccumulator = typename Config::ElementAccumulator;

  // Kernel shape config
  using ArchTag      = cutlass::arch::Xe4;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using TileShape    = typename Config::CtaTileShape_MNK;
  using ClusterShape = typename Config::ClusterShape_MNK;

  // --- Build Epilogue (reuses existing Xe4 epilogue collective, not added by this PR) ---
  // Identity epilogue: D = Acc, no fusion, void = no source C.
  // TODO: Test with a real epilogue (e.g., bias, activation) to validate end-to-end fusion.
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
      cutlass::gemm::KernelTmaWarpSpecializedXe4<3, 1>  // required by template; builder hardcodes these internally
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::StaticPersistentScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Warp counts — fixed by shared infrastructure (kernel + epilogue builders)
  // NumEpilogueWarps: set by Xe4 epilogue builder (xe4_builder.inl), controls SMEM->GMEM store throughput.
  // NumControlWarps:  set by Xe4 kernel (xe4_gemm_dma_warpspecialized.hpp), one per role:
  //                   MMA(0), Sched(1), MainloopLoad(2), EpilogueLoad(3).
  static constexpr int NumTotalWarps    = GemmKernel::MaxThreadsPerBlock / cutlass::NumThreadsPerWarp;
  static constexpr int NumEpilogueWarps = CollectiveEpilogue::DispatchPolicy::NumEpilogueWarps;
  static constexpr int NumControlWarps  = NumTotalWarps - NumEpilogueWarps;

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /// Setup and evaluation
  /////////////////////////////////////////////////////////////////////////////////////////////////

  auto dev = q.get_device();
  std::cout << "Running block-scaled GEMM on " << dev.get_info<info::device::name>() << "\n";

  auto problem_shape_mnkl = typename GemmKernel::ProblemShape {};
  cute::fill_int_tuple_from(problem_shape_mnkl, Options::get_problem_shape<Config>());

  auto [mat_m, mat_n, mat_k, mat_l] = problem_shape_mnkl;

  // Validation of unsupported configurations passed via cmd
  if (mat_l != 1) {
    std::cerr << "Error: batched GEMM (L=" << mat_l << ") is not supported by block-scaled collectives.\n";
    return false;
  }
  constexpr int TileK = cute::size<2>(TileShape{});
  if (mat_k % TileK != 0) {
    std::cerr << "Error: K=" << mat_k << " is not a multiple of TileK=" << TileK << ".\n";
    return false;
  }
  if (mat_k % SFVecSize != 0) {
    std::cerr << "Error: K=" << mat_k << " is not a multiple of SFVecSize=" << SFVecSize << ".\n";
    return false;
  }

  // Data tensor sizes (element counts)
  uint32_t sizeA = size(select<0,2,3>(problem_shape_mnkl));
  uint32_t sizeB = size(select<1,2,3>(problem_shape_mnkl));

  // SF tensor sizes: one SF per SFVecSize K-elements — MN-major layout
  // SFA: (M, K/SFVecSize, L),  SFB: (N, K/SFVecSize, L)
  int sf_k = mat_k / SFVecSize;
  uint32_t sizeSFA = mat_m * sf_k * mat_l;
  uint32_t sizeSFB = mat_n * sf_k * mat_l;

  // --- Allocate data tensors ---
  // For sub-byte types (e.g. FP4), malloc_shared allocates 1 byte per element;
  // after filling, subbyte_pack packs two values per byte.
  auto A_s = sycl::malloc_shared<ElementA>(sizeA, q);
  auto B_s = sycl::malloc_shared<ElementB>(sizeB, q);

  auto A_ten = make_tensor(make_gmem_ptr(A_s), make_layout(make_shape(mat_m, mat_k)));
  auto B_ten = make_tensor(make_gmem_ptr(B_s), make_layout(make_shape(mat_n, mat_k)));

  constexpr uint64_t seed_base = 42;
  random_fill_data(A_ten, seed_base + 2022);
  random_fill_data(B_ten, seed_base + 2021);

  // Keep copies for host-side validation (before sub-byte packing, if applicable)
  auto A_ref = sycl::malloc_shared<ElementA>(sizeA, q);
  auto B_ref = sycl::malloc_shared<ElementB>(sizeB, q);
  std::memcpy(A_ref, A_s, sizeA * sizeof(ElementA));
  std::memcpy(B_ref, B_s, sizeB * sizeof(ElementB));

  // Pack sub-byte elements (e.g., two 4-bit values per byte for FP4)
  if constexpr (cute::sizeof_bits_v<ElementA> < 8) {
    subbyte_pack(A_ten);
    subbyte_pack(B_ten);
  }

  // --- Allocate scale factor tensors ---
  auto SFA_s = sycl::malloc_shared<ElementSF>(sizeSFA, q);
  auto SFB_s = sycl::malloc_shared<ElementSF>(sizeSFB, q);

  auto SFA_ten = make_tensor(make_gmem_ptr(SFA_s), make_layout(make_shape(mat_m, sf_k)));
  auto SFB_ten = make_tensor(make_gmem_ptr(SFB_s), make_layout(make_shape(mat_n, sf_k)));
  random_fill_sf(SFA_ten, seed_base + 2024);
  random_fill_sf(SFB_ten, seed_base + 2025);

  // --- Allocate output tensor ---
  uint32_t sizeD = size(select<0,1,3>(problem_shape_mnkl));
  auto D_s = sycl::malloc_shared<ElementD>(sizeD, q);
  std::fill_n(D_s, sizeD, ElementD(0));

  // --- Strides ---
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, select<0,2,3>(problem_shape_mnkl));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, select<1,2,3>(problem_shape_mnkl));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, select<0,1,3>(problem_shape_mnkl));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, select<0,1,3>(problem_shape_mnkl));

  // --- Cluster / grid setup ---
  auto [cluster_size_y, cluster_size_x, cluster_size_z] = ClusterShape{};
  sycl::range<3> cluster_size(cluster_size_z, cluster_size_y, cluster_size_x);

  // --- Print info ---
  print("ProblemShape_MNKL: "); print(problem_shape_mnkl); print("\n");
  print("TileShape_MNK: ");     print(TileShape{});        print("\n");
  std::cout << "SFVecSize: " << SFVecSize << "\n";

  auto smem_info = get_shared_memory_info_blockscaled<GemmKernel>();
  std::cout << smem_info << std::endl;

  // --- Assemble kernel arguments ---
  auto args = typename Gemm::GemmKernel::Arguments {
    problem_shape_mnkl,
    { A_s, stride_A, B_s, stride_B, SFA_s, SFB_s },
    { typename CollectiveEpilogue::FusionCallbacks::Arguments{}, nullptr, stride_C, D_s, stride_D }
  };

  // --- Convert to device params and launch ---
  int device_id = 0;
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  cutlass::KernelHardwareInfo kernel_hw_info{device_id, sm_count, 0};

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

  // --- Validate against block-scaled reference ---
  constexpr auto layout_a = std::is_same_v<LayoutA, cutlass::layout::RowMajor>
      ? mem_layout::row_major : mem_layout::col_major;
  constexpr auto layout_b = std::is_same_v<LayoutB, cutlass::layout::RowMajor>
      ? mem_layout::row_major : mem_layout::col_major;

  for (int mat_i = 0; mat_i < mat_l; ++mat_i) {
    ElementA*  ptr_A   = A_ref + mat_i * mat_m * mat_k;
    ElementB*  ptr_B   = B_ref + mat_i * mat_n * mat_k;
    ElementD*  ptr_D   = D_s   + mat_i * mat_m * mat_n;
    ElementSF* ptr_SFA = SFA_s + mat_i * mat_m * sf_k;
    ElementSF* ptr_SFB = SFB_s + mat_i * mat_n * sf_k;

    uint32_t err_cnt = validate_mxfp_gemm_result<ElementA, ElementB, ElementD, ElementSF, float>(
      ptr_A, ptr_B, ptr_D,
      mat_m, mat_n, mat_k,
      true, true,                   // a_scaling, b_scaling
      ptr_SFA, ptr_SFB,
      layout_a,
      layout_b,
      false,                        // negative_axb
      tolerance<ElementD>{},
      SFVecSize);

    if (err_cnt > 0) {
      std::cerr << "Test FAILED at batch " << mat_i << ", error count: " << err_cnt << std::endl;
      return false;
    }
  }

  std::cout << "Ran successfully." << std::endl;

  // --- Cleanup ---
  sycl::free(A_s, q);
  sycl::free(B_s, q);
  sycl::free(A_ref, q);
  sycl::free(B_ref, q);
  sycl::free(SFA_s, q);
  sycl::free(SFB_s, q);
  sycl::free(D_s, q);

  return true;
}

template <typename Config>
bool run_if_selected(const std::vector<std::string>& configs, sycl::queue& q) {
  if (!configs.empty() &&
      std::find(configs.begin(), configs.end(), Config::Name) == configs.end())
    return true;  // skipped
  std::cout << "\n=== Running config: " << Config::Name << " ===\n";
  return run_gemm_blockscaled<Config>(q);
}

// Warn about any --config names that don't match a known Config type.
template <typename... Configs>
void warn_unknown_configs(const std::vector<std::string>& user_configs) {
  if (user_configs.empty()) return;
  const std::vector<std::string> known = { Configs::Name... };
  for (const auto& c : user_configs)
    if (std::find(known.begin(), known.end(), c) == known.end())
      std::cerr << "Warning: unknown config '" << c << "' (run --help for available configs)\n";
}

// Run all configs, then warn on unknown names. Returns false if any config failed.
// Usage: return run_configs<C1, C2, ...>(Options::configs, q) ? 0 : 1;
template <typename... Configs>
bool run_configs(const std::vector<std::string>& user_configs, sycl::queue& q) {
  bool pass = (run_if_selected<Configs>(user_configs, q) & ...);
  warn_unknown_configs<Configs...>(user_configs);
  return pass;
}
