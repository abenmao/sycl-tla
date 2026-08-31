/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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
/*! \file
    \brief Bare CuTe implementation of FP4 (e2m1) block-scaled GEMM on Intel Xe35.

    This example extracts the inlined kernel from xe_gemm.hpp and the
    collective mainloop from xe_mma_blockscaled_native.hpp into a single
    self-contained SYCL kernel, with no CUTLASS device/kernel/collective
    framework layers.  Host-side init/verify reuses the same patterns as
    example 50.

    To build & run (from your build dir):

      $ ninja 53_xe35_cute_persistent_prefetch_pipeline
      $ ./examples/53_xe35_cute_persistent_prefetch_pipeline/53_xe35_cute_persistent_prefetch_pipeline

    Call with `--help` for information about available options
*/

// --- CUTLASS host utilities (init, verify, command line) ---
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/xe_mma_blockscaled_scale_traits.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/initialize_block.hpp"
#include "cutlass/util/reference/device/sycl_tensor_fill.h"
#include "cutlass/util/sycl_event_manager.hpp"

// --- CuTe core ---
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_zip.hpp"

// --- SYCL runtime & GRF property ---
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

// --- common helpers ---
#include "../common/sycl_common.hpp"
#include "../common/helper.h"

#include <iostream>
#include <vector>

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// Command line options (same as example 50)
///////////////////////////////////////////////////////////////////////////////
struct Options {
  bool help = false;
  int m = 4096, n = 4096, k = 4096, l = 1, iterations = 20;
  int const_scale = 0, verify = 1;
  float alpha = 1.f, beta = 0.f;

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) { help = true; return; }
    cmd.get_cmd_line_argument("m", m, 4096);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);
    cmd.get_cmd_line_argument("const_scale", const_scale, 0);
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "Bare CuTe Block-Scaled GEMM Example (Xe35)\n\n"
        << "Options:\n"
        << "  --help               Display this message\n"
        << "  --m=<int>            M extent (default 4096)\n"
        << "  --n=<int>            N extent (default 4096)\n"
        << "  --k=<int>            K extent (default 4096)\n"
        << "  --l=<int>            Batch count (default 1)\n"
        << "  --alpha=<float>      Epilogue alpha (default 1.0)\n"
        << "  --beta=<float>       Epilogue beta (default 0.0)\n"
        << "  --iterations=<int>   Timing iterations (default 100)\n"
        << "  --verify=<int>       Verify results (default 1)\n"
        << "  --const_scale=<int>  Use constant scale (default 0)\n";
    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Type definitions — matches example 50 e2m1 configuration
///////////////////////////////////////////////////////////////////////////////

using ElementA     = float_e2m1_t;
using ElementB     = float_e2m1_t;
using ElementScale = cutlass::float_ue8m0_t;
using ElementAccumulator = float;
using ElementOutput = bfloat16_t;
using ElementC      = bfloat16_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using StrideA     = cutlass::gemm::TagToStrideA_t<LayoutA>;
using StrideB     = cutlass::gemm::TagToStrideB_t<LayoutB>;
using StrideC     = cutlass::gemm::TagToStrideC_t<LayoutC>;
using StrideD     = cutlass::gemm::TagToStrideC_t<LayoutD>;
using StrideScale = cute::Stride<_1, int64_t, int64_t>;

// Tile shape and thread layout
using TileShape    = Shape<_2048, _256, _128>;
using ThreadLayout = Layout<Shape<_32, _4, _1>, Stride<_4, _1, _0>>;

// MMA atom — XE_BDPAS_TT for block-scaled DPAS
using MmaOp    = XE_BDPAS_TT<8, float, ElementA>;
using TiledMma = typename TiledMMAHelper<MMA_Atom<MmaOp>, Layout<TileShape>, ThreadLayout>::TiledMMA;

// Dispatch policy
using DispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16BlockScaled<2, cute::Int<32>>;

// Derived constants
static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

static constexpr int BLK_M = get<0>(TileShape{});
static constexpr int BLK_N = get<1>(TileShape{});
static constexpr int BLK_K = get<2>(TileShape{});

static constexpr int SG_NUMS_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
static constexpr int SG_NUMS_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
static constexpr int MMA_K     = get<2>(typename TiledMma::Shape_MNK{});

static constexpr int SG_M = ceil_div(BLK_M, SG_NUMS_M);
static constexpr int SG_N = ceil_div(BLK_N, SG_NUMS_N);
static constexpr int SG_K = BLK_K; // SG_NUMS_K = 1

static constexpr int      GroupK             = 32;
static constexpr uint32_t MaxThreadsPerBlock = cute::min(512, size(TiledMma{}));

using GmemTiledCopyA = void;
using GmemTiledCopyB = void;
using GmemTiledCopyScaleA = void;
using GmemTiledCopyScaleB = void;

// Tensor type helper
template<class Element, class Stride>
using TensorType = decltype(make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                                        make_layout(make_shape(int{}, int{}, int{}), Stride{})));

///////////////////////////////////////////////////////////////////////////////
// Kernel parameters — flat struct replacing GemmKernel::Params
///////////////////////////////////////////////////////////////////////////////
struct KernelParams {
  int M, N, K, L;
  float alpha, beta;

  // Data tensors
  TensorType<ElementA, StrideA> mA_mkl;
  TensorType<ElementB, StrideB> mB_nkl;

  // Scale tensors
  TensorType<ElementScale, StrideScale> mAscale;
  TensorType<ElementScale, StrideScale> mBscale;

  // Output tensors
  decltype(make_tensor(make_gmem_ptr(static_cast<ElementC const*>(nullptr)),
                       make_layout(make_shape(int{}, int{}, int{}), StrideC{}))) mC;
  decltype(make_tensor(make_gmem_ptr(static_cast<ElementOutput*>(nullptr)),
                       make_layout(make_shape(int{}, int{}, int{}), StrideD{}))) mD;

  bool raster_along_n;
};

///////////////////////////////////////////////////////////////////////////////
// Device kernel — extracted from xe_gemm.hpp operator()
//
// This is the core of the example: the entire GEMM kernel implemented
// directly with CuTe primitives, with no CUTLASS collective/kernel/device
// framework layers.
///////////////////////////////////////////////////////////////////////////////
template <class Params>
CUTLASS_DEVICE void
bare_cute_blockscaled_gemm_kernel(Params const& params)
{
  // --- Tile coordinates ---
  int m_coord, n_coord, l_coord;
  if (params.raster_along_n) {
    m_coord = BlockIdxY();
    n_coord = BlockIdxX();
    l_coord = BlockIdxZ();
  } else {
    m_coord = BlockIdxX();
    n_coord = BlockIdxY();
    l_coord = BlockIdxZ();
  }

  auto M = params.M;
  auto N = params.N;
  auto K = params.K;

  constexpr auto workgroup_shape = TileShape{};
  constexpr auto blk_shape       = TileShape{};

  // Identity tensors for coordinate mapping
  Tensor cA = make_identity_tensor(make_shape(M, K, params.L));
  Tensor cB = make_identity_tensor(make_shape(N, K, params.L));

  Tensor gA = local_tile(cA, select<0,2>(blk_shape), make_coord(m_coord, _, l_coord));
  Tensor gB = local_tile(cB, select<1,2>(blk_shape), make_coord(n_coord, _, l_coord));

  // Allocate accumulator
  TiledMma tiled_mma;
  Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));

  auto k_tile_iter = cute::make_coord_iterator(idx2crd(0, make_shape(K)), make_shape(K));
  int k_tile_count = ceil_div(K, get<2>(workgroup_shape));

  // --- Block 2D copy objects (thread-invariant) ---
  auto batch_idx = l_coord;
  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(TiledMma{}, params.mA_mkl(_, _, batch_idx));
  auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(TiledMma{}, params.mB_nkl(_, _, batch_idx));

  // Prefetch objects
  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K));
  // How many K tiles fit in one scale group (scale reloads every k_reload_factor tiles)
  constexpr int k_reload_factor = cute::max(GroupK / BLK_K, 1);

  // === Initial staging prefetch (pre-fill pipeline before entering the MN loop) ===
  {
    int thread_idx = int(ThreadIdxX());
    auto thr_prefetch_A = prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(thread_idx);
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    const int ml_m_coord = m_coord * BLK_M + (cutlass::get_sub_group_id() / SG_NUMS_N) * SG_M;
    const int ml_n_coord = n_coord * BLK_N + (cutlass::get_sub_group_id() % SG_NUMS_N) * SG_N;

    auto [tiled_copy_scaleA, copy_iter_scaleA_, fragment_scaleA_] =
        cutlass::gemm::collective::make_scaled_copy<GmemTiledCopyScaleA, ElementScale,
                                              SG_M, SG_K, GroupK>(params.mAscale, ml_m_coord, l_coord, k_tile_count);
    auto [tiled_copy_scaleB, copy_iter_scaleB_, fragment_scaleB_] =
        cutlass::gemm::collective::make_scaled_copy<GmemTiledCopyScaleB, ElementScale,
                                              SG_N, SG_K, GroupK>(params.mBscale, ml_n_coord, l_coord, k_tile_count);
    (void)copy_iter_scaleA_; (void)fragment_scaleA_;
    (void)copy_iter_scaleB_; (void)fragment_scaleB_;

    auto [tiled_prefetch_scaleA, prefetch_iter_scaleA] =
        cutlass::gemm::collective::make_scaled_prefetch<decltype(tiled_copy_scaleA),
                                                         SG_M, SG_K, GroupK>(tiled_copy_scaleA, ml_m_coord, l_coord, k_tile_count);
    auto [tiled_prefetch_scaleB, prefetch_iter_scaleB] =
        cutlass::gemm::collective::make_scaled_prefetch<decltype(tiled_copy_scaleB),
                                                         SG_N, SG_K, GroupK>(tiled_copy_scaleB, ml_n_coord, l_coord, k_tile_count);

    auto prepared_pa = prepare_payloads(prefetch_a, pAgA(_,_,_,0));
    auto prepared_pb = prepare_payloads(prefetch_b, pBgB(_,_,_,0));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prepared_pa += SG_K, prepared_pb += SG_K) {
      prefetch(prefetch_a, prepared_pa);
      prefetch(prefetch_b, prepared_pb);

      prefetch(tiled_prefetch_scaleA, prefetch_iter_scaleA(_, _, _, i / k_reload_factor));
      prefetch(tiled_prefetch_scaleB, prefetch_iter_scaleB(_, _, _, i / k_reload_factor));
    }
  }

  // === M-tile loop: each workgroup processes MN consecutive M-tiles ===
  static constexpr auto MN = 4;

  for (int m = 0; m < MN; m++) {
    // Virtual thread index maps each M-tile iteration to a different subgroup slice
    int thread_idx = int(ThreadIdxX()) + m * MaxThreadsPerBlock;

    // Per-thread copy/MMA partitions
    auto thr_copy_a = copy_a.get_slice(thread_idx);
    auto thr_copy_b = copy_b.get_slice(thread_idx);
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    // Register fragments for MMA
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    // Register fragments for copies
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

    // Partition global tensor (proxies) for copies
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    // Prefetch partitions
    auto thr_prefetch_A = prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(thread_idx);
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);


    auto next_thr_prefetch_A = prefetch_a.get_slice(thread_idx + MaxThreadsPerBlock);
    auto next_thr_prefetch_B = prefetch_b.get_slice(thread_idx + MaxThreadsPerBlock);
    auto next_pAgA = next_thr_prefetch_A.partition_S(gA);
    auto next_pBgB = next_thr_prefetch_B.partition_S(gB);

    auto prepared_a  = prepare_payloads(copy_a, tAgA(_,_,_,0));
    auto prepared_b  = prepare_payloads(copy_b, tBgB(_,_,_,0));

    using GemmIterM = Int<decltype(size<1>(tCrA.shape()))::value>;
    using GemmIterN = Int<decltype(size<1>(tCrB.shape()))::value>;
    using GemmIterK = Int<decltype(size<2>(tCrB.shape()))::value>;

    auto m_i = thread_idx - int(ThreadIdxX());

    const int ml_m_coord = m_coord * BLK_M + m_i + (cutlass::get_sub_group_id() / SG_NUMS_N) * SG_M;
    const int ml_n_coord = n_coord * BLK_N + (cutlass::get_sub_group_id() % SG_NUMS_N) * SG_N;

    // Scale copy setup
    auto [tiled_copy_scaleA, copy_iter_scaleA, fragment_scaleA] =
        cutlass::gemm::collective::make_scaled_copy<GmemTiledCopyScaleA, ElementScale,
                                              SG_M, SG_K, GroupK>(params.mAscale, ml_m_coord, l_coord, k_tile_count);
    auto [tiled_copy_scaleB, copy_iter_scaleB, fragment_scaleB] =
        cutlass::gemm::collective::make_scaled_copy<GmemTiledCopyScaleB, ElementScale,
                                              SG_N, SG_K, GroupK>(params.mBscale, ml_n_coord, l_coord, k_tile_count);
    auto [scale_m_offsets, scale_n_offsets, scale_ak_offsets, scale_bk_offsets] =
        cutlass::gemm::collective::make_scaled_offsets<
                                                GemmIterM::value, GemmIterN::value, GemmIterK::value, MMA_K, GroupK,
                                                typename decltype(tiled_copy_scaleA)::BlockShape,
                                                typename decltype(tiled_copy_scaleB)::BlockShape>();
    auto [tiled_prefetch_scaleA, prefetch_iter_scaleA] =
        cutlass::gemm::collective::make_scaled_prefetch<decltype(tiled_copy_scaleA),
                                                         SG_M, SG_K, GroupK>(tiled_copy_scaleA, ml_m_coord, l_coord, k_tile_count);
    auto [tiled_prefetch_scaleB, prefetch_iter_scaleB] =
        cutlass::gemm::collective::make_scaled_prefetch<decltype(tiled_copy_scaleB),
                                                         SG_N, SG_K, GroupK>(tiled_copy_scaleB, ml_n_coord, l_coord, k_tile_count);

    using scaleA_vec_t = intel::vector_t<ElementScale, decltype(size(fragment_scaleA))::value>;
    using scaleB_vec_t = intel::vector_t<ElementScale, decltype(size(fragment_scaleB))::value>;

    auto prepared_pa = prepare_payloads(prefetch_a, pAgA(_,_,_,DispatchPolicy::Stages));
    auto prepared_pb = prepare_payloads(prefetch_b, pBgB(_,_,_,DispatchPolicy::Stages));

    auto next_prepared_pa = prepare_payloads(prefetch_a, next_pAgA(_,_,_,0));
    auto next_prepared_pb = prepare_payloads(prefetch_b, next_pBgB(_,_,_,0));


    int next_m_i = (m + 1) * MaxThreadsPerBlock;
    const int next_ml_m_coord = m_coord * BLK_M + next_m_i + (cutlass::get_sub_group_id() / SG_NUMS_N) * SG_M;
    const int next_ml_n_coord = n_coord * BLK_N + (cutlass::get_sub_group_id() % SG_NUMS_N) * SG_N;
    auto [next_tiled_prefetch_scaleA, next_prefetch_iter_scaleA] =
        cutlass::gemm::collective::make_scaled_prefetch<decltype(tiled_copy_scaleA),
                                                         SG_M, SG_K, GroupK>(tiled_copy_scaleA, next_ml_m_coord, l_coord, k_tile_count);
    auto [next_tiled_prefetch_scaleB, next_prefetch_iter_scaleB] =
        cutlass::gemm::collective::make_scaled_prefetch<decltype(tiled_copy_scaleB),
                                                         SG_N, SG_K, GroupK>(tiled_copy_scaleB, next_ml_n_coord, l_coord, k_tile_count);

    auto prefetch_k = DispatchPolicy::Stages;

    // =========== First K tile (with clear via gemm<true>) ===========
    {
      copy(copy_a, prepared_a, tArA);
      copy(copy_b, prepared_b, tBrB);

      copy(tiled_copy_scaleA, copy_iter_scaleA(_, _, _, 0), fragment_scaleA);
      copy(tiled_copy_scaleB, copy_iter_scaleB(_, _, _, 0), fragment_scaleB);

      prefetch(prefetch_a, prepared_pa);
      prefetch(prefetch_b, prepared_pb);

      prefetch(tiled_prefetch_scaleA, prefetch_iter_scaleA(_, _, _, prefetch_k / k_reload_factor));
      prefetch(tiled_prefetch_scaleB, prefetch_iter_scaleB(_, _, _, prefetch_k / k_reload_factor));

      // Reorder from load layout to MMA operand layout
      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);

      // Broadcast scale factors across MMA iterations
      Tensor scaleA = make_tensor(recast<scaleA_vec_t>(fragment_scaleA).data(),
                                  make_layout(Shape<_1, GemmIterM, _1>{}, Stride<_1, _0, _0>{}));
      Tensor scaleB = make_tensor(recast<scaleB_vec_t>(fragment_scaleB).data(),
                                  make_layout(Shape<_1, GemmIterN, _1>{}, Stride<_1, _0, _0>{}));

      // Software fence: ensure all loads above are visible before issuing DPAS
      asm volatile("fence_sw" : : : );

      // First K-chunk iteration: gemm<true> zero-initializes the accumulator
      cute::gemm<true>(tiled_mma,
                 make_zip_tensor(tCrA(_, _, 0), scaleA(_, _, 0), scale_m_offsets(_, _, 0), scale_ak_offsets(_, _, 0)),
                 make_zip_tensor(tCrB(_, _, 0), scaleB(_, _, 0), scale_n_offsets(_, _, 0), scale_bk_offsets(_, _, 0)),
                 accumulators);

      CUTE_UNROLL
      for (int i = 1; i < decltype(size<2>(tCrA.tensor()))::value; i++) {
        cute::gemm(tiled_mma,
                   make_zip_tensor(tCrA(_, _, i), scaleA(_, _, i), scale_m_offsets(_, _, i), scale_ak_offsets(_, _, i)),
                   make_zip_tensor(tCrB(_, _, i), scaleB(_, _, i), scale_n_offsets(_, _, i), scale_bk_offsets(_, _, i)),
                   accumulators);
      }

      prepared_a += SG_K;
      prepared_b += SG_K;

      prepared_pa += SG_K;
      prepared_pb += SG_K;

      prefetch_k++;
    }

    // =========== Main K loop (software-pipelined load + compute) ===========
    for (int k_tile = 1; k_tile < k_tile_count + k_start_idx; k_tile++, prefetch_k++, prepared_a += SG_K, prepared_b += SG_K) {
      copy(copy_a, prepared_a, tArA);
      copy(copy_b, prepared_b, tBrB);

      copy(tiled_copy_scaleA, copy_iter_scaleA(_, _, _, k_tile / k_reload_factor), fragment_scaleA);
      copy(tiled_copy_scaleB, copy_iter_scaleB(_, _, _, k_tile / k_reload_factor), fragment_scaleB);

      // --- Prefetch: current M-tile's future K tiles, or next M-tile's first K tiles ---
      if (prefetch_k < k_tile_count + k_start_idx) {
        prefetch(prefetch_a, prepared_pa);
        prefetch(prefetch_b, prepared_pb);
        prepared_pa += SG_K;
        prepared_pb += SG_K;

        prefetch(tiled_prefetch_scaleA, prefetch_iter_scaleA(_, _, _, prefetch_k / k_reload_factor));
        prefetch(tiled_prefetch_scaleB, prefetch_iter_scaleB(_, _, _, prefetch_k / k_reload_factor));
      } else if ((m + 1) < MN) {
        int next_k = prefetch_k - (k_tile_count + k_start_idx);

        prefetch(prefetch_a, next_prepared_pa);
        prefetch(prefetch_b, next_prepared_pb);
        next_prepared_pa += SG_K;
        next_prepared_pb += SG_K;

        prefetch(next_tiled_prefetch_scaleA, next_prefetch_iter_scaleA(_, _, _, next_k / k_reload_factor));
        prefetch(next_tiled_prefetch_scaleB, next_prefetch_iter_scaleB(_, _, _, next_k / k_reload_factor));
      }

      // --- Reorder from load layout to MMA operand layout ---
      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);

      // Broadcast scale factors across MMA iterations
      Tensor scaleA = make_tensor(recast<scaleA_vec_t>(fragment_scaleA).data(),
                                  make_layout(Shape<_1, GemmIterM, _1>{}, Stride<_1, _0, _0>{}));
      Tensor scaleB = make_tensor(recast<scaleB_vec_t>(fragment_scaleB).data(),
                                  make_layout(Shape<_1, GemmIterN, _1>{}, Stride<_1, _0, _0>{}));

      cute::gemm(tiled_mma,
                 make_zip_tensor(tCrA, scaleA, scale_m_offsets, scale_ak_offsets),
                 make_zip_tensor(tCrB, scaleB, scale_n_offsets, scale_bk_offsets),
                 accumulators);
    }

    // =========== Epilogue: store accumulator directly to D (identity epilogue) ===========
    {
      // Reorder accumulator from MMA register layout to memory-store layout, then write.
      auto copy_d = make_block_2d_copy_D(tiled_mma, params.mD(_, _, l_coord));

      Tensor cD = make_identity_tensor(make_shape(M, N, params.L));
      Tensor gD_tile = local_tile(cD, select<0,1>(blk_shape), make_coord(m_coord, n_coord, l_coord));

      auto thr_copy_d = copy_d.get_slice(thread_idx);
      auto tDgD = thr_copy_d.partition_D(gD_tile);       // global memory destination
      auto tDrD = thr_copy_d.partition_sg_fragment_S(gD_tile); // register fragment in store layout

      // Partition in MMA accumulator layout (matches accumulators tensor)
      auto tDrD_mma = thr_mma.partition_sg_fragment_C(gD_tile);

      // Reorder accumulators from MMA layout to store layout, then issue block 2D store
      reorder(make_subgroup_tensor(accumulators, tDrD_mma.tv_layout()), tDrD);
      copy(copy_d, tDrD, tDgD);
    }
  } // end MN loop
}

///////////////////////////////////////////////////////////////////////////////
// Host-side runner
///////////////////////////////////////////////////////////////////////////////
struct Runner {
  static constexpr int scaleGroupSize = GroupK;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  StrideScale stride_SA;
  StrideScale stride_SB;

  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA>     block_A;
  cutlass::DeviceAllocation<ElementB>     block_B;
  cutlass::DeviceAllocation<float>        block_A_dq;
  cutlass::DeviceAllocation<float>        block_B_dq;
  cutlass::DeviceAllocation<ElementC>     block_C;
  cutlass::DeviceAllocation<ElementScale> block_scaleA;
  cutlass::DeviceAllocation<ElementScale> block_scaleB;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  template <class Element>
  bool initialize_scale(cutlass::DeviceAllocation<Element>& block, Options const& options) {
    const float elt_max_f = float(cutlass::platform::numeric_limits<Element>::max());
    const float max_dequant_val = elt_max_f * 0.25f;
    const float min_dequant_val = 0.5f;
    const float scale_max = options.const_scale ? 1.0f : max_dequant_val / elt_max_f;
    const float scale_min = options.const_scale ? 1.0f : min_dequant_val / elt_max_f;
    cutlass::reference::device::BlockFillRandomUniformCopyFromHost(
        block.get(), block.size(), seed, Element(scale_max), Element(scale_min));
    return true;
  }

  // Host-side dequantize A: dst[m,k] = src[m,k] * scale[m, k/group_k]
  template <class DstElement, class SrcElement, class Layout, class ScaleElement, class ScaleLayout>
  static void apply_scale(DstElement* dq_buffer, SrcElement const* q_buffer, Layout const operand_layout,
                          ScaleElement const* scale_buffer, ScaleLayout const scale_layout,
                          Options const& options, int group_size) {
    std::vector<uint8_t> dst(size(operand_layout) * sizeof_bits_v<DstElement> / 8, 0);
    cutlass::device_memory::copy_to_host(dst.data(), (uint8_t*)dq_buffer, dst.size());
    std::vector<uint8_t> src(size(operand_layout) * sizeof_bits_v<SrcElement> / 8, 0);
    cutlass::device_memory::copy_to_host(src.data(), (uint8_t*)q_buffer, src.size());
    std::vector<uint8_t> scale(size(scale_layout) * sizeof_bits_v<ScaleElement> / 8, 0);
    cutlass::device_memory::copy_to_host(scale.data(), (uint8_t*)scale_buffer, scale.size());
    compat::wait();

    auto dst_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<DstElement*>(dst.data())), operand_layout);
    auto src_tensor = [&]() {
      if constexpr (sizeof_bits_v<SrcElement> < 8)
        return make_tensor(cute::subbyte_iterator<const SrcElement>(src.data()), operand_layout);
      else
        return make_tensor(make_gmem_ptr(reinterpret_cast<SrcElement const*>(src.data())), operand_layout);
    }();
    auto scale_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<ScaleElement const*>(scale.data())), scale_layout);

    auto MN = size<0>(src_tensor); auto K = size<1>(src_tensor); auto L = size<2>(src_tensor);
    for (int l = 0; l < L; l++)
      for (int k = 0; k < K; k++)
        for (int mn = 0; mn < MN; mn++) {
          float src_data = [&]() {
            if constexpr (sizeof_bits_v<SrcElement> >= 8)
              return (float)(src_tensor(mn, k, l));
            else
              return (float)(src_tensor(mn, k, l).get());
          }();
          float scale_data = (float)(scale_tensor(mn, k / group_size, l));
          dst_tensor(mn, k, l) = src_data * scale_data;
        }

    cutlass::device_memory::copy_to_device(dq_buffer, (DstElement*)(raw_pointer_cast(dst_tensor.data())), dst_tensor.size());
    compat::wait();
  }

  // Host-side dequantize B: dst[n,k] = src[n,k] * scale[n, k/group_k]
  template <class DstElement, class SrcElement, class Layout, class ScaleElement, class ScaleLayout>
  static void apply_scale_B(DstElement* dq_buffer, SrcElement const* q_buffer, Layout const operand_layout,
                             ScaleElement const* scale_buffer, ScaleLayout const scale_layout,
                             Options const& options, int group_k) {
    std::vector<uint8_t> dst(size(operand_layout) * sizeof_bits_v<DstElement> / 8, 0);
    cutlass::device_memory::copy_to_host(dst.data(), (uint8_t*)dq_buffer, dst.size());
    std::vector<uint8_t> src(size(operand_layout) * sizeof_bits_v<SrcElement> / 8, 0);
    cutlass::device_memory::copy_to_host(src.data(), (uint8_t*)q_buffer, src.size());
    std::vector<uint8_t> scale(size(scale_layout) * sizeof_bits_v<ScaleElement> / 8, 0);
    cutlass::device_memory::copy_to_host(scale.data(), (uint8_t*)scale_buffer, scale.size());
    compat::wait();

    auto dst_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<DstElement*>(dst.data())), operand_layout);
    auto src_tensor = [&]() {
      if constexpr (sizeof_bits_v<SrcElement> < 8)
        return make_tensor(cute::subbyte_iterator<const SrcElement>(src.data()), operand_layout);
      else
        return make_tensor(make_gmem_ptr(reinterpret_cast<SrcElement const*>(src.data())), operand_layout);
    }();
    auto scale_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<ScaleElement const*>(scale.data())), scale_layout);

    auto N = size<0>(src_tensor); auto K = size<1>(src_tensor); auto L = size<2>(src_tensor);
    for (int l = 0; l < L; l++)
      for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++) {
          float src_data = [&]() {
            if constexpr (sizeof_bits_v<SrcElement> >= 8)
              return (float)(src_tensor(n, k, l));
            else
              return (float)(src_tensor(n, k, l).get());
          }();
          float scale_data = (float)(scale_tensor(n, k / group_k, l));
          dst_tensor(n, k, l) = src_data * scale_data;
        }

    cutlass::device_memory::copy_to_device(dq_buffer, (DstElement*)(raw_pointer_cast(dst_tensor.data())), dst_tensor.size());
    compat::wait();
  }

  void initialize(Options const& options) {
    auto M = options.m, N = options.n, K = options.k, L = options.l;
    const int scale_k = cute::ceil_div(K, scaleGroupSize);
    auto shape_A  = cute::make_shape(M, K, L);
    auto shape_B  = cute::make_shape(N, K, L);
    auto shape_CD = cute::make_shape(M, N, L);

    constexpr int scaleAlign = 1;
    int padded_M = cute::round_up(M, scaleAlign);
    int padded_N = cute::round_up(N, scaleAlign);
    auto shape_scale_A_padded = cute::make_shape(padded_M, scale_k, L);
    auto shape_scale_B_padded = cute::make_shape(padded_N, scale_k, L);

    stride_A  = cutlass::make_cute_packed_stride(StrideA{}, shape_A);
    stride_B  = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
    stride_C  = cutlass::make_cute_packed_stride(StrideC{}, shape_CD);
    stride_D  = cutlass::make_cute_packed_stride(StrideD{}, shape_CD);
    stride_SA = cutlass::make_cute_packed_stride(StrideScale{}, shape_scale_A_padded);
    stride_SB = cutlass::make_cute_packed_stride(StrideScale{}, shape_scale_B_padded);

    block_A.reset(size_t(M) * K * L);
    block_A_dq.reset(size_t(M) * K * L);
    block_B.reset(size_t(K) * N * L);
    block_B_dq.reset(size_t(K) * N * L);
    block_C.reset(size_t(M) * N * L);
    block_D.reset(size_t(M) * N * L);
    block_ref_D.reset(size_t(M) * N * L);
    block_scaleA.reset(size_t(scale_k) * L * padded_M);
    block_scaleB.reset(size_t(scale_k) * L * padded_N);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);

    convert_dtype<ElementA, float, Runner>(block_A, block_A_dq);
    convert_dtype<ElementB, float, Runner>(block_B, block_B_dq);

    initialize_scale(block_scaleA, options);
    initialize_scale(block_scaleB, options);

    auto layout_A = make_layout(shape_A, stride_A);
    auto layout_B = make_layout(shape_B, stride_B);
    auto layout_scale_A = make_layout(shape_scale_A_padded, stride_SA);
    auto layout_scale_B = make_layout(shape_scale_B_padded, stride_SB);

    apply_scale(block_A_dq.get(), block_A.get(), layout_A, block_scaleA.get(), layout_scale_A, options, scaleGroupSize);
    apply_scale_B(block_B_dq.get(), block_B.get(), layout_B, block_scaleB.get(), layout_scale_B, options, scaleGroupSize);
  }

  bool verify(Options const& options) {
    auto M = options.m, N = options.n, K = options.k, L = options.l;
    cutlass::TensorRef ref_A(block_A_dq.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B_dq.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
        {M, N, K}, options.alpha, ref_A, cutlass::ComplexTransform::kNone,
        ref_B, cutlass::ComplexTransform::kNone, options.beta, ref_C, ref_D,
        ElementAccumulator(0), L, M * K, K * N, M * N, M * N);
    compat::wait();

    ElementOutput const epsilon(1e-2f);
    ElementOutput const non_zero_floor(1e-4f);
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref_D.get(), block_D.get(), block_D.size(), epsilon, non_zero_floor);
    if (!passed) {
      std::vector<ElementOutput> ref_host(block_ref_D.size());
      std::vector<ElementOutput> out_host(block_D.size());
      compat::memcpy(ref_host.data(), block_ref_D.get(), ref_host.size() * sizeof(ElementOutput));
      compat::memcpy(out_host.data(), block_D.get(), out_host.size() * sizeof(ElementOutput));
      compat::wait();
      for (int i = 0; i < (int)out_host.size() && i < 32; i++) {
        printf("i: %d , ref: %f, comp: %f\n", i, float(ref_host[i]), float(out_host[i]));
      }
    }
    return passed;
  }

  cutlass::Status run(Options const& options) {
    auto M = options.m, N = options.n, K = options.k, L = options.l;

    initialize(options);

    // Build kernel params
    auto shape_A  = cute::make_shape(M, K, L);
    auto shape_B  = cute::make_shape(N, K, L);
    auto shape_CD = cute::make_shape(M, N, L);
    const int scale_k = cute::ceil_div(K, scaleGroupSize);
    int padded_M = cute::round_up(M, 1);
    int padded_N = cute::round_up(N, 1);

    auto mA = make_tensor(make_gmem_ptr(static_cast<ElementA const*>(block_A.get())), make_layout(shape_A, stride_A));
    auto mB = make_tensor(make_gmem_ptr(static_cast<ElementB const*>(block_B.get())),
                          make_layout(shape_B, stride_B));
    auto mSA = make_tensor(make_gmem_ptr(static_cast<ElementScale const*>(block_scaleA.get())),
                           make_layout(cute::make_shape(padded_M, scale_k, L), stride_SA));
    auto mSB = make_tensor(make_gmem_ptr(static_cast<ElementScale const*>(block_scaleB.get())),
                           make_layout(cute::make_shape(padded_N, scale_k, L), stride_SB));
    auto mC = make_tensor(make_gmem_ptr(static_cast<ElementC const*>(block_C.get())),
                          make_layout(shape_CD, stride_C));
    auto mD = make_tensor(make_gmem_ptr(block_D.get()),
                          make_layout(shape_CD, stride_D));

    KernelParams params;
    params.M = M; params.N = N; params.K = K; params.L = L;
    params.alpha = options.alpha;
    params.beta = options.beta;
    params.mA_mkl = mA;
    params.mB_nkl = mB;
    params.mAscale = mSA;
    params.mBscale = mSB;
    params.mC = mC;
    params.mD = mD;
    params.raster_along_n = true;

    // Grid dimensions
    int grid_m = ceil_div(M, BLK_M);
    int grid_n = ceil_div(N, BLK_N);
    int grid_l = L;

    sycl::range<3> local_range(1, 1, MaxThreadsPerBlock);
    sycl::range<3> global_range;
    if (params.raster_along_n) {
      global_range = sycl::range<3>(grid_l, grid_m, grid_n * MaxThreadsPerBlock);
    } else {
      global_range = sycl::range<3>(grid_l, grid_n, grid_m * MaxThreadsPerBlock);
    }

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    syclex::properties kernel_props{
      syclex::sub_group_size<SubgroupSize>,
      intelex::grf_size<512>
    };

    auto Q = compat::get_default_queue();

    auto launch = [&]() {
      auto event = Q.parallel_for(
        sycl::nd_range<3>(global_range, local_range),
        kernel_props,
        [=](auto) {
          bare_cute_blockscaled_gemm_kernel(params);
        }
      );
      EventManager::getInstance().addEvent(event);
    };

    launch();
    compat::wait();

    if (options.verify) {
      bool passed = verify(options);
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
      if (!passed) return cutlass::Status::kErrorInternal;
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        launch();
      }
      compat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tflops = (2.0 * M * N * K * L) * 1e-12;
      std::cout << "Problem Size: " << M << 'x' << N << 'x' << K << 'x' << L << std::endl;
      printf("Bare CuTe GEMM Performance:  [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time * 1000);
    }

    return cutlass::Status::kSuccess;
  }
};

///////////////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv) {
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  CUTLASS_CHECK(Runner{}.run(options));
  return 0;
}
