/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation. All rights reserved.
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
//
// Split-K GEMM with Async Reduce Epilogue (Xe4 CuTe Tutorial)
//
// Demonstrates: ADMA Load atoms + AMMA MMA atoms + ADMA STORE_REDUCE atom
// Multiple workgroups compute partial K-slices and atomically reduce results
// into a shared output matrix D using XE4_ADMA_STORE_REDUCE<T, RedOp::Add>.
//
// Architecture: 2-Subgroup warp specialization
//   SG0 (Producer):  ADMA Load A/B into SLM via copy atoms, ADMA STORE_REDUCE epilogue
//   SG1 (Consumer):  AMMA async MMA via gemm() with tracking
//

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gpu_generics.h>

#include <cute/arch/mma_xe4_amma.hpp>
#include <cute/arch/mma_xe4.hpp>
#include <cute/atom/mma_traits_xe4_amma.hpp>
#include <cute/atom/copy_traits_xe4_adma.hpp>
#include <cute/arch/xe4_util.hpp>

#include "../../../common/sycl_cute_common.hpp"
#include "../../../xe4/utils/validation.hpp"

using namespace cute;
using sycl::ext::oneapi::this_work_item::get_nd_item;

constexpr static size_t SmemAlignment = 256;

template <class ElementA,
          class ElementB,
          class ElementD,
          class SmemLayoutA,
          class SmemLayoutB,
          class SmemLayoutAcc>
struct SharedStorage : cute::aligned_struct<SmemAlignment, _0>
{
  cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>, SmemAlignment> smem_A;
  cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>, SmemAlignment> smem_B;
  cute::array_aligned<ElementD, cute::cosize_v<SmemLayoutAcc>, SmemAlignment> smem_D;
};

inline void xe4_syncthreads() {
  auto group = get_nd_item<1>().get_group();
  sycl::group_barrier(group);
}

////////////////////////////////////////////////////////////////////////////////
// Device kernel: Split-K GEMM with ADMA load/store atoms and AMMA compute
////////////////////////////////////////////////////////////////////////////////
template <class ProblemShape, class CtaTiler, class TileShape,
          class TA, class SmemLayoutA, class SmemLayoutAcc, class ADMA_A,
          class TB, class SmemLayoutB, class ADMA_B, class ADMA_Reduce,
          class TD, class TiledMma>
void
gemm_reduce_device(ProblemShape shape_MNK, CtaTiler cta_tiler, TileShape tile_shape,
            TA const* A, SmemLayoutA sA_layout, SmemLayoutAcc sD_layout, ADMA_A adma_load_a,
            TB const* B, SmemLayoutB sB_layout, ADMA_B adma_load_b,
            ADMA_Reduce adma_reduce_d,
            TD* D,
            int k_tile_start, int k_tiles_this_wg,
            sycl::nd_item<3> item)
{
  static_assert(rank(shape_MNK) == 3);
  static_assert(rank(cta_tiler) == 3);
  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  // Set up hardware tensor descriptors from the ADMA atoms
  auto tdesc_a = allocate_tdesc<0>();
  auto tdesc_b = allocate_tdesc<1>();
  auto tdesc_d = allocate_tdesc<2>();

  adma_load_a.set_tensor_desc(tdesc_a);
  adma_load_b.set_tensor_desc(tdesc_b);
  adma_reduce_d.set_tensor_desc(tdesc_d);

  using SharedStorageType = SharedStorage<TA, TB, TD, SmemLayoutA, SmemLayoutB, SmemLayoutAcc>;
  auto ptr = alloc_slm_buffer<uint8_t, sizeof(SharedStorageType)>(item.get_group());
  auto& smem = *reinterpret_cast<SharedStorageType*>(ptr);

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.begin()), SmemLayoutA{});   // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.begin()), SmemLayoutB{});   // (BLK_N,BLK_K,PIPE)
  Tensor sD = make_tensor(make_smem_ptr(smem.smem_D.begin()), SmemLayoutAcc{}); // (BLK_M,BLK_N)

  // Get full TMA tensors from the atoms
  auto [M, N, K] = shape_MNK;
  auto mA = adma_load_a.get_tma_tensor(make_shape(M, K));
  auto mB = adma_load_b.get_tma_tensor(make_shape(N, K));
  auto mD = adma_reduce_d.get_tma_tensor(make_shape(M, N));

  // Get this workgroup's tiles
  auto cta_coord = make_coord(BlockIdxX(), BlockIdxY(), _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});   // (BLK_M, BLK_K, m, k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});  // (BLK_N, BLK_K, n, k)
  Tensor gD = local_tile(mD, cta_tiler, cta_coord, Step<_1, _1, X>{});   // (BLK_M, BLK_N)

  // Partition using tma_partition for ADMA atoms
  auto [tAgA, tAsA] = tma_partition(adma_load_a,
                                     group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto [tBgB, tBsB] = tma_partition(adma_load_b,
                                     group_modes<0,2>(sB), group_modes<0,2>(gB));

  constexpr int K_PIPE_MAX = decltype(size<1>(tAsA))::value;

  // Transaction bytes for barriers
  constexpr int dma_transaction_bytesA = (cosize(SmemLayoutA{}) * sizeof(TA)) / K_PIPE_MAX;
  constexpr int dma_transaction_bytesB = (cosize(SmemLayoutB{}) * sizeof(TB)) / K_PIPE_MAX;
  constexpr int dma_transaction_bytesD = (cosize(SmemLayoutAcc{}) * sizeof(TD));

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t warp_idx = get_sg_id();

  // Allocate barriers: load A/B per pipe stage, MMA per pipe stage, reduce
  auto load_a_abar = allocate_abar<0, K_PIPE_MAX>();
  auto load_b_abar = allocate_abar<1, K_PIPE_MAX>();
  auto mma_abar    = allocate_abar<2, K_PIPE_MAX>();
  auto reduce_abar = allocate_abar<3>();

  // Initialize barriers
  if (elect_one_thr && warp_idx == 0) {
    for (int i = 0; i < K_PIPE_MAX; ++i) {
      xe4_initialize_barrier(load_a_abar[i], 1);
      xe4_initialize_barrier(load_b_abar[i], 1);
    }
  } else if (elect_one_thr && warp_idx == 1) {
    for (int i = 0; i < K_PIPE_MAX; ++i) {
      xe4_initialize_barrier(mma_abar[i], 1);
    }
    xe4_initialize_barrier(reduce_abar[0], 1);
  }
  xe4_syncthreads();

  uint32_t dummy_mask = 0;

  // Set up MMA
  TiledMma mma{};
  ThrMMA thr_mma = mma.get_thread_slice(ThreadIdxX());
  Tensor tCsA = thr_mma.partition_A(sA);   // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);   // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCsD = thr_mma.partition_C(sD);   // (MMA,MMA_M,MMA_N)
  Tensor tCgD = thr_mma.partition_C(gD);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrD = thr_mma.make_fragment_C(tCsD);

  // Clear accumulator in SLM
  clear(tCsD);
  xe4_syncthreads();

  uint64_t mma_ctrl = 0x100;

  int write_pipe = 0;
  uint32_t write_phase = 0;
  int read_pipe = 0;
  uint32_t read_phase = 0;

  // Producer sub-group (SG0): ADMA loads
  if (warp_idx == 0) {
    if (elect_one_thr) {
      // Prologue: fill all pipeline stages
      int k_tile = k_tile_start;
      int prologue_count = (k_tiles_this_wg < K_PIPE_MAX) ? k_tiles_this_wg : K_PIPE_MAX;
      for (int pipe = 0; pipe < prologue_count; ++pipe) {
        xe4_set_barrier_transaction_bytes(load_a_abar[pipe], dma_transaction_bytesA);
        xe4_set_barrier_transaction_bytes(load_b_abar[pipe], dma_transaction_bytesB);
        copy(adma_load_a.with(&load_a_abar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
        copy(adma_load_b.with(&load_b_abar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
        ++k_tile;
      }
      // Mainloop: wait for MMA consumption, then refill
      for (int remaining = prologue_count; remaining < k_tiles_this_wg; ++remaining) {
        xe4_wait_barrier(mma_abar[write_pipe], write_phase);
        xe4_set_barrier_transaction_bytes(load_a_abar[write_pipe], dma_transaction_bytesA);
        xe4_set_barrier_transaction_bytes(load_b_abar[write_pipe], dma_transaction_bytesB);
        copy(adma_load_a.with(&load_a_abar[write_pipe]), tAgA(_, k_tile), tAsA(_, write_pipe));
        copy(adma_load_b.with(&load_b_abar[write_pipe]), tBgB(_, k_tile), tBsB(_, write_pipe));
        ++k_tile;
        if (++write_pipe == K_PIPE_MAX) { write_pipe = 0; write_phase ^= 1; }
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }
  // Consumer sub-group (SG1): AMMA compute
  else if (warp_idx == 1) {
    if (elect_one_thr) {
      // Mainloop: wait for loads, issue MMA with AB tracking
      for (int k_iter = 0; k_iter < k_tiles_this_wg - 1; ++k_iter) {
        xe4_wait_barrier(load_a_abar[read_pipe], read_phase);
        xe4_wait_barrier(load_b_abar[read_pipe], read_phase);
        xe4_set_barrier_transaction_bytes(mma_abar[read_pipe], 2);
        auto new_mma = mma.with(AMMA::TrackMethod<AMMA::Tracking::AB>{}, mma_ctrl,
                                &mma_abar[read_pipe], &mma_abar[read_pipe],
                                dummy_mask, dummy_mask);
        cute::gemm(new_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrD);
        mma_ctrl = 0x000;
        if (++read_pipe == K_PIPE_MAX) { read_pipe = 0; read_phase ^= 1; }
      }
      // Last iteration: track D completion (signals when accumulator is written to SLM)
      {
        xe4_wait_barrier(load_a_abar[read_pipe], read_phase);
        xe4_wait_barrier(load_b_abar[read_pipe], read_phase);
        xe4_set_barrier_transaction_bytes(mma_abar[read_pipe], 1);
        auto new_mma = mma.with(AMMA::TrackMethod<AMMA::Tracking::D>{}, mma_ctrl,
                                &mma_abar[read_pipe]);
        cute::gemm(new_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrD);
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }

  xe4_syncthreads();

  // Epilogue: ADMA STORE_REDUCE — atomic accumulation from SLM to GMEM
  if (warp_idx == 0) {
    if (elect_one_thr) {
      xe4_wait_barrier(mma_abar[read_pipe], read_phase);
      xe4_set_barrier_transaction_bytes(reduce_abar[0], dma_transaction_bytesD);
      copy(adma_reduce_d.with(&reduce_abar[0]), tCsD, tCgD);
      xe4_wait_barrier(reduce_abar[0], 0);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Host setup and launch
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  sycl::queue queue{sycl::gpu_selector_v};
  std::cout << "Running on device: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  int m = 256;
  if (argc >= 2) sscanf(argv[1], "%d", &m);
  int n = 256;
  if (argc >= 3) sscanf(argv[2], "%d", &n);
  int k = 256;
  if (argc >= 4) sscanf(argv[3], "%d", &k);

  using TA = fp16;
  using TB = fp16;
  using TD = float;

  auto prob_shape = make_shape(m, n, k);

  auto A = make_shared_usm_tensor<TA, 'R'>(queue, m, k);
  auto B = make_shared_usm_tensor<TB, 'R'>(queue, n, k);
  auto D = make_shared_usm_tensor<TD, 'R'>(queue, m, n);

  random_fill(A);
  random_fill(B);
  zero_fill(D);

  // Keep a copy of A/B before subbyte_pack for verification
  auto A_ref = make_shared_usm_tensor<TA, 'R'>(queue, m, k);
  auto B_ref = make_shared_usm_tensor<TB, 'R'>(queue, n, k);
  copy(A, A_ref);
  copy(B, B_ref);

  subbyte_pack(A);
  subbyte_pack(B);

  TA const* A_ptr = &*A.data();
  TB const* B_ptr = &*B.data();
  TD* D_ptr = &*D.data();

  auto dA = A.stride();
  auto dB = B.stride();
  auto dD = D.stride();

  Tensor mA = make_tensor(make_gmem_ptr(A_ptr), make_layout(make_shape(m, k), dA));
  Tensor mB = make_tensor(make_gmem_ptr(B_ptr), make_layout(make_shape(n, k), dB));
  Tensor mD = make_tensor(make_gmem_ptr(D_ptr), make_layout(make_shape(m, n), dD));

  // Tile and cluster shapes
  using TileShape_MNK = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;
  constexpr auto bM = get<0>(TileShape_MNK{});
  constexpr auto bN = get<1>(TileShape_MNK{});
  constexpr auto bK = get<2>(TileShape_MNK{});
  auto cta_tiler = make_shape(bM, bN, bK);
  constexpr auto bP = Int<3>{};  // Pipeline depth

  static constexpr auto majorA = cute::AMMA::Major::K;
  static constexpr auto majorB = cute::AMMA::Major::K;

  using ElementAccumulator = float;

  // MMA atom
  using TiledMma = decltype(cute::make_tiled_mma(
    cute::AMMA::ss_op_selector<
      ElementAccumulator, TA, TB, TD,
      decltype(cute::product_each(TileShape_MNK{})),
      ClusterShape_MNK, majorA, majorB>()
  ));

  // SMEM layouts (static)
  using SmemLayoutAtomA = decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenRowMajor{}));
  using SmemLayoutAtomB = decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenRowMajor{}));
  using SmemLayoutAtomD = decltype(make_layout(cute::select<0, 1>(TileShape_MNK{}), GenRowMajor{}));

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
      Step<_2, _1, _3>{}));

  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
      Step<_2, _1, _3>{}));

  using SmemLayoutAcc = decltype(tile_to_shape(
      SmemLayoutAtomD{},
      make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), Step<_2, _1>{}));

  SmemLayoutA sA_layout{};
  SmemLayoutB sB_layout{};
  SmemLayoutAcc sD_layout{};

  // Cluster layout for non-cluster case
  auto cluster_layout_vmnk = tiled_divide(make_layout(ClusterShape_MNK{}),
                                           make_tile(typename TiledMma::AtomThrID{}));

  // ADMA Load atoms for A and B
  using GmemTiledCopyA = cute::XE4_ADMA_LOAD;
  using GmemTiledCopyB = cute::XE4_ADMA_LOAD;

  auto adma_load_a = make_adma_atom_A_xe4(
      GmemTiledCopyA{},
      mA,
      SmemLayoutA{}(_, _, cute::Int<0>{}),
      TileShape_MNK{},
      TiledMma{},
      cluster_layout_vmnk);

  auto adma_load_b = make_adma_atom_B_xe4(
      GmemTiledCopyB{},
      mB,
      SmemLayoutB{}(_, _, cute::Int<0>{}),
      TileShape_MNK{},
      TiledMma{},
      cluster_layout_vmnk);

  // ADMA STORE_REDUCE atom for D (atomic float add)
  using GmemTiledCopyD = cute::XE4_ADMA_STORE_REDUCE<TD, cute::RedOp::Add, cute::BarrierType::Abarrier>;
  auto cta_tiler_mn = make_shape(bM, bN);
  auto adma_reduce_d = make_adma_copy<TD>(GmemTiledCopyD{}, mD, SmemLayoutAcc{}, cta_tiler_mn, Int<1>{});

  // Split-K configuration
  constexpr int total_k_tiles = 256 / 128;  // k / bK
  constexpr int num_k_splits = total_k_tiles;
  constexpr int k_tiles_per_wg = 1;

  // Launch configuration
  constexpr int NumWarps = 2;
  constexpr int NumThreadsPerWarp = 32;
  auto num_groups_mn = ceil_div(prob_shape, TileShape_MNK{});

  namespace syclexp = sycl::ext::oneapi::experimental;
  auto [cluster_size_y, cluster_size_x, cluster_size_z] = ClusterShape_MNK{};
  sycl::range<3> clusterSize(cluster_size_z, cluster_size_y, cluster_size_x);
  syclexp::properties Props{syclexp::work_groups_per_cluster<3>(clusterSize)};

  sycl::range<3> local_range(1, NumWarps, NumThreadsPerWarp);
  sycl::range<3> group_range(num_k_splits,
                             get<1>(num_groups_mn), get<0>(num_groups_mn));
  sycl::nd_range<3> Range(group_range * local_range, local_range);

  std::cout << "Running Split-K GEMM with ADMA STORE_REDUCE epilogue..." << std::endl;
  std::cout << "  Problem: " << m << "x" << n << "x" << k << std::endl;
  std::cout << "  Tile: " << bM << "x" << bN << "x" << bK << std::endl;
  std::cout << "  K-splits: " << num_k_splits << " (tiles/WG: " << k_tiles_per_wg << ")" << std::endl;

  auto launch_cfg = syclexp::launch_config(Range, Props);
  syclexp::submit_with_event(queue, [&](sycl::handler &handler) {
    syclexp::nd_launch(handler, launch_cfg, [=](sycl::nd_item<3> item) ALWAYS_INLINE {
      int k_split_id = item.get_group(0);
      int k_tile_start = k_split_id * k_tiles_per_wg;

      gemm_reduce_device<decltype(prob_shape), decltype(cta_tiler), decltype(TileShape_MNK{}),
                  TA, decltype(sA_layout), decltype(sD_layout), decltype(adma_load_a),
                  TB, decltype(sB_layout), decltype(adma_load_b), decltype(adma_reduce_d),
                  TD, TiledMma>(
                  prob_shape, cta_tiler, TileShape_MNK{},
                  A_ptr, sA_layout, sD_layout, adma_load_a,
                  B_ptr, sB_layout, adma_load_b, adma_reduce_d,
                  D_ptr,
                  k_tile_start, k_tiles_per_wg, item);
    });
  }).wait();

  // Verification using validate_gemm_result (MKL GEMM reference).
  std::cout << "=======TEST RESULT========" << std::endl;

  TA* A_ref_ptr = &*A_ref.data();
  TB* B_ref_ptr = &*B_ref.data();

  // A is M×K row_major; B is N×K row_major (i.e. B^T is K×N row_major), so
  // pass col_major for B to make MKL transpose it before multiplying.
  int err_cnt = validate_gemm_result(A_ref_ptr, B_ref_ptr, D_ptr, m, n, k,
                                     mem_layout::row_major, mem_layout::col_major);
  printf("Verification: %s\n", (err_cnt == 0) ? "PASSED" : "FAILED");

  return err_cnt == 0 ? 0 : 1;
}
