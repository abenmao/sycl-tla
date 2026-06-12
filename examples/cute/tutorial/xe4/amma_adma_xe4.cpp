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
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gpu_generics.h>

#include "cute/arch/mma_xe4_amma.hpp"
#include <cute/arch/mma_xe4.hpp>
#include <cute/atom/mma_traits_xe4_amma.hpp>
#include <cute/atom/copy_traits_xe4_adma.hpp>
#include <cute/arch/xe4_util.hpp>
#include "cutlass/arch/barrier.h"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "../../../common/sycl_cute_common.hpp"
#include "../../../xe4/utils/validation.hpp"

using namespace cute;

using sycl::ext::oneapi::this_work_item::get_nd_item;

constexpr static size_t SmemAlignment = 256;

template <class ElementA,
          class ElementB,
          class ElementC,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB,
          class SmemLayoutC>  // (N,K,P)
struct SharedStorage : cute::aligned_struct<SmemAlignment, _0>
{
  cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>, SmemAlignment> smem_A;
  cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>, SmemAlignment> smem_B;
  cute::array_aligned<ElementC, cute::cosize_v<SmemLayoutC>, SmemAlignment> smem_C;
};

inline void xe4_syncthreads() {
  auto group = get_nd_item<1>().get_group();
  sycl::group_barrier(group);
}

template <class ProblemShape, class CtaTiler, class TileShape,
          class TA, class SmemLayoutA, class SmemLayoutAcc, class ADMA_A,
          class TB, class SmemLayoutB, class ADMA_B, class ADMA_C,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler, TileShape tile_shape,
            TA const* A, SmemLayoutA sA_layout, SmemLayoutAcc sC_layout, ADMA_A adma_load_a,
            TB const* B, SmemLayoutB sB_layout, ADMA_B adma_load_b, ADMA_C adma_store_c,
            TC* C, CStride dC,
            Alpha alpha, Beta beta,
            sycl::nd_item<3> item)
{
  // Preconditions
  static_assert(rank(shape_MNK) == 3);                                 // (M, N, K)
  static_assert(rank(cta_tiler) == 3);                                 // (BLK_M, BLK_N, BLK_K)

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  static_assert(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));         // BLK_M
  static_assert(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));         // BLK_N
  static_assert(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));         // BLK_K
  static_assert(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));         // BLK_K
  
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  
  // Creating tensor descriptors for ADMA Copy. 
  auto tdesc_a = allocate_tdesc<0>();
  auto tdesc_b = allocate_tdesc<1>();
  auto tdesc_c = allocate_tdesc<2>();

  // Setting tensor descriptors for ADMA copy atoms.
  adma_load_a.set_tensor_desc(tdesc_a);
  adma_load_b.set_tensor_desc(tdesc_b);
  adma_store_c.set_tensor_desc(tdesc_c);

  using SharedStorageType = SharedStorage<TA, TB, TC, SmemLayoutA, SmemLayoutB, SmemLayoutAcc>;
  auto ptr = alloc_slm_buffer<uint8_t, sizeof(SharedStorageType)>(item.get_group());
  auto& smem = *reinterpret_cast<SharedStorageType*>(ptr);
  
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.begin()), SmemLayoutA{});                             // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.begin()), SmemLayoutB{});                             // (BLK_N,BLK_K,PIPE)
  Tensor sC = make_tensor(make_smem_ptr(smem.smem_C.begin()), SmemLayoutAcc{});                           // (BLK_M,BLK_N,PIPE)
  
  // Represent the full tensors
  auto [M, N, K] = shape_MNK;
  auto mA = adma_load_a.get_tma_tensor(make_shape(M, K));                                                  // (m,k)
  auto mB = adma_load_b.get_tma_tensor(make_shape(N, K));                                                  // (n,k)
  auto mC = adma_store_c.get_tma_tensor(make_shape(M, N));                                                 // (m,n)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(BlockIdxX(), BlockIdxY(), _);                                                // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});                                     // (BLK_M, BLK_K, m, k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});                                    // (BLK_N, BLK_K, n, k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});                                      // (BLK_M,BLK_N)

  // Partition the copying of A and B tiles using XE4 DMA
  auto [tAgA, tAsA] = tma_partition(adma_load_a,
                                      group_modes<0,2>(sA), group_modes<0,2>(gA));

  auto [tBgB, tBsB] = tma_partition(adma_load_b,
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));
  
  // Total stages
  auto K_PIPE_MAX = size<1>(tAsA);
  // Total count of tiles
  int K_TILE_MAX = size<1>(tAgA);
  // Current tile index in gmem to read from
  int k_tile = 0;

  // Calculate adma transaction bytes for each operand.
  constexpr int dma_transaction_bytesA = (cosize(sA_layout) * sizeof(TA)) / K_PIPE_MAX;
  constexpr int dma_transaction_bytesB = (cosize(sB_layout) * sizeof(TB)) / K_PIPE_MAX;
  constexpr int dma_transaction_bytesC = (cosize(sC_layout) * sizeof(TC));

  uint32_t elect_one_thr  = cute::elect_one_sync();  // Elected leader lane (wor-item) in a sub-group.
  uint32_t warp_idx = get_sg_id();  // Sub-group = 0 for ADMA ld/st & Sub-group = 1 for AMMA.

  // Allocate abarrier for Load A/B, MMA and Store C.

  uint32_t mma_abar_phase = 0;
  auto load_a_abar = cutlass::arch::allocate_cluster_tx_barriers<K_PIPE_MAX>();
  auto load_b_abar = cutlass::arch::allocate_cluster_tx_barriers<K_PIPE_MAX>();
  auto& mma_abar = cutlass::arch::allocate_cluster_tx_barrier();
  auto& store_c_abar = cutlass::arch::allocate_cluster_tx_barrier();
  
  // Only lanes (work-items) using abarrier should initialize it.
  if (elect_one_thr && warp_idx == 0) {
    for (int i = 0; i < K_PIPE_MAX; ++i){
      load_a_abar[i].init(1 /*numThreads*/);
      load_b_abar[i].init(1 /*numThreads*/);
    }
  } else if (elect_one_thr && warp_idx == 1) {
    mma_abar.init(1 /*numThreads*/);
    store_c_abar.init(1 /*numThreads*/);
  }
  xe4_syncthreads();
  
  // Phase barrier bit for store C barrier.
  int store_c_barrier_phase_bit = 0;
  // Dummy cluster mask = 0 for non-cluster based example.
  uint32_t dummy_mask = 0;

  // Partition for this CTA 
  TiledMma mma{};
  ThrMMA thr_mma = mma.get_thread_slice(ThreadIdxX());
  Tensor tCsA = thr_mma.partition_A(sA);                                                                    // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);                                                                    // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC); 
  Tensor tCsC = thr_mma.partition_C(sC);                                                                    // (MMA,MMA_M,MMA_N)
  
  // // Allocate fragments for A/B/C in registers representing corresponding layouts in SLM.
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCsC);                                        

  // Clear accumlators in SLM.
  clear(tCsC);
  xe4_syncthreads();

  uint64_t mma_ctrl = 0x100;

  // Using PipelineState API to manage read/write states.
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();

  if (warp_idx == 0)
  {
    if (elect_one_thr) 
    {
      int prologue_pipe_max = std::min((int) K_PIPE_MAX, K_TILE_MAX);
      // Prologue: Fill all SLM pipes for A/B.
      for (int pipe = 0; pipe < prologue_pipe_max; ++pipe)
      {
          load_a_abar[pipe].arrive_and_expect_tx(dma_transaction_bytesA);
          load_b_abar[pipe].arrive_and_expect_tx(dma_transaction_bytesB);
          copy(adma_load_a.with(reinterpret_cast<uint64_t*>(&load_a_abar[pipe])), tAgA(_,k_tile), tAsA(_,pipe));
          copy(adma_load_b.with(reinterpret_cast<uint64_t*>(&load_b_abar[pipe])), tBgB(_,k_tile), tBsB(_,pipe));
          ++k_tile;
      }
      // Mainloop: Wait for MMA[pipe] to complete, then overwrite with next tile.
      for (int k_tile_next = k_tile; k_tile_next < K_TILE_MAX; ++k_tile_next)
      {
        int write_pipe = write_state.index();
        mma_abar.try_wait(mma_abar_phase); 
        mma_abar_phase ^= 1;
        load_a_abar[write_pipe].arrive_and_expect_tx(dma_transaction_bytesA);
        load_b_abar[write_pipe].arrive_and_expect_tx(dma_transaction_bytesB);
        copy(adma_load_a.with(reinterpret_cast<uint64_t*>(&load_a_abar[write_pipe])), tAgA(_,k_tile_next), tAsA(_,write_pipe));
        copy(adma_load_b.with(reinterpret_cast<uint64_t*>(&load_b_abar[write_pipe])), tBgB(_,k_tile_next), tBsB(_,write_pipe));
        ++write_state;
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }
  else if (warp_idx == 1) 
  {
    if (elect_one_thr) 
    {
      // Mainloop: Wait for load pipe to complete, then perform MMA.
      for (int k_tile_next = 0; k_tile_next < K_TILE_MAX - 1; ++k_tile_next) 
      {
        int read_pipe = read_state.index();
        load_a_abar[read_pipe].try_wait(read_state.phase());
        load_b_abar[read_pipe].try_wait(read_state.phase());
        mma_abar.arrive_and_expect_tx(2);
        auto new_mma = mma.with(AMMA::TrackMethod<AMMA::Tracking::AB>{},
                                mma_ctrl,
                                reinterpret_cast<uint64_t*>(&mma_abar),
                                reinterpret_cast<uint64_t*>(&mma_abar),
                                dummy_mask,
                                dummy_mask);
        cute::gemm(new_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
        mma_ctrl = 0x000;
        mma_abar.try_wait(mma_abar_phase);
        mma_abar_phase ^= 1;
        ++read_state;
      }
      // Mainloop last iteration: Only track D completion for notifying store barrier.
      {
        int read_pipe = read_state.index();
        load_a_abar[read_pipe].try_wait(read_state.phase());
        load_b_abar[read_pipe].try_wait(read_state.phase());
        mma_abar.arrive_and_expect_tx(1);
        auto new_mma = mma.with(AMMA::TrackMethod<AMMA::Tracking::D>{}, mma_ctrl, reinterpret_cast<uint64_t*>(&mma_abar));
        cute::gemm(new_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
        mma_abar.try_wait(mma_abar_phase);
        mma_abar_phase ^= 1;
      }
    }
   sycl::group_barrier(item.get_sub_group());
  }

  xe4_syncthreads();

  // Epilogue: Store result back to GMEM
  if (warp_idx == 0)
  {
    if (elect_one_thr) 
    {
      int read_pipe = read_state.index();
      store_c_abar.arrive_and_expect_tx(dma_transaction_bytesC);
      copy(adma_store_c.with(reinterpret_cast<uint64_t*>(&store_c_abar)), tCsC, tCgC);
      store_c_abar.try_wait(store_c_barrier_phase_bit);
      store_c_barrier_phase_bit ^= 1;
    }
  }

}

// Setup params for a TN GEMM
template <class TensorA, class TensorB, class TensorC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TensorA const& A,
        TensorB const& B,
        Beta beta,
        TensorC& C,
        sycl::queue& queue)
{
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                                                                    // (M, N, K)

  // Extract element types and raw pointers from tensors
  using TA = typename TensorA::element_type;
  using TB = typename TensorB::element_type;
  using TC = typename TensorC::element_type;
  
  TA const* A_ptr = &*A.data();
  TB const* B_ptr = &*B.data();
  TC* C_ptr = &*C.data();
  
  // Extract strides from tensor layouts
  auto dA = A.stride();
  auto dB = B.stride();
  auto dC = C.stride();

  Tensor mA = make_tensor(make_gmem_ptr(A_ptr), make_layout(make_shape(M,K), dA));
  Tensor mB = make_tensor(make_gmem_ptr(B_ptr), make_layout(make_shape(N,K), dB));
  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), make_layout(make_shape(M,N), dC));

  // Define CTA tile sizes (static)
  using TileShape_MNK = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;
  constexpr auto bM = get<0>(TileShape_MNK{});
  constexpr auto bN = get<1>(TileShape_MNK{});
  constexpr auto bK = get<2>(TileShape_MNK{});
  auto cta_tiler = make_shape(bM, bN, bK);                                                                  // (BLK_M, BLK_N, BLK_K)
  constexpr auto bP = Int<3>{};  // Pipeline stages

  static constexpr auto majorA = cute::AMMA::Major::K;
  static constexpr auto majorB = cute::AMMA::Major::K;

  using ElementA = TA;
  using ElementB = TB;
  using ElementAccumulator = float;
  using ElementC = float;

  // Define the MMA for XE4 with K-major
  using TiledMma = decltype(cute::make_tiled_mma(
    cute::AMMA::ss_op_selector<
    ElementAccumulator,
    ElementA, ElementB,
    ElementC,
    decltype(cute::product_each(TileShape_MNK{})),
    ClusterShape_MNK, majorA, majorB>()
  ));
  
  TiledMma tiled_mma{};
  
  // Define the smem layouts (static) - XE4 K-major layouts
  using SmemLayoutAtomA =
    cute::conditional_t<
      majorA == cute::AMMA::Major::K,
      decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenRowMajor{})),
      decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenColMajor{}))
    >;
  using SmemLayoutAtomB =
    cute::conditional_t<
      majorB == cute::AMMA::Major::K,
      decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenRowMajor{})),
      decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenColMajor{}))
    >;
  using SmemLayoutAtomC =
    cute::conditional_t<
      majorA == cute::AMMA::Major::K,
      decltype(make_layout(cute::select<0, 1>(TileShape_MNK{}), GenRowMajor{})),
      decltype(make_layout(cute::select<0, 1>(TileShape_MNK{}), GenColMajor{}))
    >;
  
  using SmemLayoutA = decltype(tile_to_shape(
  SmemLayoutAtomA{},
  make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
  cute::conditional_t<majorA == cute::AMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
      cute::conditional_t<majorB == cute::AMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  using SmemLayoutAcc = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), Step<_2,_1>{}));

  SmemLayoutA sA{};
  SmemLayoutB sB{};
  SmemLayoutAcc sC{};

  using GmemTiledCopyA =
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::XE4_ADMA_LOAD,
      cute::XE4_ADMA_LOAD_MULTICAST
    >;

  using GmemTiledCopyB =
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::XE4_ADMA_LOAD,
      cute::XE4_ADMA_LOAD_MULTICAST
    >;
  
  using GmemTiledCopyC = cute::XE4_ADMA_STORE;

  // Non-cluster case gives effective layout ((_1),_1,_1,_1):((_0),_0,_0,_0)
  auto cluster_layout_vmnk = tiled_divide(make_layout(ClusterShape_MNK{}), make_tile(typename TiledMma::AtomThrID{}));

  // ADMA atoms for A/B copy
  auto adma_load_a = make_adma_atom_A_xe4(
      GmemTiledCopyA{},
      mA,
      SmemLayoutA{}(_,_,cute::Int<0>{}),
      TileShape_MNK{},
      TiledMma{},
      cluster_layout_vmnk);

  auto adma_load_b = make_adma_atom_B_xe4(
      GmemTiledCopyB{},
      mB,
      SmemLayoutB{}(_,_,cute::Int<0>{}),
      TileShape_MNK{},
      TiledMma{},
      cluster_layout_vmnk);
  
  // ADMA store atom for C.
  auto cta_tiler_mn = make_shape(bM, bN);  // 2D tiler for output
  auto adma_store_c = make_adma_copy<TC>(GmemTiledCopyC{}, mC, SmemLayoutAcc{}, cta_tiler_mn, Int<1>{});
  
  //
  // Setup and Launch
  // 

  constexpr int NumOfControlWarps = 2; // Sub-groups - 0th for ADMA and 1st for AMMA calls.
  constexpr int NumOfThreadsPerWarp = 32; // Work-items per sub-group.

  namespace syclexp = sycl::ext::oneapi::experimental;
  auto [cluster_size_y, cluster_size_x, cluster_size_z] = ClusterShape_MNK{};
  sycl::range<3> clusterSize(cluster_size_z, cluster_size_y, cluster_size_x);
  syclexp::properties Props {syclexp::work_groups_per_cluster<3>(clusterSize)};
  
  // Setup thread block configuration
  auto num_groups = ceil_div(prob_shape, TileShape_MNK{});
  sycl::range<3> local_range(1, NumOfControlWarps, NumOfThreadsPerWarp);
  sycl::range<3> group_range(1, get<1>(num_groups), get<0>(num_groups));  // (Z, Y=N_tiles, X=M_tiles)
  sycl::nd_range<3> Range(group_range * local_range, local_range);

  // Kernel Launch with cluster support
  auto launch_cfg = syclexp::launch_config(Range, Props);
  syclexp::submit_with_event(queue, [&](sycl::handler &handler) {
    syclexp::nd_launch(handler, launch_cfg, [=](sycl::nd_item<3> item) ALWAYS_INLINE {
      gemm_device<decltype(prob_shape), decltype(cta_tiler), decltype(TileShape_MNK{}),
                  TA, decltype(sA), decltype(sC), decltype(adma_load_a),
                  TB, decltype(sB), decltype(adma_load_b), decltype(adma_store_c),
                  TC, decltype(dC), TiledMma,
                  Alpha, Beta>(
                  prob_shape, cta_tiler, TileShape_MNK{},
                  A_ptr, sA, sC, adma_load_a,
                  B_ptr, sB, adma_load_b, adma_store_c,
                  C_ptr, dC,
                  alpha, beta, item);
    });
  }).wait();
}

template <class TensorA, class TensorB, class TensorC,
          class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     TensorA const& A,
     TensorB const& B,
     Beta beta,
     TensorC& C,
     sycl::queue& queue)
{
  if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, B, beta, C, queue);
  }
  assert(false && "Not implemented");
}

int main(int argc, char** argv)
{
  sycl::queue queue{sycl::gpu_selector_v};
  
  std::cout << "Running on device: " 
            << queue.get_device().get_info<sycl::info::device::name>() 
            << std::endl;

  int m = 256;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 256;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 256;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  char transA = 'T';
  if (argc >= 5)
    sscanf(argv[4], "%c", &transA);

  char transB = 'N';
  if (argc >= 6)
    sscanf(argv[5], "%c", &transB);

  using TA = fp16;
  using TB = fp16;
  using TC = float;
  using TI = float;

  TI alpha = TI(1.0f);
  TI beta  = TI(0.0f);

  auto prob_shape = make_shape(m, n, k);

  uint32_t sizeA = size(select<0,2>(prob_shape));
  uint32_t sizeB = size(select<1,2>(prob_shape));
  uint32_t sizeC = size(select<0,1>(prob_shape));

  auto A = make_shared_usm_tensor<TA, 'R'>(queue, m, k);
  auto B = make_shared_usm_tensor<TB, 'R'>(queue, n, k);
  auto C = make_shared_usm_tensor<TC, 'R'>(queue, m, n);

  random_fill(A);
  random_fill(B);
  zero_fill(C);

  bool ok = false;
  
  auto A_ref = make_shared_usm_tensor<TA, 'R'>(queue, m, k);
  auto B_ref = make_shared_usm_tensor<TB, 'R'>(queue, n, k);

  copy(A, A_ref);
  copy(B, B_ref);

  subbyte_pack(A);
  subbyte_pack(B);

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 0;

  int ldA = 0, ldB = 0, ldC = m;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false && "Unsupported transA");
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false && "Unsupported transB");
  }

  // Warmup
  gemm(transA, transB, m, n, k, alpha, A, B, beta, C, queue);
  queue.wait_and_throw();
  
  // Determine layout based on transpose flags
  mem_layout layout_a = mem_layout::row_major;
  // validate_gemm_result expects B as KxN row_major, for NxK row_major we need to send col_major
  mem_layout layout_b = mem_layout::col_major;
  
  // Extract raw pointers from tensor iterators
  TA* A_ref_ptr = &*A_ref.data();
  TB* B_ref_ptr = &*B_ref.data();
  TC* C_ptr = &*C.data();
  
  int err_cnt = validate_gemm_result(A_ref_ptr, B_ref_ptr, C_ptr, m, n, k, layout_a, layout_b);
  printf("Verification: %s\n", (err_cnt == 0) ? "PASSED" : "FAILED");

  if (err_cnt != 0) {
    throw std::runtime_error("GEMM verification failed! Output does not match expected results.");
  }

  return 0;
}
