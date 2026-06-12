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
#include <cute/arch/xe4_util.hpp>
#include "cutlass/arch/barrier.h"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/kernel_hardware_info.h"

#include "../../../common/sycl_cute_common.hpp"
#include "../../../xe4/utils/validation.hpp"

using namespace cute;

using sycl::ext::oneapi::this_work_item::get_nd_item;

constexpr static size_t SmemAlignment = 128;

template <class ElementA,
          class ElementB,
          class ElementC,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB,  // (N,K,P)
          class SmemLayoutC>  // (M,N,P)
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

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class CSmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class TiledCopyC,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, CSmemLayout sC_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, TiledCopyC copy_c,
            TC      * C, CStride dC,
            Alpha alpha, Beta beta, sycl::nd_item<3> item)
{
  TiledMma mma{};
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN
  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(BlockIdxX(), BlockIdxY(), _);  // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  using SharedStorageType = SharedStorage<TA, TB, TC, ASmemLayout, BSmemLayout, CSmemLayout>;
  auto ptr = alloc_slm_buffer<uint8_t, sizeof(SharedStorageType)>(item.get_group());
  auto& smem = *reinterpret_cast<SharedStorageType*>(ptr);
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.begin()), ASmemLayout{});                           // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.begin()), BSmemLayout{});                           // (BLK_N,BLK_K,PIPE)
  Tensor sC = make_tensor(make_smem_ptr(smem.smem_C.begin()), CSmemLayout{});                           // (BLK_M,BLK_N,PIPE)

  //
  // Partition the copying of A and B tiles across the threads
  //

  ThrCopy thr_copy_a = copy_a.get_slice(ThreadIdxX());
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(ThreadIdxX());
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

  ThrMMA thr_mma = mma.get_slice(ThreadIdxX());
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCsC = thr_mma.partition_C(sC);                               // (MMA,MMA_M,MMA_N)
  Tensor tCgC = thr_mma.partition_C(gC);  
  
  // Create fragments for A/B/C in SLM to store in Registers, 
  // which represent layouts of corresponding data in SLM. (Xe4 specific)
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCsC);

#if 1
  // Total number of k-tiles
  auto K_TILE_MAX  = size<3>(tAgA);
  // Number of pipelined k-tiles in smem
  auto K_PIPE_MAX  = size<3>(tAsA);

  // Prefetch all but the last to prefill pipelines before AMMA
  CUTE_UNROLL
  for (int k = 0; k < K_PIPE_MAX-1; ++k)
  { 
    copy(copy_a, tAgA(_,_,_,k), tAsA(_,_,_,k));
    copy(copy_b, tBgB(_,_,_,k), tBsB(_,_,_,k));
  }

  // clear accumulators in SLM
  clear(tCsC);
  xe4_syncthreads();

  // Current pipe to read from
  int k_pipe_read  = 0;

  // Current pipe to write to
  int k_pipe_write = K_PIPE_MAX-1;

  uint32_t elect_one_thr  = cute::elect_one_sync();  // Elected leader from sub-group. (0th work-item)
  uint32_t elect_one_warp = (ThreadIdxX() / 32 == 0);  // Elected 0th sub-group from each work-group to call AMMA.
  auto local_id = item.get_local_linear_id();

  // Only threads that will use abarrier should initialize it.
  // For Xe4, only the leader work-item from chosen sub-group (here 0th sub-group) will call AMMA.
  auto& mma_barrier = cutlass::arch::allocate_cluster_tx_barrier();
  if (elect_one_thr && elect_one_warp) {
    mma_barrier.init(1);
  }
  xe4_syncthreads();
  int mma_barrier_phase_bit = 0;  // Each barrier has an associated phase_bit.
  
  uint64_t mma_ctrl = 0x100;
  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    int k_tile_next = k_tile + (K_PIPE_MAX-1);
    k_tile_next = (k_tile_next >= K_TILE_MAX) ? K_TILE_MAX-1 : k_tile_next;
    //
    // Copy gmem to smem for k_tile_write
    //
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe_write));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe_write));
    // Advance k_pipe_write
    ++k_pipe_write;
    k_pipe_write = (k_pipe_write == K_PIPE_MAX) ? 0 : k_pipe_write;
    //
    // Compute on k_tile
    //
    sycl::group_barrier(item.get_group());
    // (V,M,K) x (V,N,K) => (V,M,N) - XE4 AMMA gemm call from the elected sub-group (warp) & elected lane (work-item)
    if (elect_one_thr && elect_one_warp) {
      mma_barrier.arrive_and_expect_tx(1);
      auto new_mma = mma.with(AMMA::TrackMethod<AMMA::Tracking::D>{}, mma_ctrl, reinterpret_cast<uint64_t*>(&mma_barrier));
      cute::gemm(new_mma, tCrA(_,_,_,k_pipe_read), tCrB(_,_,_,k_pipe_read), tCrC);
      mma_ctrl = 0x000;
      mma_barrier.try_wait(mma_barrier_phase_bit);
      mma_barrier_phase_bit ^= 1;
    }
    sycl::group_barrier(item.get_group());

    // Advance k_pipe_read
    ++k_pipe_read;
    k_pipe_read = (k_pipe_read == K_PIPE_MAX) ? 0 : k_pipe_read;
    
  } 
  xe4_syncthreads();
  
  //
  // Epilogue
  //
  copy(copy_c, tCsC, tCgC);
  //axpby(alpha, tCrC, beta, tCgC);
#endif
  
}

// Setup params for a TN GEMM using SYCL
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
  // Define shapes 
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

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

  // Define CTA tile sizes (static) - XE4 128x128x128 block
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<128>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  constexpr auto bP = Int<3>{};  // Pipeline Stages

  // Define the smem layouts (static) - Simple K-major layouts for TN
  static constexpr auto majorA = cute::AMMA::Major::K;
  static constexpr auto majorB = cute::AMMA::Major::K;

  // Define Tiled copy atoms using UniversalCopyAtom with k-major layouts.
  TiledCopy copyA = make_tiled_copy(
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, TA>{},
    Layout<Shape<_32,_32>, Stride<_32,_1>>{},  
    Layout<Shape<_4,_4>>{}
  );

  TiledCopy copyB = make_tiled_copy(
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, TB>{},
    Layout<Shape<_32,_32>, Stride<_32,_1>>{},
    Layout<Shape<_4,_4>>{}
  );

  TiledCopy copyC = make_tiled_copy(
    Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, TC>{},
    Layout<Shape<_32,_32>, Stride<_32,_1>>{},
    Layout<Shape<_2,_2>, Stride<_2,_1>>{}
  );
  using ElementAccumulator = float;
  
  using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using TileShape_MNK = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
  // Use element types from tensors
  using ElementA = TA;
  using ElementB = TB;
  using ElementC = TC;

  using TiledMma = decltype(cute::make_tiled_mma(
    cute::AMMA::ss_op_selector<
    ElementAccumulator,
    ElementA, ElementB,
    ElementC,
    decltype(cute::product_each(TileShape_MNK{})),
    ClusterShape_MNK, majorA, majorB>()
  ));
  
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
  
  // Define SmemLayouts specific to k-major and adding pipeline dimension for A/B.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
      cute::conditional_t<majorA == cute::AMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
      cute::conditional_t<majorB == cute::AMMA::Major::K, Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})), Step<_2,_1>{}));

  SmemLayoutA sA{};
  SmemLayoutB sB{};
  SmemLayoutC sC{};

  //
  // Setup and Launch
  //

  namespace syclexp = sycl::ext::oneapi::experimental;
  auto [cluster_size_y, cluster_size_x, cluster_size_z] = ClusterShape_MNK{};
  sycl::range<3> clusterSize(cluster_size_z, cluster_size_y, cluster_size_x);
  syclexp::properties Props {syclexp::work_groups_per_cluster<3>(clusterSize)};
  
  // Setup thread block configuration
  auto num_groups = ceil_div(prob_shape, TileShape_MNK{});
  sycl::range<3> local_range(1, 1, 32*32);
  sycl::range<3> group_range(1, get<1>(num_groups), get<0>(num_groups));  // (Z, Y=N_tiles, X=M_tiles)
  sycl::nd_range<3> Range(group_range * local_range, local_range);

  // Kernel Launch with cluster support
  auto launch_cfg = syclexp::launch_config(Range, Props);
  syclexp::submit_with_event(queue, [&](sycl::handler &handler) {
    syclexp::nd_launch(handler, launch_cfg, [=](sycl::nd_item<3> item) ALWAYS_INLINE {
      gemm_device<decltype(prob_shape), decltype(cta_tiler),
                  TA, decltype(dA), decltype(sA), decltype(sC), decltype(copyA),
                  TB, decltype(dB), decltype(sB), decltype(copyB), decltype(copyC),
                  TC, decltype(dC), TiledMma,
                  Alpha, Beta>(
                  prob_shape, cta_tiler,
                  A_ptr, dA, sA, sC, copyA,
                  B_ptr, dB, sB, copyB, copyC,
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
  // if (transA == 'N' && transB == 'T') {
  //   return gemm_nt(m, n, k, alpha, A, B, beta, C, queue);
  // } // else
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

  auto prob_shape = make_shape(m, n, k); // (256, 256, 256)

  uint32_t sizeA = size(select<0,2>(prob_shape));
  uint32_t sizeB = size(select<1,2>(prob_shape));
  uint32_t sizeC = size(select<0,1>(prob_shape));

  auto A = make_shared_usm_tensor<TA, 'R'>(queue, m, k);
  auto B = make_shared_usm_tensor<TB, 'R'>(queue, n, k);
  auto C = make_shared_usm_tensor<TC, 'R'>(queue, m, n);

  one_fill(A);
  one_fill(B);
  zero_fill(C);

  bool ok = false;
  
  auto A_ref = make_shared_usm_tensor<float, 'R'>(queue, m, k);
  auto B_ref = make_shared_usm_tensor<float, 'R'>(queue, n, k);

  copy(A, A_ref);
  copy(B, B_ref);

  subbyte_pack(A);
  subbyte_pack(B);
  
  int ldA = 0, ldB = 0, ldC = m;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }

  // Run once
  gemm(transA, transB, m, n, k,
       alpha,
       A, B,
       beta,
       C, queue);
  queue.wait_and_throw();
  // Verify correctness// Determine layout based on transpose flags
  mem_layout layout_a = mem_layout::row_major;
  mem_layout layout_b = mem_layout::row_major;
  
  // Extract raw pointers from tensor iterators
  float* A_ref_ptr = &*A_ref.data();
  float* B_ref_ptr = &*B_ref.data();
  float* C_ptr = &*C.data();
  
  int err_cnt = validate_gemm_result(A_ref_ptr, B_ref_ptr, C_ptr, m, n, k, layout_a, layout_b);
  printf("Verification: %s\n", (err_cnt == 0) ? "PASSED" : "FAILED");

  if (err_cnt != 0) {
    throw std::runtime_error("GEMM verification failed! Output does not match expected results.");
  }
  
  return 0;
}
