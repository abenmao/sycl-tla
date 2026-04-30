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
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <string>
#include <type_traits>

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
          class SmemLayoutA>  // (M,K,PIPE)
struct SharedStorage : cute::aligned_struct<SmemAlignment, _0>
{
  cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>, SmemAlignment> smem_A;
};

inline void xe4_syncthreads() {
  auto group = get_nd_item<1>().get_group();
  sycl::group_barrier(group);
}

#define PRINT(x) print(#x ": "); print(x); print("\n");

template <bool EnablePrefetch,
          class ProblemShape, int CopyElements,
          class TA, class SmemLayoutA, class SmemLayoutC, class ADMA_P,
          class ADMA_A, class ADMA_C,
          class TC, class CStride>
void
adma_linear_copy_device(ProblemShape problemSize,
            TA const* A, SmemLayoutA sA_layout, SmemLayoutC sC_layout, ADMA_P adma_prefetch_a,
            ADMA_A adma_load_a, ADMA_C adma_store_c,
            TC* C, CStride dC,
            sycl::nd_item<3> item)
{
  // Preconditions
  static_assert(rank(problemSize) == 1);
  static_assert(is_static<SmemLayoutA>::value);

  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  
  using SharedStorageType = SharedStorage<TA, SmemLayoutA>;
  auto ptr = alloc_slm_buffer<uint8_t, sizeof(SharedStorageType)>(item.get_group());
  auto& smem = *reinterpret_cast<SharedStorageType*>(ptr);
  
  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.begin()), SmemLayoutA{});
  Tensor sC = make_tensor(make_smem_ptr(smem.smem_A.begin()), SmemLayoutC{});

  // ASYNC_LINEAR_LOAD uses raw pointers, not TMA coord tensors
  auto mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(problemSize)));
  auto mC = make_tensor(make_gmem_ptr(C), make_layout(make_shape(problemSize)));

  // Get the appropriate blocks for this thread block
  constexpr auto tile_size = Int<CopyElements>{}; //size(SmemLayoutA{});
  auto cta_coord = make_coord(BlockIdxX());
  Tensor gA = local_tile(mA, make_shape(tile_size), cta_coord, Step<_1>{});  // SRC tile
  Tensor gC = local_tile(mC, make_shape(tile_size), cta_coord, Step<_1>{});  // DST tile

  // No tma_partition for ASYNC_LINEAR_LOAD — it copies the full tile in one shot
  auto tAgA = coalesce(gA);
  auto tAsA = coalesce(sA);
  auto tAgC = coalesce(gC);
  auto tAsC = coalesce(sC);

  // Calculate adma transaction bytes for each operand.
  constexpr int dma_transaction_bytesA = CopyElements * sizeof(TA);
  constexpr int dma_transaction_bytesC = CopyElements * sizeof(TC);

  uint32_t elect_one_thr  = cute::elect_one_sync();  // Elected leader lane (work-item) in a sub-group.
  uint32_t warp_idx = get_sg_id();  // Sub-group = 0 for ADMA load & Sub-group = 1 for ADMA store

  // Allocate abarrier for Load A and Store C.
  auto load_a_abar = allocate_abar<0>();
  auto store_c_abar = allocate_abar<1>();

  // Only lanes (work-items) using abarrier should initialize it.
  if (elect_one_thr && warp_idx == 0) {
    for (int i = 0; i < 1; ++i){
      xe4_initialize_barrier(load_a_abar[i], 1 /*numThreads*/);
    }
  } else if (elect_one_thr && warp_idx == 1) {
    xe4_initialize_barrier(store_c_abar[0], 1 /*numThreads*/);
  }
  xe4_syncthreads();
  
  // Phase barrier bit for store C barrier.
  int store_c_barrier_phase_bit = 0;
  // Dummy cluster mask = 0 for non-cluster based example.
  uint32_t dummy_mask = 0;

  // Using PipelineState API to manage read/write states.
  auto write_state = cutlass::PipelineState<1>();
  auto read_state  = cutlass::PipelineState<1>();

  for (int k_tile = 0; k_tile < 1; k_tile++)
  {
      if ((warp_idx == 0) && elect_one_thr)
      {
        if constexpr (EnablePrefetch) {
          copy(adma_prefetch_a, tAgA, tAsA);
        }
        xe4_set_barrier_transaction_bytes(load_a_abar[0], dma_transaction_bytesA);
        copy(adma_load_a.with(&load_a_abar[0]), tAgA, tAsA);
      }

      sycl::group_barrier(item.get_sub_group());

      if ((warp_idx == 1) && elect_one_thr)
      {
          int read_pipe = read_state.index();
          xe4_wait_barrier(load_a_abar[read_pipe], read_state.phase());
          xe4_set_barrier_transaction_bytes(store_c_abar[0], dma_transaction_bytesC);
          copy(adma_store_c.with(&store_c_abar[0]), tAsC, tAgC);
          xe4_wait_barrier(store_c_abar[0], store_c_barrier_phase_bit);
          store_c_barrier_phase_bit ^= 1;
      }
  }

}

// Setup params for a adma_linear_copy_A_k_major
template <bool EnablePrefetch, class TensorA, class TensorC>
void
adma_linear_copy_A_k_major(uint32_t probSize,
        TensorA const& A,
        TensorC& C,
        sycl::queue& queue)
{
  // Define shapes (dynamic)
  auto prob_shape = probSize;

  // Extract element types and raw pointers from tensors
  using TA = typename TensorA::element_type;
  using TC = typename TensorC::element_type;
  
  TA const* A_ptr = &*A.data();
  TC* C_ptr = &*C.data();
  
  // Extract strides from tensor layouts
  auto dA = A.stride();
  auto dC = C.stride();

  Tensor mA = make_tensor(make_gmem_ptr(A_ptr), make_layout(make_shape(probSize), 1));
  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), make_layout(make_shape(probSize), 1));

  // Define CTA tile sizes (static)
  using TileShape_copy = cute::Shape<cute::_256, cute::_256>;
  using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;
  constexpr auto dim0 = get<0>(TileShape_copy{});
  constexpr auto dim1 = get<1>(TileShape_copy{});
  auto copyElements_per_cta = size(make_shape(dim0, dim1));

  static constexpr auto majorA = cute::AMMA::Major::K;

  using ElementA = TA;
  using ElementB = TA;
  using ElementC = float;
  using ElementAccumulator = float;

  constexpr int copyElements = copyElements_per_cta;
  constexpr int extraSizeFactor = 1; // extra SMEM buffer if needed

  // Define the smem layouts (static) - XE4 row-major layouts
  using SmemLayoutA = decltype(make_layout(Int<extraSizeFactor * copyElements>{}, Int<1>{}));
  using SmemLayoutC = decltype(make_layout(Int<extraSizeFactor * copyElements>{}, Int<1>{}));

  SmemLayoutA sA{};
  SmemLayoutC sC{};

  constexpr auto copyBytesA = copyElements * sizeof(TA);
  constexpr auto copyBytesC = copyElements * sizeof(TC);

#if 0
  PRINT(sA);
  PRINT(sC);
#endif

  using GmemTiledCopyP = cute::Copy_Traits<cute::XE4_ADMA_LINEAR_PREFETCH, cute::Int<copyBytesA>>;
  using GmemTiledCopyA = cute::Copy_Traits<cute::XE4_ADMA_LINEAR_LOAD<>, cute::Int<copyBytesA>>;
  using GmemTiledCopyC = cute::Copy_Traits<cute::XE4_ADMA_LINEAR_STORE<>, cute::Int<copyBytesC>>;

  // Non-cluster case gives effective layout ((_1),_1,_1,_1):((_0),_0,_0,_0)
  //auto cluster_layout_vmnk = tiled_divide(make_layout(ClusterShape_MNK{}), make_tile(typename TiledMma::AtomThrID{}));
  auto cluster_layout_vmnk = tiled_divide(make_layout(ClusterShape_MNK{}), make_tile(_1{}, _1{}, _1{}));
  
  // ADMA_LINEAR_LOAD/STORE are raw-pointer ops, NOT TMA descriptor ops.
  // Use make_tiled_copy (not make_tma_copy) so .with(abar, bytes) routes through
  // XE4_ADMA_LINEAR_COPY_Unpack (4-arg copy) instead of XE4_COPY_Unpack (5-arg).
  auto adma_prefetch_a = make_tiled_copy(
    Copy_Atom<GmemTiledCopyP, TA>{},
    Layout<_1>{},
    Layout<Int<copyElements>>{}
  );

  auto adma_load_a = make_tiled_copy(
    Copy_Atom<GmemTiledCopyA, TA>{},
    Layout<_1>{},
    Layout<Int<copyElements>>{}
  );

  auto adma_store_c = make_tiled_copy(
    Copy_Atom<GmemTiledCopyC, TC>{},
    Layout<_1>{},
    Layout<Int<copyElements>>{}
  );

#if 0
  PRINT(adma_load_a);
  PRINT(adma_store_c);
#endif  
  //
  // Setup and Launch
  //
  constexpr int NumOfControlWarps = 2; // Sub-groups - 0th for ADMA-load, 1st for ADMA-store.
  constexpr int NumOfThreadsPerWarp = 32; // Work-items per sub-group.

  namespace syclexp = sycl::ext::oneapi::experimental;
  auto [cluster_size_y, cluster_size_x, cluster_size_z] = ClusterShape_MNK{};
  sycl::range<3> clusterSize(cluster_size_z, cluster_size_y, cluster_size_x);
  syclexp::properties Props {syclexp::work_groups_per_cluster<3>(clusterSize)};
  
  // Setup thread block configuration
  auto num_groups = ceil_div(probSize, copyElements);
  sycl::range<3> local_range(1, NumOfControlWarps, NumOfThreadsPerWarp);
  sycl::range<3> group_range(1, 1, get<0>(num_groups));  // (Z_tiles, Y_tiles, X_tiles)
  sycl::nd_range<3> Range(group_range * local_range, local_range);

#if 0
  PRINT(num_groups);
  PRINT(local_range[0]);
  PRINT(local_range[1]);
  PRINT(local_range[2]);
  PRINT(group_range[0]);
  PRINT(group_range[1]);
  PRINT(group_range[2]);
#endif

  // Kernel Launch with cluster support
  auto launch_cfg = syclexp::launch_config(Range, Props);
  syclexp::submit_with_event(queue, [&](sycl::handler &handler) {
    syclexp::nd_launch(handler, launch_cfg, [=](sycl::nd_item<3> item) ALWAYS_INLINE {
      adma_linear_copy_device<EnablePrefetch, decltype(prob_shape), copyElements,
                  TA, decltype(sA), decltype(sC), decltype(adma_prefetch_a),
                  decltype(adma_load_a), decltype(adma_store_c),
                  TC, decltype(dC)>(
                  prob_shape,
                  A_ptr, sA, sC, adma_prefetch_a,
                  adma_load_a, adma_store_c,
                  C_ptr, dC, item);
    });
  }).wait();
}


template <bool EnablePrefetch, class TensorA, class TensorC>
void
adma_linear_copy(uint32_t probSize,
     TensorA const& A,
     TensorC& C,
     sycl::queue& queue)
{
  return adma_linear_copy_A_k_major<EnablePrefetch>(probSize, A, C, queue);
}

template <bool EnablePrefetch, class TensorA, class TensorRef, class TensorC>
void run_and_verify(const char* label, uint32_t sizeA, TensorA& A, TensorRef& A_ref, TensorC& C, sycl::queue& queue)
{
  using TA = typename TensorRef::element_type;
  using TC = typename TensorC::element_type;

  zero_fill(C);

  adma_linear_copy<EnablePrefetch>(sizeA, A, C, queue);
  queue.wait_and_throw();

  TA* A_ref_ptr = &*A_ref.data();
  TC* C_ptr = &*C.data();

  int err_cnt = 0;
  for (uint32_t i = 0; i < sizeA; i++) {
    if (A_ref_ptr[i] != C_ptr[i]) {
      printf("  Mismatch at index %u: A_ref = %f, C = %f\n", i, static_cast<float>(A_ref_ptr[i]), static_cast<float>(C_ptr[i]));
      err_cnt++;
    }
  }

  printf("[%s] Verification: %s\n", label, (err_cnt == 0) ? "PASSED" : "FAILED");
  if (err_cnt != 0) {
    throw std::runtime_error("ADMA linear copy verification failed! Output does not match input.");
  }
}

int main(int argc, char** argv)
{
  std::string prefetch_val = "both";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--prefetch=", 0) == 0) {
      prefetch_val = arg.substr(11);
    } else if (arg.rfind("-prefetch=", 0) == 0) {
      prefetch_val = arg.substr(10);
    } else if (arg == "--help" || arg == "-h") {
      printf("Usage: %s [--prefetch=yes|no|both(default)]\n", argv[0]);
      return 0;
    }
  }

  bool run_prefetch    = (prefetch_val == "yes" || prefetch_val == "both");
  bool run_no_prefetch = (prefetch_val == "no"  || prefetch_val == "both");

  if (!run_prefetch && !run_no_prefetch) {
    printf("Unknown --prefetch value '%s'. Use --help for usage.\n", prefetch_val.c_str());
    return 1;
  }

  sycl::queue queue{sycl::gpu_selector_v};

  std::cout << "Running on device: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  int m = 1024;
  int k = 1024;

  using TA = fp16;
  using TC = fp16;

  auto prob_shape = make_shape(m, k);
  uint32_t sizeA = size(prob_shape);

  auto A = make_shared_usm_tensor<TA, 'R'>(queue, 1, sizeA);
  auto C = make_shared_usm_tensor<TC, 'R'>(queue, 1, sizeA);
  auto A_ref = make_shared_usm_tensor<TA, 'R'>(queue, 1, sizeA);

  random_fill(A);
  copy(A, A_ref);
  subbyte_pack(A);

  if (run_prefetch) {
    run_and_verify<true>("with prefetch", sizeA, A, A_ref, C, queue);
  }

  if (run_no_prefetch) {
    run_and_verify<false>("without prefetch", sizeA, A, A_ref, C, queue);
  }

  return 0;
}
