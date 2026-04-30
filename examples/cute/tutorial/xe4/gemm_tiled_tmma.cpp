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


#include <vector>
#include <cstdlib>
#include <random>

////////////////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>

#include "gemm_tiled_tmma.hpp" 



namespace gemm_tmm {

////////////////////////////////////////////////////////////////////////////////
// GEMM TMM via CuTe TiledMMA
////////////////////////////////////////////////////////////////////////////////
template<typename ProblemShape_MNK, 
         class WorkGroupTilerShape, typename CTA_Warp_Layout,
         typename MMA_Op, typename Copy_Op,
         typename TA, typename TB, typename TC, typename TD=TC,
         typename TAlpha=TC, typename TBeta=TC, typename TBias=TC,
         int kSubGroupSize=32, int kNumProducerSubGroups=2,
         int kWorkGroupSize=96>
CUTE_HOST_DEVICE
  void gemm_device(TA const *A, TB const *B, TC const *C, TD *D,
      TBias const *Bias_ptr, const TAlpha& alpha, const TBeta& beta,
      sycl::nd_item<3> it, sycl::stream out) {

  using namespace cute;

  ProblemShape_MNK prob_shape;
  auto M = size<0>(prob_shape);
  auto N = size<1>(prob_shape);
  auto K = size<2>(prob_shape);

  // Global-memory tensor views (identical to AMMA version)
  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(Int<M>{}, Int<K>{}),
        make_stride(Int<K>{}, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(Int<N>{}, Int<K>{}),
        make_stride(_1{}, Int<N>{}));  // transposed
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(Int<M>{}, Int<N>{}),
        make_stride(Int<N>{}, Int<1>{}));
  Tensor mBias = make_tensor(make_gmem_ptr(Bias_ptr),
      make_shape(Int<M>{}, Int<N>{}), make_stride(_0{}, _1{}) );
  Tensor mD = make_tensor(make_gmem_ptr(D), make_shape(Int<M>{}, Int<N>{}),
        make_stride(Int<N>{}, Int<1>{}));

  WorkGroupTilerShape wg_tiler;
  constexpr int bM = size<0>(wg_tiler), bN = size<1>(wg_tiler),
                bK = size<2>(wg_tiler);

  auto my_sg_id   = it.get_sub_group().get_group_id();
  auto my_lane_id = it.get_sub_group().get_local_id();

  // Tile the global tensors for this workgroup
  auto wg_coord = make_coord(BlockIdxX(), BlockIdxY(), _);
  Tensor gA    = local_tile(mA,    wg_tiler, wg_coord, Step< _1, X, _1>{});
  Tensor gB    = local_tile(mB,    wg_tiler, wg_coord, Step< X, _1, _1>{});
  Tensor gC    = local_tile(mC,    wg_tiler, wg_coord, Step< _1, _1, X>{});
  Tensor gBias = local_tile(mBias, wg_tiler, wg_coord, Step< _1, _1, X>{});
  Tensor gD    = local_tile(mD,    wg_tiler, wg_coord, Step< _1, _1, X>{});

  //////////////////////////////////////////////////////////////////////////////
  // Shared memory — only A and B tiles needed
  // REMOVED: sC and sD SLM allocations — TMM accumulates in registers!
  //////////////////////////////////////////////////////////////////////////////
  auto wg = it.get_group();

  auto smem_mult_ptr_A =
      sycl::ext::oneapi::group_local_memory_for_overwrite<TA[bM*bK]>(wg);
  TA *smem_A_ptr = sycl::address_space_cast<
    sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        *smem_mult_ptr_A).get();
  auto sA = make_tensor(make_smem_ptr(smem_A_ptr),
        make_layout(make_shape(Int<bM>{}, Int<bK>{}), 
            make_stride(Int<bK>{}, Int<1>{})));

  auto smem_mult_ptr_B =
      sycl::ext::oneapi::group_local_memory_for_overwrite<TB[bN*bK]>(wg);
  TB *smem_B_ptr = sycl::address_space_cast<
    sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        *smem_mult_ptr_B).get();
  auto sB = make_tensor(make_smem_ptr(smem_B_ptr),
        make_layout(make_shape(Int<bN>{}, Int<bK>{}), 
            make_stride(_1{}, Int<bN>{})));

  auto smem_mult_ptr_D = 
      sycl::ext::oneapi::group_local_memory_for_overwrite<TD[bM*bN]>(wg);
  TD *smem_D_ptr = sycl::address_space_cast<
    sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        *smem_mult_ptr_D).get();
  auto sD = make_tensor(make_smem_ptr(smem_D_ptr), 
        make_layout(make_shape(Int<bM>{}, Int<bN>{}),
          make_stride(Int<bN>{}, Int<1>{})));

  //////////////////////////////////////////////////////////////////////////////

  constexpr int ktile_count = size<2>(gA);
  int ktile_idx = 0;

  if (my_sg_id < kNumProducerSubGroups) {

    if (my_sg_id == 0) {
      copy(gC, sD);
    }

    while (ktile_idx < ktile_count) {
      if (ktile_idx > 0) {
        sycl::group_barrier(it.get_group());
      }
      if (my_sg_id == 0) {
        copy(gA(_, _, ktile_idx), sA);
      } else if (my_sg_id == 1) {
        copy(gB(_, _, ktile_idx), sB);
      } 
      sycl::group_barrier(it.get_group());
      ++ktile_idx;
    }
    sycl::group_barrier(it.get_group());
  } else {
    using cta_warp_layout_t = CTA_Warp_Layout;
    using xe4_tmma_op_t = MMA_Op;
    using copy_op_t = Copy_Op;

    #pragma unroll
    while (ktile_idx < ktile_count) {
      // Wait for producers to finish copying this k-tile
      sycl::group_barrier(it.get_group());

      tmma_gemm<bM, bN, bK, 
        xe4_tmma_op_t, cta_warp_layout_t, copy_op_t,
        kNumProducerSubGroups>(sA, sB, sD, it/*, out*/);

      sycl::group_barrier(it.get_group());
      ++ktile_idx;
    }
    copy(sD, gD);
  }
  sycl::group_barrier(it.get_group());
}

} // namespace gemm_tmm
////////////////////////////////////////////////////////////////////////////////

typedef cute::half_t half_t;
template<typename T>
using usm_allocator_t = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template<typename T>
using usm_vector_t = std::vector<T, usm_allocator_t<T>>;


template <typename T>
bool almost_equal(T a, T b, double rel_error=0.01) { // 1% relative error //
  return (a == b) ||
    ((std::fabs(double(a) - double(b))/std::fabs(double(a))) < rel_error);
}

template<typename TensorA, typename TensorB, typename TensorC,
         typename TensorD, typename TensorBias,
         typename TAlpha, typename TBeta>
int compare_device_gemm_with_host_gemm(const TensorA& A, const TensorB& B,
      const TensorC& C, const TensorD& D, const TensorBias& Bias,
      const TAlpha& alpha, const TBeta& beta) {

  using TC = typename TensorC::value_type;

  int M = cute::size<0>(A), K = cute::size<1>(A), N = cute::size<0>(B);

  bool ok = true;
  int bad_count = 0;

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      TC acc = TC(0);
      for (int k = 0; k < K; ++k) {
        acc += A(m, k) * B(k, n);
      }
      acc = alpha*acc + beta*C(m, n) + Bias(m, n);

      TC got = D(m, n);

      if (!almost_equal(acc, got)) {
        ok = false;
        if (bad_count < 10) {
          std::cerr << "Mismatch at (m=" << m << ", n=" << n << "): "
                    << "A*B=" << acc << " but D=" << got << "\n";
        }
        ++bad_count;
      }
    }
  }

  if (ok) {
    std::cout << "PASS: alpha*(A*B) + beta*C + Bias == D for (" << M << "x" << K << ") * ("
              << K << "x" << N << ") -> (" << M << "x" << N << ")\n";
    return 0;
  } else {
    std::cout << "FAIL: " << bad_count << " mismatches found\n";
    return 1;
  }
}

template<typename T>
void fill_random_into_vector(T& matrix) {
    // 1. Initialize the random engine with a fixed seed for reproducibility
    std::mt19937 gen(42); 
    
    // 2. Define the distribution for FP32 between 0.0 and 1.0
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // 3. Generate FP32 and cast to FP16 (sycl::half)
    for (size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] = static_cast<typename T::value_type >(dist(gen));
    }
}

int main(int argc, char **argv) {
  // problem shape
  constexpr int m=256, n=256, k=256;
  constexpr int tM=64, tN=64, tK=64;
  using TA = fp16;
  using TB = fp16;
  using TC = float;
  using TD = float;
  using TAlpha = float;
  using TBeta = float;
  using TBias = float;


  sycl::queue q;
  usm_allocator_t<TA> alloc_TA(q);
  usm_allocator_t<TB> alloc_TB(q);
  usm_allocator_t<TC> alloc_TC(q);
  usm_allocator_t<TD> alloc_TD(q);
  usm_allocator_t<TBias> alloc_TBias(q);

  usm_vector_t<TA> dataA(m*k, alloc_TA);
  usm_vector_t<TB> dataB(k*n, alloc_TB);
  usm_vector_t<TC> dataC(m*n, alloc_TC);
  usm_vector_t<TD> dataD(m*n, alloc_TD);
  usm_vector_t<TBias> dataBias(n, alloc_TBias);

	//fill_random_into_vector(dataA);
  //fill_random_into_vector(dataB);

  // Deterministic value of C[i,j] = K*i*j //
  // A[i,j] = i //
  for (size_t i=0; i<m; i++) {
    for (size_t j=0; j<k; j++) {
      dataA[i*k + j] = i;
    }
  }

  // B[i,j] = j //
  for (size_t i=0; i<k; i++) {
    for (size_t j=0; j<n; j++) {
      dataB[i*n + j] = j; 
    }
  }

  //TODO(vamsikku): bias is not yet suported //
  for (size_t i=0; i<m; i++) {
    for (size_t j=0; j<n; j++) {
      dataD[i*n + j] = 0; 
      dataC[i*n + j] = 0; //((unsigned int) rand())%range;
    }
  }

  for (size_t j=0; j<n; j++) {
    dataBias[j] = 0; //((unsigned int) rand())%range;
  }

  TAlpha alpha = TAlpha(1.0);
  TBeta beta = TBeta(0.0);


  using namespace cute;
  Shape prob_shape = make_shape(Int<m>{}, Int<n>{}, Int<k>{});

  auto bM = min(Int<m>{}, Int<tM>{});
  auto bN = min(Int<n>{}, Int<tN>{});
  auto bK = min(Int<k>{}, Int<tK>{});
  Shape wg_tile_shape = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});
  Shape work_groups_shape = ceil_div(prob_shape, wg_tile_shape);


  constexpr int kWorkGroupSize = 320;
  constexpr int kSubGroupSize = 32;
  constexpr int kNumProducerSubGroups = 2;
  constexpr size_t workgroup_size = kWorkGroupSize;
  sycl::range<3> local_range(1, 1, workgroup_size);
  sycl::range<3> group_range(1,
        get<1>(work_groups_shape), get<0>(work_groups_shape));
  sycl::nd_range<3> global_range(group_range * local_range, local_range);


  //////////////////////////////////////////////////////////////////////////////
  // Pick CuTe Atoms/Ops //
  // XE4IP-BUG(vamsikku): accuracy issues if the number of atoms increase a
  // lot can be verified by TMM_ATOM_N=1
  constexpr int TMM_ATOM_N = 8;
  using xe4_tmma_op_t = XE4_TMM<TD, TA, TB, TC, TMM_ATOM_N>; 
  using copy_op_t = UniversalCopy<uint32_t, uint32_t>;
  //////////////////////////////////////////////////////////////////////////////

  // Choose a Subgroup/Warp layout on the CTA tile to partition  the work across
  // subgroups in the workgroup. This should be  based on  available warps at 
  // the workgroup level which can run in parallel. In this case we are using
  // workgroup size is 320 (sg=32) so 10 subgroups out of which 2 are producer
  // so we have 8 subgroups which run the CTA level MMA since (atom = 32x8x16)
  // we break the 64x64 C tile matrix into 2x4 warp layout.
  //
  auto cta_warp_layout = make_layout(make_shape (_2{}, _4{}, _1{}), 
                                         make_stride(_1{}, _2{}, _0{}));
  using cta_warp_layout_t = decltype(cta_warp_layout);
  // NOTE: you can also use the following utility which automatically picks
  // the warp layout for you see test/unit/xe4/tiled_mma_utils.hpp
  // typename default_warp_layout_for_tiled_mma<xe4_tmma_op_t,
  //   bM, bN, bK, kWorkGroupSize, kSubGroupSize, kNumProducerSubGroups>::type; 

  cta_warp_layout_t cta_layout;
  std::cout << "warp_layout:" << cta_layout << std::endl;

  TA *A_ptr = dataA.data();
  TB *B_ptr = dataB.data();
  TC *C_ptr = dataC.data();
  TD *D_ptr = dataD.data();
  TBias *Bias_ptr = dataBias.data();

  q.submit(
      [&](sycl::handler &h) {
        auto out = sycl::stream(1024*1024, 1024, h);
        h.parallel_for(global_range,
           [=](sycl::nd_item<3> it)
            [[sycl::reqd_work_group_size(1, 1, workgroup_size)]]
            [[sycl::reqd_sub_group_size(kSubGroupSize)]]
              {
              gemm_tmm::gemm_device<
                  decltype(prob_shape),
                  decltype(wg_tile_shape), cta_warp_layout_t,
                  xe4_tmma_op_t, copy_op_t,
                  TA, TB, TC, TC, TAlpha, TBeta, TBias,
                    kSubGroupSize, kNumProducerSubGroups, kWorkGroupSize>(
                      A_ptr, B_ptr, C_ptr, D_ptr, Bias_ptr, alpha, beta,
                      it , out);
              }
        );
      }
  ).wait();


 Tensor A = make_tensor(make_gmem_ptr(dataA.data()),
     make_shape(Int<m>{}, Int<k>{}), make_stride(Int<k>{}, Int<1>{}));
 Tensor B = make_tensor(make_gmem_ptr(dataB.data()),
     make_shape(Int<k>{}, Int<n>{}), make_stride(Int<n>{}, Int<1>{}));
 Tensor C = make_tensor(make_gmem_ptr(dataC.data()),
     make_shape(Int<m>{}, Int<n>{}), make_stride(Int<n>{}, Int<1>{}));;
 Tensor D = make_tensor(make_gmem_ptr(dataD.data()),
     make_shape(Int<m>{}, Int<n>{}), make_stride(Int<n>{}, Int<1>{}));;
 Tensor Bias = make_tensor(make_gmem_ptr(dataBias.data()),
     make_shape(Int<m>{}, Int<n>{}), make_stride(_0{}, _1{}));

#if 0
 std::cout << "=====A=======" << std::endl;
 std::cout << A << std::endl;

 std::cout << "=====B=======" << std::endl;
 std::cout << B << std::endl;

 std::cout << "=====D=======" << std::endl;
 std::cout << D << std::endl;
#endif



 std::cout << "=======TEST RESULT========" << std::endl;
 return compare_device_gemm_with_host_gemm(A, B, C, D, Bias, alpha, beta);
}
