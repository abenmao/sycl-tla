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

////////////////////////////////////////////////////////////////////////////////
#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cute/arch/mma_xe4_amma.hpp>
#include <cute/arch/xe4_async_gmma_slm_layout.hpp>
#include <cute/arch/mma_xe4.hpp>
#include <cute/atom/mma_traits_xe4_amma.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_xe4_eu_copy.hpp>

namespace gemm_eu_copy {

template<typename Item>
inline __attribute__((always_inline)) 
bool is_chosen_leader_in_the_work_group(const Item& it,
	const int subgroup_launching_async_ops=4) {
	// NOTE: change this subgroup id based on the number of control threads and
	// worker threads in your workgroup //
	return sycl::ext::oneapi::this_work_item::get_sub_group().leader() &&
	(it.get_sub_group().get_group_id() == subgroup_launching_async_ops);
}

////////////////////////////////////////////////////////////////////////////////
// GEMM+Epilog(simple bias) : using EU copy matrix atoms
////////////////////////////////////////////////////////////////////////////////
template<typename ProblemShape_MNK, class WorkGroupTilerShape,
         typename TA, typename TB, typename TC, typename TD=TC,
         typename TAlpha=TC, typename TBeta=TC, typename TBias=TC,
         int kSubGroupSize=32, int kNumProducerSubGroups=4,
         int kSubGroupLaunchingAsyncMMA=4>
CUTE_HOST_DEVICE
  void gemm_device(TA const *A, TB const *B, TC const *C, TD *D,
      TBias const *Bias_ptr, const TAlpha& alpha, const TBeta& beta, 
      sycl::nd_item<3> it, sycl::stream out) {

  static_assert(kSubGroupLaunchingAsyncMMA+1 > kNumProducerSubGroups, 
      "Subgroup launching Async MMA must be in one of the consumers");
  static_assert(kNumProducerSubGroups >= 2, 
      "Subgroup launching Async MMA must be in one of the consumers");
  namespace syclex = sycl::ext::intel::experimental;
  using namespace cute;

  ProblemShape_MNK prob_shape;
  auto M = size<0>(prob_shape);
  auto N = size<1>(prob_shape);
  auto K = size<2>(prob_shape);


  // Create a view of tensors in the GMEM //
  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(Int<M>{}, Int<K>{}), 
        make_stride(Int<K>{}, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(Int<N>{}, Int<K>{}),
        make_stride(Int<K>{}, Int<1>{})); // transposed //
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(Int<M>{}, Int<N>{}),
        make_stride(Int<N>{}, Int<1>{})); 
  Tensor mBias = make_tensor(make_gmem_ptr(Bias_ptr), 
      make_shape(Int<M>{}, Int<N>{}), make_stride(_0{}, _1{}) );

  // output //
  Tensor mD = make_tensor(make_gmem_ptr(D), make_shape(Int<M>{}, Int<N>{}),
        make_stride(Int<N>{}, Int<1>{}));

  WorkGroupTilerShape wg_tiler;
  constexpr int bM = size<0>(wg_tiler), bN=size<1>(wg_tiler), 
                bK=size<1>(wg_tiler);

  auto my_sg_id = it.get_sub_group().get_group_id();
  auto my_lane_id = it.get_sub_group().get_local_id(); // within the subgroup //

  // Create a tile view of the global memory tensors in the workgroup//
  auto wg_coord = make_coord(BlockIdxX(), BlockIdxY(), _);
  Tensor gA = local_tile(mA, wg_tiler, wg_coord, Step< _1, X, _1>{});
  Tensor gB = local_tile(mB, wg_tiler, wg_coord, Step< X, _1, _1>{});
  Tensor gC = local_tile(mC, wg_tiler, wg_coord, Step< _1, _1, X>{});
  Tensor gBias = local_tile(mBias, wg_tiler, wg_coord, Step< _1, _1, X>{});
  Tensor gD = local_tile(mD, wg_tiler, wg_coord, Step< _1, _1, X>{});

  //////////////////////////////////////////////////////////////////////////////
  // Shared memory for the tiles//
  auto wg = it.get_group();
  auto smem_mult_ptr_A = 
      sycl::ext::oneapi::group_local_memory_for_overwrite<TA[bM*bK]>(wg);
  TA *smem_A_ptr = sycl::address_space_cast<
    sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        *smem_mult_ptr_A).get();
  auto sA = make_tensor(make_smem_ptr(smem_A_ptr),
        make_layout(make_shape(bM, bK), make_stride(Int<bK>{}, Int<1>{})));

  auto smem_mult_ptr_B = 
      sycl::ext::oneapi::group_local_memory_for_overwrite<TB[bK*bN]>(wg);
  TB *smem_B_ptr = sycl::address_space_cast<
    sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        *smem_mult_ptr_B).get();
  auto sB = make_tensor(make_smem_ptr(smem_B_ptr),
        make_layout(make_shape(bK, bN), make_stride(Int<bK>{}, Int<1>{})));

  // Note C is only needed for the epilog //
  auto smem_mult_ptr_C = 
      sycl::ext::oneapi::group_local_memory_for_overwrite<TC[bM*bN]>(wg);
  TC *smem_C_ptr = sycl::address_space_cast<
    sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        *smem_mult_ptr_C).get();
  auto sC = make_tensor(make_smem_ptr(smem_C_ptr),
        make_layout(make_shape(bM, bN), make_stride(Int<bN>{}, Int<1>{})));

  auto smem_mult_ptr_D = 
      sycl::ext::oneapi::group_local_memory_for_overwrite<TD[bM*bN]>(wg);
  TD *smem_D_ptr = sycl::address_space_cast<
    sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        *smem_mult_ptr_D).get();
  auto sD = make_tensor(make_smem_ptr(smem_D_ptr),
        make_layout(make_shape(bM, bN), make_stride(Int<bN>{}, Int<1>{})));
  //////////////////////////////////////////////////////////////////////////////
  namespace xe4_type1_kmajor = cute::xe4::slm::type1::kmajor;


  constexpr int ktile_count  = size<2>(gA);
  int ktile_idx = 0; 

  if (my_sg_id < kNumProducerSubGroups) { // producers //
    smem_D_ptr = 0; smem_C_ptr = 0;

    while (ktile_idx < ktile_count) {
      if (ktile_idx > 0) {
        sycl::group_barrier(it.get_group());
      }

      if (my_sg_id==0) { // copy the tile from gmem to slm in one subgroup
        auto gmem_tensor = gA(_, _, ktile_idx);
        auto smem_tensor = make_tensor(sA.data(),
            xe4::slm::type1::kmajor::make_slm_layout_elem<TA, bM, bK>());
        auto tiled_copy =
            make_xe4_g2s_tiled_copy<TA, bM, bK>(gmem_tensor, smem_tensor);
        auto thr_copy   = tiled_copy.get_thread_slice((int) my_lane_id);
        auto tSrc = thr_copy.partition_S(gmem_tensor);
        auto tDst = thr_copy.partition_D(smem_tensor);

        copy(tiled_copy, tSrc, tDst);
      } else if (my_sg_id==1) {
        auto gmem_tensor = gB(_, _, ktile_idx);
        auto smem_tensor = make_tensor(sB.data(),
            xe4::slm::type1::kmajor::make_slm_layout_elem<TB, bN, bK>());
        auto tiled_copy =
            make_xe4_g2s_tiled_copy<TB, bM, bK>(gmem_tensor, smem_tensor);
        auto thr_copy   = tiled_copy.get_thread_slice((int) my_lane_id);
        auto tSrc = thr_copy.partition_S(gmem_tensor);
        auto tDst = thr_copy.partition_D(smem_tensor);
        copy(tiled_copy, tSrc, tDst);
      }

      sycl::group_barrier(it.get_group());
      ++ktile_idx;
    }
    sycl::group_barrier(it.get_group());
  } else {
     // STEP-3: define the necessary MMA atom/ops 
     // async GMMA op with tracking on D completion//
     typedef cute::XE4_AMMA_D<TC, TA, TB, TC, bM, bN, bK,
               cute::AMMA::Major::K, cute::AMMA::Major::K> 
                  xe4_gmma_op_dtrack_t;
     auto sADesc = cute::AMMA::make_matrix_desc<cute::AMMA::Major::K>(sA);
     auto sBDesc = cute::AMMA::make_matrix_desc<cute::AMMA::Major::K>(sB);
     auto sDDesc = cute::AMMA::make_matrix_desc<cute::AMMA::Major::K>(sD);

    uint64_t mma_ctrl = 0x100;
    int mma_barrier_phase_bit = 0;
    uint64_t *abarrier_systolic = allocate_abar<1>();

    // Main Loop: issues GMMAs to systolic //
    #pragma unroll
    while (ktile_idx < ktile_count) {
      sycl::group_barrier(it.get_group());
      if (is_chosen_leader_in_the_work_group(it, kSubGroupLaunchingAsyncMMA)) {
        abarrier_init(abarrier_systolic,  1); 
        xe4_set_barrier_transaction_bytes(*abarrier_systolic, 1);
        xe4_gmma_op_dtrack_t::fma(MMAControl(mma_ctrl),
            sDDesc, sADesc, sBDesc, sDDesc, abarrier_systolic);
        mma_ctrl = 0x000;
        abarrier_try_wait(abarrier_systolic, mma_barrier_phase_bit);
        mma_barrier_phase_bit ^= 1;
      }
      sycl::group_barrier(it.get_group());
      ++ktile_idx;
    }

    ////////////////////////////////////////////////////////////////////////////
    ///////////////////////////// Epilogue /////////////////////////////////////
    auto tiled_copy_reg_layout =
        make_xe4_reg_layout_for_tiled_copy<TD, bM, bN>();
    auto rD = make_tensor<TD>(tiled_copy_reg_layout);
    auto rC = make_tensor<TC>(tiled_copy_reg_layout);
    auto rBias = make_tensor<TBias>(make_shape(Int<bN>{}));

    static_assert(is_rmem<decltype(rBias)>::value, "rBias is not in registers"); 
    static_assert(is_rmem<decltype(rD)>::value, "rD is not in registers"); 
    static_assert(is_rmem<decltype(rC)>::value, "rC is not in registers"); 
    static_assert(rank(tiled_copy_reg_layout) == _2{}, 
          "register rank invariant failed");
    static_assert(size<1>(tiled_copy_reg_layout) == Int<bN>{}, 
          "register row width invariant failed");

    {  // Copy MMA output into registers (from SLM) of each lane //
      auto slm_layout = 
            xe4::slm::type1::kmajor::make_slm_layout_elem<TD, bM, bN>();
      auto slm_tensor = make_tensor(sD.data(), slm_layout);

      auto tiled_copy =
          make_xe4_s2r_tiled_copy<TD, bM, bN>(slm_tensor, rD);
      auto thr_copy = tiled_copy.get_thread_slice((int) my_lane_id);

      auto tSrc = thr_copy.partition_S(slm_tensor);
      copy(tiled_copy, tSrc, rD);
    }


    { // Copy C into registers of each lane (workitem) //
      auto tiled_copy = make_xe4_g2r_tiled_copy<TC, bM, bN>();
      auto thr_copy = tiled_copy.get_thread_slice((int) my_lane_id);
      auto tSrc = thr_copy.partition_S(gC);
      copy(tiled_copy, tSrc, rC);
    }

    // All lanes use the same row bias //
    copy(gBias(0, _), rBias);

    // Row wise application of epilogue //
    CUTE_UNROLL
    for (int m=0; m<size<0>(tiled_copy_reg_layout); m++) {
      CUTE_UNROLL
      for (int k=0; k<bN; k++) {
        rD(m,k) = (alpha*rD(m,k)) + (beta*rC(m,k)) + rBias(k);
      }
    }

    { // Copy registers in each lane back to gmem //
      auto tiled_copy = make_xe4_r2g_tiled_copy<TD, bM, bN>();
      auto thr_copy = tiled_copy.get_thread_slice((int) my_lane_id);
      auto tDst = thr_copy.partition_D(gD);
      copy(tiled_copy, rD, tDst);
    }
    ////////////////////////////////////////////////////////////////////////////
  }
}

} // namespace gemm_eu_copy //
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

  // Check: recompute A*B and compare with provided C
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

int main(int argc, char **argv) {
  // problem shape //
  constexpr int m=256, n=256, k=256;
  // tile shape //
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
  usm_vector_t<TBias> dataBias(n, alloc_TBias); // bias row //

  unsigned int rseed = 4239753476;
  unsigned int range = 256;
  srand(rseed);
  // K-major input //
  for (size_t i=0; i<m; i++) {
    for (size_t j=0; j<k; j++) {
      dataA[i*k + j] = ((unsigned int) rand())%range;
    }
  }

  // N-major input (transposed) //
  int count=0;
  for (size_t j=0; j<k; j++) {
    for (size_t i=0; i<n; i++) {
      dataB[i*k + j] = ((unsigned int) rand())%range; 
    }
  }

  for (size_t i=0; i<m; i++) {
    for (size_t j=0; j<n; j++) {
      dataD[i*n + j] = 0;
      dataC[i*n + j] = ((unsigned int) rand())%range; 
    }
  }

  for (size_t j=0; j<n; j++) {
    dataBias[j] = ((unsigned int) rand())%range; 
  }

  TAlpha alpha = TAlpha(1.0);
  TBeta beta = TBeta(5.0);


  // STEP-1: define shapes (dynamic)
  using namespace cute;
  Shape prob_shape = make_shape(Int<m>{}, Int<n>{}, Int<k>{});

  // STEP-2: WorkGroup/ThreadBlock(CTA) partitioning //
  auto bM = min(Int<m>{}, Int<tM>{});
  auto bN = min(Int<n>{}, Int<tN>{});
  auto bK = min(Int<k>{}, Int<tK>{});
  // cta_tiler //
  Shape wg_tile_shape = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{}); 
  Shape work_groups_shape = ceil_div(prob_shape, wg_tile_shape);

  constexpr size_t workgroup_size = 384; // or 384 //
  sycl::range<3> local_range(1, 1, workgroup_size);
  sycl::range<3> group_range(1,
        get<1>(work_groups_shape), get<0>(work_groups_shape));
  sycl::nd_range<3> global_range(group_range * local_range, local_range);

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
              {
              gemm_eu_copy::gemm_device<
                  decltype(prob_shape),
                  decltype(wg_tile_shape),
                  TA, TB, TC>(A_ptr, B_ptr, C_ptr, D_ptr, Bias_ptr, alpha, beta,
                      it, out);
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

 std::cout << "=======TEST RESULT========" << std::endl;
  //std::cout << D << std::endl;

 // undo the transpose //
 for (int j=0; j<k; j++) {
   for (int i=j; i<n; i++) {
     TB v = B(i, j);
     B(i, j) = B(j, i);
     B(j, i) = v;
   }
 }

 return compare_device_gemm_with_host_gemm(A, B, C, D, Bias, alpha, beta);
}
