/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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

#include "cutlass_unit_test.h"
#include <iostream>
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_xe4_ldsm.hpp>


using namespace cute;
namespace sc = compat;
namespace sc_exp = compat::experimental;
namespace sycl_ext = sycl::ext::oneapi::experimental;

template <typename T, typename CopyOp, class SmemLayout, class TLayout, class VLayout,
	  typename TiledAdma, typename CTile>
CUTLASS_GLOBAL void
ldsm_test_device_cute(T* g_in, T* g_out,
		      TLayout t_layout,
		      VLayout v_layout,
		      SmemLayout smem_layout,
	              TiledAdma adma_load, CTile ctile,
		      sycl::local_ptr<char> base_smem)
{
  using namespace cute;
  auto smem = reinterpret_cast<T*>((char*)base_smem);

  if (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group_linear_id() == 0)
  {
    auto t_g_in  = make_tensor(make_gmem_ptr(g_in),  smem_layout);
    auto t_g_out = make_tensor(make_gmem_ptr(g_out), smem_layout);
    auto t_smem  = make_tensor(make_smem_ptr(smem),  smem_layout);

    // ADMA
    adma_load.cache_.set_tensor_desc(allocate_tdesc<0>());
    uint64_t* adma_load_mbar = allocate_abar<0>();
    Tensor mA = adma_load.get_tma_tensor(shape(smem_layout));
    // Tensor gA = flat_divide(mA, ctile);
    // Slice and get partitions
    auto cta_adma_load = adma_load.get_slice(Int<0>{});                            // CTA slice
    Tensor tAgA_x = cta_adma_load.partition_S(mA);                           // (TMA,TMA_M,TMA_N,REST_M,REST_N)
    Tensor tAsA_x = cta_adma_load.partition_D(t_smem);

    //Tensor tAgA = group_modes<1,rank(tAgA_x)>(tAgA_x);
    //Tensor tAsA = group_modes<1,rank(tAsA_x)>(tAsA_x);
    // Loop over the TMA stages, using smem as our buffer
    int kPhaseBitLoad = 0;
    bool electedThread = cute::elect_one_sync();
    sycl::sub_group sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    if(electedThread && sg.get_group_id() == 0) {
        xe4_initialize_barrier(adma_load_mbar[0], 1 /*numThreads*/);
    }
    //for (int stage = 0; stage < size<1>(tAgA); ++stage) {
    if (electedThread&& sg.get_group_id() == 0) {
	constexpr int kTmaTransactionBytes = sizeof(make_tensor_like(tAsA_x));
        xe4_set_barrier_transaction_bytes(adma_load_mbar[0], kTmaTransactionBytes);
        copy(adma_load.with(&adma_load_mbar[0]), tAgA_x, tAsA_x);
        xe4_wait_barrier(adma_load_mbar[0], kPhaseBitLoad);
    }
    kPhaseBitLoad ^=1;
    //}
    syncthreads();

    int tid = ThreadIdxX();
    //auto block_shape = make_shape(Int<128>{}, Int<64>{});
    TiledCopy tiled_copy = make_ldsm_tiled_copy(CopyOp{}, t_smem, t_layout, v_layout);
    //TiledCopy tiled_copy = make_ldsm_tiled_copy(CopyOp{}, tiled_tensor_S);

    auto Sshape = shape(SmemLayout{});
    auto coord_shape = make_shape(get<0>(Sshape), get<1>(Sshape));
    //Tensor coord_tile = make_coord_tensor(SmemLayout{});
    Tensor coord_tile = make_identity_tensor(coord_shape);

    auto thr_copy = tiled_copy.get_thread_slice(tid);

    auto tXsX = thr_copy.partition_S(coord_tile);
    auto tXgX = thr_copy.partition_D(t_g_out);  // (V,M,N)
    auto tXrX = thr_copy.partition_fragment_D(coord_tile);

    //auto tXrX = make_tensor<T>(shape(tXgX)); // (V,M,N)
    clear(tXrX);  // Just to make sure
    clear(tXgX);  // Just to make sure
		 
    auto item =sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    item.barrier(sycl::access::fence_space::local_space);
    copy(tiled_copy, tXsX, tXrX);
    //sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_group());
    // Output rmem -> gmem
    item.barrier(sycl::access::fence_space::local_space);
    syncthreads();
    /*
    if (thread(0)) {
      int tid = ThreadIdxX();
      printf("\n ID : %d\n", tid);
      for (int i = 0; i < size<0>(tXrX); ++i) {
        //for (int j = 0; j < size<1>(tXrX); ++j) {
	  printf("%d:%d, ", tXrX(i), tAsA_x(i));
	//}
      }
      printf("\n");
      for (int i = 0; i < size<0>(coord_tile); ++i) {
        for (int j = 0; j < size<1>(coord_tile); ++j) {
	  printf("%d, ", coord_tile(i, j));
	}
        printf("\n");
      }
    }
    */
    copy(tXrX, tXgX);
    syncthreads();
  }
}
template<typename T, typename CopyOp, uint32_t GroupSize, typename TLayout, typename VLayout, typename SLayout,
	 typename Din, typename TiledAdma, typename CTile>
void launch_tiled_kernel(TLayout t_layout, VLayout v_layout,
		         SLayout smem_layout, Din &d_in, int check_count,
			 Din &zeros, 
			 TiledAdma tiled_copy, CTile ctile)
{	
  device_vector<T> d_out=zeros;

  host_vector<T> h_in = d_in;
  sc_exp::launch<ldsm_test_device_cute<T, CopyOp, decltype(smem_layout), TLayout, VLayout,
	                               TiledAdma, CTile>>
    //( sc_exp::launch_policy{sc::dim3(1), sc::dim3(int(size(tiled_copy))),
    ( sc_exp::launch_policy{sc::dim3(1), sc::dim3(32 * GroupSize),
      sc_exp::local_mem_size{sizeof(T) * size(smem_layout)}},
      d_in.data(), d_out.data(), t_layout, v_layout, smem_layout,
      tiled_copy,ctile);
  sc::wait_and_throw();
  host_vector<T> h_out = d_out;
  auto q = sc::get_default_queue();

  Tensor ht_in = make_tensor(h_in.data(), smem_layout);
  Tensor ht_out = make_tensor(h_out.data(), smem_layout);
  //  Set limit to 8 for Unordered to pass 
  //     for (int i = 0; i < size<1>(ht_in); ++i) 
  for (int i = 0; i < size<0>(ht_in); ++i) {
    for (int j = 0; j < size<1>(ht_in); ++j) {
      int val1 = ht_in(i,j);
      int val2 = ht_out(i,j);
      EXPECT_EQ(val1, val2);
      //printf("\tData: %d : %d  %d \n", i, val1, val2);
    }
  }
  /*
  printf("\n-------------ht_in-------------------\n");
  print_tensor(ht_in);
  printf("\n-------------ht_out-------------------\n");
  print_tensor(ht_out);
  */
}

template<int GroupSize, LDSMMode Mode> struct ThrLayout {
   static constexpr int ThrM=32/GroupSize;
    // Since ThrM is div by GroupSize, ThrN is GroupSize^2
    // as there will be 32 * GroupSize threads
   static constexpr int ThrN=GroupSize*GroupSize;
   static constexpr Layout t_layout = make_layout(make_shape(Int<ThrM>{}, Int<ThrN>{}), LayoutLeft{});
   //static constexpr Layout t_layout = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<2>{}, Int<8>{}));
};

// Unordered Vetor needs seperate layout due to h/w constraints on bank acces
// first 8 threads in group 0 access 8 rows in bank 1,
// then first 8 threads in group 1 access 8 threads in back 2
// upto 4 banks. This make the stride 32
template<int GroupSize> struct ThrLayout<GroupSize, LDSMMode::UnorderedVector> {
   static constexpr Layout t_layout = make_layout(make_shape(make_shape(Int<8>{}, Int<4>{}), Int<GroupSize>{}),
    		                                  make_stride(make_stride(Int<1>{}, Int<32>{}), Int<8>{}));
};
template<int GroupSize> struct ThrLayout<GroupSize, LDSMMode::UnorderedArrOfVectors> {
   static constexpr Layout t_layout = make_layout(make_shape(make_shape(Int<8>{}, Int<4>{}), Int<GroupSize>{}),
    		                                  make_stride(make_stride(Int<1>{}, Int<32>{}), Int<8>{}));
};
template<typename T, int M, int N, LDSMMode Mode, int Vlen,
         cute::Vecdir Vdir, uint32_t Alen, cute::Arrdir Adir, uint32_t GroupSize=1>
void run_ldsm_test()
{
  constexpr int elem_alignment = 16 / sizeof(T);
  constexpr int aligned_N = ((N + elem_alignment - 1) / elem_alignment) * elem_alignment;
  int count = M * aligned_N;
  // Define a tensor shape with dynamic extents (m, n)
  auto tensor_shape = make_shape(Int<M>{}, Int<N>{});
  //int count = size(tensor_shape);
  host_vector<T> h_in(count);
  host_vector<T> zeros(count);

  using SLayout = decltype(make_layout(tensor_shape, LayoutRight{}));
  for (int i = 0; i < count; ++i) {
    h_in[i] = T(i);
    zeros[i] = 0;
  }
  {
    // Load Vector
    using CopyOp = XE4_LOAD_MATRIX<T, SLayout, Mode, Vlen, Vdir, Alen, Adir>;
    using ADMA_Load = cute::XE4_ADMA_LOAD;
    auto mem_layout = SLayout{};

    device_vector<T> d_in = h_in;
    Tensor gA = make_tensor(make_gmem_ptr(d_in.data()), mem_layout);
    auto cta_tiler = product_each(shape(mem_layout));
    auto adma_load = cute::make_adma_copy<T>(ADMA_Load{}, gA, mem_layout, cta_tiler, Int<1>{});

    Layout t_layout = ThrLayout<GroupSize, Mode>::t_layout;

    Layout v_layout_row = make_layout(make_shape(Int<1>{}, Int<Vlen*Alen>{}));
    Layout v_layout_col = make_layout(make_shape(Int<Vlen*Alen>{}, Int<1>{}));
   
    static_assert((Vdir == cute::Vecdir::Vrow || Vdir == cute::Vecdir::Vcol));
    int check_count = count;
    if (Vdir == cute::Vecdir::Vrow) {
      launch_tiled_kernel<T, CopyOp, GroupSize>(t_layout, v_layout_row, SLayout{},
		          h_in, check_count, zeros, adma_load, cta_tiler);
    } else {
      launch_tiled_kernel<T, CopyOp, GroupSize>(t_layout, v_layout_col, SLayout{},
		          h_in, check_count, zeros, adma_load, cta_tiler);
    }

    CUTLASS_TRACE_HOST("CuTe LDSM SUCCESS\n");
  }

  CUTLASS_TRACE_HOST("PASS");
}
TEST(XE4_CuTe_JGS, LDSM_ADMA_Row_UAOfVector_row_2x16)
{
  using T = uint16_t;
  static constexpr int ALEN=2;
  run_ldsm_test<T, 32, 16*4*2, LDSMMode::UnorderedArrOfVectors,16, cute::Vecdir::Vrow,
	        ALEN, cute::Arrdir::Arow, 4>();
}
