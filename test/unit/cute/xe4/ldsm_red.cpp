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
#include <algorithm>
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_xe4_ldsm.hpp>


using namespace cute;
namespace sc = compat;
namespace sc_exp = compat::experimental;
namespace sycl_ext = sycl::ext::oneapi::experimental;

template <class ElementType, class SmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> base_smem;
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> base_smem_t;
};


template <typename T, typename LoadOp, typename RedOp, class SmemLayout, class TLayout, class VLayout>
CUTLASS_GLOBAL void
ldsm_test_device_cute(T* g_in, T* g_out,
		      TLayout t_layout,
		      VLayout v_layout,
		      SmemLayout smem_layout) //, sycl::local_ptr<char> base_smem)
{
  using namespace cute;
  //auto smem = reinterpret_cast<T*>((char*)base_smem);
  //sycl::local_ptr<char> base_smem_st;
  //auto s_smem = reinterpret_cast<T*>((char*)base_smem_st);


  if (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group_linear_id() == 0)
  {

  // Use Shared Storage structure to allocate and distribute aligned SMEM addresses
  constexpr auto smem_size = cute::cosize_v<decltype(smem_layout)> * 2;
#if defined(__SYCL_DEVICE_ONLY__)
  auto item =sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto shared_memory = alloc_slm_buffer<T, smem_size>(item.get_group());
#endif
#if defined(CUTLASS_ENABLE_SYCL) && !defined(__SYCL_DEVICE_ONLY__)
  char* shared_memory; // dummy declaration to avoid compilation errors during the host compilation phase
#endif
    using SharedStorage = SharedStorage<T, SmemLayout>;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

    auto t_g_in  = make_tensor(make_gmem_ptr(g_in),  smem_layout);
    auto t_g_out = make_tensor(make_gmem_ptr(g_out), smem_layout);
    //auto t_smem  = make_tensor(make_smem_ptr(smem),  smem_layout);
    //auto ts_smem  = make_tensor(make_smem_ptr(s_smem),  smem_layout);

    Tensor t_smem= recast<T>(make_tensor(make_smem_ptr(shared_storage.base_smem.begin()), smem_layout));
    Tensor ts_smem= recast<T>(make_tensor(make_smem_ptr(shared_storage.base_smem_t.begin()), smem_layout));

    int tid = ThreadIdxX();

    // Load input gmem -> smem
    for (int i = tid; i < size(t_smem); i += 32) {
      t_smem(i) = t_g_in(i);
      ts_smem(i) = t_g_out(i);
    }
    syncthreads();
    TiledCopy tiled_load = make_ldsm_tiled_copy(LoadOp{}, t_smem, t_layout, v_layout);
    //TiledCopy tiled_load = make_ldsm_tiled_copy(CopyOp{}, tiled_tensor_S);

    auto Sshape = shape(SmemLayout{});
    auto coord_shape = make_shape(get<0>(Sshape), get<1>(Sshape));
    //Tensor coord_tile = make_coord_tensor(SmemLayout{});
    Tensor coord_tile = make_identity_tensor(coord_shape);

    auto thr_load = tiled_load.get_thread_slice(tid);

    auto tXsX = thr_load.partition_S(coord_tile);
    //auto tXgX = thr_load.partition_D(t_g_out);  // (V,M,N)
    auto tXrX = thr_load.partition_fragment_D(coord_tile);

    //auto tXrX = make_tensor<T>(shape(tXgX)); // (V,M,N)
    clear(tXrX);  // Just to make sure
    //clear(tXgX);  // Just to make sure
		 
    // Copy SMEM -> RMEM via tiled_copy (LDSM, LDS)
    copy(tiled_load, tXsX, tXrX);
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group());
    //syncthreads();

    // Store OP from RMEM to SMEM
    TiledCopy tiled_store = make_ldsm_tiled_copy_reduce(RedOp{}, ts_smem);
    auto thr_store = tiled_store.get_thread_slice(tid);
    // Dst is coord tile
    auto thr_dst_coord = thr_store.partition_D(coord_tile);
    // Src register fragment
    auto thr_src_frag = thr_store.partition_fragment_S(coord_tile);
    // Default Copy register to register
    copy(tXrX, thr_src_frag);
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group());
    // Copy (Store) from  RMEM src to SMEM
    copy(tiled_store, thr_src_frag, thr_dst_coord);
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group());

    for (int i = tid; i < size(t_smem); i += 32) {
      t_g_out(i) = ts_smem(i);
    }
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group());
    //copy(tXrX, tXgX);
  }
}
template<typename T, typename LoadOp, typename RedOp,
         typename TLayout, typename VLayout, typename SLayout,
	 typename Hin>
void launch_tiled_kernel(TLayout t_layout, VLayout v_layout,
		         SLayout smem_layout, Hin &h_in, int count,
			 Hin &incr)
{
  device_vector<T> d_in = h_in;
  device_vector<T> d_out=incr;

  sc_exp::launch<ldsm_test_device_cute<T, LoadOp, RedOp, decltype(smem_layout), TLayout, VLayout>>
    //( sc_exp::launch_policy{sc::dim3(1), sc::dim3(int(size(tiled_copy))),
    ( sc_exp::launch_policy{sc::dim3(1), sc::dim3(32)},
      //sc_exp::local_mem_size{sizeof(T) * size(smem_layout)}},
      d_in.data(), d_out.data(), t_layout, v_layout, smem_layout);
  sc::wait_and_throw();
  host_vector<T> h_out = d_out;

  for (int i = 1; i < count; ++i) {
    //printf("\tData: %d  %d \n", int(h_in[i]), int(h_out[i]));
    T cval = h_in[i];
    if (RedOp::RedOp == cute::MredOp::Add) {
       cval += incr[i];
    } else if (RedOp::RedOp == cute::MredOp::Max) { // Max Only for float
      cval = std::max(cval, incr[i]);
    } else if (RedOp::RedOp == cute::MredOp::Min) { // Min Only for float
      cval = std::min(cval, incr[i]);
    } 
    EXPECT_EQ(h_out[i], cval);
  }
}

template<typename T, int M, int N, LDSMMode Mode, int Vlen,
         cute::Vecdir Vdir, cute::MredOp Rop, int Val=0>
void run_ldsm_test()
{
  constexpr int elem_alignment = 16 / sizeof(T);
  constexpr int aligned_N = ((N + elem_alignment - 1) / elem_alignment) * elem_alignment;
  int count = M * aligned_N;
  // Define a tensor shape with dynamic extents (m, n)
  auto tensor_shape = make_shape(Int<M>{}, Int<N>{});
  host_vector<T> h_in(count);
  host_vector<T> incr(count);

  using SLayout = decltype(make_layout(tensor_shape, LayoutRight{}));
  for (int i = 0; i < count; ++i) {
    h_in[i] = T(i);
    incr[i] = Val;
  }
  {
    // Load Vector
    using LoadOp = XE4_LOAD_MATRIX<T, SLayout, Mode,
                                   Vlen, Vdir>;
    using RedOp = XE4_REDUCE_MATRIX<T, Rop, SLayout, Mode,
                                   Vlen, Vdir>;
    constexpr int THRS = 32;
    Layout t_layout = make_layout(make_shape(Int<THRS>{}, Int<1>{}), LayoutLeft{});
    Layout v_layout_row = make_layout(make_shape(Int<1>{}, Int<Vlen>{}));
    Layout v_layout_col = make_layout(make_shape(Int<Vlen>{}, Int<1>{}));
   
    if (Vdir == cute::Vecdir::Vrow) {
      launch_tiled_kernel<T, LoadOp, RedOp>(t_layout, v_layout_row, SLayout{}, h_in, count, incr);
    } else {
      launch_tiled_kernel<T, LoadOp, RedOp>(t_layout, v_layout_col, SLayout{}, h_in, count, incr);
    }
    CUTLASS_TRACE_HOST("CuTe LDSM SUCCESS\n");
  }
  CUTLASS_TRACE_HOST("PASS");
}
TEST(XE4_CuTe_JGS, LDSM_Rreduce_Vector_float_Add)
{
  // Row direction
  using T = sycl::half;
  run_ldsm_test<T, 32, 16, LDSMMode::Vector,4, cute::Vecdir::Vrow, cute::MredOp::Add, 7>();
  run_ldsm_test<T, 32, 16, LDSMMode::Vector,8, cute::Vecdir::Vrow, cute::MredOp::Add, 7>();
  run_ldsm_test<T, 32, 16, LDSMMode::Vector,4, cute::Vecdir::Vrow, cute::MredOp::Max, 153>();
  run_ldsm_test<T, 32, 16, LDSMMode::Vector,8, cute::Vecdir::Vrow, cute::MredOp::Max, 153>();
  run_ldsm_test<T, 32, 16, LDSMMode::Vector,8, cute::Vecdir::Vrow, cute::MredOp::Min, 153>();
}
TEST(XE4_CuTe_JGS, LDSM_Reduce_Vector_int_Add)
{
  // Row direction
  using T = int;
  run_ldsm_test<T, 32, 16, LDSMMode::Vector,1, cute::Vecdir::Vrow, cute::MredOp::Add, 7>();
  run_ldsm_test<T, 32, 16, LDSMMode::Vector,4, cute::Vecdir::Vrow, cute::MredOp::Add, 77>();
  run_ldsm_test<T, 32, 16, LDSMMode::Vector,8, cute::Vecdir::Vrow, cute::MredOp::Add, 77>();
}
