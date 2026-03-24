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

#pragma once

#include "cutlass_unit_test.h"

#include <cute/arch/xe4_inline_pisa.hpp>

#include <iostream>
#include <cstdint>

#include <cute/tensor.hpp>

namespace sc = compat;
namespace sc_exp = compat::experimental;
namespace sycl_ext = sycl::ext::oneapi::experimental;
using sycl::ext::oneapi::this_work_item::get_nd_item;

namespace cutlass::test {

template <class ElementType, class SmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
};

template <class T, class TmaType = T, class TiledLoad, class TiledStore, class CTA_Tiler, class GmemLayout, class SmemLayout>
CUTLASS_GLOBAL void
tma_test_device_cute(sycl::nd_item<3> &item, T* g_in, T* g_out,
                     TiledLoad adma_load, TiledStore adma_store, CTA_Tiler cta_tiler,
                     GmemLayout gmem_layout, SmemLayout smem_layout)
{
  using namespace cute;

  adma_load.set_tensor_desc(allocate_tdesc<0>());
  adma_store.set_tensor_desc(allocate_tdesc<1>());

  CUTE_STATIC_ASSERT_V(product_each(shape(cta_tiler)) == product_each(shape(smem_layout)));

  // Use Shared Storage structure to allocate and distribute aligned SMEM addresses
  constexpr auto tma_size = cute::cosize_v<decltype(smem_layout)> * sizeof(TmaType) * sizeof(TmaType) / sizeof(T);
#if defined(__SYCL_DEVICE_ONLY__)
  auto shared_memory = alloc_slm_buffer<TmaType, tma_size>(item.get_group());
#endif
#if defined(CUTLASS_ENABLE_SYCL) && !defined(__SYCL_DEVICE_ONLY__)
  char* shared_memory; // dummy declaration to avoid compilation errors during the host compilation phase
#endif

  using SharedStorage = SharedStorage<TmaType, SmemLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  // Construct SMEM tensor
  Tensor sA = recast<T>(make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout));  // (CTA_TILE_M,CTA_TILE_N,...)
  uint64_t* adma_load_mbar = allocate_abar<0>();
  uint64_t* adma_store_mbar = allocate_abar<1>();

  // TMA requires special handling of strides to deal with coord codomain mapping
  // Represent the full tensors -- get these from TMA
  Tensor mA = adma_load.get_tma_tensor(shape(gmem_layout));
  Tensor mB = adma_store.get_tma_tensor(shape(gmem_layout));

  constexpr int R = rank_v<CTA_Tiler>;
  Tensor gA = flat_divide(mA, cta_tiler);               // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)
  Tensor gB = flat_divide(mB, cta_tiler);               // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)
#if 0 
  if (thread0()) {
    print("gA\n"); print(gA);
    print("gB\n"); print(gB);
  }
#endif

  // Prepare the adma_load
  auto cta_adma_load = adma_load.get_slice(Int<0>{});                            // CTA slice
  Tensor tAgA_x = cta_adma_load.partition_S(gA);                           // (TMA,TMA_M,TMA_N,REST_M,REST_N)
  Tensor tAsA_x = cta_adma_load.partition_D(sA);                           // (TMA,TMA_M,TMA_N)

  auto cta_adma_store = adma_store.get_slice(Int<0>{});                            // CTA slice
  Tensor tAsA_x_st = cta_adma_store.partition_S(sA);                           // (TMA,TMA_M,TMA_N)
  Tensor tBgB_x = cta_adma_store.partition_D(gB);                           // (TMA,TMA_M,TMA_N,REST_M,REST_N)
                                                                           //
#if 0 
  if (thread0()) {
    print("\ntAgA_x\n"); print(tAgA_x);
    print("\ntAsA_x\n"); print(tAsA_x);
    print("\ntAsA_x_st\n"); print(tAsA_x_st);
  }
#endif 

  // INPUT: Group the REST_X modes and the TMA_X modes to easily iterate through the tiles
  Tensor tAgA = group_modes<1,rank(tAgA_x)>(tAgA_x);                 // (TMA,REST)
  Tensor tAsA = group_modes<1,rank(tAsA_x)>(tAsA_x);                 // (TMA,REST)
  static_assert(size<1>(tAsA) == 1);

  // INPUT: Group the REST_X modes and the TMA_X modes to easily iterate through the tiles
  Tensor tBgB = group_modes<1,rank(tBgB_x)>(tBgB_x);                 // (TMA,REST)
  Tensor tBsB = group_modes<1,rank(tAsA_x_st)>(tAsA_x_st);                 // (TMA,REST)
  static_assert(size<1>(tBsB) == 1);

  // Loop over the TMA stages, using smem as our buffer
  int kPhaseBitLoad = 0;
  int kPhaseBitStore = 0;

  bool electedThread = cute::elect_one_sync();
  if(electedThread) {
    xe4_initialize_barrier(adma_load_mbar[0], 1 /*numThreads*/);
    xe4_initialize_barrier(adma_store_mbar[0], 1 /*numThreads*/);
  }
  item.barrier(sycl::access::fence_space::local_space);

  for (int stage = 0; stage < size<1>(tAgA); ++stage)
  {
#if 0 
    if (thread0()){
      print("\n stage is "); print(stage);
      print("\n grouped tAgA\n"); print(tAgA);
      print("\n grouped tAsA\n"); print(tAsA);
    }
#endif

    constexpr int kTmaTransactionBytes = sizeof(make_tensor_like(tensor<0>(tAsA)));
    //constexpr int kTmaTransactionBytes = cute::cosize_v<decltype(tAsA(_,0).layout())> * sizeof(T);
    if (electedThread) {
      xe4_set_barrier_transaction_bytes(adma_load_mbar[0], kTmaTransactionBytes);
      copy(adma_load.with(&adma_load_mbar[0]), tAgA(_,stage), tAsA(_,0));
      xe4_wait_barrier(adma_load_mbar[0], kPhaseBitLoad);
    }
    kPhaseBitLoad ^=1;

    item.barrier(sycl::access::fence_space::local_space);
    if (electedThread) {
      xe4_set_barrier_transaction_bytes(adma_store_mbar[0], kTmaTransactionBytes);
      copy(adma_store.with(&adma_store_mbar[0]), tBsB(_,0), tBgB(_,stage));
      xe4_wait_barrier(adma_store_mbar[0], kPhaseBitStore);
    }
    kPhaseBitStore ^=1;
  }
  syncthreads();
}


template <class T, class TmaType = T, class LoadOp, class StoreOp, class GMEM_Layout, class SMEM_Layout, class CTA_Tile>
auto
test_tma_load(LoadOp      const& load_op,
			        StoreOp	  const& store_op,	
              GMEM_Layout const& gmem_layout,
              SMEM_Layout const& smem_layout,
              CTA_Tile    const& cta_tile)
{
  using namespace cute;

  // Allocate and initialize host test data
  size_t N = ceil_div(cosize(gmem_layout) * sizeof_bits<T>::value, 8);
  host_vector<T> h_in(N,T(-1));
  for (size_t i = 0; i < h_in.size(); ++i) {
    h_in[i] = static_cast<T>(float(int(i) % 1024));
  }
  Tensor hA_in  = make_tensor(recast_ptr<T>(h_in.data()), gmem_layout);

  // Allocate and initialize device test data
  sycl::queue queue;

  T* d_in = static_cast<T*>(malloc_device(h_in.size() * sizeof(T), queue));
  queue.memcpy(d_in, h_in.data(), sizeof(T) * h_in.size()).wait();
  T* d_out = static_cast<T*>(malloc_device(h_in.size() * sizeof(T), queue));

  // Create TMA for this device Tensor
  Tensor gA = make_tensor(make_gmem_ptr<T>(raw_pointer_cast(d_in)), gmem_layout);
  Tensor gB = make_tensor(make_gmem_ptr<T>(raw_pointer_cast(d_out)), gmem_layout);

  // Launch
  sycl::range<3> local_range(1, 1, 32);
  sycl::range<3> global_range(1, 1, 32);

  sycl::nd_range<3> range(global_range, local_range);
  auto adma_load = cute::make_adma_copy<T>(load_op, gA, smem_layout, cta_tile, Int<1>{});
  auto adma_store = cute::make_adma_copy<T>(store_op, gB, smem_layout, cta_tile, Int<1>{});

  queue.parallel_for(range, [=](sycl::nd_item<3> item) {
      tma_test_device_cute(item, d_in, d_out, adma_load, adma_store, cta_tile,
          gmem_layout, smem_layout);
      }).wait();

  // Copy results back to host
  host_vector<T> h_out(N, T(-1));
  queue.memcpy(h_out.data(), d_out, sizeof(T) * h_in.size()).wait();
  Tensor hA_out = make_tensor(recast_ptr<T>(h_out.data()), gmem_layout);

  // Validate the results. Print only the first 3 errors.
  int count = 10;
  for (int i = 0; i < int(size(hA_out)) && count > 0 ; ++i) {
    EXPECT_EQ(hA_in(i), hA_out(i));
    if (hA_in(i) != hA_out(i)) {
      --count;
    }
  }

  return adma_load;
}


} // end namespace cutlass::test
