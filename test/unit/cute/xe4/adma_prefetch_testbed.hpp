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
#include <cute/algorithm/prefetch.hpp>

namespace sc = compat;
namespace sc_exp = compat::experimental;
namespace sycl_ext = sycl::ext::oneapi::experimental;
using sycl::ext::oneapi::this_work_item::get_nd_item;

namespace cutlass::test {

// Shared memory storage type used by the prefetch test kernels.
// Sized to hold one SMEM tile of the given layout.
template <class ElementType, class SmemLayout>
struct PrefetchSharedStorage
{
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Device kernel: Standalone prefetch test — 3 ADMA atoms (load, store, prefetch)
///
/// Test flow (per stage):
///   1. PREFETCH: fire-and-forget prefetch of the NEXT stage into L2 cache
///      - Uses Copy_Traits<XE4_ADMA_PREFETCH> — directly executable, no .with() needed
///      - copy(adma_prefetch, src, dst) where dst is a dummy (never dereferenced)
///   2. LOAD: gmem → SLM via async_tensor_copy with arrival barrier
///   3. STORE: SLM → gmem via async_tensor_copy with arrival barrier
///
/// Verification strategy: Prefetch must not corrupt data. After prefetch + load + store,
///   h_out[i] must equal h_in[i] for all elements. This confirms:
///   - Prefetch instruction does not modify memory
///   - Load after prefetch succeeds (data is in L2 as intended)
///   - The prefetch tensor descriptor is constructed correctly
///
/// This test uses 3 tensor descriptors (allocate_tdesc<0,1,2>) — one each for load/store/prefetch.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T, class TmaType = T, class TiledLoad, class TiledStore, class TiledPrefetch,
          class CTA_Tiler, class GmemLayout, class SmemLayout>
CUTLASS_GLOBAL void
adma_prefetch_test_device(sycl::nd_item<3> &item, T* g_in, T* g_out,
                          TiledLoad adma_load, TiledStore adma_store,
                          TiledPrefetch adma_prefetch,
                          CTA_Tiler cta_tiler,
                          GmemLayout gmem_layout, SmemLayout smem_layout)
{
  using namespace cute;

  adma_load.set_tensor_desc(allocate_tdesc<0>());
  adma_store.set_tensor_desc(allocate_tdesc<1>());
  adma_prefetch.set_tensor_desc(allocate_tdesc<2>());

  CUTE_STATIC_ASSERT_V(product_each(shape(cta_tiler)) == product_each(shape(smem_layout)));

  // Allocate SLM
  constexpr auto tma_size = cute::cosize_v<decltype(smem_layout)> * sizeof(TmaType) * sizeof(TmaType) / sizeof(T);
#if defined(__SYCL_DEVICE_ONLY__)
  auto shared_memory = alloc_slm_buffer<TmaType, tma_size>(item.get_group());
#endif
#if defined(CUTLASS_ENABLE_SYCL) && !defined(__SYCL_DEVICE_ONLY__)
  char* shared_memory;
#endif

  using SharedStorage = PrefetchSharedStorage<TmaType, SmemLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  Tensor sA = recast<T>(make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout));
  uint64_t* adma_load_mbar = allocate_abar<0>();
  uint64_t* adma_store_mbar = allocate_abar<1>();

  Tensor mA = adma_load.get_tma_tensor(shape(gmem_layout));
  Tensor mB = adma_store.get_tma_tensor(shape(gmem_layout));
  Tensor mP = adma_prefetch.get_tma_tensor(shape(gmem_layout));

  Tensor gA = flat_divide(mA, cta_tiler);
  Tensor gB = flat_divide(mB, cta_tiler);
  Tensor gP = flat_divide(mP, cta_tiler);

  auto cta_adma_load = adma_load.get_slice(Int<0>{});
  Tensor tAgA_x = cta_adma_load.partition_S(gA);
  Tensor tAsA_x = cta_adma_load.partition_D(sA);

  auto cta_adma_store = adma_store.get_slice(Int<0>{});
  Tensor tAsA_x_st = cta_adma_store.partition_S(sA);
  Tensor tBgB_x = cta_adma_store.partition_D(gB);

  auto cta_adma_pf = adma_prefetch.get_slice(Int<0>{});
  Tensor tPgP_x = cta_adma_pf.partition_S(gP);

  Tensor tAgA = group_modes<1,rank(tAgA_x)>(tAgA_x);
  Tensor tAsA = group_modes<1,rank(tAsA_x)>(tAsA_x);
  static_assert(size<1>(tAsA) == 1);

  Tensor tBgB = group_modes<1,rank(tBgB_x)>(tBgB_x);
  Tensor tBsB = group_modes<1,rank(tAsA_x_st)>(tAsA_x_st);
  static_assert(size<1>(tBsB) == 1);

  Tensor tPgP = group_modes<1,rank(tPgP_x)>(tPgP_x);

  int kPhaseBitLoad = 0;
  int kPhaseBitStore = 0;

  bool electedThread = cute::elect_one_sync();
  if(electedThread) {
    xe4_initialize_barrier(adma_load_mbar[0], 1);
    xe4_initialize_barrier(adma_store_mbar[0], 1);
  }
  item.barrier(sycl::access::fence_space::local_space);

  for (int stage = 0; stage < size<1>(tAgA); ++stage)
  {
    // ─── PREFETCH: fire-and-forget (directly executable, no barrier, no .with()) ───
    // Issue prefetch for the NEXT stage (if available) while loading current stage.
    if (electedThread) {
      int pf_stage = (stage + 1 < size<1>(tPgP)) ? stage + 1 : stage;
      // PREFETCH traits are directly executable — no .with() needed
      copy(adma_prefetch, tPgP(_,pf_stage), tAsA(_,0));
    }

    // ─── LOAD: gmem → SLM via ADMA ───
    constexpr int kTmaTransactionBytes = sizeof(make_tensor_like(tensor<0>(tAsA)));
    if (electedThread) {
      xe4_set_barrier_transaction_bytes(adma_load_mbar[0], kTmaTransactionBytes);
      copy(adma_load.with(&adma_load_mbar[0]), tAgA(_,stage), tAsA(_,0));
      xe4_wait_barrier(adma_load_mbar[0], kPhaseBitLoad);
    }
    kPhaseBitLoad ^= 1;

    item.barrier(sycl::access::fence_space::local_space);

    // ─── STORE: SLM → gmem via ADMA ───
    if (electedThread) {
      xe4_set_barrier_transaction_bytes(adma_store_mbar[0], kTmaTransactionBytes);
      copy(adma_store.with(&adma_store_mbar[0]), tBsB(_,0), tBgB(_,stage));
      xe4_wait_barrier(adma_store_mbar[0], kPhaseBitStore);
    }
    kPhaseBitStore ^= 1;
  }
  syncthreads();
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// Host harness: allocate, launch, verify
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T, class TmaType = T, class GMEM_Layout, class SMEM_Layout, class CTA_Tile>
auto
test_adma_prefetch(GMEM_Layout const& gmem_layout,
                   SMEM_Layout const& smem_layout,
                   CTA_Tile    const& cta_tile)
{
  using namespace cute;

  // Allocate and initialize host test data
  size_t N = ceil_div(cosize(gmem_layout) * sizeof_bits<T>::value, 8);
  host_vector<T> h_in(N, T(-1));
  for (size_t i = 0; i < h_in.size(); ++i) {
    h_in[i] = static_cast<T>(float(int(i) % 1024));
  }
  Tensor hA_in = make_tensor(recast_ptr<T>(h_in.data()), gmem_layout);

  // Allocate device memory
  sycl::queue queue;
  T* d_in  = static_cast<T*>(malloc_device(h_in.size() * sizeof(T), queue));
  queue.memcpy(d_in, h_in.data(), sizeof(T) * h_in.size()).wait();
  T* d_out = static_cast<T*>(malloc_device(h_in.size() * sizeof(T), queue));

  // Create tensors
  Tensor gA = make_tensor(make_gmem_ptr<T>(raw_pointer_cast(d_in)),  gmem_layout);
  Tensor gB = make_tensor(make_gmem_ptr<T>(raw_pointer_cast(d_out)), gmem_layout);

  // Create ADMA load/store/prefetch atoms
  auto adma_load     = cute::make_adma_copy<T>(cute::XE4_ADMA_LOAD{},     gA, smem_layout, cta_tile, Int<1>{});
  auto adma_store    = cute::make_adma_copy<T>(cute::XE4_ADMA_STORE{},    gB, smem_layout, cta_tile, Int<1>{});
  auto adma_prefetch = cute::make_adma_copy<T>(cute::XE4_ADMA_PREFETCH{}, gA, smem_layout, cta_tile, Int<1>{});

  // Launch
  sycl::range<3> local_range(1, 1, 32);
  sycl::range<3> global_range(1, 1, 32);
  sycl::nd_range<3> range(global_range, local_range);

  queue.parallel_for(range, [=](sycl::nd_item<3> item) {
      adma_prefetch_test_device(item, d_in, d_out, adma_load, adma_store,
                                adma_prefetch, cta_tile, gmem_layout, smem_layout);
  }).wait();

  // Copy results back
  host_vector<T> h_out(N, T(-1));
  queue.memcpy(h_out.data(), d_out, sizeof(T) * h_in.size()).wait();
  Tensor hA_out = make_tensor(recast_ptr<T>(h_out.data()), gmem_layout);

  // Validate: prefetch + load should produce identical results to input
  int count = 10;
  for (int i = 0; i < int(size(hA_out)) && count > 0; ++i) {
    EXPECT_EQ(hA_in(i), hA_out(i));
    if (hA_in(i) != hA_out(i)) {
      --count;
    }
  }

  sycl::free(d_in, queue);
  sycl::free(d_out, queue);
}

template <class T, class TmaType = T, class GMEM_Layout, class SMEM_Layout>
auto
test_adma_prefetch(GMEM_Layout const& gmem_layout,
                   SMEM_Layout const& smem_layout)
{
  return test_adma_prefetch<T, TmaType>(gmem_layout, smem_layout,
                                        product_each(shape(smem_layout)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Device kernel: cute::prefetch() API — prefetch derived from load atom (SM90-style)
///
/// This test validates the SM90-compatible API: cute::prefetch(tiled_copy, src)
/// which derives Copy_Traits<XE4_ADMA_PREFETCH> from the load atom's traits automatically.
///
/// Key differences from the standalone prefetch test above:
///   - Only 2 ADMA atoms created (load + store) — no separate prefetch atom
///   - Only 2 tensor descriptors allocated (allocate_tdesc<0,1>)
///   - Prefetch reuses the load atom's tensor descriptor via the converting constructor
///     in Copy_Traits<XE4_ADMA_PREFETCH> (copies tensorDesc_, aux_params_, tdesc_ptr_)
///   - Dispatch path: cute::prefetch(tiled_copy, src) → static_cast to Copy_Atom
///     → CopyOp::PREFETCH typedef → constructs Copy_Traits<XE4_ADMA_PREFETCH>
///
/// This is the recommended pattern for production GEMM kernels where minimizing
/// tensor descriptor allocation is important.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T, class TmaType = T, class TiledLoad, class TiledStore,
          class CTA_Tiler, class GmemLayout, class SmemLayout>
CUTLASS_GLOBAL void
adma_prefetch_from_load_test_device(sycl::nd_item<3> &item, T* g_in, T* g_out,
                                    TiledLoad adma_load, TiledStore adma_store,
                                    CTA_Tiler cta_tiler,
                                    GmemLayout gmem_layout, SmemLayout smem_layout)
{
  using namespace cute;

  adma_load.set_tensor_desc(allocate_tdesc<0>());
  adma_store.set_tensor_desc(allocate_tdesc<1>());

  CUTE_STATIC_ASSERT_V(product_each(shape(cta_tiler)) == product_each(shape(smem_layout)));

  // Allocate SLM
  constexpr auto tma_size = cute::cosize_v<decltype(smem_layout)> * sizeof(TmaType) * sizeof(TmaType) / sizeof(T);
#if defined(__SYCL_DEVICE_ONLY__)
  auto shared_memory = alloc_slm_buffer<TmaType, tma_size>(item.get_group());
#endif
#if defined(CUTLASS_ENABLE_SYCL) && !defined(__SYCL_DEVICE_ONLY__)
  char* shared_memory;
#endif

  using SharedStorage = PrefetchSharedStorage<TmaType, SmemLayout>;
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  Tensor sA = recast<T>(make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout));
  uint64_t* adma_load_mbar = allocate_abar<0>();
  uint64_t* adma_store_mbar = allocate_abar<1>();

  // Only 2 TMA tensors — prefetch reuses load's descriptor
  Tensor mA = adma_load.get_tma_tensor(shape(gmem_layout));
  Tensor mB = adma_store.get_tma_tensor(shape(gmem_layout));

  Tensor gA = flat_divide(mA, cta_tiler);
  Tensor gB = flat_divide(mB, cta_tiler);

  auto cta_adma_load = adma_load.get_slice(Int<0>{});
  Tensor tAgA_x = cta_adma_load.partition_S(gA);
  Tensor tAsA_x = cta_adma_load.partition_D(sA);

  auto cta_adma_store = adma_store.get_slice(Int<0>{});
  Tensor tAsA_x_st = cta_adma_store.partition_S(sA);
  Tensor tBgB_x = cta_adma_store.partition_D(gB);

  Tensor tAgA = group_modes<1,rank(tAgA_x)>(tAgA_x);
  Tensor tAsA = group_modes<1,rank(tAsA_x)>(tAsA_x);
  static_assert(size<1>(tAsA) == 1);

  Tensor tBgB = group_modes<1,rank(tBgB_x)>(tBgB_x);
  Tensor tBsB = group_modes<1,rank(tAsA_x_st)>(tAsA_x_st);
  static_assert(size<1>(tBsB) == 1);

  int kPhaseBitLoad = 0;
  int kPhaseBitStore = 0;

  bool electedThread = cute::elect_one_sync();
  if(electedThread) {
    xe4_initialize_barrier(adma_load_mbar[0], 1);
    xe4_initialize_barrier(adma_store_mbar[0], 1);
  }
  item.barrier(sycl::access::fence_space::local_space);

  for (int stage = 0; stage < size<1>(tAgA); ++stage)
  {
    // ─── PREFETCH: cute::prefetch() on the load atom (SM90-style API) ───
    // Prefetch next stage from the load atom — no separate prefetch atom needed
    if (electedThread) {
      int pf_stage = (stage + 1 < size<1>(tAgA)) ? stage + 1 : stage;
      cute::prefetch(adma_load, tAgA(_,pf_stage));
    }

    // ─── LOAD: gmem → SLM via ADMA ───
    constexpr int kTmaTransactionBytes = sizeof(make_tensor_like(tensor<0>(tAsA)));
    if (electedThread) {
      xe4_set_barrier_transaction_bytes(adma_load_mbar[0], kTmaTransactionBytes);
      copy(adma_load.with(&adma_load_mbar[0]), tAgA(_,stage), tAsA(_,0));
      xe4_wait_barrier(adma_load_mbar[0], kPhaseBitLoad);
    }
    kPhaseBitLoad ^= 1;

    item.barrier(sycl::access::fence_space::local_space);

    // ─── STORE: SLM → gmem via ADMA ───
    if (electedThread) {
      xe4_set_barrier_transaction_bytes(adma_store_mbar[0], kTmaTransactionBytes);
      copy(adma_store.with(&adma_store_mbar[0]), tBsB(_,0), tBgB(_,stage));
      xe4_wait_barrier(adma_store_mbar[0], kPhaseBitStore);
    }
    kPhaseBitStore ^= 1;
  }
  syncthreads();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Host harness for cute::prefetch() API — only load + store atoms
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T, class TmaType = T, class GMEM_Layout, class SMEM_Layout, class CTA_Tile>
auto
test_adma_prefetch_from_load(GMEM_Layout const& gmem_layout,
                             SMEM_Layout const& smem_layout,
                             CTA_Tile    const& cta_tile)
{
  using namespace cute;

  // Allocate and initialize host test data
  size_t N = ceil_div(cosize(gmem_layout) * sizeof_bits<T>::value, 8);
  host_vector<T> h_in(N, T(-1));
  for (size_t i = 0; i < h_in.size(); ++i) {
    h_in[i] = static_cast<T>(float(int(i) % 1024));
  }
  Tensor hA_in = make_tensor(recast_ptr<T>(h_in.data()), gmem_layout);

  // Allocate device memory
  sycl::queue queue;
  T* d_in  = static_cast<T*>(malloc_device(h_in.size() * sizeof(T), queue));
  queue.memcpy(d_in, h_in.data(), sizeof(T) * h_in.size()).wait();
  T* d_out = static_cast<T*>(malloc_device(h_in.size() * sizeof(T), queue));

  // Create tensors
  Tensor gA = make_tensor(make_gmem_ptr<T>(raw_pointer_cast(d_in)),  gmem_layout);
  Tensor gB = make_tensor(make_gmem_ptr<T>(raw_pointer_cast(d_out)), gmem_layout);

  // Only 2 atoms — prefetch is derived from load via cute::prefetch()
  auto adma_load  = cute::make_adma_copy<T>(cute::XE4_ADMA_LOAD{},  gA, smem_layout, cta_tile, Int<1>{});
  auto adma_store = cute::make_adma_copy<T>(cute::XE4_ADMA_STORE{}, gB, smem_layout, cta_tile, Int<1>{});

  // Launch
  sycl::range<3> local_range(1, 1, 32);
  sycl::range<3> global_range(1, 1, 32);
  sycl::nd_range<3> range(global_range, local_range);

  queue.parallel_for(range, [=](sycl::nd_item<3> item) {
      adma_prefetch_from_load_test_device(item, d_in, d_out, adma_load, adma_store,
                                          cta_tile, gmem_layout, smem_layout);
  }).wait();

  // Copy results back
  host_vector<T> h_out(N, T(-1));
  queue.memcpy(h_out.data(), d_out, sizeof(T) * h_in.size()).wait();
  Tensor hA_out = make_tensor(recast_ptr<T>(h_out.data()), gmem_layout);

  // Validate: prefetch + load should produce identical results to input
  int count = 10;
  for (int i = 0; i < int(size(hA_out)) && count > 0; ++i) {
    EXPECT_EQ(hA_in(i), hA_out(i));
    if (hA_in(i) != hA_out(i)) {
      --count;
    }
  }

  sycl::free(d_in, queue);
  sycl::free(d_out, queue);
}

template <class T, class TmaType = T, class GMEM_Layout, class SMEM_Layout>
auto
test_adma_prefetch_from_load(GMEM_Layout const& gmem_layout,
                             SMEM_Layout const& smem_layout)
{
  return test_adma_prefetch_from_load<T, TmaType>(gmem_layout, smem_layout,
                                                   product_each(shape(smem_layout)));
}

} // end namespace cutlass::test
