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

// Unit tests for geometry-only make_ldsm_copy_C / make_ldsm_copy_D APIs (no TiledMMA)
// Round-trip validation: GMEM -> SLM(src) -> Reg [copy_C] -> SLM(dst) [copy_D] -> GMEM
// These test the overloads that derive layouts purely from SLM tensor geometry.

#include "cutlass_unit_test.h"
#include <iostream>
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_xe4_ldsm.hpp>

using namespace cute;
namespace sc = compat;
namespace sc_exp = compat::experimental;

template <class ElementType, class SmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem_src;
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem_dst;
};

// Device kernel: round-trip via geometry-only make_ldsm_copy_C (S2R) and make_ldsm_copy_D (R2S)
// No TiledMMA — layouts derived purely from SLM tensor geometry and ThrGroupSize.
template <typename T, int ThrGroupSize, LDSMMode PreferredMode,
          class SmemLayout>
CUTLASS_GLOBAL void
ldsm_roundtrip_kernel(T* g_in, T* g_out, SmemLayout smem_layout)
{
  using namespace cute;

  if (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group_linear_id() == 0)
  {
  // Allocate SLM for two buffers (src + dst)
  constexpr auto smem_size = cute::cosize_v<decltype(smem_layout)> * sizeof(T) * 2;
#if defined(__SYCL_DEVICE_ONLY__)
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto shared_memory = alloc_slm_buffer<T, smem_size>(item.get_group());
#endif
#if defined(CUTLASS_ENABLE_SYCL) && !defined(__SYCL_DEVICE_ONLY__)
  char* shared_memory; // dummy declaration for host compilation
#endif
    using Storage = SharedStorage<T, SmemLayout>;
    Storage& shared_storage = *reinterpret_cast<Storage*>(shared_memory);

    auto t_g_in  = make_tensor(make_gmem_ptr(g_in),  smem_layout);
    auto t_g_out = make_tensor(make_gmem_ptr(g_out), smem_layout);
    Tensor t_smem_src = recast<T>(make_tensor(make_smem_ptr(shared_storage.smem_src.begin()), smem_layout));
    Tensor t_smem_dst = recast<T>(make_tensor(make_smem_ptr(shared_storage.smem_dst.begin()), smem_layout));

    int tid = ThreadIdxX();

    // Step 1: GMEM -> SLM(src)
    for (int i = tid; i < size(t_smem_src); i += 32) {
      t_smem_src(i) = t_g_in(i);
    }
    syncthreads();

    // Step 2: S2R via geometry-only make_ldsm_copy_C (no TiledMMA)
    auto tiled_load = make_ldsm_copy_C<ThrGroupSize, PreferredMode>(t_smem_src);

    auto Sshape = shape(SmemLayout{});
    auto coord_shape = make_shape(get<0>(Sshape), get<1>(Sshape));
    Tensor coord_tile = make_identity_tensor(coord_shape);

    auto thr_load = tiled_load.get_thread_slice(tid);
    auto tXsX = thr_load.partition_S(coord_tile);
    auto tXrX = thr_load.partition_fragment_D(coord_tile);
    clear(tXrX);

    copy(tiled_load, tXsX, tXrX);
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group());

    // Step 3: R2S via geometry-only make_ldsm_copy_D (no TiledMMA)
    auto tiled_store = make_ldsm_copy_D<ThrGroupSize, PreferredMode>(t_smem_dst);

    auto thr_store = tiled_store.get_thread_slice(tid);
    auto thr_dst_coord = thr_store.partition_D(coord_tile);
    auto thr_src_frag = thr_store.partition_fragment_S(coord_tile);
    // Register-to-register copy to match store layout
    copy(tXrX, thr_src_frag);
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group());

    copy(tiled_store, thr_src_frag, thr_dst_coord);
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group());

    // Step 4: SLM(dst) -> GMEM
    for (int i = tid; i < size(t_smem_dst); i += 32) {
      t_g_out(i) = t_smem_dst(i);
    }
    sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group());
  }
}

// Host-side test runner for geometry-only LDSM APIs (no TiledMMA).
// Layout derivation is based purely on SLM tensor shape and ThrGroupSize.
template <typename T, int M, int N, int ThrGroupSize = 4,
          LDSMMode PreferredMode = UnorderedVector>
void run_ldsm_roundtrip_test()
{
  auto tensor_shape = make_shape(Int<M>{}, Int<N>{});
  using SLayout = decltype(make_layout(tensor_shape, LayoutRight{}));
  constexpr int count = M * N;

  host_vector<T> h_in(count);
  host_vector<T> h_zero(count);
  for (int i = 0; i < count; ++i) {
    h_in[i] = T(i);
    h_zero[i] = T(0);
  }

  device_vector<T> d_in = h_in;
  device_vector<T> d_out = h_zero;

  sc_exp::launch<ldsm_roundtrip_kernel<T, ThrGroupSize, PreferredMode, SLayout>>
    (sc_exp::launch_policy{sc::dim3(1), sc::dim3(32 * ThrGroupSize)},
     d_in.data(), d_out.data(), SLayout{});
  sc::wait_and_throw();

  host_vector<T> h_out = d_out;
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(h_out[i], h_in[i]);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// UnorderedVector (default mode) — highest throughput, ThrGroupSize=4, 128 threads
// Geometry-only: layout derived from SLM tensor shape, no MMA information used.
////////////////////////////////////////////////////////////////////////////////////////////////////

// bf16/fp16 (16-bit): uint16_t covers both since LDSM moves raw bits
TEST(XE4_CuTe_JGS, LDSM_Epilogue_RoundTrip_bf16)
{
  // UnorderedVector, Vlen=16, tile 32x64, ThrGroupSize=4
  run_ldsm_roundtrip_test<uint16_t, 32, 64>();
}

// fp32 (32-bit)
TEST(XE4_CuTe_JGS, LDSM_Epilogue_RoundTrip_fp32)
{
  // UnorderedVector, Vlen=8, tile 32x32, ThrGroupSize=4
  run_ldsm_roundtrip_test<float, 32, 32>();
}

// int8 (8-bit)
TEST(XE4_CuTe_JGS, LDSM_Epilogue_RoundTrip_int8)
{
  // UnorderedVector, Vlen=32, tile 32x128, ThrGroupSize=4
  run_ldsm_roundtrip_test<uint8_t, 32, 128>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// CoopVector mode — ordered cooperative, ThrGroupSize=1, 32 threads
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe_JGS, LDSM_Epilogue_RoundTrip_bf16_CoopVector)
{
  // CoopVector, Vlen=16, tile 32x32, ThrGroupSize=1
  run_ldsm_roundtrip_test<uint16_t, 32, 32, 1, LDSMMode::CoopVector>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Vector mode — widest compatibility, ThrGroupSize=1, 32 threads
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe_JGS, LDSM_Epilogue_RoundTrip_bf16_Vector)
{
  // Vector, Vlen=16, tile 32x32, ThrGroupSize=1
  run_ldsm_roundtrip_test<uint16_t, 32, 32, 1, LDSMMode::Vector>();
}

TEST(XE4_CuTe_JGS, LDSM_Epilogue_RoundTrip_fp32_Vector)
{
  // Vector, Vlen=8, tile 32x32, ThrGroupSize=1
  run_ldsm_roundtrip_test<float, 32, 32, 1, LDSMMode::Vector>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-atom tiling tests — WarpTileN > Vlen
// These exercise the case where each thread handles more elements than a single LDSM
// invocation (atom) can load/store. The TiledCopy automatically tiles multiple atoms
// along the N dimension per thread.
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe_JGS, LDSM_Epilogue_RoundTrip_bf16_CoopVector_WideN)
{
  // CoopVector, tile 32x64, ThrGroupSize=1
  // WarpTileN=64, Vlen=16 → 4 atom invocations per thread along N
  run_ldsm_roundtrip_test<uint16_t, 32, 64, 1, LDSMMode::CoopVector>();
}

TEST(XE4_CuTe_JGS, LDSM_Epilogue_RoundTrip_fp32_CoopVector_WideN)
{
  // CoopVector, tile 32x32, ThrGroupSize=1
  // WarpTileN=32, Vlen=8 → 4 atom invocations per thread along N
  run_ldsm_roundtrip_test<float, 32, 32, 1, LDSMMode::CoopVector>();
}

TEST(XE4_CuTe_JGS, LDSM_Epilogue_RoundTrip_bf16_Vector_WideN)
{
  // Vector, tile 32x64, ThrGroupSize=1
  // WarpTileN=64, Vlen=16 → 4 atom invocations per thread along N
  run_ldsm_roundtrip_test<uint16_t, 32, 64, 1, LDSMMode::Vector>();
}

TEST(XE4_CuTe_JGS, LDSM_Epilogue_RoundTrip_fp32_Vector_WideN)
{
  // Vector, tile 32x64, ThrGroupSize=1
  // WarpTileN=64, Vlen=8 → 8 atom invocations per thread along N
  run_ldsm_roundtrip_test<float, 32, 64, 1, LDSMMode::Vector>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Mode fallback tests — verify graceful degradation
// When preferred UnorderedVector can't meet thread count requirements (< 128),
// the selector falls back to CoopVector while preserving correct behavior.
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe_JGS, LDSM_Epilogue_RoundTrip_bf16_Unordered_Fallback_Coop)
{
  // Preferred UnorderedVector with ThrGroupSize=1 (32 threads < 128)
  // LdsmModeVlenSelector falls back to CoopVector mode
  run_ldsm_roundtrip_test<uint16_t, 32, 32, 1, LDSMMode::UnorderedVector>();
}
