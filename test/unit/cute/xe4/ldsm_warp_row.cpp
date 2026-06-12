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

// Unit tests for make_ldsm_copy_warp_row_C / make_ldsm_copy_warp_row_D APIs in
// include/cute/atom/copy_traits_xe4_ldsm.hpp.
//
// These factories build a TiledCopy with a hand-rolled (T,V)→(M,N) bijection
// such that row-mates (work-items sharing an m coordinate) live within a
// single 32-thread warp. They are used in the FMHA4 softmax epilogue to make
// HW cooperative-row LDSM coexist with within-warp `fred.max` reductions.
//
// Each test does a GMEM → SLM(src) → Reg [warp_row_C] → SLM(dst) [warp_row_D]
// → GMEM round-trip and checks byte-identical recovery.
//
// Mode coverage:
//   * UnorderedVector — `PreferredMode = UnorderedVector` and dimensions chosen so
//     the LdsmWarpRowModeSelector predicate holds (NumValPerWIPerIter is a
//     {1×, 2×, 4×} multiple of coop_vlen<BitWidth>, TotalThreads % 128 == 0).
//   * Vector (small TileN)        — UnorderedVector preferred but
//     NumValPerWIPerIter < coop_vlen → selector falls back to Vector.
//   * Vector (PreferredMode=Vector) — explicit Vector request.
//   * Vector (PreferredMode=CoopVector) — the warp-row selector does not
//     produce CoopVector; CoopVector preference falls back to Vector. The
//     test verifies the fallback round-trips correctly.
//
// Variations:
//   * (M, N) sweeps for both fp16 and fp32 element types.
//   * NumWarps in {4, 8} (cohort decomposition EuCount × EuSgCount = 4 × 1 / 4 × 2).
//   * TotalRowsPerThread > 1 (TileM = numRowsPerIter × k for k ∈ {1, 2}).
//   * FMHA-shaped interleaving — recast fp16 SLM to uint32 and feed to the
//     factory, mirroring CollectiveSoftmaxEpilogue::update() in
//     examples/xe4/fmha4/collective/xe4_fmha_fwd_softmax_epilogue.hpp.

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

// Direct round-trip via make_ldsm_copy_warp_row_C / _D.
// Operates on the SLM tensor as-is (no recast). The coord tile passed to
// partition_S/D matches the factory's view of (TileM, TileN).
template <typename T, int NumWarps, LDSMMode PreferredMode, class SmemLayout>
CUTLASS_GLOBAL void
ldsm_warp_row_kernel(T* g_in, T* g_out, SmemLayout smem_layout)
{
  using namespace cute;

  if (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group_linear_id() == 0)
  {
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

    // GMEM → SLM(src). Stride by total thread count so each element is
    // written by exactly one thread (no idempotent races).
    constexpr int kTotalThreads = 32 * NumWarps;
    for (int i = tid; i < size(t_smem_src); i += kTotalThreads) {
      t_smem_src(i) = t_g_in(i);
    }
    syncthreads();

    // S2R via make_ldsm_copy_warp_row_C.
    auto tiled_load = make_ldsm_copy_warp_row_C<NumWarps, /*EuCount=*/4,
                                                 /*RowsPerWi=*/2, PreferredMode>(t_smem_src);

    auto Sshape = shape(SmemLayout{});
    auto coord_shape = make_shape(get<0>(Sshape), get<1>(Sshape));
    Tensor coord_tile = make_identity_tensor(coord_shape);

    auto thr_load = tiled_load.get_thread_slice(tid);
    auto tXsX = thr_load.partition_S(coord_tile);
    auto tXrX = thr_load.partition_fragment_D(coord_tile);
    clear(tXrX);

    copy(tiled_load, tXsX, tXrX);
    syncthreads();

    // R2S via make_ldsm_copy_warp_row_D.
    auto tiled_store = make_ldsm_copy_warp_row_D<NumWarps, /*EuCount=*/4,
                                                  /*RowsPerWi=*/2, PreferredMode>(t_smem_dst);
    auto thr_store = tiled_store.get_thread_slice(tid);
    auto thr_dst_coord = thr_store.partition_D(coord_tile);
    auto thr_src_frag  = thr_store.partition_fragment_S(coord_tile);
    copy(tXrX, thr_src_frag);
    syncthreads();

    copy(tiled_store, thr_src_frag, thr_dst_coord);
    syncthreads();

    // SLM(dst) → GMEM.
    for (int i = tid; i < size(t_smem_dst); i += kTotalThreads) {
      t_g_out(i) = t_smem_dst(i);
    }
    syncthreads();
  }
}

template <typename T, int M, int N, int NumWarps, LDSMMode PreferredMode>
void run_warp_row_test()
{
  auto tensor_shape = make_shape(Int<M>{}, Int<N>{});
  using SLayout = decltype(make_layout(tensor_shape, LayoutRight{}));
  constexpr int count = M * N;

  host_vector<T> h_in(count);
  host_vector<T> h_zero(count);
  for (int i = 0; i < count; ++i) {
    h_in[i] = T(i & 0xFFFF);  // mask so uint16_t doesn't wrap; uint32_t fine
    h_zero[i] = T(0);
  }

  device_vector<T> d_in = h_in;
  device_vector<T> d_out = h_zero;

  sc_exp::launch<ldsm_warp_row_kernel<T, NumWarps, PreferredMode, SLayout>>
    (sc_exp::launch_policy{sc::dim3(1), sc::dim3(32 * NumWarps)},
     d_in.data(), d_out.data(), SLayout{});
  sc::wait_and_throw();

  host_vector<T> h_out = d_out;
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(h_out[i], h_in[i]) << "mismatch at index " << i;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// UnorderedVector — TotalThreads % 128 == 0 and NumValPerWIPerIter ∈ {1×, 2×, 4×} coop_vlen.
// Selector emits ld_matrix.unordered.al<Alen>.cooprow.<bw>.
////////////////////////////////////////////////////////////////////////////////////////////////////

// fp32, NumWarps=4 → TotalThreads=128, coop_vlen<32>=8. NumValPerWIPerIter=8 → Alen=1.
TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_fp32_UnorderedVector)
{
  // TileM=8 = numRowsPerIter (TotalRowsPerThread=1).
  run_warp_row_test<float, 8, 128, 4, LDSMMode::UnorderedVector>();
}

// Same as above but TotalRowsPerThread=2 (TileM = 2×numRowsPerIter).
TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_fp32_UnorderedVector_TotalRows2)
{
  run_warp_row_test<float, 16, 128, 4, LDSMMode::UnorderedVector>();
}

// fp32, NumWarps=4, TileN=256 → NumValPerWIPerIter=16 = 2·coop_vlen → Alen=2.
TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_fp32_UnorderedVector_Alen2)
{
  run_warp_row_test<float, 8, 256, 4, LDSMMode::UnorderedVector>();
}

// fp16, NumWarps=4, TileN=256 → NumValPerWIPerIter=16 = 1·coop_vlen<16>=16 → Alen=1.
TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_fp16_UnorderedVector)
{
  run_warp_row_test<uint16_t, 8, 256, 4, LDSMMode::UnorderedVector>();
}

// fp16, NumWarps=4, TotalRowsPerThread=2.
TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_fp16_UnorderedVector_TotalRows2)
{
  run_warp_row_test<uint16_t, 16, 256, 4, LDSMMode::UnorderedVector>();
}

// NumWarps=8 (EuSgCount=2): exercises a different cohort decomposition and
// 256-thread launch. fp32 TileN=128, NumValPerWIPerIter=8, Alen=1.
TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_fp32_UnorderedVector_NumWarps8)
{
  // numRowsPerIter = 2*8 = 16; TileM = 16 → TotalRowsPerThread = 1.
  run_warp_row_test<float, 16, 128, 8, LDSMMode::UnorderedVector>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Vector (selector fallback from UnorderedVector when NumValPerWIPerIter < coop_vlen).
// The factory still emits a working per-warp Vector load/store.
////////////////////////////////////////////////////////////////////////////////////////////////////

// fp32, NumWarps=4, TileN=64 → NumValPerWIPerIter=4 < coop_vlen<32>=8.
// Selector → Vector mode, Vlen = min(4, vec_vlen<32,Type1>=8) = 4.
TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_fp32_VectorFallback_SmallTileN)
{
  run_warp_row_test<float, 8, 64, 4, LDSMMode::UnorderedVector>();
}

// fp16, NumWarps=4, TileN=128 → NumValPerWIPerIter=8 < coop_vlen<16>=16.
// Selector → Vector mode, Vlen = min(8, vec_vlen<16,Type1>=16) = 8.
TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_fp16_VectorFallback_SmallTileN)
{
  run_warp_row_test<uint16_t, 8, 128, 4, LDSMMode::UnorderedVector>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Vector — explicit PreferredMode = Vector. Selector picks Vector directly.
////////////////////////////////////////////////////////////////////////////////////////////////////

// fp32, NumWarps=4, TileN=128, PreferredMode=Vector.
TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_fp32_Vector)
{
  run_warp_row_test<float, 8, 128, 4, LDSMMode::Vector>();
}

// fp16, NumWarps=4, TileN=256, PreferredMode=Vector.
TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_fp16_Vector)
{
  run_warp_row_test<uint16_t, 8, 256, 4, LDSMMode::Vector>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// CoopVector preference — falls back to Vector inside the warp-row selector.
// `LdsmWarpRowModeSelector::can_unordered_` requires `PreferredMode == UnorderedVector`,
// so any other preference (including CoopVector) goes through the Vector branch.
// Test verifies the round-trip still works.
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_fp32_CoopVectorFallback)
{
  run_warp_row_test<float, 8, 128, 4, LDSMMode::CoopVector>();
}

TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_fp16_CoopVectorFallback)
{
  run_warp_row_test<uint16_t, 8, 256, 4, LDSMMode::CoopVector>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FMHA-shaped interleaving — recast fp16 SLM to uint32 and feed the warp-row factory,
// mirroring CollectiveSoftmaxEpilogue::update() in fmha4. The recast halves the column
// count (sizeof(uint32) / sizeof(fp16) = 2) so the factory sees a (TileM, TileN_packed)
// uint32 view with HW-correct Pitch from make_ldsm_matrix_descriptor.
//
// Selector with BitWidth=32, coop_vlen<32>=8: NumValPerWIPerIter=8 → UV, Vlen=8, Alen=1.
// This is the exact path FMHA softmax uses for sS / sP loads/stores.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumWarps, int M_fp16, int N_fp16>
CUTLASS_GLOBAL void
ldsm_warp_row_fmha_kernel(uint16_t* g_in, uint16_t* g_out)
{
  using namespace cute;

  if (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group_linear_id() == 0)
  {
    using SmemLayoutFp16 = decltype(make_layout(
        make_shape(Int<M_fp16>{}, Int<N_fp16>{}), LayoutRight{}));
    constexpr auto smem_size = cute::cosize_v<SmemLayoutFp16> * sizeof(uint16_t) * 2;
#if defined(__SYCL_DEVICE_ONLY__)
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto shared_memory = alloc_slm_buffer<uint16_t, smem_size>(item.get_group());
#endif
#if defined(CUTLASS_ENABLE_SYCL) && !defined(__SYCL_DEVICE_ONLY__)
    char* shared_memory;
#endif
    using Storage = SharedStorage<uint16_t, SmemLayoutFp16>;
    Storage& shared_storage = *reinterpret_cast<Storage*>(shared_memory);

    auto t_g_in  = make_tensor(make_gmem_ptr(g_in),  SmemLayoutFp16{});
    auto t_g_out = make_tensor(make_gmem_ptr(g_out), SmemLayoutFp16{});
    Tensor sS = recast<uint16_t>(make_tensor(make_smem_ptr(shared_storage.smem_src.begin()),
                                             SmemLayoutFp16{}));
    Tensor sP = recast<uint16_t>(make_tensor(make_smem_ptr(shared_storage.smem_dst.begin()),
                                             SmemLayoutFp16{}));

    int tid = ThreadIdxX();
    constexpr int kTotalThreads = 32 * NumWarps;

    // GMEM → SLM(src) (fp16 view).
    for (int i = tid; i < size(sS); i += kTotalThreads) {
      sS(i) = t_g_in(i);
    }
    syncthreads();

    // FMHA convention: recast fp16 SLM to uint32 before handing to the factory.
    // make_ldsm_matrix_descriptor's Pitch = stride>>2 is HW-correct for 4-byte
    // elements; the recast halves the column count so the descriptor matches.
    auto sS_packed = recast<uint32_t>(sS);
    auto sP_packed = recast<uint32_t>(sP);

    auto tc_load  = make_ldsm_copy_warp_row_C<NumWarps>(sS_packed);
    auto tc_store = make_ldsm_copy_warp_row_D<NumWarps>(sP_packed);

    constexpr int N_packed = N_fp16 / 2;  // sizeof(uint32) / sizeof(fp16)
    auto coord = make_identity_tensor(make_shape(Int<M_fp16>{}, Int<N_packed>{}));

    auto thr_load  = tc_load .get_thread_slice(tid);
    auto thr_store = tc_store.get_thread_slice(tid);

    // Bulk-fragment copy: load the whole (CPY, CPY_M, CPY_N) fragment in one
    // call, bridge load → store via register-to-register copy, then store.
    // This matches the simpler ldsm_epilogue.cpp pattern; FMHA's iter-loop
    // pattern (copy(_, i, _0{})) is equivalent but harder to validate
    // without the surrounding compute that imposes the iter granularity.
    auto tXsX     = thr_load .partition_S(coord);
    auto tXrX     = thr_load .partition_fragment_D(coord);
    auto thr_dst  = thr_store.partition_D(coord);
    auto thr_src  = thr_store.partition_fragment_S(coord);
    clear(tXrX);

    copy(tc_load, tXsX, tXrX);
    syncthreads();

    copy(tXrX, thr_src);  // register-to-register bridge
    syncthreads();

    copy(tc_store, thr_src, thr_dst);
    syncthreads();

    // SLM(dst) → GMEM (fp16 view).
    for (int i = tid; i < size(sP); i += kTotalThreads) {
      t_g_out(i) = sP(i);
    }
    syncthreads();
  }
}

template <int NumWarps, int M_fp16, int N_fp16>
void run_warp_row_fmha_test()
{
  constexpr int count = M_fp16 * N_fp16;
  host_vector<uint16_t> h_in(count);
  host_vector<uint16_t> h_zero(count, 0);
  for (int i = 0; i < count; ++i) {
    h_in[i] = uint16_t(i & 0xFFFF);
  }

  device_vector<uint16_t> d_in = h_in;
  device_vector<uint16_t> d_out = h_zero;

  sc_exp::launch<ldsm_warp_row_fmha_kernel<NumWarps, M_fp16, N_fp16>>
    (sc_exp::launch_policy{sc::dim3(1), sc::dim3(32 * NumWarps)},
     d_in.data(), d_out.data());
  sc::wait_and_throw();

  host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(h_out[i], h_in[i]) << "mismatch at index " << i;
  }
}

// FMHA-shaped: NumWarps=4, M=8, N=256 (fp16). After recast: TileM=8, TileN_packed=128,
// NumValPerWIPerIter=8=coop_vlen<32>, UV with Alen=1, TotalRowsPerThread=1.
TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_FMHA_Interleaved_fp16_NW4)
{
  run_warp_row_fmha_test</*NumWarps=*/4, /*M_fp16=*/8, /*N_fp16=*/256>();
}

// FMHA-shaped with TotalRowsPerThread > 1: NumWarps=4, M=16, N=256.
TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_FMHA_Interleaved_fp16_NW4_TotalRows2)
{
  run_warp_row_fmha_test</*NumWarps=*/4, /*M_fp16=*/16, /*N_fp16=*/256>();
}

// FMHA-shaped with NumWarps=8: numRowsPerIter=16, TileM=16 → TotalRowsPerThread=1.
TEST(XE4_CuTe_JGS, LDSM_WarpRow_RoundTrip_FMHA_Interleaved_fp16_NW8)
{
  run_warp_row_fmha_test</*NumWarps=*/8, /*M_fp16=*/16, /*N_fp16=*/256>();
}
