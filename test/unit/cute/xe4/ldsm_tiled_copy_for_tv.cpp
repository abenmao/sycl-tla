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

// Unit tests for cute::make_ldsm_tiled_copy_noninjective — the public entry point
// for building an LDSM TiledCopy from a hand-rolled (T, V) layout + Tiler_MN
// pair, bypassing CuTe's `make_tiled_copy` (which uses raked_product +
// right_inverse and rejects non-injective layouts).
//
// These tests exercise the API directly: each test constructs the FMHA
// within-warp-row TV layout and tiler externally (via cute::detail helpers),
// picks a load and store op type, calls `make_ldsm_tiled_copy_noninjective`, and
// performs a GMEM → SLM(src) → Reg → SLM(dst) → GMEM round-trip.  Identical
// recovery confirms the API correctly drives the LDSM atoms when handed a
// non-injective TV layout that `make_tiled_copy` would reject.
//
// Coverage:
//   * UnorderedVector — TotalThreads % 128 == 0 and NumValPerWIPerIter
//     ∈ {1×, 2×, 4×} coop_vlen<BitWidth>.
//   * Vector (selector → Vector when UV preconditions fail).
//   * Vector explicit (PreferredMode = Vector).
//   * Several (M, N, NumWarps) variations including FMHA-shaped fp32 tiles
//     and the recast-to-uint32 fp16 interleaving used by the softmax.

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

// -----------------------------------------------------------------------------
// Small constexpr helpers mirroring the warp-row factory's selector.  We
// re-derive (Mode, Vlen, Alen) at the test level so the test owns the
// op-type construction end-to-end — exactly what a future custom caller
// of `make_ldsm_tiled_copy_noninjective` would do.
// -----------------------------------------------------------------------------
template <int BitWidth, int NumValPerWIPerIter, LDSMMode PreferredMode,
          int TotalThreads>
struct WarpRowSelector {
  static constexpr int  coop_vlen_ = ldsm_coop_vlen<BitWidth>();
  static constexpr int  vec_vlen_  = ldsm_vector_row_vlen<BitWidth, true>();
  static constexpr bool can_uv_ =
      (PreferredMode == UnorderedVector) &&
      (TotalThreads % 128 == 0) &&
      (NumValPerWIPerIter >= coop_vlen_) &&
      (NumValPerWIPerIter % coop_vlen_ == 0) &&
      (NumValPerWIPerIter / coop_vlen_ == 1 ||
       NumValPerWIPerIter / coop_vlen_ == 2 ||
       NumValPerWIPerIter / coop_vlen_ == 4);
  static constexpr LDSMMode mode = can_uv_ ? UnorderedVector : Vector;
  static constexpr int      vlen =
      can_uv_ ? coop_vlen_
              : ((NumValPerWIPerIter < vec_vlen_) ? NumValPerWIPerIter
                                                  : vec_vlen_);
  static constexpr int      alen = can_uv_ ? (NumValPerWIPerIter / coop_vlen_) : 0;
};

template <typename T, int NumWarps, LDSMMode PreferredMode, class SmemLayout>
CUTLASS_GLOBAL void
ldsm_for_tv_kernel(T* g_in, T* g_out, SmemLayout smem_layout)
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
    char* shared_memory;
#endif
    using Storage = SharedStorage<T, SmemLayout>;
    Storage& shared_storage = *reinterpret_cast<Storage*>(shared_memory);

    auto t_g_in  = make_tensor(make_gmem_ptr(g_in),  smem_layout);
    auto t_g_out = make_tensor(make_gmem_ptr(g_out), smem_layout);
    Tensor t_smem_src = recast<T>(make_tensor(make_smem_ptr(shared_storage.smem_src.begin()), smem_layout));
    Tensor t_smem_dst = recast<T>(make_tensor(make_smem_ptr(shared_storage.smem_dst.begin()), smem_layout));

    int tid = ThreadIdxX();
    constexpr int kTotalThreads = 32 * NumWarps;

    // GMEM → SLM(src).
    for (int i = tid; i < size(t_smem_src); i += kTotalThreads) {
      t_smem_src(i) = t_g_in(i);
    }
    syncthreads();

    // ----- Build the warp-row TV layout and tiler externally -----
    constexpr int BitWidth        = sizeof_bits_v<T>;
    constexpr int NumThreadsPerWarp = 32;
    constexpr int TotalThreads    = NumWarps * NumThreadsPerWarp;
    constexpr int EuCount         = 4;
    constexpr int RowsPerWi       = 2;
    constexpr int EuSgCount       = NumWarps / EuCount;
    constexpr int NumThreadPerRow = NumThreadsPerWarp / RowsPerWi;
    constexpr int TileM = decltype(size<0>(SmemLayout{}))::value;
    constexpr int TileN = decltype(size<1>(SmemLayout{}))::value;
    constexpr int NumValPerWIPerIter = TileN / NumThreadPerRow;

    using Sel = WarpRowSelector<BitWidth, NumValPerWIPerIter, PreferredMode,
                                 TotalThreads>;
    constexpr LDSMMode Mode = Sel::mode;
    constexpr int      Vlen = Sel::vlen;
    constexpr int      Alen = Sel::alen;
    constexpr int      numRowsPerIter = RowsPerWi * NumWarps;

    using RepSLayout = decltype(cute::make_layout(
        cute::make_shape(cute::Int<numRowsPerIter>{}, cute::Int<TileN>{}),
        cute::LayoutRight{}));

    // Pick load and store ops with the same (Mode, Vlen, Alen) triple.
    using LoadOpCoop  = XE4_LOAD_MATRIX <T, RepSLayout, Mode, Vlen,
                                          cute::Vecdir::Vrow,
                                          (Alen ? Alen : 1),
                                          cute::Arrdir::Arow>;
    using LoadOpVec   = XE4_LOAD_MATRIX <T, RepSLayout, Mode, Vlen,
                                          cute::Vecdir::Vrow>;
    using StoreOpCoop = XE4_STORE_MATRIX<T, RepSLayout, Mode, Vlen,
                                          cute::Vecdir::Vrow,
                                          (Alen ? Alen : 1),
                                          cute::Arrdir::Arow>;
    using StoreOpVec  = XE4_STORE_MATRIX<T, RepSLayout, Mode, Vlen,
                                          cute::Vecdir::Vrow>;
    using LoadOp  = std::conditional_t<
        (Mode == UnorderedVector || Mode == CoopVector), LoadOpCoop,  LoadOpVec>;
    using StoreOp = std::conditional_t<
        (Mode == UnorderedVector || Mode == CoopVector), StoreOpCoop, StoreOpVec>;

    auto tv_layout = detail::make_warp_row_tv_layout<
        RowsPerWi, NumThreadPerRow, EuCount, EuSgCount,
        NumValPerWIPerIter, TileM>();
    auto m_tiler = detail::make_warp_row_m_tiler_layout<
        RowsPerWi, EuCount, EuSgCount>();
    auto n_tiler = cute::make_layout(cute::Int<TileN>{}, cute::_1{});
    auto tiler   = cute::make_tile(m_tiler, n_tiler);

    auto tc_load  = make_ldsm_tiled_copy_noninjective(LoadOp{},  t_smem_src, tv_layout, tiler);
    auto tc_store = make_ldsm_tiled_copy_noninjective(StoreOp{}, t_smem_dst, tv_layout, tiler);

    auto coord = make_identity_tensor(make_shape(Int<TileM>{}, Int<TileN>{}));
    auto thr_load  = tc_load .get_thread_slice(tid);
    auto thr_store = tc_store.get_thread_slice(tid);

    auto tXsX     = thr_load .partition_S(coord);
    auto tXrX     = thr_load .partition_fragment_D(coord);
    auto thr_dst  = thr_store.partition_D(coord);
    auto thr_src  = thr_store.partition_fragment_S(coord);

    clear(tXrX);
    copy(tc_load, tXsX, tXrX);
    syncthreads();

    copy(tXrX, thr_src);  // register bridge
    syncthreads();

    copy(tc_store, thr_src, thr_dst);
    syncthreads();

    for (int i = tid; i < size(t_smem_dst); i += kTotalThreads) {
      t_g_out(i) = t_smem_dst(i);
    }
    syncthreads();
  }
}

template <typename T, int M, int N, int NumWarps, LDSMMode PreferredMode>
void run_for_tv_test()
{
  auto tensor_shape = make_shape(Int<M>{}, Int<N>{});
  using SLayout = decltype(make_layout(tensor_shape, LayoutRight{}));
  constexpr int count = M * N;

  host_vector<T> h_in(count);
  host_vector<T> h_zero(count);
  for (int i = 0; i < count; ++i) {
    h_in[i] = T(i & 0xFFFF);
    h_zero[i] = T(0);
  }

  device_vector<T> d_in = h_in;
  device_vector<T> d_out = h_zero;

  sc_exp::launch<ldsm_for_tv_kernel<T, NumWarps, PreferredMode, SLayout>>
    (sc_exp::launch_policy{sc::dim3(1), sc::dim3(32 * NumWarps)},
     d_in.data(), d_out.data(), SLayout{});
  sc::wait_and_throw();

  host_vector<T> h_out = d_out;
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(h_out[i], h_in[i]) << "mismatch at index " << i;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// UnorderedVector — selector picks UV when TotalThreads % 128 == 0 and
// NumValPerWIPerIter ∈ {1×, 2×, 4×} coop_vlen.
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe_JGS, LDSM_TiledCopyForTV_RoundTrip_fp32_UV_Alen1) {
  // fp32, NumWarps=4 → TotalThreads=128, coop_vlen<32>=8, NEPT=8 → Alen=1.
  run_for_tv_test<float, 8, 128, 4, LDSMMode::UnorderedVector>();
}

TEST(XE4_CuTe_JGS, LDSM_TiledCopyForTV_RoundTrip_fp32_UV_Alen2) {
  // fp32, NumWarps=4, TileN=256 → NEPT=16 = 2·coop_vlen → Alen=2.
  run_for_tv_test<float, 8, 256, 4, LDSMMode::UnorderedVector>();
}

TEST(XE4_CuTe_JGS, LDSM_TiledCopyForTV_RoundTrip_fp16_UV_Alen1) {
  // fp16, NumWarps=4, TileN=256 → coop_vlen<16>=16, NEPT=16 → Alen=1.
  run_for_tv_test<uint16_t, 8, 256, 4, LDSMMode::UnorderedVector>();
}

TEST(XE4_CuTe_JGS, LDSM_TiledCopyForTV_RoundTrip_fp32_UV_TotalRows2) {
  // TotalRowsPerThread = 2 (TileM = 2·numRowsPerIter).
  run_for_tv_test<float, 16, 128, 4, LDSMMode::UnorderedVector>();
}

TEST(XE4_CuTe_JGS, LDSM_TiledCopyForTV_RoundTrip_fp32_UV_NumWarps8) {
  // NumWarps=8 → EuSgCount=2, larger cohort.
  run_for_tv_test<float, 16, 128, 8, LDSMMode::UnorderedVector>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Vector (selector fallback) — UV preferred but NEPT < coop_vlen.
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe_JGS, LDSM_TiledCopyForTV_RoundTrip_fp32_VectorFallback) {
  // fp32, NumWarps=4, TileN=64 → NEPT=4 < coop_vlen<32>=8 → Vector.
  run_for_tv_test<float, 8, 64, 4, LDSMMode::UnorderedVector>();
}

TEST(XE4_CuTe_JGS, LDSM_TiledCopyForTV_RoundTrip_fp16_VectorFallback) {
  // fp16, NumWarps=4, TileN=128 → NEPT=8 < coop_vlen<16>=16 → Vector.
  run_for_tv_test<uint16_t, 8, 128, 4, LDSMMode::UnorderedVector>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Vector (explicit PreferredMode).
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe_JGS, LDSM_TiledCopyForTV_RoundTrip_fp32_Vector) {
  run_for_tv_test<float, 8, 128, 4, LDSMMode::Vector>();
}

TEST(XE4_CuTe_JGS, LDSM_TiledCopyForTV_RoundTrip_fp16_Vector) {
  run_for_tv_test<uint16_t, 8, 256, 4, LDSMMode::Vector>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// CoopVector preference — selector falls back to Vector inside the warp-row
// path (UV is the only cooperative form `WarpRowSelector` accepts).  The
// round-trip should still pass.
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe_JGS, LDSM_TiledCopyForTV_RoundTrip_fp32_CoopVectorFallback) {
  run_for_tv_test<float, 8, 128, 4, LDSMMode::CoopVector>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FMHA-shaped interleaving — recast fp16 SLM to uint32 then build the TV
// layout and tiler over the uint32 view.  Mirrors the FMHA softmax path.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumWarps, int M_fp16, int N_fp16>
CUTLASS_GLOBAL void
ldsm_for_tv_fmha_kernel(uint16_t* g_in, uint16_t* g_out)
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
    for (int i = tid; i < size(sS); i += kTotalThreads) sS(i) = t_g_in(i);
    syncthreads();

    auto sS_packed = recast<uint32_t>(sS);
    auto sP_packed = recast<uint32_t>(sP);

    using T_packed = uint32_t;
    constexpr int BitWidth        = sizeof_bits_v<T_packed>;
    constexpr int NumThreadsPerWarp = 32;
    constexpr int TotalThreads    = NumWarps * NumThreadsPerWarp;
    constexpr int EuCount         = 4;
    constexpr int RowsPerWi       = 2;
    constexpr int EuSgCount       = NumWarps / EuCount;
    constexpr int NumThreadPerRow = NumThreadsPerWarp / RowsPerWi;
    constexpr int TileM           = M_fp16;
    constexpr int TileN_packed    = N_fp16 / 2;  // sizeof(uint32) / sizeof(fp16)
    constexpr int NumValPerWIPerIter = TileN_packed / NumThreadPerRow;
    constexpr int numRowsPerIter = RowsPerWi * NumWarps;

    using Sel = WarpRowSelector<BitWidth, NumValPerWIPerIter,
                                 LDSMMode::UnorderedVector, TotalThreads>;
    constexpr LDSMMode Mode = Sel::mode;
    constexpr int      Vlen = Sel::vlen;
    constexpr int      Alen = Sel::alen;

    using RepSLayout = decltype(cute::make_layout(
        cute::make_shape(cute::Int<numRowsPerIter>{}, cute::Int<TileN_packed>{}),
        cute::LayoutRight{}));
    using LoadOp  = XE4_LOAD_MATRIX <T_packed, RepSLayout, Mode, Vlen,
                                      cute::Vecdir::Vrow,
                                      (Alen ? Alen : 1),
                                      cute::Arrdir::Arow>;
    using StoreOp = XE4_STORE_MATRIX<T_packed, RepSLayout, Mode, Vlen,
                                      cute::Vecdir::Vrow,
                                      (Alen ? Alen : 1),
                                      cute::Arrdir::Arow>;

    auto tv_layout = detail::make_warp_row_tv_layout<
        RowsPerWi, NumThreadPerRow, EuCount, EuSgCount,
        NumValPerWIPerIter, TileM>();
    auto m_tiler = detail::make_warp_row_m_tiler_layout<
        RowsPerWi, EuCount, EuSgCount>();
    auto n_tiler = cute::make_layout(cute::Int<TileN_packed>{}, cute::_1{});
    auto tiler   = cute::make_tile(m_tiler, n_tiler);

    auto tc_load  = make_ldsm_tiled_copy_noninjective(LoadOp{},  sS_packed, tv_layout, tiler);
    auto tc_store = make_ldsm_tiled_copy_noninjective(StoreOp{}, sP_packed, tv_layout, tiler);

    auto coord = make_identity_tensor(make_shape(Int<TileM>{}, Int<TileN_packed>{}));
    auto thr_load  = tc_load .get_thread_slice(tid);
    auto thr_store = tc_store.get_thread_slice(tid);
    auto tXsX     = thr_load .partition_S(coord);
    auto tXrX     = thr_load .partition_fragment_D(coord);
    auto thr_dst  = thr_store.partition_D(coord);
    auto thr_src  = thr_store.partition_fragment_S(coord);

    clear(tXrX);
    copy(tc_load, tXsX, tXrX);
    syncthreads();
    copy(tXrX, thr_src);
    syncthreads();
    copy(tc_store, thr_src, thr_dst);
    syncthreads();

    for (int i = tid; i < size(sP); i += kTotalThreads) t_g_out(i) = sP(i);
    syncthreads();
  }
}

template <int NumWarps, int M_fp16, int N_fp16>
void run_for_tv_fmha_test()
{
  constexpr int count = M_fp16 * N_fp16;
  host_vector<uint16_t> h_in(count);
  host_vector<uint16_t> h_zero(count, 0);
  for (int i = 0; i < count; ++i) h_in[i] = uint16_t(i & 0xFFFF);

  device_vector<uint16_t> d_in = h_in;
  device_vector<uint16_t> d_out = h_zero;

  sc_exp::launch<ldsm_for_tv_fmha_kernel<NumWarps, M_fp16, N_fp16>>
    (sc_exp::launch_policy{sc::dim3(1), sc::dim3(32 * NumWarps)},
     d_in.data(), d_out.data());
  sc::wait_and_throw();

  host_vector<uint16_t> h_out = d_out;
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(h_out[i], h_in[i]) << "mismatch at index " << i;
  }
}

TEST(XE4_CuTe_JGS, LDSM_TiledCopyForTV_RoundTrip_FMHA_fp16_NW4) {
  run_for_tv_fmha_test</*NumWarps=*/4, /*M=*/8, /*N=*/256>();
}

TEST(XE4_CuTe_JGS, LDSM_TiledCopyForTV_RoundTrip_FMHA_fp16_NW4_TotalRows2) {
  run_for_tv_fmha_test</*NumWarps=*/4, /*M=*/16, /*N=*/256>();
}

TEST(XE4_CuTe_JGS, LDSM_TiledCopyForTV_RoundTrip_FMHA_fp16_NW8) {
  run_for_tv_fmha_test</*NumWarps=*/8, /*M=*/16, /*N=*/256>();
}
