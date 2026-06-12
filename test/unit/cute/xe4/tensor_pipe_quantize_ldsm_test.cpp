/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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

// Unit tests for TensorPipe Quantize (downconvert) using LDSM for register load/store.
//
// Flow: GMEM(wide) --(ADMA load)--> SLM(src) --(LDSM load)--> Registers(wide)
//                  --(TCVD in N_tile slices)--> Registers(narrow)
//                  --(LDSM store)--> SLM(dst) --(ADMA store)--> GMEM(narrow)
//
// Key design: LDSM loads/stores the full tile. N is chosen large enough to satisfy
// LDSM Vlen for both source and destination — no padding is needed.
// TCVD operates on N_tile slices within the register.
//
// LDSM Vlen constraints (minimum columns):
//   32-bit (float): 8 | 16-bit (bf16/fp16): 16 | 8-bit (uint8_t): 32
//
// For bf8/hf8 dest (1 byte/elem): N_dst_bytes = N, need N >= 32
// For fp4 dest (0.5 byte/elem):   N_dst_bytes = N/2, need N >= 64
//
// N_tile=3 limitations:
// - f32 source: N_tile=3 not supported because vector_t<uint32_t, 3> (ext_vector_type(3)) has
//   sizeof=16 due to padding, vs marray<uint32_t, 3> sizeof=12, making sycl::bit_cast fail.
//   Will be addressed later by modifying the gtp_tcvd inline pISA wrapper to handle N_u32_src=3.
// - fp4 dest: Odd N_tile (1, 3) not supported because N_tile * 4 bits is not byte-aligned
//   (e.g. 3*4=12 bits=1.5 bytes). The test harness computes N_dst_bytes_per_tile = N_tile * 4 / 8
//   using integer division which truncates, and the kernel loop cannot correctly stitch
//   sub-byte nibbles across tile boundaries.
//

#include "cutlass_unit_test.h"
#include "tensor_pipe_quantize_dequantize_api.hpp"
#include <cute/atom/mma_traits_xe4_amma.hpp>
#include <iostream>
#include <cstring>

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shared storage: separate src (wide) and dst (narrow) SLM buffers.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class SrcType, class SrcSmemLayout, class DstType, class DstSmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<SrcType, cute::cosize_v<SrcSmemLayout>> smem_src;
  cute::ArrayEngine<DstType, cute::cosize_v<DstSmemLayout>> smem_dst;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Device kernel: ADMA G2S → LDSM load → TCVD (N_tile slices) → LDSM store → ADMA S2G
//
// SrcType: LDSM/GMEM source element type (float or uint16_t for bf16/fp16 raw bits)
// DstType: GMEM destination storage type (uint8_t for 8-bit fp8 / packed fp4)
// TcvdDstType: TCVD semantic dest type (bf8, hf8, fp4_e2m1) — selects pISA variant
// TcvdSrcType: TCVD semantic src type (float, bf16, fp16) — selects pISA variant
// N: source elements per row (satisfies LDSM Vlen for both src and dst, no padding)
// N_tile: per-TCVD slice width (must match a supported tcvd.*.m32nN dimension)
// N_dst: destination bytes per row = N * sizeof_bits<TcvdDstType> / 8
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType, typename TcvdDstType, typename TcvdSrcType,
          int N, int N_tile, int N_dst,
          class SrcSmemLayout, class DstSmemLayout,
          class AdmaLoad, class AdmaStore>
CUTLASS_GLOBAL void
tcvd_ldsm_kernel(sycl::nd_item<3>& item,
                 SrcType* g_in, DstType* g_out,
                 SrcSmemLayout src_smem_layout, DstSmemLayout dst_smem_layout,
                 AdmaLoad adma_ld, AdmaStore adma_st)
{
  using namespace cute;
  namespace api = cutlass::xe4;

  constexpr int M = 32;

  adma_ld.set_tensor_desc(allocate_tdesc<0>());
  adma_st.set_tensor_desc(allocate_tdesc<1>());

  constexpr auto smem_size = cute::cosize_v<decltype(src_smem_layout)> * sizeof(SrcType)
                           + cute::cosize_v<decltype(dst_smem_layout)> * sizeof(DstType);
#if defined(__SYCL_DEVICE_ONLY__)
  auto shared_memory = alloc_slm_buffer<uint8_t, smem_size>(item.get_group());
#endif
#if defined(CUTLASS_ENABLE_SYCL) && !defined(__SYCL_DEVICE_ONLY__)
  char* shared_memory;
#endif

  using Storage = SharedStorage<SrcType, SrcSmemLayout, DstType, DstSmemLayout>;
  Storage& shared_storage = *reinterpret_cast<Storage*>(shared_memory);

  Tensor t_smem_src = recast<SrcType>(
      make_tensor(make_smem_ptr(shared_storage.smem_src.begin()), src_smem_layout));
  Tensor t_smem_dst = recast<DstType>(
      make_tensor(make_smem_ptr(shared_storage.smem_dst.begin()), dst_smem_layout));

  int tid = ThreadIdxX();

  uint64_t* load_mbar = allocate_abar<0>();
  uint64_t* store_mbar = allocate_abar<1>();

  bool elected = cute::elect_one_sync();
  if (elected) {
    xe4_initialize_barrier(load_mbar[0], 1);
    xe4_initialize_barrier(store_mbar[0], 1);
  }
  item.barrier(sycl::access::fence_space::local_space);

  // Step 1: GMEM → SLM(src) via ADMA
  api::adma_load(t_smem_src, adma_ld, src_smem_layout, load_mbar, elected, item);

  // Step 2: SLM(src) → Registers via LDSM (full N-wide tile)
  auto tXrX = api::load_matrix(t_smem_src, src_smem_layout, tid, item);

  // Step 3: TCVD — downconvert in N_tile slices
  constexpr int num_col_slices = N / N_tile;
  constexpr int N_dst_bytes_per_tile = N_tile * ::sizeof_bits<TcvdDstType>() / 8;

  DstType dst_reg[N_dst];
  memset(dst_reg, 0, sizeof(dst_reg));

  for (int cs = 0; cs < num_col_slices; ++cs) {
    auto src_slice = make_tensor(make_rmem_ptr(&tXrX(cs * N_tile)),
                                 make_shape(Int<N_tile>{}));
    DstType tile_dst[N_dst_bytes_per_tile];
    auto dst_tensor = make_tensor(make_rmem_ptr(tile_dst),
                                  make_shape(Int<N_dst_bytes_per_tile>{}));
    cute::tensor_pipe_quantize<TcvdDstType, TcvdSrcType>(src_slice, dst_tensor);

    for (int i = 0; i < N_dst_bytes_per_tile; ++i) {
      dst_reg[cs * N_dst_bytes_per_tile + i] = tile_dst[i];
    }
  }
  sycl::group_barrier(item.get_group());

  // Step 4: Registers → SLM(dst) via LDSM store
  api::store_matrix<DstType, N_dst>(t_smem_dst, dst_reg, dst_smem_layout, tid, item);

  // Step 5: SLM(dst) → GMEM via ADMA
  api::adma_store(t_smem_dst, adma_st, dst_smem_layout, store_mbar, elected);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Host-side test runner for downconvert (TCVD)
//
// N must satisfy LDSM Vlen constraints for BOTH source and destination:
//   - Source: N >= Vlen(SrcType)         (float: 8, uint16_t: 16)
//   - Dest:   N_dst >= Vlen(DstType)     (uint8_t: 32)
//     bf8/hf8:  N_dst = N     → N >= 32
//     fp4:      N_dst = N/2   → N >= 64
// No padding — N is chosen at the test case level to satisfy both.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType, typename TcvdDstType, typename TcvdSrcType,
          typename HostRefType, int M, int N, int N_tile>
void run_ldsm_quantize_test()
{
  static_assert(M == 32, "TCVD requires M=32 (subgroup size)");
  static_assert(N % N_tile == 0, "N must be evenly divisible by N_tile");

  constexpr int N_dst = N * ::sizeof_bits<TcvdDstType>() / 8;

  using SrcSLayout = decltype(make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
  using DstSLayout = decltype(make_layout(make_shape(Int<M>{}, Int<N_dst>{}), LayoutRight{}));

  constexpr int src_count = M * N;
  constexpr int dst_count = M * N_dst;

  host_vector<SrcType> h_in(src_count, SrcType(0));
  cutlass::xe4::fill_source_data<SrcType, TcvdSrcType, M, N>(h_in);

  host_vector<DstType> h_zero(dst_count, DstType(0));

  sycl::queue queue;
  SrcType* d_in = static_cast<SrcType*>(malloc_device(src_count * sizeof(SrcType), queue));
  DstType* d_out = static_cast<DstType*>(malloc_device(dst_count * sizeof(DstType), queue));
  queue.memcpy(d_in, h_in.data(), src_count * sizeof(SrcType)).wait();
  queue.memcpy(d_out, h_zero.data(), dst_count * sizeof(DstType)).wait();

  auto gmem_layout_src = make_layout(make_shape(M, N), LayoutRight{});
  auto gmem_layout_dst = make_layout(make_shape(M, N_dst), LayoutRight{});

  Tensor gA = make_tensor(make_gmem_ptr<SrcType>(d_in), gmem_layout_src);
  Tensor gB = make_tensor(make_gmem_ptr<DstType>(d_out), gmem_layout_dst);

  auto adma_load = cute::make_adma_copy<SrcType>(
      XE4_ADMA_LOAD{}, gA, SrcSLayout{}, product_each(shape(SrcSLayout{})), Int<1>{});
  auto adma_store = cute::make_adma_copy<DstType>(
      XE4_ADMA_STORE{}, gB, DstSLayout{}, product_each(shape(DstSLayout{})), Int<1>{});

  sycl::nd_range<3> range(sycl::range<3>(1,1,32), sycl::range<3>(1,1,32));

  queue.parallel_for(range, [=](sycl::nd_item<3> item) {
    tcvd_ldsm_kernel<SrcType, DstType, TcvdDstType, TcvdSrcType, N, N_tile, N_dst,
                     SrcSLayout, DstSLayout>(
        item, d_in, d_out, SrcSLayout{}, DstSLayout{}, adma_load, adma_store);
  }).wait();

  host_vector<DstType> h_out(dst_count, DstType(0));
  queue.memcpy(h_out.data(), d_out, dst_count * sizeof(DstType)).wait();

  cutlass::xe4::validate_quantize_output<SrcType, DstType, TcvdDstType, TcvdSrcType,
                                          HostRefType, M, N, N, N_dst>(h_out, h_in);

  sycl::free(d_in, queue);
  sycl::free(d_out, queue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test cases: float -> bf8 (E5M2)
// N=32: 32 src floats (Vlen=8 ✓), 32 dst bytes (Vlen=32 ✓)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_bf8_N32_Nt1)  { run_ldsm_quantize_test<float, uint8_t, bf8, float, bf8, 32, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_bf8_N32_Nt2)  { run_ldsm_quantize_test<float, uint8_t, bf8, float, bf8, 32, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_bf8_N32_Nt4)  { run_ldsm_quantize_test<float, uint8_t, bf8, float, bf8, 32, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_bf8_N32_Nt8)  { run_ldsm_quantize_test<float, uint8_t, bf8, float, bf8, 32, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_bf8_N32_Nt16) { run_ldsm_quantize_test<float, uint8_t, bf8, float, bf8, 32, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_bf8_N32_Nt32) { run_ldsm_quantize_test<float, uint8_t, bf8, float, bf8, 32, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// float -> hf8 (E4M3)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_hf8_N32_Nt1)  { run_ldsm_quantize_test<float, uint8_t, hf8, float, hf8, 32, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_hf8_N32_Nt2)  { run_ldsm_quantize_test<float, uint8_t, hf8, float, hf8, 32, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_hf8_N32_Nt4)  { run_ldsm_quantize_test<float, uint8_t, hf8, float, hf8, 32, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_hf8_N32_Nt8)  { run_ldsm_quantize_test<float, uint8_t, hf8, float, hf8, 32, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_hf8_N32_Nt16) { run_ldsm_quantize_test<float, uint8_t, hf8, float, hf8, 32, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_hf8_N32_Nt32) { run_ldsm_quantize_test<float, uint8_t, hf8, float, hf8, 32, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// float -> fp4_e2m1 (E2M1)
// N=64: 64 src floats (Vlen=8 ✓), 32 dst bytes (Vlen=32 ✓). N_tile >= 2 for fp4.
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_fp4_N64_Nt2)  { run_ldsm_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 32, 64, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_fp4_N64_Nt4)  { run_ldsm_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 32, 64, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_fp4_N64_Nt8)  { run_ldsm_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 32, 64, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_fp4_N64_Nt16) { run_ldsm_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 32, 64, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, f32_to_fp4_N64_Nt32) { run_ldsm_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 32, 64, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// bf16 -> bf8 (E5M2)
// N=32: 32 src uint16_t (Vlen=16 ✓), 32 dst bytes (Vlen=32 ✓)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_bf8_N32_Nt1)  { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_bf8_N32_Nt2)  { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_bf8_N96_Nt3)  { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 96, 3>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_bf8_N32_Nt4)  { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_bf8_N32_Nt8)  { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_bf8_N32_Nt16) { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_bf8_N32_Nt32) { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// bf16 -> hf8 (E4M3)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_hf8_N32_Nt1)  { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_hf8_N32_Nt2)  { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_hf8_N96_Nt3)  { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 96, 3>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_hf8_N32_Nt4)  { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_hf8_N32_Nt8)  { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_hf8_N32_Nt16) { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_hf8_N32_Nt32) { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 32, 32>(); }


////////////////////////////////////////////////////////////////////////////////////////////////////
// bf16 -> fp4_e2m1 (E2M1)
// N=64: 64 src uint16_t (Vlen=16 ✓), 32 dst bytes (Vlen=32 ✓). N_tile >= 2 for fp4.
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_fp4_N64_Nt2)  { run_ldsm_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 32, 64, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_fp4_N64_Nt4)  { run_ldsm_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 32, 64, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_fp4_N64_Nt8)  { run_ldsm_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 32, 64, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_fp4_N64_Nt16) { run_ldsm_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 32, 64, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, bf16_to_fp4_N64_Nt32) { run_ldsm_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 32, 64, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// fp16 -> bf8 (E5M2)
// N=32: 32 src uint16_t (Vlen=16 ✓), 32 dst bytes (Vlen=32 ✓)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_bf8_N32_Nt1)  { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 32, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_bf8_N32_Nt2)  { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 32, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_bf8_N96_Nt3)  { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 32, 96, 3>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_bf8_N32_Nt4)  { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 32, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_bf8_N32_Nt8)  { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 32, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_bf8_N32_Nt16) { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 32, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_bf8_N32_Nt32) { run_ldsm_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 32, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// fp16 -> hf8 (E4M3)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_hf8_N32_Nt1)  { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 32, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_hf8_N32_Nt2)  { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 32, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_hf8_N96_Nt3)  { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 32, 96, 3>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_hf8_N32_Nt4)  { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 32, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_hf8_N32_Nt8)  { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 32, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_hf8_N32_Nt16) { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 32, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_hf8_N32_Nt32) { run_ldsm_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 32, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// fp16 -> fp4_e2m1 (E2M1)
// N=64: 64 src uint16_t (Vlen=16 ✓), 32 dst bytes (Vlen=32 ✓). N_tile >= 2 for fp4.
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_fp4_N64_Nt2)  { run_ldsm_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 32, 64, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_fp4_N64_Nt4)  { run_ldsm_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 32, 64, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_fp4_N64_Nt8)  { run_ldsm_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 32, 64, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_fp4_N64_Nt16) { run_ldsm_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 32, 64, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM, fp16_to_fp4_N64_Nt32) { run_ldsm_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 32, 64, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Multi-warp LDSM quantize tests (M=64, M=128)
//
// These tests use MMA-aware make_ldsm_copy_C for the source load, mimicking the
// real epilogue path where MMA accumulator output stored in SLM is loaded back to
// registers for quantization. Each warp handles 32 rows independently; the
// multi-warp tiling is handled by the MMA-aware LDSM API.
//
// Flow: GMEM --(ADMA load)--> SLM(src) --(MMA-aware LDSM S2R)--> Registers(wide)
//             --(TCVD in N_tile slices)--> Registers(narrow)
//             --(LDSM R2S)--> SLM(dst) --(ADMA store)--> GMEM(narrow)
//
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// Device kernel: ADMA G2S → MMA-aware LDSM S2R → TCVD → LDSM R2S → ADMA S2G
//
// Multi-warp: M can be 64, 128, etc. ThrGroupSize warps run in parallel, each
// handling a 32-row chunk. The MMA-aware LDSM copy_C distributes work across warps
// based on the TiledMMA's atom shape.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType, typename TcvdDstType, typename TcvdSrcType,
          int M, int N, int N_tile, int ThrGroupSize,
          class SrcSmemLayout, class DstSmemLayout,
          class TiledMMA,
          class AdmaLoad, class AdmaStore>
CUTLASS_GLOBAL void
tcvd_ldsm_mma_kernel(sycl::nd_item<3>& item,
                     SrcType* g_in, DstType* g_out,
                     SrcSmemLayout src_smem_layout, DstSmemLayout dst_smem_layout,
                     AdmaLoad adma_ld, AdmaStore adma_st)
{
  using namespace cute;
  namespace api = cutlass::xe4;

  constexpr int N_dst = N * ::sizeof_bits<TcvdDstType>() / 8;

  adma_ld.set_tensor_desc(allocate_tdesc<0>());
  adma_st.set_tensor_desc(allocate_tdesc<1>());

  constexpr auto smem_size = cute::cosize_v<SrcSmemLayout> * sizeof(SrcType)
                           + cute::cosize_v<DstSmemLayout> * sizeof(DstType);
#if defined(__SYCL_DEVICE_ONLY__)
  auto shared_memory = alloc_slm_buffer<uint8_t, smem_size>(item.get_group());
#endif
#if defined(CUTLASS_ENABLE_SYCL) && !defined(__SYCL_DEVICE_ONLY__)
  char* shared_memory;
#endif

  using Storage = SharedStorage<SrcType, SrcSmemLayout, DstType, DstSmemLayout>;
  Storage& shared_storage = *reinterpret_cast<Storage*>(shared_memory);

  Tensor t_smem_src = recast<SrcType>(
      make_tensor(make_smem_ptr(shared_storage.smem_src.begin()), src_smem_layout));
  Tensor t_smem_dst = recast<DstType>(
      make_tensor(make_smem_ptr(shared_storage.smem_dst.begin()), dst_smem_layout));

  int tid = ThreadIdxX();

  uint64_t* load_mbar = allocate_abar<0>();
  uint64_t* store_mbar = allocate_abar<1>();

  // Only thread 0 initializes barriers and triggers ADMA (multi-warp safe)
  bool elected = (tid == 0);
  if (elected) {
    xe4_initialize_barrier(load_mbar[0], 1);
    xe4_initialize_barrier(store_mbar[0], 1);
  }
  item.barrier(sycl::access::fence_space::local_space);

  // Step 1: GMEM → SLM(src) via ADMA
  api::adma_load(t_smem_src, adma_ld, src_smem_layout, load_mbar, elected, item);

  // Step 2: SLM(src) → Registers via MMA-aware LDSM copy_C
  TiledMMA mma{};
  auto tXrX = api::load_matrix<ThrGroupSize>(mma, t_smem_src, src_smem_layout, tid, item);

  // Step 3: TCVD — downconvert in N_tile slices
  // Per-warp geometry: each warp handles 32 rows, WarpTileN source columns
  constexpr int NumWarpsAlongM = M / 32;
  constexpr int NumWarpsAlongN = ThrGroupSize / NumWarpsAlongM;
  constexpr int WarpTileN_src = N / NumWarpsAlongN;
  constexpr int WarpTileN_dst = N_dst / NumWarpsAlongN;
  static_assert(WarpTileN_src % N_tile == 0, "N_tile must evenly divide WarpTileN_src");

  constexpr int num_col_slices = WarpTileN_src / N_tile;
  constexpr int N_dst_bytes_per_tile = N_tile * ::sizeof_bits<TcvdDstType>() / 8;

  DstType dst_reg[WarpTileN_dst];
  memset(dst_reg, 0, sizeof(dst_reg));

  for (int cs = 0; cs < num_col_slices; ++cs) {
    auto src_slice = make_tensor(make_rmem_ptr(&tXrX(cs * N_tile)),
                                 make_shape(Int<N_tile>{}));
    DstType tile_dst[N_dst_bytes_per_tile];
    auto dst_tensor = make_tensor(make_rmem_ptr(tile_dst),
                                  make_shape(Int<N_dst_bytes_per_tile>{}));
    cute::tensor_pipe_quantize<TcvdDstType, TcvdSrcType>(src_slice, dst_tensor);

    for (int i = 0; i < N_dst_bytes_per_tile; ++i) {
      dst_reg[cs * N_dst_bytes_per_tile + i] = tile_dst[i];
    }
  }
  sycl::group_barrier(item.get_group());

  // Step 4: Registers → SLM(dst) via geometry-only LDSM copy_D
  api::store_matrix<DstType, WarpTileN_dst, ThrGroupSize>(
      t_smem_dst, dst_reg, dst_smem_layout, tid, item);

  // Step 5: SLM(dst) → GMEM via ADMA
  api::adma_store(t_smem_dst, adma_st, dst_smem_layout, store_mbar, elected);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Host-side test runner for multi-warp LDSM quantize
//
// M must be a multiple of 32. ThrGroupSize controls the number of warps (32 threads each).
// Warp distribution: NumWarpsAlongM = M/32, NumWarpsAlongN = ThrGroupSize / NumWarpsAlongM.
// For M=32 with ThrGroupSize > 1, warps are distributed along N (each warp handles a column slice).
// The MMA is constructed to match the source tile (M×N) so make_ldsm_copy_C can
// derive proper warp distribution from the MMA atom shape.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType, typename TcvdDstType, typename TcvdSrcType,
          typename HostRefType, int M, int N, int N_tile, int ThrGroupSize = M / 32,
          typename MmaAB = bf16, typename MmaCD = float, int MmaK = 16>
void run_ldsm_mma_quantize_test()
{
  static_assert(M >= 32, "M must be at least 32 (subgroup size)");
  static_assert(M % 32 == 0, "M must be a multiple of 32");
  static_assert(N % N_tile == 0, "N must be evenly divisible by N_tile");
  static_assert(ThrGroupSize >= 1, "ThrGroupSize must be at least 1");
  static_assert(ThrGroupSize % (M / 32) == 0, "ThrGroupSize must be divisible by M/32");
  constexpr int N_dst = N * ::sizeof_bits<TcvdDstType>() / 8;

  using SrcSLayout = decltype(make_layout(make_shape(Int<M>{}, Int<N>{}), LayoutRight{}));
  using DstSLayout = decltype(make_layout(make_shape(Int<M>{}, Int<N_dst>{}), LayoutRight{}));

  // Construct TiledMMA matching source tile shape for MMA-aware LDSM distribution
  using MMAOp = XE4_AMMA<MmaCD, MmaAB, MmaAB, MmaCD, M, N, MmaK,
                          AMMA::Major::K, AMMA::Major::K>;
  using SrcTiledMMA = decltype(make_tiled_mma(MMA_Atom<MMA_Traits<MMAOp>>{}));

  constexpr int src_count = M * N;
  constexpr int dst_count = M * N_dst;

  host_vector<SrcType> h_in(src_count, SrcType(0));
  cutlass::xe4::fill_source_data<SrcType, TcvdSrcType, M, N>(h_in);

  host_vector<DstType> h_zero(dst_count, DstType(0));

  sycl::queue queue;
  SrcType* d_in = static_cast<SrcType*>(malloc_device(src_count * sizeof(SrcType), queue));
  DstType* d_out = static_cast<DstType*>(malloc_device(dst_count * sizeof(DstType), queue));
  queue.memcpy(d_in, h_in.data(), src_count * sizeof(SrcType)).wait();
  queue.memcpy(d_out, h_zero.data(), dst_count * sizeof(DstType)).wait();

  auto gmem_layout_src = make_layout(make_shape(M, N), LayoutRight{});
  auto gmem_layout_dst = make_layout(make_shape(M, N_dst), LayoutRight{});

  Tensor gA = make_tensor(make_gmem_ptr<SrcType>(d_in), gmem_layout_src);
  Tensor gB = make_tensor(make_gmem_ptr<DstType>(d_out), gmem_layout_dst);

  auto adma_load = cute::make_adma_copy<SrcType>(
      XE4_ADMA_LOAD{}, gA, SrcSLayout{}, product_each(shape(SrcSLayout{})), Int<1>{});
  auto adma_store = cute::make_adma_copy<DstType>(
      XE4_ADMA_STORE{}, gB, DstSLayout{}, product_each(shape(DstSLayout{})), Int<1>{});

  sycl::nd_range<3> range(sycl::range<3>(1, 1, 32 * ThrGroupSize),
                          sycl::range<3>(1, 1, 32 * ThrGroupSize));

  queue.parallel_for(range, [=](sycl::nd_item<3> item) {
    tcvd_ldsm_mma_kernel<SrcType, DstType, TcvdDstType, TcvdSrcType,
                         M, N, N_tile, ThrGroupSize,
                         SrcSLayout, DstSLayout, SrcTiledMMA>(
        item, d_in, d_out, SrcSLayout{}, DstSLayout{}, adma_load, adma_store);
  }).wait();

  host_vector<DstType> h_out(dst_count, DstType(0));
  queue.memcpy(h_out.data(), d_out, dst_count * sizeof(DstType)).wait();

  cutlass::xe4::validate_quantize_output<SrcType, DstType, TcvdDstType, TcvdSrcType,
                                          HostRefType, M, N, N, N_dst>(h_out, h_in);

  sycl::free(d_in, queue);
  sycl::free(d_out, queue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: float -> bf8 (E5M2)
// M=64 (ThrGroupSize=2, 64 threads), N=32
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M64_N32_Nt1)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 64, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M64_N32_Nt2)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 64, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M64_N32_Nt4)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 64, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M64_N32_Nt8)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 64, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M64_N32_Nt16) { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 64, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M64_N32_Nt32) { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 64, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// M=128 (ThrGroupSize=4, 128 threads), N=32
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M128_N32_Nt1)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 128, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M128_N32_Nt2)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 128, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M128_N32_Nt4)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 128, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M128_N32_Nt8)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 128, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M128_N32_Nt16) { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 128, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M128_N32_Nt32) { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 128, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: float -> hf8 (E4M3)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M64_N32_Nt1)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 64, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M64_N32_Nt2)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 64, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M64_N32_Nt4)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 64, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M64_N32_Nt8)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 64, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M64_N32_Nt16) { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 64, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M64_N32_Nt32) { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 64, 32, 32>(); }

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M128_N32_Nt1)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 128, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M128_N32_Nt2)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 128, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M128_N32_Nt4)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 128, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M128_N32_Nt8)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 128, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M128_N32_Nt16) { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 128, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M128_N32_Nt32) { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 128, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: float -> fp4_e2m1 (E2M1)
// N=64 required: N_dst = N/2 = 32 (satisfies LDSM Vlen=32 for uint8_t)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M64_N64_Nt2)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 64, 64, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M64_N64_Nt4)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 64, 64, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M64_N64_Nt8)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 64, 64, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M64_N64_Nt16) { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 64, 64, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M64_N64_Nt32) { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 64, 64, 32>(); }

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M128_N64_Nt2)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 128, 64, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M128_N64_Nt4)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 128, 64, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M128_N64_Nt8)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 128, 64, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M128_N64_Nt16) { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 128, 64, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M128_N64_Nt32) { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 128, 64, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: bf16 -> bf8 (E5M2)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M64_N32_Nt1)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 64, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M64_N32_Nt2)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 64, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M64_N96_Nt3)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 64, 96, 3>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M64_N32_Nt4)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 64, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M64_N32_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 64, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M64_N32_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 64, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M64_N32_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 64, 32, 32>(); }

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M128_N32_Nt1)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 128, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M128_N32_Nt2)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 128, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M128_N96_Nt3)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 128, 96, 3>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M128_N32_Nt4)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 128, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M128_N32_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 128, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M128_N32_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 128, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M128_N32_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 128, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: bf16 -> hf8 (E4M3)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M64_N32_Nt1)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 64, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M64_N32_Nt2)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 64, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M64_N96_Nt3)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 64, 96, 3>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M64_N32_Nt4)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 64, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M64_N32_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 64, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M64_N32_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 64, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M64_N32_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 64, 32, 32>(); }

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M128_N32_Nt1)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 128, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M128_N32_Nt2)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 128, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M128_N96_Nt3)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 128, 96, 3>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M128_N32_Nt4)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 128, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M128_N32_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 128, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M128_N32_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 128, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M128_N32_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 128, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: bf16 -> fp4_e2m1 (E2M1)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M64_N64_Nt2)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 64, 64, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M64_N64_Nt4)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 64, 64, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M64_N64_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 64, 64, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M64_N64_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 64, 64, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M64_N64_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 64, 64, 32>(); }

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M128_N64_Nt2)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 128, 64, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M128_N64_Nt4)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 128, 64, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M128_N64_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 128, 64, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M128_N64_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 128, 64, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M128_N64_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 128, 64, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: fp16 -> bf8 (E5M2)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M64_N32_Nt1)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 64, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M64_N32_Nt2)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 64, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M64_N96_Nt3)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 64, 96, 3>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M64_N32_Nt4)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 64, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M64_N32_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 64, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M64_N32_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 64, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M64_N32_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 64, 32, 32>(); }

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M128_N32_Nt1)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 128, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M128_N32_Nt2)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 128, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M128_N96_Nt3)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 128, 96, 3>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M128_N32_Nt4)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 128, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M128_N32_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 128, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M128_N32_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 128, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M128_N32_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 128, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: fp16 -> hf8 (E4M3)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M64_N32_Nt1)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 64, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M64_N32_Nt2)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 64, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M64_N96_Nt3)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 64, 96, 3>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M64_N32_Nt4)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 64, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M64_N32_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 64, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M64_N32_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 64, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M64_N32_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 64, 32, 32>(); }

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M128_N32_Nt1)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 128, 32, 1>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M128_N32_Nt2)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 128, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M128_N96_Nt3)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 128, 96, 3>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M128_N32_Nt4)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 128, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M128_N32_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 128, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M128_N32_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 128, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M128_N32_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 128, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: fp16 -> fp4_e2m1 (E2M1)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_fp4_M64_N64_Nt2)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 64, 64, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_fp4_M64_N64_Nt4)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 64, 64, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_fp4_M64_N64_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 64, 64, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_fp4_M64_N64_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 64, 64, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_fp4_M64_N64_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 64, 64, 32>(); }

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_fp4_M128_N64_Nt2)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 128, 64, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_fp4_M128_N64_Nt4)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 128, 64, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_fp4_M128_N64_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 128, 64, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_fp4_M128_N64_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 128, 64, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_fp4_M128_N64_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 128, 64, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests with M=32, ThrGroupSize > 1 (warps distributed along N)
// Mimics real epilogue where multiple warps share a 32-row tile, each handling
// a column slice. E.g., ThrGroupSize=2 → 2 warps, each handling N/2 columns.
////////////////////////////////////////////////////////////////////////////////////////////////////

// float -> bf8, M=32, ThrGroupSize=2 (2 warps along N, WarpTileN=16)
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M32_N32_TG2_Nt16) { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 32, 32, 16, 2>(); }
// float -> bf8, M=32, ThrGroupSize=4 (4 warps along N, WarpTileN=8)
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M32_N32_TG4_Nt8)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 32, 32, 8, 4>(); }

// float -> hf8, M=32, ThrGroupSize=2
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M32_N32_TG2_Nt16) { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 32, 32, 16, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M32_N32_TG4_Nt8)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 32, 32, 8, 4>(); }

// bf16 -> bf8, M=32, ThrGroupSize=2
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M32_N32_TG2_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 32, 16, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M32_N32_TG4_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 32, 8, 4>(); }

// bf16 -> hf8, M=32, ThrGroupSize=2
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M32_N32_TG2_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 32, 16, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M32_N32_TG4_Nt8)  { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 32, 8, 4>(); }

// fp16 -> bf8, M=32, ThrGroupSize=2
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_bf8_M32_N32_TG2_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, fp16, bf8, 32, 32, 16, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_hf8_M32_N32_TG2_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, fp16, hf8, 32, 32, 16, 2>(); }

// float -> fp4, M=32, ThrGroupSize=2 (N=64, WarpTileN=32 → WarpTileN_dst=16)
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M32_N64_TG2_Nt16) { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 32, 64, 16, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M32_N64_TG2_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 32, 64, 16, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, fp16_to_fp4_M32_N64_TG2_Nt16) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 32, 64, 16, 2>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Large tile tests: 128×256, 256×128, 256×256, 32×256, 32×128
//
// ThrGroupSize defaults to M/32. Warp distribution:
//   128×256: ThrGroupSize=4, NumWarpsAlongM=4, NumWarpsAlongN=1, WarpTileN=256
//   256×128: ThrGroupSize=8, NumWarpsAlongM=8, NumWarpsAlongN=1, WarpTileN=128
//   256×256: ThrGroupSize=8, NumWarpsAlongM=8, NumWarpsAlongN=1, WarpTileN=256
//   32×256:  ThrGroupSize=4, NumWarpsAlongM=1, NumWarpsAlongN=4, WarpTileN=64
//   32×128:  ThrGroupSize=2, NumWarpsAlongM=1, NumWarpsAlongN=2, WarpTileN=64
////////////////////////////////////////////////////////////////////////////////////////////////////

// --- 128 × 256 (ThrGroupSize=4, WarpTileN_src=256) ---

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M128_N256_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 128, 256, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M128_N256_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 128, 256, 32>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M128_N256_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 128, 256, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M128_N256_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 128, 256, 32>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M128_N256_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 128, 256, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M128_N256_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 128, 256, 32>(); }

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M128_N256_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 128, 256, 32>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M128_N256_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 128, 256, 32>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M128_N256_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 128, 256, 32>(); }

// --- 256 × 128 (ThrGroupSize=8, WarpTileN_src=128) ---

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M256_N128_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 256, 128, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M256_N128_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 256, 128, 32>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M256_N128_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 256, 128, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M256_N128_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 256, 128, 32>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M256_N128_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 256, 128, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M256_N128_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 256, 128, 32>(); }

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M256_N128_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 256, 128, 32>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M256_N128_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 256, 128, 32>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M256_N128_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 256, 128, 32>(); }

// --- 256 × 256 (ThrGroupSize=8, WarpTileN_src=256) ---

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M256_N256_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 256, 256, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M256_N256_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 256, 256, 32>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M256_N256_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 256, 256, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M256_N256_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 256, 256, 32>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M256_N256_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 256, 256, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M256_N256_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 256, 256, 32>(); }

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M256_N256_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 256, 256, 32>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M256_N256_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 256, 256, 32>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M256_N256_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 256, 256, 32>(); }

// --- 32 × 256 (ThrGroupSize=4, NumWarpsAlongN=4, WarpTileN_src=64) ---

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M32_N256_TG4_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 32, 256, 16, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M32_N256_TG4_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 32, 256, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M32_N256_TG4_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 32, 256, 16, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M32_N256_TG4_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 32, 256, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M32_N256_TG4_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 32, 256, 16, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M32_N256_TG4_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 32, 256, 32, 4>(); }

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M32_N256_TG4_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 256, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M32_N256_TG4_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 256, 32, 4>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M32_N256_TG4_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 32, 256, 32, 4>(); }

// --- 32 × 128 (ThrGroupSize=2, NumWarpsAlongN=2, WarpTileN_src=64) ---

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M32_N128_TG2_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 32, 128, 16, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_bf8_M32_N128_TG2_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, bf8, float, bf8, 32, 128, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M32_N128_TG2_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 32, 128, 16, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_hf8_M32_N128_TG2_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, hf8, float, hf8, 32, 128, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M32_N128_TG2_Nt16)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 32, 128, 16, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, f32_to_fp4_M32_N128_TG2_Nt32)  { run_ldsm_mma_quantize_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 32, 128, 32, 2>(); }

TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_bf8_M32_N128_TG2_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 128, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_hf8_M32_N128_TG2_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 128, 32, 2>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_LDSM_MMA, bf16_to_fp4_M32_N128_TG2_Nt32) { run_ldsm_mma_quantize_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 32, 128, 32, 2>(); }
