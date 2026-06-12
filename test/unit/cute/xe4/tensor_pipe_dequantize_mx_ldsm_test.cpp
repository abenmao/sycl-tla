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

// Unit tests for TensorPipe Dequantize with MX scaling (TCVUMX) using LDSM for register load/store.
//
// Flow: GMEM(narrow) --(ADMA load)--> SLM(src) --(LDSM load)--> Registers(narrow)
//                    --(TCVUMX in N_tile slices with e8m0 meta)--> Registers(wide)
//                    --(LDSM store)--> SLM(dst) --(ADMA store)--> GMEM(wide)
//
// TCVUMX operation: dst = src_value * 2^(meta - 127)
//   where meta is a per-row uint8_t e8m0 scale factor (.mxnd mode).
//
// This is the inverse of TCVDMX (quantize): narrow → wide with MX scaling.
//
// LDSM Vlen constraints (minimum columns):
//   32-bit (float): 8 | 16-bit (bf16/fp16): 16 | 8-bit (uint8_t): 32
//
// For bf8/hf8 src (1 byte/elem):  N_src_bytes = N, need N >= 32
// For fp4 src (0.5 byte/elem):    N_src_bytes = N/2, need N >= 64
// Destination (bf16/fp16, 2 bytes/elem): N_dst_elems = N, need N >= 16 (always satisfied)
//
// N_tile=3 limitation for fp4 source:
// - Odd N_tile (1, 3) not supported because N_tile * 4 bits is not byte-aligned
//   (e.g. 3*4=12 bits=1.5 bytes). The test harness computes N_src_bytes_per_tile = N_tile * 4 / 8
//   using integer division which truncates, and the kernel loop cannot correctly stitch
//   sub-byte nibbles across tile boundaries.
//

#include "cutlass_unit_test.h"
#include "tensor_pipe_quantize_dequantize_api.hpp"
#include <cute/atom/mma_traits_xe4_amma.hpp>
#include <iostream>
#include <cstring>
#include <cmath>

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shared storage: separate src (narrow) and dst (wide) SLM buffers.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class SrcType, class SrcSmemLayout, class DstType, class DstSmemLayout>
struct SharedStorage
{
  cute::ArrayEngine<SrcType, cute::cosize_v<SrcSmemLayout>> smem_src;
  cute::ArrayEngine<DstType, cute::cosize_v<DstSmemLayout>> smem_dst;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Section 1: Basic .mxnd (per-row scaling) dequantization
//
// tcvumx.toty.fromty.m32nN.mxnd dst, src, m
//   - dst:  OUTPUT wider type (bf16/f16) in registers
//   - src:  INPUT narrow MX type (bf8/hf8/fp4) in registers
//   - m:    INPUT meta (pre-computed e8m0 scale, per-row)
//
// Validation: dst[row][col] = src_value[row][col] * 2^(meta[row] - 127)
//   where src_value is the float representation of the narrow-type element.
//
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// Device kernel: ADMA G2S → LDSM S2R → TCVUMX (dequantize) → LDSM R2S → ADMA S2G
//
// SrcType: GMEM/SLM source storage type (uint8_t for packed narrow types)
// DstType: GMEM/SLM destination storage type (uint16_t for bf16/fp16)
// TcvumxDstType: TCVUMX semantic dest type (bf16, fp16) — selects pISA .toty
// TcvumxSrcType: TCVUMX semantic src type (bf8, hf8, fp4_e2m1) — selects pISA .fromty
// N: number of source elements per row
// N_tile: per-TCVUMX slice width (must match supported tcvumx.*.m32nN dimension)
// N_src: source bytes per row = N * sizeof_bits<TcvumxSrcType> / 8
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType, typename TcvumxDstType, typename TcvumxSrcType,
          int N, int N_tile, int N_src,
          class SrcSmemLayout, class DstSmemLayout,
          class AdmaLoad, class AdmaStore>
CUTLASS_GLOBAL void
tcvumx_ldsm_kernel(sycl::nd_item<3>& item,
                   SrcType* g_in, DstType* g_out, const uint8_t* g_meta,
                   SrcSmemLayout src_smem_layout, DstSmemLayout dst_smem_layout,
                   AdmaLoad adma_ld, AdmaStore adma_st)
{
  using namespace cute;
  namespace api = cutlass::xe4;

  constexpr int M = 32;
  // N_dst_elems = N (one wide element per source element)
  constexpr int N_dst = N;

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

  // Step 1: GMEM(narrow) → SLM(src) via ADMA
  api::adma_load(t_smem_src, adma_ld, src_smem_layout, load_mbar, elected, item);

  // Step 2: SLM(src) → Registers via LDSM (narrow data)
  auto tXrX = api::load_matrix(t_smem_src, src_smem_layout, tid, item);

  // Load per-row meta — each thread's lane determines which meta to use
  uint8_t meta = g_meta[tid];

  // Step 3: TCVUMX — upconvert with MX scaling in N_tile slices
  constexpr int num_col_slices = N / N_tile;
  // Destination bytes per tile: N_tile elements × 2 bytes (bf16/fp16)
  constexpr int N_dst_bytes_per_tile = N_tile * sizeof(DstType);

  // Source registers per tile: ceil(N_tile * bits_per_src_elem / 32)
  constexpr int N_reg_src = (N_tile * ::sizeof_bits<TcvumxSrcType>() / BITS_PER_BYTE
                             + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  // Destination registers per tile: ceil(N_tile * 16 / 32) = ceil(N_tile / 2)
  constexpr int N_reg_dst = (N_tile * sizeof(DstType) + sizeof(uint32_t) - 1) / sizeof(uint32_t);

  // Source bytes per element for indexing into the narrow register file
  constexpr int src_bits_per_elem = ::sizeof_bits<TcvumxSrcType>();
  constexpr int src_bytes_per_tile = N_tile * src_bits_per_elem / BITS_PER_BYTE;

  DstType dst_reg[N_dst];
  memset(dst_reg, 0, sizeof(dst_reg));

  for (int cs = 0; cs < num_col_slices; ++cs) {
    // Pack source narrow elements into uint32 registers
    uint32_t src_u32[N_reg_src];
    memset(src_u32, 0, sizeof(src_u32));
    SrcType src_narrow[src_bytes_per_tile];
    for (int j = 0; j < src_bytes_per_tile; ++j) {
      src_narrow[j] = tXrX(cs * src_bytes_per_tile + j);
    }
    memcpy(src_u32, src_narrow, src_bytes_per_tile);

    // TCVUMX: upconvert narrow → wide with MX scaling
    uint32_t dst_u32[N_reg_dst];
    memset(dst_u32, 0, sizeof(dst_u32));
    gtp_tcvumx<TcvumxDstType, TcvumxSrcType, N_tile>(dst_u32, src_u32, meta);

    // Unpack wide destination from uint32 registers
    DstType tile_dst[N_tile];
    memcpy(tile_dst, dst_u32, N_tile * sizeof(DstType));

    for (int i = 0; i < N_tile; ++i) {
      dst_reg[cs * N_tile + i] = tile_dst[i];
    }
  }
  sycl::group_barrier(item.get_group());

  // Step 4: Registers → SLM(dst) via LDSM store (wide data)
  api::store_matrix<DstType, N_dst>(t_smem_dst, dst_reg, dst_smem_layout, tid, item);

  // Step 5: SLM(dst) → GMEM via ADMA
  api::adma_store(t_smem_dst, adma_st, dst_smem_layout, store_mbar, elected);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test runner for TCVUMX (.mxnd per-row dequantization)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType, typename TcvumxDstType, typename TcvumxSrcType,
          typename HostRefType, int M, int N, int N_tile>
void run_ldsm_tcvumx_test()
{
  static_assert(M == 32, "TCVUMX requires M=32 (subgroup size)");
  static_assert(N % N_tile == 0, "N must be evenly divisible by N_tile");

  // Source bytes per row (narrow packed data)
  constexpr int N_src = N * ::sizeof_bits<TcvumxSrcType>() / BITS_PER_BYTE;
  // Destination: one wide element per source element
  constexpr int N_dst = N;

  using SrcSLayout = decltype(make_layout(make_shape(Int<M>{}, Int<N_src>{}), LayoutRight{}));
  using DstSLayout = decltype(make_layout(make_shape(Int<M>{}, Int<N_dst>{}), LayoutRight{}));

  constexpr int src_count = M * N_src;
  constexpr int dst_count = M * N_dst;

  // Per-row meta: cycle through scale exponents
  host_vector<uint8_t> h_meta(M, uint8_t(0));
  cutlass::xe4::fill_meta_per_row<M>(h_meta);

  // Source data: narrow-type values that will be upconverted
  host_vector<SrcType> h_in(src_count, SrcType(0));
  cutlass::xe4::fill_lower_precision_source_data_mx<SrcType, TcvumxSrcType, M, N, N_src>(h_in);

  host_vector<DstType> h_zero(dst_count, DstType(0));

  sycl::queue queue;
  SrcType* d_in = static_cast<SrcType*>(malloc_device(src_count * sizeof(SrcType), queue));
  DstType* d_out = static_cast<DstType*>(malloc_device(dst_count * sizeof(DstType), queue));
  uint8_t* d_meta = static_cast<uint8_t*>(malloc_device(M * sizeof(uint8_t), queue));
  queue.memcpy(d_in, h_in.data(), src_count * sizeof(SrcType)).wait();
  queue.memcpy(d_out, h_zero.data(), dst_count * sizeof(DstType)).wait();
  queue.memcpy(d_meta, h_meta.data(), M * sizeof(uint8_t)).wait();

  auto gmem_layout_src = make_layout(make_shape(M, N_src), LayoutRight{});
  auto gmem_layout_dst = make_layout(make_shape(M, N_dst), LayoutRight{});

  Tensor gA = make_tensor(make_gmem_ptr<SrcType>(d_in), gmem_layout_src);
  Tensor gB = make_tensor(make_gmem_ptr<DstType>(d_out), gmem_layout_dst);

  auto adma_load = cute::make_adma_copy<SrcType>(
      XE4_ADMA_LOAD{}, gA, SrcSLayout{}, product_each(shape(SrcSLayout{})), Int<1>{});
  auto adma_store = cute::make_adma_copy<DstType>(
      XE4_ADMA_STORE{}, gB, DstSLayout{}, product_each(shape(DstSLayout{})), Int<1>{});

  sycl::nd_range<3> range(sycl::range<3>(1,1,32), sycl::range<3>(1,1,32));

  queue.parallel_for(range, [=](sycl::nd_item<3> item) {
    tcvumx_ldsm_kernel<SrcType, DstType, TcvumxDstType, TcvumxSrcType, N, N_tile, N_src,
                       SrcSLayout, DstSLayout>(
        item, d_in, d_out, d_meta, SrcSLayout{}, DstSLayout{}, adma_load, adma_store);
  }).wait();

  host_vector<DstType> h_out(dst_count, DstType(0));
  queue.memcpy(h_out.data(), d_out, dst_count * sizeof(DstType)).wait();

  cutlass::xe4::validate_tcvumx_output<SrcType, DstType, TcvumxDstType, TcvumxSrcType,
                          HostRefType, M, N, N_src, N_dst>(h_out, h_in, h_meta);

  sycl::free(d_in, queue);
  sycl::free(d_out, queue);
  sycl::free(d_meta, queue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Section 1 tests: bf8 (e5m2) → bf16, per-row scaling (.mxnd)
// N=32 (src: 32 uint8_t Vlen=32 ✓, dst: 32 uint16_t Vlen=16 ✓)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TCVUMX, bf8_to_bf16_N32_Nt1)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 32, 32, 1>(); }
TEST(XE4_TCVUMX, bf8_to_bf16_N32_Nt2)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 32, 32, 2>(); }
TEST(XE4_TCVUMX, bf8_to_bf16_N96_Nt3)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 32, 96, 3>(); }
TEST(XE4_TCVUMX, bf8_to_bf16_N32_Nt4)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 32, 32, 4>(); }
TEST(XE4_TCVUMX, bf8_to_bf16_N32_Nt8)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 32, 32, 8>(); }
TEST(XE4_TCVUMX, bf8_to_bf16_N32_Nt16) { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 32, 32, 16>(); }
TEST(XE4_TCVUMX, bf8_to_bf16_N32_Nt32) { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 32, 32, 32>(); }

// bf8 → fp16
TEST(XE4_TCVUMX, bf8_to_fp16_N32_Nt1)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 32, 32, 1>(); }
TEST(XE4_TCVUMX, bf8_to_fp16_N32_Nt2)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 32, 32, 2>(); }
TEST(XE4_TCVUMX, bf8_to_fp16_N96_Nt3)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 32, 96, 3>(); }
TEST(XE4_TCVUMX, bf8_to_fp16_N32_Nt4)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 32, 32, 4>(); }
TEST(XE4_TCVUMX, bf8_to_fp16_N32_Nt8)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 32, 32, 8>(); }
TEST(XE4_TCVUMX, bf8_to_fp16_N32_Nt16) { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 32, 32, 16>(); }
TEST(XE4_TCVUMX, bf8_to_fp16_N32_Nt32) { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 32, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// hf8 (e4m3) → bf16
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TCVUMX, hf8_to_bf16_N32_Nt1)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 32, 32, 1>(); }
TEST(XE4_TCVUMX, hf8_to_bf16_N32_Nt2)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 32, 32, 2>(); }
TEST(XE4_TCVUMX, hf8_to_bf16_N96_Nt3)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 32, 96, 3>(); }
TEST(XE4_TCVUMX, hf8_to_bf16_N32_Nt4)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 32, 32, 4>(); }
TEST(XE4_TCVUMX, hf8_to_bf16_N32_Nt8)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 32, 32, 8>(); }
TEST(XE4_TCVUMX, hf8_to_bf16_N32_Nt16) { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 32, 32, 16>(); }
TEST(XE4_TCVUMX, hf8_to_bf16_N32_Nt32) { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 32, 32, 32>(); }

// hf8 → fp16
TEST(XE4_TCVUMX, hf8_to_fp16_N32_Nt1)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 32, 32, 1>(); }
TEST(XE4_TCVUMX, hf8_to_fp16_N32_Nt2)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 32, 32, 2>(); }
TEST(XE4_TCVUMX, hf8_to_fp16_N96_Nt3)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 32, 96, 3>(); }
TEST(XE4_TCVUMX, hf8_to_fp16_N32_Nt4)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 32, 32, 4>(); }
TEST(XE4_TCVUMX, hf8_to_fp16_N32_Nt8)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 32, 32, 8>(); }
TEST(XE4_TCVUMX, hf8_to_fp16_N32_Nt16) { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 32, 32, 16>(); }
TEST(XE4_TCVUMX, hf8_to_fp16_N32_Nt32) { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 32, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// fp4 (e2m1) → bf16, N=64 (src: 32 bytes Vlen=32 ✓, dst: 64 uint16_t Vlen=16 ✓)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TCVUMX, fp4_to_bf16_N64_Nt2)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 32, 64, 2>(); }
TEST(XE4_TCVUMX, fp4_to_bf16_N64_Nt4)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 32, 64, 4>(); }
TEST(XE4_TCVUMX, fp4_to_bf16_N64_Nt8)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 32, 64, 8>(); }
TEST(XE4_TCVUMX, fp4_to_bf16_N64_Nt16) { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 32, 64, 16>(); }
TEST(XE4_TCVUMX, fp4_to_bf16_N64_Nt32) { run_ldsm_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 32, 64, 32>(); }

// fp4 → fp16
TEST(XE4_TCVUMX, fp4_to_fp16_N64_Nt2)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 32, 64, 2>(); }
TEST(XE4_TCVUMX, fp4_to_fp16_N64_Nt4)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 32, 64, 4>(); }
TEST(XE4_TCVUMX, fp4_to_fp16_N64_Nt8)  { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 32, 64, 8>(); }
TEST(XE4_TCVUMX, fp4_to_fp16_N64_Nt16) { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 32, 64, 16>(); }
TEST(XE4_TCVUMX, fp4_to_fp16_N64_Nt32) { run_ldsm_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 32, 64, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Section 2: MMA-aware multi-warp TCVUMX (.mxnd, per-row scaling)
//
// Same MMA-aware pattern as the quantize MMA tests. Demonstrates the full GEMM
// input dequantization path:
//
//   GMEM(narrow MX weights) → ADMA G2S → MMA-aware LDSM S2R → TCVUMX (dequant)
//     → LDSM R2S → SLM(wide) → ready for MMA consumption
//
// Multi-warp: M=64/128, ThrGroupSize=2/4.
//
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// MMA-aware kernel for TCVUMX (dequantization)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType, typename TcvumxDstType, typename TcvumxSrcType,
          int M, int N, int N_tile, int ThrGroupSize,
          class SrcSmemLayout, class DstSmemLayout,
          class SrcTiledMMA,
          class AdmaLoad, class AdmaStore>
CUTLASS_GLOBAL void
tcvumx_ldsm_mma_kernel(sycl::nd_item<3>& item,
                        SrcType* g_in, DstType* g_out, const uint8_t* g_meta,
                        SrcSmemLayout src_smem_layout, DstSmemLayout dst_smem_layout,
                        AdmaLoad adma_ld, AdmaStore adma_st)
{
  using namespace cute;
  namespace api = cutlass::xe4;

  constexpr int N_src = N * ::sizeof_bits<TcvumxSrcType>() / BITS_PER_BYTE;
  constexpr int N_dst = N;

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

  bool elected = (tid == 0);
  if (elected) {
    xe4_initialize_barrier(load_mbar[0], 1);
    xe4_initialize_barrier(store_mbar[0], 1);
  }
  item.barrier(sycl::access::fence_space::local_space);

  // Step 1: GMEM(narrow) → SLM(src) via ADMA
  api::adma_load(t_smem_src, adma_ld, src_smem_layout, load_mbar, elected, item);

  // Step 2: SLM(src) → Registers via MMA-aware LDSM copy_C (narrow data)
  SrcTiledMMA mma{};
  auto tXrX = api::load_matrix<ThrGroupSize>(mma, t_smem_src, src_smem_layout, tid, item);

  // Per-row meta for multi-warp
  int lane_id = tid % 32;
  int warp_id = tid / 32;
  constexpr int NumWarpsAlongM = M / 32;
  constexpr int NumWarpsAlongN = ThrGroupSize / NumWarpsAlongM;
  int warp_m = warp_id % NumWarpsAlongM;
  int row = warp_m * 32 + lane_id;
  uint8_t meta = g_meta[row];

  // Step 3: TCVUMX — upconvert with MX scaling in N_tile slices
  constexpr int WarpTileN_src_bytes = N_src / NumWarpsAlongN;
  constexpr int WarpTileN_elems = N / NumWarpsAlongN;
  static_assert(WarpTileN_elems % N_tile == 0, "N_tile must evenly divide WarpTileN_elems");

  constexpr int num_col_slices = WarpTileN_elems / N_tile;
  constexpr int src_bits_per_elem = ::sizeof_bits<TcvumxSrcType>();
  constexpr int src_bytes_per_tile = N_tile * src_bits_per_elem / BITS_PER_BYTE;
  constexpr int N_reg_src = (src_bytes_per_tile + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  constexpr int N_reg_dst = (N_tile * sizeof(DstType) + sizeof(uint32_t) - 1) / sizeof(uint32_t);

  DstType dst_reg[WarpTileN_elems];
  memset(dst_reg, 0, sizeof(dst_reg));

  for (int cs = 0; cs < num_col_slices; ++cs) {
    uint32_t src_u32[N_reg_src];
    memset(src_u32, 0, sizeof(src_u32));
    SrcType src_narrow[src_bytes_per_tile];
    for (int j = 0; j < src_bytes_per_tile; ++j) {
      src_narrow[j] = tXrX(cs * src_bytes_per_tile + j);
    }
    memcpy(src_u32, src_narrow, src_bytes_per_tile);

    uint32_t dst_u32[N_reg_dst];
    memset(dst_u32, 0, sizeof(dst_u32));
    gtp_tcvumx<TcvumxDstType, TcvumxSrcType, N_tile>(dst_u32, src_u32, meta);

    DstType tile_dst[N_tile];
    memcpy(tile_dst, dst_u32, N_tile * sizeof(DstType));

    for (int i = 0; i < N_tile; ++i) {
      dst_reg[cs * N_tile + i] = tile_dst[i];
    }
  }
  sycl::group_barrier(item.get_group());

  // Step 4: Registers → SLM(dst) via geometry-only LDSM copy_D (wide data)
  api::store_matrix<DstType, WarpTileN_elems, ThrGroupSize>(t_smem_dst, dst_reg, dst_smem_layout, tid, item);

  // Step 5: SLM(dst) → GMEM via ADMA
  api::adma_store(t_smem_dst, adma_st, dst_smem_layout, store_mbar, elected);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// MMA-aware test runner for TCVUMX
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType, typename TcvumxDstType, typename TcvumxSrcType,
          typename HostRefType, int M, int N, int N_tile, int ThrGroupSize = M / 32,
          typename MmaAB = bf16, typename MmaCD = float, int MmaK = 16>
void run_ldsm_mma_tcvumx_test()
{
  static_assert(M >= 32, "M must be at least 32 (subgroup size)");
  static_assert(M % 32 == 0, "M must be a multiple of 32");
  static_assert(N % N_tile == 0, "N must be evenly divisible by N_tile");
  static_assert(ThrGroupSize >= 1, "ThrGroupSize must be at least 1");
  static_assert(ThrGroupSize % (M / 32) == 0, "ThrGroupSize must be divisible by M/32");

  constexpr int N_src = N * ::sizeof_bits<TcvumxSrcType>() / BITS_PER_BYTE;
  constexpr int N_dst = N;

  using SrcSLayout = decltype(make_layout(make_shape(Int<M>{}, Int<N_src>{}), LayoutRight{}));
  using DstSLayout = decltype(make_layout(make_shape(Int<M>{}, Int<N_dst>{}), LayoutRight{}));

  // MMA matching SOURCE layout for MMA-aware LDSM copy_C
  using SrcMMAOp = XE4_AMMA<MmaCD, MmaAB, MmaAB, MmaCD, M, N_src, MmaK,
                             AMMA::Major::K, AMMA::Major::K>;
  using SrcTiledMMA = decltype(make_tiled_mma(MMA_Atom<MMA_Traits<SrcMMAOp>>{}));

  constexpr int src_count = M * N_src;
  constexpr int dst_count = M * N_dst;

  host_vector<uint8_t> h_meta(M, uint8_t(0));
  cutlass::xe4::fill_meta_per_row<M>(h_meta);

  host_vector<SrcType> h_in(src_count, SrcType(0));
  cutlass::xe4::fill_lower_precision_source_data_mx<SrcType, TcvumxSrcType, M, N, N_src>(h_in);

  host_vector<DstType> h_zero(dst_count, DstType(0));

  sycl::queue queue;
  SrcType* d_in = static_cast<SrcType*>(malloc_device(src_count * sizeof(SrcType), queue));
  DstType* d_out = static_cast<DstType*>(malloc_device(dst_count * sizeof(DstType), queue));
  uint8_t* d_meta = static_cast<uint8_t*>(malloc_device(M * sizeof(uint8_t), queue));
  queue.memcpy(d_in, h_in.data(), src_count * sizeof(SrcType)).wait();
  queue.memcpy(d_out, h_zero.data(), dst_count * sizeof(DstType)).wait();
  queue.memcpy(d_meta, h_meta.data(), M * sizeof(uint8_t)).wait();

  auto gmem_layout_src = make_layout(make_shape(M, N_src), LayoutRight{});
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
    tcvumx_ldsm_mma_kernel<SrcType, DstType, TcvumxDstType, TcvumxSrcType,
                            M, N, N_tile, ThrGroupSize,
                            SrcSLayout, DstSLayout, SrcTiledMMA>(
        item, d_in, d_out, d_meta, SrcSLayout{}, DstSLayout{}, adma_load, adma_store);
  }).wait();

  host_vector<DstType> h_out(dst_count, DstType(0));
  queue.memcpy(h_out.data(), d_out, dst_count * sizeof(DstType)).wait();

  cutlass::xe4::validate_tcvumx_output<SrcType, DstType, TcvumxDstType, TcvumxSrcType,
                          HostRefType, M, N, N_src, N_dst>(h_out, h_in, h_meta);

  sycl::free(d_in, queue);
  sycl::free(d_out, queue);
  sycl::free(d_meta, queue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: bf8 -> fp16
// M=64 (ThrGroupSize=2, 64 threads), N=32
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M64_N32_Nt1)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 64, 32, 1>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M64_N32_Nt2)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 64, 32, 2>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M64_N96_Nt3)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 64, 96, 3>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M64_N32_Nt4)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 64, 32, 4>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M64_N32_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 64, 32, 8>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M64_N32_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 64, 32, 16>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M64_N32_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 64, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// M=128 (ThrGroupSize=4, 128 threads), N=32
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M128_N32_Nt1)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 128, 32, 1>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M128_N32_Nt2)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 128, 32, 2>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M128_N96_Nt3)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 128, 96, 3>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M128_N32_Nt4)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 128, 32, 4>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M128_N32_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 128, 32, 8>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M128_N32_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 128, 32, 16>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M128_N32_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 128, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: bf8 -> bf16
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M64_N32_Nt1)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 64, 32, 1>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M64_N32_Nt2)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 64, 32, 2>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M64_N96_Nt3)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 64, 96, 3>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M64_N32_Nt4)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 64, 32, 4>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M64_N32_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 64, 32, 8>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M64_N32_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 64, 32, 16>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M64_N32_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 64, 32, 32>(); }

TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M128_N32_Nt1)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 128, 32, 1>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M128_N32_Nt2)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 128, 32, 2>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M128_N96_Nt3)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 128, 96, 3>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M128_N32_Nt4)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 128, 32, 4>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M128_N32_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 128, 32, 8>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M128_N32_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 128, 32, 16>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M128_N32_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 128, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: hf8 -> fp16
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M64_N32_Nt1)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 64, 32, 1>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M64_N32_Nt2)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 64, 32, 2>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M64_N96_Nt3)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 64, 96, 3>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M64_N32_Nt4)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 64, 32, 4>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M64_N32_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 64, 32, 8>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M64_N32_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 64, 32, 16>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M64_N32_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 64, 32, 32>(); }

TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M128_N32_Nt1)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 128, 32, 1>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M128_N32_Nt2)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 128, 32, 2>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M128_N96_Nt3)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 128, 96, 3>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M128_N32_Nt4)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 128, 32, 4>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M128_N32_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 128, 32, 8>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M128_N32_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 128, 32, 16>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M128_N32_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 128, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: hf8 -> bf16
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M64_N32_Nt1)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 64, 32, 1>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M64_N32_Nt2)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 64, 32, 2>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M64_N96_Nt3)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 64, 96, 3>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M64_N32_Nt4)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 64, 32, 4>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M64_N32_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 64, 32, 8>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M64_N32_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 64, 32, 16>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M64_N32_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 64, 32, 32>(); }

TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M128_N32_Nt1)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 128, 32, 1>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M128_N32_Nt2)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 128, 32, 2>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M128_N96_Nt3)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 128, 96, 3>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M128_N32_Nt4)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 128, 32, 4>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M128_N32_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 128, 32, 8>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M128_N32_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 128, 32, 16>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M128_N32_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 128, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: fp4_e2m1 -> fp16
// N=64: N_src = 32 bytes per row. MMA shape M × 32.
// N_tile >= 2 required (fp4 N_tile=1 produces 0 source bytes)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TCVUMX_MMA, fp4_to_fp16_M64_N64_Nt2)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 64, 64, 2>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_fp16_M64_N64_Nt4)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 64, 64, 4>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_fp16_M64_N64_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 64, 64, 8>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_fp16_M64_N64_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 64, 64, 16>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_fp16_M64_N64_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 64, 64, 32>(); }

TEST(XE4_TCVUMX_MMA, fp4_to_fp16_M128_N64_Nt2)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 128, 64, 2>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_fp16_M128_N64_Nt4)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 128, 64, 4>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_fp16_M128_N64_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 128, 64, 8>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_fp16_M128_N64_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 128, 64, 16>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_fp16_M128_N64_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 128, 64, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests: fp4_e2m1 -> bf16
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TCVUMX_MMA, fp4_to_bf16_M64_N64_Nt2)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 64, 64, 2>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_bf16_M64_N64_Nt4)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 64, 64, 4>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_bf16_M64_N64_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 64, 64, 8>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_bf16_M64_N64_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 64, 64, 16>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_bf16_M64_N64_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 64, 64, 32>(); }

TEST(XE4_TCVUMX_MMA, fp4_to_bf16_M128_N64_Nt2)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 128, 64, 2>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_bf16_M128_N64_Nt4)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 128, 64, 4>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_bf16_M128_N64_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 128, 64, 8>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_bf16_M128_N64_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 128, 64, 16>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_bf16_M128_N64_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 128, 64, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Multi-warp tests with M=32, ThrGroupSize > 1 (warps distributed along N)
// Mimics real prologue where multiple warps share a 32-row tile, each handling
// a column slice of the narrow source.
////////////////////////////////////////////////////////////////////////////////////////////////////

// bf8 -> fp16, M=32, ThrGroupSize=2 (2 warps along N, WarpTileSrc=16)
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M32_N32_TG2_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 32, 32, 16, 2>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M32_N32_TG4_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 32, 32, 8, 4>(); }

// bf8 -> bf16, M=32, ThrGroupSize=2
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M32_N32_TG2_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 32, 32, 16, 2>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M32_N32_TG4_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 32, 32, 8, 4>(); }

// hf8 -> fp16, M=32, ThrGroupSize=2
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M32_N32_TG2_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 32, 32, 16, 2>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M32_N32_TG4_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 32, 32, 8, 4>(); }

// hf8 -> bf16, M=32, ThrGroupSize=2
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M32_N32_TG2_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 32, 32, 16, 2>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M32_N32_TG4_Nt8)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 32, 32, 8, 4>(); }

// fp4 -> fp16, M=32, ThrGroupSize=2 (N=64, N_src=32, WarpTileSrc=16)
TEST(XE4_TCVUMX_MMA, fp4_to_fp16_M32_N64_TG2_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 32, 64, 16, 2>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_bf16_M32_N64_TG2_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 32, 64, 16, 2>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Large tile tests: 256×32 (bf8/hf8), 256×64 (fp4)
//
// ThrGroupSize defaults to M/32. Warp distribution:
//   256×32 (bf8/hf8): ThrGroupSize=8, NumWarpsAlongM=8, NumWarpsAlongN=1, WarpTileSrc=32
//   256×64 (fp4):     ThrGroupSize=8, NumWarpsAlongM=8, NumWarpsAlongN=1, WarpTileSrc=32
////////////////////////////////////////////////////////////////////////////////////////////////////

// --- 256 × 32 bf8/hf8 (ThrGroupSize=8, WarpTileSrc=32) ---

TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M256_N32_Nt16)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 256, 32, 16>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_fp16_M256_N32_Nt32)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, bf8, fp16, 256, 32, 32>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M256_N32_Nt16)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 256, 32, 16>(); }
TEST(XE4_TCVUMX_MMA, bf8_to_bf16_M256_N32_Nt32)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, bf8, bf16, 256, 32, 32>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M256_N32_Nt16)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 256, 32, 16>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_fp16_M256_N32_Nt32)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, hf8, fp16, 256, 32, 32>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M256_N32_Nt16)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 256, 32, 16>(); }
TEST(XE4_TCVUMX_MMA, hf8_to_bf16_M256_N32_Nt32)  { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, hf8, bf16, 256, 32, 32>(); }

// --- 256 × 64 fp4 (ThrGroupSize=8, WarpTileSrc=32, WarpTileN=64) ---

TEST(XE4_TCVUMX_MMA, fp4_to_fp16_M256_N64_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 256, 64, 16>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_fp16_M256_N64_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, fp16, fp4_e2m1, fp16, 256, 64, 32>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_bf16_M256_N64_Nt16) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 256, 64, 16>(); }
TEST(XE4_TCVUMX_MMA, fp4_to_bf16_M256_N64_Nt32) { run_ldsm_mma_tcvumx_test<uint8_t, uint16_t, bf16, fp4_e2m1, bf16, 256, 64, 32>(); }
