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

// Unit tests for TensorPipe Quantize (downconvert) using high-level API wrappers.
//
// Flow: GMEM --(adma_load)--> SLM(src) --(load_matrix)--> Registers
//             --(tensor_pipe_quantize)--> Registers
//             --(store_matrix)--> SLM(dst) --(adma_store)--> GMEM
//
// API wrappers (defined in tensor_pipe_quantize_api.hpp):
//   cutlass::xe4::adma_load()             — GMEM → SLM via async DMA
//   cutlass::xe4::load_matrix()           — SLM → Registers via LDSM
//   cutlass::xe4::tensor_pipe_quantize()  — Register-to-register type downconversion
//   cutlass::xe4::store_matrix()          — Registers → SLM via LDSM
//   cutlass::xe4::adma_store()            — SLM → GMEM via async DMA
//

#include "cutlass_unit_test.h"
#include "tensor_pipe_quantize_api.hpp"
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
// Device kernel using API wrappers
//
// SrcType: LDSM/GMEM source element type (float or uint16_t for bf16 raw bits)
// DstType: GMEM destination storage type (uint8_t for 8-bit fp8 / packed fp4)
// TcvdDstType: TCVD semantic dest type (bf8, hf8, fp4_e2m1) — selects pISA variant
// TcvdSrcType: TCVD semantic src type (float, bf16, fp16) — selects pISA variant
// N_tcvd: per-thread element count for TCVD (must match a supported tcvd.*.m32nN dim)
// N_src_pad: padded source SLM width (>= N_tcvd, satisfies LDSM Vlen constraint)
// N_dst_pad: padded destination SLM width (>= N_dst_valid, satisfies LDSM Vlen constraint)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType, typename TcvdDstType, typename TcvdSrcType,
          int N_tcvd, int N_src_pad, int N_dst_pad,
          class SrcSmemLayout, class DstSmemLayout,
          class AdmaLoad, class AdmaStore>
CUTLASS_GLOBAL void
tcvd_api_test_kernel(sycl::nd_item<3>& item,
                     SrcType* g_in, DstType* g_out,
                     SrcSmemLayout src_smem_layout, DstSmemLayout dst_smem_layout,
                     AdmaLoad adma_ld, AdmaStore adma_st)
{
  using namespace cute;
  namespace api = cutlass::xe4;

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

  // Step 1: GMEM -> SLM(src) via ADMA
  api::adma_load(t_smem_src, adma_ld, src_smem_layout, load_mbar, elected, item);

  // Step 2: SLM(src) -> Registers via LDSM
  auto tXrX = api::load_matrix(t_smem_src, src_smem_layout, tid, item);

  // Step 3: Quantize (downconvert) in registers via TCVD
  constexpr int N_dst_bytes = N_tcvd * ::sizeof_bits<TcvdDstType>() / 8;
  DstType dst_reg[N_dst_bytes];
  api::tensor_pipe_quantize<TcvdDstType, TcvdSrcType, N_tcvd, SrcType>(dst_reg, tXrX);
  sycl::group_barrier(item.get_group());

  // Step 4: Registers -> SLM(dst) via LDSM
  api::store_matrix<DstType, N_dst_bytes>(t_smem_dst, dst_reg, dst_smem_layout, tid, item);

  // Step 5: SLM(dst) -> GMEM via ADMA
  api::adma_store(t_smem_dst, adma_st, dst_smem_layout, store_mbar, elected);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Host-side test runner
//
// SrcType: LDSM element type (float or uint16_t for bf16 raw bits)
// DstType: output storage type (uint8_t for 8-bit fp8 / packed fp4)
// TcvdDstType: TCVD semantic dest type (bf8, hf8, fp4_e2m1) — selects pISA instruction variant
// TcvdSrcType: TCVD semantic src type (float, bf16, fp16) — selects pISA instruction variant
// HostRefType: host-side type for computing reference (must construct from float)
// M, N: tile dimensions (M must be 32 for m32nN TCVD constraint)
//
// Padding for LDSM Vlen constraint is computed and applied automatically:
//   - N_src_pad = max(N, ldsm_min_n for SrcType bit width)
//   - N_dst_pad = max(N_dst_valid, ldsm_min_n for DstType bit width)
//   - GMEM is allocated with padded pitch for ADMA compatibility
//   - Only the first N valid columns contain test data; padding columns are zero
//   - Validation checks only valid columns, padding is excluded
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType, typename TcvdDstType, typename TcvdSrcType,
          typename HostRefType, int M, int N>
void run_api_test()
{
  static_assert(M == 32, "TCVD requires M=32 (subgroup size)");

  // Compute padded widths for SLM (LDSM Vlen constraint)
  constexpr int SrcBitWidth = sizeof(SrcType) * 8;
  constexpr int DstBitWidth = sizeof(DstType) * 8;
  constexpr int N_src_pad = cutlass::xe4::slm_pad_n<N, SrcBitWidth>;

  // Valid destination bytes per row: N for byte types (bf8/hf8), N/2 for sub-byte (fp4)
  constexpr int N_dst_valid = N * sizeof_bits_v<TcvdDstType> / 8;
  constexpr int N_dst_pad = cutlass::xe4::slm_pad_n<N_dst_valid, DstBitWidth>;

  using SrcSLayout = decltype(make_layout(make_shape(Int<M>{}, Int<N_src_pad>{}), LayoutRight{}));
  using DstSLayout = decltype(make_layout(make_shape(Int<M>{}, Int<N_dst_pad>{}), LayoutRight{}));

  constexpr int src_count = M * N_src_pad;
  constexpr int dst_count = M * N_dst_pad;

  // Prepare host input: valid data in columns 0..N-1, zero padding in N..N_src_pad-1
  host_vector<SrcType> h_in(src_count, SrcType(0));
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      int src_idx = row * N_src_pad + col;
      float val = static_cast<float>((row * N + col) % 16) * 0.5f;
      if constexpr (std::is_same_v<SrcType, float>) {
        h_in[src_idx] = val;
      } else if constexpr (std::is_same_v<SrcType, uint16_t>) {
        TcvdSrcType tmp(val);
        memcpy(&h_in[src_idx], &tmp, sizeof(uint16_t));
      }
    }
  }

  host_vector<DstType> h_zero(dst_count, DstType(0));

  // Allocate device memory with padded pitch
  sycl::queue queue;
  SrcType* d_in = static_cast<SrcType*>(malloc_device(src_count * sizeof(SrcType), queue));
  DstType* d_out = static_cast<DstType*>(malloc_device(dst_count * sizeof(DstType), queue));
  queue.memcpy(d_in, h_in.data(), src_count * sizeof(SrcType)).wait();
  queue.memcpy(d_out, h_zero.data(), dst_count * sizeof(DstType)).wait();

  // GMEM tensors with padded pitch (matching SLM layout for ADMA compatibility)
  auto gmem_layout_src = make_layout(make_shape(M, N_src_pad), LayoutRight{});
  auto gmem_layout_dst = make_layout(make_shape(M, N_dst_pad), LayoutRight{});

  Tensor gA = make_tensor(make_gmem_ptr<SrcType>(d_in), gmem_layout_src);
  Tensor gB = make_tensor(make_gmem_ptr<DstType>(d_out), gmem_layout_dst);

  auto adma_load = cute::make_adma_copy<SrcType>(
      XE4_ADMA_LOAD{}, gA, SrcSLayout{}, product_each(shape(SrcSLayout{})), Int<1>{});
  auto adma_store = cute::make_adma_copy<DstType>(
      XE4_ADMA_STORE{}, gB, DstSLayout{}, product_each(shape(DstSLayout{})), Int<1>{});

  sycl::nd_range<3> range(sycl::range<3>(1,1,32), sycl::range<3>(1,1,32));

  queue.parallel_for(range, [=](sycl::nd_item<3> item) {
    tcvd_api_test_kernel<SrcType, DstType, TcvdDstType, TcvdSrcType, N, N_src_pad, N_dst_pad,
                          SrcSLayout, DstSLayout>(
        item, d_in, d_out, SrcSLayout{}, DstSLayout{}, adma_load, adma_store);
  }).wait();

  host_vector<DstType> h_out(dst_count, DstType(0));
  queue.memcpy(h_out.data(), d_out, dst_count * sizeof(DstType)).wait();

  // Validate only the N valid columns (stride = N_src_pad / N_dst_pad for padded GMEM)
  cutlass::xe4::validate_quantize_output<SrcType, DstType, TcvdDstType, TcvdSrcType,
                                          HostRefType, M, N, N_src_pad, N_dst_pad>(h_out, h_in);

  sycl::free(d_in, queue);
  sycl::free(d_out, queue);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test cases: float -> bf8 (E5M2) via tcvd.e5m2.f32.m32nN
// Supported N for float->bf8: 8, 16, 32
// No source padding needed (float max Vlen=8, all N >= 8)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_API, f32_to_bf8_N8)   { run_api_test<float, uint8_t, bf8, float, bf8, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_API, f32_to_bf8_N16)  { run_api_test<float, uint8_t, bf8, float, bf8, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_API, f32_to_bf8_N32)  { run_api_test<float, uint8_t, bf8, float, bf8, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test cases: float -> hf8 (E4M3) via tcvd.e4m3.f32.m32nN
// Supported N for float->hf8: 16, 32
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_API, f32_to_hf8_N16)  { run_api_test<float, uint8_t, hf8, float, hf8, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_API, f32_to_hf8_N32)  { run_api_test<float, uint8_t, hf8, float, hf8, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test cases: float -> fp4_e2m1 (E2M1) via tcvd.e2m1.f32.m32nN
// Supported N for float->fp4_e2m1: 16, 32
// DstType=uint8_t (packed output: 2 fp4 values per byte)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_API, f32_to_fp4_N16)  { run_api_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_API, f32_to_fp4_N32)  { run_api_test<float, uint8_t, fp4_e2m1, float, fp4_e2m1, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test cases: bf16 -> bf8 (E5M2) via tcvd.e5m2.bf16.m32nN
// SrcType=uint16_t (raw bf16 bits), TcvdSrcType=bf16
// Supported N for bf16->bf8: 8, 16, 32
// N=8: source SLM auto-padded to 16 (bf16 max Vlen=16)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_API, bf16_to_bf8_N8)  { run_api_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_API, bf16_to_bf8_N16) { run_api_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_API, bf16_to_bf8_N32) { run_api_test<uint16_t, uint8_t, bf8, bf16, bf8, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test cases: bf16 -> hf8 (E4M3) via tcvd.e4m3.bf16.m32nN
// SrcType=uint16_t (raw bf16 bits), TcvdSrcType=bf16
// Supported N for bf16->hf8: 8, 16, 32
// N=8: source SLM auto-padded to 16 (bf16 max Vlen=16)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_API, bf16_to_hf8_N8)  { run_api_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_API, bf16_to_hf8_N16) { run_api_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_API, bf16_to_hf8_N32) { run_api_test<uint16_t, uint8_t, hf8, bf16, hf8, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test cases: fp16 -> bf8 (E5M2) via tcvd.e5m2.f16.m32nN
// SrcType=uint16_t (raw fp16 bits), TcvdSrcType=fp16
// Supported N for fp16->bf8: 8, 16, 32
// N=8: source SLM auto-padded to 16 (fp16 max Vlen=16)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_API, fp16_to_bf8_N8)  { run_api_test<uint16_t, uint8_t, bf8, fp16, bf8, 32, 8>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_API, fp16_to_bf8_N16) { run_api_test<uint16_t, uint8_t, bf8, fp16, bf8, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_API, fp16_to_bf8_N32) { run_api_test<uint16_t, uint8_t, bf8, fp16, bf8, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test cases: fp16 -> hf8 (E4M3) via tcvd.e4m3.f16.m32nN
// SrcType=uint16_t (raw fp16 bits), TcvdSrcType=fp16
// Supported N for fp16->hf8: 16, 32
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_API, fp16_to_hf8_N16) { run_api_test<uint16_t, uint8_t, hf8, fp16, hf8, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_API, fp16_to_hf8_N32) { run_api_test<uint16_t, uint8_t, hf8, fp16, hf8, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test cases: bf16 -> fp4_e2m1 (E2M1) via tcvd.e2m1.bf16.m32nN
// SrcType=uint16_t (raw bf16 bits), TcvdSrcType=bf16
// DstType=uint8_t (packed output: 2 fp4 values per byte)
// Supported N for bf16->fp4_e2m1: 16, 32
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_API, bf16_to_fp4_N16) { run_api_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_API, bf16_to_fp4_N32) { run_api_test<uint16_t, uint8_t, fp4_e2m1, bf16, fp4_e2m1, 32, 32>(); }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Test cases: fp16 -> fp4_e2m1 (E2M1) via tcvd.e2m1.f16.m32nN
// SrcType=uint16_t (raw fp16 bits), TcvdSrcType=fp16
// DstType=uint8_t (packed output: 2 fp4 values per byte)
// Supported N for fp16->fp4_e2m1: 16, 32
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_TENSORPIPE_QUANTIZE_API, fp16_to_fp4_N16) { run_api_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 32, 16>(); }
TEST(XE4_TENSORPIPE_QUANTIZE_API, fp16_to_fp4_N32) { run_api_test<uint16_t, uint8_t, fp4_e2m1, fp16, fp4_e2m1, 32, 32>(); }
