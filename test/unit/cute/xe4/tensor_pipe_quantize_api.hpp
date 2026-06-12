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

#pragma once

// API wrappers for ADMA, LDSM, and Tensor Pipe quantize (downconvert) operations.
//
// The LDSM Vlen padding is handled internally — the caller works with actual
// (M, N) data dimensions. Padding to satisfy the LDSM hardware constraint is
// computed automatically and applied to both the GMEM allocation (for ADMA
// compatibility) and the SLM layout. Only valid columns contain test data;
// padding columns are zero-filled and excluded from validation.
//
// API functions:
//   - adma_load:             GMEM → SLM via async DMA
//   - adma_store:            SLM → GMEM via async DMA
//   - load_matrix:           SLM → Registers via LDSM
//   - store_matrix:          Registers → SLM via LDSM
//   - tensor_pipe_quantize:  Register-to-register type downconversion via TCVD
//
// Padding rules (applied internally):
//   32-bit (float):    N_pad = max(N,  8)
//   16-bit (bf16/fp16): N_pad = max(N, 16)
//    8-bit (bf8/hf8):  N_pad = max(N, 32)
//
// Supported TCVD conversions (tcvd.toty.fromty.m32nN):
//   .toty   = { .e5m2, .e4m3, .e3m2, .e2m3, .e2m1, .s8, .s4 }
//   .fromty = { .f16, .bf16, .f32 }

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_xe4_ldsm.hpp>
#include <cute/atom/copy_traits_xe4_adma.hpp>
#include <cutlass/arch/barrier.h>
#include <cute/algorithm/tensor_processing.hpp>

namespace cutlass::xe4 {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// fp4_e2m1_tcvd: float → fp4_e2m1 conversion matching TCVD hardware rounding.
//
// This is a separate function rather than a fix to fp4_e2m1(float) because
// the existing constructor is used by other components and changing its
// rounding behavior could break them. This function fixes two issues in
// the original:
//   1. LUT is sorted so std::lower_bound (binary search) works correctly.
//      The original LUT has positive values followed by negative values,
//      which is not sorted and causes undefined behavior with lower_bound.
//   2. Tie-breaking uses round-to-nearest-even (picks the encoding with
//      even mantissa bit[0]=0) to match the gtp_tcvd pISA instruction.
//      The original picks the lower value on ties, which disagrees with
//      hardware on cases like 3.5 (equidistant from 3.0 and 4.0).
////////////////////////////////////////////////////////////////////////////////////////////////////

inline fp4_e2m1 fp4_e2m1_tcvd(float val) {
  if (std::isnan(val) || std::isinf(val)) { return fp4_e2m1(uint8_t(0x8)); }

  static std::vector<float> LUT = {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6};
  static std::vector<uint8_t> D_LUT = {0xf, 0xe, 0xd, 0xc, 0xb, 0xa, 0x9, 0x0,
                                       0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7};

  auto it = std::lower_bound(LUT.begin(), LUT.end(), val);

  uint8_t enc;
  if (it == LUT.begin()) {
    enc = D_LUT.front();
  } else if (it == LUT.end()) {
    enc = D_LUT.back();
  } else {
    int idx = it - LUT.begin();
    float dist_hi = std::abs(LUT[idx] - val);
    float dist_lo = std::abs(LUT[idx - 1] - val);
    if (dist_hi < dist_lo) {
      enc = D_LUT[idx];
    } else if (dist_lo < dist_hi) {
      enc = D_LUT[idx - 1];
    } else {
      uint8_t enc_hi = D_LUT[idx];
      uint8_t enc_lo = D_LUT[idx - 1];
      enc = (enc_hi & 1) == 0 ? enc_hi : enc_lo;
    }
  }
  return fp4_e2m1(enc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Compile-time padding helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BitWidth>
struct ldsm_min_n {
  static constexpr int value =
    (BitWidth >= 32) ?  8 :
    (BitWidth >= 16) ? 16 :
                       32;
};

template <int N, int BitWidth>
static constexpr int slm_pad_n = (N < ldsm_min_n<BitWidth>::value)
                                   ? ldsm_min_n<BitWidth>::value : N;

////////////////////////////////////////////////////////////////////////////////////////////////////
// adma_load: GMEM → SLM via async DMA
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class SmemTensor, class AdmaCopy, class SmemLayout>
CUTE_DEVICE void
adma_load(SmemTensor& t_smem,
          AdmaCopy& adma_copy,
          SmemLayout const& smem_layout,
          uint64_t* mbar,
          bool elected,
          sycl::nd_item<3>& item)
{
  Tensor mA = adma_copy.get_tma_tensor(shape(smem_layout));
  auto cta_slice = adma_copy.get_slice(Int<0>{});
  Tensor tAgA = cta_slice.partition_S(mA);
  Tensor tAsA = cta_slice.partition_D(t_smem);

  constexpr int kBytes = sizeof(make_tensor_like(tensor<0>(tAsA)));
  if (elected) {
    xe4_set_barrier_transaction_bytes(mbar[0], kBytes);
    copy(adma_copy.with(&mbar[0]), tAgA, tAsA);
    xe4_wait_barrier(mbar[0], 0);
  }
  item.barrier(sycl::access::fence_space::local_space);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// adma_store: SLM → GMEM via async DMA
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class SmemTensor, class AdmaCopy, class SmemLayout>
CUTE_DEVICE void
adma_store(SmemTensor& t_smem,
           AdmaCopy& adma_copy,
           SmemLayout const& smem_layout,
           uint64_t* mbar,
           bool elected)
{
  Tensor mB = adma_copy.get_tma_tensor(shape(smem_layout));
  auto cta_slice = adma_copy.get_slice(Int<0>{});
  Tensor tBsB = cta_slice.partition_S(t_smem);
  Tensor tBgB = cta_slice.partition_D(mB);

  constexpr int kBytes = sizeof(make_tensor_like(tensor<0>(tBsB)));
  if (elected) {
    xe4_set_barrier_transaction_bytes(mbar[0], kBytes);
    copy(adma_copy.with(&mbar[0]), tBsB, tBgB);
    xe4_wait_barrier(mbar[0], 0);
  }
  syncthreads();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// load_matrix: SLM → Registers via LDSM
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class SmemTensor, class SmemLayout>
CUTE_DEVICE auto
load_matrix(SmemTensor& t_smem,
            SmemLayout const& smem_layout,
            int tid,
            sycl::nd_item<3>& item)
{
  auto tiled_load = make_ldsm_copy_C<1, LDSMMode::Vector>(t_smem);

  auto Sshape = shape(SmemLayout{});
  auto coord_shape = make_shape(get<0>(Sshape), get<1>(Sshape));
  Tensor coord_tile = make_identity_tensor(coord_shape);

  auto thr_load = tiled_load.get_thread_slice(tid);
  auto tXsX = thr_load.partition_S(coord_tile);
  auto tXrX = thr_load.partition_fragment_D(coord_tile);
  clear(tXrX);

  copy(tiled_load, tXsX, tXrX);
  sycl::group_barrier(item.get_group());

  return tXrX;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// store_matrix: Registers → SLM via LDSM
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class DstType, int N_dst_bytes, class SmemTensor, class SmemLayout>
CUTE_DEVICE void
store_matrix(SmemTensor& t_smem,
             DstType const* dst_reg,
             SmemLayout const& smem_layout,
             int tid,
             sycl::nd_item<3>& item)
{
  auto tiled_store = make_ldsm_copy_D<1, LDSMMode::Vector>(t_smem);

  auto Dshape = shape(SmemLayout{});
  auto dst_coord_shape = make_shape(get<0>(Dshape), get<1>(Dshape));
  Tensor dst_coord_tile = make_identity_tensor(dst_coord_shape);

  auto thr_store = tiled_store.get_thread_slice(tid);
  auto thr_dst_coord = thr_store.partition_D(dst_coord_tile);
  auto thr_store_frag = thr_store.partition_fragment_S(dst_coord_tile);
  clear(thr_store_frag);

  for (int i = 0; i < N_dst_bytes; ++i) {
    thr_store_frag(i) = dst_reg[i];
  }

  copy(tiled_store, thr_store_frag, thr_dst_coord);
  sycl::group_barrier(item.get_group());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// tensor_pipe_quantize: Register-to-register type downconversion via TCVD
//
// Thin wrapper around cute::tensor_pipe_quantize() (defined in
// cute/algorithm/tensor_processing.hpp).
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class TcvdDstType, class TcvdSrcType, int N_tcvd,
          class SrcType, class DstType, class RegFragment>
CUTE_DEVICE void
tensor_pipe_quantize(DstType* dst_reg,
                     RegFragment& tXrX)
{
  constexpr int N_dst_bytes = N_tcvd * ::sizeof_bits<TcvdDstType>() / 8;

  auto src_tensor = make_tensor(make_rmem_ptr(&tXrX(0)), make_shape(Int<N_tcvd>{}));
  auto dst_tensor = make_tensor(make_rmem_ptr(dst_reg), make_shape(Int<N_dst_bytes>{}));
  cute::tensor_pipe_quantize<TcvdDstType, TcvdSrcType>(src_tensor, dst_tensor);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// validate_quantize_output: Host-side validation
//
// Compares device output against host-computed reference values.
// For byte-sized types (bf8, hf8): one element per output byte, N_dst_stride = N.
// For sub-byte types (fp4_e2m1): two elements packed per byte, N_dst_stride = N/2.
//
// Parameters:
//   h_out: device output copied back to host (stride = N_dst_stride)
//   h_in:  host input buffer (stride = N_src_stride)
//   N_src_stride: row stride of the source GMEM buffer
//   N_dst_stride: row stride of the destination GMEM buffer
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType, typename TcvdDstType, typename TcvdSrcType,
          typename HostRefType, int M, int N, int N_src_stride, int N_dst_stride>
void validate_quantize_output(host_vector<DstType> const& h_out,
                              host_vector<SrcType> const& h_in)
{
  auto get_src_float = [&](int row, int col) -> float {
    int src_idx = row * N_src_stride + col;
    if constexpr (std::is_same_v<SrcType, float>) {
      return h_in[src_idx];
    } else {
      TcvdSrcType tmp;
      memcpy(&tmp, &h_in[src_idx], sizeof(uint16_t));
      return static_cast<float>(tmp);
    }
  };

  if constexpr (std::is_same_v<TcvdDstType, fp4_e2m1>) {
    constexpr int N_dst_valid = N / 2;
    static_assert(N % 2 == 0, "fp4_e2m1 requires even N");
    for (int row = 0; row < M; ++row) {
      for (int byte_col = 0; byte_col < N_dst_valid; ++byte_col) {
        int dst_idx = row * N_dst_stride + byte_col;
        float val0 = get_src_float(row, byte_col * 2);
        float val1 = get_src_float(row, byte_col * 2 + 1);
        uint8_t enc0 = fp4_e2m1_tcvd(val0).data;
        uint8_t enc1 = fp4_e2m1_tcvd(val1).data;
        uint8_t expected = (enc0 & 0xF) | ((enc1 & 0xF) << 4);
        EXPECT_EQ(h_out[dst_idx], expected)
            << " at row=" << row << " byte_col=" << byte_col
            << " (val0=" << val0 << ", val1=" << val1 << ")";
      }
    }
  } else {
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < N; ++col) {
        int dst_idx = row * N_dst_stride + col;
        float src_val = get_src_float(row, col);
        HostRefType ref(src_val);
        DstType expected;
        memcpy(&expected, &ref, sizeof(DstType));
        EXPECT_EQ(h_out[dst_idx], expected)
            << " at row=" << row << " col=" << col << " (src_float=" << src_val << ")";
      }
    }
  }
}

} // namespace cutlass::xe4
