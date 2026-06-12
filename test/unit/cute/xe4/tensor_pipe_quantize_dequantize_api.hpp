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
#include <cute/atom/copy_traits_xe4_eu_copy.hpp>
#include <cute/arch/xe4_async_gmma_slm_layout.hpp>
#include <cute/arch/xe4_inline_pisa.hpp>
#include <cute/algorithm/tensor_processing.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

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
//
// Overload 1 (no MMA): uses geometry-only make_ldsm_copy_C with ThrGroupSize (default 1).
// Overload 2 (with MMA): uses MMA-aware make_ldsm_copy_C for multi-warp partitioning
//   driven by the TiledMMA atom shape.
////////////////////////////////////////////////////////////////////////////////////////////////////

// Overload without MMA
template <int ThrGroupSize = 1, class SmemTensor, class SmemLayout>
CUTE_DEVICE auto
load_matrix(SmemTensor& t_smem,
            SmemLayout const&,
            int tid,
            sycl::nd_item<3>& item)
{
  auto tiled_load = make_ldsm_copy_C<ThrGroupSize, LDSMMode::Vector>(t_smem);

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

// Overload with MMA (multi-warp, MMA-aware partitioning)
template <int ThrGroupSize = 1, class TiledMMA, class SmemTensor, class SmemLayout>
CUTE_DEVICE auto
load_matrix(TiledMMA const& mma,
            SmemTensor& t_smem,
            SmemLayout const&,
            int tid,
            sycl::nd_item<3>& item)
{
  auto tiled_load = make_ldsm_copy_C<ThrGroupSize, LDSMMode::Vector>(mma, t_smem);

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
//
// ThrGroupSize (default 1) controls multi-warp partitioning of the destination
// SLM tile. For single-warp use, omit ThrGroupSize or pass 1.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class DstType, int N_dst_bytes, int ThrGroupSize = 1,
          class SmemTensor, class SmemLayout>
CUTE_DEVICE void
store_matrix(SmemTensor& t_smem,
             DstType const* dst_reg,
             SmemLayout const&,
             int tid,
             sycl::nd_item<3>& item)
{
  auto tiled_store = make_ldsm_copy_D<ThrGroupSize, LDSMMode::Vector>(t_smem);

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

////////////////////////////////////////////////////////////////////////////////////////////////////
// validate_dequantize_output: Host-side validation for upconvert (TCVU)
//
// Compares device output (wide type) against host-computed reference values.
// Source is narrow type packed as uint8_t bytes; destination is wide type (fp16/bf16).
//
// For byte-sized sources (bf8, hf8): one element per source byte.
// For sub-byte sources (fp4_e2m1): two elements packed per source byte
//   (low nibble = even element, high nibble = odd element).
//
// Parameters:
//   h_out: device output copied back to host (DstType elements, stride = N_dst_stride)
//   h_in:  host input buffer (SrcType=uint8_t bytes, stride = N_src_stride)
//   M, N:  tile dimensions (N = number of narrow elements per row)
//   N_src_stride: row stride of the source buffer (in SrcType elements)
//   N_dst_stride: row stride of the destination buffer (in DstType elements)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType, typename TcvuDstType, typename TcvuSrcType,
          int M, int N, int N_src_stride, int N_dst_stride>
void validate_dequantize_output(host_vector<DstType> const& h_out,
                                host_vector<SrcType> const& h_in)
{
  // Convert a source narrow element (identified by row, col) to float
  auto get_src_float = [&](int row, int col) -> float {
    if constexpr (std::is_same_v<TcvuSrcType, fp4_e2m1>) {
      int byte_col = col / 2;
      int src_idx = row * N_src_stride + byte_col;
      uint8_t packed = h_in[src_idx];
      uint8_t nibble = (col % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
      fp4_e2m1 narrow(nibble);
      return static_cast<float>(narrow);
    } else {
      int src_idx = row * N_src_stride + col;
      TcvuSrcType narrow;
      memcpy(&narrow, &h_in[src_idx], sizeof(uint8_t));
      return static_cast<float>(narrow);
    }
  };

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      int dst_idx = row * N_dst_stride + col;
      float src_val = get_src_float(row, col);
      TcvuDstType ref(src_val);
      DstType expected;
      memcpy(&expected, &ref, sizeof(DstType));
      EXPECT_EQ(h_out[dst_idx], expected)
          << " at row=" << row << " col=" << col << " (src_float=" << src_val << ")";
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// fill_source_data: Generate test source data for quantize tests.
//
// Fills h_in with a repeating pattern: val = ((row * N + col) % 8) * 0.5f
// Values range 0.0–3.5, safely representable by all narrow target types.
//
// For float SrcType: stores val directly.
// For uint16_t SrcType: converts through TcvdSrcType (bf16/fp16) and stores raw bits.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename TcvdSrcType, int M, int N>
void fill_source_data(host_vector<SrcType>& h_in)
{
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      int src_idx = row * N + col;
      float val = static_cast<float>((row * N + col) % 8) * 0.5f;
      if constexpr (std::is_same_v<SrcType, float>) {
        h_in[src_idx] = val;
      } else if constexpr (std::is_same_v<SrcType, uint16_t>) {
        TcvdSrcType tmp(val);
        memcpy(&h_in[src_idx], &tmp, sizeof(uint16_t));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// fill_meta_per_row: Generate per-row MX scale metadata.
//
// meta[row] = 127 + (row % 3) → exponents 0, 1, 2 → scales 1, 2, 4
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int M>
void fill_meta_per_row(host_vector<uint8_t>& h_meta)
{
  for (int row = 0; row < M; ++row) {
    h_meta[row] = static_cast<uint8_t>(127 + (row % 3));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// fill_meta_per_col: Generate per-column MX scale metadata.
//
// meta[col] = 127 + (col % 3) → exponents 0, 1, 2 → scales 1, 2, 4
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
void fill_meta_per_col(host_vector<uint8_t>& h_meta)
{
  for (int col = 0; col < N; ++col) {
    h_meta[col] = static_cast<uint8_t>(127 + (col % 3));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// fill_lower_precision_source_data: Generate packed narrow source data for dequantize tests.
//
// Fills h_in (uint8_t buffer) with narrow-type encoded values using the pattern:
//   val = ((row * N + col) % 8) * 0.5f
//
// For byte-sized narrow types (bf8, hf8): one element per byte.
// For sub-byte types (fp4_e2m1): two elements packed per byte (low nibble = even col).
//
// Parameters:
//   M: number of rows
//   N: number of logical elements per row
//   N_src: source bytes per row = N * sizeof_bits<TcvuSrcType> / 8
//   TcvuSrcType: narrow semantic type (bf8, hf8, fp4_e2m1)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename TcvuSrcType, int M, int N, int N_src>
void fill_lower_precision_source_data(host_vector<SrcType>& h_in)
{
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float val = static_cast<float>((row * N + col) % 8) * 0.5f;
      if constexpr (std::is_same_v<TcvuSrcType, fp4_e2m1>) {
        int byte_col = col / 2;
        int src_idx = row * N_src + byte_col;
        fp4_e2m1 narrow(val);
        if (col % 2 == 0) {
          h_in[src_idx] = narrow.data & 0xF;
        } else {
          h_in[src_idx] |= (narrow.data & 0xF) << 4;
        }
      } else {
        int src_idx = row * N_src + col;
        TcvuSrcType narrow(val);
        memcpy(&h_in[src_idx], &narrow, sizeof(uint8_t));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// validate_tcvdmx_mx_output: Unified host-side validation for MX quantize
//
// MxScaleDir::PerRow: meta indexed by row — dst = quantize(src / 2^(meta[row] - 127))
// MxScaleDir::PerCol: meta indexed by col — dst = quantize(src / 2^(meta[col] - 127))
//
// When AllowSrndTolerance=true, comparison uses ±1 ULP tolerance (for stochastic rounding).
// When AllowSrndTolerance=false (default), comparison is exact (EXPECT_EQ).
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class MxScaleDir { PerRow, PerCol };

template <MxScaleDir ScaleDir = MxScaleDir::PerRow, bool AllowSrndTolerance = false,
          typename SrcType, typename DstType, typename TcvdDstType, typename TcvdSrcType,
          typename HostRefType, int M, int N, int N_src_stride, int N_dst_stride>
void validate_tcvdmx_mx_output(host_vector<DstType> const& h_out,
                                host_vector<SrcType> const& h_in,
                                host_vector<uint8_t> const& h_meta)
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

  auto apply_mx_scale = [&](float val, int idx) -> float {
    int exponent = static_cast<int>(h_meta[idx]) - 127;
    return val / std::ldexp(1.0f, exponent);
  };

  // Helper: get the scale index based on direction
  auto scale_idx = [](int row, int col) -> int {
    if constexpr (ScaleDir == MxScaleDir::PerRow) return row;
    else return col;
  };

  if constexpr (std::is_same_v<TcvdDstType, fp4_e2m1> && !AllowSrndTolerance) {
    constexpr int N_dst_valid = N / 2;
    static_assert(N % 2 == 0, "fp4_e2m1 requires even N");
    for (int row = 0; row < M; ++row) {
      for (int byte_col = 0; byte_col < N_dst_valid; ++byte_col) {
        int dst_idx = row * N_dst_stride + byte_col;
        int col0 = byte_col * 2;
        int col1 = byte_col * 2 + 1;
        float val0 = apply_mx_scale(get_src_float(row, col0), scale_idx(row, col0));
        float val1 = apply_mx_scale(get_src_float(row, col1), scale_idx(row, col1));
        uint8_t enc0 = fp4_e2m1_tcvd(val0).data;
        uint8_t enc1 = fp4_e2m1_tcvd(val1).data;
        uint8_t expected = (enc0 & 0xF) | ((enc1 & 0xF) << 4);
        EXPECT_EQ(h_out[dst_idx], expected)
            << " at row=" << row << " byte_col=" << byte_col
            << " (val0=" << val0 << ", val1=" << val1
            << ", meta0=" << static_cast<int>(h_meta[scale_idx(row, col0)])
            << ", meta1=" << static_cast<int>(h_meta[scale_idx(row, col1)]) << ")";
      }
    }
  } else {
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < N; ++col) {
        int dst_idx = row * N_dst_stride + col;
        float src_val = apply_mx_scale(get_src_float(row, col), scale_idx(row, col));
        HostRefType ref(src_val);
        DstType expected;
        memcpy(&expected, &ref, sizeof(DstType));
        if constexpr (AllowSrndTolerance) {
          int diff = static_cast<int>(h_out[dst_idx]) - static_cast<int>(expected);
          EXPECT_LE(std::abs(diff), 1)
              << " srnd output out of range at row=" << row << " col=" << col
              << " actual=" << static_cast<int>(h_out[dst_idx])
              << " rne=" << static_cast<int>(expected)
              << " src=" << get_src_float(row, col)
              << " scaled=" << src_val
              << " meta=" << static_cast<int>(h_meta[scale_idx(row, col)]);
        } else {
          EXPECT_EQ(h_out[dst_idx], expected)
              << " at row=" << row << " col=" << col
              << " (src_float=" << get_src_float(row, col)
              << ", scaled=" << src_val
              << ", meta=" << static_cast<int>(h_meta[scale_idx(row, col)]) << ")";
        }
      }
    }
  }
}

// Convenience aliases matching the old function names
template <typename SrcType, typename DstType, typename TcvdDstType, typename TcvdSrcType,
          typename HostRefType, int M, int N, int N_src_stride, int N_dst_stride>
void validate_tcvdmx_output(host_vector<DstType> const& h_out,
                            host_vector<SrcType> const& h_in,
                            host_vector<uint8_t> const& h_meta)
{
  validate_tcvdmx_mx_output<MxScaleDir::PerRow, false,
      SrcType, DstType, TcvdDstType, TcvdSrcType, HostRefType, M, N, N_src_stride, N_dst_stride>(
      h_out, h_in, h_meta);
}

template <typename SrcType, typename DstType, typename TcvdDstType, typename TcvdSrcType,
          typename HostRefType, int M, int N, int N_src_stride, int N_dst_stride>
void validate_tcvdmx_mxmd_output(host_vector<DstType> const& h_out,
                                  host_vector<SrcType> const& h_in,
                                  host_vector<uint8_t> const& h_meta)
{
  validate_tcvdmx_mx_output<MxScaleDir::PerCol, false,
      SrcType, DstType, TcvdDstType, TcvdSrcType, HostRefType, M, N, N_src_stride, N_dst_stride>(
      h_out, h_in, h_meta);
}

template <typename SrcType, typename DstType, typename TcvdDstType, typename TcvdSrcType,
          typename HostRefType, int M, int N, int N_src_stride, int N_dst_stride>
void validate_tcvdmx_srnd_output(host_vector<DstType> const& h_out,
                                  host_vector<SrcType> const& h_in,
                                  host_vector<uint8_t> const& h_meta)
{
  validate_tcvdmx_mx_output<MxScaleDir::PerRow, true,
      SrcType, DstType, TcvdDstType, TcvdSrcType, HostRefType, M, N, N_src_stride, N_dst_stride>(
      h_out, h_in, h_meta);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// fill_lower_precision_source_data_mx: Generate packed narrow source data for MX dequantize tests.
//
// Similar to fill_lower_precision_source_data but with smaller value ranges to avoid
// overflow when scaled by MX meta exponent (2^(meta-127) with meta up to 129 → scale of 4).
//
// For byte-sized narrow types (bf8, hf8): val = ((row*N + col) % 8) * 0.25f
// For sub-byte types (fp4_e2m1): val = ((row*N + col) % 4) * 0.5f, packed 2 per byte.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename TcvumxSrcType, int M, int N, int N_src>
void fill_lower_precision_source_data_mx(host_vector<SrcType>& h_in)
{
  for (int row = 0; row < M; ++row) {
    if constexpr (std::is_same_v<TcvumxSrcType, fp4_e2m1>) {
      // fp4: pack two 4-bit elements per byte
      for (int byte_col = 0; byte_col < N_src; ++byte_col) {
        int col0 = byte_col * 2;
        int col1 = byte_col * 2 + 1;
        float val0 = static_cast<float>((row * N + col0) % 4) * 0.5f;
        float val1 = static_cast<float>((row * N + col1) % 4) * 0.5f;
        fp4_e2m1 enc0(val0);
        fp4_e2m1 enc1(val1);
        uint8_t packed = (enc0.data & 0xF) | ((enc1.data & 0xF) << 4);
        h_in[row * N_src + byte_col] = packed;
      }
    } else {
      // 8-bit types: encode float → narrow type directly
      for (int col = 0; col < N; ++col) {
        float val = static_cast<float>((row * N + col) % 8) * 0.25f;
        TcvumxSrcType encoded(val);
        uint8_t raw;
        memcpy(&raw, &encoded, sizeof(uint8_t));
        h_in[row * N_src + col] = raw;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// validate_tcvumx_output: Host-side validation for TCVUMX (.mxnd per-row MX dequantization)
//
// Expected: dst[row][col] = src_value[row][col] * 2^(meta[row] - 127)
//   where src_value is the float representation of the narrow-type element.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename SrcType, typename DstType, typename TcvumxDstType, typename TcvumxSrcType,
          typename HostRefType, int M, int N, int N_src_stride, int N_dst_stride>
void validate_tcvumx_output(host_vector<DstType> const& h_out,
                            host_vector<SrcType> const& h_in,
                            host_vector<uint8_t> const& h_meta)
{
  // Decode narrow source element to float
  auto get_src_float = [&](int row, int byte_offset, int elem_in_byte) -> float {
    if constexpr (std::is_same_v<TcvumxSrcType, fp4_e2m1>) {
      int src_idx = row * N_src_stride + byte_offset;
      uint8_t packed = h_in[src_idx];
      uint8_t nibble = (elem_in_byte == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
      fp4_e2m1 val;
      val.data = nibble;
      return static_cast<float>(val);
    } else {
      int src_idx = row * N_src_stride + byte_offset;
      TcvumxSrcType val;
      memcpy(&val, &h_in[src_idx], sizeof(uint8_t));
      return static_cast<float>(val);
    }
  };

  // Apply per-ROW MX scaling: multiply by 2^(meta[row] - 127)
  auto apply_mx_scale = [&](float val, int row) -> float {
    int exponent = static_cast<int>(h_meta[row]) - 127;
    return val * std::ldexp(1.0f, exponent);
  };

  if constexpr (std::is_same_v<TcvumxSrcType, fp4_e2m1>) {
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < N; ++col) {
        int byte_offset = col / 2;
        int elem_in_byte = col % 2;
        float src_val = get_src_float(row, byte_offset, elem_in_byte);
        float scaled = apply_mx_scale(src_val, row);

        int dst_idx = row * N_dst_stride + col;
        HostRefType expected_ref(scaled);
        DstType expected;
        memcpy(&expected, &expected_ref, sizeof(DstType));
        EXPECT_EQ(h_out[dst_idx], expected)
            << " at row=" << row << " col=" << col
            << " (src_float=" << src_val
            << ", scaled=" << scaled
            << ", meta=" << static_cast<int>(h_meta[row]) << ")";
      }
    }
  } else {
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < N; ++col) {
        float src_val = get_src_float(row, col, 0);
        float scaled = apply_mx_scale(src_val, row);

        int dst_idx = row * N_dst_stride + col;
        HostRefType expected_ref(scaled);
        DstType expected;
        memcpy(&expected, &expected_ref, sizeof(DstType));
        EXPECT_EQ(h_out[dst_idx], expected)
            << " at row=" << row << " col=" << col
            << " (src_float=" << src_val
            << ", scaled=" << scaled
            << ", meta=" << static_cast<int>(h_meta[row]) << ")";
      }
    }
  }
}

} // namespace cutlass::xe4
