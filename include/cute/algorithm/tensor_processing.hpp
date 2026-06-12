/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Intel CORPORATION. All rights reserved.
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

#include <iostream>

#include <cute/config.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/fill.hpp>
#include <cute/arch/xe4_inline_pisa.hpp>
#include <cute/numeric/integral_constant.hpp>

namespace cute
{

template <uint32_t M, uint32_t N>
struct round_up_ {
  constexpr static uint32_t value = (M + N - 1) / N;
};


// Helper function to convert tensor data to marray for inline PISA instructions
// Expects src to be a single row tensor: shape (N,) or (1, N)
template <typename src_type, typename reg_type, uint32_t N_reg, uint32_t N, typename SrcEngine, typename SrcLayout>
CUTE_HOST_DEVICE
auto tensor_to_marray(Tensor<SrcEngine, SrcLayout> const& src)
{
  sycl::marray<reg_type, N_reg> src_marray;

  memcpy(&src_marray, reinterpret_cast<src_type*>(raw_pointer_cast(src.data())), N * sizeof(src_type));
  return src_marray;
}

// Tensor processing reduction wrapper for tensor_pipe_reduce
// Expects src to be a single row tensor per work-item: shape (N,) or (1, N)
// User should slice the tensor before calling: tensor_pipe_tred(full_tensor(work_item_id, _))
template <tred_red_dim red_dim,
          tred_algo algo,
          tred_round_mode tr_mode = tred_round_mode::none,
          bool dsat = false,
          typename SrcEngine, 
          typename SrcLayout>
CUTE_HOST_DEVICE
auto tensor_pipe_reduce(Tensor<SrcEngine, SrcLayout> const& src)
{
  using src_type = typename SrcEngine::value_type;

  // Extract tensor shape - should be (N,) for a single row
  auto tensor_shape = shape(src);
  constexpr uint32_t N = size(tensor_shape);

  // Validate N is a supported size
  static_assert((N >= 1 && N <= 4) || N == 8 || N == 16 || N == 32,
                "N must be 1, 2, 3, 4, 8, 16, or 32");

  // Determine dst_type based on algo
  using dst_type = cute::conditional_t<algo == tred_algo::f32add, float, src_type>;

  static_assert(cute::is_same_v<src_type, fp16> || cute::is_same_v<src_type, bf16>, 
                "tensor_pipe_reduce only supports fp16 or bf16 source types");

  // For f32add, red_dim must be rednd
  static_assert(algo != tred_algo::f32add || red_dim == tred_red_dim::rednd,
                "For f32add algorithm, red_dim must be rednd");

  constexpr uint32_t dtype_reg_size = sizeof(uint32_t);
  constexpr uint32_t N_reg = round_up_<N * sizeof(src_type), dtype_reg_size>::value;

  // Convert tensor data to marray
  auto src_marray = tensor_to_marray<src_type, uint32_t, N_reg, N>(src);

  // Call the inline PISA API without accumulation
  return ::tensor_pipe_tred_pisa<src_type, dst_type, N, uint32_t, red_dim, algo, tr_mode, dsat, false>(src_marray);
}

// Overload with accumulation support
// Expects src to be a single row tensor per work-item: shape (N,) or (1, N)
// User should slice the tensor before calling: tensor_pipe_tred(full_tensor(work_item_id, _), acc)
template <tred_red_dim red_dim,
          tred_algo algo,
          tred_round_mode tr_mode = tred_round_mode::none,
          bool dsat = false,
          typename SrcEngine,
          typename SrcLayout,
          typename AccType>
CUTE_HOST_DEVICE
auto tensor_pipe_reduce(Tensor<SrcEngine, SrcLayout> const& src, AccType const& acc)
{
  using src_type = typename SrcEngine::value_type;
  
  // Extract tensor shape - should be (N,) for a single row
  auto tensor_shape = shape(src);
  constexpr uint32_t N = size(tensor_shape);

  // Validate N is a supported size
  static_assert((N >= 1 && N <= 4) || N == 8 || N == 16 || N == 32,
                "N must be 1, 2, 3, 4, 8, 16, or 32");

  // Determine dst_type based on algo
  using dst_type = cute::conditional_t<algo == tred_algo::f32add, float, src_type>;

  static_assert(cute::is_same_v<src_type, fp16> || cute::is_same_v<src_type, bf16>, 
                "tensor_pipe_reduce only supports fp16 or bf16 source types");

  static_assert(cute::is_same_v<AccType, dst_type>,
                "Accumulator type must match dst_type");

  // For f32add, red_dim must be rednd
  static_assert(algo != tred_algo::f32add || red_dim == tred_red_dim::rednd,
                "For f32add algorithm, red_dim must be rednd");

  constexpr uint32_t dtype_reg_size = sizeof(uint32_t);
  constexpr uint32_t N_reg = round_up_<N * sizeof(src_type), dtype_reg_size>::value;

  // Convert tensor data to marray
  auto src_marray = tensor_to_marray<src_type, uint32_t, N_reg, N>(src);

  // Call the inline PISA API with accumulation
  return ::tensor_pipe_tred_pisa<src_type, dst_type, N, uint32_t, red_dim, algo, tr_mode, dsat, true>(src_marray, acc);
}

// Tensor processing exponent reduction wrapper for tensor_pipe_exp_reduce
// Expects src and dst to be single row tensors per work-item: shape (N,) or (1, N)
// User should slice before calling: tensor_pipe_exp_red(src_tensor(work_item_id, _), dst_tensor(work_item_id, _))
template <tred_red_dim red_dim = tred_red_dim::none,
          bool mxnd = false,
          tred_round_mode tr_mode = tred_round_mode::none, 
          bool dsat = false,
          bool xch = false,
          typename SrcEngine, 
          typename SrcLayout,
          typename DstEngine,
          typename DstLayout>
CUTE_HOST_DEVICE
auto tensor_pipe_exp_reduce(Tensor<SrcEngine, SrcLayout> const &src,
                            Tensor<DstEngine, DstLayout> const &dst, 
                            float m=0.0f, float dsrc1=0.0f)
{
  using src_type = typename SrcEngine::value_type;
  using dst_type = typename DstEngine::value_type;

  // Extract tensor shape - should be (N,) for a single row
  auto src_shape = shape(src);
  auto dst_shape = shape(dst);
  constexpr uint32_t N = size(src_shape);

  // Validate N is a supported size
  static_assert((N >= 1 && N <= 4) || N == 8 || N == 16 || N == 32,
                "N must be 1, 2, 3, 4, 8, 16, or 32");

  static_assert(cute::is_same_v<src_type, fp16> || cute::is_same_v<src_type, bf16> ||
                cute::is_same_v<src_type, float>,
                "tensor_pipe_exp_reduce only supports fp16/bf16/float source types");

  static_assert((red_dim == tred_red_dim::none) || (red_dim == tred_red_dim::rednd),
                "Only not reduce or rednd are supported");

  constexpr uint32_t dtype_reg_size = sizeof(uint32_t);
  constexpr uint32_t N_reg = round_up_<N * sizeof(src_type), dtype_reg_size>::value;
  constexpr uint32_t N_reg_dst = round_up_<N * sizeof(dst_type), dtype_reg_size>::value;

  // Convert tensor data to marray
  auto src_marray = tensor_to_marray<src_type, uint32_t, N_reg, N>(src);
  sycl::marray<uint32_t, N_reg_dst> dst_marray;

  // Call the inline PISA API
  float ret_dsrc1 = ::tensor_pipe_exp_red_pisa<src_type, dst_type, N, uint32_t, red_dim, mxnd, tr_mode, dsat, xch>(src_marray, dst_marray, m, dsrc1);
  
  // Copy result back to destination tensor
  memcpy(raw_pointer_cast(dst.data()), &dst_marray, N * sizeof(dst_type));

  return ret_dsrc1;
}
  
// Tensor processing exponent base-2 reduction wrapper for tensor_pipe_exp2_reduce
// Expects src and dst to be single row tensors per work-item: shape (N,) or (1, N)
// User should slice before calling: tensor_pipe_exp2_red(src_tensor(work_item_id, _), dst_tensor(work_item_id, _))
template <tred_red_dim red_dim = tred_red_dim::none,
          bool mxnd = false,
          tred_round_mode tr_mode = tred_round_mode::none, 
          bool dsat = false,
          bool xch = false,
          typename SrcEngine, 
          typename SrcLayout,
          typename DstEngine,
          typename DstLayout>
CUTE_HOST_DEVICE
auto tensor_pipe_exp2_reduce(Tensor<SrcEngine, SrcLayout> const &src,
                             Tensor<DstEngine, DstLayout> const &dst, 
                             float m=0.0f, float dsrc1=0.0f)
{
  using src_type = typename SrcEngine::value_type;
  using dst_type = typename DstEngine::value_type;

  // Extract tensor shape - should be (N,) for a single row
  auto src_shape = shape(src);
  auto dst_shape = shape(dst);
  constexpr uint32_t N = size(src_shape);

  // Validate N is a supported size
  static_assert((N >= 1 && N <= 4) || N == 8 || N == 16 || N == 32,
                "N must be 1, 2, 3, 4, 8, 16, or 32");

  static_assert(cute::is_same_v<src_type, fp16> || cute::is_same_v<src_type, bf16> ||
                cute::is_same_v<src_type, float>,
                "tensor_pipe_exp_reduce only supports fp16/bf16/float source types");

  static_assert((red_dim == tred_red_dim::none) || (red_dim == tred_red_dim::rednd),
                "Only not reduce or rednd are supported");


  constexpr uint32_t dtype_reg_size = sizeof(uint32_t);
  constexpr uint32_t N_reg = round_up_<N * sizeof(src_type), dtype_reg_size>::value;
  constexpr uint32_t N_reg_dst = round_up_<N * sizeof(dst_type), dtype_reg_size>::value;

  // Convert tensor data to marray
  auto src_marray = tensor_to_marray<src_type, uint32_t, N_reg, N>(src);
  sycl::marray<uint32_t, N_reg_dst> dst_marray;

  // Call the inline PISA API
  float ret_dsrc1 = ::tensor_pipe_exp2_red_pisa<src_type, dst_type, N, uint32_t, red_dim, mxnd, tr_mode, dsat, xch>(src_marray, dst_marray, m, dsrc1);
  
  // Copy result back to destination tensor
  memcpy(raw_pointer_cast(dst.data()), &dst_marray, N * sizeof(dst_type));

  return ret_dsrc1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tensor processing downconvert wrapper for tensor_pipe_quantize (gtp_tcvd)
//
// Register-to-register type downconversion via the gtp_tcvd pISA instruction.
// Converts N source elements from a wider type (float, bf16, fp16) to a narrower
// type (bf8, hf8, fp4_e2m1) in registers.
//
// Expects src to be a single row tensor per work-item: shape (N,) or (1, N)
// User should slice the tensor before calling:
//   tensor_pipe_quantize<bf8, bf16>(src_tensor(work_item_id, _), dst_tensor(work_item_id, _))
//
// Template parameters:
//   TcvdDstType: semantic destination (narrow) type — bf8, hf8, or fp4_e2m1.
//                Selects the pISA .toty encoding.
//   TcvdSrcType: semantic source (wide) type — fp16, bf16, or float.
//                Selects the pISA .fromty encoding.
//
//   Both must be explicitly specified because:
//   (1) The narrow-side tensor stores raw uint8_t bytes (LDSM loads/stores
//       narrow data as raw bytes without semantic type information), so the
//       semantic narrow type (bf8/hf8/fp4) cannot be deduced from
//       DstEngine::value_type.
//   (2) The wide-side tensor may use uint16_t as its storage type (LDSM
//       loads/stores 16-bit data as raw uint16_t without distinguishing
//       bf16 from fp16), so the semantic wide type cannot be deduced from
//       SrcEngine::value_type either.
//   The pISA gtp_tcvd instruction requires both types to select the correct
//   hardware conversion micro-op.
//
//
// Supported conversions (tcvd.toty.fromty.m32nN):
//   .toty   = { .e5m2, .e4m3, .e2m1 }
//   .fromty = { .f16, .bf16, .f32 }
//
// For sub-byte types (fp4_e2m1), output bytes pack multiple elements
// (e.g., 2 fp4 values per byte, lower nibble = even element, upper = odd).
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TcvdDstType,
          typename TcvdSrcType,
          typename SrcEngine,
          typename SrcLayout,
          typename DstEngine,
          typename DstLayout>
CUTE_HOST_DEVICE
void tensor_pipe_quantize(Tensor<SrcEngine, SrcLayout> const& src,
                          Tensor<DstEngine, DstLayout>& dst)
{
  using src_type = typename SrcEngine::value_type;
  using dst_type = typename DstEngine::value_type;

  auto src_shape = shape(src);
  constexpr uint32_t N = size(src_shape);

  static_assert((N >= 1 && N <= 4) || N == 8 || N == 16 || N == 32,
                "N must be 1, 2, 3, 4, 8, 16, or 32");

   // Layouts must be compact (contiguous) for memcpy to be correct.
  static_assert(cute::size_v<SrcLayout> == cute::cosize_v<SrcLayout>,
                "Source tensor layout must be compact (size == cosize)");
  static_assert(cute::size_v<DstLayout> == cute::cosize_v<DstLayout>,
                "Destination tensor layout must be compact (size == cosize)");
 
  // Number of packed narrow bytes produced by the conversion.
  constexpr uint32_t N_dst_bytes = N * ::sizeof_bits<TcvdDstType>() / 8;
 
  // Source and destination sizes must be consistent.
  // The destination tensor stores packed narrow bytes (uint8_t). This check ensures
  // the caller provided a correctly-sized output buffer to prevent buffer overflow
  // in the final memcpy from the local register buffer (dst_u32) to dst.data().
  static_assert(cute::size_v<DstLayout> == N_dst_bytes,
                "Source and destination sizes must be consistent: size(dst) == N * sizeof_bits(TcvdDstType) / 8");

  static_assert(cute::is_same_v<TcvdSrcType, fp16> ||
                cute::is_same_v<TcvdSrcType, bf16> ||
                cute::is_same_v<TcvdSrcType, float>,
                "TcvdSrcType (.fromty) must be fp16, bf16, or float");

  static_assert(cute::is_same_v<TcvdDstType, bf8>       ||
                cute::is_same_v<TcvdDstType, hf8>       ||
                cute::is_same_v<TcvdDstType, fp4_e2m1>,
                "TcvdDstType (.toty) must be bf8 (e5m2), hf8 (e4m3), fp4_e2m1 (e2m1)");

  // Number of uint32_t registers needed to hold N source elements and destination bytes.
  // uint32_t is the register-width operand format required by the pISA gtp_tcvd instruction.
  constexpr uint32_t dtype_reg_size = sizeof(uint32_t);
  constexpr uint32_t N_reg_src = round_up_<N * sizeof(src_type), dtype_reg_size>::value;
  constexpr uint32_t N_reg_dst = round_up_<N_dst_bytes, dtype_reg_size>::value;

#if defined(__SYCL_DEVICE_ONLY__)
  // Marshal src tensor into uint32_t register buffers for the pISA instruction.
  uint32_t src_u32[N_reg_src];
  memcpy(src_u32, reinterpret_cast<const src_type*>(raw_pointer_cast(src.data())), N * sizeof(src_type));

  // Tenor Pipe downconvert: N wide elements -> N narrow elements.
  // Sub-byte types (e.g. fp4) are packed by the instruction itself (2×fp4 per byte).
  uint32_t dst_u32[N_reg_dst];
  gtp_tcvd<TcvdDstType, TcvdSrcType, N, uint32_t>(dst_u32, src_u32);

  // Copy packed result back to the destination tensor.
  memcpy(raw_pointer_cast(dst.data()), dst_u32, N_dst_bytes);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// tensor_pipe_dequantize: Use the TensorCore TCVU instruction to upconvert
// N narrow elements to wider representation.
//
// Template parameters:
//   TcvuDstType — destination (wider) type — fp16, bf16, bf8, or hf8.
//                 Selects the pISA .toty encoding.
//   TcvuSrcType — source (narrower) type — bf8, hf8, or fp4_e2m1.
//                 Selects the pISA .fromty encoding.
//
//   Both must be explicitly specified because:
//   (1) The narrow-side (source) tensor stores raw uint8_t bytes (LDSM
//       loads/stores narrow data as raw bytes without semantic type
//       information), so the semantic narrow type (bf8/hf8/fp4) cannot be
//       deduced from SrcEngine::value_type.
//   (2) The wide-side (destination) tensor may use uint16_t as its storage
//       type (LDSM loads/stores 16-bit data as raw uint16_t without
//       distinguishing bf16 from fp16), so the semantic wide type cannot
//       be deduced from DstEngine::value_type either.
//   The pISA gtp_tcvu instruction requires both types to select the correct
//   hardware conversion micro-op.
//
// Supported conversions (tcvu.toty.fromty.m32nN):
//   .toty   = { .f16, .bf16, .e5m2, .e4m3 }
//   .fromty = { .e5m2, .e4m3, .e2m1 }
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TcvuDstType,
          typename TcvuSrcType,
          typename SrcEngine,
          typename SrcLayout,
          typename DstEngine,
          typename DstLayout>
CUTE_HOST_DEVICE
void tensor_pipe_dequantize(Tensor<SrcEngine, SrcLayout> const& src,
                            Tensor<DstEngine, DstLayout>& dst)
{
  using dst_type = typename DstEngine::value_type;

  auto dst_shape = shape(dst);
  constexpr uint32_t N = size(dst_shape);

  static_assert((N >= 1 && N <= 4) || N == 8 || N == 16 || N == 32,
                "N must be 1, 2, 3, 4, 8, 16, or 32");

  // Layouts must be compact (contiguous) for memcpy to be correct.
  static_assert(cute::size_v<SrcLayout> == cute::cosize_v<SrcLayout>,
                "Source tensor layout must be compact (size == cosize)");
  static_assert(cute::size_v<DstLayout> == cute::cosize_v<DstLayout>,
                "Destination tensor layout must be compact (size == cosize)");
 
  // Number of packed narrow bytes in the source.
  constexpr uint32_t N_src_bytes = N * ::sizeof_bits<TcvuSrcType>() / 8;
 
  // Source and destination sizes must be consistent.
  // The source tensor stores packed narrow bytes (uint8_t). This check ensures
  // the caller provided a correctly-sized input buffer to prevent reading beyond
  // the source tensor's allocation in the memcpy to the local register buffer (src_u32).
  static_assert(cute::size_v<SrcLayout> == N_src_bytes,
                "Source and destination sizes must be consistent: size(src) == N * sizeof_bits(TcvuSrcType) / 8");
 

  static_assert(cute::is_same_v<TcvuSrcType, bf8>        ||
                cute::is_same_v<TcvuSrcType, hf8>        ||
                cute::is_same_v<TcvuSrcType, fp4_e2m1>,
                "TcvuSrcType (.fromty) must be bf8 (e5m2), hf8 (e4m3), fp4_e2m1 (e2m1)");

  static_assert(cute::is_same_v<TcvuDstType, fp16>  ||
                cute::is_same_v<TcvuDstType, bf16>   ||
                cute::is_same_v<TcvuDstType, bf8>    ||
                cute::is_same_v<TcvuDstType, hf8>,
                "TcvuDstType (.toty) must be fp16, bf16, bf8 (e5m2), or hf8 (e4m3)");

  // Number of uint32_t registers needed to hold N source narrow bytes and destination wide elements.
  constexpr uint32_t dtype_reg_size = sizeof(uint32_t);
  constexpr uint32_t N_reg_src = round_up_<N_src_bytes, dtype_reg_size>::value;
  constexpr uint32_t N_dst_bytes = N * sizeof(dst_type);
  constexpr uint32_t N_reg_dst = round_up_<N_dst_bytes, dtype_reg_size>::value;

#if defined(__SYCL_DEVICE_ONLY__)
  // Marshal src narrow bytes into uint32_t register buffers for the pISA instruction.
  uint32_t src_u32[N_reg_src];
  memcpy(src_u32, reinterpret_cast<const uint8_t*>(raw_pointer_cast(src.data())), N_src_bytes);

  // Tensor Pipe upconvert: N narrow elements -> N wide elements.
  uint32_t dst_u32[N_reg_dst];
  gtp_tcvu<TcvuDstType, TcvuSrcType, N, uint32_t>(dst_u32, src_u32);

  // Copy result back to the destination tensor.
  memcpy(raw_pointer_cast(dst.data()), dst_u32, N_dst_bytes);
#endif
}

}
