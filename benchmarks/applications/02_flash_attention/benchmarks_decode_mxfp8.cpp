/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation. All rights reserved.
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

#include "benchmark_runner.hpp"
#include "fmha_configuration.hpp"

using namespace cutlass::flash_attention;

using mxfp8_e5m2 = cutlass::mx_float8_t<float_e5m2_t>;
using mxfp8_e4m3 = cutlass::mx_float8_t<float_e4m3_t>;

/* ---------------------------------------- HeadDim = 128 ----------------------------------------- */
// True MXFP8 block-scaled decode, aligned with the develop-branch decode example
// (examples/06_bmg_flash_attention/06_xe_fmha_fwd_mxfp.cpp, PR #933) ShapeQK8 tile:
//   ShapeQK  = Shape<_8, _256, _64>
//   ShapePV  = Shape<_8, _64, _256>
//   ShapeOut = Shape<_8, _128>
//   SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>
// => WgTileQ=8, WgTileK=256, WgTileV=64, SgTileQ=8 (num_sg_q=1), SgTileK=32 (num_sg_k=8),
//    HeadDimQK=64, HeadDimV=128.
// WgTileQ=8 gives SGTileQ=8 => BDPAS RepeatCount M=gcd(8,8)=8, satisfying XE_BDPAS_TT's
// static_assert(M==8) that WgTileQ=1 would violate.  BlockScale=true selects the block-scaled
// BDPAS MMA (true MXFP8), and GqaFusion=true packs gqa_group query heads into the 8 Q rows,
// matching the example's non-split GQA-fusion path.

// mxfp8_e4m3
using CriFMHADecode_MXFP8E4M3_MXFP8E4M3_MXFP8E4M3_BF16_RCR_WgQ8K256V64_SgQ8K32_HDimQK64V128_Causal_FixedLen = FMHAConfigGenWithTileShape</*Mode*/FMHAMode::Decode,
  /*ElementQ*/ mxfp8_e4m3::DataType, /*ElementK*/ mxfp8_e4m3::DataType, /*ElementV*/ mxfp8_e4m3::DataType, /*ElementO*/ cutlass::bfloat16_t,
  /*LayoutQ*/ cutlass::layout::RowMajor, /*LayoutK*/ cutlass::layout::ColumnMajor, /*LayoutV*/ cutlass::layout::RowMajor, /*LayoutO*/ cutlass::layout::RowMajor,
  /*ElementScale*/ mxfp8_e4m3::ScaleFactorType, /*Causal*/ true, /*VarLen*/ false, /*CachedKV*/ false, /*PagedKV*/ false, /*Persistent*/ false, /*BlockScale*/ true, /*WgTileQ*/ 8, /*WgTileK*/ 256, /*WgTileV*/ 64,
  /*SgTileQ*/ 8, /*SgTileK*/ 32, /*HeadDimQK*/ 64, /*HeadDimV*/ 128, /*GqaFusion*/ true
>::type;

using CriFMHADecode_MXFP8E4M3_MXFP8E4M3_MXFP8E4M3_BF16_RCR_WgQ8K256V64_SgQ8K32_HDimQK64V128_NonCausal_FixedLen = FMHAConfigGenWithTileShape</*Mode*/FMHAMode::Decode,
  /*ElementQ*/ mxfp8_e4m3::DataType, /*ElementK*/ mxfp8_e4m3::DataType, /*ElementV*/ mxfp8_e4m3::DataType, /*ElementO*/ cutlass::bfloat16_t,
  /*LayoutQ*/ cutlass::layout::RowMajor, /*LayoutK*/ cutlass::layout::ColumnMajor, /*LayoutV*/ cutlass::layout::RowMajor, /*LayoutO*/ cutlass::layout::RowMajor,
  /*ElementScale*/ mxfp8_e4m3::ScaleFactorType, /*Causal*/ false, /*VarLen*/ false, /*CachedKV*/ false, /*PagedKV*/ false, /*Persistent*/ false, /*BlockScale*/ true, /*WgTileQ*/ 8, /*WgTileK*/ 256, /*WgTileV*/ 64,
  /*SgTileQ*/ 8, /*SgTileK*/ 32, /*HeadDimQK*/ 64, /*HeadDimV*/ 128, /*GqaFusion*/ true
>::type;

// mxfp8_e5m2
using CriFMHADecode_MXFP8E5M2_MXFP8E5M2_MXFP8E5M2_BF16_RCR_WgQ8K256V64_SgQ8K32_HDimQK64V128_Causal_FixedLen = FMHAConfigGenWithTileShape</*Mode*/FMHAMode::Decode,
  /*ElementQ*/ mxfp8_e5m2::DataType, /*ElementK*/ mxfp8_e5m2::DataType, /*ElementV*/ mxfp8_e5m2::DataType, /*ElementO*/ cutlass::bfloat16_t,
  /*LayoutQ*/ cutlass::layout::RowMajor, /*LayoutK*/ cutlass::layout::ColumnMajor, /*LayoutV*/ cutlass::layout::RowMajor, /*LayoutO*/ cutlass::layout::RowMajor,
  /*ElementScale*/ mxfp8_e5m2::ScaleFactorType, /*Causal*/ true, /*VarLen*/ false, /*CachedKV*/ false, /*PagedKV*/ false, /*Persistent*/ false, /*BlockScale*/ true, /*WgTileQ*/ 8, /*WgTileK*/ 256, /*WgTileV*/ 64,
  /*SgTileQ*/ 8, /*SgTileK*/ 32, /*HeadDimQK*/ 64, /*HeadDimV*/ 128, /*GqaFusion*/ true
>::type;

using CriFMHADecode_MXFP8E5M2_MXFP8E5M2_MXFP8E5M2_BF16_RCR_WgQ8K256V64_SgQ8K32_HDimQK64V128_NonCausal_FixedLen = FMHAConfigGenWithTileShape</*Mode*/FMHAMode::Decode,
  /*ElementQ*/ mxfp8_e5m2::DataType, /*ElementK*/ mxfp8_e5m2::DataType, /*ElementV*/ mxfp8_e5m2::DataType, /*ElementO*/ cutlass::bfloat16_t,
  /*LayoutQ*/ cutlass::layout::RowMajor, /*LayoutK*/ cutlass::layout::ColumnMajor, /*LayoutV*/ cutlass::layout::RowMajor, /*LayoutO*/ cutlass::layout::RowMajor,
  /*ElementScale*/ mxfp8_e5m2::ScaleFactorType, /*Causal*/ false, /*VarLen*/ false, /*CachedKV*/ false, /*PagedKV*/ false, /*Persistent*/ false, /*BlockScale*/ true, /*WgTileQ*/ 8, /*WgTileK*/ 256, /*WgTileV*/ 64,
  /*SgTileQ*/ 8, /*SgTileK*/ 32, /*HeadDimQK*/ 64, /*HeadDimV*/ 128, /*GqaFusion*/ true
>::type;

CUTLASS_CREATE_FMHA_BENCHMARK(CriFMHADecode_MXFP8E4M3_MXFP8E4M3_MXFP8E4M3_BF16_RCR_WgQ8K256V64_SgQ8K32_HDimQK64V128_Causal_FixedLen);
CUTLASS_CREATE_FMHA_BENCHMARK(CriFMHADecode_MXFP8E4M3_MXFP8E4M3_MXFP8E4M3_BF16_RCR_WgQ8K256V64_SgQ8K32_HDimQK64V128_NonCausal_FixedLen);
CUTLASS_CREATE_FMHA_BENCHMARK(CriFMHADecode_MXFP8E5M2_MXFP8E5M2_MXFP8E5M2_BF16_RCR_WgQ8K256V64_SgQ8K32_HDimQK64V128_Causal_FixedLen);
CUTLASS_CREATE_FMHA_BENCHMARK(CriFMHADecode_MXFP8E5M2_MXFP8E5M2_MXFP8E5M2_BF16_RCR_WgQ8K256V64_SgQ8K32_HDimQK64V128_NonCausal_FixedLen);

static void register_flash_attention_decode_benchmarks_mxfp8() {
  CUTLASS_FMHA_BENCHMARK(CriFMHADecode_MXFP8E4M3_MXFP8E4M3_MXFP8E4M3_BF16_RCR_WgQ8K256V64_SgQ8K32_HDimQK64V128_Causal_FixedLen);
  CUTLASS_FMHA_BENCHMARK(CriFMHADecode_MXFP8E4M3_MXFP8E4M3_MXFP8E4M3_BF16_RCR_WgQ8K256V64_SgQ8K32_HDimQK64V128_NonCausal_FixedLen);
  CUTLASS_FMHA_BENCHMARK(CriFMHADecode_MXFP8E5M2_MXFP8E5M2_MXFP8E5M2_BF16_RCR_WgQ8K256V64_SgQ8K32_HDimQK64V128_Causal_FixedLen);
  CUTLASS_FMHA_BENCHMARK(CriFMHADecode_MXFP8E5M2_MXFP8E5M2_MXFP8E5M2_BF16_RCR_WgQ8K256V64_SgQ8K32_HDimQK64V128_NonCausal_FixedLen);
}
