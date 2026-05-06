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

#include "cutlass_unit_test.h"

#include "./adma_reduce_testbed.hpp"

using namespace cute;
using namespace cutlass::test;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// XE4 ADMA Store-Reduce Unit Tests
///
/// Tests XE4_ADMA_STORE_REDUCE<T, Rop, BType> — atomic SLM → gmem reduction via
/// async_tensor_fred (float types) and async_tensor_ired (integer types).
///
/// Three test categories:
///
/// 1. Float reductions (fred): half/float/bf16 Add, half Min/Max, 64x64, dynamic stride
/// 2. Integer reductions (ired): int/uint32 Add, Smin/Smax, Umin/Umax, And/Or/Xor
/// 3. Barrier type variants: explicit Abarrier for float Add and int Add
///
/// IMPORTANT: These tests require REAL HARDWARE (not TBX simulator).
/// The TBX simulator does not support the send.dma.tensor.reduce_l2g instruction variant.
/// All 18 tests compile and generate correct asm but return base_val (no reduction)
/// on TBX. Expected to pass on actual XE4 hardware.
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  FLOAT STORE-REDUCTIONS (fred) — SLM → gmem with reduction
///
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe, ADMA_Fred_StoreReduce_Add_half_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<half_t, RedOp::Add>(gmem_layout, smem_layout,
                                             half_t(10), half_t(3));
}

TEST(XE4_CuTe, ADMA_Fred_StoreReduce_Add_float_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<float, RedOp::Add>(gmem_layout, smem_layout, 10.0f, 3.0f);
}

TEST(XE4_CuTe, ADMA_Fred_StoreReduce_Add_bf16_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<sycl::ext::oneapi::bfloat16, RedOp::Add>(
      gmem_layout, smem_layout,
      sycl::ext::oneapi::bfloat16(10), sycl::ext::oneapi::bfloat16(3));
}

TEST(XE4_CuTe, ADMA_Fred_StoreReduce_Min_half_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  // base=100, operand=42 → min=42
  test_adma_store_reduce<half_t, RedOp::Min>(gmem_layout, smem_layout,
                                             half_t(100), half_t(42));
}

TEST(XE4_CuTe, ADMA_Fred_StoreReduce_Max_half_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  // base=42, operand=100 → max=100
  test_adma_store_reduce<half_t, RedOp::Max>(gmem_layout, smem_layout,
                                             half_t(42), half_t(100));
}

TEST(XE4_CuTe, ADMA_Fred_StoreReduce_Add_half_64x64)
{
  Layout smem_layout = Layout<Shape<_64,_64>, Stride<_1,_64>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<half_t, RedOp::Add>(gmem_layout, smem_layout,
                                             half_t(5), half_t(7));
}

TEST(XE4_CuTe, ADMA_Fred_StoreReduce_Add_float_DynStride)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = make_layout(make_shape(32,32), make_stride(Int<1>{}, 1024));
  auto cta_tile = product_each(shape(smem_layout));
  test_adma_store_reduce<float, RedOp::Add>(gmem_layout, smem_layout, cta_tile, 10.0f, 3.0f);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  INTEGER STORE-REDUCTIONS (ired) — SLM → gmem with reduction
///
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe, ADMA_Ired_StoreReduce_Add_int32_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<int, RedOp::Add>(gmem_layout, smem_layout, 10, 7);
}

TEST(XE4_CuTe, ADMA_Ired_StoreReduce_Add_uint32_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<uint32_t, RedOp::Add>(gmem_layout, smem_layout, 10u, 7u);
}

TEST(XE4_CuTe, ADMA_Ired_StoreReduce_Smin_int32_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<int, RedOp::Smin>(gmem_layout, smem_layout, 100, -5);
}

TEST(XE4_CuTe, ADMA_Ired_StoreReduce_Smax_int32_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<int, RedOp::Smax>(gmem_layout, smem_layout, -5, 100);
}

TEST(XE4_CuTe, ADMA_Ired_StoreReduce_Umin_uint32_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<uint32_t, RedOp::Umin>(gmem_layout, smem_layout, 100u, 5u);
}

TEST(XE4_CuTe, ADMA_Ired_StoreReduce_Umax_uint32_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<uint32_t, RedOp::Umax>(gmem_layout, smem_layout, 5u, 100u);
}

TEST(XE4_CuTe, ADMA_Ired_StoreReduce_And_uint32_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<uint32_t, RedOp::And>(gmem_layout, smem_layout, 0xFF00u, 0x0FF0u);
}

TEST(XE4_CuTe, ADMA_Ired_StoreReduce_Or_uint32_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<uint32_t, RedOp::Or>(gmem_layout, smem_layout, 0xFF00u, 0x00FFu);
}

TEST(XE4_CuTe, ADMA_Ired_StoreReduce_Xor_uint32_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<uint32_t, RedOp::Xor>(gmem_layout, smem_layout, 0xAAAAu, 0x5555u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
///  BARRIER TYPE VARIANTS
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe, ADMA_StoreReduce_Abarrier_float_Add)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<float, RedOp::Add, cute::BarrierType::Abarrier>(
      gmem_layout, smem_layout, 10.0f, 3.0f);
}

TEST(XE4_CuTe, ADMA_StoreReduce_Abarrier_int_Add)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<int, RedOp::Add, cute::BarrierType::Abarrier>(
      gmem_layout, smem_layout, 10, 7);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  DOUBLE FLOAT STORE-REDUCTIONS (fred) — validates rd_type<double> fix
///
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe, ADMA_Fred_StoreReduce_Add_double_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<double, RedOp::Add>(gmem_layout, smem_layout, 10.0, 3.0);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  INT64 INTEGER STORE-REDUCTIONS (ired) — validates int64_t classifier fix
///
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe, ADMA_Ired_StoreReduce_Add_int64_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<int64_t, RedOp::Add>(gmem_layout, smem_layout,
                                              int64_t(10), int64_t(3));
}

TEST(XE4_CuTe, ADMA_Ired_StoreReduce_Smin_int64_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<int64_t, RedOp::Smin>(gmem_layout, smem_layout,
                                               int64_t(100), int64_t(-5));
}

TEST(XE4_CuTe, ADMA_Ired_StoreReduce_Smax_int64_32x32)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_store_reduce<int64_t, RedOp::Smax>(gmem_layout, smem_layout,
                                               int64_t(-5), int64_t(100));
}
