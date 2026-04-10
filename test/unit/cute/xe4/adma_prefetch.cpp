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

#include "./adma_prefetch_testbed.hpp"

using namespace cute;
using namespace cutlass::test;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// XE4 ADMA Prefetch Unit Tests
///
/// Two test categories:
///
/// 1. Standalone prefetch (ADMA_Prefetch_*):
///    Creates 3 ADMA atoms: LOAD + STORE + PREFETCH.
///    Tests that XE4_ADMA_PREFETCH fires correctly as a directly-executable atom
///    (no .with() needed). Validates copy(adma_prefetch, src, dst) path.
///
/// 2. Prefetch-from-load (ADMA_PrefetchFromLoad_*):
///    Creates only 2 ADMA atoms: LOAD + STORE.
///    Tests the SM90-style API: cute::prefetch(tiled_copy, src) which derives
///    prefetch traits from the load atom via CopyOp::PREFETCH typedef.
///    Validates the converting constructor in Copy_Traits<XE4_ADMA_PREFETCH>.
///
/// All tests verify: h_in == h_out (prefetch must not corrupt data).
/// Tested on TBX simulator — all 10 tests pass.
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// Prefetch + Load roundtrip: 32x32 Col-major
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe, ADMA_Prefetch_32x32_Col_half)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  {
    // Static stride, matching gmem/smem
    Layout gmem_layout = smem_layout;
    test_adma_prefetch<half_t>(gmem_layout, smem_layout);
  }
  {
    // Dynamic col-major stride
    Layout gmem_layout = make_layout(make_shape(32,32), GenColMajor{});
    test_adma_prefetch<half_t>(gmem_layout, smem_layout);
  }
  {
    // Large dynamic stride (stride gap > tile)
    Layout gmem_layout = make_layout(make_shape(32,32), make_stride(Int<1>{}, 1024));
    test_adma_prefetch<half_t>(gmem_layout, smem_layout);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Prefetch + Load roundtrip: 32x32 Row-major
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe, ADMA_Prefetch_32x32_Row_half)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_32, _1>>{};
  {
    Layout gmem_layout = smem_layout;
    test_adma_prefetch<half_t>(gmem_layout, smem_layout);
  }
  {
    Layout gmem_layout = make_layout(make_shape(32,32), make_stride(1024, Int<1>{}));
    test_adma_prefetch<half_t>(gmem_layout, smem_layout);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Prefetch + Load roundtrip: 32x32, 32-bit elements (float)
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe, ADMA_Prefetch_32x32_Col_float)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_prefetch<float>(gmem_layout, smem_layout);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Prefetch + Load roundtrip: Larger tile 64x64
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe, ADMA_Prefetch_64x64_Col_half)
{
  Layout smem_layout = Layout<Shape<_64,_64>, Stride<_1,_64>>{};
  Layout gmem_layout = smem_layout;
  test_adma_prefetch<half_t>(gmem_layout, smem_layout);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Prefetch + Load roundtrip: Multi-stage (exercises prefetch-ahead pattern)
// The testbed kernel already prefetches stage N+1 while loading stage N.
// This test uses a gmem that is 2x the smem tile to exercise multi-stage.
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe, ADMA_Prefetch_MultiStage_half)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  // gmem is 2 tiles wide → 2 stages → exercises prefetch-ahead
  Layout gmem_layout = make_layout(make_shape(32,64), make_stride(Int<1>{}, 32));
  auto cta_tile = product_each(shape(smem_layout));
  test_adma_prefetch<half_t>(gmem_layout, smem_layout, cta_tile);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// cute::prefetch() API tests — prefetch derived from load atom (SM90-style)
// Exercises cute::prefetch(tiled_copy, src) from prefetch.hpp.
// Uses XE4_ADMA_LOAD::PREFETCH typedef to derive Copy_Traits<XE4_ADMA_PREFETCH> from load traits.
// No separate XE4_ADMA_PREFETCH atom is created; only 2 tensor descriptors are allocated.
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(XE4_CuTe, ADMA_PrefetchFromLoad_32x32_Col_half)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  {
    Layout gmem_layout = smem_layout;
    test_adma_prefetch_from_load<half_t>(gmem_layout, smem_layout);
  }
  {
    Layout gmem_layout = make_layout(make_shape(32,32), GenColMajor{});
    test_adma_prefetch_from_load<half_t>(gmem_layout, smem_layout);
  }
  {
    Layout gmem_layout = make_layout(make_shape(32,32), make_stride(Int<1>{}, 1024));
    test_adma_prefetch_from_load<half_t>(gmem_layout, smem_layout);
  }
}

TEST(XE4_CuTe, ADMA_PrefetchFromLoad_32x32_Row_half)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_32, _1>>{};
  {
    Layout gmem_layout = smem_layout;
    test_adma_prefetch_from_load<half_t>(gmem_layout, smem_layout);
  }
  {
    Layout gmem_layout = make_layout(make_shape(32,32), make_stride(1024, Int<1>{}));
    test_adma_prefetch_from_load<half_t>(gmem_layout, smem_layout);
  }
}

TEST(XE4_CuTe, ADMA_PrefetchFromLoad_32x32_Col_float)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = smem_layout;
  test_adma_prefetch_from_load<float>(gmem_layout, smem_layout);
}

TEST(XE4_CuTe, ADMA_PrefetchFromLoad_64x64_Col_half)
{
  Layout smem_layout = Layout<Shape<_64,_64>, Stride<_1,_64>>{};
  Layout gmem_layout = smem_layout;
  test_adma_prefetch_from_load<half_t>(gmem_layout, smem_layout);
}

TEST(XE4_CuTe, ADMA_PrefetchFromLoad_MultiStage_half)
{
  Layout smem_layout = Layout<Shape<_32,_32>, Stride<_1,_32>>{};
  Layout gmem_layout = make_layout(make_shape(32,64), make_stride(Int<1>{}, 32));
  auto cta_tile = product_each(shape(smem_layout));
  test_adma_prefetch_from_load<half_t>(gmem_layout, smem_layout, cta_tile);
}
