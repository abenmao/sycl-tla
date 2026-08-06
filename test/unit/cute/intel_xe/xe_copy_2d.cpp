/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the disclaimer.
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
 * OF THIS SOFTWARE, EVEN IF ADVISED OF POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include "cutlass/detail/layout.hpp"

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <cute/arch/copy_xe_2d.hpp>
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include "cutlass_unit_test.h"
#include "utils.hpp"

using namespace cute;
using namespace cutlass;
using namespace compat::experimental;

#define SUBGROUP_SIZE (16)

#if (IGC_VERSION_MAJOR > 2) || (IGC_VERSION_MAJOR == 2 && IGC_VERSION_MINOR >= 18)

// Kernel name for unique identification
template<class...> class XECopy2DKernelName;

// Device kernel for XE_LOAD_2D testing
// Single tile copy at a specified offset to test boundary correctness.
template <class SrcTensor, class DstTensor, int Bits, int Height, int Width>
void xe_copy_2d_kernel(SrcTensor src, DstTensor dst) {
  using namespace cute;
  using Element = typename SrcTensor::value_type;

  constexpr int TileN = Width * Bits / sizeof_bits_v<Element>;

  auto nd_item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  auto local_id = int(nd_item.get_local_id(0));

  // Create block 2D copy for load
  using CopyOp = XE_LOAD_2D<Bits, Height, Width>;
  auto tiled_copy = make_block_2d_copy(CopyOp{}, src);
  auto thr_copy = tiled_copy.get_slice(local_id);

  // Create coordinate tensor for the tile
  auto coord_shape = make_shape(Int<Height>{}, Int<TileN>{});
  Tensor coord_tile = make_identity_tensor(coord_shape);

  // Partition and load
  auto thr_src_coord = thr_copy.partition_S(coord_tile);
  auto thr_dst_frag = thr_copy.partition_fragment_D(coord_tile);
  copy(tiled_copy, thr_src_coord, thr_dst_frag);

  // Create block 2D copy for store
  using StoreOp = XE_STORE_2D<Bits, Height, Width>;
  auto tiled_store = make_block_2d_copy(StoreOp{}, dst);
  auto thr_store = tiled_store.get_slice(local_id);

  auto thr_dst_coord = thr_store.partition_D(coord_tile);
  auto thr_src_frag = thr_store.partition_fragment_S(coord_tile);

  // Copy loaded data to store fragment and store
  copy(thr_dst_frag, thr_src_frag);
  copy(tiled_store, thr_src_frag, thr_dst_coord);
}

// Host test function template
// Strategy: copy a single tile into the CENTER of a larger matrix pre-filled with sentinel values.
// The matrix has padding on all four sides (top, bottom, left, right).
// Verify: (1) the tile region matches src, (2) everything outside is untouched.
template <typename Element, int Bits, int Height, int Width, int BlockWidth = Width>
void test_xe_copy_2d() {
  using namespace cute;

  // Tile dimensions in elements
  constexpr int TileN = Width * Bits / sizeof_bits_v<Element>;

  // Padding on all four sides — must be cache-line aligned for 2D block ops
  // 2D block load/store requires surface base address to be 64-byte aligned.
  constexpr int cache_line_elems = 64 / sizeof(Element);
  constexpr int PadRows = 4;
  constexpr int PadCols = cache_line_elems;  // ensures column offset is 64-byte aligned
  constexpr int M = PadRows + Height + PadRows;
  constexpr int N = PadCols + TileN + PadCols;

  // Align row width to cache line to maintain 64-byte alignment per row
  constexpr int elem_alignment = cache_line_elems;
  constexpr int aligned_N = ((N + elem_alignment - 1) / elem_alignment) * elem_alignment;

  // Sentinel value — fill dst with this to detect out-of-bounds writes
  Element sentinel;
  if constexpr (std::is_floating_point_v<Element> ||
                std::is_same_v<Element, half_t> ||
                std::is_same_v<Element, bfloat16_t> ||
                std::is_same_v<Element, tfloat32_t>) {
    sentinel = Element(-1.0f);
  } else {
    sentinel = static_cast<Element>(0xDE);
  }

  // Allocate and initialize ALL host data
  cutlass::host_vector<Element> host_src(M * aligned_N);
  cutlass::host_vector<Element> host_dst(M * aligned_N, sentinel);

  // Initialize entire source with a known pattern (for robustness)
  // Use values 1-127 to guarantee no overlap with sentinel (0xDE=222 for int, -1.0 for float)
  for (size_t i = 0; i < host_src.size(); ++i) {
    if constexpr (std::is_floating_point_v<Element> ||
                  std::is_same_v<Element, half_t> ||
                  std::is_same_v<Element, bfloat16_t> ||
                  std::is_same_v<Element, tfloat32_t>) {
      float val = static_cast<float>((i % 127) + 1) / 255.0f;  // 1/255 .. 127/255, never -1.0
      host_src[i] = Element(val);
    } else {
      host_src[i] = static_cast<Element>((i % 127) + 1);  // 1..127, never 0xDE(222)
    }
  }

  // Copy to device
  cutlass::device_vector<Element> device_src = host_src;
  cutlass::device_vector<Element> device_dst(M * aligned_N);
  // Fill entire device dst with sentinel
  {
    cutlass::host_vector<Element> sentinel_vec(M * aligned_N, sentinel);
    device_dst = sentinel_vec;
  }

  // The tile is placed at offset (PadRows, PadCols) in the matrix.
  // Create sub-tensors pointing to that offset for kernel use.
  auto* src_offset_ptr = device_src.data() + PadRows * aligned_N + PadCols;
  auto* dst_offset_ptr = device_dst.data() + PadRows * aligned_N + PadCols;

  Tensor tensor_src =
        make_tensor(make_gmem_ptr(src_offset_ptr),
                    make_layout(make_shape(Int<Height>{}, Int<TileN>{}),
                                make_stride(Int<aligned_N>{}, _1{})));

  Tensor tensor_dst =
        make_tensor(make_gmem_ptr(dst_offset_ptr),
                    make_layout(make_shape(Int<Height>{}, Int<TileN>{}),
                                make_stride(Int<aligned_N>{}, _1{})));

  // Launch kernel — single work group, single tile copy
  auto blockDim = compat::dim3(SUBGROUP_SIZE);
  auto gridDim = compat::dim3(1);

  launch<xe_copy_2d_kernel<decltype(tensor_src), decltype(tensor_dst), Bits, Height, Width>,
         XECopy2DKernelName<decltype(tensor_src), decltype(tensor_dst)>>(
    launch_policy{
      gridDim, blockDim,
      kernel_properties{sycl_exp::sub_group_size<SUBGROUP_SIZE>}
    },
    tensor_src, tensor_dst);

  compat::wait_and_throw();
  host_dst = device_dst;

  // Verify (1): tile region [PadRows..PadRows+Height) x [PadCols..PadCols+TileN) must match src
  for (int row = 0; row < Height; ++row) {
    for (int col = 0; col < TileN; ++col) {
      int dst_idx = (PadRows + row) * aligned_N + (PadCols + col);
      int src_idx = dst_idx;  // same layout
      EXPECT_EQ(host_dst[dst_idx], host_src[src_idx])
          << "Tile data mismatch at row=" << row << " col=" << col;
    }
  }

  // Verify (2): everything outside the tile must still be sentinel
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < aligned_N; ++col) {
      bool in_tile = (row >= PadRows && row < PadRows + Height &&
                      col >= PadCols && col < PadCols + TileN);
      if (in_tile) continue;
      int idx = row * aligned_N + col;
      EXPECT_EQ(host_dst[idx], sentinel)
          << "Boundary violation at row=" << row << " col=" << col
          << " (outside tile [" << PadRows << ".." << PadRows + Height
          << ") x [" << PadCols << ".." << PadCols + TileN << "))";
    }
  }
}

TEST(PVC_CuTe_Xe, XE_COPY_2D_char) {
  test_xe_copy_2d<char, 8, 1, 16>();
  test_xe_copy_2d<char, 8, 2, 16>();
  test_xe_copy_2d<char, 8, 3, 16>();
  test_xe_copy_2d<char, 8, 4, 16>();
  test_xe_copy_2d<char, 8, 5, 16>();
  test_xe_copy_2d<char, 8, 6, 16>();
  test_xe_copy_2d<char, 8, 7, 16>();
  test_xe_copy_2d<char, 8, 8, 16>();

  test_xe_copy_2d<char, 8, 1, 32>();
  test_xe_copy_2d<char, 8, 2, 32>();
  test_xe_copy_2d<char, 8, 3, 32>();
  test_xe_copy_2d<char, 8, 4, 32>();
  test_xe_copy_2d<char, 8, 5, 32>();
  test_xe_copy_2d<char, 8, 6, 32>();
  test_xe_copy_2d<char, 8, 7, 32>();
  test_xe_copy_2d<char, 8, 8, 32>();

  test_xe_copy_2d<char, 8, 1, 64>();
  test_xe_copy_2d<char, 8, 2, 64>();
  test_xe_copy_2d<char, 8, 3, 64>();
  test_xe_copy_2d<char, 8, 4, 64>();
  test_xe_copy_2d<char, 8, 5, 64>();
  test_xe_copy_2d<char, 8, 6, 64>();
  test_xe_copy_2d<char, 8, 7, 64>();
  test_xe_copy_2d<char, 8, 8, 64>();
}

TEST(PVC_CuTe_Xe, XE_COPY_2D_uint8) {
  test_xe_copy_2d<uint8_t, 8, 1, 16>();
  test_xe_copy_2d<uint8_t, 8, 2, 16>();
  test_xe_copy_2d<uint8_t, 8, 3, 16>();
  test_xe_copy_2d<uint8_t, 8, 4, 16>();
  test_xe_copy_2d<uint8_t, 8, 5, 16>();
  test_xe_copy_2d<uint8_t, 8, 6, 16>();
  test_xe_copy_2d<uint8_t, 8, 7, 16>();
  test_xe_copy_2d<uint8_t, 8, 8, 16>();

  test_xe_copy_2d<uint8_t, 8, 1, 32>();
  test_xe_copy_2d<uint8_t, 8, 2, 32>();
  test_xe_copy_2d<uint8_t, 8, 3, 32>();
  test_xe_copy_2d<uint8_t, 8, 4, 32>();
  test_xe_copy_2d<uint8_t, 8, 5, 32>();
  test_xe_copy_2d<uint8_t, 8, 6, 32>();
  test_xe_copy_2d<uint8_t, 8, 7, 32>();
  test_xe_copy_2d<uint8_t, 8, 8, 32>();

  test_xe_copy_2d<uint8_t, 8, 1, 64>();
  test_xe_copy_2d<uint8_t, 8, 2, 64>();
  test_xe_copy_2d<uint8_t, 8, 3, 64>();
  test_xe_copy_2d<uint8_t, 8, 4, 64>();
  test_xe_copy_2d<uint8_t, 8, 5, 64>();
  test_xe_copy_2d<uint8_t, 8, 6, 64>();
  test_xe_copy_2d<uint8_t, 8, 7, 64>();
  test_xe_copy_2d<uint8_t, 8, 8, 64>();
}

TEST(PVC_CuTe_Xe, XE_COPY_2D_int8) {
  test_xe_copy_2d<int8_t, 8, 1, 16>();
  test_xe_copy_2d<int8_t, 8, 2, 16>();
  test_xe_copy_2d<int8_t, 8, 3, 16>();
  test_xe_copy_2d<int8_t, 8, 4, 16>();
  test_xe_copy_2d<int8_t, 8, 5, 16>();
  test_xe_copy_2d<int8_t, 8, 6, 16>();
  test_xe_copy_2d<int8_t, 8, 7, 16>();
  test_xe_copy_2d<int8_t, 8, 8, 16>();

  test_xe_copy_2d<int8_t, 8, 1, 32>();
  test_xe_copy_2d<int8_t, 8, 2, 32>();
  test_xe_copy_2d<int8_t, 8, 3, 32>();
  test_xe_copy_2d<int8_t, 8, 4, 32>();
  test_xe_copy_2d<int8_t, 8, 5, 32>();
  test_xe_copy_2d<int8_t, 8, 6, 32>();
  test_xe_copy_2d<int8_t, 8, 7, 32>();
  test_xe_copy_2d<int8_t, 8, 8, 32>();

  test_xe_copy_2d<int8_t, 8, 1, 64>();
  test_xe_copy_2d<int8_t, 8, 2, 64>();
  test_xe_copy_2d<int8_t, 8, 3, 64>();
  test_xe_copy_2d<int8_t, 8, 4, 64>();
  test_xe_copy_2d<int8_t, 8, 5, 64>();
  test_xe_copy_2d<int8_t, 8, 6, 64>();
  test_xe_copy_2d<int8_t, 8, 7, 64>();
  test_xe_copy_2d<int8_t, 8, 8, 64>();
}

TEST(PVC_CuTe_Xe, XE_COPY_2D_uint16) {
  test_xe_copy_2d<uint16_t, 16, 1, 16>();
  test_xe_copy_2d<uint16_t, 16, 2, 16>();
  test_xe_copy_2d<uint16_t, 16, 3, 16>();
  test_xe_copy_2d<uint16_t, 16, 4, 16>();
  test_xe_copy_2d<uint16_t, 16, 5, 16>();
  test_xe_copy_2d<uint16_t, 16, 6, 16>();
  test_xe_copy_2d<uint16_t, 16, 7, 16>();
  test_xe_copy_2d<uint16_t, 16, 8, 16>();

  test_xe_copy_2d<uint16_t, 16, 1, 32>();
  test_xe_copy_2d<uint16_t, 16, 2, 32>();
  test_xe_copy_2d<uint16_t, 16, 3, 32>();
  test_xe_copy_2d<uint16_t, 16, 4, 32>();
  test_xe_copy_2d<uint16_t, 16, 5, 32>();
  test_xe_copy_2d<uint16_t, 16, 6, 32>();
  test_xe_copy_2d<uint16_t, 16, 7, 32>();
  test_xe_copy_2d<uint16_t, 16, 8, 32>();
}

TEST(PVC_CuTe_Xe, XE_COPY_2D_int16) {
  test_xe_copy_2d<int16_t, 16, 1, 16>();
  test_xe_copy_2d<int16_t, 16, 2, 16>();
  test_xe_copy_2d<int16_t, 16, 3, 16>();
  test_xe_copy_2d<int16_t, 16, 4, 16>();
  test_xe_copy_2d<int16_t, 16, 5, 16>();
  test_xe_copy_2d<int16_t, 16, 6, 16>();
  test_xe_copy_2d<int16_t, 16, 7, 16>();
  test_xe_copy_2d<int16_t, 16, 8, 16>();

  test_xe_copy_2d<int16_t, 16, 1, 32>();
  test_xe_copy_2d<int16_t, 16, 2, 32>();
  test_xe_copy_2d<int16_t, 16, 3, 32>();
  test_xe_copy_2d<int16_t, 16, 4, 32>();
  test_xe_copy_2d<int16_t, 16, 5, 32>();
  test_xe_copy_2d<int16_t, 16, 6, 32>();
  test_xe_copy_2d<int16_t, 16, 7, 32>();
  test_xe_copy_2d<int16_t, 16, 8, 32>();
}

TEST(PVC_CuTe_Xe, XE_COPY_2D_half) {
  test_xe_copy_2d<half_t, 16, 1, 16>();
  test_xe_copy_2d<half_t, 16, 2, 16>();
  test_xe_copy_2d<half_t, 16, 3, 16>();
  test_xe_copy_2d<half_t, 16, 4, 16>();
  test_xe_copy_2d<half_t, 16, 5, 16>();
  test_xe_copy_2d<half_t, 16, 6, 16>();
  test_xe_copy_2d<half_t, 16, 7, 16>();
  test_xe_copy_2d<half_t, 16, 8, 16>();

  test_xe_copy_2d<half_t, 16, 1, 32>();
  test_xe_copy_2d<half_t, 16, 2, 32>();
  test_xe_copy_2d<half_t, 16, 3, 32>();
  test_xe_copy_2d<half_t, 16, 4, 32>();
  test_xe_copy_2d<half_t, 16, 5, 32>();
  test_xe_copy_2d<half_t, 16, 6, 32>();
  test_xe_copy_2d<half_t, 16, 7, 32>();
  test_xe_copy_2d<half_t, 16, 8, 32>();
}

TEST(PVC_CuTe_Xe, XE_COPY_2D_bfloat16) {
  test_xe_copy_2d<bfloat16_t, 16, 1, 16>();
  test_xe_copy_2d<bfloat16_t, 16, 2, 16>();
  test_xe_copy_2d<bfloat16_t, 16, 3, 16>();
  test_xe_copy_2d<bfloat16_t, 16, 4, 16>();
  test_xe_copy_2d<bfloat16_t, 16, 5, 16>();
  test_xe_copy_2d<bfloat16_t, 16, 6, 16>();
  test_xe_copy_2d<bfloat16_t, 16, 7, 16>();
  test_xe_copy_2d<bfloat16_t, 16, 8, 16>();

  test_xe_copy_2d<bfloat16_t, 16, 1, 32>();
  test_xe_copy_2d<bfloat16_t, 16, 2, 32>();
  test_xe_copy_2d<bfloat16_t, 16, 3, 32>();
  test_xe_copy_2d<bfloat16_t, 16, 4, 32>();
  test_xe_copy_2d<bfloat16_t, 16, 5, 32>();
  test_xe_copy_2d<bfloat16_t, 16, 6, 32>();
  test_xe_copy_2d<bfloat16_t, 16, 7, 32>();
  test_xe_copy_2d<bfloat16_t, 16, 8, 32>();
}

TEST(PVC_CuTe_Xe, XE_COPY_2D_uint32) {
  test_xe_copy_2d<uint32_t, 32, 1, 16>();
  test_xe_copy_2d<uint32_t, 32, 2, 16>();
  test_xe_copy_2d<uint32_t, 32, 3, 16>();
  test_xe_copy_2d<uint32_t, 32, 4, 16>();
  test_xe_copy_2d<uint32_t, 32, 5, 16>();
  test_xe_copy_2d<uint32_t, 32, 6, 16>();
  test_xe_copy_2d<uint32_t, 32, 7, 16>();
  test_xe_copy_2d<uint32_t, 32, 8, 16>();
}

TEST(PVC_CuTe_Xe, XE_COPY_2D_int32) {
  test_xe_copy_2d<int32_t, 32, 1, 16>();
  test_xe_copy_2d<int32_t, 32, 2, 16>();
  test_xe_copy_2d<int32_t, 32, 3, 16>();
  test_xe_copy_2d<int32_t, 32, 4, 16>();
  test_xe_copy_2d<int32_t, 32, 5, 16>();
  test_xe_copy_2d<int32_t, 32, 6, 16>();
  test_xe_copy_2d<int32_t, 32, 7, 16>();
  test_xe_copy_2d<int32_t, 32, 8, 16>();
}

TEST(PVC_CuTe_Xe, XE_COPY_2D_float) {
  test_xe_copy_2d<float, 32, 1, 16>();
  test_xe_copy_2d<float, 32, 2, 16>();
  test_xe_copy_2d<float, 32, 3, 16>();
  test_xe_copy_2d<float, 32, 4, 16>();
  test_xe_copy_2d<float, 32, 5, 16>();
  test_xe_copy_2d<float, 32, 6, 16>();
  test_xe_copy_2d<float, 32, 7, 16>();
  test_xe_copy_2d<float, 32, 8, 16>();
}

TEST(PVC_CuTe_Xe, XE_COPY_2D_tfloat32) {
  test_xe_copy_2d<tfloat32_t, 32, 1, 16>();
  test_xe_copy_2d<tfloat32_t, 32, 2, 16>();
  test_xe_copy_2d<tfloat32_t, 32, 3, 16>();
  test_xe_copy_2d<tfloat32_t, 32, 4, 16>();
  test_xe_copy_2d<tfloat32_t, 32, 5, 16>();
  test_xe_copy_2d<tfloat32_t, 32, 6, 16>();
  test_xe_copy_2d<tfloat32_t, 32, 7, 16>();
  test_xe_copy_2d<tfloat32_t, 32, 8, 16>();
}

#else

// For the fallback case
#include "cutlass_unit_test.h"

TEST(PVC_CuTe_Xe, XE_COPY_2D_SKIPPED) {
  GTEST_SKIP() << "XE_COPY_2D tests require IGC version 2.18 or higher. skipped";
}

#endif
