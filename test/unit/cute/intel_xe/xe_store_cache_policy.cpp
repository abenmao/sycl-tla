/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include "cutlass/detail/layout.hpp"

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <cute/arch/copy_xe_2d.hpp>
#include <cute/util/compat.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <type_traits>

#include "cutlass_unit_test.h"
#include "utils.hpp"

using namespace cute;
using namespace cutlass;
using namespace compat::experimental;

static_assert(std::is_same_v<XE_STORE_2D<8, 2, 64>,
                             XE_STORE_2D<8, 2, 64, XeStoreCachePolicy::kDefault>>);
static_assert(is_xe_store_cache_v<cute::C<XeStoreCachePolicy::kWT_UC_UC>>);
static_assert(is_xe_block_2d_atom_v<XE_STORE_2D<8, 2, 64, XeStoreCachePolicy::kWB_WB_UC>>);

template <class...>
class XeStoreCachePolicyKernelName;

template <class SrcTensor, class DstTensor, XeStoreCachePolicy CachePolicy>
void xe_store_cache_policy_kernel(SrcTensor src, DstTensor dst) {
  auto nd_item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  auto local_id = int(nd_item.get_local_id(0));

  using LoadOp = XE_LOAD_2D<8, 2, 64>;
  auto tiled_load = make_block_2d_copy(LoadOp{}, src);
  auto thr_load = tiled_load.get_slice(local_id);
  auto coord_tile = make_identity_tensor(make_shape(Int<2>{}, Int<64>{}));
  auto thr_src_coord = thr_load.partition_S(coord_tile);
  auto thr_dst_frag = thr_load.partition_fragment_D(coord_tile);
  copy(tiled_load, thr_src_coord, thr_dst_frag);

  using StoreOp = XE_STORE_2D<8, 2, 64, CachePolicy>;
  auto tiled_store = make_block_2d_copy(StoreOp{}, dst);
  auto thr_store = tiled_store.get_slice(local_id);
  auto thr_dst_coord = thr_store.partition_D(coord_tile);
  auto thr_src_frag = thr_store.partition_fragment_S(coord_tile);
  copy(thr_dst_frag, thr_src_frag);
  copy(tiled_store, thr_src_frag, thr_dst_coord);
}

template <XeStoreCachePolicy CachePolicy>
void test_xe_store_cache_policy() {
  constexpr int height = 2;
  constexpr int width = 64;
  constexpr int pad_rows = 4;
  constexpr int pad_cols = 64;
  constexpr int rows = pad_rows + height + pad_rows;
  constexpr int columns = pad_cols + width + pad_cols;
  constexpr uint8_t sentinel = 0xde;

  cutlass::host_vector<uint8_t> host_src(rows * columns);
  cutlass::host_vector<uint8_t> host_dst(rows * columns, sentinel);
  for (size_t i = 0; i < host_src.size(); ++i) {
    host_src[i] = static_cast<uint8_t>((i % 127) + 1);
  }

  cutlass::device_vector<uint8_t> device_src = host_src;
  cutlass::device_vector<uint8_t> device_dst = host_dst;
  auto offset = pad_rows * columns + pad_cols;
  auto tensor_src = make_tensor(
      make_gmem_ptr(device_src.data() + offset),
      make_layout(make_shape(Int<height>{}, Int<width>{}),
                  make_stride(Int<columns>{}, _1{})));
  auto tensor_dst = make_tensor(
      make_gmem_ptr(device_dst.data() + offset),
      make_layout(make_shape(Int<height>{}, Int<width>{}),
                  make_stride(Int<columns>{}, _1{})));

  launch<xe_store_cache_policy_kernel<decltype(tensor_src), decltype(tensor_dst), CachePolicy>,
         XeStoreCachePolicyKernelName<std::integral_constant<XeStoreCachePolicy, CachePolicy>>>(
      launch_policy{
          compat::dim3(1), compat::dim3(16),
          kernel_properties{sycl_exp::sub_group_size<16>}},
      tensor_src, tensor_dst);
  compat::wait_and_throw();
  host_dst = device_dst;

  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      bool in_tile = row >= pad_rows && row < pad_rows + height &&
                     column >= pad_cols && column < pad_cols + width;
      auto index = row * columns + column;
      if (in_tile) {
        EXPECT_EQ(host_dst[index], host_src[index]);
      } else {
        EXPECT_EQ(host_dst[index], sentinel);
      }
    }
  }
}

TEST(XE35_CuTe_Xe, XE_STORE_2D_cache_policies) {
  test_xe_store_cache_policy<XeStoreCachePolicy::kDefault>();
  test_xe_store_cache_policy<XeStoreCachePolicy::kUC_UC_UC>();
  test_xe_store_cache_policy<XeStoreCachePolicy::kUC_UC_WB>();
  test_xe_store_cache_policy<XeStoreCachePolicy::kUC_WB_UC>();
  test_xe_store_cache_policy<XeStoreCachePolicy::kUC_WB_WB>();
  test_xe_store_cache_policy<XeStoreCachePolicy::kWT_UC_UC>();
  test_xe_store_cache_policy<XeStoreCachePolicy::kWT_UC_WB>();
  test_xe_store_cache_policy<XeStoreCachePolicy::kWT_WB_UC>();
  test_xe_store_cache_policy<XeStoreCachePolicy::kWT_WB_WB>();
  test_xe_store_cache_policy<XeStoreCachePolicy::kWB_UC_UC>();
  test_xe_store_cache_policy<XeStoreCachePolicy::kWB_UC_WB>();
  test_xe_store_cache_policy<XeStoreCachePolicy::kWB_WB_UC>();
}
