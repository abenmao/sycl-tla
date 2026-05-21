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

#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/atom/mma_traits_xe4_tmm.hpp>

namespace gemm_tmm {

using namespace cute;

template<typename T>
struct show_type_t;

// A dummy stream that absorbs all operator<< calls and does nothing
struct NullStream {
	template <typename T>
	CUTE_HOST_DEVICE constexpr const NullStream& operator<<(const T&) 
	const noexcept { return *this; }
}; // NullStream //

template<int bM, int bN, int bK, 
    typename MMA_Op, typename CTA_Warp_Layout, typename Copy_Op, 
    int kNumProducerSGs, 
    class TensorA, class TensorB, class TensorD,
		class StreamT = NullStream>
CUTE_HOST_DEVICE
void tmma_gemm(TensorA const& gA, TensorB const& gB, TensorD & gD, 
    sycl::nd_item<3> it, StreamT out = StreamT{} ) {

  (void)out;

	constexpr int kSubGroupSize = cute::size(typename MMA_Op::ThrID{});
  static_assert(kSubGroupSize == 32,
      "tmma_gemm requires a subgroup size of exactly 32!");

  using TA = typename TensorA::value_type;
  using TB = typename TensorB::value_type;
  using TD = typename TensorD::value_type;

  // assumes local_range(1,1,workgroup_size) //
  auto my_wg_level_lane_id =  // lane-id within the workgroup //
    (it.get_local_id(2) - (kSubGroupSize*kNumProducerSGs)); 
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Op>{}, CTA_Warp_Layout{});

  auto thr_mma = tiled_mma.get_slice((int)my_wg_level_lane_id);
  // Partition the tensors into atoms tiles//
  auto tAgA = thr_mma.partition_A(gA);
  auto tBgB = thr_mma.partition_B(gB);
  auto tDgD = thr_mma.partition_C(gD);

  // Allocate registers for each of the lanes //
  auto tArA = thr_mma.make_fragment_A(tAgA);
  auto tBrB = thr_mma.make_fragment_B(tBgB);
  auto tDrD = thr_mma.make_fragment_C(tDgD);

  // copy to registers //
  auto tiled_copy_A = make_tiled_copy_A(Copy_Atom<Copy_Op, TA>{}, tiled_mma);
  auto tiled_copy_B = make_tiled_copy_B(Copy_Atom<Copy_Op, TB>{}, tiled_mma);
  auto tiled_copy_D = make_tiled_copy_C(Copy_Atom<Copy_Op, TD>{}, tiled_mma);

#if 0
  copy(tiled_copy_A, tAgA, tArA);
  copy(tiled_copy_B, tAgB, tArB);
  copy(tiled_copy_D, tAgD, tArD);
#endif

  for (int i=0; i<size(tAgA); i++) {
    tArA(i) = tAgA(i);
  }

  for (int i=0; i<size(tBgB); i++) {
    tBrB(i) = tBgB(i);
  }

  for (int i=0; i<size(tDgD); i++) {
    tDrD(i) = tDgD(i);
  }


  // Call cute::gemm with tiled_mma directly.
  cute::gemm(tiled_mma, tArA, tBrB, tDrD);
  copy(tDrD, tDgD);
}

} // namespace gemm_tmm
