/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2025 INTEL CORPORATION & AFFILIATES. All rights reserved.
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
#include <cute/config.hpp>
#include <cute/numeric/numeric_types.hpp>
#include "xe4_util.hpp"

#if defined(SYCL_INTEL_XE4_TARGET)
// Config
namespace cute {

CUTE_DEVICE void cluster_arrive()
{
  INLINE_PISA("cbarrier.arrive;" ::);
}

CUTE_DEVICE void cluster_arrive_relaxed()
{
  INLINE_PISA("cbarrier.arrive.relaxed;" ::);
}

CUTE_DEVICE void cluster_wait()
{
  INLINE_PISA("cbarrier.wait;" ::);
}

CUTE_DEVICE void cluster_wait_relaxed()
{
  INLINE_PISA("cbarrier.wait.relaxed;" ::);
}

CUTE_DEVICE void cluster_sync()
{
  cluster_arrive();
  cluster_wait();
}

// Returns the relative dim3 block rank local to the cluster.
CUTE_DEVICE dim3 block_id_in_cluster()
{
  return {0,0,0};
}

// Returns the dim3 cluster shape.
CUTE_DEVICE dim3 cluster_shape()
{
  return {1,1,1};
}
// Get 1D ctaid in a cluster.
CUTE_DEVICE uint32_t block_rank_in_cluster()
{
  return 0;
}

// Elect one thread in the warp. The elected thread gets its predicate set to true, all others obtain false.
CUTE_HOST_DEVICE uint32_t elect_one_sync()
{
#if defined(SYCL_INTEL_TARGET)
  return sycl::ext::oneapi::this_work_item::get_sub_group().leader();
#else
  return true;
#endif
}

} // end namespace cute
  //
#endif
