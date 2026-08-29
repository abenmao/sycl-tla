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

#include "cutlass/util/device_memory.h"
#include "cutlass/util/initialize_block.hpp"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "../common.hpp"

#include <vector>
#include <string>

namespace cutlass::benchmark {

// Cache flush mode enum
enum class CacheFlushMode {
  FlushKernel = 0,     // Run a cache-flushing kernel between iterations (default)
  RotateBuffers = 1    // Pre-allocate multiple buffer sets, rotate input pointers per iteration
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// CacheFlushHelper — manages cache flush logic for benchmark iterations.
///
/// Encapsulates both FlushKernel mode (runs a flush kernel between iterations)
/// and RotateBuffers mode (pre-allocates multiple buffer sets, rotates pointers).
/// The benchmark loop uses a uniform interface regardless of mode.
///////////////////////////////////////////////////////////////////////////////////////////////////
template <
  class ElementA,
  class ElementB,
  class ElementC,
  class ElementScaleA,
  class ElementScaleB,
  bool IsBlockScaled>
struct CacheFlushHelper {

  // Buffer copies: 1 for FlushKernel mode, iterations+1 for RotateBuffers mode
  std::vector<DeviceAllocation<ElementA>> rotate_blocks_A;
  std::vector<DeviceAllocation<ElementB>> rotate_blocks_B;
  std::vector<DeviceAllocation<ElementC>> rotate_blocks_C;
  std::vector<DeviceAllocation<ElementScaleA>> rotate_blocks_scaleA;
  std::vector<DeviceAllocation<ElementScaleB>> rotate_blocks_scaleB;
  int32_t num_rotate_copies = 0;
  int32_t current_idx_ = 0;
  CacheFlushMode mode = CacheFlushMode::FlushKernel;

  void flush_kernel() {
    auto q = compat::get_default_queue();

    using cache_dtype = uint32_t;

    static const size_t cache_size = cutlass::get_llc_size();

    constexpr size_t local_size = 512;
    size_t num_elements = cutlass::ceil_div(cache_size, sizeof(cache_dtype));
    size_t global_size = cutlass::ceil_div(num_elements, local_size) * local_size;

    static cutlass::DeviceAllocation<uint8_t> dev_cache_block(global_size * sizeof(cache_dtype) + 64);
    cache_dtype* mem_to = reinterpret_cast<cache_dtype*>(dev_cache_block.get());
    cache_dtype* mem_from = reinterpret_cast<cache_dtype*>(dev_cache_block.get() + sizeof(cache_dtype));

    q.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](auto idx) {
      size_t i = idx.get_global_id();
      if (i < num_elements) {
        mem_to[0] += mem_from[i];
      }
    });

    q.wait();
  }

  /// Initialize the helper.
  /// @param flush_mode       Which cache flush strategy to use.
  /// @param iterations       Number of benchmark iterations.
  /// @param M, N, K, L       Problem dimensions.
  /// @param scale_k          ceil_div(K, GROUP_SIZE) for block-scaled.
  /// @param seed             Random seed base for buffer initialization.
  /// @param blk_A, blk_B, blk_C  Source buffer pointers to copy from.
  /// @param blk_SA, blk_SB   Source scale buffer pointers (block-scaled only).
  template <class InitScaleFn>
  void initialize(
      CacheFlushMode flush_mode,
      int iterations,
      int M, int N, int K, int L,
      int scale_k,
      uint64_t seed,
      ElementA const* blk_A,
      ElementB const* blk_B,
      ElementC const* blk_C,
      ElementScaleA const* blk_SA,
      ElementScaleB const* blk_SB,
      InitScaleFn&& init_scale_fn) {

    mode = flush_mode;

    // -- Compute element counts and byte sizes for each buffer --
    auto size_A       = static_cast<std::size_t>(M) * K * L;
    auto size_B       = static_cast<std::size_t>(K) * N * L;
    auto size_C       = static_cast<std::size_t>(M) * N * L;
    auto size_A_bytes = size_A * sizeof_bits_v<ElementA> / 8;
    auto size_B_bytes = size_B * sizeof_bits_v<ElementB> / 8;
    auto size_C_bytes = size_C * sizeof_bits_v<ElementC> / 8;

    auto size_SA       = static_cast<std::size_t>(scale_k) * L * M;
    auto size_SB       = static_cast<std::size_t>(scale_k) * L * N;
    auto size_SA_bytes = size_SA * sizeof_bits_v<ElementScaleA> / 8;
    auto size_SB_bytes = size_SB * sizeof_bits_v<ElementScaleB> / 8;

    // -- Determine how many buffer copies to allocate --
    //  Cap total allocation at 4x LLC to avoid OOM on large problems.
    static const size_t cache_size = cutlass::get_llc_size();
    const auto kMaxSize = cache_size * 4;

    //  Total footprint of one copy (includes scale buffers when block-scaled).
    auto total_size = IsBlockScaled
                    ? size_A_bytes + size_B_bytes + size_C_bytes + size_SA_bytes + size_SB_bytes
                    : size_A_bytes + size_B_bytes + size_C_bytes;

    // FlushKernel:    1 copy  — all iterations reuse the same buffers; cache flushed between them.
    // RotateBuffers:  N copies — index 0 for verification, 1..N-1 for benchmark iterations.
    //                 Capped so total allocation stays within kMaxSize.
    if (mode == CacheFlushMode::RotateBuffers) {
      num_rotate_copies = min(iterations + 1, ceil_div(kMaxSize, total_size));
    } else if (mode == CacheFlushMode::FlushKernel) {
      num_rotate_copies = 1;
    } else {
      throw std::runtime_error("Unknown CacheFlushMode");
    }

    rotate_blocks_A.resize(num_rotate_copies);
    rotate_blocks_B.resize(num_rotate_copies);
    rotate_blocks_C.resize(num_rotate_copies);

    // All copies: initialized with deterministic random data.
    // Copy 0 uses the same seeds as initialize() so GEMM results match verification.
    for (int i = 0; i < num_rotate_copies; i++) {
      rotate_blocks_A[i].reset(size_A);
      rotate_blocks_B[i].reset(size_B);
      rotate_blocks_C[i].reset(size_C);
      initialize_block(rotate_blocks_A[i], seed + 2023 + i);
      initialize_block(rotate_blocks_B[i], seed + 2022 + i);
      initialize_block(rotate_blocks_C[i], seed + 2021 + i);
    }

    if constexpr (IsBlockScaled) {
      rotate_blocks_scaleA.resize(num_rotate_copies);
      rotate_blocks_scaleB.resize(num_rotate_copies);

      for (int i = 0; i < num_rotate_copies; i++) {
        rotate_blocks_scaleA[i].reset(size_SA);
        rotate_blocks_scaleB[i].reset(size_SB);
        init_scale_fn(rotate_blocks_scaleA[i]);
        init_scale_fn(rotate_blocks_scaleB[i]);
      }
    }
  }

  /// Call before each benchmark iteration (inside PauseTiming).
  /// Advances the internal buffer index and flushes cache if needed.
  void prepare() {
    ++current_idx_;
    if (mode == CacheFlushMode::FlushKernel) {
      flush_kernel();
    }
  }

  /// Get the A pointer for the current iteration.
  auto* ptr_A() const {
    return rotate_blocks_A[current_idx_ % num_rotate_copies].get();
  }

  /// Get the B pointer for the current iteration.
  auto* ptr_B() const {
    return rotate_blocks_B[current_idx_ % num_rotate_copies].get();
  }

  /// Get the C pointer for the current iteration.
  auto* ptr_C() const {
    return rotate_blocks_C[current_idx_ % num_rotate_copies].get();
  }

  /// Get the ScaleA pointer for the current iteration.
  auto* ptr_SA() const {
    return rotate_blocks_scaleA[current_idx_ % num_rotate_copies].get();
  }

  /// Get the ScaleB pointer for the current iteration.
  auto* ptr_SB() const {
    return rotate_blocks_scaleB[current_idx_ % num_rotate_copies].get();
  }
};

} // namespace cutlass::benchmark
