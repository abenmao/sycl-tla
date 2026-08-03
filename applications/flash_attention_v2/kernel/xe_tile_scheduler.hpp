/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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


#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"
#include "cute/layout.hpp"

namespace cutlass::fmha::kernel {
namespace detail {
struct EmptyDivmod {};
}

#ifndef FMHA_SATURATION_CORES
#define FMHA_SATURATION_CORES 0   // 0 => use the caller-provided fallback
#endif

template <int DefaultCores = FMHA_SATURATION_CORES>
inline int fmha_split_saturation_cores(int fallback_cores) {
  // //TODO: remove this after we tuned the saturation cores
  // static const int env_cores = [] {
  //   const char* env = std::getenv("FMHA_SATURATION_CORES");
  //   if (env == nullptr) { return -1; }
  //   int v = std::atoi(env);
  //   return v > 0 ? v : -1;
  // }();
  // if (env_cores > 0) { return env_cores; }
  if constexpr (DefaultCores > 0) { return DefaultCores; }
  return fallback_cores;
}

inline int fmha_dynamic_num_partitions(
    int saturation_cores, int num_batch_heads, int local_k_blocks,
    int max_num_partitions) {
  if (num_batch_heads <= 0 || local_k_blocks <= 0) {
    return 1;
  }

  int target_partitions = cute::max(1, saturation_cores / num_batch_heads);
  target_partitions = cute::min(target_partitions, max_num_partitions);
  target_partitions = cute::min(target_partitions, local_k_blocks);

  int blocks_per_partition = cute::ceil_div(local_k_blocks, target_partitions);
  return cute::ceil_div(local_k_blocks, blocks_per_partition);
}

// Minimum packed-Q rows (head_group_q * seq_len_qo) for KV prefetch to pay off on
// the GQA-fusion path. Measured on CRI/32 XeCore: 8 rows -17%, 16 rows -4%,
// 24 rows +15%, 32 rows +18%, >32 rows (halved BLK_K) +55..85%.
inline constexpr int kFmhaGqaKVPrefetchMinRows = 24;

// The host picks the packed-Q tile from a ladder keyed on the same row count, so
// BLK_Q is an exact compile-time proxy for it.
template <class TileShapeQK>
CUTLASS_HOST_DEVICE constexpr bool fmha_gqa_disable_kv_prefetch() {
  return int(cute::get<0>(TileShapeQK{})) < kFmhaGqaKVPrefetchMinRows;
}

template <bool OneBatch = false, bool NoGQA = false, bool CausalMask = false,
          bool GqaFusion = false, bool DisablePrefetchV = false,
          bool DisableKVPrefetch = true,
          bool InterleaveCausalQTiles = DisablePrefetchV>
struct XeFHMAIndividualTileScheduler {
  static constexpr bool kGqaFusion = GqaFusion;
  static constexpr bool kDisablePrefetchV = DisablePrefetchV;
  static constexpr bool kDisableKVPrefetch = DisableKVPrefetch;
  static constexpr bool kInterleaveCausalQTiles = InterleaveCausalQTiles;
  // Grid-layout selector. The causal grid (V, batch*heads, Q) places the Q tile
  // in the slow (z) dimension with reverse dispatch -- good for prefill causal
  // load balancing, but for Q_PACKED_DECODE fusion it scatters the packed M-tiles
  // that SHARE a KV head into different dispatch waves, so the second m-tile
  // re-reads the (long) KV from HBM instead of reusing L2. For the fusion+gather
  // case we therefore use the non-causal layout (M-tile in the fast y dimension)
  // so a KV head's m-tiles co-schedule and share its K/V in L2. The causal MASK
  // is still applied in the mainloop (driven by CausalMask), independent of this.
#if defined(Q_PACKED_DECODE)
  static constexpr bool kUseCausalGrid = CausalMask && !GqaFusion;
#else
  static constexpr bool kUseCausalGrid = CausalMask;
#endif
  using NumHeadsDivmod   = cute::conditional_t<OneBatch, detail::EmptyDivmod, FastDivmod>;
  using HeadGroupDivmod  = cute::conditional_t<NoGQA || GqaFusion, detail::EmptyDivmod, FastDivmod>;

  struct Params {
    dim3 grid;
    NumHeadsDivmod  divmod_num_heads;
    HeadGroupDivmod divmod_head_group_q;
    int gqa_group_size;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeFHMAIndividualTileScheduler(Params const& params) : params(params) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape, KernelHardwareInfo hw_info,
      TileShape const& tile_shape)
  {
    using namespace cute;

    int heads_in_grid = GqaFusion ? shape.num_heads_kv : shape.num_heads_q;

    // Number of rows packed into the Q (M) dimension of the grid. For the native
    // kGqaFusion path this is just seq_len_qo (the head_group_q * seq_len_qo rows
    // are looped serially inside a single WG). With Q_PACKED_DECODE the packed
    // M dimension (head_group_q * seq_len_qo) is instead SPLIT across parallel
    // WGs, so each WG runs one M-tile with a single KV pass -- this removes the
    // serial per-m-tile KV re-read that regresses seq_len_qo > 1 decode.
    // NOTE: unlike the .cur Q_PACKED_DECODE this is the *parallelization* half
    // only -- it keeps the native GqaFusion strided Q/O view and does NOT do a
    // physical gather/scatter into a global scratch buffer.
    int q_rows_in_grid = int(shape.seq_len_qo);
#if defined(Q_PACKED_DECODE)
    if constexpr (GqaFusion) {
      q_rows_in_grid = (shape.num_heads_q / shape.num_heads_kv) * int(shape.seq_len_qo);
    }
#endif

    dim3 grid;
    if constexpr (kUseCausalGrid) {
      // Causal: grid layout (V, batch*heads, Q) groups all heads for the same
      // Q tile adjacent, enabling wave-level load balancing under causal mask.
      grid = dim3(size(ceil_div(shape.head_size_vo, get<1>(tile_shape))),   // V
            size(shape.batch * heads_in_grid),                         // (h,b)
                  size(ceil_div(q_rows_in_grid,     get<0>(tile_shape))));  // Q
    } else {
      // Non-causal (and Q_PACKED_DECODE fusion): grid layout (V, Q, batch*heads)
      // keeps a KV head's packed M-tiles adjacent for L2 reuse.
      grid = dim3(size(ceil_div(shape.head_size_vo, get<1>(tile_shape))),   // V
                  size(ceil_div(q_rows_in_grid,     get<0>(tile_shape))),   // Q
            size(shape.batch * heads_in_grid));                        // (h,b)
    }
    Params p{};
    p.grid = grid;
    p.gqa_group_size = shape.num_heads_q / shape.num_heads_kv;
    if constexpr (!OneBatch) {
      p.divmod_num_heads = FastDivmod(heads_in_grid);
    }
    if constexpr (!NoGQA && !GqaFusion) {
      p.divmod_head_group_q = FastDivmod(shape.num_heads_q / shape.num_heads_kv);
    }
    return p;
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    int head;
    int idx_b;
    if constexpr (kUseCausalGrid) {
      // Causal grid layout: (V, batch*heads, Q).
      if constexpr (OneBatch) {
        // Single batch: grid.y == num_heads_q. No divmod needed.
        head  = BlockIdxY();
        idx_b = 0;
      } else {
        idx_b = BlockIdxY();
        params.divmod_num_heads(idx_b, head, idx_b);
      }

      if constexpr (InterleaveCausalQTiles) {
        // Interleave heavy and light Q tiles.
        int const q_idx = int(BlockIdxZ());
        int const num_q_tiles = params.grid.z;
        int q_tile = 0;
        if ((q_idx & 1) == 0) {
          q_tile = num_q_tiles - 1 - (q_idx >> 1);
        } else {
          q_tile = (q_idx >> 1);
        }
        return make_coord(q_tile, int(BlockIdxX()), head, idx_b);
      } else {
        // Restore the previous causal dispatch
        int q_tile = params.grid.z - 1 - BlockIdxZ();
        return make_coord(q_tile, BlockIdxX(), head, idx_b);
      }
    } else {
      // Non-causal grid layout: (V, Q, batch*heads).
      if constexpr (OneBatch) {
        // Single batch: grid.z == num_heads_q. No divmod needed.
        head  = BlockIdxZ();
        idx_b = 0;
      } else {
        idx_b = BlockIdxZ();
        params.divmod_num_heads(idx_b, head, idx_b);
      }
      // Reverse Q dispatch: last Q tile first.
      int q_tile = params.grid.y - 1 - BlockIdxY();
      return make_coord(q_tile, BlockIdxX(), head, idx_b);
    }
  }

  CUTLASS_DEVICE
  int divide_head_group(int head_q) const {
    if constexpr (NoGQA || GqaFusion) {
      return head_q;
    } else {
      return params.divmod_head_group_q.div(head_q);
    }
  }

  CUTLASS_DEVICE
  int get_gqa_group_size() const { return params.gqa_group_size; }

  CUTLASS_DEVICE
  XeFHMAIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

struct XeFHMAIndividualPersistentTileScheduler {

  struct Params {
    dim3 grid;
    FastDivmod divmod_num_partitions;
    int num_partitions;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeFHMAIndividualPersistentTileScheduler(Params const& params) : params(params) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape, KernelHardwareInfo hw_info,
      TileShape const& tile_shape, int saturation_cores_hint,
      int local_k_blocks, int max_num_partitions)
  {
    using namespace cute;

    int head_group_q = shape.num_heads_q / shape.num_heads_kv;
    int packed_q_rows = int(shape.seq_len_qo) * head_group_q;
    int num_batch_heads = shape.batch * shape.num_heads_kv;
    int saturation_cores = fmha_split_saturation_cores(
      saturation_cores_hint > 0 ? saturation_cores_hint : hw_info.sm_count);
    int num_partitions = fmha_dynamic_num_partitions(
        saturation_cores, num_batch_heads, local_k_blocks, max_num_partitions);
    dim3 grid(size(ceil_div(shape.head_size_vo, get<1>(tile_shape))),     // V
              size(ceil_div(packed_q_rows,      get<0>(tile_shape))),     // packed Q
              size(num_batch_heads * num_partitions));                    // (batch_head, partition)

    return Params{grid, {num_partitions}, num_partitions};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    int linear_idx = BlockIdxZ();
    int batch_head_id;
    int partition_id;
    params.divmod_num_partitions(batch_head_id, partition_id, linear_idx);
    return make_coord(BlockIdxY(), BlockIdxX(), batch_head_id, partition_id);
  }

  CUTLASS_DEVICE
  XeFHMAIndividualPersistentTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

}  // namespace cutlass::fmha::kernel
