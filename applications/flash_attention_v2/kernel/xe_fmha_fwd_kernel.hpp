/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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

#include <array>
#include <cstddef>

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp"
#include "cute/util/type_traits.hpp"
#include "cute/util/xe_split_barrier.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
template <bool IsVarLen_ = false>
struct FMHAProblemShape {
  using SeqLenType = cute::conditional_t<IsVarLen_, cutlass::fmha::collective::VariableLength, int>;
  int batch;
  int num_heads_q, num_heads_kv;
  SeqLenType seq_len_qo, seq_len_kv, seq_len_kv_cache;
  int head_size_qk, head_size_vo;
};

static constexpr int kBlockScaleRowAlign = 64;

CUTLASS_HOST_DEVICE
int fmha_scaleq_rows(bool packs_gqa_q, int seq_len_qo, int head_group_q) {
  return packs_gqa_q ? cutlass::round_up(head_group_q * seq_len_qo, kBlockScaleRowAlign)
                     : seq_len_qo;
}

CUTLASS_HOST_DEVICE
int fmha_scale_cache_rows(int kv_cache_rows) {
  return cutlass::round_up(kv_cache_rows, kBlockScaleRowAlign);
}

template <class TensorScaleQ, class TensorScaleK, class TensorScaleV>
struct FMHABlockScaleTensors {
  TensorScaleQ Q;
  TensorScaleK K;
  TensorScaleV V;
  TensorScaleV P;
  TensorScaleK K_cache;
  TensorScaleV V_cache;

  CUTLASS_DEVICE static FMHABlockScaleTensors make_null() {
    auto null_ptr = make_gmem_ptr(static_cast<typename TensorScaleQ::element_type*>(nullptr));
    auto null_shape = make_shape(0, 0, 0, 0);
    auto null_k = make_tensor(null_ptr, make_layout(null_shape, decltype(stride(TensorScaleK{})){}));
    auto null_v = make_tensor(null_ptr, make_layout(null_shape, decltype(stride(TensorScaleV{})){}));
    return {make_tensor(null_ptr, make_layout(null_shape, decltype(stride(TensorScaleQ{})){})),
            null_k, null_v, null_v, null_k, null_v};
  }
};

template <bool PackedStrides = false, class KernelParams>
CUTLASS_DEVICE auto
make_blockscale_tensors(KernelParams const& p, int rows_q, int heads_q,
                        int seq_len_kv, int batch_dim,
                        int offset_q = 0, int offset_k = 0, int offset_v = 0,
                        int rows_kv_cache = 0, int offset_k_cache = 0, int offset_v_cache = 0)
{
  using ElementScale = cute::remove_const_t<cute::remove_pointer_t<decltype(p.scaleQ)>>;
  using StrideScaleQ = cute::remove_cvref_t<decltype(p.dScaleQ)>;
  using StrideScaleK = cute::remove_cvref_t<decltype(p.dScaleK)>;
  using StrideScaleV = cute::remove_cvref_t<decltype(p.dScaleV)>;

  auto const& s = p.shape;
  int groups_qk = cute::ceil_div(s.head_size_qk, p.group_size);
  int groups_v  = cute::ceil_div(seq_len_kv, p.group_size);
  int groups_v_cache = cute::ceil_div(rows_kv_cache, p.group_size);

  auto shape_Q = make_shape(rows_q, groups_qk, heads_q, batch_dim);
  auto shape_K = make_shape(seq_len_kv, groups_qk, s.num_heads_kv, batch_dim);
  auto shape_V = make_shape(s.head_size_vo, groups_v, s.num_heads_kv, batch_dim);
  auto shape_K_cache = make_shape(rows_kv_cache, groups_qk, s.num_heads_kv, batch_dim);
  auto shape_V_cache = make_shape(s.head_size_vo, groups_v_cache, s.num_heads_kv, batch_dim);

  StrideScaleQ stride_Q = p.dScaleQ;
  StrideScaleK stride_K = p.dScaleK;
  StrideScaleV stride_V = p.dScaleV;
  StrideScaleK stride_K_cache = p.dScaleK_cache;
  StrideScaleV stride_V_cache = p.dScaleV_cache;
  if constexpr (PackedStrides) {
    stride_Q = cutlass::make_cute_packed_stride(StrideScaleQ{}, shape_Q);
    stride_K = cutlass::make_cute_packed_stride(StrideScaleK{}, shape_K);
    stride_V = cutlass::make_cute_packed_stride(StrideScaleV{}, shape_V);
    stride_K_cache = cutlass::make_cute_packed_stride(StrideScaleK{}, shape_K_cache);
    stride_V_cache = cutlass::make_cute_packed_stride(StrideScaleV{}, shape_V_cache);
  }

  Tensor ScaleQ = make_tensor(make_gmem_ptr(const_cast<ElementScale*>(p.scaleQ + offset_q)),
                              make_layout(shape_Q, stride_Q));
  Tensor ScaleK = make_tensor(make_gmem_ptr(const_cast<ElementScale*>(p.scaleK + offset_k)),
                              make_layout(shape_K, stride_K));
  Tensor ScaleV = make_tensor(make_gmem_ptr(const_cast<ElementScale*>(p.scaleV + offset_v)),
                              make_layout(shape_V, stride_V));
  Tensor ScaleP = make_tensor(make_gmem_ptr(const_cast<ElementScale*>(p.scaleP + offset_v)),
                              make_layout(shape_V, stride_V));
  Tensor ScaleK_cache = make_tensor(make_gmem_ptr(const_cast<ElementScale*>(p.scaleK_cache + offset_k_cache)),
                                    make_layout(shape_K_cache, stride_K_cache));
  Tensor ScaleV_cache = make_tensor(make_gmem_ptr(const_cast<ElementScale*>(p.scaleV_cache + offset_v_cache)),
                                    make_layout(shape_V_cache, stride_V_cache));

  return FMHABlockScaleTensors<decltype(ScaleQ), decltype(ScaleK), decltype(ScaleV)>{
      ScaleQ, ScaleK, ScaleV, ScaleP, ScaleK_cache, ScaleV_cache};
}

///////////////////////////////////////////////////////////////////////////////

// A sequence of length seq_len_kv_cache occupies ceil_div(seq_len_kv_cache, page_size) pages
CUTLASS_HOST_DEVICE
int paged_kv_cache_rows(int seq_len_kv_cache, int page_size) {
  return page_size > 0 ? cute::ceil_div(seq_len_kv_cache, page_size) * page_size
                       : seq_len_kv_cache;
}

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeFMHAFwdKernel {

public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len = cutlass::fmha::collective::is_variable_length_v<typename ProblemShape::SeqLenType>;
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using SubgroupLayoutQK = typename CollectiveMainloop::SubgroupLayoutQK;
  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;
  using ElementScale = typename CollectiveMainloop::TensorScaleQ::element_type;
  using StrideScaleQ = decltype(stride(typename CollectiveMainloop::TensorScaleQ{}));
  using StrideScaleK = decltype(stride(typename CollectiveMainloop::TensorScaleK{}));
  using StrideScaleV = decltype(stride(typename CollectiveMainloop::TensorScaleV{}));
  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));
  static constexpr bool BlockScale = CollectiveMainloop::BlockScale;
  using ScaleTensors = FMHABlockScaleTensors<typename CollectiveMainloop::TensorScaleQ,
                                             typename CollectiveMainloop::TensorScaleK,
                                             typename CollectiveMainloop::TensorScaleV>;

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using FragSPartialRow = typename CollectiveMainloop::FragSPartialRow;

  // Tile scheduler derived types
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr bool kPacksGqaQ = TileScheduler::kGqaFusion;

  static_assert(!(BlockScale && kPacksGqaQ && is_var_len),
                "BlockScale with GQA-fused Q rows does not support variable-length sequences");

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));
  static constexpr bool kReducePhaseSeparate = false;

  using ElementLSE = void;

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                     : sizeof(SharedStorage);

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ *Q;
    StrideQ dQ;
    const ElementK *K;
    StrideK dK;
    const ElementV *V;
    StrideV dV;
    ElementO *O;
    StrideO dO;
    // Arguments order affects kernel-entry payload layout and instruction
    // pipeline, keep hot arguments at the front to improve performance.
    float scale_k = 1.f;
    float scale_v = 1.f;
    float scale_q = 1.f;
    const ElementK *K_cache;
    StrideK dK_cache{};
    const ElementV *V_cache;
    StrideV dV_cache{};
    const ElementScale *scaleQ = nullptr;
    StrideScaleQ dScaleQ{};
    const ElementScale *scaleK = nullptr;
    StrideScaleK dScaleK{};
    const ElementScale *scaleV = nullptr;
    StrideScaleV dScaleV{};
    int group_size = 32;
    const ElementScale *scaleP = nullptr;
    StrideScaleV dScaleP{};
    const ElementScale *scaleK_cache = nullptr;
    StrideScaleK dScaleK_cache{};
    const ElementScale *scaleV_cache = nullptr;
    StrideScaleV dScaleV_cache{};
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  static Params to_underlying_arguments(Arguments const &args, void *workspace) {
    return {args.kernel,
            CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
            TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{})};
  }

  static bool can_implement(Arguments const &args) {
    return CollectiveMainloop::can_implement(args.mainloop)
        && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const &args) { return 0; }

  static cutlass::Status initialize_workspace(Arguments const &args, void *workspace = nullptr,
                                              cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const &params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(SGPerWG::value * intel::sg_size, 1, 1); }

  CUTLASS_DEVICE
  Shape<int, int, int> get_sequence_length_shape(ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      auto sequence_lengths = cutlass::fmha::collective::apply_variable_length(
          Shape<VariableLength, VariableLength, VariableLength>{
              problem_shape.seq_len_qo, problem_shape.seq_len_kv, problem_shape.seq_len_kv_cache}, batch);
      return Shape<int, int, int>{get<0>(sequence_lengths), get<1>(sequence_lengths),
          CollectiveMainloop::PagedKV ? get<2>(sequence_lengths) : 0};
    } else {
      return Shape<int, int, int>{problem_shape.seq_len_qo, problem_shape.seq_len_kv,
          CollectiveMainloop::PagedKV ? problem_shape.seq_len_kv_cache : 0};
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf)
  {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    auto &p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_q, idx_b] = tile_scheduler.get_block_coord(); // (Q,V,h,b)
      auto blk_qv = make_coord(blk_q, blk_v);
      int head = tile_scheduler.divide_head_group(head_q);

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv, seq_len_kv_cache] = sequence_length_shape;
      if (blk_q * get<0>(TileShapeQK{}) >= seq_len_qo) continue;

      int discard_seq_coord = 0;
      int full_tile_offset = 0;
      int seq_len_new = seq_len_kv;
      int seq_len_new_wg = seq_len_kv;

#if defined(CUTLASS_TEST_FOR_CRI)
      constexpr bool kIndependentSubgroups =
          is_empty_v<MainloopSharedStorage> && is_empty_v<EpilogueSharedStorage>
          && !TileScheduler::kGqaFusion;
#else
      constexpr bool kIndependentSubgroups = false;
#endif
      int q_offset_sg = 0;
      if constexpr (CollectiveMainloop::CausalMask || kIndependentSubgroups) {
        auto cS = make_identity_tensor(take<0,2>(TiledMMAQK{}.tile_mnk()));
        auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
        auto q_offset_wi = get<0>(tScS(0));
        q_offset_sg = group_broadcast(sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);
      }
#if defined(CUTLASS_TEST_FOR_CRI)
      if constexpr (kIndependentSubgroups) {
        if (blk_q * get<0>(TileShapeQK{}) + q_offset_sg >= seq_len_qo) continue;
      }
#endif

      if constexpr (CollectiveMainloop::CausalMask) {
        int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));
        int offset = cute::min(seq_len_qo, seq_len_kv);
        discard_seq_coord = seq_len_qo - offset;
        full_tile_offset = seq_len_kv - offset;
        int seq_coord = cute::min(seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));
        if (seq_coord < discard_seq_coord) continue;
        seq_len_new = full_tile_offset + cute::min(seq_len_kv, seq_coord - discard_seq_coord) + q_sg_tile;

        int q_offset_sg_max = get<0>(TileShapeQK{}) - q_sg_tile;
        int seq_coord_wg = cute::min(seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg_max));
        seq_len_new_wg = full_tile_offset + cute::min(seq_len_kv, seq_coord_wg - discard_seq_coord) + q_sg_tile;
      }
      const int seq_len = seq_len_new + seq_len_kv_cache;
      // Compute k_blocks as sum of cache tiles + new tiles to avoid losing new data
      // when seq_len_kv_cache is not a multiple of the tile size.
      int k_blocks;
      int k_blocks_prefetch;
      if constexpr (CollectiveMainloop::CausalMask || CollectiveMainloop::PagedKV) {
        const int kblocks_cache = CollectiveMainloop::PagedKV ? cute::ceil_div(seq_len_kv_cache, get<1>(TileShapeQK{})) : 0;
        const int kblocks_new = cute::ceil_div(seq_len_new, get<1>(TileShapeQK{}));
        k_blocks = kblocks_cache + kblocks_new;
        const int kblocks_new_wg = cute::ceil_div(seq_len_new_wg, get<1>(TileShapeQK{}));
        k_blocks_prefetch = kblocks_cache + kblocks_new_wg;
      } else {
        k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));
        k_blocks_prefetch = k_blocks;
      }

      int kv_cache_rows = seq_len_kv_cache;
      if constexpr (CollectiveMainloop::PagedKV) {
        kv_cache_rows = paged_kv_cache_rows(seq_len_kv_cache, params.mainloop.page_size);
      }

      int offset_q = 0, offset_k = 0, offset_v = 0, offset_o = 0;
      int offset_k_cache = 0, offset_v_cache = 0;
      int batch_dim = s.batch;
      int l_coord = idx_b;
      if constexpr (is_var_len) {
        batch_dim = 1;
        l_coord = 0;
        auto qo_cumulative = s.seq_len_qo.cumulative_length;
        auto kv_cumulative = s.seq_len_kv.cumulative_length;
        offset_q = s.num_heads_q * s.head_size_qk * qo_cumulative[idx_b];
        offset_k = s.num_heads_kv * s.head_size_qk * kv_cumulative[idx_b];
        offset_v = s.num_heads_kv * s.head_size_vo * kv_cumulative[idx_b];
        offset_o = s.num_heads_q * s.head_size_vo * qo_cumulative[idx_b];
        if (s.seq_len_kv_cache.cumulative_length) {
          auto kv_cumulative_cache = s.seq_len_kv_cache.cumulative_length;
          int rows_before_batch = kv_cumulative_cache[idx_b];
          if constexpr (CollectiveMainloop::PagedKV) {
            if (params.mainloop.num_pages_per_seq) {
              rows_before_batch = params.mainloop.num_pages_per_seq[idx_b] * params.mainloop.page_size;
            }
          }
          offset_k_cache = s.num_heads_kv * s.head_size_qk * rows_before_batch;
          offset_v_cache = s.num_heads_kv * s.head_size_vo * rows_before_batch;
        }
      }

      auto shape_Q = make_shape(seq_len_qo, s.head_size_qk, s.num_heads_q, batch_dim);
      auto shape_K = make_shape(seq_len_kv, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V = make_shape(s.head_size_vo, seq_len_kv, s.num_heads_kv, batch_dim);
      auto shape_O = make_shape(seq_len_qo, s.head_size_vo, s.num_heads_q, batch_dim);

      auto shape_K_cache = make_shape(kv_cache_rows, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V_cache = make_shape(s.head_size_vo, kv_cache_rows, s.num_heads_kv, batch_dim);

      auto dcQ = make_subbyte_aware_ptr<ElementQ>(p.Q, offset_q);
      auto dcK = make_subbyte_aware_ptr<ElementK>(p.K, offset_k);
      auto dcV = make_subbyte_aware_ptr<ElementV>(p.V, offset_v);
      decltype(make_subbyte_aware_ptr<ElementK>(p.K_cache, offset_k_cache)) dcK_cache{};
      decltype(make_subbyte_aware_ptr<ElementV>(p.V_cache, offset_v_cache)) dcV_cache{};
      if constexpr (CollectiveMainloop::PagedKV) {
        dcK_cache = make_subbyte_aware_ptr<ElementK>(p.K_cache, offset_k_cache);
        dcV_cache = make_subbyte_aware_ptr<ElementV>(p.V_cache, offset_v_cache);
      }
      auto ptrO = p.O + offset_o;

      StrideQ stride_q = p.dQ;
      StrideK stride_k = p.dK;
      StrideV stride_v = p.dV;
      StrideO stride_o = p.dO;
      StrideK stride_k_cache{};
      StrideV stride_v_cache{};
      if constexpr (CollectiveMainloop::PagedKV) {
        stride_k_cache = p.dK_cache;
        stride_v_cache = p.dV_cache;
      }
      if constexpr (is_var_len) {
        stride_q = cutlass::make_cute_packed_stride(StrideQ{}, shape_Q);
        stride_k = cutlass::make_cute_packed_stride(StrideK{}, shape_K);
        stride_v = cutlass::make_cute_packed_stride(StrideV{}, shape_V);
        stride_o = cutlass::make_cute_packed_stride(StrideO{}, shape_O);
        if constexpr (CollectiveMainloop::PagedKV) {
          stride_k_cache = cutlass::make_cute_packed_stride(StrideK{}, shape_K_cache);
          stride_v_cache = cutlass::make_cute_packed_stride(StrideV{}, shape_V_cache);
        }
      }

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, stride_q));
      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, stride_k));
      Tensor V = make_tensor(make_gmem_ptr(dcV), make_layout(shape_V, stride_v));
      Tensor K_cache = make_tensor(make_gmem_ptr(dcK_cache), make_layout(shape_K_cache, stride_k_cache));
      Tensor V_cache = make_tensor(make_gmem_ptr(dcV_cache), make_layout(shape_V_cache, stride_v_cache));
      Tensor O = make_tensor(make_gmem_ptr(ptrO), make_layout(shape_O, stride_o));

      // O accumulator types
      FragA tArA;
      FragARow tA_max;
      FragSPartialRow tA_sum;

      // Main loop
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      static constexpr int QK_BLK_M = decltype(get<0>(TileShapeQK{}))::value;
      constexpr bool kGqaFusion = TileScheduler::kGqaFusion;
      constexpr bool kDisablePrefetchV = TileScheduler::kDisablePrefetchV;
      constexpr bool kDisableKVPrefetch = TileScheduler::kDisableKVPrefetch;

      [[maybe_unused]] ScaleTensors scales = ScaleTensors::make_null();
      if constexpr (BlockScale) {
        int offset_scaleQ = 0, offset_scaleK = 0, offset_scaleV = 0;
        int offset_scaleK_cache = 0, offset_scaleV_cache = 0;
        if constexpr (is_var_len) {
          int scale_d = cute::ceil_div(s.head_size_qk, p.group_size);
          offset_scaleQ = s.num_heads_q  * scale_d * s.seq_len_qo.cumulative_length[idx_b];
          offset_scaleK = s.num_heads_kv * scale_d * s.seq_len_kv.cumulative_length[idx_b];
          offset_scaleV = s.num_heads_kv * s.seq_len_kv.cumulative_scale_length[idx_b];
          if (s.seq_len_kv_cache.cumulative_scale_length) {
            offset_scaleK_cache = s.num_heads_kv * scale_d * s.seq_len_kv_cache.cumulative_length[idx_b];
            offset_scaleV_cache = s.num_heads_kv * s.seq_len_kv_cache.cumulative_scale_length[idx_b];
          }
        }
        scales = make_blockscale_tensors<is_var_len>(
            p,
            fmha_scaleq_rows(kGqaFusion, int(seq_len_qo), head_group_q),
            kGqaFusion ? s.num_heads_kv : s.num_heads_q,
            int(seq_len_kv), batch_dim, offset_scaleQ, offset_scaleK, offset_scaleV,
            fmha_scale_cache_rows(kv_cache_rows), offset_scaleK_cache, offset_scaleV_cache);
      }

      if constexpr (kGqaFusion) {
        assert(get<0>(blk_qv) == 0 && "GQA fusion assumes seq_len_qo <= QK_BLK_M");
        const int head_kv = head;
        const int g = tile_scheduler.get_gqa_group_size();

        const int q_len = seq_len_qo;
        const int total_rows = g * q_len;
        const int num_q_blocks = (total_rows + int(QK_BLK_M) - 1) / int(QK_BLK_M);

        int fusion_seq_len          = seq_len;
        int fusion_k_blocks         = k_blocks;
        int fusion_full_tile_offset = full_tile_offset;
        int fusion_discard          = discard_seq_coord;
        int gqa_fusion_q_per_head   = 0;
        if constexpr (CollectiveMainloop::CausalMask) {
          gqa_fusion_q_per_head   = q_len;
          fusion_seq_len          = seq_len_kv_cache + seq_len_kv;
          fusion_full_tile_offset = seq_len_kv - cute::min(q_len, seq_len_kv);
          fusion_discard          = 0;
          const int fusion_cache_k_blocks = CollectiveMainloop::PagedKV
              ? cute::ceil_div(seq_len_kv_cache, get<1>(TileShapeQK{})) : 0;
          fusion_k_blocks = fusion_cache_k_blocks
                          + cute::ceil_div(seq_len_kv, get<1>(TileShapeQK{}));
        }

        const int idx_b_l = idx_b;
        const auto q_group_off = head_kv * g * stride<2>(Q.layout());
        const auto o_group_off = head_kv * g * stride<2>(O.layout());
        for (int qb = 0; qb < num_q_blocks; ++qb) {
          const int row_start  = qb * int(QK_BLK_M);
          const int valid_rows = cute::min(int(QK_BLK_M), total_rows - row_start);

          auto make_gqa_view_q = [&, idx_b_l, row_start, valid_rows]() {
            auto offset = idx_b_l * stride<3>(Q.layout())
                        + q_group_off
                        + row_start * stride<0>(Q.layout());
            return make_tensor(
              Q.data() + offset,
              make_layout(make_shape(valid_rows, int(s.head_size_qk)),
                          make_stride(int(stride<0>(Q.layout())), stride<1>(Q.layout()))));
          };
          auto make_gqa_view_o = [&, idx_b_l, row_start, valid_rows]() {
            auto offset = idx_b_l * stride<3>(O.layout())
                        + o_group_off
                        + row_start * stride<0>(O.layout());
            return make_tensor(
              O.data() + offset,
              make_layout(make_shape(valid_rows, int(s.head_size_vo)),
                          make_stride(int(stride<0>(O.layout())), stride<1>(O.layout()))));
          };

          if (qb > 0) {
            sycl::group_barrier(get_work_group<3>());
          }

          FragA tArA;
          FragARow tA_max;
          FragSPartialRow tA_sum;

          mainloop.template operator()<kDisableKVPrefetch, kDisablePrefetchV>(
                  make_gqa_view_q(),
                  K(_,_,head_kv,idx_b),
                  V(_,_,head_kv,idx_b),
                  tArA, tA_max, tA_sum,
                  blk_qv, 0, fusion_k_blocks, fusion_k_blocks, fusion_k_blocks,
                  thr_id,
                  fusion_seq_len, seq_len_kv_cache, idx_b,
                  fusion_full_tile_offset, fusion_discard,
                  row_start, gqa_fusion_q_per_head,
                  K_cache(_,_,head,l_coord),
                  V_cache(_,_,head,l_coord),
                  p.scale_k, p.scale_v, p.scale_q,
                  scales.Q(_,_,head_kv,idx_b_l),
                  scales.K(_,_,head_kv,idx_b_l),
                  scales.V(_,_,head_kv,idx_b_l),
                  scales.P(_,_,head_kv,idx_b_l),
                  scales.K_cache(_,_,head_kv,idx_b_l),
                  scales.V_cache(_,_,head_kv,idx_b_l));

          if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
            sycl::group_barrier(get_work_group<3>());
          }

          epilogue(make_gqa_view_o(),
                  tArA, tA_max, tA_sum,
                  blk_qv, thr_id, p.scale_v);
        }
        return;
      }

      mainloop.template operator()<false, kDisablePrefetchV>(Q(_,_,head_q,l_coord),
               K(_,_,head,l_coord),
               V(_,_,head,l_coord),
               tArA, tA_max, tA_sum,
               blk_qv, 0, k_blocks, k_blocks, k_blocks_prefetch,
               thr_id, seq_len, seq_len_kv_cache, idx_b,
               full_tile_offset, discard_seq_coord,
               0,0,
               K_cache(_,_,head,l_coord),
               V_cache(_,_,head,l_coord),
               p.scale_k, p.scale_v, p.scale_q,
               scales.Q(_,_,head_q,l_coord),
               scales.K(_,_,head,l_coord),
               scales.V(_,_,head,l_coord),
               scales.P(_,_,head,l_coord),
               scales.K_cache(_,_,head,l_coord),
               scales.V_cache(_,_,head,l_coord));

      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Epilogue
      epilogue(O(_,_,head_q,l_coord),
               tArA, tA_max, tA_sum,
               blk_qv, thr_id, p.scale_v);
    }
  }
};

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_,
          class CollectiveEpilogueSplit_, class TileScheduler_>
class XeFMHAFwdDynamicSplitKernel {

public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;

  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;
  using ElementScale = typename CollectiveMainloop::TensorScaleQ::element_type;

  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));
  using StrideScaleQ = decltype(stride(typename CollectiveMainloop::TensorScaleQ{}));
  using StrideScaleK = decltype(stride(typename CollectiveMainloop::TensorScaleK{}));
  using StrideScaleV = decltype(stride(typename CollectiveMainloop::TensorScaleV{}));

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using SingleFragA = typename CollectiveMainloop::SingleFragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using FragSPartialRow = typename CollectiveMainloop::FragSPartialRow;
  // element dtype for MmaPV results
  using ElementA = typename CollectiveMainloop::ElementA;

  // Tile scheduler derived types
  static_assert(is_same_v<TileScheduler_, XeFHMAIndividualPersistentTileScheduler>);
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using CollectiveEpilogueSplit = CollectiveEpilogueSplit_;
  using SplitEpilogueArguments = typename CollectiveEpilogueSplit::Arguments;
  using SplitEpilogueParams = typename CollectiveEpilogueSplit::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));
  using ElementPartialO = typename CollectiveEpilogueSplit::TensorO::element_type;
  using StridePartialO = decltype(stride(typename CollectiveEpilogueSplit::TensorO{}));
  using ElementLSE = typename CollectiveEpilogueSplit::ElementLSE;
  static_assert(is_same_v<TileShapeO, typename CollectiveEpilogueSplit::TileShapeO>,
                "Final and split epilogues require the same output tile shape");
  static_assert(is_same_v<StrideO, StridePartialO>,
                "Final and split outputs require the same stride type");
  static_assert(!is_void_v<ElementLSE>,
                "XeFMHAFwdDynamicSplitKernel requires a split-capable epilogue");

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  using SplitEpilogueSharedStorage = typename CollectiveEpilogueSplit::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
    SplitEpilogueSharedStorage split_epilogue;
  };

  static constexpr bool BlockScale = CollectiveMainloop::BlockScale;
  using ScaleTensors = FMHABlockScaleTensors<typename CollectiveMainloop::TensorScaleQ,
                                             typename CollectiveMainloop::TensorScaleK,
                                             typename CollectiveMainloop::TensorScaleV>;

  static constexpr bool kPacksGqaQ = true;

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                     : sizeof(SharedStorage);
  static constexpr int max_num_partitions = SGPerWG::value * intel::sg_size;

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ *Q;
    StrideQ dQ;
    const ElementK *K;
    StrideK dK;
    const ElementV *V;
    StrideV dV;
    ElementO *O;
    StrideO dO;
    const ElementK *K_cache = nullptr;
    StrideK dK_cache{};
    const ElementV *V_cache = nullptr;
    StrideV dV_cache{};
    float scale_k = 1.f;
    float scale_v = 1.f;
    float scale_q = 1.f;
    const ElementScale *scaleQ = nullptr;
    StrideScaleQ dScaleQ{};
    const ElementScale *scaleK = nullptr;
    StrideScaleK dScaleK{};
    const ElementScale *scaleV = nullptr;
    StrideScaleV dScaleV{};
    const ElementScale *scaleP = nullptr;
    StrideScaleV dScaleP{};
    int group_size = 32;
    const ElementScale *scaleK_cache = nullptr;
    StrideScaleK dScaleK_cache{};
    const ElementScale *scaleV_cache = nullptr;
    StrideScaleV dScaleV_cache{};
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    // Split-K saturation core count
    int saturation_cores_hint = 0;
    SplitEpilogueArguments split_epilogue{};
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    SplitEpilogueParams split_epilogue;
    TileSchedulerParams scheduler;
    ElementPartialO *partial_output_ptr = nullptr;
    ElementLSE *exp_sums_ptr = nullptr;
    ElementLSE *max_logits_ptr = nullptr;
    int num_partitions = 1;
  };

  //
  // Methods
  //

  static int get_local_k_blocks(ProblemShape const& shape) {
    constexpr int kv_tile_size = int(get<1>(TileShapeQK{}));
    int cache_k_blocks = CollectiveMainloop::PagedKV
        ? cute::ceil_div(int(shape.seq_len_kv_cache), kv_tile_size) : 0;
    return cache_k_blocks + cute::ceil_div(int(shape.seq_len_kv), kv_tile_size);
  }

  static bool has_split_partitions(TileSchedulerParams const& scheduler) {
    return scheduler.num_partitions > 1;
  }

  static size_t partial_output_elements(ProblemShape const& shape, int max_parts) {
    return size_t(shape.batch) * shape.num_heads_q * int(shape.seq_len_qo) * int(shape.head_size_vo) * max_parts;
  }

  static size_t stats_elements(ProblemShape const& shape, int max_parts) {
    return size_t(shape.batch) * shape.num_heads_q * int(shape.seq_len_qo) * max_parts;
  }

  static size_t align_up(size_t size, size_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
  }

  static Params to_underlying_arguments(Arguments const &args, void *workspace) {
    int local_k_blocks = get_local_k_blocks(args.kernel.shape);
    auto scheduler = TileScheduler::to_underlying_arguments(
      args.kernel.shape, args.hw_info, TileShapeO{},
      args.saturation_cores_hint, local_k_blocks, max_num_partitions);
    int num_partitions = scheduler.num_partitions;
    ElementPartialO *partial_output_ptr = nullptr;
    ElementLSE *exp_sums_ptr = nullptr;
    ElementLSE *max_logits_ptr = nullptr;
    if (workspace != nullptr) {
      auto workspace_ptr = reinterpret_cast<uint8_t *>(workspace);
      size_t partial_bytes = partial_output_elements(args.kernel.shape, num_partitions) * sizeof(ElementPartialO);
      size_t stats_offset = align_up(partial_bytes, alignof(ElementLSE));
      size_t stats_bytes = stats_elements(args.kernel.shape, num_partitions) * sizeof(ElementLSE);
      partial_output_ptr = reinterpret_cast<ElementPartialO *>(workspace_ptr);
      exp_sums_ptr = reinterpret_cast<ElementLSE *>(workspace_ptr + stats_offset);
      max_logits_ptr = reinterpret_cast<ElementLSE *>(workspace_ptr + stats_offset + stats_bytes);
    }
    return {args.kernel,
            CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
            CollectiveEpilogueSplit::to_underlying_arguments(args.split_epilogue, workspace),
            scheduler,
            partial_output_ptr, exp_sums_ptr, max_logits_ptr, num_partitions
          };
  }

  static bool can_implement(Arguments const &args) {
    return CollectiveMainloop::can_implement(args.mainloop)
        && CollectiveEpilogue::can_implement(args.epilogue)
        && CollectiveEpilogueSplit::can_implement(args.split_epilogue);
  }

  static size_t get_workspace_size(Arguments const &args) {
    int local_k_blocks = get_local_k_blocks(args.kernel.shape);
    auto scheduler = TileScheduler::to_underlying_arguments(
      args.kernel.shape, args.hw_info, TileShapeO{},
      args.saturation_cores_hint, local_k_blocks, max_num_partitions);
    if (!has_split_partitions(scheduler)) {
      return 0;
    }

    int num_partitions = scheduler.num_partitions;
    size_t partial_bytes = partial_output_elements(args.kernel.shape, num_partitions) * sizeof(ElementPartialO);
    size_t stats_offset = align_up(partial_bytes, alignof(ElementLSE));
    return stats_offset + 2 * stats_elements(args.kernel.shape, num_partitions) * sizeof(ElementLSE);
  }

  static cutlass::Status initialize_workspace(Arguments const &args, void *workspace = nullptr,
                                              cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const &params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(SGPerWG::value * intel::sg_size, 1, 1); }

  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf)
  {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    auto &p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());
    constexpr int kv_tile_size = get<1>(TileShapeQK{});
    int cache_k_blocks = CollectiveMainloop::PagedKV
      ? cute::ceil_div(s.seq_len_kv_cache, kv_tile_size): 0;
    int local_k_blocks = cache_k_blocks + cute::ceil_div(s.seq_len_kv, kv_tile_size);
    int num_partitions = params.num_partitions;

    // Final output tensor and epilogue.
    auto shape_O = make_shape(s.seq_len_qo, s.head_size_vo, s.num_heads_q, s.batch);
    Tensor O = make_tensor(make_gmem_ptr(p.O), make_layout(shape_O, p.dO));    // (q,v,h,b)
    const int total_rows_o = head_group_q * s.seq_len_qo;

    auto do_epilogue = [&](int bh, FragA &out, FragARow &mx, auto &sm, auto const &bqv) {
      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }
      int hkv = bh % s.num_heads_kv;
      int idx_b_o = bh / s.num_heads_kv;
      const auto o_group_off = hkv * head_group_q * stride<2>(O.layout());
      auto o_view = make_tensor(
          O.data() + idx_b_o * stride<3>(O.layout()) + o_group_off,
          make_layout(make_shape(total_rows_o, int(s.head_size_vo)),
                      make_stride(int(stride<0>(O.layout())), stride<1>(O.layout()))));
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      epilogue(o_view, out, mx, sm, bqv, thr_id, p.scale_v);
    };

    auto shape_Oaccum = make_shape(total_rows_o, s.head_size_vo, s.num_heads_kv * num_partitions, s.batch);
    auto shape_stats = make_shape(total_rows_o, num_partitions, s.num_heads_kv, s.batch);
    auto stride_Oaccum = cutlass::make_cute_packed_stride(StridePartialO{}, shape_Oaccum);
    auto stride_stats = cutlass::make_cute_packed_stride(StrideO{}, shape_stats);
    Tensor Oaccum = make_tensor(
        make_gmem_ptr(params.partial_output_ptr),
        make_layout(shape_Oaccum, stride_Oaccum));
    Tensor exp_sums = make_tensor(
        make_gmem_ptr(params.exp_sums_ptr),
        make_layout(shape_stats, stride_stats));
    Tensor max_logits = make_tensor(
        make_gmem_ptr(params.max_logits_ptr),
        make_layout(shape_stats, stride_stats));

    auto store_partition = [&](int bh, int part, FragA &out, FragARow &mx,
                   auto &sm, auto const &bqv) {
      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<SplitEpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }
      int hkv = bh % s.num_heads_kv;
      int idx_b_o = bh / s.num_heads_kv;
      CollectiveEpilogueSplit split_epilogue{
          params.split_epilogue, shared_storage.split_epilogue};
      split_epilogue(
          Oaccum(_,_,part * s.num_heads_kv + hkv,idx_b_o),
          out, mx, sm, bqv, thr_id,
          exp_sums(_,_,hkv,idx_b_o),
          max_logits(_,_,hkv,idx_b_o),
          part, total_rows_o, p.scale_v);
    };

    [[maybe_unused]] ScaleTensors scales = ScaleTensors::make_null();
    if constexpr (BlockScale) {
      int scale_cache_rows = s.seq_len_kv_cache;
      if constexpr (CollectiveMainloop::PagedKV) {
        scale_cache_rows = paged_kv_cache_rows(s.seq_len_kv_cache, params.mainloop.page_size);
      }
      scales = make_blockscale_tensors(
          p,
          fmha_scaleq_rows(kPacksGqaQ, int(s.seq_len_qo), head_group_q),
          s.num_heads_kv, int(s.seq_len_kv), s.batch,
          0, 0, 0, fmha_scale_cache_rows(scale_cache_rows));
    }

    TileScheduler tile_scheduler{params.scheduler};
    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, batch_head_id, partition_id] = tile_scheduler.get_block_coord();
      auto blk_qv = make_coord(blk_q, blk_v);

      auto shape_Q = make_shape(s.seq_len_qo, s.head_size_qk, s.num_heads_q,  s.batch);
      auto shape_K = make_shape(s.seq_len_kv, s.head_size_qk, s.num_heads_kv, s.batch);
      auto shape_V = make_shape(s.head_size_vo, s.seq_len_kv, s.num_heads_kv, s.batch);

      auto dcQ = make_subbyte_aware_ptr<ElementQ>(p.Q);
      auto dcK = make_subbyte_aware_ptr<ElementK>(p.K);
      auto dcV = make_subbyte_aware_ptr<ElementV>(p.V);

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, p.dQ));    // (q,d,h,b)
      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, p.dK));    // (k,d,h,b)
      Tensor V = make_tensor(make_gmem_ptr(dcV), make_layout(shape_V, p.dV));    // (v,k,h,b)

      int kv_cache_rows = s.seq_len_kv_cache;
      if constexpr (CollectiveMainloop::PagedKV) {
        kv_cache_rows = paged_kv_cache_rows(s.seq_len_kv_cache, params.mainloop.page_size);
      }
      auto shape_K_cache = make_shape(kv_cache_rows, s.head_size_qk, s.num_heads_kv, s.batch);
      auto shape_V_cache = make_shape(s.head_size_vo, kv_cache_rows, s.num_heads_kv, s.batch);
      auto dcK_cache = make_subbyte_aware_ptr<ElementK>(p.K_cache);
      auto dcV_cache = make_subbyte_aware_ptr<ElementV>(p.V_cache);
      Tensor K_cache = make_tensor(make_gmem_ptr(dcK_cache), make_layout(shape_K_cache, p.dK_cache));
      Tensor V_cache = make_tensor(make_gmem_ptr(dcV_cache), make_layout(shape_V_cache, p.dV_cache));
      
      // O accumulator types
      FragA tArA;
      FragARow tA_max;
      FragSPartialRow tA_sum_partial;

      int blocks_per_partition = cute::ceil_div(local_k_blocks, num_partitions);
      int start_blk = partition_id * blocks_per_partition;
      int end_blk = cute::min(start_blk + blocks_per_partition, local_k_blocks);
      int head_kv = batch_head_id % s.num_heads_kv;
      int idx_b = batch_head_id / s.num_heads_kv;

      // Main loop
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      const int total_rows_q = head_group_q * s.seq_len_qo;
      const auto q_group_off = head_kv * head_group_q * stride<2>(Q.layout());
      auto make_gqa_view_q = [&]() {
        auto offset = idx_b * stride<3>(Q.layout()) + q_group_off;
        return make_tensor(
          Q.data() + offset,
          make_layout(make_shape(total_rows_q, int(s.head_size_qk)),
                      make_stride(int(stride<0>(Q.layout())), stride<1>(Q.layout()))));
      };

      int split_full_tile_offset = 0;
      int split_q_per_head = 0;
      if constexpr (CollectiveMainloop::CausalMask) {
        split_q_per_head = s.seq_len_qo;
        split_full_tile_offset = s.seq_len_kv - cute::min(int(s.seq_len_qo), int(s.seq_len_kv));
      }
      // keep the prefetch on when splitKV
      mainloop.template operator()<false>(
            make_gqa_view_q(),
            K(_,_,head_kv,idx_b),
            V(_,_,head_kv,idx_b),
            tArA, tA_max, tA_sum_partial,
            blk_qv, start_blk, end_blk, local_k_blocks, end_blk,
            thr_id, s.seq_len_kv_cache + s.seq_len_kv, s.seq_len_kv_cache, idx_b,
            split_full_tile_offset, 0, 0, split_q_per_head,
            K_cache(_,_,head_kv,idx_b),
            V_cache(_,_,head_kv,idx_b),
            p.scale_k, p.scale_v, p.scale_q,
            scales.Q(_,_,head_kv,idx_b),
            scales.K(_,_,head_kv,idx_b),
            scales.V(_,_,head_kv,idx_b),
            scales.P(_,_,head_kv,idx_b),
            scales.K_cache(_,_,head_kv,idx_b),
            scales.V_cache(_,_,head_kv,idx_b));

      if (num_partitions == 1) {
        do_epilogue(batch_head_id, tArA, tA_max, tA_sum_partial, blk_qv);
      } else {
        store_partition(batch_head_id, partition_id, tArA, tA_max, tA_sum_partial, blk_qv);
      }
    }
  }
};

} // namespace cutlass::fmha::kernel

namespace cutlass::reduction::kernel {

template <class FMHAKernel_>
class ReduceDynamicSplitK {
public:
  using FMHAParams = typename FMHAKernel_::Params;
  using ProblemShape = typename FMHAKernel_::ProblemShape;
  using ElementO = typename FMHAKernel_::ElementO;
  using StrideO = typename FMHAKernel_::StrideO;
  using ElementPartialO = typename FMHAKernel_::ElementPartialO;
  using StridePartialO = typename FMHAKernel_::StridePartialO;
  using ElementLSE = typename FMHAKernel_::ElementLSE;
  using SGPerWG = typename FMHAKernel_::SGPerWG;

  static constexpr int max_num_partitions = SGPerWG::value * intel::sg_size;

  static constexpr int reduction_wg_size = 8 * intel::sg_size;

  struct Params {
    ProblemShape shape;
    ElementO *O;
    StrideO dO;
    const ElementPartialO *partial_output_ptr;
    const ElementLSE *exp_sums_ptr;
    const ElementLSE *max_logits_ptr;
    int num_partitions;
  };

  struct SharedStorage {
    cutlass::Array<ElementLSE, max_num_partitions> max_logits;
    cutlass::Array<ElementLSE, max_num_partitions> exp_sums;
  };

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                     : sizeof(SharedStorage);

  static Params to_underlying_arguments(FMHAParams const &params) {
    return {
      params.kernel.shape,
      params.kernel.O,
      params.kernel.dO,
      params.partial_output_ptr,
      params.exp_sums_ptr,
      params.max_logits_ptr,
      params.num_partitions
    };
  }

  static bool requires_reduction(FMHAParams const &params) {
    return FMHAKernel_::has_split_partitions(params.scheduler);
  }

  static dim3 get_grid_shape(Params const &params) {
    return dim3(params.shape.seq_len_qo, params.shape.num_heads_q, params.shape.batch);
  }

  static dim3 get_block_shape() {
    return dim3(reduction_wg_size, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);
    auto const& s = params.shape;
    int thr_id = int(ThreadIdxX());
    int seq_idx = int(BlockIdxX());
    int head_q = int(BlockIdxY());
    int idx_b = int(BlockIdxZ());
    int num_partitions = params.num_partitions;

    auto shape_O = make_shape(s.seq_len_qo, s.head_size_vo, s.num_heads_q, s.batch);
    auto shape_Oaccum = make_shape(s.seq_len_qo, s.head_size_vo, s.num_heads_q * num_partitions, s.batch);
    auto shape_stats = make_shape(s.seq_len_qo, num_partitions, s.num_heads_q, s.batch);
    auto stride_Oaccum = cutlass::make_cute_packed_stride(StridePartialO{}, shape_Oaccum);
    auto stride_stats = cutlass::make_cute_packed_stride(StrideO{}, shape_stats);

    Tensor O = make_tensor(make_gmem_ptr(params.O), make_layout(shape_O, params.dO));
    Tensor Oaccum = make_tensor(
        make_gmem_ptr(const_cast<ElementPartialO *>(params.partial_output_ptr)),
        make_layout(shape_Oaccum, stride_Oaccum));
    Tensor exp_sums = make_tensor(
        make_gmem_ptr(const_cast<ElementLSE *>(params.exp_sums_ptr)),
        make_layout(shape_stats, stride_stats));
    Tensor max_logits = make_tensor(
        make_gmem_ptr(const_cast<ElementLSE *>(params.max_logits_ptr)),
        make_layout(shape_stats, stride_stats));

    constexpr int wg_size = reduction_wg_size;
    ElementLSE thread_max{cutlass::platform::numeric_limits<ElementLSE>::lowest()};
    if (thr_id < num_partitions) {
      ElementLSE cur_max = max_logits(seq_idx, thr_id, head_q, idx_b);
      thread_max = cur_max;
      shared_storage.max_logits[thr_id] = cur_max;
      shared_storage.exp_sums[thr_id] = exp_sums(seq_idx, thr_id, head_q, idx_b);
    }

    sycl::group_barrier(get_work_group<3>());
    ElementLSE global_max = reduce_over_group(get_work_group<1>(), thread_max, sycl::maximum<>());

    cutlass::Array<ElementLSE, 2> partial_buffer;
    for (int idx_v = thr_id; idx_v < s.head_size_vo; idx_v += wg_size) {
      ElementLSE output_accum = 0;
      ElementLSE global_exp_sum = 0;
      partial_buffer[0] = static_cast<ElementLSE>(Oaccum(seq_idx, idx_v, head_q, idx_b));

      #pragma unroll 2
      for (int part = 0; part < num_partitions; ++part) {
        if (part + 1 < num_partitions) {
          partial_buffer[(part + 1) & 1] = static_cast<ElementLSE>(
              Oaccum(seq_idx, idx_v, (part + 1) * s.num_heads_q + head_q, idx_b));
        }

        ElementLSE local_max = shared_storage.max_logits[part];
        ElementLSE local_exp_sum = shared_storage.exp_sums[part];
        ElementLSE rescale = sycl::native::exp2(local_max - global_max);
        ElementLSE weighted_sum = local_exp_sum * rescale;
        output_accum += partial_buffer[part & 1] * weighted_sum;
        global_exp_sum += weighted_sum;
      }

      O(seq_idx, idx_v, head_q, idx_b) = static_cast<ElementO>(output_accum / global_exp_sum);
    }
  }
};

} // namespace cutlass::reduction::kernel
