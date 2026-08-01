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
    const ElementScale *scaleQ = nullptr;
    StrideScaleQ dScaleQ{};
    const ElementScale *scaleK = nullptr;
    StrideScaleK dScaleK{};
    const ElementScale *scaleV = nullptr;
    StrideScaleV dScaleV{};
    float scale_k = 1.f;
    float scale_v = 1.f;
    float scale_q = 1.f;
    int group_size = 32;
    const ElementK *K_cache;
    StrideK dK_cache{};
    const ElementV *V_cache;
    StrideV dV_cache{};
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
      return cutlass::fmha::collective::apply_variable_length(Shape<VariableLength, VariableLength, VariableLength>{problem_shape.seq_len_qo, problem_shape.seq_len_kv, problem_shape.seq_len_kv_cache}, batch);
    } else {
      return Shape<int, int, int>{problem_shape.seq_len_qo, problem_shape.seq_len_kv, problem_shape.seq_len_kv_cache};
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
      auto cS = make_identity_tensor(take<0,2>(TiledMMAQK{}.tile_mnk()));
      auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
      auto q_offset_wi = get<0>(tScS(0));
      auto q_offset_sg = group_broadcast(
          sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);
      constexpr bool kIndependentSubgroups =
          is_empty_v<MainloopSharedStorage> && is_empty_v<EpilogueSharedStorage>
          && !TileScheduler::kGqaFusion;
      if constexpr (kIndependentSubgroups) {
        if (blk_q * get<0>(TileShapeQK{}) + q_offset_sg >= seq_len_qo) continue;
      }

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
      if constexpr (CollectiveMainloop::CausalMask || CollectiveMainloop::CachedKV) {
        const int kblocks_cache = CollectiveMainloop::CachedKV ? cute::ceil_div(seq_len_kv_cache, get<1>(TileShapeQK{})) : 0;
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

      auto dcQ = const_cast<ElementQ*>(p.Q + offset_q);
      auto dcK = const_cast<ElementK*>(p.K + offset_k);
      auto dcV = const_cast<ElementV*>(p.V + offset_v);
      auto dcK_cache = const_cast<ElementK*>(p.K_cache + offset_k_cache);
      auto dcV_cache = const_cast<ElementV*>(p.V_cache + offset_v_cache);
      auto ptrO = p.O + offset_o;

      StrideQ stride_q = p.dQ;
      StrideK stride_k = p.dK;
      StrideV stride_v = p.dV;
      StrideO stride_o = p.dO;
      StrideK stride_k_cache = p.dK_cache;
      StrideV stride_v_cache = p.dV_cache;
      if constexpr (is_var_len) {
        stride_q = cutlass::make_cute_packed_stride(StrideQ{}, shape_Q);
        stride_k = cutlass::make_cute_packed_stride(StrideK{}, shape_K);
        stride_v = cutlass::make_cute_packed_stride(StrideV{}, shape_V);
        stride_o = cutlass::make_cute_packed_stride(StrideO{}, shape_O);
        stride_k_cache = cutlass::make_cute_packed_stride(StrideK{}, shape_K_cache);
        stride_v_cache = cutlass::make_cute_packed_stride(StrideV{}, shape_V_cache);
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
          const int fusion_cache_k_blocks = CollectiveMainloop::CachedKV
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

          mainloop.template operator()<true, kDisablePrefetchV>(
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

// Compute the maximum number of partitions (WGs) that can contribute to a single
// batch_head. This depends on sm_count and num_batch_heads at runtime.
// Free function (template-independent) so it can be unit-tested without
// instantiating the full kernel type.
inline int compute_max_num_partitions(int num_partition_wgs, int num_batch_heads) {
  return cute::ceil_div(num_partition_wgs, cute::max(1, num_batch_heads)) + 1;
}

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
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

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr bool BlockScale = CollectiveMainloop::BlockScale;
  using ScaleTensors = FMHABlockScaleTensors<typename CollectiveMainloop::TensorScaleQ,
                                             typename CollectiveMainloop::TensorScaleK,
                                             typename CollectiveMainloop::TensorScaleV>;

  static constexpr bool kPacksGqaQ = true;

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                     : sizeof(SharedStorage);

  // Important: make sure multiple of 16 element for each copy
  // this is for storing partial results from different KV partitions
  static constexpr int num_elem_per_thread = (size(FragA{}.shape()) + 2 * size(FragARow{}.shape()) + 15) / 16 * 16;

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
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
    // workspace for storing partial results of different KV partitions
    ElementA *partial_results_ptr = nullptr;
    // max partitions per batch_head (from saturation_cores_hint)
    int max_num_partitions = 0;
    // 0 = compute phase (store partials), 1 = reduce phase (merge + epilogue)
    int phase = 0;
  };

  static constexpr bool kReducePhaseSeparate = true;

  //
  // Methods
  //

  static int num_partition_wgs(int sm_count) {
    return sm_count / 2;
  }

  static Params to_underlying_arguments(Arguments const &args, void *workspace) {
    int num_batch_heads = args.kernel.shape.batch * args.kernel.shape.num_heads_kv;
    int max_parts = compute_max_num_partitions(
        fmha_split_saturation_cores(args.saturation_cores_hint), num_batch_heads);
    ElementA *partial_results_ptr = reinterpret_cast<ElementA *>(workspace);
    return {args.kernel,
            CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
            TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{}, args.saturation_cores_hint),
            partial_results_ptr, max_parts, /*phase=*/0
          };
  }

  static bool can_implement(Arguments const &args) {
    return CollectiveMainloop::can_implement(args.mainloop)
        && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const &args) {
    int num_batch_heads = args.kernel.shape.batch * args.kernel.shape.num_heads_kv;
    int max_parts = compute_max_num_partitions(
        fmha_split_saturation_cores(args.saturation_cores_hint), num_batch_heads);
    const int wg_size = SGPerWG::value * intel::sg_size;
    // one partial blob (attn out + per-row max/sum) per (batch_head, partition)
    return (max_parts * num_batch_heads) * wg_size * num_elem_per_thread * sizeof(ElementA);
  }

  static cutlass::Status initialize_workspace(Arguments const &args, void *workspace = nullptr,
                                              cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const &params) {
    if (params.phase == 1) {
      int num_batch_heads = params.kernel.shape.batch * params.kernel.shape.num_heads_kv;
      return dim3(1, 1, num_batch_heads);
    }
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(SGPerWG::value * intel::sg_size, 1, 1); }

  CUTLASS_DEVICE
  int get_partition_id(const int cur_wg_id, const int batch_head_id, const int num_blocks_per_wg, const int local_k_blocks) {
    int partition_id = 0;
    if (batch_head_id == 0) {
      return cur_wg_id;
    }
    int start_wg_id = batch_head_id * local_k_blocks / num_blocks_per_wg;
    partition_id = cur_wg_id - start_wg_id;
    return partition_id;
  }

  CUTLASS_DEVICE
  int get_num_partitions(const int batch_head_id, const int num_blocks_per_wg, const int local_k_blocks) {
    int num_partitions = 1;
    int start_wg_id = batch_head_id * local_k_blocks / num_blocks_per_wg;
    int end_wg_id = (batch_head_id + 1) * local_k_blocks / num_blocks_per_wg;
    num_partitions = end_wg_id - start_wg_id + 1;
    // end_wg_id is the starting wg id of next batch head id
    if (((batch_head_id + 1) * local_k_blocks) % num_blocks_per_wg == 0) {
      num_partitions -= 1;
    }
    return num_partitions;
  }

  template <class Params, class FragA, class FragARow>
  CUTLASS_DEVICE
  void reduce_split2(const Params &params, FragA &out1, FragARow& max_val1, FragARow& exp_sum_val1, FragA &out2, FragARow& max_val2, FragARow& exp_sum_val2) {
    // global max value
    FragARow max_prev1 = max_val1;
    FragARow max_prev2 = max_val2;

    auto scale = params.mainloop.scale;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < max_val1.size(); i++) {
      max_val1(i) = sycl::max(max_val1(i), max_val2(i));
    }

    FragARow rescale1, rescale2;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < max_val1.size(); i++) {
      rescale1(i) = sycl::native::exp2(max_prev1(i) - max_val1(i));
      rescale2(i) = sycl::native::exp2(max_prev2(i) - max_val1(i));
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < exp_sum_val1.size(); i++) {
      exp_sum_val1(i) = exp_sum_val1(i) * rescale1(i) + exp_sum_val2(i) * rescale2(i);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < out1.size(); i++)
      out1(i) = out1(i) * broadcast<0>(rescale1, out1, i) + out2(i) * broadcast<0>(rescale2, out2, i);
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
    int sg_id = thr_id / intel::sg_size;
    int tid_in_sg = thr_id % intel::sg_size;
    int num_batch_heads = s.batch * s.num_heads_kv;

    constexpr int kv_tile_size = get<1>(TileShapeQK{});
    int cache_k_blocks = CollectiveMainloop::CachedKV
      ? cute::ceil_div(s.seq_len_kv_cache, kv_tile_size): 0;
    int local_k_blocks = cache_k_blocks + cute::ceil_div(s.seq_len_kv, kv_tile_size);
    // total number of blocks need to be processed across all wgs
    int total_k_blocks = local_k_blocks * num_batch_heads;

    // Output tensor + epilogue are shared by both phases.
    auto shape_O = make_shape(s.seq_len_qo, s.head_size_vo, s.num_heads_q, s.batch);
    Tensor O = make_tensor(make_gmem_ptr(p.O), make_layout(shape_O, p.dO));    // (q,v,h,b)
    CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};

    auto do_epilogue = [&](int bh, FragA &out, FragARow &mx, FragARow &sm, auto const &bqv) {
      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }
      int hkv = bh % s.num_heads_kv;
      int idx_b_o = bh / s.num_heads_kv;
      const int total_rows_o = head_group_q * s.seq_len_qo;
      const auto o_group_off = hkv * head_group_q * stride<2>(O.layout());
      auto o_view = make_tensor(
          O.data() + idx_b_o * stride<3>(O.layout()) + o_group_off,
          make_layout(make_shape(total_rows_o, int(s.head_size_vo)),
                      make_stride(int(stride<0>(O.layout())), stride<1>(O.layout()))));
      epilogue.template operator()<true>(o_view, out, mx, sm, bqv, thr_id, p.scale_v);
    };

    auto load_partition = [&](int bh, int part, FragA &out, FragARow &mx, FragARow &sm) {
      int offset = bh * params.max_num_partitions * SGPerWG::value * intel::sg_size * num_elem_per_thread
                 + part * SGPerWG::value * intel::sg_size * num_elem_per_thread
                 + sg_id * intel::sg_size * num_elem_per_thread
                 + tid_in_sg * num_elem_per_thread;
      Tensor tPartial = make_tensor(params.partial_results_ptr + offset, make_shape(Int<num_elem_per_thread>{}));
      Tensor merged_res = make_tensor<ElementA>(Int<num_elem_per_thread>{});
      copy(tPartial, merged_res);
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < size(FragA{}.shape()); ++e) {
        out(e) = merged_res(e);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < size(FragARow{}.shape()); ++e) {
        mx(e) = merged_res(2 * e + size(FragA{}.shape()));
        sm(e) = merged_res(2 * e + 1 + size(FragA{}.shape()));
      }
    };

    // Reduce phase: one work-group per batch_head merges all its partitions
    if (params.phase == 1) {
      int bh = int(BlockIdxZ());
      if (bh >= num_batch_heads) return;
      // Partition count is derived from the compute grid (params.scheduler.grid.z).
      int compute_grid_z = int(params.scheduler.grid.z);
      int num_blocks_per_wg = cute::ceil_div(total_k_blocks, compute_grid_z);
      int num_partitions = get_num_partitions(bh, num_blocks_per_wg, local_k_blocks);
      if (num_partitions <= 1) return;  // already written directly by the compute phase

      FragA acc;
      FragARow accMax, accSum;
      load_partition(bh, 0, acc, accMax, accSum);
      CUTLASS_PRAGMA_NO_UNROLL
      for (int i = 1; i < num_partitions; ++i) {
        FragA pOut;
        FragARow pMax, pSum;
        load_partition(bh, i, pOut, pMax, pSum);
        reduce_split2(params, acc, accMax, accSum, pOut, pMax, pSum);
      }
      do_epilogue(bh, acc, accMax, accSum, make_coord(0, 0));
      return;
    }

    // Compute phase: each partition computes its KV slice and either writes O
    int wg_id = int(BlockIdxZ());
    int num_blocks_per_wg = cute::ceil_div(total_k_blocks, GridDimZ());

    auto store_partition = [&](int bh, int part, FragA const &out, FragARow const &mx, FragARow const &sm) {
      int offset = bh * params.max_num_partitions * SGPerWG::value * intel::sg_size * num_elem_per_thread
                 + part * SGPerWG::value * intel::sg_size * num_elem_per_thread
                 + sg_id * intel::sg_size * num_elem_per_thread
                 + tid_in_sg * num_elem_per_thread;
      Tensor tPartial = make_tensor(params.partial_results_ptr + offset, make_shape(Int<num_elem_per_thread>{}));
      Tensor merged_res = make_tensor<ElementA>(Int<num_elem_per_thread>{});
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < size(FragA{}.shape()); ++e) {
        merged_res(e) = out(e);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < size(FragARow{}.shape()); ++e) {
        merged_res(2 * e + size(FragA{}.shape())) = mx(e);
        merged_res(2 * e + 1 + size(FragA{}.shape())) = sm(e);
      }
      copy(merged_res, tPartial);
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

    TileScheduler tile_scheduler{params.scheduler, get<1>(TileShapeQK{}), local_k_blocks, num_batch_heads};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, start_batch_head_id] = tile_scheduler.get_block_coord(); // (Q,V, batch_head_idx)
      auto blk_qv = make_coord(blk_q, blk_v);

      auto shape_Q = make_shape(s.seq_len_qo, s.head_size_qk, s.num_heads_q,  s.batch);
      auto shape_K = make_shape(s.seq_len_kv, s.head_size_qk, s.num_heads_kv, s.batch);
      auto shape_V = make_shape(s.head_size_vo, s.seq_len_kv, s.num_heads_kv, s.batch);

      auto dcQ = const_cast<ElementQ*>(p.Q);  // de-const these for uniformity
      auto dcK = const_cast<ElementK*>(p.K);
      auto dcV = const_cast<ElementV*>(p.V);

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, p.dQ));    // (q,d,h,b)
      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, p.dK));    // (k,d,h,b)
      Tensor V = make_tensor(make_gmem_ptr(dcV), make_layout(shape_V, p.dV));    // (v,k,h,b)

      int kv_cache_rows = s.seq_len_kv_cache;
      if constexpr (CollectiveMainloop::PagedKV) {
        kv_cache_rows = paged_kv_cache_rows(s.seq_len_kv_cache, params.mainloop.page_size);
      }
      auto shape_K_cache = make_shape(kv_cache_rows, s.head_size_qk, s.num_heads_kv, s.batch);
      auto shape_V_cache = make_shape(s.head_size_vo, kv_cache_rows, s.num_heads_kv, s.batch);
      auto dcK_cache = const_cast<ElementK*>(p.K_cache);
      auto dcV_cache = const_cast<ElementV*>(p.V_cache);
      Tensor K_cache = make_tensor(make_gmem_ptr(dcK_cache), make_layout(shape_K_cache, p.dK_cache));
      Tensor V_cache = make_tensor(make_gmem_ptr(dcV_cache), make_layout(shape_V_cache, p.dV_cache));
      
      // O accumulator types
      FragA tArA;
      FragARow tA_max, tA_sum;
      FragSPartialRow tA_sum_partial;

      // compute num computed blocks for start batch head id
      int num_computed_blocks = wg_id * num_blocks_per_wg - start_batch_head_id * local_k_blocks;
      int start_blk, end_blk, idx_b, head_kv;

      // Main loop
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);

      // compute blocks budget remained for each wg
      int block_budget_remained = num_blocks_per_wg;
      int batch_head_id = start_batch_head_id;
      bool is_update_batch_head_id = false;
      // Skip excess WGs whose start_batch_head_id is already out of range.
      // This happens when total_k_blocks < GridDimZ (more WGs than work).
      if (batch_head_id >= num_batch_heads) {
        block_budget_remained = 0;
      }
      while (block_budget_remained > 0) {
        int num_new_blocks = local_k_blocks - num_computed_blocks;
        if (num_new_blocks <= block_budget_remained) {
          // finished current batch head id
          start_blk = num_computed_blocks;
          end_blk = start_blk + num_new_blocks;

          // update states
          num_computed_blocks = 0;
          block_budget_remained -= num_new_blocks;
          is_update_batch_head_id = true;
        } else {
          // budget cannot afford finishing current batch head id
          start_blk = num_computed_blocks;
          end_blk = start_blk + block_budget_remained;

          block_budget_remained = 0;
          is_update_batch_head_id = false;
        }

        head_kv = batch_head_id % s.num_heads_kv;
        idx_b = batch_head_id / s.num_heads_kv;
        const int total_rows_q  = head_group_q * s.seq_len_qo;
        const auto q_group_off  = head_kv * head_group_q * stride<2>(Q.layout());
        auto make_gqa_view_q = [&]() {
          auto offset = idx_b * stride<3>(Q.layout()) + q_group_off;
          return make_tensor(
            Q.data() + offset,
            make_layout(make_shape(total_rows_q, int(s.head_size_qk)),
                        make_stride(int(stride<0>(Q.layout())), stride<1>(Q.layout()))));
        };

        int split_full_tile_offset = 0;
        int split_q_per_head       = 0;
        if constexpr (CollectiveMainloop::CausalMask) {
          split_q_per_head       = s.seq_len_qo;
          split_full_tile_offset = s.seq_len_kv - cute::min(int(s.seq_len_qo), int(s.seq_len_kv));
        }

        mainloop.template operator()<true>(
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

        tA_sum = reduce<0, cute::ReduceMode::Horizontal>(tA_sum_partial, sycl::plus<void>{});

        // partition id of start batch head id in current wg
        int partition_id = get_partition_id(wg_id, batch_head_id, num_blocks_per_wg, local_k_blocks);
        int num_partitions = get_num_partitions(batch_head_id, num_blocks_per_wg, local_k_blocks);

        if (num_partitions == 1) {
          do_epilogue(batch_head_id, tArA, tA_max, tA_sum, blk_qv);
        } else {
          // Store this partition's partial; the reduce phase will merge them.
          store_partition(batch_head_id, partition_id, tArA, tA_max, tA_sum);
        }

        if (is_update_batch_head_id) {
          batch_head_id += 1;
          if (batch_head_id >= num_batch_heads) {
            break;
          }
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// Split-KV (flash-decoding) compute kernel.
//
// Splits the KV sequence into `num_kv_splits` partitions, each processed by an
// independent work-group, to obtain high parallelism for long-KV decode. GQA
// query heads sharing a KV head are packed into the Q tile dimension. Each WG
// stores its locally-normalized partial output (Oaccum) plus per-split softmax
// statistics (exp sum + max logit); a subsequent ReduceSplitK kernel merges the
// partitions with a numerically-stable log-sum-exp rescale.
//
// Assumes decode (seq_len_qo == 1) and GQA group size <= 8 (DPAS max repeat).
///////////////////////////////////////////////////////////////////////////////
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeFMHAFwdSplitKVKernel {

public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len = cutlass::fmha::collective::is_variable_length_v<typename ProblemShape::SeqLenType>;
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

  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using FragSPartialRow = typename CollectiveMainloop::FragSPartialRow;

  // Tile scheduler derived types
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  // dtype for storing intermediate exp sums and max logits
  using ElementLSE = typename CollectiveEpilogue::ElementLSE;

  static constexpr bool BlockScale = CollectiveMainloop::BlockScale;
  static_assert(!BlockScale, "XeFMHAFwdSplitKVKernel does not support BlockScale");

  static constexpr bool kPacksGqaQ = true;

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                     : sizeof(SharedStorage);

  // Maximum number of KV splits is bounded by the work-group size (one thread
  // reduces one split in the reduction kernel).
  static constexpr int max_num_kv_splits = SGPerWG::value * intel::sg_size;
  static constexpr int dpas_max_repeat_count = 8;
  static constexpr bool kReducePhaseSeparate = false;

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ *Q;
    StrideQ dQ;
    const ElementK *K;
    StrideK dK;
    const ElementV *V;
    StrideV dV;
    ElementO *O;                 // partial output (Oaccum)
    StrideO dO;
    const ElementK *K_cache = nullptr;
    StrideK dK_cache{};
    const ElementV *V_cache = nullptr;
    StrideV dV_cache{};
    ElementLSE *exp_sums = nullptr;
    StrideO dExp_sums{};
    ElementLSE *max_logits = nullptr;
    StrideO dMax_logits{};
    float scale_k = 1.f;
    float scale_v = 1.f;
    float scale_q = 1.f;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = -1; // no split by default
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
            TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const &args) {
    // Query positions are packed alongside GQA heads into the Q tile dimension,
    // so seq_len_qo > 1 is supported for the non-var-len, non-causal path.
    //  - Causal + seq_len_qo > 1 is NOT supported: the mainloop is invoked
    //    without GQA fusion, so a packed row index no longer maps to a real
    //    query position and the causal mask would use the wrong coordinate.
    //  - Var-len forces seq_len_qo = 1 internally, so reject longer var-len
    //    sequences rather than silently produce wrong results.
    if constexpr (is_var_len) {
      if (args.kernel.shape.seq_len_qo.max_length != 1) {
        return false;
      }
    } else {
      if (CollectiveMainloop::CausalMask && args.kernel.shape.seq_len_qo != 1) {
        return false;
      }
    }
    if (args.num_kv_splits > max_num_kv_splits) {
      return false;
    }
    if (args.num_kv_splits == 0) {
      // 0 is invalid; -1 means "auto" (clamped to 1 downstream).
      return false;
    }
    // GQA group size limited to DPAS max repeat count
    if (args.kernel.shape.num_heads_q / args.kernel.shape.num_heads_kv > dpas_max_repeat_count) {
      return false;
    }
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
      return cutlass::fmha::collective::apply_variable_length(Shape<VariableLength, VariableLength, VariableLength>{problem_shape.seq_len_qo, problem_shape.seq_len_kv, problem_shape.seq_len_kv_cache}, batch);
    } else {
      return Shape<int, int, int>{problem_shape.seq_len_qo, problem_shape.seq_len_kv, problem_shape.seq_len_kv_cache};
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
    int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));

    auto cS = make_identity_tensor(take<0,2>(TiledMMAQK{}.tile_mnk()));
    auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
    auto q_offset_wi = get<0>(tScS(0));
    auto q_offset_sg = group_broadcast(sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);

    TileScheduler tile_scheduler{params.scheduler};
    auto num_kv_splits = params.scheduler.num_kv_splits_;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head, idx_b, idx_kv_split] = tile_scheduler.get_block_coord(); // (Q,V,h_kv,b,kv_split)
      auto blk_qv = make_coord(blk_q, blk_v);

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv, seq_len_kv_cache] = sequence_length_shape;

      auto seq_len_qo_packed = seq_len_qo * head_group_q;

      if (blk_q * get<0>(TileShapeQK{}) >= seq_len_qo_packed) continue;

      auto offset = cute::min(seq_len_qo, seq_len_kv);
      auto discard_seq_coord = seq_len_qo - offset;
      auto full_tile_offset = seq_len_kv - offset;
      int seq_coord = cute::min(seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));

      if (CollectiveMainloop::CausalMask && seq_coord < discard_seq_coord) continue;
      const int seq_len_new = CollectiveMainloop::CausalMask ? full_tile_offset + cute::min(seq_len_kv, seq_coord - discard_seq_coord) + q_sg_tile : seq_len_kv;
      const int seq_len = seq_len_new + seq_len_kv_cache;

      const int k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));

      int kv_cache_rows = seq_len_kv_cache;
      if constexpr (CollectiveMainloop::PagedKV) {
        kv_cache_rows = paged_kv_cache_rows(seq_len_kv_cache, params.mainloop.page_size);
      }

      int offset_q = 0, offset_k = 0, offset_v = 0, offset_o = 0;
      int offset_k_cache = 0, offset_v_cache = 0;
      int offset_exp_sums = 0, offset_max_logits = 0;
      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;
        auto kv_cumulative = s.seq_len_kv.cumulative_length;
        offset_q = s.num_heads_q * s.head_size_qk * qo_cumulative[idx_b];
        offset_k = s.num_heads_kv * s.head_size_qk * kv_cumulative[idx_b];
        offset_v = s.num_heads_kv * s.head_size_vo * kv_cumulative[idx_b];

        offset_o = s.num_heads_q * s.head_size_vo * qo_cumulative[idx_b] * num_kv_splits;
        offset_exp_sums = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];
        offset_max_logits = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];

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

        // for gqa packing, seq_len_qo must be 1
        seq_len_qo = 1;
      }

      auto batch_dim = is_var_len ? 1 : s.batch;
      auto shape_Q = make_shape(seq_len_qo_packed, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_K = make_shape(seq_len_kv, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V = make_shape(s.head_size_vo, seq_len_kv, s.num_heads_kv, batch_dim);
      auto shape_O = make_shape(seq_len_qo_packed, s.head_size_vo, s.num_heads_kv * num_kv_splits, batch_dim);
      auto shape_exp_sums = make_shape(seq_len_qo_packed, num_kv_splits, s.num_heads_kv, batch_dim);
      auto shape_max_logits = make_shape(seq_len_qo_packed, num_kv_splits, s.num_heads_kv, batch_dim);

      auto shape_K_cache = make_shape(kv_cache_rows, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V_cache = make_shape(s.head_size_vo, kv_cache_rows, s.num_heads_kv, batch_dim);

      int num_blocks_per_split = cute::ceil_div(k_blocks, num_kv_splits);
      int kv_split_offset = idx_kv_split * num_blocks_per_split;
      int num_effective_kv_blocks = cute::min(k_blocks - kv_split_offset, num_blocks_per_split);

      if (num_effective_kv_blocks <= 0) {
        // nothing to compute for this split
        continue;
      }

      auto dcQ = const_cast<ElementQ*>(p.Q + offset_q);
      auto dcK = const_cast<ElementK*>(p.K + offset_k);
      auto dcV = const_cast<ElementV*>(p.V + offset_v);
      auto dcK_cache = const_cast<ElementK*>(p.K_cache + offset_k_cache);
      auto dcV_cache = const_cast<ElementV*>(p.V_cache + offset_v_cache);
      auto ptrO = p.O + offset_o;
      auto ptrExp_sums = p.exp_sums + offset_exp_sums;
      auto ptrMax_logits = p.max_logits + offset_max_logits;

      auto stride_q = cutlass::make_cute_packed_stride(StrideQ{}, shape_Q);
      auto stride_k = is_var_len ? cutlass::make_cute_packed_stride(StrideK{}, shape_K) : p.dK;
      auto stride_v = is_var_len ? cutlass::make_cute_packed_stride(StrideV{}, shape_V) : p.dV;
      auto stride_k_cache = is_var_len ? cutlass::make_cute_packed_stride(StrideK{}, shape_K_cache) : p.dK_cache;
      auto stride_v_cache = is_var_len ? cutlass::make_cute_packed_stride(StrideV{}, shape_V_cache) : p.dV_cache;
      auto stride_o = cutlass::make_cute_packed_stride(StrideO{}, shape_O);
      auto stride_exp_sums = cutlass::make_cute_packed_stride(StrideO{}, shape_exp_sums);
      auto stride_max_logits = cutlass::make_cute_packed_stride(StrideO{}, shape_max_logits);

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, stride_q));
      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, stride_k));
      Tensor V = make_tensor(make_gmem_ptr(dcV), make_layout(shape_V, stride_v));
      Tensor K_cache = make_tensor(make_gmem_ptr(dcK_cache), make_layout(shape_K_cache, stride_k_cache));
      Tensor V_cache = make_tensor(make_gmem_ptr(dcV_cache), make_layout(shape_V_cache, stride_v_cache));
      Tensor O = make_tensor(make_gmem_ptr(ptrO), make_layout(shape_O, stride_o));
      Tensor exp_sums = make_tensor(make_gmem_ptr(ptrExp_sums), make_layout(shape_exp_sums, stride_exp_sums));
      Tensor max_logits = make_tensor(make_gmem_ptr(ptrMax_logits), make_layout(shape_max_logits, stride_max_logits));

      // O accumulator types
      FragA tArA;
      FragARow tA_max;
      FragSPartialRow tA_sum;

      int l_coord = is_var_len ? 0 : idx_b;

      int start_blk = kv_split_offset;
      int end_blk = kv_split_offset + num_effective_kv_blocks;

      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);

      mainloop.template operator()<>(
              Q(_,_,head,l_coord),
              K(_,_,head,l_coord),
              V(_,_,head,l_coord),
              tArA, tA_max, tA_sum,
              blk_qv, start_blk, end_blk, k_blocks, end_blk,
              thr_id, seq_len, seq_len_kv_cache, idx_b,
              full_tile_offset, discard_seq_coord,
              0, 0,
              K_cache(_,_,head,l_coord),
              V_cache(_,_,head,l_coord),
              p.scale_k, p.scale_v, p.scale_q);

      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Epilogue: store locally-normalized partial output + per-split LSE stats.
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      epilogue(O(_,_,idx_kv_split * s.num_heads_kv + head,l_coord),
                tArA, tA_max, tA_sum,
                blk_qv, thr_id,
                exp_sums(_,_,head,l_coord),
                max_logits(_,_,head,l_coord),
                idx_kv_split, seq_len_qo_packed,
                p.scale_v);
    }
  }
};

} // namespace cutlass::fmha::kernel
