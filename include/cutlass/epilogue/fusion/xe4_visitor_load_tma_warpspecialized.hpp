/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \file
  \brief Visitor tree load operations for the xe4 TMA warp-specialized (ws) epilogue
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/epilogue/fusion/xe_visitor.hpp"

#include "xe_visitor.hpp"

#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

using namespace cute;
using namespace detail;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Broadcast Load Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// Row vector broadcast
template<
  int Stages,
  class CtaTileShapeMNK,
  class ElementInput_,
  class ElementCompute = cute::remove_pointer_t<ElementInput_>,
  class StrideMNL_ = Stride<_0,_1,_0>,
  int Alignment = 128 / sizeof_bits_v<cute::remove_pointer_t<ElementInput_>>,
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
>
struct Xe4RowBroadcast {
  using StrideMNL = StrideMNL_;
  // Get base element input type.
  using ElementInput = cute::remove_pointer_t<ElementInput_>;
  // Check if input is an array of pointers.
  static constexpr bool IsArrayOfPointers = is_same_v<ElementInput*, ElementInput_>;
  using PtrRowType = cute::conditional_t<IsArrayOfPointers, ElementInput const* const*, ElementInput const*>;

  static constexpr size_t CtaTileN = size<1>(CtaTileShapeMNK{});
  using CopyOpG2S = xe4::ASYNC_LINEAR_LOAD<ElementInput, CtaTileN>;

  static_assert(Stages == 0, "Row broadcast doesn't support smem pipelining");

  static constexpr bool IsDynamicBroadcast = is_same_v<remove_cvref_t<decltype(get<1>(StrideMNL{}))>, bool>; // row vector or scalar broadcast
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))> || IsDynamicBroadcast); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_0,_1>{} || IsDynamicBroadcast);

  struct SharedStorage {
    array_aligned<ElementInput, CtaTileN> smem;
  };

  struct Arguments {
    PtrRowType ptr_row = nullptr;
    ElementInput null_default = ElementInput(0);
    StrideMNL dRow = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Xe4RowBroadcast() { }

  CUTLASS_HOST_DEVICE
  Xe4RowBroadcast(Params const& params, SharedStorage const& shared_storage)
      : params(params), is_zero_(false),
        smem(const_cast<ElementInput*>(shared_storage.smem.data())) {
    auto const& [stride_M, stride_N, stride_L] = params.dRow;
    // Nullptr default
    if (EnableNullptr && params.ptr_row == nullptr) {
      is_zero_ = params.null_default == ElementCompute(0);
    }
    // Dynamic non-batched scalar broadcast
    else if (IsDynamicBroadcast && stride_N == bool(0) && stride_L == repeat_like(stride_L, 0)) {
       if constexpr (!IsArrayOfPointers) {
         is_zero_ = params.ptr_row[0] == ElementInput(0);
       }
    }
  }

  Params params;
  bool is_zero_ = false;
  ElementInput *smem = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return true;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return is_zero_;
  }

  template <class GS_GTensor, class GS_STensor, class Tiled_G2S>
  struct ProducerLoadCallbacks : EmptyProducerLoadCallbacks {
    CUTLASS_DEVICE
    ProducerLoadCallbacks(GS_GTensor tGS_gRow_, GS_STensor tGS_sRow_, Tiled_G2S tiled_g2s_, Params const& params_)
      : tGS_gRow(tGS_gRow_), tGS_sRow(tGS_sRow_), tiled_G2S(tiled_g2s_), params(params_) {}

    GS_GTensor tGS_gRow;                                                         // (CPY,CPY_M,CPY_N)
    GS_STensor tGS_sRow;                                                         // (CPY,CPY_M,CPY_N)
    Tiled_G2S tiled_G2S;

    Params const& params;

    CUTLASS_DEVICE void
    step(uint64_t* full_mbarrier_ptr, int epi_m, int epi_n, int load_iteration, bool issue_tma_load) {
      if (issue_tma_load && epi_n == 0) {
        // Increment the expected transaction bytes of the current stage's mbarrier by the subtile's byte-size
        constexpr uint32_t copy_bytes = CtaTileN * sizeof_bits_v<ElementInput> / 8;
        cutlass::arch::ClusterTransactionBarrier::expect_transaction(full_mbarrier_ptr, copy_bytes);

        Tensor tGS_gRow_flt = filter_zeros(tGS_gRow);
        Tensor tGS_sRow_flt = filter_zeros(tGS_sRow);
        Tensor gRow = coalesce(tGS_gRow_flt);
        Tensor sRow = coalesce(tGS_sRow_flt);
        copy(tiled_G2S.with(full_mbarrier_ptr, copy_bytes), gRow, sRow);
      }
    }
  };

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    auto coord_shape = make_coord(m, n, l);

    auto layout_N = [&] () CUTLASS_LAMBDA_FUNC_INLINE {
      auto shape_N = get<1>(args.problem_shape_mnkl);
      if constexpr (IsDynamicBroadcast) {
        auto stride_N = repeat_like(shape_N, int(0));
        if (get<1>(params.dRow) == bool(1)) {
          stride_N = transform_leaf(compact_major<LayoutLeft>(shape_N),
            [] (auto const& stride) { return static_cast<int>(stride); }
          );
        }
        return make_layout(shape_N, stride_N);
      }
      else {
        return make_layout(shape_N);
      }
    }();

    auto layout_M = make_layout(M, repeat_like(M, _0{}));
    auto layout_L = make_layout(L, get<2>(params.dRow));

    ElementInput const* ptr_row = [&] () CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr(IsArrayOfPointers) {
        return params.ptr_row[l];
      } else {
        return params.ptr_row;
      }
    }();

    Tensor mRow = make_tensor(make_gmem_ptr(ptr_row), make_layout(layout_M,layout_N,layout_L));
    Tensor gRow = local_tile(mRow(_,_,l), take<0,2>(args.tile_shape_mnk), make_coord(m, n));        // (CTA_M, CTA_N)
    Tensor sRow = make_tensor(make_smem_ptr(smem),
      make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{})), make_shape(_0{}, _1{}));  // (CTA_M, CTA_N)

    //// G2S: Gmem to Smem
    auto tiled_g2s = make_tiled_copy(
      Copy_Atom<CopyOpG2S, ElementInput>{},
      Layout<Shape<_1,_1>, Stride<_0,_1>>{},
      Layout<Shape<_1,Int<CtaTileN>>, Stride<_0,_1>>{}
    );

    return ProducerLoadCallbacks(gRow, sRow, tiled_g2s, params);
  }

  template <class SR_STensor, class SR_RTensor, class SubGroup>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(SR_STensor tSR_sRow_, SR_RTensor tSR_rRow_, SubGroup sg_)
      : tSR_sRow(tSR_sRow_), tSR_rRow(tSR_rRow_), sg(sg_) {}

    SR_STensor tSR_sRow;                                                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    SR_RTensor tSR_rRow;                                                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

    SubGroup sg;
    int last_start_offset = -1;

    CUTLASS_DEVICE void
    begin_loop(int epi_m, int epi_n) {
      auto tSR_sRow_slice = tSR_sRow(_, _, _, epi_m, epi_n);
      auto start_offset = tSR_sRow_slice.layout()(make_coord(_0{}, _0{}, _0{}));

      if (start_offset != last_start_offset) {
        copy(tSR_sRow_slice, tSR_rRow);
        last_start_offset = start_offset;
      }
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<ElementCompute, FragmentSize> frg_row;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        frg_row[i] = sycl::select_from_group(sg, tSR_rRow(0), epi_v * FragmentSize + i);
      }

      return frg_row;
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    Tensor sRow = make_tensor(make_smem_ptr(smem),
        make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{})), make_shape(_0{}, _1{}));  // (CTA_M, CTA_N)

    auto tiled_s2r = make_tiled_copy(
      Copy_Atom<UniversalCopy<ElementInput>, ElementInput>{},
      make_layout(args.epi_tile, GenRowMajor{}),
      Layout<Shape<_1,_1>, Stride<_0,_1>>{}
    );

    //// S2R: Smem to Reg
    Tensor tSR_sRow = xe_partition_for_epilogue<ReferenceSrc>(sRow, args.epi_tile, tiled_s2r, args.thread_idx);
    Tensor tSR_rRow = make_tensor_like<ElementCompute>(take<0,3>(tSR_sRow));  // (CPY,CPY_M,CPY_N)

    auto sg = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_sub_group();

    return ConsumerStoreCallbacks(tSR_sRow, tSR_rRow, sg);
  }
};

// Column vector broadcast (M-dimension)
template<
  int Stages,
  class CtaTileShapeMNK,
  class ElementInput_,
  class ElementCompute = cute::remove_pointer_t<ElementInput_>,
  class StrideMNL_ = Stride<_1,_0,_0>,
  int Alignment = 128 / sizeof_bits_v<cute::remove_pointer_t<ElementInput_>>,
  bool EnableNullptr = true
>
struct Xe4ColBroadcast {
  using StrideMNL = StrideMNL_;
  using ElementInput = cute::remove_pointer_t<ElementInput_>;
  static constexpr bool IsArrayOfPointers = is_same_v<ElementInput*, ElementInput_>;
  using PtrColType = cute::conditional_t<IsArrayOfPointers, ElementInput const* const*, ElementInput const*>;

  static constexpr size_t CtaTileM = size<0>(CtaTileShapeMNK{});
  using CopyOpG2S = xe4::ASYNC_LINEAR_LOAD<ElementInput, CtaTileM>;

  static_assert(Stages == 0, "Column broadcast doesn't support smem pipelining");

  static constexpr bool IsDynamicBroadcast = is_same_v<remove_cvref_t<decltype(get<0>(StrideMNL{}))>, bool>;
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))> || IsDynamicBroadcast);
  static_assert(take<0,2>(StrideMNL{}) == Stride<_1,_0>{} || IsDynamicBroadcast);

  struct SharedStorage {
    array_aligned<ElementInput, CtaTileM> smem;
  };

  struct Arguments {
    PtrColType ptr_col = nullptr;
    ElementInput null_default = ElementInput(0);
    StrideMNL dCol = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Xe4ColBroadcast() { }

  CUTLASS_HOST_DEVICE
  Xe4ColBroadcast(Params const& params, SharedStorage const& shared_storage)
      : params(params), is_zero_(false),
        smem(const_cast<ElementInput*>(shared_storage.smem.data())) {
    auto const& [stride_M, stride_N, stride_L] = params.dCol;
    if (EnableNullptr && params.ptr_col == nullptr) {
      is_zero_ = params.null_default == ElementCompute(0);
    }
    else if (IsDynamicBroadcast && stride_M == bool(0) && stride_L == repeat_like(stride_L, 0)) {
       if constexpr (!IsArrayOfPointers) {
         is_zero_ = params.ptr_col[0] == ElementInput(0);
       }
    }
  }

  Params params;
  bool is_zero_ = false;
  ElementInput *smem = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return true;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return is_zero_;
  }

  template <class GS_GTensor, class GS_STensor, class Tiled_G2S>
  struct ProducerLoadCallbacks : EmptyProducerLoadCallbacks {
    CUTLASS_DEVICE
    ProducerLoadCallbacks(GS_GTensor tGS_gCol_, GS_STensor tGS_sCol_, Tiled_G2S tiled_g2s_, Params const& params_)
      : tGS_gCol(tGS_gCol_), tGS_sCol(tGS_sCol_), tiled_G2S(tiled_g2s_), params(params_) {}

    GS_GTensor tGS_gCol;
    GS_STensor tGS_sCol;
    Tiled_G2S tiled_G2S;

    Params const& params;

    CUTLASS_DEVICE void
    step(uint64_t* full_mbarrier_ptr, int epi_m, int epi_n, int load_iteration, bool issue_tma_load) {
      if (issue_tma_load && epi_m == 0) {
        constexpr uint32_t copy_bytes = CtaTileM * sizeof_bits_v<ElementInput> / 8;
        cutlass::arch::ClusterTransactionBarrier::expect_transaction(full_mbarrier_ptr, copy_bytes);

        Tensor tGS_gCol_flt = filter_zeros(tGS_gCol);
        Tensor tGS_sCol_flt = filter_zeros(tGS_sCol);
        Tensor gCol = coalesce(tGS_gCol_flt);
        Tensor sCol = coalesce(tGS_sCol_flt);
        copy(tiled_G2S.with(full_mbarrier_ptr, copy_bytes), gCol, sCol);
      }
    }
  };

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    auto layout_M = [&] () CUTLASS_LAMBDA_FUNC_INLINE {
      auto shape_M = get<0>(args.problem_shape_mnkl);
      if constexpr (IsDynamicBroadcast) {
        auto stride_M = repeat_like(shape_M, int(0));
        if (get<0>(params.dCol) == bool(1)) {
          stride_M = transform_leaf(compact_major<LayoutLeft>(shape_M),
            [] (auto const& stride) { return static_cast<int>(stride); }
          );
        }
        return make_layout(shape_M, stride_M);
      }
      else {
        return make_layout(shape_M);
      }
    }();

    auto layout_N = make_layout(N, repeat_like(N, _0{}));
    auto layout_L = make_layout(L, get<2>(params.dCol));

    ElementInput const* ptr_col = [&] () CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr(IsArrayOfPointers) {
        return params.ptr_col[l];
      } else {
        return params.ptr_col;
      }
    }();

    Tensor mCol = make_tensor(make_gmem_ptr(ptr_col), make_layout(layout_M,layout_N,layout_L));
    Tensor gCol = local_tile(mCol(_,_,l), take<0,2>(args.tile_shape_mnk), make_coord(m, n));        // (CTA_M, CTA_N)
    Tensor sCol = make_tensor(make_smem_ptr(smem),
      make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{})), make_shape(_1{}, _0{}));  // (CTA_M, CTA_N)

    //// G2S: Gmem to Smem
    auto tiled_g2s = make_tiled_copy(
      Copy_Atom<CopyOpG2S, ElementInput>{},
      Layout<Shape<_1,_1>, Stride<_1,_0>>{},
      Layout<Shape<Int<CtaTileM>,_1>, Stride<_1,_0>>{}
    );

    return ProducerLoadCallbacks(gCol, sCol, tiled_g2s, params);
  }

  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(ElementInput* smem_, int m_in_epi_tile_, int epi_tile_m_)
      : smem(smem_), m_in_epi_tile(m_in_epi_tile_), epi_tile_m(epi_tile_m_) {}

    ElementInput* smem;
    int m_in_epi_tile;
    int epi_tile_m;

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<ElementCompute, FragmentSize> frg_col;
      ElementCompute val = ElementCompute(smem[epi_m * epi_tile_m + m_in_epi_tile]);

      for (int i = 0; i < FragmentSize; ++i) {
        frg_col[i] = val;
      }

      return frg_col;
    }
  };

  template <
    bool ReferenceSrc,
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto sg = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_sub_group();

    constexpr int EpiTileM = size<0>(decltype(args.epi_tile){});
    constexpr int EpiTileN = size<1>(decltype(args.epi_tile){});
    constexpr int ElementsPerWarpN = 32;
    constexpr int NumWarpsAlongN = EpiTileN / ElementsPerWarpN;

    int lane_id = sg.get_local_linear_id();
    int warp_m = args.thread_idx / (cutlass::NumThreadsPerWarp * NumWarpsAlongN);
    int m_in_epi_tile = warp_m * cutlass::NumThreadsPerWarp + lane_id;

    return ConsumerStoreCallbacks(smem, m_in_epi_tile, EpiTileM);
  }
};

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
