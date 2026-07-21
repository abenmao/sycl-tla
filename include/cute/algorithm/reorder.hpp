/***************************************************************************************************
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

#include <cute/config.hpp>            // CUTE_HOST_DEVICE
#include <cute/tensor_impl.hpp>       // cute::Tensor
#include <cute/atom/reorder_atom.hpp>

// Subgroup-level ("warp" in CUDA terminology) register-to-register reorder operations.
// Currently implemented for Xe only.


namespace cute
{

namespace detail {

// Modify subgroup TV layout for subbyte types.
//
// In general on Xe successive elements in registers are assigned to work-items in
//   round-robin order (interleaved at element granularity). However, subbyte types are
//   only interleaved at byte granularity.
//
// This routine modifies the incoming layout to appear as though work-item ownership for subbyte
//   types is also at element granularity, to uniformize later logic.
template <class T, class InLayout>
CUTE_HOST_DEVICE
constexpr decltype(auto)
subbyte_sg_tv_swizzle(const InLayout &layout)
{
#ifdef SYCL_INTEL_TARGET
  using namespace intel;
  if constexpr (sizeof_bits_v<T> >= 8)
    return layout;
  else {
    static_assert(is_static_v<InLayout>, "Layout must be static");
    constexpr auto values = size(InLayout{}) / sg_size;
    constexpr auto per_byte = 8 / sizeof_bits_v<T>;
    static_assert(values % per_byte == 0, "Partially-occupied bytes in layout");
    return composition(layout, Layout<Shape<Shape<C<per_byte>, C<sg_size/per_byte>>, Shape<C<per_byte>, C<values/per_byte>>>,
                                      Stride<Stride<_SGSize, _1>, Stride<C<sg_size/per_byte>, C<sg_size*per_byte>>>>{});
  }
#else
  return layout;
#endif
}

#ifdef SYCL_INTEL_TARGET
template <class ReorderAtom, class SrcType, class DstType>
struct CoalescedReorderAtom {
  static constexpr bool value = false;
};

template <class ReorderAtom, class = void>
struct CoalescedReorderStagedAtom {
  static constexpr bool value = false;
};

template <class ReorderAtom>
struct CoalescedReorderStagedAtom<ReorderAtom, void_t<typename ReorderAtom::CoalescedStage>> {
  static constexpr bool value = true;
};

template <>
struct CoalescedReorderAtom<Xe_Reorder<ReorderKind::UU_Universal, float, cutlass::float_e5m2_t>,
                            float,
                            cutlass::float_e5m2_t> {
  static constexpr bool value = true;
};

template <>
struct CoalescedReorderAtom<Xe_Reorder<ReorderKind::UU_Universal, float, cutlass::float_e4m3_t>,
                            float,
                            cutlass::float_e4m3_t> {
  static constexpr bool value = true;
};

template <>
struct CoalescedReorderAtom<Xe_Reorder<ReorderKind::UU, float, bfloat16_t>,
                            float,
                            bfloat16_t> {
  static constexpr bool value = true;
};

template <int Dst, int Src, int Values, class ValueLayout>
struct CoalescedReorderFindSrc {
  static constexpr int mapped = decltype(ValueLayout{}(Int<Src>{}))::value;
  static constexpr int value = (mapped == Dst)
    ? Src
    : CoalescedReorderFindSrc<Dst, Src + 1, Values, ValueLayout>::value;
};

template <int Dst, int Values, class ValueLayout>
struct CoalescedReorderFindSrc<Dst, Values, Values, ValueLayout> {
  static constexpr int value = -1;
};

template <int Dst, int Offset, int ChunkValues, int Values, class ValueLayout>
struct CoalescedReorderChunkSupported {
  static constexpr bool value =
    CoalescedReorderFindSrc<Dst + Offset, 0, Values, ValueLayout>::value >= 0 &&
    CoalescedReorderChunkSupported<Dst, Offset + 1, ChunkValues, Values, ValueLayout>::value;
};

template <int Dst, int ChunkValues, int Values, class ValueLayout>
struct CoalescedReorderChunkSupported<Dst, ChunkValues, ChunkValues, Values, ValueLayout> {
  static constexpr bool value = true;
};

template <int Dst, int ChunkValues, int Values, class ValueLayout>
struct CoalescedReorderLayoutSupported {
  static constexpr bool chunk_in_range = (Dst + ChunkValues) <= Values;
  static constexpr bool chunk_supported = chunk_in_range &&
    CoalescedReorderChunkSupported<Dst, 0, ChunkValues, Values, ValueLayout>::value;
  static constexpr bool value = chunk_supported &&
    CoalescedReorderLayoutSupported<Dst + ChunkValues, ChunkValues, Values, ValueLayout>::value;
};

template <int ChunkValues, int Values, class ValueLayout>
struct CoalescedReorderLayoutSupported<Values, ChunkValues, Values, ValueLayout> {
  static constexpr bool value = true;
};

template <class DstType>
struct CoalescedReorderDstStorage {
  static constexpr int register_bits = 512 / intel::_SGSize::value;
  static constexpr int dst_bits = sizeof_bits_v<DstType>;
  static constexpr int storage_bits = bytes_to_bits(bits_to_bytes(dst_bits));
  static_assert(512 % intel::_SGSize::value == 0,
                "Coalesced reorder GRF bits must divide evenly across the subgroup");
  static_assert(register_bits % dst_bits == 0,
                "Coalesced reorder output register must contain a whole number of destination values");
  static_assert(register_bits * intel::_SGSize::value == 512,
                "Coalesced reorder output register must fill one 64B GRF across the subgroup");
  using DRegister = intel::vector_t<uint_byte_t<bits_to_bytes(dst_bits)>, register_bits / storage_bits>;
  static constexpr int value = register_bits / dst_bits;
};

template <class ReorderAtom, int ChunkValues, int Dst, int Values, class ValueLayout, class TensorSrc, int Chunks, int... Is>
CUTE_HOST_DEVICE void
coalesced_reorder_stage_chunk(TensorSrc const& src, typename ReorderAtom::CoalescedStage (&staged)[Chunks], int_sequence<Is...>)
{
  static_assert(CoalescedReorderChunkSupported<Dst, 0, ChunkValues, Values, ValueLayout>::value,
                "Coalesced reorder could not derive a complete dst-to-src mapping");
  ReorderAtom::reorder_stage(
      src(CoalescedReorderFindSrc<Dst + Is, 0, Values, ValueLayout>::value)...,
      staged[Dst / ChunkValues]);
}

template <class ReorderAtom, int ChunkValues, int Dst, int Values, class ValueLayout, class TensorSrc, int Chunks>
CUTE_HOST_DEVICE void
coalesced_reorder_stage(TensorSrc const& src, typename ReorderAtom::CoalescedStage (&staged)[Chunks])
{
  if constexpr (Dst < Values) {
    coalesced_reorder_stage_chunk<ReorderAtom, ChunkValues, Dst, Values, ValueLayout>(
        src, staged, make_int_sequence<ChunkValues>{});
    coalesced_reorder_stage<ReorderAtom, ChunkValues, Dst + ChunkValues, Values, ValueLayout>(src, staged);
  }
}

template <class ReorderAtom, class DRegister, int ChunkValues, int Dst, int Values, class TensorDst, int Chunks>
CUTE_HOST_DEVICE void
coalesced_reorder_store(typename ReorderAtom::CoalescedStage const (&staged)[Chunks], TensorDst& dst)
{
  if constexpr (Dst < Values) {
    ReorderAtom::reorder_pack(
        staged[Dst / ChunkValues],
        reinterpret_cast<DRegister&>(dst(Dst)));
    coalesced_reorder_store<ReorderAtom, DRegister, ChunkValues, Dst + ChunkValues, Values>(staged, dst);
  }
}

template <class ReorderAtom, class DRegister, int ChunkValues, int Values, class ValueLayout, class TensorSrc, class TensorDst>
CUTE_HOST_DEVICE void
coalesced_reorder_staged(TensorSrc const& src, TensorDst& dst)
{
  static_assert(Values % ChunkValues == 0,
                "Staged coalesced reorder expects whole chunks");
  constexpr int Chunks = Values / ChunkValues;
  typename ReorderAtom::CoalescedStage staged[Chunks];
  coalesced_reorder_stage<ReorderAtom, ChunkValues, 0, Values, ValueLayout>(src, staged);
  coalesced_reorder_store<ReorderAtom, DRegister, ChunkValues, 0, Values>(staged, dst);
}
#endif

} /* namespace detail */

// Subgroup-cooperative reorder.
//          src, dst: WI-owned fragments
//  slayout, dlayout: subgroup TV-layouts for these fragments.
//
// The layout of src/dst can be arbitrary. The TV layouts
//   are used to map values in src to values in dst.
template <class SEngine, class SLayoutWI, class SLayout,
          class DEngine, class DLayoutWI, class DLayout>
CUTE_HOST_DEVICE
void
reorder(Tensor<SEngine,SLayoutWI> const& src,       // WI fragment
        Tensor<DEngine,DLayoutWI> &      dst,       // WI fragment
        SLayout                   const& slayout,   // (src thr, src val) -> coord
        DLayout                   const& dlayout)   // (dst thr, dst val) -> coord
{
  using SType = typename SEngine::element_type;
  using DType = typename DEngine::element_type;

  static_assert(is_static_v<SLayout>, "Reorder source layout must be static");
  static_assert(is_static_v<DLayout>, "Reorder destination layout must be static");

  auto sl0 = detail::subbyte_sg_tv_swizzle<SType>(project_strides(slayout));
  auto dl0 = detail::subbyte_sg_tv_swizzle<DType>(project_strides(dlayout));

#ifdef SYCL_INTEL_TARGET
  auto impl = choose_xe_reorder_impl<SType, DType>(sl0, dl0);   // -> atom or dispatch tag
#else
  static_assert("Reorder only implemented on Xe");
#endif

  reorder_impl(impl, src, dst, sl0, dl0);
}

template <class SEngine, class SLayoutWI, class SLayout,
          class DEngine, class DLayoutWI, class DLayout>
CUTE_HOST_DEVICE
void
reorder(SubgroupTensor<SEngine,SLayoutWI,SLayout> const& src,
        SubgroupTensor<DEngine,DLayoutWI,DLayout> &      dst)
{
  reorder(src, dst, src.tv_layout(), dst.tv_layout());
}

// Accept mutable temporaries
template <class SEngine, class SLayoutWI, class SLayout,
          class DEngine, class DLayoutWI, class DLayout>
CUTE_HOST_DEVICE
void
reorder(Tensor<SEngine,SLayoutWI> const& src,       // WI fragment
        Tensor<DEngine,DLayoutWI>     && dst,       // WI fragment
        SLayout                   const& slayout,   // (src thr, src val) -> coord
        DLayout                   const& dlayout)   // (dst thr, dst val) -> coord
{
  reorder(src, dst, slayout, dlayout);
}

template <class SEngine, class SLayoutWI, class SLayout,
          class DEngine, class DLayoutWI, class DLayout>
CUTE_HOST_DEVICE
void
reorder(SubgroupTensor<SEngine,SLayoutWI,SLayout> const& src,
        SubgroupTensor<DEngine,DLayoutWI,DLayout>     && dst)
{
  reorder(src, dst);
}

// Base case for reorders: loop over reorder atoms
template <class ReorderAtom,
          class SEngine, class SLayoutWI, class SLayout,
          class DEngine, class DLayoutWI, class DLayout>
CUTE_HOST_DEVICE
void
reorder_impl(ReorderAtom               const& atom,
             Tensor<SEngine,SLayoutWI> const& src,       // WI fragment
             Tensor<DEngine,DLayoutWI> &      dst,       // WI fragment
             SLayout                   const& slayout,   // (src thr, src val) -> coord
             DLayout                   const& dlayout)   // (dst thr, dst val) -> coord
{
  using _SG = intel::_SGSize;
  using SType = typename SEngine::element_type;
  using RegistersSrc = typename ReorderAtom::SRegisters;
  using RegistersDst = typename ReorderAtom::DRegisters;
  using RegTypeSrc   = typename remove_extent<RegistersSrc>::type;
  using RegTypeDst   = typename remove_extent<RegistersDst>::type;
  constexpr int RegNumSrc = extent<RegistersSrc>::value;
  constexpr int RegNumDst = extent<RegistersDst>::value;
  constexpr int values = size(SLayout{}) / size<0>(SLayout{});
  constexpr int vchunk = sizeof_bits_v<RegistersSrc> / sizeof_bits_v<SType>;

  static constexpr bool has_broadcast = (size(DLayoutWI{}) > size(SLayoutWI{}));

  if (!has_broadcast) {
    // Calculate mapping from src val -> dst val on a chunk-by-chunk basis. Unlike a plain copy, there is no intrinsic
    //   correspondence of src/dst values for subgroup reorders.
    auto rlayout = coalesce(composition(right_inverse(dlayout), slayout));                 // src index -> dst index
    auto vrlayout = composition(composition(Layout<Shape<_SG, Int<values>>, Stride<_0, _1>>{},
                                            rlayout),
                                Layout<Shape<_1, Int<values>>, Stride<_0, _SG>>{});        // src val -> dst val

#ifdef SYCL_INTEL_TARGET
    if constexpr (detail::CoalescedReorderAtom<ReorderAtom, SType, typename DEngine::element_type>::value) {
      using DstStorage = detail::CoalescedReorderDstStorage<typename DEngine::element_type>;
      using DRegister = typename DstStorage::DRegister;
      using ValueLayout = decltype(vrlayout);
      constexpr int packed_values = DstStorage::value;
      if constexpr (values % packed_values == 0) {
        if constexpr (detail::CoalescedReorderLayoutSupported<0, packed_values, values, ValueLayout>::value) {
          static_assert(detail::CoalescedReorderStagedAtom<ReorderAtom>::value,
                        "Coalesced reorder atoms must define CoalescedStage and staged reorder_pack hooks");
          detail::coalesced_reorder_staged<ReorderAtom, DRegister, packed_values, values, ValueLayout>(src, dst);
          return;
        }
      }
    }
#endif

    CUTE_UNROLL
    for (int sv = 0; sv < values; sv += vchunk) {
      auto pS = recast_ptr<RegTypeSrc>(src.data() + sv);
      auto pD = recast_ptr<RegTypeDst>(dst.data() + vrlayout(sv));

      detail::explode(detail::CallReorder<ReorderAtom>{},
                      pS, make_int_sequence<RegNumSrc>{},
                      pD, make_int_sequence<RegNumDst>{});
    }
  } else {
    // If there is broadcast happening, then we need to loop over dst values instead.
    auto rlayout = coalesce(composition(right_inverse(slayout), dlayout));                 // dst index -> src index
    auto vrlayout = composition(composition(Layout<Shape<_SG, Int<values>>, Stride<_0, _1>>{},
                                            rlayout),
                                Layout<Shape<_1, Int<values>>, Stride<_0, _SG>>{});        // dst val -> src val

    CUTE_UNROLL
    for (int dv = 0; dv < values; dv += vchunk) {
      auto pS = recast_ptr<RegTypeSrc>(src.data() + vrlayout(dv));
      auto pD = recast_ptr<RegTypeDst>(dst.data() + dv);

      detail::explode(detail::CallReorder<ReorderAtom>{},
                      pS, make_int_sequence<RegNumSrc>{},
                      pD, make_int_sequence<RegNumDst>{});
    }
  }
}

template <typename T>
using upcast_subbyte_t = conditional_t<is_subbyte_v<T>,
                                       conditional_t<cutlass::platform::numeric_limits<T>::is_integer,
                                                     conditional_t<cutlass::platform::numeric_limits<T>::is_signed,
                                                                   int8_t, uint8_t>,
                                                     half_t>,
                                       T>;

// Reorder strategy: type conversion, then layout change.
template <class SEngine, class SLayoutWI, class SLayout,
          class DEngine, class DLayoutWI, class DLayout>
CUTE_HOST_DEVICE
void
reorder_impl(ReorderDispatchConvertRelayout const&,
             Tensor<SEngine,SLayoutWI> const& src,       // WI fragment
             Tensor<DEngine,DLayoutWI> &      dst,       // WI fragment
             SLayout                   const& slayout,   // (src thr, src val) -> coord
             DLayout                   const& dlayout)   // (dst thr, dst val) -> coord
{
  using SrcType = typename SEngine::element_type;
  using DstType = typename DEngine::element_type;
  using NewSrcType = conditional_t<is_subbyte_v<SrcType>, upcast_subbyte_t<SrcType>, DstType>;
  auto src_c = make_fragment_like<NewSrcType>(src);

  reorder(src, src_c, slayout, slayout);
  reorder(src_c, dst, slayout, dlayout);
}

// Reorder strategy: layout change, then type conversion
template <class SEngine, class SLayoutWI, class SLayout,
          class DEngine, class DLayoutWI, class DLayout>
CUTE_HOST_DEVICE
void
reorder_impl(ReorderDispatchRelayoutConvert const&,
             Tensor<SEngine,SLayoutWI> const& src,       // WI fragment
             Tensor<DEngine,DLayoutWI> &      dst,       // WI fragment
             SLayout                   const& slayout,   // (src thr, src val) -> coord
             DLayout                   const& dlayout)   // (dst thr, dst val) -> coord
{
  using SrcType = typename SEngine::element_type;
  using DstType = typename DEngine::element_type;
  using NewDstType = conditional_t<is_same_v<SrcType, DstType>, upcast_subbyte_t<DstType>, SrcType>;
  auto dst_c = make_fragment_like<NewDstType>(dst);

  reorder(src, dst_c, slayout, dlayout);
  reorder(dst_c, dst, dlayout, dlayout);
}


} // end namespace cute
