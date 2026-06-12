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

#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/algorithm/prefetch.hpp>
#include <cute/arch/mma_xe4_desc.hpp>
#include <cute/arch/copy_xe4_ldsm.hpp>
#include <cute/arch/async_tensor_copy.hpp>
#include <cute/arch/xe4_async_gmma_slm_layout.hpp>
#include <cute/atom/copy_traits_xe4_adma.hpp>

namespace cute {

// MInfo: Matrix descriptor info carrier for LDSM Copy_Traits.
// Wraps the raw 32-bit hardware matrix descriptor that encodes the SLM surface
// (start address, strides, dimensions). Passed to the LDSM load/store/reduce
// intrinsics at copy time. Default-constructible so that Copy_Traits and
// Copy_Atom can be instantiated without a descriptor (e.g., during type-level
// TiledCopy construction); the actual descriptor is set later via
// make_ldsm_tiled_copy() or make_ldsm_matrix_descriptor().
struct MInfo {
  MInfo() = default;

  MInfo(const MatrixDescriptor mdesc) : matrix_desc_(mdesc.raw_) {}
  uint32_t matrix_desc_;
};


using namespace cute::detail;

// Xe4LDSMTraitsBase: Base traits for descriptor-based LDSM copy operations.
//
// This defines the thread-to-data mapping for XE4_LOAD_MATRIX, XE4_STORE_MATRIX,
// and XE4_REDUCE_MATRIX operations that use hardware matrix descriptors.
//
// Key design choice: ThrID = Layout<Int<1>> (single-thread model)
//   Unlike legacy XE4_LDSM traits which use ThrID = Layout<Shape<_32>> (32-thread warp),
//   descriptor-based LDSM ops model each thread independently. The hardware matrix
//   descriptor encodes the full SLM surface, and each thread computes its own coordinate
//   via the identity tensor partition. The thread-level cooperation (e.g., 128 threads
//   for UnorderedVector) is handled by the TiledCopy's thread layout, not by ThrID.
//
// SrcLayout/DstLayout: Shape<Int<1>, Int<DataSize>>
//   The value mode (mode-1) carries all data bits for this thread: BitWidth * Vlen * Alen.
//   The thread mode (mode-0) is 1 since each atom describes one thread's work.
//
template<class OP>
struct Xe4LDSMTraitsBase {
    using Op = OP;
    /*
     /For Matrix Type 1
     BitWidth = 16
     Vlen =1  Layout = (1, 16*1)
     Vlen =2  Layout = (1, 16*2)
    */
    static constexpr uint32_t Alen = (Op::Alen == 0) ? 1 : Op::Alen;
    static constexpr uint32_t DataSize = Op::BitWidth * Op::Vlen * Alen;
    using SrcLayout = Layout<Shape<Int<1>, Int<DataSize>>>;
    using DstLayout = SrcLayout;
    using RefLayout = DstLayout;
    using ThrID = Layout<Int<1>>;

    static constexpr int ValBits = Op::BitWidth;
    static_assert(Op::Super::CopyBitsPerThread % ValBits == 0, "Type is incompatible with this copy atom");

    CUTE_HOST_DEVICE
    Xe4LDSMTraitsBase() {}

};

// swap_coord_for_ldsm: Convert a CuTe ArithmeticTuple coordinate to the
// sycl::marray<uint16_t, 2> format expected by the LDSM hardware intrinsics.
//
// "Swap" because JGS (the hardware interface) expects (x, y) = (col, row),
// but CuTe's ArithmeticTuple stores (row, col) — i.e., get<0> is the row (y)
// and get<1> is the column (x). So we swap the order.
//
// This is a template over T0, T1 (not restricted to int/C<0>) because the
// epilogue's identity tensors can produce coordinates with varying integer
// types. For example:
//   - make_identity_tensor(make_shape(Int<64>{}, Int<256>{}))
//     produces ArithmeticTuple<unsigned int, unsigned int>
//   - Unit tests may produce ArithmeticTuple<int, int> or
//     ArithmeticTuple<int, C<0>>
// A single template handles all these cases via static_cast<uint16_t>.
template <class T0, class T1>
sycl::marray<uint16_t, 2>
swap_coord_for_ldsm(cute::ArithmeticTuple<T0, T1> const& t) {
  uint16_t x = static_cast<uint16_t>(cute::get<1>(t));
  uint16_t y = static_cast<uint16_t>(cute::get<0>(t));
  return sycl::marray<uint16_t, 2>{x, y};
}

// Overload for flattened hierarchical coordinates.
// When make_ldsm_copy_C is used with ThrGroupSize > 1 (multi-warp), the identity
// tensor partition produces coordinates of type:
//   ArithmeticTuple<ArithmeticTuple<row, row_offset>, ArithmeticTuple<col, col_offset>>
// After flatten_to_tuple, this becomes cute::tuple<row, row_offset, col, col_offset>.
// We reconstruct the 2D coordinate by summing each pair: row = get<0>+get<1>, col = get<2>+get<3>.
template <class T0, class T1, class T2, class T3>
sycl::marray<uint16_t, 2>
swap_coord_for_ldsm(cute::tuple<T0, T1, T2, T3> const& t) {
  uint16_t x = static_cast<uint16_t>(cute::get<2>(t) + cute::get<3>(t));
  uint16_t y = static_cast<uint16_t>(cute::get<0>(t) + cute::get<1>(t));
  return sycl::marray<uint16_t, 2>{x, y};
}

template<typename T, class SrLayout, LDSMMode Mode, uint32_t Vlen,
         cute::Vecdir Vdir, class MatInfo,
	 uint32_t Alen, cute::Arrdir Adir>
struct Copy_Traits<XE4_LOAD_MATRIX<T, SrLayout, Mode, Vlen, Vdir, Alen, Adir>, MatInfo>
       : Xe4LDSMTraitsBase<XE4_LOAD_MATRIX<T, SrLayout, Mode, Vlen, Vdir, Alen, Adir>> {

    using Op = XE4_LOAD_MATRIX<T, SrLayout, Mode, Vlen, Vdir, Alen, Adir>;
    using Super = Xe4LDSMTraitsBase<Op>;
    using ThrID = typename Super::ThrID;
    // Logical thread id to thread idx
    using ThrLayout = Layout<_32>;
    // Map from (src-thr,src-val) to bit
    using SrcLayout = typename Super::SrcLayout;
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = SrcLayout;
    // Reference map from (thr,val) to bit
    using RefLayout = typename Super::RefLayout;
    using Traits = Copy_Traits<Op, MInfo>;
    MatInfo cache_;
    Copy_Traits() : Super(), cache_() {}
    Copy_Traits(MatInfo minfo)
	: Super(),
       	cache_(minfo) {}

  // Execution.
  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_DEVICE friend constexpr void
  copy_unpack(Traits const&                   traits,
              Tensor<SEngine, SLayout> const& src,
              Tensor<DEngine, DLayout> &      dst) {
    using SType = typename SEngine::value_type;
    using DType = typename DEngine::value_type;
    using SrcLayout = typename Traits::SrcLayout;
    using DstLayout = typename Traits::DstLayout;
    constexpr auto DBits = sizeof_bits_v<DType>;

    //static_assert(is_counting_layout_v<SLayout>, "Source tensor must be a coordinate tensor.");
    static_assert(is_rmem_v<DEngine>, "Destination tensor must be in registers.");
    static_assert(size(SLayout{}) * DBits == size<1>(SrcLayout{}),
                  "Source tensor size does not match copy atom size.");
    static_assert(size(DLayout{}) * DBits == size<1>(DstLayout{}),
                  "Destination tensor size does not match copy atom size.");

    auto as_xe4_coord = [](auto const& t) {
      return swap_coord_for_ldsm(flatten_to_tuple(t));
    };
    auto coord = as_xe4_coord(src.data().coord_);
    Op::copy(dst.data(), traits.cache_.matrix_desc_, coord);
  }
};

// Single-parameter Copy_Traits specialization for XE4_LOAD_MATRIX.
//
// WHY THIS IS NEEDED:
// The epilogue builder constructs copy atoms via:
//   Copy_Atom<CopyOp, ElementD>
// which internally resolves to (see copy_atom.hpp line ~48):
//   Copy_Atom<Copy_Traits<CopyOp>, ElementD>
// This means Copy_Traits<CopyOp> (single param) must exist. But the actual
// implementation lives in Copy_Traits<CopyOp, MatInfo> (two params) because
// LDSM ops need a matrix descriptor (MatInfo).
//
// This bridge specialization inherits from the two-param version with MInfo
// as default, making it default-constructible. The descriptor is populated
// later when make_ldsm_tiled_copy() constructs the real TiledCopy with
// a valid MatrixDescriptor.
template<typename T, class SrLayout, LDSMMode Mode, uint32_t Vlen,
         cute::Vecdir Vdir, uint32_t Alen, cute::Arrdir Adir>
struct Copy_Traits<XE4_LOAD_MATRIX<T, SrLayout, Mode, Vlen, Vdir, Alen, Adir>>
       : Copy_Traits<XE4_LOAD_MATRIX<T, SrLayout, Mode, Vlen, Vdir, Alen, Adir>, MInfo> {
    using Super = Copy_Traits<XE4_LOAD_MATRIX<T, SrLayout, Mode, Vlen, Vdir, Alen, Adir>, MInfo>;
    Copy_Traits() : Super() {}
};

template<typename T, class SrLayout, LDSMMode Mode, uint32_t Vlen,
         cute::Vecdir Vdir, class MatInfo,
	 uint32_t Alen, cute::Arrdir Adir>
struct Copy_Traits<XE4_STORE_MATRIX<T, SrLayout, Mode, Vlen, Vdir, Alen, Adir>, MatInfo>
       : Xe4LDSMTraitsBase<XE4_STORE_MATRIX<T, SrLayout, Mode, Vlen, Vdir, Alen, Adir>> {

  using Op= XE4_STORE_MATRIX<T, SrLayout, Mode, Vlen, Vdir, Alen, Adir>;
  using Super = Xe4LDSMTraitsBase<Op>;
  using ThrID = typename Super::ThrID;
  // Logical thread id to thread idx
  using ThrLayout = Layout<_32>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = typename Super::SrcLayout;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = SrcLayout;
  // Reference map from (thr,val) to bit
  using RefLayout = typename Super::RefLayout;
  using Traits = Copy_Traits<Op, MInfo>;
  MatInfo cache_;
  Copy_Traits() : Super(), cache_() {}
  Copy_Traits(MatInfo minfo)
        : Super(),
          cache_(minfo) {}

  // Execution.
  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_DEVICE friend constexpr void
  copy_unpack(Traits const&                   traits,
              Tensor<SEngine, SLayout> const& src,
              Tensor<DEngine, DLayout> &      dst) {

    using SType = typename SEngine::value_type;
    using DType = typename DEngine::value_type;
    using SrcLayout = typename Traits::SrcLayout;
    using DstLayout = typename Traits::DstLayout;
    constexpr auto SBits = sizeof_bits_v<SType>;

    static_assert(is_counting_layout_v<DLayout>, "Destination tensor must be a coordinate tensor.");
    static_assert(is_rmem_v<SEngine>, "Source tensor must be in registers.");
    static_assert(size(SLayout{}) * SBits == size<1>(SrcLayout{}),
                  "Source tensor size does not match copy atom size.");
    static_assert(size(DLayout{}) * SBits == size<1>(DstLayout{}),
                  "Destination tensor size does not match copy atom size.");

    auto as_xe4_coord = [](auto const& t) {
      return swap_coord_for_ldsm(flatten_to_tuple(t));
    };
    auto coord = as_xe4_coord(dst.data().coord_);
    Op::copy(src.data(), traits.cache_.matrix_desc_, coord);
  }
};

// Single-parameter Copy_Traits specialization for XE4_STORE_MATRIX.
// Same bridge pattern as XE4_LOAD_MATRIX above — enables Copy_Atom<CopyOp, ElementD>
// resolution chain. See the XE4_LOAD_MATRIX single-param comment for full explanation.
template<typename T, class SrLayout, LDSMMode Mode, uint32_t Vlen,
         cute::Vecdir Vdir, uint32_t Alen, cute::Arrdir Adir>
struct Copy_Traits<XE4_STORE_MATRIX<T, SrLayout, Mode, Vlen, Vdir, Alen, Adir>>
       : Copy_Traits<XE4_STORE_MATRIX<T, SrLayout, Mode, Vlen, Vdir, Alen, Adir>, MInfo> {
    using Super = Copy_Traits<XE4_STORE_MATRIX<T, SrLayout, Mode, Vlen, Vdir, Alen, Adir>, MInfo>;
    Copy_Traits() : Super() {}
};

template<typename T, cute::MredOp Rop, class SrLayout, LDSMMode Mode, uint32_t Vlen,
         cute::Vecdir Vdir, class MatInfo,
	 uint32_t Alen, cute::Arrdir Adir>
struct Copy_Traits<XE4_REDUCE_MATRIX<T, Rop, SrLayout, Mode, Vlen, Vdir, Alen, Adir>, MatInfo>
       : Xe4LDSMTraitsBase<XE4_REDUCE_MATRIX<T, Rop, SrLayout, Mode, Vlen, Vdir, Alen, Adir>> {

  using Op= XE4_REDUCE_MATRIX<T, Rop, SrLayout, Mode, Vlen, Vdir, Alen, Adir>;
  using Super = Xe4LDSMTraitsBase<Op>;
  using ThrID = typename Super::ThrID;
  // Logical thread id to thread idx
  using ThrLayout = Layout<_32>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = typename Super::SrcLayout;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = SrcLayout;
  // Reference map from (thr,val) to bit
  using RefLayout = typename Super::RefLayout;
  using Traits = Copy_Traits<Op, MInfo>;
  MatInfo cache_;
  Copy_Traits() : Super(), cache_() {}
  Copy_Traits(MatInfo minfo)
        : Super(),
          cache_(minfo) {}

  // Execution.
  template <class SEngine, class SLayout,
            class DEngine, class DLayout>
  CUTE_DEVICE friend constexpr void
  copy_unpack(Traits const&                   traits,
              Tensor<SEngine, SLayout> const& src,
              Tensor<DEngine, DLayout> &      dst) {

    using SType = typename SEngine::value_type;
    using DType = typename DEngine::value_type;
    using SrcLayout = typename Traits::SrcLayout;
    using DstLayout = typename Traits::DstLayout;
    constexpr auto SBits = sizeof_bits_v<SType>;

    static_assert(is_counting_layout_v<DLayout>, "Destination tensor must be a coordinate tensor.");
    static_assert(is_rmem_v<SEngine>, "Source tensor must be in registers.");
    static_assert(size(SLayout{}) * SBits == size<1>(SrcLayout{}),
                  "Source tensor size does not match copy atom size.");
    static_assert(size(DLayout{}) * SBits == size<1>(DstLayout{}),
                  "Destination tensor size does not match copy atom size.");

    auto as_xe4_coord = [](auto const& t) {
      return swap_coord_for_ldsm(flatten_to_tuple(t));
    };
    auto coord = as_xe4_coord(dst.data().coord_);
    Op::copy(src.data(), traits.cache_.matrix_desc_, coord);
  }
};

// Single-parameter Copy_Traits specialization for XE4_REDUCE_MATRIX.
// Same bridge pattern as XE4_LOAD_MATRIX above — enables Copy_Atom<CopyOp, ElementD>
// resolution chain. See the XE4_LOAD_MATRIX single-param comment for full explanation.
template<typename T, cute::MredOp Rop, class SrLayout, LDSMMode Mode, uint32_t Vlen,
         cute::Vecdir Vdir, uint32_t Alen, cute::Arrdir Adir>
struct Copy_Traits<XE4_REDUCE_MATRIX<T, Rop, SrLayout, Mode, Vlen, Vdir, Alen, Adir>>
       : Copy_Traits<XE4_REDUCE_MATRIX<T, Rop, SrLayout, Mode, Vlen, Vdir, Alen, Adir>, MInfo> {
    using Super = Copy_Traits<XE4_REDUCE_MATRIX<T, Rop, SrLayout, Mode, Vlen, Vdir, Alen, Adir>, MInfo>;
    Copy_Traits() : Super() {}
};

// make_ldsm_matrix_descriptor: Build a hardware MatrixDescriptor from an SLM tensor.
//
// The MatrixDescriptor encodes the SLM surface properties (start address, strides,
// dimensions) that the LDSM load/store intrinsics use to access shared memory.
//
// ComposedLayout handling:
//   When the epilogue calls make_slm_tensor(), it applies CoreMatrix::retile<T>()
//   which adds a swizzle for SLM bank-conflict avoidance. This produces a
//   ComposedLayout<Swizzle<...>, Offset, Layout<...>> instead of a plain Layout.
//   ComposedLayout does NOT support .stride() directly (it would be a deleted function).
//   Since the hardware handles swizzling separately, we strip the swizzle layer via
//   layout_b() to get the underlying base layout for descriptor construction.
//   The is_composed_layout<> trait (from layout_composed.hpp) detects this case.
//
// Rank handling:
//   SLM layouts can be rank > 2 (e.g., after tiling). If so, we coalesce to
//   reduce to rank-2 before passing to make_matrix_descriptor().
//
// StartAddress:
//   The hardware descriptor requires the SLM start address right-shifted by 9
//   (i.e., in units of 512 bytes). slm_space_cast converts to SLM address space.
//
template <class GEngine, class SLayout>
CUTE_HOST
auto
make_ldsm_matrix_descriptor(
    Tensor<GEngine,SLayout> const& stensor,
    bool is_B_matrix =true
) {
  auto slayout = layout(stensor);
  // Extract base layout, stripping swizzle from ComposedLayout if present.
  // LDSM descriptors encode 2D surface strides; core-matrix swizzling is
  // handled by the hardware and should not be included in the descriptor.
  auto base_layout = [&]() {
    if constexpr (is_composed_layout<decltype(slayout)>::value) {
      // ComposedLayout = Swizzle ∘ Offset ∘ BaseLayout
      // layout_b() returns the BaseLayout (the actual shape/stride info)
      return slayout.layout_b();
    } else {
      return slayout;
    }
  }();
  MatrixDescriptor matrix_desc{};
  // Coalesce higher-rank layouts to rank-2 for the hardware descriptor
  if constexpr(decltype(rank(flatten(base_layout.shape())))::value > 2)
    matrix_desc = make_matrix_descriptor(coalesce(base_layout), !is_B_matrix);
  else
    matrix_desc = make_matrix_descriptor(base_layout, !is_B_matrix);

  // Start address: SLM address in 512-byte units (right-shift by 9 bits)
  matrix_desc.StartAddress = static_cast<uint32_t>(
      reinterpret_cast<uint64_t>(slm_space_cast(&*stensor.data()))) >> 9;

  return matrix_desc;
}

template<int Vlen, cute::Vecdir Vdir> struct LdsmValLayout {
   static constexpr Layout v_layout = make_layout(make_shape(Int<1>{}, Int<Vlen>{}));
};
template<int Vlen> struct LdsmValLayout<Vlen, cute::Vecdir::Vcol> {
   static constexpr Layout v_layout = make_layout(make_shape(Int<Vlen>{}, Int<1>{}));
};

template<uint32_t ThCount, uint32_t GroupSize, bool UnorderedType=false> struct LdsmThrLayout {
   static constexpr Layout t_layout = make_layout(make_shape(Int<ThCount>{}, Int<1>{}));
};
template<uint32_t ThCount, uint32_t GroupSize> struct LdsmThrLayout<ThCount,
                                	GroupSize, true> {
  static_assert((GroupSize > 0) && (GroupSize % 4 == 0));
  static constexpr Layout t_layout = make_layout(
		   make_shape(make_shape(Int<8>{}, Int<ThCount/8>{}), Int<GroupSize>{}),
                   make_stride(make_stride(Int<1>{}, Int<32>{}), Int<8>{}));
};

// MMA-aware thread layout for multi-warp LDSM copies.
// For UnorderedVector: delegates to LdsmThrLayout's hardware-required interleaved pattern.
// For other modes: uses ordered layout distributing warps along M then N, matching the
// epilogue pattern from xe4_epilogue_adma_warpspecialized.hpp.
template<int NumWarpsAlongM, int NumWarpsAlongN, bool IsUnordered>
struct LdsmMmaThrLayout {
  static constexpr int kNumThreadsPerWarp = 32;
  static constexpr auto t_layout = make_layout(
    make_shape(make_shape(Int<kNumThreadsPerWarp>{}, Int<NumWarpsAlongM>{}),
               Int<NumWarpsAlongN>{}),
    make_stride(make_stride(Int<1>{}, Int<kNumThreadsPerWarp * NumWarpsAlongN>{}),
                Int<kNumThreadsPerWarp>{}));
};
template<int NumWarpsAlongM, int NumWarpsAlongN>
struct LdsmMmaThrLayout<NumWarpsAlongM, NumWarpsAlongN, true> {
  static constexpr int ThrGroupSize = NumWarpsAlongM * NumWarpsAlongN;
  static constexpr auto t_layout = LdsmThrLayout<32, ThrGroupSize, true>::t_layout;
};

template <int ThrGroupSize=1,
	  class CopyOp,
          class GEngine,
          class SLayout>
CUTE_HOST_DEVICE
auto
make_ldsm_tiled_copy(const CopyOp& Op,
	             Tensor<GEngine,SLayout> const& stensor,
		     bool is_B_matrix=true)
{
  using Traits = Copy_Traits<CopyOp, MInfo>;
  using ThrLayout = typename Traits::ThrLayout;
  constexpr auto ThrCount = get<0>(ThrLayout{}.shape());
  constexpr bool UnorderedType = ((CopyOp::CopyMode == UnorderedVector) ||
	                          (CopyOp::CopyMode == UnorderedArrOfVectors));
  auto t_layout = LdsmThrLayout<ThrCount, ThrGroupSize, UnorderedType>::t_layout;
  auto v_layout = LdsmValLayout<CopyOp::Vlen, CopyOp::Vdir>::v_layout;

  return make_ldsm_tiled_copy(Op, stensor, t_layout, v_layout, is_B_matrix);
}

template <class CopyOp,
          class GEngine,
          class SLayout,
          class TLayout,
          class VLayout>
auto make_ldsm_tiled_copy(const CopyOp& Op,
		          Tensor<GEngine,SLayout> const& stensor,
			  TLayout t_layout,
			  VLayout v_layout,
			  bool is_B_matrix=true)
{
    using ValType = typename GEngine::value_type;
    MatrixDescriptor matrix_desc = make_ldsm_matrix_descriptor(stensor, is_B_matrix);
    //constexpr auto Vdir = get_vector_dir(SLayout{});
    //using CopyOp = XE4_LOAD_MATRIX<ValType, SLayout, Mode, Vlen, Vdir>;
    if constexpr (CopyOp::CopyMode == LDSMMode::UnorderedVector ||
		  CopyOp::CopyMode == LDSMMode::UnorderedArrOfVectors )
      static_assert(size(t_layout) % 128 ==0);
    using  Traits = Copy_Traits<CopyOp, MInfo>;
    using  Atom = Copy_Atom<Traits, ValType>;
    Traits traits{matrix_desc};
    Copy_Atom atom = Atom{traits};
    auto tiled_copy = make_tiled_copy(atom, t_layout, v_layout);
    return tiled_copy;
}

// ---------------------------------------------------------------------------
// make_ldsm_tiled_copy_noninjective — caller-supplies-TV variant
// ---------------------------------------------------------------------------
//
// PURPOSE
// -------
// Build an LDSM `TiledCopy` from a hand-rolled `(T, V) → (M, N)` layout and
// an explicit `Tiler_MN`, bypassing CuTe's `make_tiled_copy` (which uses
// `raked_product → right_inverse` and rejects non-injective layouts).
//
// This is the public entry point that packages the
// `make_tiled_copy_impl(atom, tv_layout, tiler)` shape used internally by
// `make_ldsm_copy_warp_row_CD_int`.  Use it whenever the desired
// (T, V) → (M, N) bijection is non-injective from `make_tiled_copy`'s point
// of view but is exactly representable as a single `Layout` — e.g., the
// FMHA softmax within-warp-row layout where row-mates must live in the
// same warp.
//
// CALLER CONTRACT
// ---------------
//   * `tv_layout` maps `(T, V) → tile_block_linear`.  Its codomain must be
//     bijective onto the per-call atom block; CuTe will compose it with the
//     tile_block layout (derived from `tiler_mn`) to produce the
//     `(T, V) → (m, n)` partition.
//   * `tiler_mn` is the per-call `Tile<M_layout, N_layout>` whose codomain
//     spans the rows/cols touched by one HW atom invocation.  Its
//     `complement` in `(TileM, TileN)` becomes the `RestM/RestN` axis of
//     `partition_S/D`.
//   * Same descriptor / atom plumbing as the simpler `make_ldsm_tiled_copy`
//     overloads — `slm_tensor` is recast to a 4-byte element type when the
//     underlying SLM data is sub-32-bit (so `make_ldsm_matrix_descriptor`'s
//     Pitch is HW-correct).
//
// USAGE
// -----
//   // Build TV layout + tiler externally (see detail::make_warp_row_tv_layout
//   // and detail::make_warp_row_m_tiler_layout for the FMHA pattern).
//   auto tv  = my_make_tv_layout<...>();
//   auto m_t = my_m_tiler<...>();
//   auto n_t = cute::make_layout(cute::Int<TileN>{}, cute::_1{});
//   auto tlr = cute::make_tile(m_t, n_t);
//
//   // Pick op type (load/store, mode, Vlen, Alen) yourself.
//   using Op = XE4_LOAD_MATRIX<ValType, RepSLayout, Mode, Vlen,
//                              Vecdir::Vrow, Alen, Arrdir::Arow>;
//
//   auto tc = cute::make_ldsm_tiled_copy_noninjective(Op{}, slm_tensor, tv, tlr);
//
// COMPARED WITH `make_ldsm_tiled_copy(op, stensor, t_layout, v_layout, …)`:
//   That overload computes the TV layout via `make_tiled_copy(atom, t, v)`,
//   which fails for non-injective (T, V) layouts.  This overload skips that
//   step — the caller has already done the (T, V) → tile_block math.
//
// COMPARED WITH `make_ldsm_copy_warp_row_C/D`:
//   Those wrappers hard-code the FMHA within-warp-row formula.  This
//   overload is the building block: any caller with a custom non-injective
//   TV layout can call it directly.
template <class CopyOp,
          class GEngine, class SLayout,
          class TVLayout, class TilerMN>
CUTE_HOST_DEVICE
auto
make_ldsm_tiled_copy_noninjective(const CopyOp& /*op*/,
                            Tensor<GEngine, SLayout> const& slm_tensor,
                            TVLayout                        tv_layout,
                            TilerMN                         tiler_mn,
                            bool                            is_B_matrix = true)
{
  using ValType = typename GEngine::value_type;
  MatrixDescriptor matrix_desc = make_ldsm_matrix_descriptor(slm_tensor, is_B_matrix);
  using Traits = Copy_Traits<CopyOp, MInfo>;
  using Atom   = Copy_Atom<Traits, ValType>;
  Atom atom = Atom{Traits{matrix_desc}};
  return make_tiled_copy_impl(atom, tv_layout, tiler_mn);
}

template <class CopyReduceOp,
          class GEngine,
          class SLayout>
CUTE_HOST_DEVICE
auto
make_ldsm_tiled_copy_reduce(const CopyReduceOp& Op,
	             Tensor<GEngine,SLayout> const& stensor,
		     bool is_B_matrix=true)
{
  return make_ldsm_tiled_copy(Op, stensor, is_B_matrix);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// LDSM Op Auto-Selector (Module 1)
/// Selects optimal XE4_LOAD_MATRIX / XE4_STORE_MATRIX parameters from SLM layout.
/// Mode priority: UnorderedVector (Cooprow) > CoopVector (Cooprow) > Vector (Vrow)
/// See MODULE_PLAN.md - Module 1 and ARCHITECTURE.md for design rationale.
////////////////////////////////////////////////////////////////////////////////////////////////////

// Max Vlen for cooperative row access (UnorderedVector / CoopVector, Type1 only)
// Constraint source: XE4_LDSTMatrixBase::check_coop_row_vector_constraints()
template <int BitWidth>
CUTE_HOST_DEVICE constexpr int
ldsm_coop_vlen() {
  static_assert(BitWidth == 64 || BitWidth == 32 || BitWidth == 16 ||
                BitWidth == 8  || BitWidth == 6  || BitWidth == 4,
                "Unsupported BitWidth for LDSM cooperative row access");
  if constexpr (BitWidth >= 64) return 4;
  else if constexpr (BitWidth == 32) return 8;
  else if constexpr (BitWidth == 16) return 16;
  else return 32; // 8b, 6b, 4b
}

// Max Vlen for standard Vector row access
// Constraint source: XE4_LDSTMatrixBase::check_row_vector_constraints()
template <int BitWidth, bool IsType1>
CUTE_HOST_DEVICE constexpr int
ldsm_vector_row_vlen() {
  if constexpr (IsType1) {
    // Type1 row: same max Vlen as cooperative row
    return ldsm_coop_vlen<BitWidth>();
  } else {
    // Type2 row: restricted Vlen range
    static_assert(BitWidth == 64 || BitWidth == 32 || BitWidth == 16 || BitWidth == 8,
                  "Unsupported BitWidth for LDSM Type2 vector row access");
    if constexpr (BitWidth == 64) return 4;
    else if constexpr (BitWidth == 32) return 8;
    else return 8; // 16b and 8b: max 8 for Type2
  }
}

// Detect if SLM layout is Type1 (N-contiguous, stride<1>=1).
// Simple Layout: check stride directly. ComposedLayout (swizzled SLM): always Type1
// in the CUTLASS framework (builder guarantees row-major SLM).
template<class SLayout>
struct LdsmIsType1 : true_type {};  // Default: assume Type1

template<class Shape, class Stride>
struct LdsmIsType1<Layout<Shape, Stride>> {
  static constexpr bool value = (get<1>(Stride{}) == 1);
};

// MMA-aware mode and Vlen selection for LDSM copies.
// Given the per-warp tile width (WarpTileN), preferred mode, and type properties,
// selects the optimal LDSM mode with graceful fallback:
//   1. If preferred mode's Vlen constraint can't be met (WarpTileN < coop_vlen),
//      fall back from cooperative modes to Vector.
//   2. If UnorderedVector is preferred but thread count < 128, fall back to CoopVector.
//   3. Vlen is capped at WarpTileN so data divides evenly among threads.
template <int BitWidth, int WarpTileN, LDSMMode PreferredMode, bool IsType1, int ThrGroupSize>
struct LdsmModeVlenSelector {
private:
  static constexpr int coop_vlen_   = ldsm_coop_vlen<BitWidth>();
  static constexpr int vec_vlen_    = ldsm_vector_row_vlen<BitWidth, IsType1>();
  static constexpr int total_threads_ = ThrGroupSize * 32;

  static constexpr bool can_use_unordered_ =
    IsType1 && WarpTileN >= coop_vlen_ && total_threads_ % 128 == 0;
  static constexpr bool can_use_coop_ =
    IsType1 && WarpTileN >= coop_vlen_;

public:
  static constexpr LDSMMode mode =
    (PreferredMode == UnorderedVector && can_use_unordered_) ? UnorderedVector :
    ((PreferredMode == UnorderedVector || PreferredMode == CoopVector) && can_use_coop_) ? CoopVector :
    Vector;

  static constexpr int max_vlen =
    (mode == UnorderedVector || mode == CoopVector) ? coop_vlen_ : vec_vlen_;

  static constexpr int vlen = (WarpTileN < max_vlen) ? WarpTileN : max_vlen;

  static_assert(WarpTileN % vlen == 0,
                "WarpTileN must be evenly divisible by selected Vlen for even data distribution");
};

// Helper: construct typed LOAD or STORE matrix op
template <class ValType, bool IsStore, class SLayout,
          LDSMMode Mode, int Vlen, cute::Vecdir Vdir>
CUTE_HOST_DEVICE constexpr auto
make_ldsm_op_type() {
  if constexpr (!IsStore)
    return XE4_LOAD_MATRIX<ValType, SLayout, Mode, Vlen, Vdir>{};
  else
    return XE4_STORE_MATRIX<ValType, SLayout, Mode, Vlen, Vdir>{};
}

// LDSM op auto-selector
//
// Given an SLM layout, selects the optimal LDSM operation type with mode priority:
//   1. UnorderedVector (Cooprow) — highest throughput, requires Type1, thread_count % 128 == 0
//   2. CoopVector (Cooprow)     — ordered cooperative, requires Type1
//   3. Vector (Vrow)            — widest compatibility, supports Type1 and Type2
//
// For Type2 matrices (stride<1> != 1), always falls back to Vector regardless of PreferredMode.
// Vdir is set to Vrow; XE4_LDSTMatrixBase::getVdir() converts to Cooprow for cooperative modes.
//
// SLayout must be a rank-2 static CuTe Layout encoding the SLM tile shape and strides.
// The caller (make_ldsm_copy_C/D) is responsible for extracting rank-2 from higher-rank layouts.
//
// Usage:
//   auto op = ldsm_selector<bf16, false>(smem_layout);              // UnorderedVector (default)
//   auto op = ldsm_selector<bf16, false, CoopVector>(smem_layout);  // CoopVector fallback
//   auto op = ldsm_selector<bf16, true>(smem_layout);               // Store variant
//
template <class ValType, bool IsStore,
          LDSMMode PreferredMode = UnorderedVector, class SLayout>
CUTE_HOST_DEVICE constexpr auto
ldsm_selector(SLayout const&)
{
  constexpr int BitWidth = sizeof_bits_v<ValType>;
  constexpr bool is_type1 = LdsmIsType1<SLayout>::value;

  if constexpr (!is_type1) {
    // Type2: cooperative modes not supported, use Vector
    constexpr int Vlen = ldsm_vector_row_vlen<BitWidth, false>();
    return make_ldsm_op_type<ValType, IsStore, SLayout, Vector, Vlen, cute::Vecdir::Vrow>();
  }
  else if constexpr (PreferredMode == UnorderedVector) {
    // Priority 1: Unordered cooperative row — highest throughput
    constexpr int Vlen = ldsm_coop_vlen<BitWidth>();
    return make_ldsm_op_type<ValType, IsStore, SLayout, UnorderedVector, Vlen, cute::Vecdir::Vrow>();
  }
  else if constexpr (PreferredMode == CoopVector) {
    // Priority 2: Ordered cooperative row
    constexpr int Vlen = ldsm_coop_vlen<BitWidth>();
    return make_ldsm_op_type<ValType, IsStore, SLayout, CoopVector, Vlen, cute::Vecdir::Vrow>();
  }
  else {
    // Priority 3: Standard vector mode
    constexpr int Vlen = ldsm_vector_row_vlen<BitWidth, true>();
    return make_ldsm_op_type<ValType, IsStore, SLayout, Vector, Vlen, cute::Vecdir::Vrow>();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// make_ldsm_copy_C / make_ldsm_copy_D (Module 2)
/// Convenience APIs that auto-select LDSM op and construct a TiledCopy for epilogue S2R / R2S.
/// Follows make_block_2d_copy_C/D pattern from copy_traits_xe_2d.hpp.
/// See MODULE_DESIGN.md - Module 2 for design rationale.
////////////////////////////////////////////////////////////////////////////////////////////////////

// Internal helper: derive LDSM TiledCopy from SLM tensor geometry and thread group size.
//
// Layout derivation steps:
//   1. Extract tile dimensions from SLM tensor layout (NOT from MMA — the SLM
//      tensor may be an EpilogueTile spanning multiple MMA atoms)
//   2. Compute per-warp tile geometry (WarpTileN) from tile dimensions
//   3. Select LDSM mode + Vlen that evenly divides WarpTileN among threads
//   4. Build thread layout (interleaved for UnorderedVector, ordered for others)
//   5. Build value layout = (1, WarpTileN) covering full per-warp N dimension
//
// NOTE: TiledMMA is accepted by the public make_ldsm_copy_C/D wrappers for API
// consistency with make_block_2d_copy_C/D, but is NOT forwarded here. All geometry
// is derived from ThrGroupSize and SLayout.
//
template <bool IsStore,
          int ThrGroupSize,
          LDSMMode PreferredMode,
          class GEngine, class SLayout>
CUTE_HOST_DEVICE
auto
make_ldsm_copy_CD_impl(Tensor<GEngine, SLayout> const& slm_tensor,
                        bool is_B_matrix)
{
  using ValType = typename GEngine::value_type;
  namespace hw = xe4::slm::type1::kmajor;
  constexpr int BitWidth = sizeof_bits_v<ValType>;
  constexpr int NumThreadsPerWarp = hw::kSubGroupSize;  // 32

  // --- Step 1: Extract tile dimensions from SLM tensor layout ---
  // Use SLayout dimensions (from the actual SLM tensor) for per-warp geometry,
  // NOT MMA tile dimensions. The SLM tensor may be an EpilogueTile that spans
  // multiple MMA atoms (e.g., EpilogueTile 64x256 vs MMA atom 32x64).
  constexpr int TileM = size<0>(SLayout{});
  constexpr int TileN = size<1>(SLayout{});

  // --- Step 2: Compute per-warp geometry from tile dimensions ---
  static_assert(TileM % NumThreadsPerWarp == 0,
                "Tile M must be divisible by kSubGroupSize (32)");
  constexpr int NumWarpsAlongM = TileM / NumThreadsPerWarp;

  static_assert(ThrGroupSize % NumWarpsAlongM == 0,
                "ThrGroupSize must be divisible by NumWarpsAlongM");
  constexpr int NumWarpsAlongN = ThrGroupSize / NumWarpsAlongM;

  static_assert(TileN % NumWarpsAlongN == 0,
                "Tile N must be evenly divided among warps along N");
  constexpr int WarpTileN = TileN / NumWarpsAlongN;

  // --- Step 3: Select LDSM mode and Vlen for even data distribution ---
  // Vlen is capped at WarpTileN so each thread's data divides evenly.
  // Mode falls back gracefully if hardware constraints can't be met.
  constexpr bool is_type1 = LdsmIsType1<SLayout>::value;
  using Selector = LdsmModeVlenSelector<BitWidth, WarpTileN, PreferredMode,
                                         is_type1, ThrGroupSize>;
  constexpr LDSMMode EffectiveMode = Selector::mode;
  constexpr int Vlen = Selector::vlen;

  // Representative Type1 layout for op type — simple (32, WarpTileN) LayoutRight.
  // The actual SLM tensor may have a ComposedLayout (with swizzle), but the op type
  // only needs dimensions and stride direction for hardware constraint checking.
  // This mirrors what xe4_get_ldsm_load_op/store_op do in the builder.
  using RepSLayout = decltype(make_layout(
      make_shape(Int<NumThreadsPerWarp>{}, Int<WarpTileN>{}), LayoutRight{}));
  auto op = make_ldsm_op_type<ValType, IsStore, RepSLayout,
                               EffectiveMode, Vlen, cute::Vecdir::Vrow>();

  // --- Step 4: Build thread and value layouts ---
  // Thread layout: interleaved for UnorderedVector when ThrGroupSize ≤ 4 (injective),
  // ordered for larger groups where the interleaved layout becomes non-injective.
  // LdsmThrLayout<32, G, true> has max_index = 95 + 8*G; injective requires G ≤ 4.
  // CuTe's TiledCopy complement() requires injective thread layouts.
  constexpr bool UseInterleaved = (EffectiveMode == UnorderedVector ||
                                   EffectiveMode == UnorderedArrOfVectors) &&
                                  ThrGroupSize <= 4;
  auto thr_layout = LdsmMmaThrLayout<NumWarpsAlongM, NumWarpsAlongN,
                                      UseInterleaved>::t_layout;
  // Value layout: covers full per-warp N dimension. When WarpTileN > Vlen,
  // the TiledCopy automatically tiles multiple atom invocations along N.
  auto val_layout = make_layout(make_shape(Int<1>{}, Int<WarpTileN>{}));

  // --- Step 5: Construct TiledCopy with derived layouts ---
  return make_ldsm_tiled_copy(op, slm_tensor, thr_layout, val_layout, is_B_matrix);
}

// Load from SLM to registers (S2R epilogue path) — geometry-only variant.
// Derives thread/value layouts purely from SLM tensor geometry and ThrGroupSize.
// Does not use TiledMMA information. For MMA-aware layout derivation, use the
// overload that accepts a TiledMMA<Args...> parameter.
template <int ThrGroupSize = 4,
          LDSMMode PreferredMode = UnorderedVector,
          class GEngine, class SLayout>
CUTE_HOST_DEVICE
auto
make_ldsm_copy_C(Tensor<GEngine, SLayout> const& slm_tensor,
                  bool is_B_matrix = true)
{
  return make_ldsm_copy_CD_impl</*IsStore=*/false, ThrGroupSize, PreferredMode>(
      slm_tensor, is_B_matrix);
}

// Store from registers to SLM (R2S epilogue path) — geometry-only variant.
// Derives thread/value layouts purely from SLM tensor geometry and ThrGroupSize.
// Does not use TiledMMA information. For MMA-aware layout derivation, use the
// overload that accepts a TiledMMA<Args...> parameter.
template <int ThrGroupSize = 4,
          LDSMMode PreferredMode = UnorderedVector,
          class GEngine, class SLayout>
CUTE_HOST_DEVICE
auto
make_ldsm_copy_D(Tensor<GEngine, SLayout> const& slm_tensor,
                  bool is_B_matrix = true)
{
  return make_ldsm_copy_CD_impl</*IsStore=*/true, ThrGroupSize, PreferredMode>(
      slm_tensor, is_B_matrix);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// MMA-aware make_ldsm_copy_C / make_ldsm_copy_D
///
/// These overloads derive thread/value layouts from TiledMMA's C-fragment
/// distribution, following the same TV-layout extraction algorithm used by
/// make_block_2d_copy_CD (copy_traits_xe_2d.hpp) and make_adma_atom_A/B_xe4
/// (copy_traits_xe4_adma.hpp).  See XE2DandADMA.md Section 10 and
/// LDSM_New_CDApis.md for design rationale.
///
/// Algorithm summary:
///   1. Extract MMA atom shape for C/D from TiledMMA::AtomShape_MNK
///   2. Compute how many atoms span the SLM tile along M and N
///   3. Derive NumWarpsAlongM/N using the MMA atom distribution, clamped by
///      LDSM's 32-thread warp constraint (each warp occupies 32 rows)
///   4. Select LDSM mode/Vlen and build thread/value layouts from the
///      MMA-derived warp distribution
///
/// Compared to the geometry-only variant (above), this approach:
///   - Respects MMA atom boundaries when distributing warps
///   - Produces identical results for current configurations (backward compat)
///   - Correctly handles future MMA configs where AtomShape_M != 32
////////////////////////////////////////////////////////////////////////////////////////////////////

// Internal helper: derive LDSM TiledCopy from TiledMMA's C-fragment distribution.
template <bool IsStore,
          int ThrGroupSize,
          LDSMMode PreferredMode,
          class... MmaArgs, class GEngine, class SLayout>
CUTE_HOST_DEVICE
auto
make_ldsm_copy_CD_mma_impl(TiledMMA<MmaArgs...> const& mma,
                            Tensor<GEngine, SLayout> const& slm_tensor,
                            bool is_B_matrix)
{
  using MMA = TiledMMA<MmaArgs...>;
  using ValType = typename GEngine::value_type;
  namespace hw = xe4::slm::type1::kmajor;
  constexpr int BitWidth = sizeof_bits_v<ValType>;
  constexpr int NumThreadsPerWarp = hw::kSubGroupSize;  // 32

  // --- Step 1: Extract SLM tile dimensions ---
  constexpr int TileM = size<0>(SLayout{});
  constexpr int TileN = size<1>(SLayout{});

  // --- Step 2: Extract MMA atom shape for C/D (M, N) ---
  constexpr int AtomM = decltype(size<0>(typename MMA::AtomShape_MNK{}))::value;
  constexpr int AtomN = decltype(size<1>(typename MMA::AtomShape_MNK{}))::value;

  // Clamp atom dimensions to the tile. The SLM tile (EpilogueTile) can be a
  // sub-tile of the MMA atom — e.g., EpilogueTile (64,256) within a (256,256)
  // atom. When the atom is larger than the tile, atom boundary alignment is
  // meaningless; we use the tile dimension itself.
  constexpr int EffAtomM = (AtomM <= TileM) ? AtomM : TileM;
  constexpr int slm_atoms_m = TileM / EffAtomM;

  // --- Step 3: Derive warp distribution from MMA atom layout ---
  // LDSM hardware constraint: each warp = 32 threads = 32 rows in M.
  static_assert(TileM % NumThreadsPerWarp == 0,
                "Tile M must be divisible by warp size (32)");
  constexpr int max_warps_m = TileM / NumThreadsPerWarp;

  // MMA-informed distribution:
  //   When EffAtomM > warp size (32), multiple warps necessarily share one atom,
  //   so atom boundary alignment is impossible. Use max_warps_m (geometry-only).
  //   When EffAtomM <= warp size, each warp handles >= 1 whole atom, and we can
  //   align warp boundaries to atom boundaries via min(max_warps_m, slm_atoms_m).
  //   NumWarpsAlongM = gcd(ThrGroupSize, preferred_warps_m)
  //     Ensures NumWarpsAlongM divides ThrGroupSize (integer NumWarpsAlongN)
  //     Picks the largest M-split compatible with both constraints
  constexpr int preferred_warps_m = (EffAtomM > NumThreadsPerWarp)
      ? max_warps_m
      : ((max_warps_m < slm_atoms_m) ? max_warps_m : slm_atoms_m);
  constexpr int NumWarpsAlongM = cute::gcd(ThrGroupSize, preferred_warps_m);

  static_assert(ThrGroupSize % NumWarpsAlongM == 0,
                "ThrGroupSize must be divisible by NumWarpsAlongM");
  constexpr int NumWarpsAlongN = ThrGroupSize / NumWarpsAlongM;

  static_assert(TileN % NumWarpsAlongN == 0,
                "Tile N must be evenly divided among warps along N");
  constexpr int WarpTileN = TileN / NumWarpsAlongN;

  // --- Step 4: Select LDSM mode and Vlen ---
  constexpr bool is_type1 = LdsmIsType1<SLayout>::value;
  using Selector = LdsmModeVlenSelector<BitWidth, WarpTileN, PreferredMode,
                                         is_type1, ThrGroupSize>;
  constexpr LDSMMode EffectiveMode = Selector::mode;
  constexpr int Vlen = Selector::vlen;

  // Representative Type1 layout for op type construction
  using RepSLayout = decltype(make_layout(
      make_shape(Int<NumThreadsPerWarp>{}, Int<WarpTileN>{}), LayoutRight{}));
  auto op = make_ldsm_op_type<ValType, IsStore, RepSLayout,
                               EffectiveMode, Vlen, cute::Vecdir::Vrow>();

  // --- Step 5: Build thread and value layouts ---
  constexpr bool UseInterleaved = (EffectiveMode == UnorderedVector ||
                                   EffectiveMode == UnorderedArrOfVectors) &&
                                  ThrGroupSize <= 4;
  auto thr_layout = LdsmMmaThrLayout<NumWarpsAlongM, NumWarpsAlongN,
                                      UseInterleaved>::t_layout;
  auto val_layout = make_layout(make_shape(Int<1>{}, Int<WarpTileN>{}));

  // --- Step 6: Construct TiledCopy ---
  return make_ldsm_tiled_copy(op, slm_tensor, thr_layout, val_layout, is_B_matrix);
}

// Load from SLM to registers (S2R epilogue path) — MMA-aware variant.
// Derives warp distribution from TiledMMA's C-fragment atom layout, respecting
// MMA atom boundaries when distributing warps across M and N dimensions.
// Follows the make_block_2d_copy_CD pattern from copy_traits_xe_2d.hpp.
template <int ThrGroupSize = 4,
          LDSMMode PreferredMode = UnorderedVector,
          class... MmaArgs, class GEngine, class SLayout>
CUTE_HOST_DEVICE
auto
make_ldsm_copy_C(TiledMMA<MmaArgs...> const& mma,
                  Tensor<GEngine, SLayout> const& slm_tensor,
                  bool is_B_matrix = true)
{
  return make_ldsm_copy_CD_mma_impl</*IsStore=*/false, ThrGroupSize, PreferredMode>(
      mma, slm_tensor, is_B_matrix);
}

// Store from registers to SLM (R2S epilogue path) — MMA-aware variant.
// Derives warp distribution from TiledMMA's C-fragment atom layout, respecting
// MMA atom boundaries when distributing warps across M and N dimensions.
// Follows the make_block_2d_copy_CD pattern from copy_traits_xe_2d.hpp.
template <int ThrGroupSize = 4,
          LDSMMode PreferredMode = UnorderedVector,
          class... MmaArgs, class GEngine, class SLayout>
CUTE_HOST_DEVICE
auto
make_ldsm_copy_D(TiledMMA<MmaArgs...> const& mma,
                  Tensor<GEngine, SLayout> const& slm_tensor,
                  bool is_B_matrix = true)
{
  return make_ldsm_copy_CD_mma_impl</*IsStore=*/true, ThrGroupSize, PreferredMode>(
      mma, slm_tensor, is_B_matrix);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// make_ldsm_copy_warp_row_CD_int  (internal — call via make_ldsm_copy_warp_row_C/D)
///
/// LDSM TiledCopy factory whose distinguishing functional property is that
/// **row-mates (work-items sharing an M-coord) land within a single warp**,
/// regardless of `NumWarps`.  This is what enables within-warp cross-lane
/// reductions (e.g. `fred.max` with mask 0x55555555) on the loaded fragment
/// for any thread-group size.
///
/// Compared with the generic `make_ldsm_copy_CD_(mma_)impl` factories:
///   * Generic factory:  derives the TV layout from `LdsmMmaThrLayout` via
///     `make_tiled_copy`'s `raked_product → right_inverse` chain.  The
///     interleaved variant is non-injective for `ThrGroupSize > 4`, so the
///     factory falls back to the ordered layout where row-mates land
///     **cross-warp**.
///   * `_warp_row` factory:  bypasses `make_tiled_copy` entirely and calls
///     `make_tiled_copy_impl(atom, layout_TV, tiler)` with a hand-rolled
///     bit-decomposed TV layout.  The (T, V) → (M, N) bijection is encoded
///     directly using distinct power-of-2 strides, so every WI's row-mates
///     stay inside one 32-thread warp.
///
/// (T, V) → (M, N) mapping baked into the returned TiledCopy:
///
///     m = wi_lo + RowsPerEu·eu_id + 32·eu_sg_id + RowsPerWi·iter
///     n = wi_hi · NumValPerWIPerIter + atom_val
///
///   where the worker_id bit decomposition is
///     wi_lo    = worker_id  & 1                        size 2
///     wi_hi    = (worker_id / 2) & (NumThreadPerRow-1) size NumThreadPerRow
///     eu_id    = (worker_id / 32) & (EuCount-1)        size EuCount
///     eu_sg_id =  worker_id / 128                      size EuSgCount
///
/// This is a SPARSE iter→row mapping: iter `i` covers exactly
/// `numRowsPerIteration = NumWarps * RowsPerWi` rows, distributed across
/// [0, TileM) at stride 2 (interleaved with wi_lo, eu_id, eu_sg_id bits) —
/// NOT a contiguous M-band.  The matching M-tiler in
/// `detail::make_warp_row_m_tiler_layout` produces this scattered codomain.
///
/// Why this is safe for LDSM atoms specifically:
///   Xe4LDSMTraitsBase has ThrID = Layout<Int<1>>, AtomLayoutRef =
///   Layout<Shape<Int<1>, Int<NumValPerAtom>>>, so AtomNumThr = 1 and
///   right_inverse(AtomLayoutRef).compose(AtomLayoutSrc) = identity.
///   `tidfrg_S/D` then uses the supplied TiledLayout_TV verbatim — no
///   further injectivity constraint is imposed.  The (T,V)→(M,N) bijection
///   we encode is exact because every stride is a distinct power of 2.
///
/// HW path emitted:
///   With UnorderedVector mode the HW issues
///   `ld_matrix.unordered.al<A>.as1.arow.vl<V>.cooprow.<bw>` cohorting
///   128 threads per atom.  When the per-WI value count or the total thread
///   count can't satisfy cooperative-row constraints the selector falls
///   back to per-warp `Vector` mode.  See `detail::LdsmWarpRowModeSelector`.
///
/// API (single function, store/load selected by template bool):
///   auto tc = make_ldsm_copy_warp_row_CD_int<IsStore, NumWarps, EuCount,
///                                            RowsPerWi, PreferredMode>(slm_tensor);
/// where:
///   IsStore       : false → S2R load (XE4_LOAD_MATRIX),
///                   true  → R2S store (XE4_STORE_MATRIX).
///   NumWarps      : ThrGroupSize.  Must be a multiple of EuCount.
///   EuCount       : Xe4 type-1 CM EU count (default 4).
///   RowsPerWi     : per-WI rows owned within a warp = 32/NumThreadPerRow
///                   (default 2).
///   PreferredMode : LDSMMode preference; falls back from UnorderedVector
///                   to Vector when HW constraints can't be met.
///
/// USE
/// ---
///     // Load (S2R) — preferred public form
///     auto tc       = cute::make_ldsm_copy_warp_row_C<NumWarps>(sX_packed);
///     auto coord    = make_identity_tensor(make_shape(Int<TileM>{},
///                                                      Int<TileN_packed>{}));
///     auto thr_part = tc.get_slice(worker_id).partition_S(coord);
///     // thr_part shape = (CPY, CPY_M=TotalRowsPerThread, CPY_N=1).
///     for (int i = 0; i < TotalRowsPerThread; ++i)
///         copy(tc, thr_part(_, i, _0{}), reg_slot_i);
///
///     // Store (R2S) — preferred public form
///     auto tc_store = cute::make_ldsm_copy_warp_row_D<NumWarps>(sX_packed);
///     auto thr_dst  = tc_store.get_slice(worker_id).partition_D(coord);
///     for (int i = 0; i < TotalRowsPerThread; ++i)
///         copy(tc_store, reg_slot_i, thr_dst(_, i, _0{}));
///
/// Mode selection priority:
///   1. UnorderedVector (HW cooperative-row).  Picked when
///      `PreferredMode == UnorderedVector`, `TotalThreads % 128 == 0`, and
///      `NumValPerWIPerIter / coop_vlen<BitWidth>() ∈ {1, 2, 4}`.
///   2. Vector (per-warp, non-cooperative).  Used when (1) can't apply.
///   See `detail::LdsmWarpRowModeSelector` for the full predicate.
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// ---------------------------------------------------------------------------
// LdsmWarpRowModeSelector — pick (mode, Vlen, Alen) for the per-call atom
// ---------------------------------------------------------------------------
//
// Two-priority selector mirroring the legacy `cm_vrow_*_unordered` ↔
// `cm_vrow_*` (ordered) fallback at master_next softmax_epilogue.hpp:738.
//
//   1. UnorderedVector (HW cooperative-row).  Requires:
//        * `PreferredMode == UnorderedVector`.
//        * `TotalThreads % 128 == 0`  (the 128-thread cohort fits whole).
//        * `NumValPerWIPerIter >= coop_vlen` and `% coop_vlen == 0`.
//        * `NumValPerWIPerIter / coop_vlen ∈ {1, 2, 4}`  (HW-legal Alen).
//      Picks `vlen = coop_vlen<BitWidth>` and
//            `alen = NumValPerWIPerIter / coop_vlen`.
//
//   2. Vector (per-warp, non-cooperative).  Used when (1) can't apply.
//      Picks `vlen = min(NumValPerWIPerIter, ldsm_vector_row_vlen<BitWidth,
//      Type1>)` and `alen = 0`.
//
// Concrete decisions for the live FMHA config (PROJECT_SPEC §5.3 table):
//   sS  / sP   : fp16, NEPT=32, coop_vlen<16>=16  → UnorderedVector, vlen=16, alen=2
//   sOacc      : fp32, NEPT=8,  coop_vlen<32>=8   → UnorderedVector, vlen=8,  alen=1
//   sO  (FP16) : fp16, NEPT=8 < 16                → Vector,          vlen=8,  alen=0
//   sO  (FP32) : fp32, NEPT=8                     → UnorderedVector, vlen=8,  alen=1
//
// (The same table is replayed in `make_warp_row_tv_layout`'s comment block.)
template <int BitWidth, int NumValPerWIPerIter, LDSMMode PreferredMode,
          int TotalThreads>
struct LdsmWarpRowModeSelector {
private:
  static constexpr int coop_vlen_ = ldsm_coop_vlen<BitWidth>();
  static constexpr int vec_vlen_  = ldsm_vector_row_vlen<BitWidth, true>();
  static constexpr bool can_unordered_ =
      (PreferredMode == UnorderedVector) &&
      (TotalThreads % 128 == 0) &&
      (NumValPerWIPerIter >= coop_vlen_) &&
      (NumValPerWIPerIter % coop_vlen_ == 0) &&
      (NumValPerWIPerIter / coop_vlen_ == 1 ||
       NumValPerWIPerIter / coop_vlen_ == 2 ||
       NumValPerWIPerIter / coop_vlen_ == 4);
public:
  static constexpr LDSMMode mode = can_unordered_ ? UnorderedVector : Vector;
  static constexpr int      vlen =
      can_unordered_ ? coop_vlen_
                     : ((NumValPerWIPerIter < vec_vlen_) ? NumValPerWIPerIter
                                                         : vec_vlen_);
  static constexpr int      alen = can_unordered_ ? (NumValPerWIPerIter / coop_vlen_) : 0;
};

// ---------------------------------------------------------------------------
// make_warp_row_tv_layout — TV (thread, value) → tile-block-linear mapping
// ---------------------------------------------------------------------------
//
// CONTEXT
// -------
// `make_tiled_copy_impl(atom, layout_TV, tiler)` builds a TiledCopy whose
// `partition_S(stensor)` resolves
//
//     (T, V, RestM, RestN) →  original (m, n)
//
// via two compositions:
//   (a)  `zipped_divide(stensor, tiler)` produces a layout
//             ((TileBlock_M, TileBlock_N), (RestM, RestN)) → (m, n)
//        where the inside of the first mode is the tile_block layout (the
//        per-call atom block) — its strides come from the *tilers* themselves.
//   (b)  the TV layout maps `(T, V) → tile_block_linear` and is composed
//        with the tile_block to give `(T, V) → (m, n)`.
//
// **Crucial subtlety (B4d in FINDINGS.md §7).**  The TV layout's linear value
// is consumed *as a tile-block-linear index* — not as an original-tile linear
// index.  That tile-block layout is **column-major by default** in CuTe.
//
// Within-warp-row formula (mirrors the legacy cm_vrow_*_unordered shape):
//
//     m = wi_lo + RowsPerEu·eu_id + 32·eu_sg_id   + RowsPerWi·iter
//     n = wi_hi · NumValPerWIPerIter + atom_val
//
// where `worker_id` decomposes as
//     wi_lo    = worker_id  & 1                        size 2
//     wi_hi    = (worker_id / 2) & (NumThreadPerRow-1) size NumThreadPerRow
//     eu_id    = (worker_id / 32) & (EuCount-1)        size EuCount
//     eu_sg_id =  worker_id / 128                      size EuSgCount
//
// TV-LAYOUT DERIVATION (TILE-BLOCK-LINEAR)
// ----------------------------------------
// The m-tiler is `Layout<Shape<_RowsPerWi,_EuCount,_EuSgCount>,
//                        Stride<_1,_RowsPerEu,_32>>` (codomain = warp-row
// scatter pattern).  The n-tiler is `Layout<_TileN, _1>` (single contiguous
// N-block per call).  The tile_block layout is therefore
//
//     ((RowsPerWi, EuCount, EuSgCount), TileN_per_call)
//
// with **column-major** linear strides
//
//     ( (1, RowsPerWi, RowsPerWi·EuCount), RowsPerWi·EuCount·EuSgCount )
//   = ( (1, 2, 8),                          32 )                            .
//
// To make T-bit `b` advance the tile_block coord by the right amount we set
// the stride in tile-block linear to the corresponding tile_block stride:
//
//    | T/V bit  | T-stride | size            | tile-block sub-mode advance | tile-block-linear stride |
//    |----------|---------:|----------------:|----------------------------:|------------------------:|
//    | wi_lo    |        1 | RowsPerWi=2     | sub_a += 1                  | 1                       |
//    | wi_hi    |        2 | NumThrPerRow=16 | n_inner += NumVal           | 32 · NumValPerWIPerIter |
//    | eu_id    |       32 | EuCount=4       | sub_b += 1                  | RowsPerWi=2             |
//    | eu_sg_id |      128 | EuSgCount=4     | sub_c += 1                  | RowsPerWi·EuCount=8     |
//    | atom_val |      V=1 | NumValPerWIPerIter | n_inner += 1             | 32                      |
//
// Substituting RowsPerWi=2, EuCount=4, EuSgCount=4, NumValPerWIPerIter=16:
// the tile-block-linear strides resolve to (1, 32·16=512, 2, 8) for T-modes
// and 32 for V — so the linear value computed for any worker_id reproduces
// the within-warp-row `(m, n)` after composition with the tile_block layout.
//
// HW COHORT GEOMETRY
// ------------------
// The 128 threads with the same eu_sg_id together populate rows
// [eu_sg_id·32, eu_sg_id·32 + 32) — exactly one type-1 CM block per cohort —
// which is the geometry the `unordered.cooprow.<bw>` instruction expects.
//
// CONSTRUCTION-TIME PARAMETERS
// ----------------------------
// `RowsPerWi`         : per-WI rows owned within a warp (2 for FMHA softmax).
// `NumThreadPerRow`   : warp lanes per row (= 32/RowsPerWi = 16 for FMHA).
// `EuCount`, `EuSgCount` : type-1 CM cohort decomposition (4 × 4 for FMHA).
// `NumValPerWIPerIter`: per-WI columns per copy() call (uint32 view).
// The last template parameter is unused — kept for API stability with the
// earlier (TileN-row-major) prototype; do not rely on it.
template <int RowsPerWi, int NumThreadPerRow, int EuCount, int EuSgCount,
          int NumValPerWIPerIter, int /*unused — kept for API stability*/>
CUTE_HOST_DEVICE constexpr auto
make_warp_row_tv_layout() {
  // Tile-block has RowsPerWi*EuCount*EuSgCount M-cells.  The N stride per
  // single n_inner step in tile-block linear equals that count (column-major).
  constexpr int kMCellsPerCall = RowsPerWi * EuCount * EuSgCount;  // 32 for FMHA
  constexpr int wi_lo_stride    = 1;
  constexpr int eu_id_stride    = RowsPerWi;                       // 2
  constexpr int eu_sg_id_stride = RowsPerWi * EuCount;             // 8
  constexpr int atom_val_stride = kMCellsPerCall;                  // 32
  constexpr int wi_hi_stride    = atom_val_stride * NumValPerWIPerIter; // 32·NumVal
  return cute::make_layout(
      cute::make_shape(
          cute::make_shape(cute::Int<RowsPerWi>{},
                           cute::Int<NumThreadPerRow>{},
                           cute::Int<EuCount>{},
                           cute::Int<EuSgCount>{}),
          cute::Int<NumValPerWIPerIter>{}),
      cute::make_stride(
          cute::make_stride(cute::Int<wi_lo_stride>{},
                            cute::Int<wi_hi_stride>{},
                            cute::Int<eu_id_stride>{},
                            cute::Int<eu_sg_id_stride>{}),
          cute::Int<atom_val_stride>{}));
}

// ---------------------------------------------------------------------------
// make_warp_row_m_tiler_layout — interspersed-row M-tiler for the per-call atom
// ---------------------------------------------------------------------------
//
// PURPOSE
// -------
// The within-warp-row iter→row mapping is **non-contiguous**: iter `i`
// covers rows
//
//     {wi_lo + RowsPerEu·eu_id + 32·eu_sg_id + RowsPerWi·i}
//      = { 0,1, 8,9, 16,17, 24,25, 32,33, ..., 120,121 }   // for i=0
//      = { 2,3, 10,11, ..., 122,123 }                       // for i=1
//      = ...
//
// `Tiler_MN`'s M-mode must therefore be a **Layout** (not a Shape) whose
// codomain is the per-call scatter set, so that
// `complement(m_tiler, TileM)` is the iter axis at row-stride RowsPerWi.
//
// LAYOUT
// ------
// Returned shape:    `Layout<Shape<RowsPerWi, EuCount, EuSgCount>,
//                            Stride<_1,        kRowsPerEu, _32>>`.
// For FMHA (RowsPerWi=2, EuCount=4, EuSgCount=4):
//     shape  = (2, 4, 4)        size 32
//     stride = (1, 8, 32)       cosize = 1+24+96+1 = 122
//     codomain ⊂ [0, 122) ⊂ [0, 128=TileM) — bijective with the per-call rows.
// `complement(m_tiler, _128) = Layout<_4, _2>` — the 4 iters at row-stride 2.
//
// USAGE
// -----
// Combined with a flat n-tiler `Layout<_TileN, _1>` via `make_tile`, this is
// passed as `Tiler_MN` to `make_tiled_copy_impl(atom, tv_layout, tiler)`.
// `partition_S(make_identity_tensor((TileM, TileN)))` then yields
//     (CPY = NumValPerWIPerIter, CPY_M = TotalRowsPerThread, CPY_N = 1)
// with the CPY_M axis indexing the iter at the within-warp-row row-stride.
template <int RowsPerWi, int EuCount, int EuSgCount>
CUTE_HOST_DEVICE constexpr auto
make_warp_row_m_tiler_layout() {
  constexpr int kRowsPerEu = 32 / EuCount;  // 8 for EuCount=4
  return cute::make_layout(
      cute::make_shape(cute::Int<RowsPerWi>{},
                       cute::Int<EuCount>{},
                       cute::Int<EuSgCount>{}),
      cute::make_stride(cute::Int<1>{},
                        cute::Int<kRowsPerEu>{},
                        cute::Int<32>{}));
}

}  // namespace detail

// ---------------------------------------------------------------------------
// make_ldsm_copy_warp_row_CD_int — internal TiledCopy factory whose row-mates
// stay within a single warp (S2R load when IsStore=false, R2S store when
// IsStore=true).  Public callers should use the make_ldsm_copy_warp_row_C
// (load) / make_ldsm_copy_warp_row_D (store) wrappers below.
// ---------------------------------------------------------------------------
//
// Builds a `TiledCopy` with a hand-rolled (T, V) → (M, N) mapping
//
//     m = wi_lo + RowsPerEu·eu_id + 32·eu_sg_id + RowsPerWi·iter
//     n = wi_hi · NumValPerWIPerIter + atom_val
//
// — guaranteeing that 'row-mates' (threads that share an `m` after a row
// reduction) live in the **same warp**, which is the precondition for
// within-warp cross-lane reductions (e.g. FMHA softmax `fred.max` with
// mask 0x55555555).
//
// Why this exists vs. `make_ldsm_copy_C/D`:
//   `make_ldsm_copy_C/D` derives its thread layout via
//   `LdsmMmaThrLayout<NumWarpsAlongM, NumWarpsAlongN, IsUnordered>`. For
//   `ThrGroupSize > 4`, the interleaved variant is non-injective and
//   `make_tiled_copy::right_inverse` rejects it; the factory falls back to
//   the ordered layout where row-mates land cross-warp, breaking
//   within-warp lane reductions.  This factory bypasses that path by
//   calling `make_tiled_copy_impl(atom, tv_layout, tiler)` directly with a
//   hand-rolled TV layout (see `detail::make_warp_row_tv_layout`).
//
// CALLER CONTRACT
// ---------------
//   * `slm_tensor` should be the per-stage SLM tile, **recast to a 4-byte
//     element type** (e.g. uint32) when the underlying data is sub-32-bit.
//     This makes `make_ldsm_matrix_descriptor`'s `Pitch = stride>>2`
//     produce HW-correct units.  For fp32 SLM no recast is needed.
//   * `is_B_matrix = true` (default) for row-major SLM (Type1).
//
// USE
// ---
//     // Load — preferred public form
//     auto tc       = cute::make_ldsm_copy_warp_row_C<NumWarps>(sX_packed);
//     auto coord    = make_identity_tensor(make_shape(Int<TileM>{},
//                                                      Int<TileN_packed>{}));
//     auto thr_part = tc.get_slice(worker_id).partition_S(coord);
//     // thr_part shape = (CPY, CPY_M=TotalRowsPerThread, CPY_N=1).
//     for (int i = 0; i < TotalRowsPerThread; ++i)
//         copy(tc, thr_part(_, i, _0{}), reg_slot_i);
//
//     // Store — preferred public form
//     auto tc_store = cute::make_ldsm_copy_warp_row_D<NumWarps>(sX_packed);
//     auto thr_dst  = tc_store.get_slice(worker_id).partition_D(coord);
//     for (int i = 0; i < TotalRowsPerThread; ++i)
//         copy(tc_store, reg_slot_i, thr_dst(_, i, _0{}));
//
// TEMPLATE PARAMETERS
// -------------------
//   IsStore       : false → S2R load (XE4_LOAD_MATRIX),
//                   true  → R2S store (XE4_STORE_MATRIX).
//   NumWarps      : sub-group warp count = ThrGroupSize.
//                   Must be a multiple of EuCount.
//   EuCount       : type-1 CM cohort size in EUs (default 4 for Xe4).
//   RowsPerWi     : per-WI rows owned within a warp = NumThrPerWarp/NumThreadPerRow
//                   (default 2).
//   PreferredMode : preferred LDSMMode; falls back to Vector if cooperative
//                   constraints (NumValPerWIPerIter ≥ coop_vlen, etc.)
//                   aren't met.  See `detail::LdsmWarpRowModeSelector`.
template <bool IsStore,
          int NumWarps,
          int EuCount   = 4,
          int RowsPerWi = 2,
          LDSMMode PreferredMode = UnorderedVector,
          class GEngine,
          class SLayout>
CUTE_HOST_DEVICE
auto
make_ldsm_copy_warp_row_CD_int(Tensor<GEngine, SLayout> const& slm_tensor,
                               bool is_B_matrix = true)
{
  using ValType = typename GEngine::value_type;
  constexpr int BitWidth         = sizeof_bits_v<ValType>;
  constexpr int NumThreadsPerWarp= 32;
  constexpr int TotalThreads     = NumWarps * NumThreadsPerWarp;
  static_assert(NumWarps % EuCount == 0,
                "NumWarps must be a multiple of EuCount for cooperative-row layout");
  constexpr int EuSgCount        = NumWarps / EuCount;
  constexpr int NumThreadPerRow  = NumThreadsPerWarp / RowsPerWi;
  constexpr int TileM            = cute::size<0>(SLayout{});
  constexpr int TileN            = cute::size<1>(SLayout{});
  static_assert(TileN % NumThreadPerRow == 0,
                "TileN must be divisible by NumThreadPerRow");
  constexpr int NumValPerWIPerIter = TileN / NumThreadPerRow;
  constexpr int numRowsPerIter   = RowsPerWi * NumWarps;
  static_assert(TileM % numRowsPerIter == 0,
                "TileM must be divisible by RowsPerWi*NumWarps (numRowsPerIter)");
  constexpr int TotalRowsPerThread = TileM / numRowsPerIter;

  (void)TotalRowsPerThread;
  using Selector = detail::LdsmWarpRowModeSelector<
      BitWidth, NumValPerWIPerIter, PreferredMode, TotalThreads>;
  constexpr LDSMMode Mode = Selector::mode;
  constexpr int      Vlen = Selector::vlen;
  constexpr int      Alen = Selector::alen;

  using RepSLayout = decltype(cute::make_layout(
      cute::make_shape(cute::Int<numRowsPerIter>{}, cute::Int<TileN>{}),
      cute::LayoutRight{}));

  // Pick load vs. store op (with cooperative-array form when Mode admits it).
  using OpCoop = std::conditional_t<
      IsStore,
      XE4_STORE_MATRIX<ValType, RepSLayout, Mode, Vlen, cute::Vecdir::Vrow,
                       Alen, cute::Arrdir::Arow>,
      XE4_LOAD_MATRIX<ValType, RepSLayout, Mode, Vlen, cute::Vecdir::Vrow,
                      Alen, cute::Arrdir::Arow>>;
  using OpVec = std::conditional_t<
      IsStore,
      XE4_STORE_MATRIX<ValType, RepSLayout, Mode, Vlen, cute::Vecdir::Vrow>,
      XE4_LOAD_MATRIX <ValType, RepSLayout, Mode, Vlen, cute::Vecdir::Vrow>>;
  using Op = std::conditional_t<
      (Mode == UnorderedVector || Mode == CoopVector), OpCoop, OpVec>;

  // Per-call TV layout encoding the within-warp-row coord formula.
  auto tv_layout = detail::make_warp_row_tv_layout<
      RowsPerWi, NumThreadPerRow, EuCount, EuSgCount,
      NumValPerWIPerIter, TileM>();
  // Tiler_MN: M is the interspersed-row layout
  // (m = wi_lo + 8*eu_id + 32*eu_sg_id), N is the full-N range as one block.
  // The complement of the M-tiler in (TileM, 1) is the iter axis with
  // row-stride RowsPerWi — partition_S/D then exposes that as the outer
  // CPY_M axis sized TotalRowsPerThread, matching `thr(_, i, _0{})` indexing.
  auto m_tiler = detail::make_warp_row_m_tiler_layout<
      RowsPerWi, EuCount, EuSgCount>();
  auto n_tiler = cute::make_layout(cute::Int<TileN>{}, cute::_1{});
  auto tiler   = cute::make_tile(m_tiler, n_tiler);

  // Defer to the public make_ldsm_tiled_copy_noninjective helper so that consumers
  // building their own non-injective TV layouts can use the same path.
  return make_ldsm_tiled_copy_noninjective(Op{}, slm_tensor, tv_layout, tiler, is_B_matrix);
}

// ---------------------------------------------------------------------------
// make_ldsm_copy_warp_row_C — public S2R (load) wrapper
// ---------------------------------------------------------------------------
// Thin wrapper over make_ldsm_copy_warp_row_CD_int<IsStore=false,...> that
// builds a within-warp-row LDSM TiledCopy for the load (S2R) direction.
// See the make_ldsm_copy_warp_row_CD_int comment block above for the full
// (T, V) → (M, N) mapping, mode-selection, and caller contract.
template <int NumWarps,
          int EuCount   = 4,
          int RowsPerWi = 2,
          LDSMMode PreferredMode = UnorderedVector,
          class GEngine,
          class SLayout>
CUTE_HOST_DEVICE
auto
make_ldsm_copy_warp_row_C(Tensor<GEngine, SLayout> const& slm_tensor,
                           bool is_B_matrix = true)
{
  return make_ldsm_copy_warp_row_CD_int<
      /*IsStore=*/false, NumWarps, EuCount, RowsPerWi, PreferredMode>(
      slm_tensor, is_B_matrix);
}

// ---------------------------------------------------------------------------
// make_ldsm_copy_warp_row_D — public R2S (store) wrapper
// ---------------------------------------------------------------------------
// Thin wrapper over make_ldsm_copy_warp_row_CD_int<IsStore=true,...> that
// builds a within-warp-row LDSM TiledCopy for the store (R2S) direction.
// See the make_ldsm_copy_warp_row_CD_int comment block above for the full
// (T, V) → (M, N) mapping, mode-selection, and caller contract.
template <int NumWarps,
          int EuCount   = 4,
          int RowsPerWi = 2,
          LDSMMode PreferredMode = UnorderedVector,
          class GEngine,
          class SLayout>
CUTE_HOST_DEVICE
auto
make_ldsm_copy_warp_row_D(Tensor<GEngine, SLayout> const& slm_tensor,
                           bool is_B_matrix = true)
{
  return make_ldsm_copy_warp_row_CD_int<
      /*IsStore=*/true, NumWarps, EuCount, RowsPerWi, PreferredMode>(
      slm_tensor, is_B_matrix);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Type traits for detecting LDSM descriptor-based ops (Module 3)
///
/// The epilogue has two code paths for SLM↔Register copies:
///   1. Legacy path: XE4_LDSM / XE4_STSM — pointer-based vector loads/stores
///   2. Descriptor path: XE4_LOAD_MATRIX / XE4_STORE_MATRIX — hardware matrix descriptor ops
///
/// These traits let the epilogue (xe4_epilogue_adma_warpspecialized.hpp) use
/// `if constexpr (is_ldsm_load_matrix_v<CopyOp>)` to dispatch to the correct path.
/// The builder (xe4_builder.inl) controls which CopyOp type is selected:
///   - Xe4AdmaBuilderImpl uses legacy ops (xe4_get_smem_load_op)
///   - Xe4LdsmAdmaBuilderImpl uses descriptor ops (xe4_get_ldsm_load_op)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class CopyOp>
struct is_ldsm_load_matrix : false_type {};

template <class T, class SL, LDSMMode M, uint32_t V, Vecdir VD, uint32_t A, Arrdir AD>
struct is_ldsm_load_matrix<XE4_LOAD_MATRIX<T, SL, M, V, VD, A, AD>> : true_type {};

template <class CopyOp>
struct is_ldsm_store_matrix : false_type {};

template <class T, class SL, LDSMMode M, uint32_t V, Vecdir VD, uint32_t A, Arrdir AD>
struct is_ldsm_store_matrix<XE4_STORE_MATRIX<T, SL, M, V, VD, A, AD>> : true_type {};

template <class CopyOp>
inline constexpr bool is_ldsm_load_matrix_v = is_ldsm_load_matrix<CopyOp>::value;

template <class CopyOp>
inline constexpr bool is_ldsm_store_matrix_v = is_ldsm_store_matrix<CopyOp>::value;

// Convenience: true if the CopyOp is any descriptor-based LDSM operation (load or store)
template <class CopyOp>
inline constexpr bool is_ldsm_descriptor_op_v =
    is_ldsm_load_matrix_v<CopyOp> || is_ldsm_store_matrix_v<CopyOp>;

}
