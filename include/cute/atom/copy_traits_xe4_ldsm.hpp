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
