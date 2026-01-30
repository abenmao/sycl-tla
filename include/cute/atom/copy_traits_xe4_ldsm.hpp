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
#include <cute/atom/copy_traits_xe4_adma.hpp>

namespace cute {

struct MInfo {
  MInfo() = default;

  MInfo(const MatrixDescriptor mdesc) : matrix_desc_(mdesc.raw_) {}
  uint32_t matrix_desc_;
};


using namespace cute::detail;
template<class OP> 
struct Xe4LDSMTraitsBase {
    using Op = OP;
    static constexpr auto Group_Size = (Op::Super::Coop) ? (Op::BitWidth==4 ? 4 :2): 1;

    /*
     /For Matrix Type 1
     BitWidth = 16
     Vlen =1 (1, 16*1) (16*1, 1)
     Vlen =2 (1, 16*2) (16*2, 1)
    */
    using SrcLayout = Layout<Shape<Int<Group_Size>, Int<Op::BitWidth * Op::Vlen>>,
                          typename std::conditional_t<
		              (Op::Type == MatrixType::Type1), // For Matrix Type1
		              Stride<Int<Op::BitWidth * Op::Vlen>, _1>,
		              Stride<_1, Int<Op::BitWidth * Op::Vlen>> 
			  >
                      >;
    using DstLayout = SrcLayout; 
    using RefLayout = DstLayout;  
    // using ThrID = Layout<intel::_SGSize>;
    // using ThrID = Layout<_1>;
    using ThrID = Layout<Int<Group_Size>>;

    static constexpr int ValBits = Op::BitWidth;
    static_assert(Op::Super::CopyBitsPerThread % ValBits == 0, "Type is incompatible with this copy atom");

    CUTE_HOST_DEVICE
    Xe4LDSMTraitsBase() {}

};

sycl::marray<uint16_t, 2>
swap_coord_for_ldsm(cute::ArithmeticTuple<int, int> const& t) {
  // Element access via get<>() 
  uint16_t x = static_cast<uint16_t>(cute::get<1>(t));
  uint16_t y = static_cast<uint16_t>(cute::get<0>(t));

  // marray can be list-initialized from scalars
  return sycl::marray<uint16_t, 2>{x, y};
}
sycl::marray<uint16_t, 2>
swap_coord_for_ldsm(cute::ArithmeticTuple<int, cute::C<0>> const& t) {
  // Element access via get<>() just like std::tuple. [web:17][web:42]
  uint16_t x = 0;
  uint16_t y = static_cast<uint16_t>(cute::get<0>(t));

  // marray can be list-initialized from scalars. [web:74]
  return sycl::marray<uint16_t, 2>{x, y};
}

template<typename T, class SrLayout, LDSMMode Mode, uint32_t Vlen,
         cute::Vecdir Vdir, class MatInfo>
struct Copy_Traits<XE4_LOAD_MATRIX<T, SrLayout, Mode, Vlen, Vdir>, MatInfo> 
       : Xe4LDSMTraitsBase<XE4_LOAD_MATRIX<T, SrLayout, Mode, Vlen, Vdir>> {

    using Op = XE4_LOAD_MATRIX<T, SrLayout, Mode, Vlen, Vdir>;
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
      auto asd = flatten_to_tuple(t);
      return swap_coord_for_ldsm(flatten_to_tuple(t));
    };
    //auto src_coord = as_xe4_coord(take<0,1>(dst(Int<0>{}))); //src(Int<0>{}));
    auto coord = as_xe4_coord(src.data().coord_);
    Op::copy(dst.data(), traits.cache_.matrix_desc_, coord);
  }
};

template<typename T, class SrLayout, LDSMMode Mode, uint32_t Vlen,
         cute::Vecdir Vdir, class MatInfo>
struct Copy_Traits<XE4_STORE_MATRIX<T, SrLayout, Mode, Vlen, Vdir>, MatInfo> 
       : Xe4LDSMTraitsBase<XE4_STORE_MATRIX<T, SrLayout, Mode, Vlen, Vdir>> {

  using Op= XE4_STORE_MATRIX<T, SrLayout, Mode, Vlen, Vdir>;
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

    // NEED To call righ Copy here
    //Op::copy(traits.payload, recast_ptr<int_byte_t<bits_to_bytes(Super::ValBits)>>(&*dst.data()));
    auto as_xe4_coord = [](auto const& t) {
      return swap_coord_for_ldsm(flatten_to_tuple(t));
    };
    //auto src_coord = as_xe4_coord(take<0,1>(dst(Int<0>{}))); //src(Int<0>{}));
    auto coord = as_xe4_coord(dst.data().coord_);
    //Op::copy(traits.cache_.matrix_desc_, src.data(), coord);
    Op::copy(src.data(), traits.cache_.matrix_desc_, coord);
  }
};


template <class GEngine, class SLayout>
CUTE_HOST
auto
make_ldsm_matrix_descriptor(
    Tensor<GEngine,SLayout> const& stensor,
    bool is_B_matrix =true
) {
  auto slayout =layout(stensor);
  MatrixDescriptor matrix_desc{};
  if constexpr(decltype(rank(flatten(slayout.shape())))::value > 2)
    matrix_desc = make_matrix_descriptor(coalesce(slayout), !is_B_matrix);
  else
    matrix_desc = make_matrix_descriptor(slayout, !is_B_matrix);

  //auto src_ptr = recast_ptr<ValType>(&stensor.data());
  matrix_desc.StartAddress = static_cast<uint32_t>(
      reinterpret_cast<uint64_t>(slm_space_cast(&*stensor.data()))) >> 9;

  return matrix_desc;
}

template<int Vlen, cute::Vecdir Vdir> struct LdsmLayout { 
   static constexpr Layout v_layout = make_layout(Shape(Int<1>{}, Int<Vlen>{}));
};
template<int Vlen> struct LdsmLayout<Vlen, cute::Vecdir::Vrow> {
   static constexpr Layout v_layout = make_layout(Shape(Int<1>{}, Int<Vlen>{}));
};
template<int Vlen> struct LdsmLayout<Vlen, cute::Vecdir::Vcol> {
   static constexpr Layout v_layout = make_layout(Shape(Int<Vlen>{}, Int<1>{}));
};

template <class CopyOp, 
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
  using RefLayout = typename Traits::RefLayout;
  constexpr auto ThCount = get<0>(ThrLayout{}.shape());
  auto t_layout = make_layout(Shape(Int<ThCount>{}, Int<1>{}));
  auto v_layout = LdsmLayout<CopyOp::Vlen, CopyOp::Vdir>::v_layout;

  //static_assert(CopyOp::Vdir == cute::Vecdir::Vrow || CopyOp::Vdir == cute::Vecdir::Vcol);
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
    using  Traits = Copy_Traits<CopyOp, MInfo>;
    using  Atom = Copy_Atom<Traits, ValType>;
    Traits traits{matrix_desc};
    Copy_Atom atom = Atom{traits};
    auto tiled_copy = make_tiled_copy(atom, t_layout, v_layout);
    return tiled_copy;
}

}
