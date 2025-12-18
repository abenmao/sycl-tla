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

#include <cute/arch/mma_xe.hpp>
#include <cute/atom/mma_traits.hpp>
#include "cute/arch/util.hpp"

#include <cute/layout.hpp>

namespace cute
{

namespace detail
{

template <typename ValType, typename LayoutIn>
CUTE_HOST_DEVICE
constexpr auto
wi_interleave(LayoutIn const&)
{
  using namespace intel;
  constexpr LayoutIn layout{};
  constexpr int per_byte = ceil_div(8, sizeof_bits_v<ValType>);
  constexpr int vals = ceil_div(size(layout), sg_size);
  auto tv_interleaved = Layout<Shape<_16,          Shape<C<per_byte>, C<vals/per_byte>>>,
                              Stride<C<per_byte>, Stride<_1,          C<sg_size*per_byte>>>>{};
  return coalesce(composition(layout, tv_interleaved), Step<_1,_1>{});
}

template <typename ValType, typename LayoutIn>
using wi_interleave_t = remove_cvref_t<decltype(wi_interleave<ValType>(LayoutIn{}))>;

} // end namespace detail


template <int M, typename TD, typename TA, typename TB, typename TC>
struct MMA_Traits<XE_DPAS_TT<M, TD, TA, TB, TC>>
{
  using Op = XE_DPAS_TT<M, TD, TA, TB, TC>;

  static constexpr int BV = 32 / sizeof_bits_v<TB>;
  static constexpr int K = Op::K;

  using ValTypeD = TD;
  using ValTypeA = TA;
  using ValTypeB = TB;
  using ValTypeC = TC;
  using _M = Int<M>;
  using _K = Int<K>;

  using Shape_MNK = Shape<_M, _16, _K>;
  using ThrID = Layout<intel::_SGSize>;

  // A layout: (T,V) -> (M,K)
  //   M x K row major, work-items interleaved.
  using ALayout = detail::wi_interleave_t<TA, Layout<Shape<_K, _M>, Stride<_M, _1>>>;

  // B layout: (T,V) -> (N,K)
  //   K x 16 VNNI-transformed row major, work-items interleaved.
  using BLayout = detail::wi_interleave_t<TB, Layout<Shape<Int<BV>, _16, Int<K/BV>>,
                                                     Stride<_16,    _1,  Int<16*BV>>>>;

  // C layout: (T,V) -> (M,N)
  //   M x 16 row major, work-items interleaved.
  using CLayout = Layout<Shape<_16, _M>, Stride<_M, _1>>;
};

template <int M, typename TD, typename TA, typename TB, typename TC>
struct MMA_Traits<XE_BDPAS_TT<M, TD, TA, TB, TC>> : public MMA_Traits<XE_DPAS_TT<M, TD, TA, TB, TC>>
{
  using MMAOp = XE_BDPAS_TT<M, TD, TA, TB, TC>;

  template <class TD1, class DLayout,
            class TA1, class ALayout,
            class TB1, class BLayout,
            class TC1, class CLayout>
  CUTE_DEVICE friend void
  mma_unpack(MMA_Traits<MMAOp>    const& traits,
            Tensor<TD1, DLayout>      & D,
            Tensor<TA1, ALayout> const& A_zipped,
            Tensor<TB1, BLayout> const& B_zipped,
            Tensor<TC1, CLayout> const& C)
  {
    static_assert(is_rmem<TD>::value, "Expected registers in MMA_Atom::call");
    static_assert(is_rmem<TA>::value, "Expected registers in MMA_Atom::call");
    static_assert(is_rmem<TB>::value, "Expected registers in MMA_Atom::call");
    static_assert(is_rmem<TC>::value, "Expected registers in MMA_Atom::call");

    // Register value types from the MMA_Operation register arrays
    using          RegTypeD = typename remove_extent<typename MMAOp::DRegisters>::type;
    using          RegTypeA = typename remove_extent<typename MMAOp::ARegisters>::type;
    using          RegTypeB = typename remove_extent<typename MMAOp::BRegisters>::type;
    using          RegTypeC = typename remove_extent<typename MMAOp::CRegisters>::type;

    constexpr int   RegNumD = extent<typename MMAOp::DRegisters>::value;
    constexpr int   RegNumA = extent<typename MMAOp::ARegisters>::value;
    constexpr int   RegNumB = extent<typename MMAOp::BRegisters>::value;
    constexpr int   RegNumC = extent<typename MMAOp::CRegisters>::value;

    auto  [A, SFA, SFA_M_OFFSET, SFA_K_OFFSET] = unzip_tensor(A_zipped);
    auto  [B, SFB, SFB_N_OFFSET, SFB_K_OFFSET] = unzip_tensor(B_zipped);

    using          RegTypeSFA = typename decltype(SFA)::value_type;
    using          RegTypeSFB = typename decltype(SFB)::value_type;

    Tensor rA = recast<RegTypeA>(A);
    Tensor rB = recast<RegTypeB>(B);
    CUTE_STATIC_ASSERT_V(size(rA) == Int<RegNumA>{});
    CUTE_STATIC_ASSERT_V(size(rB) == Int<RegNumB>{});

    Tensor rD = recast<RegTypeD>(D);
    Tensor rC = recast<RegTypeC>(C);
    CUTE_STATIC_ASSERT_V(size(rD) == Int<RegNumD>{});
    CUTE_STATIC_ASSERT_V(size(rC) == Int<RegNumC>{});

    auto sfa_offset = SFA_M_OFFSET[0] + SFA_K_OFFSET[0];
    auto sfb_offset = SFB_N_OFFSET[0] + SFB_K_OFFSET[0];

    if constexpr (sizeof_bits_v<typename MMAOp::AType> < 8) {
      constexpr auto scaleA_size = 3;
      constexpr auto scaleB_size = 3;

      auto tensor_sfa = make_tensor<cutlass::float_ue8m0_t>(Shape<Int<scaleA_size>>{});
      auto tensor_sfb = make_tensor<cutlass::float_ue8m0_t>(Shape<Int<scaleB_size>>{});

      auto rSFA = make_tensor(recast<intel::vector_t<cutlass::float_ue8m0_t, scaleA_size>>(tensor_sfa).data(), Shape<_1>{});
      auto rSFB = make_tensor(recast<intel::vector_t<cutlass::float_ue8m0_t, scaleB_size>>(tensor_sfb).data(), Shape<_1>{});

      #if defined(CUTE_ARCH_MMA_XE_ENABLED)
        asm ( \
            "{\n" \
            ".decl A_UB v_type=G type=UB num_elts=32 alias=<%1,%2>\n" \
            "mov (M1_NM, 16) %0(0,0)<1> A_UB(0,0)<1;1,0>\n" \
            "mov (M1_NM, 16) %0(0,32)<1> A_UB(0,16)<1;1,0>\n" \
            "}\n" : "=rw"(rSFA[0]) : "rw"(SFA[0]), "P"(sfa_offset) \
        );
        asm ( \
            "{\n" \
            ".decl B_UB v_type=G type=UB num_elts=32 alias=<%1,%2>\n" \
            "mov (M1_NM, 16) %0(0,0)<1> B_UB(0,0)<1;1,0>\n" \
            "mov (M1_NM, 16) %0(0,32)<1> B_UB(0,16)<1;1,0>\n" \
            "}\n" : "=rw"(rSFB[0]) : "rw"(SFB[0]), "P"(sfb_offset) \
        ); 
      #endif

      cute::detail::explode_mma<MMAOp>(
              rD,   make_int_sequence<RegNumD>{},
              rA,   make_int_sequence<RegNumA>{},
              rB,   make_int_sequence<RegNumB>{},
              rC,   make_int_sequence<RegNumC>{},
              rSFA, make_int_sequence<1>{},
              rSFB, make_int_sequence<1>{},
              0,
              0);
    } else {
      cute::detail::explode_mma<MMAOp>(
              rD,   make_int_sequence<RegNumD>{},
              rA,   make_int_sequence<RegNumA>{},
              rB,   make_int_sequence<RegNumB>{},
              rC,   make_int_sequence<RegNumC>{},
              SFA, make_int_sequence<1>{},
              SFB, make_int_sequence<1>{},
              sfa_offset,
              sfb_offset);
    }
  }

};

} /* namespace cute */
