#pragma once

#include "cute/arch/mma_xe4_amma.hpp"
#include "cute/arch/mma_xe4_desc.hpp"
#include "util.hpp"

namespace cute::xe4 {

template <bool cm_major_x, class SEngine, class SLayout>
CUTE_HOST_DEVICE constexpr
auto make_matrix_desc(Tensor<SEngine, SLayout> const& sTensor) {
  using Stride = decltype(stride(SLayout{}));
  using Element = typename SEngine::element_type;

  constexpr int leading_dim = cutlass::gemm::detail::is_mn_major<Stride>() ? 0 : 1;
  constexpr uint32_t cm_size = cm_major_x ? 32 / sizeof(Element) : 32;
  constexpr uint32_t cm_stride = size<leading_dim>(shape(SLayout{})) / cm_size;

  MatDesc mat_desc = reinterpret_cast<uint64_t>(slm_space_cast(raw_pointer_cast(sTensor.data()))) >> 9;
  mat_desc |= (cm_stride << 16);

  return mat_desc;
}

struct MatDescIterator
{
  using reference    = MatDesc;
  using element_type = MatDesc;
  using value_type   = MatDesc;

  MatDesc desc_;

  constexpr MatDescIterator(MatDesc desc): desc_(desc) {}

  // Dereference returns the MatDesc
  CUTE_HOST_DEVICE constexpr
  reference operator*() const { return desc_; }

  // Advance and return a new MatDesc
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return *(*this + i); }

  // Return an advanced iterator
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  MatDescIterator operator+(Index const& offset) const
  {
    return { MatDesc{desc_ + (MatDesc(offset) >> 9)} };
  }

  CUTE_HOST_DEVICE friend void
  print(MatDescIterator const& iter) { printf("xe4::MatDescIterator(%p)", iter.desc_); }
};

template <class T>
CUTE_HOST_DEVICE constexpr
MatDesc
raw_pointer_cast(MatDescIterator const& ptr) {
  return ptr.desc_;
}

template <bool cm_major_x>
struct slm_desc : MatDescIterator { };

template <int M, int K>
using ABLayout = Layout<Shape<_1,Shape<Int<M>,Int<K>>>, Stride<_0,Stride<_1,Int<M>>>>;

} // end namespace cute::xe4

namespace cute {

template <bool cm_major_x>
struct MakeTensor<xe4::slm_desc<cm_major_x>>
{
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    auto mat_desc = xe4::make_matrix_desc<cm_major_x>(tensor<0>(smem_tensor));
    auto new_layout = replace<0>(recast<uint8_t const>(smem_tensor).layout(), Layout<_1,_0>{});
    return make_tensor(xe4::MatDescIterator{mat_desc}, new_layout);
  }
};

struct XE4_ASYNC_GMMA_OP {};
struct XE4_ASYNC_GMMA_SCALE_OP {};

template<bool B, auto TrueVal, auto FalseVal>
constexpr auto condition_v = B ? TrueVal : FalseVal;

template <class TupleC, class TA, class TB, class Shape_MNK_, xe4::GMMA::Major tnspA_, xe4::GMMA::Major tnspB_>
struct MMA_Traits<XE4_ASYNC_GMMA<TupleC, TA, TB, Shape_MNK_, tnspA_, tnspB_>>
{
  using ValTypeD = tuple_element_t<1, TupleC>;
  using ValTypeA = TA;
  using ValTypeB = TB;
  using ValTypeC = tuple_element_t<0, TupleC>;

  using FrgTypeA = xe4::slm_desc<tnspA_==xe4::GMMA::Major::K>;
  using FrgTypeB = xe4::slm_desc<true>;
  using FrgTypeC = xe4::slm_desc<true>;

  using Shape_MNK = Shape_MNK_;
  using ThrID   = Layout<_1>;
  using ALayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using BLayout = xe4::ABLayout<get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using CLayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<1>(Shape_MNK{})>;

  static constexpr xe4::GMMA::Major tnspA = tnspA_;
  static constexpr xe4::GMMA::Major tnspB = tnspB_;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) <= 4)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    using MMA_Op = XE4_ASYNC_GMMA<TupleC, TA, TB, Shape_MNK_, tnspA_, tnspB_>;
    auto opargs = make_tuple(static_cast<TraitsArgs&&>(args)...);
    return MMA_Traits<XE4_ASYNC_GMMA_OP, decltype(opargs), MMA_Op>{{}, opargs};
  }

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) >= 5)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    using MMA_Op = XE4_ASYNC_GMMA<TupleC, TA, TB, Shape_MNK_, tnspA_, tnspB_>;
    auto opargs = make_tuple(static_cast<TraitsArgs&&>(args)...);
    auto tmp_opargs = remove<sizeof...(args)-1>(opargs);
    auto opargs_reduced = remove<sizeof...(args)-3>(tmp_opargs);
    return MMA_Traits<XE4_ASYNC_GMMA_OP, decltype(opargs_reduced), MMA_Op>{{}, opargs_reduced};
  }
};

template<typename OpArgs, typename MMA_Op>
struct MMA_Traits<XE4_ASYNC_GMMA_OP, OpArgs, MMA_Op>: public MMA_Traits<MMA_Op> {
  OpArgs const opargs_;

  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE friend constexpr
  void
  mma_unpack(MMA_Traits const& traits,
       Tensor<TD, DLayout>      & D,
       Tensor<TA, ALayout> const& A,
       Tensor<TB, BLayout> const& B,
       Tensor<TC, CLayout> const& C)
  {
    auto matdesc_tuple = make_tuple(*D.data(), *C.data(), *A.data(), *B.data());
    return detail::explode_tuple(detail::CallFMA<MMA_Op>{},
                                 matdesc_tuple, tuple_seq<decltype(matdesc_tuple)>{},
                                 traits.opargs_, tuple_seq<decltype(traits.opargs_)>{});
  }
};

template <class TupleC, class TA, class TB, class Shape_MNK_, xe4::GMMA::Major tnspA_, xe4::GMMA::Major tnspB_>
struct MMA_Traits<XE4_ASYNC_GMMA_MULTICAST<TupleC, TA, TB, Shape_MNK_, tnspA_, tnspB_>>
{
  using ValTypeD = tuple_element_t<1, TupleC>;
  using ValTypeA = TA;
  using ValTypeB = TB;
  using ValTypeC = tuple_element_t<0, TupleC>;

  using FrgTypeA = xe4::slm_desc<tnspA_==xe4::GMMA::Major::K>;
  using FrgTypeB = xe4::slm_desc<true>;
  using FrgTypeC = xe4::slm_desc<true>;

  using Shape_MNK = Shape_MNK_;
  using ThrID   = Layout<_1>;
  using ALayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using BLayout = xe4::ABLayout<get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using CLayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<1>(Shape_MNK{})>;

  static constexpr xe4::GMMA::Major tnspA = tnspA_;
  static constexpr xe4::GMMA::Major tnspB = tnspB_;

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) <= 4)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    if constexpr (sizeof...(args) <= 4) {
      return MMA_Traits<XE4_ASYNC_GMMA<TupleC, TA, TB, Shape_MNK_, tnspA_, tnspB_>>::with(static_cast<TraitsArgs&&>(args)...);
    }
  }

  template<typename... TraitsArgs, __CUTE_REQUIRES(sizeof...(TraitsArgs) > 4)>
  CUTE_HOST_DEVICE static auto
  with(TraitsArgs&&... args) {
    using MMA_Op = XE4_ASYNC_GMMA_MULTICAST<TupleC, TA, TB, Shape_MNK_, tnspA_, tnspB_>;
    auto opargs = make_tuple(static_cast<TraitsArgs&&>(args)...);
    return MMA_Traits<XE4_ASYNC_GMMA_OP, decltype(opargs), MMA_Op>{{}, opargs};
  }
};

} // namespace cute
