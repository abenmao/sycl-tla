#pragma once

#include "cute/arch/mma_xe4_amma.hpp"
#include "util.hpp"

namespace cute::xe4 {

template <typename T = void>
struct MatDescIterator
{
  using underlying_type = T;
  using reference       = uint32_t;
  using element_type    = uint32_t;
  using value_type      = uint32_t;

  MatDesc desc_;

  MatDescIterator(MatDesc const& desc): desc_(desc) {}

  // Dereference returns the MatDesc
  CUTE_HOST_DEVICE constexpr
  reference operator*() const { return desc_.get(); }

  // Advance and return a new MatDesc
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return *(*this + i); }

  // Return an advanced iterator
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  MatDescIterator operator+(Index const& offset) const
  {
    return { MatDesc{desc_ + uint32_t(offset)} };
  }

  CUTE_HOST_DEVICE friend void
  print(MatDescIterator const& iter) { printf("xe4::MatDescIterator(%p)", *iter); }
};

template <bool cm_major_x, class SEngine, class SLayout>
CUTE_HOST_DEVICE
auto make_matrix_desc(Tensor<SEngine, SLayout> const& sTensor) {
  using Stride = decltype(stride(SLayout{}));
  using Element = typename SEngine::element_type;

  constexpr int non_leading_dim = cutlass::gemm::detail::is_mn_major<Stride>() ? 1 : 0;
  constexpr uint32_t matrix_stride = size<non_leading_dim>(Stride{});
  constexpr slm_matrix_type cm_type = cm_major_x ? slm_matrix_type::type1 : slm_matrix_type::type2;

  auto slm_ptr = slm_space_cast(raw_pointer_cast(sTensor.data()));
  matrix_desc_t mat_desc(slm_ptr, matrix_stride, cm_type);

  return MatDescIterator<Element>{mat_desc};
}

template <class T, class U>
CUTE_HOST_DEVICE constexpr
uint32_t
raw_pointer_cast(MatDescIterator<U> const& ptr) {
  return *ptr;
}

template <bool cm_major_x>
struct slm_desc : MatDescIterator<void> { };

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
    auto layout = recast<uint8_t const>(smem_tensor).layout();

    constexpr auto get_cm_stride = [dim0_size = size<0>(layout)](auto stride_val) {
      constexpr int ColsPerCoreMatrix = 32;
      constexpr int BytesPerCoreMatrix = 1024;
      constexpr int ScaleFactor = BytesPerCoreMatrix / ColsPerCoreMatrix;
      constexpr int new_stride_val = (stride_val >= dim0_size) ? stride_val : ScaleFactor * stride_val;
      return C<new_stride_val>{};
    };

    constexpr auto stride = layout.stride();
    constexpr auto new_stride_1 = get_cm_stride(get<1>(stride));
    constexpr auto new_stride_2 = get_cm_stride(get<2>(stride));

    auto new_layout = make_layout(
      tuple_cat(make_tuple(_1{}), take<1,-1>(shape(layout))),
      tuple_cat(make_tuple(_0{}, new_stride_1, new_stride_2), take<3,-1>(stride))
    );

    auto mat_desc_iter = xe4::make_matrix_desc<cm_major_x>(tensor<0>(smem_tensor));

    return make_tensor(mat_desc_iter, new_layout);
  }
};

struct XE4_ASYNC_GMMA_OP {};
struct XE4_ASYNC_GMMA_SCALE_OP {};

template<bool B, auto TrueVal, auto FalseVal>
constexpr auto condition_v = B ? TrueVal : FalseVal;

template <class TupleC, class TA, class TB, class Shape_MNK_, SM90::GMMA::Major tnspA_, SM90::GMMA::Major tnspB_>
struct MMA_Traits<XE4_ASYNC_GMMA<TupleC, TA, TB, Shape_MNK_, tnspA_, tnspB_>>
{
  using ValTypeD = tuple_element_t<1, TupleC>;
  using ValTypeA = TA;
  using ValTypeB = TB;
  using ValTypeC = tuple_element_t<0, TupleC>;

  using FrgTypeA = xe4::slm_desc<tnspA_==SM90::GMMA::Major::K>;
  using FrgTypeB = xe4::slm_desc<true>;
  using FrgTypeC = xe4::slm_desc<true>;

  using Shape_MNK = Shape_MNK_;
  using ThrID   = Layout<_1>;
  using ALayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using BLayout = xe4::ABLayout<get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using CLayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<1>(Shape_MNK{})>;

  static constexpr SM90::GMMA::Major tnspA = tnspA_;
  static constexpr SM90::GMMA::Major tnspB = tnspB_;

  template<class MmaCtrl, class Abarriers>
  CUTE_HOST_DEVICE static auto
  with(MmaCtrl mmaCtrl, Abarriers barriers) {
    static_assert(is_tuple_v<Abarriers>, "Abarriers must be a tuple");
    using MMA_Op = XE4_ASYNC_GMMA<TupleC, TA, TB, Shape_MNK_, tnspA_, tnspB_>;
    auto opargs = tuple_cat(make_tuple(mmaCtrl), barriers);
    return MMA_Traits<XE4_ASYNC_GMMA_OP, decltype(opargs), MMA_Op>{{}, opargs};
  }

  template<class MmaCtrl, class Abarriers, class McastMasks>
  CUTE_HOST_DEVICE static auto
  with(MmaCtrl mmaCtrl, Abarriers barriers, McastMasks masks) {
    return with(mmaCtrl, barriers);
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
    auto matdesc_tuple = make_tuple(D.data(), C.data(), A.data(), B.data());
    return detail::explode_tuple(detail::CallFMA<MMA_Op>{},
                                 matdesc_tuple, tuple_seq<decltype(matdesc_tuple)>{},
                                 traits.opargs_, tuple_seq<decltype(traits.opargs_)>{});
  }
};

template <class TupleC, class TA, class TB, class Shape_MNK_, SM90::GMMA::Major tnspA_, SM90::GMMA::Major tnspB_>
struct MMA_Traits<XE4_ASYNC_GMMA_MULTICAST<TupleC, TA, TB, Shape_MNK_, tnspA_, tnspB_>>
{
  using ValTypeD = tuple_element_t<1, TupleC>;
  using ValTypeA = TA;
  using ValTypeB = TB;
  using ValTypeC = tuple_element_t<0, TupleC>;

  using FrgTypeA = xe4::slm_desc<tnspA_==SM90::GMMA::Major::K>;
  using FrgTypeB = xe4::slm_desc<true>;
  using FrgTypeC = xe4::slm_desc<true>;

  using Shape_MNK = Shape_MNK_;
  using ThrID   = Layout<_1>;
  using ALayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using BLayout = xe4::ABLayout<get<1>(Shape_MNK{}), get<2>(Shape_MNK{})>;
  using CLayout = xe4::ABLayout<get<0>(Shape_MNK{}), get<1>(Shape_MNK{})>;

  static constexpr SM90::GMMA::Major tnspA = tnspA_;
  static constexpr SM90::GMMA::Major tnspB = tnspB_;

  template<class MmaCtrl, class Abarriers>
  CUTE_HOST_DEVICE static auto
  with(MmaCtrl mmaCtrl, Abarriers barriers) {
    return MMA_Traits<XE4_ASYNC_GMMA<TupleC, TA, TB, Shape_MNK_, tnspA_, tnspB_>>::with(mmaCtrl, barriers);
  }

  template<class MmaCtrl, class Abarriers, class McastMasks>
  CUTE_HOST_DEVICE static auto
  with(MmaCtrl mmaCtrl, Abarriers barriers, McastMasks masks) {
    static_assert(is_tuple_v<Abarriers>, "Abarriers must be a tuple");
    static_assert(is_tuple_v<McastMasks>, "McastMasks must be a tuple");

    using MMA_Op = XE4_ASYNC_GMMA_MULTICAST<TupleC, TA, TB, Shape_MNK_, tnspA_, tnspB_>;
    auto opargs = tuple_cat(make_tuple(mmaCtrl), barriers, masks);
    return MMA_Traits<XE4_ASYNC_GMMA_OP, decltype(opargs), MMA_Op>{{}, opargs};
  }
};

} // namespace cute
