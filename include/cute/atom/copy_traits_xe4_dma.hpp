#pragma once

#include "cute/arch/copy_xe4_dma.hpp"
#include "cutlass/detail/layout.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cutlass/gemm/gemm.h"

namespace cute
{

template <class CopyOp>
struct XE4_SLM_COPY_Unpack
{
  template <class... Args,
            class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    auto src_ptr = cute::raw_pointer_cast(src.data());
    auto dst_ptr = cute::raw_pointer_cast(dst.data());
    return detail::explode_tuple(detail::CallCOPY<CopyOp>{}, make_tuple(src_ptr, dst_ptr), seq<0,1>{});
  }
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// ASYNC_TENSOR_LOAD / ASYNC_TENSOR_STORE ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyOperation>
struct Xe4CopyOpWrapper : CopyOperation {};

template <class CopyOperation, class NumBitsPerTMA, class DmaCache>
struct Copy_Traits<Xe4CopyOp<CopyOperation>, NumBitsPerTMA, DmaCache>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  DmaCache cache_;

  template<class ABarrier>
  CUTE_HOST_DEVICE constexpr
  auto with(ABarrier const* abar_ptr, [[maybe_unused]] uint32_t const& multicast_mask = 0) const {
    using Wrapper = Xe4CopyOpWrapper<CopyOperation>;
    using OpUnpack = typename DmaCache::template OpUnpack<Wrapper>;

    auto opargs = cache_.make_args_tuple(abar_ptr);
    return Copy_Traits<Wrapper, NumBitsPerTMA, decltype(opargs), OpUnpack>{opargs};
  }

  template<class ABarrier, class DimIndex>
  CUTE_HOST_DEVICE constexpr
  auto with(DimIndex const& dim_index, uint32_t const& dim_size, ABarrier const* abar_ptr, [[maybe_unused]] uint32_t const& multicast_mask = 0) const {
    using Wrapper = Xe4CopyOpWrapper<CopyOperation>;
    using OpUnpack = typename DmaCache::template OpUnpack<Wrapper>;

    auto opargs = cache_.make_args_tuple(dim_index, dim_size, abar_ptr);
    return Copy_Traits<Wrapper, NumBitsPerTMA, decltype(opargs), OpUnpack>{opargs};
  }

  CUTE_HOST_DEVICE constexpr
  auto get_tensor_desc() const {
    return cache_.get_tensor_desc();
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    return cache_.get_tma_tensor(g_shape);
  }

  // Don't try to execute a copy with XE4_TMA_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

template <class CopyOperation, class NumBitsPerTMA, class OpArgsTuple, template<class> class OpUnpack>
struct Copy_Traits<Xe4CopyOpWrapper<CopyOperation>, NumBitsPerTMA, OpArgsTuple, OpUnpack<Xe4CopyOpWrapper<CopyOperation>>> : OpUnpack<Xe4CopyOpWrapper<CopyOperation>>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  OpArgsTuple const opargs_;

  Copy_Traits(OpArgsTuple const& opargs) : opargs_(opargs) {}
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// ASYNC_TENSOR_LOAD_MULTICAST / ASYNC_TENSOR_STORE_MULTICAST /////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <slm_matrix_type cm_type, class NumBitsPerTMA, class DmaCache>
struct Copy_Traits<Xe4CopyOp<xe4::ASYNC_TENSOR_LOAD_MULTICAST<cm_type>>, NumBitsPerTMA, DmaCache>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  DmaCache cache_;

  template<class ABarrier>
  CUTE_HOST_DEVICE constexpr
  auto with(ABarrier const* abar_ptr, uint32_t const& multicast_mask) const {
    using CopyOperation = xe4::ASYNC_TENSOR_LOAD_MULTICAST<cm_type>;
    using Wrapper = Xe4CopyOpWrapper<CopyOperation>;
    using OpUnpack = typename DmaCache::template OpUnpack<Wrapper>;

    auto opargs = cache_.make_args_tuple(abar_ptr, multicast_mask);
    return Copy_Traits<Wrapper, NumBitsPerTMA, decltype(opargs), OpUnpack>{opargs};
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    return cache_.get_tma_tensor(g_shape);
  }

  // Don't try to execute a copy with XE4_TMA_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// SLM_VLOAD / SLM_VSTORE ///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<int VS, class S, class D>
struct Copy_Traits<xe4::XE4_LDSM<VS,S,D>> : XE4_SLM_COPY_Unpack<xe4::XE4_LDSM<VS,S,D>>
{
  using ThrID     = Layout<Shape<_32>>;
  using ValID     = Layout<Shape<_1,Int<sizeof_bits_v<S>*VS>>>;
  using SrcLayout = decltype(make_ordered_layout(Shape<_32,Shape<_1,Int<sizeof_bits_v<S>*VS>>>{}, Step<_2,Step<_1,_0>>{}));
  using DstLayout = decltype(make_ordered_layout(Shape<_32,Shape<_1,Int<sizeof_bits_v<D>*VS>>>{}, Step<_2,Step<_1,_0>>{}));
  using RefLayout = SrcLayout;
};

template<class S, int VS, class D>
struct Copy_Traits<xe4::XE4_STSM<VS,S,D>> : XE4_SLM_COPY_Unpack<xe4::XE4_STSM<VS,S,D>>
{
  using ThrID     = Layout<Shape<_32>>;
  using ValID     = Layout<Shape<_1,Int<sizeof_bits_v<S>*VS>>>;
  using SrcLayout = decltype(make_ordered_layout(Shape<_32,Shape<_1,Int<sizeof_bits_v<S>*VS>>>{}, Step<_2,Step<_1,_0>>{}));
  using DstLayout = decltype(make_ordered_layout(Shape<_32,Shape<_1,Int<sizeof_bits_v<D>*VS>>>{}, Step<_2,Step<_1,_0>>{}));
  using RefLayout = SrcLayout;
};
} // namespace cute
