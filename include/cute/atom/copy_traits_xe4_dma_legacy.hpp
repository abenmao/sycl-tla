#pragma once

#include "cute/arch/copy_xe4_dma_legacy.hpp"
#include "cutlass/detail/layout.hpp"
#include "cute/atom/copy_traits_xe4_tma.hpp"
#include "cutlass/gemm/gemm.h"

namespace cute
{

template <class CopyOp>
struct XE4_COPY_Unpack
{
  template <class... Args,
            class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    constexpr auto isLoadOperation = !cute::is_base_of<xe4::DMA_STORE, CopyOp>::value;
    constexpr auto isIm2ColOperation = cute::is_base_of<xe4::ASYNC_ROW_IM2COL, CopyOp>::value;

    auto as_xe4_coord = [](auto const& t) {
      auto flat_t = flatten_to_tuple(t);
      constexpr size_t N = tuple_size<decltype(flat_t)>::value;
      sycl::marray<int32_t, N> result;
      for_each(make_seq<N>{}, [&] (auto i) { result[i] = get<i>(flat_t); });
      return result;
    };

    if constexpr (isLoadOperation) {
      auto dst_ptr = cute::raw_pointer_cast(dst.data());
      if constexpr(isIm2ColOperation) {
        auto src_coord = as_xe4_coord(src(Int<0>{}));
        return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                    traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                    make_tuple(dst_ptr, src_coord), seq<0, 1>{});
      } else {
        auto src_coord = as_xe4_coord(src.data().coord_);
        return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                    traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                    make_tuple(dst_ptr, src_coord), seq<0, 1>{});
      }
    } else {
      auto src_ptr = cute::raw_pointer_cast(src.data());
      if constexpr(isIm2ColOperation) {
        auto dst_coord = as_xe4_coord(take<0,3>(dst(Int<0>{})));
        return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                    traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                    make_tuple(src_ptr, dst_coord), seq<0, 1>{});
      } else {
        auto dst_coord = as_xe4_coord(dst.data().coord_);
        return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                    traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                    make_tuple(src_ptr, dst_coord), seq<0, 1>{});
      }

    }
  }
};

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

template <class CopyOp>
struct XE4_ASYNC_LINEAR_COPY_Unpack
{
  template <class... Args,
            class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    static_assert(is_gmem<TS>::value, "ASYNC_LINEAR_LOAD requires the source be global memory.");
    static_assert(is_smem<TD>::value, "ASYNC_LINEAR_LOAD requires the destination be shared memory.");

    auto src_ptr = cute::raw_pointer_cast(src.data());
    auto dst_ptr = cute::raw_pointer_cast(dst.data());

    return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
      traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
      make_tuple(src_ptr, dst_ptr), seq<0, 1>{});
  }
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// ASYNC_TENSOR_LOAD / ASYNC_TENSOR_STORE ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyOperation>
struct Xe4CopyOp {};

template <typename CopyOperation>
struct Xe4CopyOpWrapper : CopyOperation {};

template <class GmemDetails, class AuxParams, class GmemPtr>
struct Xe4DmaCache {
  template <typename CopyOp>
  using OpUnpack = XE4_COPY_Unpack<CopyOp>;

  Xe4DmaCache() = default;

  Xe4DmaCache(GmemDetails const& gmem_details, AuxParams const& aux_params, GmemPtr gmem_ptr)
    : gmem_details_(gmem_details), aux_params_(aux_params), gmem_ptr_(gmem_ptr) {}

  CUTE_DEVICE void
  set_gmem_ptr(GmemPtr gmem_ptr) {
    gmem_ptr_ = gmem_ptr;
  }

  CUTE_DEVICE void
  set_tensor_desc(TmaDescriptor tensor_desc) const {
    constexpr int tma_dim = rank_v<typename AuxParams::TmaGmemBasis>;
    auto [gmem_shape, gmem_stride, roi_shape, element_stride] = gmem_details_;

    tensordesc_fill_dim_size<tma_dim>(tensor_desc, gmem_shape);
    tensordesc_fill_dim_stride<tma_dim>(tensor_desc, gmem_stride);
    tensordesc_fill_roitensor_dim_size<tma_dim>(tensor_desc, roi_shape);
    tensordesc_fill_element_stride<tma_dim>(tensor_desc, element_stride);

    tdesc_ptr_ = tensor_desc;
  }

  CUTE_DEVICE void
  update_gmem_details_and_params(GmemDetails gmem_details_new, AuxParams aux_params_new, GmemPtr gmem_ptr_new) const {
    gmem_details_ = gmem_details_new;
    aux_params_ = aux_params_new;
    gmem_ptr_ = gmem_ptr_new;
  }

  CUTE_HOST_DEVICE constexpr
  auto get_tensor_desc() const {
    return tdesc_ptr_;
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  template <typename... Args>
  CUTE_HOST_DEVICE constexpr
  auto make_args_tuple(Args&&... args) const {
    return make_tuple(tdesc_ptr_, gmem_ptr_, static_cast<Args&&>(args)...);
  }

  mutable GmemDetails gmem_details_;
  mutable AuxParams aux_params_;
  mutable GmemPtr gmem_ptr_ {nullptr};
  mutable TmaDescriptor tdesc_ptr_ { nullptr };
};

template <class CopyOperation, class NumBitsPerTMA, class DmaCache>
struct Copy_Traits<Xe4CopyOp<CopyOperation>, NumBitsPerTMA, DmaCache>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  DmaCache cache_;

  CUTE_DEVICE void
  set_tensor_desc(TmaDescriptor tensor_desc) const {
    cache_.set_tensor_desc(tensor_desc);
  }

  template <class GmemDetails, class AuxParams, class GmemPtr>
  CUTE_DEVICE void
  update_gmem_details_and_params(
      GmemDetails const& gmem_details,
      AuxParams const& aux_params,
      GmemPtr gmem_ptr) const {
    cache_.update_gmem_details_and_params(gmem_details, aux_params, gmem_ptr);
  }

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

template <slm_matrix_type cm_type, uint32_t stride, class NumBitsPerTMA, class DmaCache>
struct Copy_Traits<Xe4CopyOp<xe4::ASYNC_TENSOR_LOAD_MULTICAST<cm_type, stride>>, NumBitsPerTMA, DmaCache>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;
  DmaCache cache_;

  CUTE_DEVICE void
  set_tensor_desc(TmaDescriptor tensor_desc) const {
    cache_.set_tensor_desc(tensor_desc);
  }

  template <class GmemDetails, class AuxParams, class GmemPtr>
  CUTE_DEVICE void
  update_gmem_details_and_params(
      GmemDetails const& gmem_details,
      AuxParams const& aux_params,
      GmemPtr gmem_ptr) const {
    cache_.update_gmem_details_and_params(gmem_details, aux_params, gmem_ptr);
  }

  template<class ABarrier>
  CUTE_HOST_DEVICE constexpr
  auto with(ABarrier const* abar_ptr, uint32_t const& multicast_mask) const {
    using CopyOperation = xe4::ASYNC_TENSOR_LOAD_MULTICAST<cm_type, stride>;
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
/////////////////////////////////////// ASYNC_LINEAR_LOAD ////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t N>
struct Copy_Traits<xe4::ASYNC_LINEAR_LOAD<T, N>>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, Int<N*sizeof_bits_v<T>>>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  template<class ABarrier>
  CUTE_HOST_DEVICE constexpr
  auto with(ABarrier const* abar_ptr, uint32_t copy_size, [[maybe_unused]] uint32_t const& multicast_mask = 0) const {
    using Wrapper = Xe4CopyOpWrapper<xe4::ASYNC_LINEAR_LOAD<T, N>>;
    auto opargs = make_tuple(abar_ptr, copy_size);
    return Copy_Traits<Wrapper, decltype(opargs)>{opargs};
  }

  // Don't try to execute a copy with XE4_TMA_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

template <class OpArgsTuple, class T, size_t N>
struct Copy_Traits<Xe4CopyOpWrapper<xe4::ASYNC_LINEAR_LOAD<T, N>>, OpArgsTuple> : XE4_ASYNC_LINEAR_COPY_Unpack<Xe4CopyOpWrapper<xe4::ASYNC_LINEAR_LOAD<T, N>>>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, Int<N*sizeof_bits_v<T>>>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  OpArgsTuple const opargs_;

  Copy_Traits(OpArgsTuple const& opargs) : opargs_(opargs) {}
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
