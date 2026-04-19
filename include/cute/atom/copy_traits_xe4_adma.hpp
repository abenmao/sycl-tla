#pragma once

#include <sycl/sycl.hpp>

#include <cute/arch/mma_xe4_desc.hpp>
#include <cute/arch/copy_xe4_adma.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/atom/copy_atom.hpp>

#include <cute/atom/copy_traits_xe4_tma.hpp>

#include <cute/layout.hpp>

namespace cute {

// Type trait to detect XE4_ADMA_STORE_REDUCE (and its _OP marker variant).
// Used in make_adma_atom_A/B_xe4 to handle STORE_REDUCE in the multicast lambda
// (STORE_REDUCE never multicasts, so it always returns Int<1>{}).
// The _OP specialization is added below near XE4_ADMA_STORE_REDUCE_OP.
template <typename T>
struct is_xe4_adma_store_reduce : cute::false_type {};

template <typename T, RedOp Rop, BarrierType BType>
struct is_xe4_adma_store_reduce<XE4_ADMA_STORE_REDUCE<T, Rop, BType>>
  : cute::true_type {};

template <typename T>
inline constexpr bool is_xe4_adma_store_reduce_v = is_xe4_adma_store_reduce<T>::value;

template <class CopyOp, class... Args>
struct ADMA_LOAD_Unpack
{
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    // static_assert(is_smem<TD>::value, "XE4_ADMA_LOAD requires the destination be shared memory.");
    constexpr int M = tuple_size<decltype(traits.opargs_)>::value;

    auto as_xe4_coord = [](auto const& t) {
      return to_vec<int32_t>(flatten_to_tuple(t));
    };
    auto src_coord = as_xe4_coord(src.data().coord_);
    auto dst_ptr = cute::raw_pointer_cast(dst.data());
    return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                traits.opargs_, make_range<0, M -2>{},
                                make_tuple(dst_ptr, src_coord), seq<0, 1>{},
                                traits.opargs_, make_range<M-2, M>{});
  }
};

template <class CopyOp, class... Args>
struct ADMA_STORE_Unpack
{
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    // static_assert(is_smem<TS>::value, "XE4_ADMA_STORE requires the source be shared memory.");
    constexpr int M = tuple_size<decltype(traits.opargs_)>::value;

    auto as_xe4_coord = [](auto const& t) {
      return to_vec<int32_t>(flatten_to_tuple(t));
    };
    auto dst_coord = as_xe4_coord(dst.data().coord_);
    auto src_ptr = cute::raw_pointer_cast(src.data());
    return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                traits.opargs_, make_range<0, M-1>{},
                                make_tuple(src_ptr, dst_coord), seq<0, 1>{},
                                traits.opargs_, make_range<M-1, M>{});
  }
};

struct XE4_ADMA_LOAD_OP : XE4_ADMA_LOAD {};

template <typename T, class NumBitsPerADMA, class AuxParams_>
struct Copy_Traits<XE4_ADMA_LOAD, T, NumBitsPerADMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  TensorDescriptor<T> tensorDesc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;
  mutable uint64_t* tdesc_ptr_ { nullptr };

  CUTE_DEVICE void
  set_tensor_desc(uint64_t* tensor_desc) const {
    dupTensorPayload(tensor_desc, (uint64_t *)&tensorDesc_.payload);
    tdesc_ptr_ = tensor_desc;
  }

  CUTE_HOST_DEVICE constexpr
  TensorDescriptor<T>* get_tensor_desc() const {
    return reinterpret_cast<TensorDescriptor<T>*>(tdesc_ptr_);
  }

  // TODO: rename, get_adma_tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  template <typename... Args>
  CUTE_HOST_DEVICE constexpr
  auto make_args_tuple(Args&&... args) const {
    return make_tuple(
        tdesc_ptr_, tensorDesc_.g_pointer,
        tensorDesc_.matrix_desc, static_cast<Args&&>(args)...);
  }

  template <detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_LOAD_OP, T, NumBitsPerADMA, detail::CacheHint<CC>, detail::FillMode<FM>>
  with(uint64_t* abar_ptr, [[maybe_unused]] uint32_t const& multicast_mask = 0,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return {tdesc_ptr_, tensorDesc_.g_pointer, tensorDesc_.matrix_desc, abar_ptr};
  }

  // Don't try to execute a copy with this Copy_Traits specialization before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

template <typename T, class NumBitsPerADMA, detail::CacheCtrl CC, detail::FillMethod FM>
struct Copy_Traits<XE4_ADMA_LOAD_OP, T, NumBitsPerADMA, detail::CacheHint<CC>, detail::FillMode<FM>>
  : ADMA_LOAD_Unpack<XE4_ADMA_LOAD_OP, T, NumBitsPerADMA, detail::CacheHint<CC>, detail::FillMode<FM>>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // XE4_ADMA_LOAD arguments with CacheHint and FillMode
  tuple<
  uint64_t*,
  T const*,
  uint32_t,
  uint64_t*,
  detail::CacheHint<CC>,
  detail::FillMode<FM>
  > const opargs_;

  CUTE_HOST_DEVICE
  Copy_Traits(uint64_t * desc, T const* adrs, uint32_t mdesc, uint64_t* mbar)
    : opargs_(desc, adrs, mdesc, mbar, {}, {}) {}

  CUTE_HOST_DEVICE constexpr
  TensorDescriptor<T> const*
  get_tma_descriptor() const {
    return reinterpret_cast<TensorDescriptor<T> const*>(get<0>(opargs_));
  }
};

struct XE4_ADMA_LOAD_MULTICAST_OP : XE4_ADMA_LOAD_MULTICAST {};

template <typename T, class NumBitsPerADMA, class AuxParams_>
struct Copy_Traits<XE4_ADMA_LOAD_MULTICAST, T, NumBitsPerADMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  TensorDescriptor<T> tensorDesc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;
  mutable uint64_t* tdesc_ptr_ { nullptr };

  CUTE_DEVICE void
  set_tensor_desc(uint64_t* tensor_desc) const {
    dupTensorPayload(tensor_desc, (uint64_t *)&tensorDesc_.payload);
    tdesc_ptr_ = tensor_desc;
  }

  CUTE_HOST_DEVICE constexpr
  TensorDescriptor<T>* get_tensor_desc() const {
    return reinterpret_cast<TensorDescriptor<T>*>(tdesc_ptr_);
  }

  // TODO: rename, get_adma_tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  template <typename... Args>
  CUTE_HOST_DEVICE constexpr
  auto make_args_tuple(Args&&... args) const {
    return make_tuple(
        tdesc_ptr_, tensorDesc_.g_pointer,
        tensorDesc_.matrix_desc, static_cast<Args&&>(args)...);
  }

  template <detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_LOAD_MULTICAST_OP, T, NumBitsPerADMA, detail::CacheHint<CC>, detail::FillMode<FM>>
  with(uint64_t* abar_ptr, uint32_t const& multicast_mask = 0,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return {tdesc_ptr_, tensorDesc_.g_pointer, tensorDesc_.matrix_desc, abar_ptr, multicast_mask};
  }

  // Don't try to execute a copy with XE4_ADMA_LOAD_MULTICAST before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

template <typename T, class NumBitsPerADMA, detail::CacheCtrl CC, detail::FillMethod FM>
struct Copy_Traits<XE4_ADMA_LOAD_MULTICAST_OP, T, NumBitsPerADMA, detail::CacheHint<CC>, detail::FillMode<FM>>
  : ADMA_LOAD_Unpack<XE4_ADMA_LOAD_MULTICAST_OP, T, NumBitsPerADMA, detail::CacheHint<CC>, detail::FillMode<FM>>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // XE4_ADMA_LOAD_MULTICAST arguments with CacheHint and FillMode
  tuple<
  uint64_t*,
  T const*,
  uint32_t,
  uint64_t*,
  uint32_t,
  detail::CacheHint<CC>,
  detail::FillMode<FM>
  > const opargs_;

  CUTE_HOST_DEVICE
  Copy_Traits(uint64_t * desc, T const* adrs, uint32_t mdesc, uint64_t* mbar, uint32_t mask)
    : opargs_(desc, adrs, mdesc, mbar, mask, {}, {}) {}

  CUTE_HOST_DEVICE constexpr
  TensorDescriptor<T> const*
  get_tma_descriptor() const {
    return reinterpret_cast<TensorDescriptor<T> const*>(get<0>(opargs_));
  }
};

struct XE4_ADMA_STORE_OP : XE4_ADMA_STORE {};

template <typename T, class NumBitsPerADMA, class AuxParams_>
struct Copy_Traits<XE4_ADMA_STORE, T, NumBitsPerADMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  TensorDescriptor<T> tensorDesc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;
  mutable uint64_t* tdesc_ptr_ { nullptr };

  CUTE_DEVICE void
  set_tensor_desc(uint64_t* tensor_desc) const {
    dupTensorPayload(tensor_desc, (uint64_t *)&tensorDesc_.payload);
    tdesc_ptr_ = tensor_desc;
  }

  CUTE_HOST_DEVICE constexpr
  TensorDescriptor<T>* get_tensor_desc() const {
    return reinterpret_cast<TensorDescriptor<T>*>(tdesc_ptr_);
  }

  // TODO: rename, get_adma_tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  template <typename... Args>
  CUTE_HOST_DEVICE constexpr
  auto make_args_tuple(Args&&... args) const {
    return make_tuple(
        const_cast<T*>(tensorDesc_.g_pointer), tdesc_ptr_,
        tensorDesc_.matrix_desc, static_cast<Args&&>(args)...);
  }

  template <detail::CacheCtrl CC = detail::CacheCtrl::L2wb_L3uc>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_STORE_OP, T, NumBitsPerADMA, detail::CacheHint<CC>>
  with(uint64_t* abar_ptr, detail::CacheHint<CC> = {}) const {
    // Store writes to global memory (gmem_ptr first per store(gmem_ptr, desc, ...) convention).
    // TensorDescriptor::g_pointer is const T* for shared load/store descriptor representation,
    // but STORE always targets writable memory, so the cast is safe here.
    return {const_cast<T*>(tensorDesc_.g_pointer), tdesc_ptr_, tensorDesc_.matrix_desc, abar_ptr};
  }

  // Don't try to execute a copy with XE4_ADMA_STORE before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

// The executable version
template <typename T, class NumBitsPerADMA, detail::CacheCtrl CC>
struct Copy_Traits<XE4_ADMA_STORE_OP, T, NumBitsPerADMA, detail::CacheHint<CC>>
  : ADMA_STORE_Unpack<XE4_ADMA_STORE_OP, T, NumBitsPerADMA, detail::CacheHint<CC>>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // XE4_ADMA_STORE arguments with CacheHint
  tuple<
  T*,
  uint64_t*,
  uint32_t,
  uint64_t*,
  detail::CacheHint<CC>
  > const opargs_;

  CUTE_HOST_DEVICE
  Copy_Traits(T* adrs, uint64_t* desc, uint32_t mdesc, uint64_t* mbar)
    : opargs_(adrs, desc, mdesc, mbar, {}) {}

  CUTE_HOST_DEVICE constexpr
  TensorDescriptor<T> const*
  get_tma_descriptor() const {
    return reinterpret_cast<TensorDescriptor<T> const*>(get<1>(opargs_));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// PREFETCH — Directly Executable Traits (no .with() needed)
///
/// Unlike LOAD/STORE traits which follow the two-phase pattern:
///   non-executable builder → .with(abar) → executable _OP traits
/// PREFETCH traits are directly executable because prefetch is fire-and-forget:
///   - No barrier (no .with() needed)
///   - No SLM destination (dst tensor in copy_unpack is ignored)
///   - No completion signal — the instruction just hints the cache
///
/// Key design: copy_unpack is defined inline (not deleted + _OP indirection).
/// This means `copy(prefetch_traits, src, dst)` works immediately.
///
/// Converting constructor: Enables constructing prefetch traits from any other ADMA traits
/// (LOAD, LOAD_MULTICAST, etc.) by copying tensorDesc_, aux_params_, and tdesc_ptr_.
/// This is the mechanism that makes cute::prefetch(load_atom, src) work — it constructs
/// Copy_Traits<XE4_ADMA_PREFETCH> from Copy_Traits<XE4_ADMA_LOAD> at the call site.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, class NumBitsPerADMA, class AuxParams_>
struct Copy_Traits<XE4_ADMA_PREFETCH, T, NumBitsPerADMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  using DstLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  using RefLayout = SrcLayout;

  TensorDescriptor<T> tensorDesc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;
  mutable uint64_t* tdesc_ptr_ { nullptr };

  // Default constructor
  Copy_Traits() = default;
  Copy_Traits(TensorDescriptor<T> const& td, AuxParams const& ap)
    : tensorDesc_(td), aux_params_(ap) {}

  // Converting constructor: creates prefetch traits from any other ADMA traits.
  // Required by cute::prefetch() in prefetch.hpp which constructs
  // Copy_Traits<XE4_ADMA_PREFETCH, ...> from Copy_Traits<XE4_ADMA_LOAD, ...>.
  // Copies the tensor descriptor (same gmem shape/stride/base address) and
  // the device-side tdesc_ptr_ so that the prefetch uses the same HW descriptor.
  template <class OtherCopyOp, class OtherT, class OtherBits, class OtherAux>
  Copy_Traits(Copy_Traits<OtherCopyOp, OtherT, OtherBits, OtherAux> const& other)
    : tensorDesc_(other.tensorDesc_), aux_params_(other.aux_params_),
      tdesc_ptr_(other.tdesc_ptr_) {}

  CUTE_DEVICE void
  set_tensor_desc(uint64_t* tensor_desc) const {
    dupTensorPayload(tensor_desc, (uint64_t *)&tensorDesc_.payload);
    tdesc_ptr_ = tensor_desc;
  }

  CUTE_HOST_DEVICE constexpr
  TensorDescriptor<T>* get_tensor_desc() const {
    return reinterpret_cast<TensorDescriptor<T>*>(tdesc_ptr_);
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  // DIRECTLY EXECUTABLE — no .with() needed, no barrier for prefetch.
  // The dst tensor is ignored (prefetch has no SLM destination).
  // Dispatches to XE4_ADMA_PREFETCH::copy() → AsyncTensorGlobalPrefetch.
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    auto src_coord = to_vec<int32_t>(flatten_to_tuple(src.data().coord_));
    XE4_ADMA_PREFETCH::copy(traits.tdesc_ptr_, traits.tensorDesc_.g_pointer, src_coord);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// STORE_REDUCE — Two-Phase Traits (non-executable → .with(abar) → executable)
///
/// Follows the same pattern as XE4_ADMA_LOAD / XE4_ADMA_STORE:
///   1. Non-executable builder traits hold the tensor descriptor and aux params.
///      Calling copy() on these is a deleted function (compile error).
///   2. .with(abar_ptr) returns executable _OP traits that hold the flattened opargs tuple.
///      These inherit ADMA_STORE_Unpack to implement copy_unpack.
///
/// The _OP marker type inherits from STORE_REDUCE so that CopyOp::copy() dispatches
/// correctly through the same virtual-ish mechanism (CRTP via ADMA_STORE_Unpack).
////////////////////////////////////////////////////////////////////////////////////////////////////

// Executable marker type — inherits STORE_REDUCE so CopyOp::copy() resolves correctly.
// ADMA_STORE_Unpack<_OP> calls detail::CallCOPY<_OP>{} which invokes _OP::copy()
// (inherited from STORE_REDUCE), dispatching to AsyncTensorReduce.
template <typename T, RedOp Rop, BarrierType BType>
struct XE4_ADMA_STORE_REDUCE_OP : XE4_ADMA_STORE_REDUCE<T, Rop, BType> {};

// Extend type trait to also match the _OP variant (used in make_adma_atom_A/B_xe4 multicast lambda)
template <typename T, RedOp Rop, BarrierType BType>
struct is_xe4_adma_store_reduce<XE4_ADMA_STORE_REDUCE_OP<T, Rop, BType>>
  : cute::true_type {};

// Non-executable builder traits: holds tensor descriptor + aux params.
// .with(abar_ptr) packs (tdesc, gmem, mat_desc, abar, 0) into the executable _OP traits.
template <typename T, RedOp Rop, BarrierType BType,
          class NumBitsPerADMA, class AuxParams_>
struct Copy_Traits<XE4_ADMA_STORE_REDUCE<T, Rop, BType>, T, NumBitsPerADMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  using DstLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  using RefLayout = SrcLayout;

  TensorDescriptor<T> tensorDesc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;
  mutable uint64_t* tdesc_ptr_ { nullptr };

  CUTE_DEVICE void
  set_tensor_desc(uint64_t* tensor_desc) const {
    dupTensorPayload(tensor_desc, (uint64_t *)&tensorDesc_.payload);
    tdesc_ptr_ = tensor_desc;
  }

  CUTE_HOST_DEVICE constexpr
  TensorDescriptor<T>* get_tensor_desc() const {
    return reinterpret_cast<TensorDescriptor<T>*>(tdesc_ptr_);
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_STORE_REDUCE_OP<T, Rop, BType>, T, NumBitsPerADMA>
  with(uint64_t* abar_ptr) const {
    return {tdesc_ptr_, tensorDesc_.g_pointer, tensorDesc_.matrix_desc, abar_ptr};
  }

  // Not directly executable — must call .with(abar) first
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

// Executable traits — reuses ADMA_STORE_Unpack (store direction: src=smem, dst=gmem coord).
// ADMA_STORE_Unpack::copy_unpack explodes the 4-element opargs_ tuple + (slm_ptr, coord)
// from the src/dst tensors, then calls XE4_ADMA_STORE_REDUCE_OP::copy() with all 6 args.
template <typename T, RedOp Rop, BarrierType BType,
          class NumBitsPerADMA>
struct Copy_Traits<XE4_ADMA_STORE_REDUCE_OP<T, Rop, BType>, T, NumBitsPerADMA>
  : ADMA_STORE_Unpack<XE4_ADMA_STORE_REDUCE_OP<T, Rop, BType>, T, NumBitsPerADMA>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  using DstLayout = Layout<Shape<_1,NumBitsPerADMA>>;
  using RefLayout = SrcLayout;

  // 5-element opargs matching unified 7-arg CopyOp (Unpack adds slm_ptr + coord)
  tuple<
  uint64_t*,
  T const*,
  uint32_t,
  uint64_t*
  > const opargs_;

  CUTE_HOST_DEVICE
  Copy_Traits(uint64_t* desc, T const* adrs, uint32_t mdesc, uint64_t* mbar)
    : opargs_(desc, adrs, mdesc, mbar) {}

  CUTE_HOST_DEVICE constexpr
  TensorDescriptor<T> const*
  get_tma_descriptor() const {
    return reinterpret_cast<TensorDescriptor<T> const*>(get<0>(opargs_));
  }
};

namespace detail {

// Use a sidx2gmode to read through the GMEM tensor
//   and construct a TMA Descriptor for the resulting instruction
// At the same time, construct the Tma Tensor's Stride to generate
//   the TMA coordinates that the instruction consumes.
//
template <class InternalType,
          class GEngine, class GLayout,
          class TShape, class TStride>
CUTE_HOST_RTC
auto
make_adma_copy_desc(
    Tensor<GEngine,GLayout> const& gtensor,         // The original GMEM Tensor
    Layout<TShape,TStride>  const& adma_gbasis,     // ADMA mode -> GMEM mode mapping
    uint32_t                       matrix_desc,
    uint32_t                       num_multicast)   // The number of CTAs in multicasting
{
  //
  // Tensor desc creation
  //

  constexpr int t_dim = decltype(rank(adma_gbasis))::value;

  //
  // Tensor Descriptor info
  //

  // Recast the original tensor for shape/stride inspections
  Tensor gtensor_T = recast<InternalType>(gtensor);

  auto* gmem_address = raw_pointer_cast(gtensor_T.data());
  auto  gmem_layout  = gtensor_T.layout();

  cute::array<uint64_t, 5> gmem_prob_shape  = {1,1,1,1,1};
  cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};

  fill_tma_gmem_shape_stride(gtensor_T, stride(adma_gbasis), gmem_prob_shape, gmem_prob_stride);

#if 0
  assert((reinterpret_cast<uint64_t>(gmem_address) & 0b1111) == 0);  // Address must be 16B-aligned

  assert(gmem_prob_shape[0] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[0] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[1] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[1] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[2] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[2] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[3] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[3] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[4] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[4] <= (uint64_t(1) << 32));         // Size must be max 2^32

  // TMA descriptor does not store the zeroth stride and assumes it is 1 (TmaInternalType element).
  assert(gmem_prob_stride[0] == 1 && "Majorness of smem doesn't match majorness of gmem");
#endif

  // convert strides to byte strides
  for(uint64_t& stride : gmem_prob_stride) {
    stride = (stride * sizeof_bits_v<InternalType>) / 8;
  }

#if 0
  // Assert the byte strides. Tma Descriptor uses byte strides
  assert((gmem_prob_stride[1]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[1] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[2]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[2] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[3]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[3] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[4]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[4] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
#endif

  //
  // TMA smem desc info
  //
  cute::array<uint16_t, 5> smem_box_shape  = {1,1,1,1,1};
  cute::array<uint32_t, 5> smem_box_stride = {1,1,1,1,1};

  // The smem box is simply given by the sizes of the modes in adma_gbasis
  for_each(make_seq<t_dim>{}, [&](auto i) {
    smem_box_shape[i] *= size<i>(adma_gbasis);
  });
  // Finally, truncate the tma box by the num_multicast
  for (uint32_t i = t_dim-1, multicast = num_multicast; multicast > 1; --i) {
#if 0
    assert(smem_box_shape[i] % multicast == 0 || multicast % smem_box_shape[i] == 0);
#endif
    uint32_t new_mult = ceil_div(multicast, smem_box_shape[i]);
    smem_box_shape[i] = ceil_div(smem_box_shape[i], multicast);
    multicast = new_mult;
  }

#if 0
  assert(smem_box_shape[0] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[0] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[1] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[1] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[2] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[2] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[3] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[3] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[4] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[4] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256

  assert(smem_box_stride[0] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[0] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[1] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[1] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[2] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[2] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[3] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[3] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[4] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[4] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
#endif

    //
    // Construct the descriptor
    //

    TensorDescriptor<InternalType> tensor_desc{};

    //
    // TMA general info
    //

#if 0

    CUtensorMapDataType     tma_format      = TMA::to_CUtensorMapDataType<TmaInternalType>();
    CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    // TMA smem swizzle type
    TMA::SmemSwizzleBits swizzle_bits = get_tma_swizzle_bits(swizzle);
    TMA::SmemSwizzleBase swizzle_base = get_tma_swizzle_base(swizzle);
    CUtensorMapSwizzle smem_swizzle = TMA::to_CUtensorMapSwizzle(swizzle_bits, swizzle_base);
    CUresult result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &tma_desc,
        tma_format,
        t_dim,
        gmem_address,
        gmem_prob_shape.data(),
        gmem_prob_stride.data() + 1,  // gmem_prob_stride[0] implicitly 1
        smem_box_shape.data(),
        smem_box_stride.data(),
        tma_interleave,
        smem_swizzle,
        tma_l2Promotion,
        tma_oobFill);

    if (result != CUDA_SUCCESS) {
      std::cerr << "TMA Desc Addr:   " << &tma_desc
                << "\nformat         " << tma_format
                << "\ndim            " << t_dim
                << "\ngmem_address   " << gmem_address
                << "\nglobalDim      " << gmem_prob_shape
                << "\nglobalStrides  " << gmem_prob_stride
                << "\nboxDim         " << smem_box_shape
                << "\nelementStrides " << smem_box_stride
                << "\ninterleave     " << tma_interleave
                << "\nswizzle        " << smem_swizzle
                << "\nl2Promotion    " << tma_l2Promotion
                << "\noobFill        " << tma_oobFill << std::endl;
      std::cerr << "Error: Failed to initialize the TMA descriptor " << result << std::endl;
      assert(false);
    }

#endif // (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)

  auto recast_ratio = cute::trait_ratio(sizeof_bits<typename GEngine::value_type>{},
                                        sizeof_bits<InternalType>{});

  auto gbasis = make_basis_like(shape(gtensor));

  // Finally, get the inverse permutation of the E<i> bases for the mocked gmem stride
  auto gmem_tma_basis_stride = transform_leaf(gbasis, [&](auto ei) {
    auto si = basis_get(ei,  shape(gmem_layout));
    auto di = basis_get(ei, stride(gmem_layout));
    if constexpr (is_constant<1, decltype(si)>::value || is_constant<0, decltype(di)>::value) {
      return Int<0>{};                  // If size-1 or stride-0, return arithmetic identity -- no contribution to the TMA
    } else {
      auto tma_gmem_basis_stride = stride(adma_gbasis);
      // Find j such that E<i> is in stride<j>(adma_gbasis)
      using EI = decltype(ei);
      [[maybe_unused]] auto j = find_if(tma_gmem_basis_stride, [&](auto tma_stride_j) { return any_of(tma_stride_j, [&](auto dj) { return dj == EI{}; }); });
      if constexpr (decltype(j == rank(tma_gmem_basis_stride))::value) {
        return Int<0>{};               // If not-found, return arithmetic identity -- no contribution to the TMA
      } else
      if constexpr (decltype(j == Int<0>{})::value) {
        auto scale = recast_ratio * basis_get(ei, stride(gtensor));
        return E<j>{} * scale;         // Return TMA Coord basis -- with a recast scale factor
      } else
      if constexpr (decltype(rank<j>(tma_gmem_basis_stride) == Int<1>{})::value) {
        return E<j>{};                 // Return TMA Coord basis -- known scale of Int<1>{}
      } else {
        int32_t scale = ceil_div(int32_t(di * sizeof_bits_v<InternalType> / cute::max(gmem_prob_stride[j], uint64_t{16})), 8);
        return E<j>{} * scale;         // Return TMA Coord basis -- with a dynamic scale factor
      }
    }
  });

#if 0
    print("gmem_tma_basis_stride : "); print(gmem_tma_basis_stride); print("\n");
#endif

  using AuxParams = AuxTmaParams<decltype(gmem_tma_basis_stride),
                                 decltype(adma_gbasis),
                                 Swizzle<0, 4, 3>>; // XXX: do we need this?

  // Carry tensor descriptor message out of here.
  cute::array<uint32_t, t_dim> gmem_shape;
  for_each(make_seq<t_dim>{}, [&](auto i) {gmem_shape[i] = gmem_prob_shape[i];});
  cute::array<uint64_t, t_dim-1> gmem_stride;
  for_each(make_seq<t_dim-1>{}, [&](auto i) {gmem_stride[i] = gmem_prob_stride[i+1];});
  cute::array<uint16_t, t_dim> sbox_shape;
  for_each(make_seq<t_dim>{}, [&](auto i) {sbox_shape[i] = smem_box_shape[i];});
  cute::array<uint32_t, t_dim> sbox_stride;
  for_each(make_seq<t_dim>{}, [&](auto i) {sbox_stride[i] = smem_box_stride[i];});

  fillTensorDescriptorDimSize((uint64_t *)&tensor_desc, gmem_shape);
  fillTensorDescriptorDimStride((uint64_t *)&tensor_desc, gmem_stride);
  fillTensorDescriptorROI((uint64_t *)&tensor_desc, sbox_shape);
  fillTensorDescriptorElementStride((uint64_t *)&tensor_desc, sbox_stride);
  tensor_desc.matrix_desc = matrix_desc;
  tensor_desc.g_pointer = sycl::address_space_cast<
    sycl::access::address_space::global_space,
    sycl::access::decorated::yes
  >(gmem_address).get();

  return cute::make_tuple(tensor_desc, AuxParams{gmem_tma_basis_stride});
}

template <class InternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class VShape, class VStride>
CUTE_HOST_RTC
auto
make_adma_copy_atom(
    CopyOp,
    Tensor<GEngine,GLayout> const& gtensor,       // Full GMEM Tensor
    SLayout                 const& slayout,       // Group Tile of SMEM, potentially swizzled
    uint32_t                const& num_multicast, // The number of Groups involved in multicasting
    uint32_t                const& matrix_desc,
    Layout<VShape,VStride>  const& cta_v_map)     // V: CTA val idx -> gmem mode
{
  //
  // AMMA truncated layout
  //

  // auto smem_swizzle = get_swizzle_portion(slayout);
  auto smem_layout  = get_nonswizzle_portion(slayout);

  auto adma_gbasis = detail::construct_tma_gbasis<InternalType>(gtensor, smem_layout, cta_v_map);

  //
  // Construct the AMMA Desc and the strides of the AMMA Tensor
  //

  auto [tma_desc, aux_params] = detail::make_adma_copy_desc<InternalType>(
      gtensor, adma_gbasis, matrix_desc, num_multicast);

  //
  // Construct the Copy_Traits
  //
  constexpr int num_bits_per_tma = size(adma_gbasis) * sizeof_bits_v<InternalType>;
  using Traits = Copy_Traits<CopyOp, InternalType, cute::C<num_bits_per_tma>, decltype(aux_params)>;
  using Atom   = Copy_Atom<Traits, typename GEngine::value_type>;

  Traits tma_traits{tma_desc, aux_params};

#if 0
  print("num_bits_per_tma :  "); print(num_bits_per_tma); print("\n");
  print("g_stride_bases   :  "); print(tma_traits.aux_params_.g_stride_); print("\n");
#endif

  // Return the Copy_Atom
  return Atom{tma_traits};
}

template <class SLayout>
CUTE_HOST
auto
make_matrix_descriptor(
    SLayout const& slayout, bool is_A_matrix = false
) {
  auto strides = slayout.stride();

  MatrixDescriptor matrix_desc{};
  if (is_A_matrix) {
    matrix_desc.Type = get<1>(strides) == 1 /* runtime check */?
      MatrixDescriptor::Type1 : MatrixDescriptor::Type2;
  } else {
    matrix_desc.Type = MatrixDescriptor::Type1;
  }

  matrix_desc.Pitch = get<1>(strides) == 1 ?
    get<0>(strides) >> 2 : get<1>(strides) >> 2;

  return matrix_desc.raw_;
}

// The "logical TMA tid" is a map from the CTA rank to its logical id
// within the instruction.  It works like a mask or ordering on the
// CTAs.  For non-multicast TMA, all CTAs should map to 0.  For
// multicast TMA of size 4, CTAs will be mapped to {0,1,2,3}.
template <class InternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class TShape, class TStride,
          class VShape, class VStride>
CUTE_HOST_RTC
auto
make_adma_copy_tiled(
    CopyOp                  const& copy_op,
    Tensor<GEngine,GLayout> const& gtensor,     // Full GMEM Tensor
    SLayout                 const& slayout,     // CTA Tile of SMEM
    Layout<TShape,TStride>  const& cta_t_map,   // T: CTA thr idx -> logical TMA tid
    Layout<VShape,VStride>  const& cta_v_map)   // V: CTA val idx -> gmem mode
{
  MatrixDescriptor matrix_desc{};
  if constexpr(decltype(rank(flatten(slayout.shape())))::value > 2)
    matrix_desc = make_matrix_descriptor(coalesce(slayout));
  else
    matrix_desc = make_matrix_descriptor(slayout);
  Copy_Atom atom = make_adma_copy_atom<InternalType>(
      copy_op, gtensor, slayout, cosize(cta_t_map), matrix_desc, cta_v_map);

  //
  // Construct the TiledCopy
  //

  [[maybe_unused]] auto cta_tiler = product_each(shape(cta_v_map));

  auto num_elems_per_tma = size<1>(typename decltype(atom)::RefLayout{}) / static_value<sizeof_bits<typename GEngine::value_type>>();

  // smem idx -> smem coord
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
  // CTA V -> smem_coord
  auto layout_v = composition(inv_smem_layout, num_elems_per_tma);
  // Scale that up to cover all of the smem_coords
  auto layout_V = tile_to_shape(make_layout(layout_v), size(cta_v_map));
  // CTA T -> smem idx
  auto layout_t = make_layout(cosize(cta_t_map), safe_div(num_elems_per_tma, cosize(cta_t_map)));
  // CTA TID -> smem coord
  auto layout_T = composition(inv_smem_layout, composition(layout_t, cta_t_map));
  // Combine with the T mapping
  [[maybe_unused]] auto layout_TV = make_layout(layout_T, layout_V);

#if 0
  print("cta_tiler : "); print(cta_tiler); print("\n");
  print("layout_v : "); print(layout_v); print("\n");
  print("layout_V : "); print(layout_V); print("\n");
  print("layout_t : "); print(layout_t); print("\n");
  print("layout_T : "); print(layout_T); print("\n");
  print("layout_TV : "); print(layout_TV); print("\n");
#endif

  return TiledCopy<decltype(atom), decltype(layout_TV), decltype(cta_tiler)>{atom};
}

}

template <class InternalType = void,
         class CopyOp,
         class GEngine, class GLayout,
         class SLayout,
         class MMA_Tiler,
         class... Args,
         class ClusterShapeVMNK>
CUTE_HOST
auto
make_adma_atom_A_xe4(
    CopyOp                   const& copy_op,
    Tensor<GEngine, GLayout> const& gtensor,        // (M, K, ...)
    SLayout                  const& slayout,        // (MMA, MMA_M, MMA_K, ...)
    MMA_Tiler                const& mma_tiler,      // (TILE_M, TILE_N, TILE_K, ...)
    TiledMMA<Args...>        const& mma,
    ClusterShapeVMNK         const& cluster_shape   // (CTA_V, CTA_M, CTA_N, CTA_K)
) {
  // Keep only MK modes from MNK
  auto mma_tiler_mk = remove<1>(mma_tiler);

  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(mma_tiler_mk);

  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_A(g_tile))(_, repeat<rank(g_tile)>(_));

  // Matrix descriptor derivation:
  // Detect SF / MX-metadata layouts at compile time: they contain stride-0 broadcast
  // modes from SfAtom (size(slayout) != size(filter_zeros(slayout))). Data layouts
  // never have stride-0 modes. This allows make_adma_atom_A_xe4 to be used for both
  // data A and scale-factor SFA without separate functions.
  //   SF (broadcast detected)  -> Type3, Pitch = TILE_M >> 2
  //   Data (no broadcast)      -> Type1/Type2, Pitch derived from SMEM strides
  constexpr bool is_sf_layout = size(SLayout{}) != size(filter_zeros(SLayout{}));
  auto matrix_desc = [&]() -> uint32_t {
    if constexpr (is_sf_layout) {
      MatrixDescriptor sf_md{};
      sf_md.Type = MatrixDescriptor::Type3;
      sf_md.Pitch = get<0>(mma_tiler) >> 2;  // TILE_M >> 2
      return sf_md.raw_;
    } else {
      // TODO: fix with better implementation to handle all shapes.
      auto slayout_2d = [&]() {
        if constexpr (decltype(rank(slayout))::value >= 3) {
          return layout<0>(slayout);  // Hierarchical: extract first mode
        } else {
          return coalesce(slayout);   // Flat 2D: coalesce to handle any trivial modes
        }
      }();
      return detail::make_matrix_descriptor(slayout_2d, true);
    }
  }();

#if 0
  print("(tma_a) slayout:      "); print(slayout);      print("\n");
  print("(tma_a) g_tile:       "); print(g_tile);       print("\n");
  print("(tma_a) mma_tiler:    "); print(mma_tiler);    print("\n");
  print("(tma_a) cta_v_tile:   "); print(cta_v_tile);   print("\n");
#endif

  auto num_multicast = [&]() {
    if constexpr (is_same_v<CopyOp, XE4_ADMA_LOAD_MULTICAST>)
      return size<2>(cluster_shape);
    else if constexpr (is_same_v<CopyOp, XE4_ADMA_LOAD>)
      return Int<1>{};
    else if constexpr (is_same_v<CopyOp, XE4_ADMA_STORE>)
      return Int<1>{};
    else if constexpr (is_xe4_adma_store_reduce_v<CopyOp>)
      return Int<1>{};
    else if constexpr (is_same_v<CopyOp, XE4_ADMA_PREFETCH>)
      return Int<1>{};
    else
      static_assert(dependent_false<CopyOp>, "Unsupported CopyOp");
  }();

  using AmmaType = conditional_t<is_same<void, InternalType>::value, typename GEngine::value_type, InternalType>;
  return detail::make_adma_copy_atom<AmmaType>(
      copy_op, gtensor, slayout, num_multicast, matrix_desc, cta_v_tile);
}

template <class InternalType = void,
         class CopyOp,
         class GEngine, class GLayout,
         class SLayout,
         class MMA_Tiler,
         class... Args,
         class ClusterShapeVMNK>
CUTE_HOST
auto
make_adma_atom_B_xe4(
    CopyOp                   const& copy_op,
    Tensor<GEngine, GLayout> const& gtensor,        // (M, K, ...)
    SLayout                  const& slayout,        // (MMA, MMA_M, MMA_K, ...)
    MMA_Tiler                const& mma_tiler,      // (TILE_M, TILE_N, TILE_K, ...)
    TiledMMA<Args...>        const& mma,
    ClusterShapeVMNK         const& cluster_shape   // (CTA_V, CTA_M, CTA_N, CTA_K)
) {
  // Keep only NK modes from MNK
  auto mma_tiler_nk = remove<0>(mma_tiler);

  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(mma_tiler_nk);

  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_B(g_tile))(_, repeat<rank(g_tile)>(_));

  // Matrix descriptor derivation:
  // Detect SF / MX-metadata layouts at compile time via stride-0 broadcast modes.
  // See make_adma_atom_A_xe4 for full rationale.
  //   SF (broadcast detected)  -> Type3, Pitch = TILE_N >> 2
  //   Data (no broadcast)      -> Type1, Pitch derived from SMEM strides
  constexpr bool is_sf_layout = size(SLayout{}) != size(filter_zeros(SLayout{}));
  auto matrix_desc = [&]() -> uint32_t {
    if constexpr (is_sf_layout) {
      MatrixDescriptor sf_md{};
      sf_md.Type = MatrixDescriptor::Type3;
      sf_md.Pitch = get<1>(mma_tiler) >> 2;  // TILE_N >> 2
      return sf_md.raw_;
    } else {
      // TODO: fix with better implementation to handle all shapes.
      auto slayout_2d = [&]() {
        if constexpr (decltype(rank(slayout))::value >= 3) {
          return layout<0>(slayout);  // Hierarchical: extract first mode
        } else {
          return coalesce(slayout);   // Flat 2D: coalesce to handle any trivial modes
        }
      }();
      return detail::make_matrix_descriptor(slayout_2d, false);
    }
  }();

#if 0
  print("(tma_b) slayout:      "); print(slayout);      print("\n");
  print("(tma_b) g_tile:       "); print(g_tile);       print("\n");
  print("(tma_b) mma_tiler:    "); print(mma_tiler);    print("\n");
  print("(tma_b) cta_v_tile:   "); print(cta_v_tile);   print("\n");
#endif

  auto num_multicast = [&]() {
    if constexpr (is_same_v<CopyOp, XE4_ADMA_LOAD_MULTICAST>)
      return size<1>(cluster_shape);
    else if constexpr (is_same_v<CopyOp, XE4_ADMA_LOAD>)
      return Int<1>{};
    else if constexpr (is_same_v<CopyOp, XE4_ADMA_STORE>)
      return Int<1>{};
    else if constexpr (is_xe4_adma_store_reduce_v<CopyOp>)
      return Int<1>{};
    else if constexpr (is_same_v<CopyOp, XE4_ADMA_PREFETCH>)
      return Int<1>{};
    else
      static_assert(dependent_false<CopyOp>, "Unsupported CopyOp");
  }();

  using AmmaType = conditional_t<is_same<void, InternalType>::value, typename GEngine::value_type, InternalType>;
  return detail::make_adma_copy_atom<AmmaType>(
      copy_op, gtensor, slayout, num_multicast, matrix_desc, cta_v_tile);
}

template <class InternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size>
CUTE_HOST_RTC
auto
make_adma_copy(CopyOp                 const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout,
              CTA_Tiler               const& cta_tiler,
              Cluster_Size            const& cluster_size)
{
  if constexpr (false) { // TODO: Add im2col support
    return make_im2col_tma_copy(copy_op,
                                gtensor,
                                slayout,
                                cta_tiler,
                                cluster_size);
  } else {
    auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
    auto cta_t_tile = make_layout(cluster_size);
    // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
    using AdmaType = conditional_t<is_same<void, InternalType>::value, typename GEngine::value_type, InternalType>;
    return detail::make_adma_copy_tiled<AdmaType>(copy_op,
                                                gtensor, slayout,
                                                cta_t_tile, cta_v_tile);
  }
}

// Convenience functions for creating ADMA prefetch atoms for A and B operands.
// These delegate to make_adma_atom_A/B_xe4 with XE4_ADMA_PREFETCH as the CopyOp,
// producing a directly-executable Copy_Atom<Copy_Traits<XE4_ADMA_PREFETCH, ...>>.
// Usage: auto pf_atom_a = make_adma_prefetch_atom_A_xe4(gA, slayout, tiler, mma, cluster);
//        copy(pf_atom_a, tAgA(_,stage), dummy_dst);  // fire-and-forget

template <class InternalType = void,
          class GEngine, class GLayout,
          class SLayout,
          class MMA_Tiler,
          class... Args,
          class ClusterShapeVMNK>
CUTE_HOST
auto
make_adma_prefetch_atom_A_xe4(
    Tensor<GEngine, GLayout> const& gtensor,
    SLayout                  const& slayout,
    MMA_Tiler                const& mma_tiler,
    TiledMMA<Args...>        const& mma,
    ClusterShapeVMNK         const& cluster_shape)
{
  return make_adma_atom_A_xe4<InternalType>(
      XE4_ADMA_PREFETCH{}, gtensor, slayout, mma_tiler, mma, cluster_shape);
}

template <class InternalType = void,
          class GEngine, class GLayout,
          class SLayout,
          class MMA_Tiler,
          class... Args,
          class ClusterShapeVMNK>
CUTE_HOST
auto
make_adma_prefetch_atom_B_xe4(
    Tensor<GEngine, GLayout> const& gtensor,
    SLayout                  const& slayout,
    MMA_Tiler                const& mma_tiler,
    TiledMMA<Args...>        const& mma,
    ClusterShapeVMNK         const& cluster_shape)
{
  return make_adma_atom_B_xe4<InternalType>(
      XE4_ADMA_PREFETCH{}, gtensor, slayout, mma_tiler, mma, cluster_shape);
}

}
