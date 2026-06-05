#pragma once

#include <sycl/sycl.hpp>

#include <cute/arch/mma_xe4_desc.hpp>
#include <cute/arch/copy_xe4_adma.hpp>
#include <cute/arch/xe4_util.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/atom/copy_atom.hpp>

#include <cute/atom/copy_traits_xe4_tma.hpp>

#include <cute/layout.hpp>
#include <stdexcept>
#include <cstdio>

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

template <BarrierType BType> struct XE4_ADMA_LINEAR_LOAD_OP;
template <BarrierType BType> struct XE4_ADMA_LINEAR_STORE_OP;
struct XE4_ADMA_LINEAR_LOAD_MULTICAST_CLUSTER_OP;

template <BarrierType BType, class CopySizeInBytes>
struct Copy_Traits<XE4_ADMA_LINEAR_LOAD<BType>, CopySizeInBytes>
{
  static_assert(int32_t(CopySizeInBytes::value) % 16 == 0,
                "ADMA Linear Load requires copy size in Bytes to be aligned to 16B.");

  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, decltype(CopySizeInBytes{} * C<8>{})>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  template <detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_LINEAR_LOAD_OP<BType>, CopySizeInBytes, detail::CacheHint<CC>>
  with(uint64_t* abar_ptr, detail::CacheHint<CC> = {}) const {
    return {abar_ptr};
  }

  // Not directly executable — must call .with(abar_ptr) first.
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

template <BarrierType BType>
struct XE4_ADMA_LINEAR_LOAD_OP : XE4_ADMA_LINEAR_LOAD<BType> {};

template <BarrierType BType, class CopySizeInBytes, detail::CacheCtrl CC>
struct Copy_Traits<XE4_ADMA_LINEAR_LOAD_OP<BType>, CopySizeInBytes, detail::CacheHint<CC>>
{
  static_assert(int32_t(CopySizeInBytes::value) % 16 == 0,
                "ADMA Linear Load requires copy size in Bytes to be aligned to 16B.");

  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, decltype(CopySizeInBytes{} * C<8>{})>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  uint64_t* abar_ptr_;

  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem src for XE4_ADMA_LINEAR_LOAD");
    static_assert(is_smem<TD>::value, "Expected smem dst for XE4_ADMA_LINEAR_LOAD");
    XE4_ADMA_LINEAR_LOAD<BType>::template copy<CC>(
        raw_pointer_cast(dst.data()),
        const_cast<void*>(static_cast<const void*>(raw_pointer_cast(src.data()))),
        int32_t(CopySizeInBytes::value),
        traits.abar_ptr_);
  }
};

template <BarrierType BType, class CopySizeInBytes>
struct Copy_Traits<XE4_ADMA_LINEAR_STORE<BType>, CopySizeInBytes>
{
  static_assert(int32_t(CopySizeInBytes::value) % 16 == 0,
                "ADMA Linear Store requires copy size in Bytes to be aligned to 16B.");

  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, decltype(CopySizeInBytes{} * C<8>{})>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  template <detail::CacheCtrl CC = detail::CacheCtrl::L2wb_L3uc>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_LINEAR_STORE_OP<BType>, CopySizeInBytes, detail::CacheHint<CC>>
  with(uint64_t* abar_ptr, detail::CacheHint<CC> = {}) const {
    return {abar_ptr};
  }

  // Not directly executable — must call .with(abar_ptr) first.
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

template <BarrierType BType>
struct XE4_ADMA_LINEAR_STORE_OP : XE4_ADMA_LINEAR_STORE<BType> {};

template <BarrierType BType, class CopySizeInBytes, detail::CacheCtrl CC>
struct Copy_Traits<XE4_ADMA_LINEAR_STORE_OP<BType>, CopySizeInBytes, detail::CacheHint<CC>>
{
  static_assert(int32_t(CopySizeInBytes::value) % 16 == 0,
                "ADMA Linear Store requires copy size in Bytes to be aligned to 16B.");

  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, decltype(CopySizeInBytes{} * C<8>{})>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  uint64_t* abar_ptr_;

  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_smem<TS>::value, "Expected smem src for XE4_ADMA_LINEAR_STORE");
    static_assert(is_gmem<TD>::value, "Expected gmem dst for XE4_ADMA_LINEAR_STORE");
    XE4_ADMA_LINEAR_STORE<BType>::template copy<CC>(
        raw_pointer_cast(dst.data()),
        const_cast<void*>(static_cast<const void*>(raw_pointer_cast(src.data()))),
        int32_t(CopySizeInBytes::value),
        traits.abar_ptr_);
  }
};

template <class CopySizeInBytes>
struct Copy_Traits<XE4_ADMA_LINEAR_PREFETCH, CopySizeInBytes>
{
  static_assert(int32_t(CopySizeInBytes::value) % 16 == 0,
                "ADMA Linear Prefetch requires copy size in Bytes to be aligned to 16B.");

  using ThrID = Layout<_1>;
  using SrcLayout = Layout<Shape<_1,decltype(CopySizeInBytes{} * C<8>{})>>;
  using DstLayout = Layout<Shape<_1,decltype(CopySizeInBytes{} * C<8>{})>>;
  using RefLayout = SrcLayout;

  template <class TS, class SLayout,
            class TD, class DLayout>
  friend CUTE_HOST_DEVICE constexpr void
  copy_unpack([[maybe_unused]] Copy_Traits const& traits,
              Tensor<TS,SLayout>          const& src,
              [[maybe_unused]] Tensor<TD,DLayout>& dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem src for XE4_ADMA_LINEAR_PREFETCH");
    XE4_ADMA_LINEAR_PREFETCH::copy(
        const_cast<void*>(static_cast<const void*>(raw_pointer_cast(src.data()))),
        int32_t(CopySizeInBytes::value));
  }
};

template <typename T, RedOp Rop, BarrierType BType,
          class CopySizeInBytes, class... OpArgs>
struct Copy_Traits<XE4_ADMA_LINEAR_REDUCE<T, Rop, BType>, CopySizeInBytes, OpArgs...>
{
  static_assert(int32_t(CopySizeInBytes::value) % 16 == 0,
                "ADMA Linear Reduce requires copy size in Bytes to be aligned to 16B.");

  using ThrID = Layout<_1>;
  using SrcLayout = Layout<Shape<_1,decltype(CopySizeInBytes{} * C<8>{})>>;
  using DstLayout = Layout<Shape<_1,decltype(CopySizeInBytes{} * C<8>{})>>;
  using RefLayout = SrcLayout;

  cute::tuple<OpArgs...> reduce_abar_;

  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_LINEAR_REDUCE<T, Rop, BType>, CopySizeInBytes, uint64_t*>
  with(uint64_t* abar_ptr) const {
    return {abar_ptr};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  friend CUTE_HOST_DEVICE constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_same<cute::tuple<OpArgs...>, cute::tuple<uint64_t*>>::value,
                  "Extra arguments not set. Set .with() before use.");
    static_assert(is_smem<TS>::value, "Expected smem src for XE4_ADMA_LINEAR_REDUCE");
    static_assert(is_gmem<TD>::value, "Expected gmem dst for XE4_ADMA_LINEAR_REDUCE");
    XE4_ADMA_LINEAR_REDUCE<T, Rop, BType>::copy(
                               raw_pointer_cast(dst.data()),
                               const_cast<void*>(static_cast<const void*>(raw_pointer_cast(src.data()))),
                               int32_t(CopySizeInBytes::value), get<0>(traits.reduce_abar_));
  }
};
template <class CopySizeInBytes>
struct Copy_Traits<XE4_ADMA_LINEAR_LOAD_MULTICAST_CLUSTER, CopySizeInBytes>
{
  static_assert(int32_t(CopySizeInBytes::value) % 16 == 0,
                "ADMA Linear Load (cluster) requires copy size in bytes to be 16B-aligned.");

  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, decltype(CopySizeInBytes{} * C<8>{})>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  template <detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_LINEAR_LOAD_MULTICAST_CLUSTER_OP, CopySizeInBytes, detail::CacheHint<CC>>
  with(uint64_t* abar_ptr, uint32_t multicast_mask, detail::CacheHint<CC> = {}) const {
    return {abar_ptr, multicast_mask};
  }

  // Not directly executable — must call .with(abar_ptr, multicast_mask) first.
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS, SLayout> const& src,
              Tensor<TD, DLayout>      & dst) = delete;
};

struct XE4_ADMA_LINEAR_LOAD_MULTICAST_CLUSTER_OP : XE4_ADMA_LINEAR_LOAD_MULTICAST_CLUSTER {};

template <class CopySizeInBytes, detail::CacheCtrl CC>
struct Copy_Traits<XE4_ADMA_LINEAR_LOAD_MULTICAST_CLUSTER_OP, CopySizeInBytes, detail::CacheHint<CC>>
{
  static_assert(int32_t(CopySizeInBytes::value) % 16 == 0,
                "ADMA Linear Load (cluster) requires copy size in bytes to be 16B-aligned.");

  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, decltype(CopySizeInBytes{} * C<8>{})>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  uint64_t* abar_ptr_;
  uint32_t  multicast_mask_;

  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS, SLayout> const& src,
              Tensor<TD, DLayout>      & dst)
  {
    static_assert(is_gmem<TS>::value,
                  "XE4_ADMA_LINEAR_LOAD_MULTICAST_CLUSTER requires global-memory source.");
    static_assert(is_smem<TD>::value,
                  "XE4_ADMA_LINEAR_LOAD_MULTICAST_CLUSTER requires shared-memory destination.");
    XE4_ADMA_LINEAR_LOAD_MULTICAST_CLUSTER::copy<CC>(
        raw_pointer_cast(dst.data()),
        const_cast<void*>(static_cast<const void*>(raw_pointer_cast(src.data()))),
        int32_t(CopySizeInBytes::value),
        traits.abar_ptr_,
        traits.multicast_mask_);
  }
};

template <class CopySizeInBytes, class... OpArgs>
struct Copy_Traits<XE4_ADMA_LINEAR_LOAD_LOCAL_TO_REMOTE_SLM_CLUSTER, CopySizeInBytes, OpArgs...>
{
  static_assert(int32_t(CopySizeInBytes::value) % 16 == 0,
                "ADMA Linear Load (Local to Remote SLM) requires copy size in bytes to be 16B-aligned.");

  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, decltype(CopySizeInBytes{} * C<8>{})>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  cute::tuple<OpArgs...> opargs_;

  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_LINEAR_LOAD_LOCAL_TO_REMOTE_SLM_CLUSTER, CopySizeInBytes, uint64_t*, uint32_t>
  with(uint64_t* abar_ptr, uint32_t multicast_mask) const {
    return {make_tuple(abar_ptr, multicast_mask)};
  }

  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS, SLayout> const& src,
              Tensor<TD, DLayout>      & dst)
  {
    static_assert(is_same<cute::tuple<OpArgs...>, cute::tuple<uint64_t*, uint32_t>>::value,
                  "abar and multicast_mask not set. Call .with(abar_ptr, multicast_mask) before use.");
    static_assert(is_smem<TS>::value,
                  "XE4_ADMA_LINEAR_LOAD_LOCAL_TO_REMOTE_SLM_CLUSTER requires shared-memory source.");
    static_assert(is_smem<TD>::value,
                  "XE4_ADMA_LINEAR_LOAD_LOCAL_TO_REMOTE_SLM_CLUSTER requires shared-memory destination.");
    XE4_ADMA_LINEAR_LOAD_LOCAL_TO_REMOTE_SLM_CLUSTER::copy(
        raw_pointer_cast(dst.data()),
        const_cast<void*>(static_cast<const void*>(raw_pointer_cast(src.data()))),
        int32_t(CopySizeInBytes::value),
        get<0>(traits.opargs_),
        get<1>(traits.opargs_));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copy_Traits for XE4_ADMA_ROW_COPY_LINEAR_LOAD - Unified for all addressing modes
/// RowSize is a compile-time power-of-2 in [16, 2048] selecting the hardware instruction variant.
/// IsPerWarp selects atom thread layout: false = per-lane (Layout<_1>), true = per-warp (Layout<_32>).
/// Mode selects the addressing scheme; baked into the trait type so .with() and copy_unpack
/// dispatch via `if constexpr (Mode == ...)` without runtime tags.
///   - A64:  .with(uint64_t gmem_addr, uint32_t size, uint64_t* abar_ptr, ...)        // no offset
///   - A32S: .with(DataType* gmem_ptr, int32_t  offset, uint32_t size, uint64_t* abar_ptr, ...)
///   - A32U: .with(DataType* gmem_ptr, uint32_t offset, uint32_t size, uint64_t* abar_ptr, ...)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class DataType, class NumBytes, class RowSizeT,
          bool IsPerWarp, AddressingMode Mode, class... OpArgs>
struct Copy_Traits<XE4_ADMA_ROW_COPY_LINEAR_LOAD,
                   DataType, NumBytes, RowSizeT,
                   cute::C<IsPerWarp>, cute::C<Mode>, OpArgs...>
{
  static_assert(int32_t(NumBytes::value) % 16 == 0,
                "ADMA Row Load requires copy size in Bytes to be aligned to 16B.");
  static constexpr uint32_t kRowSize = RowSizeT::value;
  static_assert(detail::is_valid_row_size_v<kRowSize>,
                "RowSize must be a power of 2 in [16, 2048]");

  using ThrID = std::conditional_t<IsPerWarp, Layout<_32>, Layout<_1>>;
  using SrcLayout = std::conditional_t<
      IsPerWarp,
      Layout<Shape<_32, decltype(NumBytes{} * C<8>{})>,
             Stride<decltype(NumBytes{} * C<8>{}), _1>>,
      Layout<Shape<_1, decltype(NumBytes{} * C<8>{})>>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  // Arguments stored by .with() - exact args depend on Mode.
  cute::tuple<OpArgs...> row_load_args_;

  // A64: .with(uint64_t gmem_addr, uint32_t size, uint64_t* abar_ptr, ...)
  // A64 has no offset operand — gmem_addr is a 64-bit absolute address.
  template <AddressingMode M = Mode,
            detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero,
            std::enable_if_t<M == AddressingMode::A64, int> = 0>
  CUTE_HOST_DEVICE constexpr auto
  with(uint64_t gmem_addr, uint32_t size, uint64_t* abar_ptr,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return Copy_Traits<XE4_ADMA_ROW_COPY_LINEAR_LOAD,
                       DataType, NumBytes, RowSizeT,
                       cute::C<IsPerWarp>, cute::C<Mode>,
                       uint64_t, uint32_t, uint64_t*,
                       detail::CacheHint<CC>, detail::FillMode<FM>>{
      cute::make_tuple(gmem_addr, size, abar_ptr,
                       detail::CacheHint<CC>{}, detail::FillMode<FM>{})};
  }

  // A32S: .with(DataType* gmem_ptr, int32_t offset, uint32_t size, uint64_t* abar_ptr, ...)
  template <AddressingMode M = Mode,
            detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero,
            std::enable_if_t<M == AddressingMode::A32S, int> = 0>
  CUTE_HOST_DEVICE constexpr auto
  with(DataType* gmem_ptr, int32_t offset, uint32_t size, uint64_t* abar_ptr,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return Copy_Traits<XE4_ADMA_ROW_COPY_LINEAR_LOAD,
                       DataType, NumBytes, RowSizeT,
                       cute::C<IsPerWarp>, cute::C<Mode>,
                       DataType*, int32_t, uint32_t, uint64_t*,
                       detail::CacheHint<CC>, detail::FillMode<FM>>{
      cute::make_tuple(gmem_ptr, offset, size, abar_ptr,
                       detail::CacheHint<CC>{}, detail::FillMode<FM>{})};
  }

  // A32U: .with(DataType* gmem_ptr, uint32_t offset, uint32_t size, uint64_t* abar_ptr, ...)
  template <AddressingMode M = Mode,
            detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero,
            std::enable_if_t<M == AddressingMode::A32U, int> = 0>
  CUTE_HOST_DEVICE constexpr auto
  with(DataType* gmem_ptr, uint32_t offset, uint32_t size, uint64_t* abar_ptr,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return Copy_Traits<XE4_ADMA_ROW_COPY_LINEAR_LOAD,
                       DataType, NumBytes, RowSizeT,
                       cute::C<IsPerWarp>, cute::C<Mode>,
                       DataType*, uint32_t, uint32_t, uint64_t*,
                       detail::CacheHint<CC>, detail::FillMode<FM>>{
      cute::make_tuple(gmem_ptr, offset, size, abar_ptr,
                       detail::CacheHint<CC>{}, detail::FillMode<FM>{})};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  friend CUTE_HOST_DEVICE constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem src for XE4_ADMA_ROW_COPY_LINEAR_LOAD");
    static_assert(is_smem<TD>::value, "Expected smem dst for XE4_ADMA_ROW_COPY_LINEAR_LOAD");
    static_assert(std::is_same_v<typename TD::value_type, DataType>,
                  "ADMA row load requires smem element type to match gmem DataType");
    static_assert(sizeof...(OpArgs) > 0, "Extra arguments not set. Set .with() before use.");

    if constexpr (Mode == AddressingMode::A64) {
      // A64: (gmem_addr, size, abar_ptr, CacheHint, FillMode) — no offset.
      constexpr auto CC = decltype(get<3>(traits.row_load_args_))::value;
      constexpr auto FM = decltype(get<4>(traits.row_load_args_))::value;
      XE4_ADMA_ROW_COPY_LINEAR_LOAD::template copy<AddressingMode::A64, kRowSize, DataType, CC, FM>(
          raw_pointer_cast(dst.data()),
          get<0>(traits.row_load_args_),  // gmem_addr (uint64_t absolute address)
          get<1>(traits.row_load_args_),  // size
          get<2>(traits.row_load_args_),  // abar_ptr
          get<3>(traits.row_load_args_),  // CacheHint
          get<4>(traits.row_load_args_)); // FillMode
    } else {
      // A32S / A32U: (gmem_ptr, offset, size, abar_ptr, CacheHint, FillMode)
      constexpr auto CC = decltype(get<4>(traits.row_load_args_))::value;
      constexpr auto FM = decltype(get<5>(traits.row_load_args_))::value;
      XE4_ADMA_ROW_COPY_LINEAR_LOAD::template copy<Mode, kRowSize, DataType, CC, FM>(
          raw_pointer_cast(dst.data()),
          get<0>(traits.row_load_args_),  // gmem_ptr
          get<1>(traits.row_load_args_),  // offset (int32_t / uint32_t)
          get<2>(traits.row_load_args_),  // size
          get<3>(traits.row_load_args_),  // abar_ptr
          get<4>(traits.row_load_args_),  // CacheHint
          get<5>(traits.row_load_args_)); // FillMode
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copy_Traits for XE4_ADMA_ROW_COPY_LINEAR_LOAD_MULTICAST - Same as LOAD but with wg_mask.
/// IsPerWarp + Mode behave as in XE4_ADMA_ROW_COPY_LINEAR_LOAD.
///   - A64:  .with(uint64_t gmem_addr, uint32_t size, uint64_t* abar_ptr, uint32_t wg_mask, ...)
///   - A32S: .with(DataType* gmem_ptr, int32_t  offset, uint32_t size, uint64_t* abar_ptr, uint32_t wg_mask, ...)
///   - A32U: .with(DataType* gmem_ptr, uint32_t offset, uint32_t size, uint64_t* abar_ptr, uint32_t wg_mask, ...)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class DataType, class NumBytes, class RowSizeT,
          bool IsPerWarp, AddressingMode Mode, class... OpArgs>
struct Copy_Traits<XE4_ADMA_ROW_COPY_LINEAR_LOAD_MULTICAST,
                   DataType, NumBytes, RowSizeT,
                   cute::C<IsPerWarp>, cute::C<Mode>, OpArgs...>
{
  static_assert(int32_t(NumBytes::value) % 16 == 0,
                "ADMA Row Load Multicast requires copy size in Bytes to be aligned to 16B.");
  static constexpr uint32_t kRowSize = RowSizeT::value;
  static_assert(detail::is_valid_row_size_v<kRowSize>,
                "RowSize must be a power of 2 in [16, 2048]");

  using ThrID = std::conditional_t<IsPerWarp, Layout<_32>, Layout<_1>>;
  using SrcLayout = std::conditional_t<
      IsPerWarp,
      Layout<Shape<_32, decltype(NumBytes{} * C<8>{})>,
             Stride<decltype(NumBytes{} * C<8>{}), _1>>,
      Layout<Shape<_1, decltype(NumBytes{} * C<8>{})>>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  cute::tuple<OpArgs...> row_load_args_;

  // A64: gmem_addr is a 64-bit absolute address; no offset operand.
  template <AddressingMode M = Mode,
            detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero,
            std::enable_if_t<M == AddressingMode::A64, int> = 0>
  CUTE_HOST_DEVICE constexpr auto
  with(uint64_t gmem_addr, uint32_t size, uint64_t* abar_ptr, uint32_t wg_mask,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return Copy_Traits<XE4_ADMA_ROW_COPY_LINEAR_LOAD_MULTICAST,
                       DataType, NumBytes, RowSizeT,
                       cute::C<IsPerWarp>, cute::C<Mode>,
                       uint64_t, uint32_t, uint64_t*, uint32_t,
                       detail::CacheHint<CC>, detail::FillMode<FM>>{
      cute::make_tuple(gmem_addr, size, abar_ptr, wg_mask,
                       detail::CacheHint<CC>{}, detail::FillMode<FM>{})};
  }

  // A32S
  template <AddressingMode M = Mode,
            detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero,
            std::enable_if_t<M == AddressingMode::A32S, int> = 0>
  CUTE_HOST_DEVICE constexpr auto
  with(DataType* gmem_ptr, int32_t offset, uint32_t size, uint64_t* abar_ptr, uint32_t wg_mask,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return Copy_Traits<XE4_ADMA_ROW_COPY_LINEAR_LOAD_MULTICAST,
                       DataType, NumBytes, RowSizeT,
                       cute::C<IsPerWarp>, cute::C<Mode>,
                       DataType*, int32_t, uint32_t, uint64_t*, uint32_t,
                       detail::CacheHint<CC>, detail::FillMode<FM>>{
      cute::make_tuple(gmem_ptr, offset, size, abar_ptr, wg_mask,
                       detail::CacheHint<CC>{}, detail::FillMode<FM>{})};
  }

  // A32U
  template <AddressingMode M = Mode,
            detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero,
            std::enable_if_t<M == AddressingMode::A32U, int> = 0>
  CUTE_HOST_DEVICE constexpr auto
  with(DataType* gmem_ptr, uint32_t offset, uint32_t size, uint64_t* abar_ptr, uint32_t wg_mask,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return Copy_Traits<XE4_ADMA_ROW_COPY_LINEAR_LOAD_MULTICAST,
                       DataType, NumBytes, RowSizeT,
                       cute::C<IsPerWarp>, cute::C<Mode>,
                       DataType*, uint32_t, uint32_t, uint64_t*, uint32_t,
                       detail::CacheHint<CC>, detail::FillMode<FM>>{
      cute::make_tuple(gmem_ptr, offset, size, abar_ptr, wg_mask,
                       detail::CacheHint<CC>{}, detail::FillMode<FM>{})};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  friend CUTE_HOST_DEVICE constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem src for XE4_ADMA_ROW_COPY_LINEAR_LOAD_MULTICAST");
    static_assert(is_smem<TD>::value, "Expected smem dst for XE4_ADMA_ROW_COPY_LINEAR_LOAD_MULTICAST");
    static_assert(std::is_same_v<typename TD::value_type, DataType>,
                  "ADMA row load multicast requires smem element type to match gmem DataType");
    static_assert(sizeof...(OpArgs) > 0, "Extra arguments not set. Set .with() before use.");

    if constexpr (Mode == AddressingMode::A64) {
      // A64: (gmem_addr, size, abar_ptr, wg_mask, CacheHint, FillMode) — no offset.
      constexpr auto CC = decltype(get<4>(traits.row_load_args_))::value;
      constexpr auto FM = decltype(get<5>(traits.row_load_args_))::value;
      XE4_ADMA_ROW_COPY_LINEAR_LOAD_MULTICAST::template copy<AddressingMode::A64, kRowSize, DataType, CC, FM>(
          raw_pointer_cast(dst.data()),
          get<0>(traits.row_load_args_),  // gmem_addr (uint64_t absolute address)
          get<1>(traits.row_load_args_),  // size
          get<2>(traits.row_load_args_),  // abar_ptr
          get<3>(traits.row_load_args_),  // wg_mask
          get<4>(traits.row_load_args_),  // CacheHint
          get<5>(traits.row_load_args_)); // FillMode
    } else {
      // A32S / A32U: (gmem_ptr, offset, size, abar_ptr, wg_mask, CacheHint, FillMode)
      constexpr auto CC = decltype(get<5>(traits.row_load_args_))::value;
      constexpr auto FM = decltype(get<6>(traits.row_load_args_))::value;
      XE4_ADMA_ROW_COPY_LINEAR_LOAD_MULTICAST::template copy<Mode, kRowSize, DataType, CC, FM>(
          raw_pointer_cast(dst.data()),
          get<0>(traits.row_load_args_),  // gmem_ptr
          get<1>(traits.row_load_args_),  // offset (int32_t / uint32_t)
          get<2>(traits.row_load_args_),  // size
          get<3>(traits.row_load_args_),  // abar_ptr
          get<4>(traits.row_load_args_),  // wg_mask
          get<5>(traits.row_load_args_),  // CacheHint
          get<6>(traits.row_load_args_)); // FillMode
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copy_Traits for XE4_ADMA_ROW_COPY_LINEAR_STORE - Unified for all addressing modes
/// RowSize is a compile-time power-of-2 in [16, 2048] selecting the hardware instruction variant.
/// Mode (baked into the trait type) selects the addressing scheme:
///   - A64:  .with(uint64_t gmem_addr, uint32_t size, uint64_t* abar_ptr, ...)        // no offset
///   - A32S: .with(DataType* gmem_ptr, int32_t  offset, uint32_t size, uint64_t* abar_ptr, ...)
///   - A32U: .with(DataType* gmem_ptr, uint32_t offset, uint32_t size, uint64_t* abar_ptr, ...)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class DataType, class NumBytes, class RowSizeT,
          bool IsPerWarp, AddressingMode Mode, class... OpArgs>
struct Copy_Traits<XE4_ADMA_ROW_COPY_LINEAR_STORE,
                   DataType, NumBytes, RowSizeT,
                   cute::C<IsPerWarp>, cute::C<Mode>, OpArgs...>
{
  static_assert(int32_t(NumBytes::value) % 16 == 0,
                "ADMA Row Store requires copy size in Bytes to be aligned to 16B.");
  static constexpr uint32_t kRowSize = RowSizeT::value;
  static_assert(detail::is_valid_row_size_v<kRowSize>,
                "RowSize must be a power of 2 in [16, 2048]");

  using ThrID = std::conditional_t<IsPerWarp, Layout<_32>, Layout<_1>>;
  using SrcLayout = std::conditional_t<
      IsPerWarp,
      Layout<Shape<_32, decltype(NumBytes{} * C<8>{})>,
             Stride<decltype(NumBytes{} * C<8>{}), _1>>,
      Layout<Shape<_1, decltype(NumBytes{} * C<8>{})>>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  cute::tuple<OpArgs...> row_store_args_;

  // A64: gmem_addr is a 64-bit absolute address; no offset operand.
  template <AddressingMode M = Mode,
            detail::CacheCtrl CC = detail::CacheCtrl::L2wb_L3uc,
            detail::CompletionMode CM = detail::CompletionMode::CM_Unspecified,
            std::enable_if_t<M == AddressingMode::A64, int> = 0>
  CUTE_HOST_DEVICE constexpr auto
  with(uint64_t gmem_addr, uint32_t size, uint64_t* abar_ptr,
       detail::CacheHint<CC> = {}, detail::CompletionModeHint<CM> = {}) const {
    return Copy_Traits<XE4_ADMA_ROW_COPY_LINEAR_STORE,
                       DataType, NumBytes, RowSizeT,
                       cute::C<IsPerWarp>, cute::C<Mode>,
                       uint64_t, uint32_t, uint64_t*,
                       detail::CacheHint<CC>, detail::CompletionModeHint<CM>>{
      cute::make_tuple(gmem_addr, size, abar_ptr,
                       detail::CacheHint<CC>{}, detail::CompletionModeHint<CM>{})};
  }

  // A32S
  template <AddressingMode M = Mode,
            detail::CacheCtrl CC = detail::CacheCtrl::L2wb_L3uc,
            detail::CompletionMode CM = detail::CompletionMode::CM_Unspecified,
            std::enable_if_t<M == AddressingMode::A32S, int> = 0>
  CUTE_HOST_DEVICE constexpr auto
  with(DataType* gmem_ptr, int32_t offset, uint32_t size, uint64_t* abar_ptr,
       detail::CacheHint<CC> = {}, detail::CompletionModeHint<CM> = {}) const {
    return Copy_Traits<XE4_ADMA_ROW_COPY_LINEAR_STORE,
                       DataType, NumBytes, RowSizeT,
                       cute::C<IsPerWarp>, cute::C<Mode>,
                       DataType*, int32_t, uint32_t, uint64_t*,
                       detail::CacheHint<CC>, detail::CompletionModeHint<CM>>{
      cute::make_tuple(gmem_ptr, offset, size, abar_ptr,
                       detail::CacheHint<CC>{}, detail::CompletionModeHint<CM>{})};
  }

  // A32U
  template <AddressingMode M = Mode,
            detail::CacheCtrl CC = detail::CacheCtrl::L2wb_L3uc,
            detail::CompletionMode CM = detail::CompletionMode::CM_Unspecified,
            std::enable_if_t<M == AddressingMode::A32U, int> = 0>
  CUTE_HOST_DEVICE constexpr auto
  with(DataType* gmem_ptr, uint32_t offset, uint32_t size, uint64_t* abar_ptr,
       detail::CacheHint<CC> = {}, detail::CompletionModeHint<CM> = {}) const {
    return Copy_Traits<XE4_ADMA_ROW_COPY_LINEAR_STORE,
                       DataType, NumBytes, RowSizeT,
                       cute::C<IsPerWarp>, cute::C<Mode>,
                       DataType*, uint32_t, uint32_t, uint64_t*,
                       detail::CacheHint<CC>, detail::CompletionModeHint<CM>>{
      cute::make_tuple(gmem_ptr, offset, size, abar_ptr,
                       detail::CacheHint<CC>{}, detail::CompletionModeHint<CM>{})};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  friend CUTE_HOST_DEVICE constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_smem<TS>::value, "Expected smem src for XE4_ADMA_ROW_COPY_LINEAR_STORE");
    static_assert(is_gmem<TD>::value, "Expected gmem dst for XE4_ADMA_ROW_COPY_LINEAR_STORE");
    static_assert(std::is_same_v<typename TS::value_type, DataType>,
                  "ADMA row store requires smem element type to match gmem DataType");
    static_assert(sizeof...(OpArgs) > 0, "Extra arguments not set. Set .with() before use.");

    if constexpr (Mode == AddressingMode::A64) {
      // A64: (gmem_addr, size, abar_ptr, CacheHint, CompletionModeHint) — no offset.
      constexpr auto CC = decltype(get<3>(traits.row_store_args_))::value;
      constexpr auto CM = decltype(get<4>(traits.row_store_args_))::value;
      XE4_ADMA_ROW_COPY_LINEAR_STORE::template copy<AddressingMode::A64, kRowSize, DataType, CC, CM>(
          const_cast<DataType*>(raw_pointer_cast(src.data())),
          get<0>(traits.row_store_args_),  // gmem_addr (uint64_t absolute address)
          get<1>(traits.row_store_args_),  // size
          get<2>(traits.row_store_args_),  // abar_ptr
          get<3>(traits.row_store_args_),  // CacheHint
          get<4>(traits.row_store_args_)); // CompletionModeHint
    } else {
      // A32S / A32U: (gmem_ptr, offset, size, abar, CacheHint, CompletionModeHint)
      constexpr auto CC = decltype(get<4>(traits.row_store_args_))::value;
      constexpr auto CM = decltype(get<5>(traits.row_store_args_))::value;
      XE4_ADMA_ROW_COPY_LINEAR_STORE::template copy<Mode, kRowSize, DataType, CC, CM>(
          const_cast<DataType*>(raw_pointer_cast(src.data())),
          get<0>(traits.row_store_args_),  // gmem_ptr
          get<1>(traits.row_store_args_),  // offset (int32_t / uint32_t)
          get<2>(traits.row_store_args_),  // size
          get<3>(traits.row_store_args_),  // abar_ptr
          get<4>(traits.row_store_args_),  // CacheHint
          get<5>(traits.row_store_args_)); // CompletionModeHint
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copy_Traits for XE4_ADMA_ROW_PREFETCH - Unified for all addressing modes
/// RowSize is a compile-time power-of-2 in [16, 2048] selecting the hardware instruction variant.
/// Atom thread granularity is selected by the cute::C<bool IsPerWarp> slot:
///   cute::C<false> = one atom per lane    (ThrLayoutCopy = Layout<_1>)
///   cute::C<true>  = one atom per 32-lane warp (ThrLayoutCopy = Layout<_32>)
/// Addressing mode is determined by .with() overload signature:
///   - A64:  .with(void* gmem_ptr, uint32_t size, ...)
///   - A32S: .with(void* gmem_ptr, int32_t offset, uint32_t size, ...)
///   - A32U: .with(void* gmem_ptr, uint32_t offset, uint32_t size, ...)
/// CacheCtrl supported: L2uc_L3c, L2c_L3uc (default), L2c_L3c.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class NumBytes, class RowSizeT,
          bool IsPerWarp, AddressingMode Mode, class... OpArgs>
struct Copy_Traits<XE4_ADMA_ROW_PREFETCH,
                   NumBytes, RowSizeT,
                   cute::C<IsPerWarp>, cute::C<Mode>, OpArgs...>
{
  static_assert(int32_t(NumBytes::value) % 16 == 0,
                "ADMA Row Prefetch requires copy size in Bytes to be aligned to 16B.");
  static constexpr uint32_t kRowSize = RowSizeT::value;
  static_assert(detail::is_valid_row_size_v<kRowSize>,
                "RowSize must be a power of 2 in [16, 2048]");

  using ThrID = std::conditional_t<IsPerWarp, Layout<_32>, Layout<_1>>;
  using SrcLayout = std::conditional_t<
      IsPerWarp,
      Layout<Shape<_32, decltype(NumBytes{} * C<8>{})>,
             Stride<decltype(NumBytes{} * C<8>{}), _1>>,
      Layout<Shape<_1, decltype(NumBytes{} * C<8>{})>>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  cute::tuple<OpArgs...> row_prefetch_args_;

  // A64: .with(void* gmem_ptr, uint32_t size, ...) — no separate offset (gmem_ptr carries it)
  template <AddressingMode M = Mode,
            detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            std::enable_if_t<M == AddressingMode::A64, int> = 0>
  CUTE_HOST_DEVICE constexpr auto
  with(void* gmem_ptr, uint32_t size, detail::CacheHint<CC> = {}) const {
    static_assert(detail::is_valid_row_prefetch_cc_v<CC>,
                  "XE4_ADMA_ROW_PREFETCH supports only L2uc_L3c, L2c_L3uc, L2c_L3c cache controls");
    return Copy_Traits<XE4_ADMA_ROW_PREFETCH,
                       NumBytes, RowSizeT,
                       cute::C<IsPerWarp>, cute::C<Mode>,
                       void*, uint32_t, detail::CacheHint<CC>>{
      cute::make_tuple(gmem_ptr, size, detail::CacheHint<CC>{})};
  }

  // A32S: .with(void* gmem_ptr, int32_t offset, uint32_t size, ...)
  template <AddressingMode M = Mode,
            detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            std::enable_if_t<M == AddressingMode::A32S, int> = 0>
  CUTE_HOST_DEVICE constexpr auto
  with(void* gmem_ptr, int32_t offset, uint32_t size, detail::CacheHint<CC> = {}) const {
    static_assert(detail::is_valid_row_prefetch_cc_v<CC>,
                  "XE4_ADMA_ROW_PREFETCH supports only L2uc_L3c, L2c_L3uc, L2c_L3c cache controls");
    return Copy_Traits<XE4_ADMA_ROW_PREFETCH,
                       NumBytes, RowSizeT,
                       cute::C<IsPerWarp>, cute::C<Mode>,
                       void*, int32_t, uint32_t, detail::CacheHint<CC>>{
      cute::make_tuple(gmem_ptr, offset, size, detail::CacheHint<CC>{})};
  }

  // A32U: .with(void* gmem_ptr, uint32_t offset, uint32_t size, ...)
  template <AddressingMode M = Mode,
            detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            std::enable_if_t<M == AddressingMode::A32U, int> = 0>
  CUTE_HOST_DEVICE constexpr auto
  with(void* gmem_ptr, uint32_t offset, uint32_t size, detail::CacheHint<CC> = {}) const {
    static_assert(detail::is_valid_row_prefetch_cc_v<CC>,
                  "XE4_ADMA_ROW_PREFETCH supports only L2uc_L3c, L2c_L3uc, L2c_L3c cache controls");
    return Copy_Traits<XE4_ADMA_ROW_PREFETCH,
                       NumBytes, RowSizeT,
                       cute::C<IsPerWarp>, cute::C<Mode>,
                       void*, uint32_t, uint32_t, detail::CacheHint<CC>>{
      cute::make_tuple(gmem_ptr, offset, size, detail::CacheHint<CC>{})};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  friend CUTE_HOST_DEVICE constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              [[maybe_unused]] Tensor<TD,DLayout>& dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem src for XE4_ADMA_ROW_PREFETCH");
    static_assert(sizeof...(OpArgs) > 0, "Extra arguments not set. Set .with() before use.");

    if constexpr (Mode == AddressingMode::A64) {
      // (gmem_ptr, size, CacheHint)
      constexpr auto CC = decltype(get<2>(traits.row_prefetch_args_))::value;
      XE4_ADMA_ROW_PREFETCH::template copy<AddressingMode::A64, kRowSize, CC>(
          get<0>(traits.row_prefetch_args_),    // gmem_ptr (void*)
          get<1>(traits.row_prefetch_args_),    // size
          get<2>(traits.row_prefetch_args_));   // CacheHint
    } else {
      // A32S / A32U: (gmem_ptr, offset, size, CacheHint)
      constexpr auto CC = decltype(get<3>(traits.row_prefetch_args_))::value;
      XE4_ADMA_ROW_PREFETCH::template copy<Mode, kRowSize, CC>(
          get<0>(traits.row_prefetch_args_),    // gmem_ptr
          get<1>(traits.row_prefetch_args_),    // offset (int32_t / uint32_t)
          get<2>(traits.row_prefetch_args_),    // size
          get<3>(traits.row_prefetch_args_));   // CacheHint
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copy_Traits for XE4_ADMA_ROW_COPY_TILED_LOAD - Unified for all addressing modes
/// RowSize is a compile-time power-of-2 in [16, 2048] selecting the hardware instruction variant.
/// Addressing mode is determined by .with() overload signature:
///   - A64:  .with(uint32_t mat_desc, uint64_t offset, ...)
///   - A32S: .with(uint32_t mat_desc, DataType* gmem_ptr, int32_t offset, ...)
///   - A32U: .with(uint32_t mat_desc, DataType* gmem_ptr, uint32_t offset, ...)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class DataType, class NumBytes, class RowSizeT, bool ThreadedAtom, class... OpArgs>
struct Copy_Traits<XE4_ADMA_ROW_COPY_TILED_LOAD, DataType, NumBytes, RowSizeT, cute::C<ThreadedAtom>, OpArgs...>
{
  static_assert(int32_t(NumBytes::value) % 16 == 0,
                "ADMA Tiled Row Load requires copy size in Bytes to be aligned to 16B.");
  static constexpr uint32_t kRowSize = RowSizeT::value;
  static_assert(detail::is_valid_row_size_v<kRowSize>,
                "RowSize must be a power of 2 in [16, 2048]");

  // ThreadedAtom selects between two atom layout flavors:
  //   true  (default for tiled use): 32-thread atom, each thread owns 1 row.
  //   false (single-thread / scalar): 1-thread atom carrying the full payload.
  using ThrID     = std::conditional_t<ThreadedAtom, Layout<_32>, Layout<_1>>;
  using SrcLayout = std::conditional_t<ThreadedAtom,
                      Layout<Shape<_32, decltype(NumBytes{} * C<8>{})>,
                             Stride<decltype(NumBytes{} * C<8>{}), _1>>,
                      Layout<Shape<_1,  decltype(NumBytes{} * C<8>{})>>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  cute::tuple<OpArgs...> row_load_args_;

  // A64 mode
  template <detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_ROW_COPY_TILED_LOAD, DataType, NumBytes, RowSizeT, cute::C<ThreadedAtom>, uint32_t, uint64_t, uint32_t, uint64_t*, detail::CacheHint<CC>, detail::FillMode<FM>>
  with(uint32_t mat_desc, uint64_t offset, uint32_t size, uint64_t* abar_ptr,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return {cute::make_tuple(mat_desc, offset, size, abar_ptr, detail::CacheHint<CC>{}, detail::FillMode<FM>{})};
  }

  // A32S mode
  template <detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_ROW_COPY_TILED_LOAD, DataType, NumBytes, RowSizeT, cute::C<ThreadedAtom>, uint32_t, DataType*, int32_t, uint32_t, uint64_t*, detail::CacheHint<CC>, detail::FillMode<FM>>
  with(uint32_t mat_desc, DataType* gmem_ptr, int32_t offset, uint32_t size, uint64_t* abar_ptr,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return {cute::make_tuple(mat_desc, gmem_ptr, offset, size, abar_ptr, detail::CacheHint<CC>{}, detail::FillMode<FM>{})};
  }

  // A32U mode
  template <detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_ROW_COPY_TILED_LOAD, DataType, NumBytes, RowSizeT, cute::C<ThreadedAtom>, uint32_t, DataType*, uint32_t, uint32_t, uint64_t*, detail::CacheHint<CC>, detail::FillMode<FM>>
  with(uint32_t mat_desc, DataType* gmem_ptr, uint32_t offset, uint32_t size, uint64_t* abar_ptr,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return {cute::make_tuple(mat_desc, gmem_ptr, offset, size, abar_ptr, detail::CacheHint<CC>{}, detail::FillMode<FM>{})};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  friend CUTE_HOST_DEVICE constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem src for XE4_ADMA_ROW_COPY_TILED_LOAD");
    static_assert(is_smem<TD>::value, "Expected smem dst for XE4_ADMA_ROW_COPY_TILED_LOAD");
    static_assert(std::is_same_v<typename TD::value_type, DataType>,
                  "ADMA tiled row load requires smem element type to match gmem DataType");

    if constexpr (sizeof...(OpArgs) == 6) {
      // A64 mode: (mat_desc, offset, size, abar_ptr, CacheHint, FillMode)
      using OffsetType = cute::tuple_element_t<1, cute::tuple<OpArgs...>>;
      if constexpr (std::is_same_v<OffsetType, uint64_t>) {
        using CacheHintType = decltype(get<4>(traits.row_load_args_));
        using FillModeType = decltype(get<5>(traits.row_load_args_));
        constexpr auto CC = CacheHintType::value;
        constexpr auto FM = FillModeType::value;
        XE4_ADMA_ROW_COPY_TILED_LOAD::template copy<AddressingMode::A64, kRowSize, DataType, CC, FM>(
            get<0>(traits.row_load_args_),  // mat_desc
            nullptr,                         // gmem_ptr unused in A64 mode
            get<1>(traits.row_load_args_),  // offset (uint64_t)
            get<2>(traits.row_load_args_),  // size
            get<3>(traits.row_load_args_),  // abar_ptr
            get<4>(traits.row_load_args_),  // CacheHint
            get<5>(traits.row_load_args_)); // FillMode
      }
    } else if constexpr (sizeof...(OpArgs) == 7) {
      // A32S or A32U mode: (mat_desc, gmem_ptr, offset, size, abar_ptr, CacheHint, FillMode)
      using OffsetType = cute::tuple_element_t<2, cute::tuple<OpArgs...>>;
      using CacheHintType = decltype(get<5>(traits.row_load_args_));
      using FillModeType = decltype(get<6>(traits.row_load_args_));
      constexpr auto CC = CacheHintType::value;
      constexpr auto FM = FillModeType::value;

      if constexpr (std::is_same_v<OffsetType, int32_t>) {
        // A32S mode
        XE4_ADMA_ROW_COPY_TILED_LOAD::template copy<AddressingMode::A32S, kRowSize, DataType, CC, FM>(
            get<0>(traits.row_load_args_),  // mat_desc
            get<1>(traits.row_load_args_),  // gmem_ptr
            get<2>(traits.row_load_args_),  // offset (int32_t)
            get<3>(traits.row_load_args_),  // size
            get<4>(traits.row_load_args_),  // abar_ptr
            get<5>(traits.row_load_args_),  // CacheHint
            get<6>(traits.row_load_args_)); // FillMode
      } else if constexpr (std::is_same_v<OffsetType, uint32_t>) {
        // A32U mode
        XE4_ADMA_ROW_COPY_TILED_LOAD::template copy<AddressingMode::A32U, kRowSize, DataType, CC, FM>(
            get<0>(traits.row_load_args_),  // mat_desc
            get<1>(traits.row_load_args_),  // gmem_ptr
            get<2>(traits.row_load_args_),  // offset (uint32_t)
            get<3>(traits.row_load_args_),  // size
            get<4>(traits.row_load_args_),  // abar_ptr
            get<5>(traits.row_load_args_),  // CacheHint
            get<6>(traits.row_load_args_)); // FillMode
      }
    } else {
      static_assert(sizeof...(OpArgs) == 0, "Extra arguments not set. Set .with() before use.");
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copy_Traits for XE4_ADMA_ROW_COPY_TILED_LOAD_MULTICAST - Same as TILED_LOAD but with wg_mask
/// Addressing mode is determined by .with() overload signature:
///   - A64:  .with(uint32_t mat_desc, uint64_t offset, uint32_t size, uint64_t* abar_ptr, uint32_t wg_mask, ...)
///   - A32S: .with(uint32_t mat_desc, DataType* gmem_ptr, int32_t offset, uint32_t size, uint64_t* abar_ptr, uint32_t wg_mask, ...)
///   - A32U: .with(uint32_t mat_desc, DataType* gmem_ptr, uint32_t offset, uint32_t size, uint64_t* abar_ptr, uint32_t wg_mask, ...)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class DataType, class NumBytes, class RowSizeT, bool ThreadedAtom, class... OpArgs>
struct Copy_Traits<XE4_ADMA_ROW_COPY_TILED_LOAD_MULTICAST, DataType, NumBytes, RowSizeT, cute::C<ThreadedAtom>, OpArgs...>
{
  static_assert(int32_t(NumBytes::value) % 16 == 0,
                "ADMA Tiled Row Load Multicast requires copy size in Bytes to be aligned to 16B.");
  static constexpr uint32_t kRowSize = RowSizeT::value;
  static_assert(detail::is_valid_row_size_v<kRowSize>,
                "RowSize must be a power of 2 in [16, 2048]");

  // ThreadedAtom selects between two atom layout flavors:
  //   true  (default for tiled use): 32-thread atom, each thread owns 1 row.
  //   false (single-thread / scalar): 1-thread atom carrying the full payload.
  using ThrID     = std::conditional_t<ThreadedAtom, Layout<_32>, Layout<_1>>;
  using SrcLayout = std::conditional_t<ThreadedAtom,
                      Layout<Shape<_32, decltype(NumBytes{} * C<8>{})>,
                             Stride<decltype(NumBytes{} * C<8>{}), _1>>,
                      Layout<Shape<_1,  decltype(NumBytes{} * C<8>{})>>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  cute::tuple<OpArgs...> row_load_args_;

  // A64 mode
  template <detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_ROW_COPY_TILED_LOAD_MULTICAST, DataType, NumBytes, RowSizeT, cute::C<ThreadedAtom>, uint32_t, uint64_t, uint32_t, uint64_t*, uint32_t, detail::CacheHint<CC>, detail::FillMode<FM>>
  with(uint32_t mat_desc, uint64_t offset, uint32_t size, uint64_t* abar_ptr, uint32_t wg_mask,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return {cute::make_tuple(mat_desc, offset, size, abar_ptr, wg_mask, detail::CacheHint<CC>{}, detail::FillMode<FM>{})};
  }

  // A32S mode
  template <detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_ROW_COPY_TILED_LOAD_MULTICAST, DataType, NumBytes, RowSizeT, cute::C<ThreadedAtom>, uint32_t, DataType*, int32_t, uint32_t, uint64_t*, uint32_t, detail::CacheHint<CC>, detail::FillMode<FM>>
  with(uint32_t mat_desc, DataType* gmem_ptr, int32_t offset, uint32_t size, uint64_t* abar_ptr, uint32_t wg_mask,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return {cute::make_tuple(mat_desc, gmem_ptr, offset, size, abar_ptr, wg_mask, detail::CacheHint<CC>{}, detail::FillMode<FM>{})};
  }

  // A32U mode
  template <detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_ROW_COPY_TILED_LOAD_MULTICAST, DataType, NumBytes, RowSizeT, cute::C<ThreadedAtom>, uint32_t, DataType*, uint32_t, uint32_t, uint64_t*, uint32_t, detail::CacheHint<CC>, detail::FillMode<FM>>
  with(uint32_t mat_desc, DataType* gmem_ptr, uint32_t offset, uint32_t size, uint64_t* abar_ptr, uint32_t wg_mask,
       detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}) const {
    return {cute::make_tuple(mat_desc, gmem_ptr, offset, size, abar_ptr, wg_mask, detail::CacheHint<CC>{}, detail::FillMode<FM>{})};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  friend CUTE_HOST_DEVICE constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_gmem<TS>::value, "Expected gmem src for XE4_ADMA_ROW_COPY_TILED_LOAD_MULTICAST");
    static_assert(is_smem<TD>::value, "Expected smem dst for XE4_ADMA_ROW_COPY_TILED_LOAD_MULTICAST");
    static_assert(std::is_same_v<typename TD::value_type, DataType>,
                  "ADMA tiled row load multicast requires smem element type to match gmem DataType");

    if constexpr (sizeof...(OpArgs) == 7) {
      // A64 mode: (mat_desc, offset, size, abar_ptr, wg_mask, CacheHint, FillMode)
      using OffsetType = cute::tuple_element_t<1, cute::tuple<OpArgs...>>;
      if constexpr (std::is_same_v<OffsetType, uint64_t>) {
        using CacheHintType = decltype(get<5>(traits.row_load_args_));
        using FillModeType = decltype(get<6>(traits.row_load_args_));
        constexpr auto CC = CacheHintType::value;
        constexpr auto FM = FillModeType::value;
        XE4_ADMA_ROW_COPY_TILED_LOAD_MULTICAST::template copy<AddressingMode::A64, kRowSize, DataType, CC, FM>(
            get<0>(traits.row_load_args_),  // mat_desc
            nullptr,                         // gmem_ptr unused in A64 mode
            get<1>(traits.row_load_args_),  // offset (uint64_t)
            get<2>(traits.row_load_args_),  // size
            get<3>(traits.row_load_args_),  // abar_ptr
            get<4>(traits.row_load_args_),  // wg_mask
            get<5>(traits.row_load_args_),  // CacheHint
            get<6>(traits.row_load_args_)); // FillMode
      }
    } else if constexpr (sizeof...(OpArgs) == 8) {
      // A32S or A32U mode: (mat_desc, gmem_ptr, offset, size, abar_ptr, wg_mask, CacheHint, FillMode)
      using OffsetType = cute::tuple_element_t<2, cute::tuple<OpArgs...>>;
      using CacheHintType = decltype(get<6>(traits.row_load_args_));
      using FillModeType = decltype(get<7>(traits.row_load_args_));
      constexpr auto CC = CacheHintType::value;
      constexpr auto FM = FillModeType::value;

      if constexpr (std::is_same_v<OffsetType, int32_t>) {
        // A32S mode
        XE4_ADMA_ROW_COPY_TILED_LOAD_MULTICAST::template copy<AddressingMode::A32S, kRowSize, DataType, CC, FM>(
            get<0>(traits.row_load_args_),  // mat_desc
            get<1>(traits.row_load_args_),  // gmem_ptr
            get<2>(traits.row_load_args_),  // offset (int32_t)
            get<3>(traits.row_load_args_),  // size
            get<4>(traits.row_load_args_),  // abar_ptr
            get<5>(traits.row_load_args_),  // wg_mask
            get<6>(traits.row_load_args_),  // CacheHint
            get<7>(traits.row_load_args_)); // FillMode
      } else if constexpr (std::is_same_v<OffsetType, uint32_t>) {
        // A32U mode
        XE4_ADMA_ROW_COPY_TILED_LOAD_MULTICAST::template copy<AddressingMode::A32U, kRowSize, DataType, CC, FM>(
            get<0>(traits.row_load_args_),  // mat_desc
            get<1>(traits.row_load_args_),  // gmem_ptr
            get<2>(traits.row_load_args_),  // offset (uint32_t)
            get<3>(traits.row_load_args_),  // size
            get<4>(traits.row_load_args_),  // abar_ptr
            get<5>(traits.row_load_args_),  // wg_mask
            get<6>(traits.row_load_args_),  // CacheHint
            get<7>(traits.row_load_args_)); // FillMode
      }
    } else {
      static_assert(sizeof...(OpArgs) == 0, "Extra arguments not set. Set .with() before use.");
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Copy_Traits for XE4_ADMA_ROW_COPY_TILED_STORE - Unified for all addressing modes
/// The SLM source is identified by a matrix descriptor (uint32_t mat_desc) supplied via .with();
/// the smem Tensor is only used for type / static checks.
/// Addressing mode is determined by .with() overload signature:
///   - A64:  .with(uint32_t mat_desc, uint64_t offset, ...)
///   - A32S: .with(uint32_t mat_desc, DataType* gmem_ptr, int32_t offset, ...)
///   - A32U: .with(uint32_t mat_desc, DataType* gmem_ptr, uint32_t offset, ...)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class DataType, class NumBytes, class RowSizeT, bool ThreadedAtom, class... OpArgs>
struct Copy_Traits<XE4_ADMA_ROW_COPY_TILED_STORE, DataType, NumBytes, RowSizeT, cute::C<ThreadedAtom>, OpArgs...>
{
  static_assert(int32_t(NumBytes::value) % 16 == 0,
                "ADMA Tiled Row Store requires copy size in Bytes to be aligned to 16B.");
  static constexpr uint32_t kRowSize = RowSizeT::value;
  static_assert(detail::is_valid_row_size_v<kRowSize>,
                "RowSize must be a power of 2 in [16, 2048]");

  // ThreadedAtom selects between two atom layout flavors:
  //   true  (default for tiled use): 32-thread atom, each thread owns 1 row.
  //   false (single-thread / scalar): 1-thread atom carrying the full payload.
  using ThrID     = std::conditional_t<ThreadedAtom, Layout<_32>, Layout<_1>>;
  using SrcLayout = std::conditional_t<ThreadedAtom,
                      Layout<Shape<_32, decltype(NumBytes{} * C<8>{})>,
                             Stride<decltype(NumBytes{} * C<8>{}), _1>>,
                      Layout<Shape<_1,  decltype(NumBytes{} * C<8>{})>>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  cute::tuple<OpArgs...> row_store_args_;

  // A64 mode
  template <detail::CacheCtrl CC = detail::CacheCtrl::L2wb_L3uc,
            detail::CompletionMode CM = detail::CompletionMode::CM_Unspecified>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_ROW_COPY_TILED_STORE, DataType, NumBytes, RowSizeT, cute::C<ThreadedAtom>, uint32_t, uint64_t, uint32_t, uint64_t*, detail::CacheHint<CC>, detail::CompletionModeHint<CM>>
  with(uint32_t mat_desc, uint64_t offset, uint32_t size, uint64_t* abar_ptr,
       detail::CacheHint<CC> = {}, detail::CompletionModeHint<CM> = {}) const {
    return {cute::make_tuple(mat_desc, offset, size, abar_ptr,
                             detail::CacheHint<CC>{}, detail::CompletionModeHint<CM>{})};
  }

  // A32S mode
  template <detail::CacheCtrl CC = detail::CacheCtrl::L2wb_L3uc,
            detail::CompletionMode CM = detail::CompletionMode::CM_Unspecified>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_ROW_COPY_TILED_STORE, DataType, NumBytes, RowSizeT, cute::C<ThreadedAtom>, uint32_t, DataType*, int32_t, uint32_t, uint64_t*, detail::CacheHint<CC>, detail::CompletionModeHint<CM>>
  with(uint32_t mat_desc, DataType* gmem_ptr, int32_t offset, uint32_t size, uint64_t* abar_ptr,
       detail::CacheHint<CC> = {}, detail::CompletionModeHint<CM> = {}) const {
    return {cute::make_tuple(mat_desc, gmem_ptr, offset, size, abar_ptr,
                             detail::CacheHint<CC>{}, detail::CompletionModeHint<CM>{})};
  }

  // A32U mode
  template <detail::CacheCtrl CC = detail::CacheCtrl::L2wb_L3uc,
            detail::CompletionMode CM = detail::CompletionMode::CM_Unspecified>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_ROW_COPY_TILED_STORE, DataType, NumBytes, RowSizeT, cute::C<ThreadedAtom>, uint32_t, DataType*, uint32_t, uint32_t, uint64_t*, detail::CacheHint<CC>, detail::CompletionModeHint<CM>>
  with(uint32_t mat_desc, DataType* gmem_ptr, uint32_t offset, uint32_t size, uint64_t* abar_ptr,
       detail::CacheHint<CC> = {}, detail::CompletionModeHint<CM> = {}) const {
    return {cute::make_tuple(mat_desc, gmem_ptr, offset, size, abar_ptr,
                             detail::CacheHint<CC>{}, detail::CompletionModeHint<CM>{})};
  }

  template <class TS, class SLayout,
            class TD, class DLayout>
  friend CUTE_HOST_DEVICE constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst)
  {
    static_assert(is_smem<TS>::value, "Expected smem src for XE4_ADMA_ROW_COPY_TILED_STORE");
    static_assert(is_gmem<TD>::value, "Expected gmem dst for XE4_ADMA_ROW_COPY_TILED_STORE");
    static_assert(std::is_same_v<typename TS::value_type, DataType>,
                  "ADMA tiled row store requires smem element type to match gmem DataType");

    if constexpr (sizeof...(OpArgs) == 6) {
      // A64 mode: (mat_desc, offset, size, abar_ptr, CacheHint, CompletionModeHint)
      using OffsetType = cute::tuple_element_t<1, cute::tuple<OpArgs...>>;
      if constexpr (std::is_same_v<OffsetType, uint64_t>) {
        using CacheHintType = decltype(get<4>(traits.row_store_args_));
        using CMHintType    = decltype(get<5>(traits.row_store_args_));
        constexpr auto CC = CacheHintType::value;
        constexpr auto CM = CMHintType::value;
        XE4_ADMA_ROW_COPY_TILED_STORE::template copy<AddressingMode::A64, kRowSize, DataType, CC, CM>(
            get<0>(traits.row_store_args_),  // mat_desc
            nullptr,                          // gmem_ptr unused in A64 mode
            get<1>(traits.row_store_args_),  // offset (uint64_t)
            get<2>(traits.row_store_args_),  // size
            get<3>(traits.row_store_args_),  // abar_ptr
            get<4>(traits.row_store_args_),  // CacheHint
            get<5>(traits.row_store_args_)); // CompletionModeHint
      }
    } else if constexpr (sizeof...(OpArgs) == 7) {
      // A32S or A32U mode: (mat_desc, gmem_ptr, offset, size, abar_ptr, CacheHint, CompletionModeHint)
      using OffsetType = cute::tuple_element_t<2, cute::tuple<OpArgs...>>;
      using CacheHintType = decltype(get<5>(traits.row_store_args_));
      using CMHintType    = decltype(get<6>(traits.row_store_args_));
      constexpr auto CC = CacheHintType::value;
      constexpr auto CM = CMHintType::value;

      if constexpr (std::is_same_v<OffsetType, int32_t>) {
        // A32S mode
        XE4_ADMA_ROW_COPY_TILED_STORE::template copy<AddressingMode::A32S, kRowSize, DataType, CC, CM>(
            get<0>(traits.row_store_args_),  // mat_desc
            get<1>(traits.row_store_args_),  // gmem_ptr
            get<2>(traits.row_store_args_),  // offset (int32_t)
            get<3>(traits.row_store_args_),  // size
            get<4>(traits.row_store_args_),  // abar_ptr
            get<5>(traits.row_store_args_),  // CacheHint
            get<6>(traits.row_store_args_)); // CompletionModeHint
      } else if constexpr (std::is_same_v<OffsetType, uint32_t>) {
        // A32U mode
        XE4_ADMA_ROW_COPY_TILED_STORE::template copy<AddressingMode::A32U, kRowSize, DataType, CC, CM>(
            get<0>(traits.row_store_args_),  // mat_desc
            get<1>(traits.row_store_args_),  // gmem_ptr
            get<2>(traits.row_store_args_),  // offset (uint32_t)
            get<3>(traits.row_store_args_),  // size
            get<4>(traits.row_store_args_),  // abar_ptr
            get<5>(traits.row_store_args_),  // CacheHint
            get<6>(traits.row_store_args_)); // CompletionModeHint
      }
    } else {
      static_assert(sizeof...(OpArgs) == 0, "Extra arguments not set. Set .with() before use.");
    }
  }
};

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

  mutable TensorDescriptor<T> tensorDesc_;
  using AuxParams = AuxParams_;
  mutable AuxParams aux_params_;
  mutable uint64_t* tdesc_ptr_ { nullptr };

  CUTE_DEVICE void
  set_tensor_desc(uint64_t* tensor_desc) const {
    dupTensorPayload(tensor_desc, (uint64_t *)&tensorDesc_.payload);
    tdesc_ptr_ = tensor_desc;
  }

  CUTE_DEVICE void
  update_gmem_details_and_params(TensorDescriptor<T> tensor_desc, AuxParams aux_params) const {
    tensorDesc_ = tensor_desc;
    aux_params_ = aux_params;
  }

  template <class GmemLayout, class SBoxLayout>
  CUTE_DEVICE void
  update_gmem_details_and_params(
        TensorDescriptor<T> tensor_desc,
        AuxParams aux_params,
        GmemLayout const& gmem_layout,
        SBoxLayout const& sbox_layout
  ) const {
    static_assert(is_layout<GmemLayout>::value && is_layout<SBoxLayout>::value,
                  "GmemLayout and SBoxLayout must be valid CuTe layout types.");
    constexpr int t_dim = rank_v<GmemLayout>;
    static_assert(rank_v<SBoxLayout> == t_dim,
                  "GmemLayout and SBoxLayout must have equal rank.");
    // gmem_layout stride is expected to include the contiguous innermost stride at index 0.
    tensorDesc_ = tensor_desc;
    aux_params_ = aux_params;
    auto gmem_shape = to_array<uint32_t>(shape(gmem_layout));
    auto gmem_stride = to_array<uint64_t>(stride(gmem_layout));
    auto sbox_shape = to_array<uint32_t>(shape(sbox_layout));
    auto sbox_stride = to_array<uint32_t>(stride(sbox_layout));
    auto* payload = reinterpret_cast<TensorPayload *>(&tensorDesc_.payload);
    for (int i = 0; i < t_dim; ++i) {
      payload->DimSize[i] = gmem_shape[i] - 1;
      payload->ROITensorDimSize[i] = sbox_shape[i] - 1;
    }
    for (int i = 0; i < t_dim - 1; ++i) {
      // DimStride omits the innermost dimension (always contiguous stride-1).
      // The input layout stride includes that innermost stride at index 0,
      // so DimStride[i] maps to gmem_stride[i + 1].
      payload->DimStride[i] = gmem_stride[i + 1];
    }
    payload->ElementStride0 = sbox_stride[0] - 1;
    if constexpr (t_dim > 1) { payload->ElementStride1 = sbox_stride[1] - 1; }
    if constexpr (t_dim > 2) { payload->ElementStride2 = sbox_stride[2] - 1; }
    if constexpr (t_dim > 3) { payload->ElementStride3 = sbox_stride[3] - 1; }
    if constexpr (t_dim > 4) { payload->ElementStride4 = sbox_stride[4] - 1; }
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

  mutable TensorDescriptor<T> tensorDesc_;
  using AuxParams = AuxParams_;
  mutable AuxParams aux_params_;
  mutable uint64_t* tdesc_ptr_ { nullptr };

  CUTE_DEVICE void
  set_tensor_desc(uint64_t* tensor_desc) const {
    dupTensorPayload(tensor_desc, (uint64_t *)&tensorDesc_.payload);
    tdesc_ptr_ = tensor_desc;
  }

  CUTE_DEVICE void
  update_gmem_details_and_params(TensorDescriptor<T> tensor_desc, AuxParams aux_params) const {
    tensorDesc_ = tensor_desc;
    aux_params_ = aux_params;
  }

  template <class GmemLayout, class SBoxLayout>
  CUTE_DEVICE void
  update_gmem_details_and_params(
        TensorDescriptor<T> tensor_desc,
        AuxParams aux_params,
        GmemLayout const& gmem_layout,
        SBoxLayout const& sbox_layout
  ) const {
    static_assert(is_layout<GmemLayout>::value && is_layout<SBoxLayout>::value,
                  "GmemLayout and SBoxLayout must be valid CuTe layout types.");
    constexpr int t_dim = rank_v<GmemLayout>;
    static_assert(rank_v<SBoxLayout> == t_dim,
                  "GmemLayout and SBoxLayout must have equal rank.");
    // gmem_layout stride is expected to include the contiguous innermost stride at index 0.
    tensorDesc_ = tensor_desc;
    aux_params_ = aux_params;
    auto gmem_shape = to_array<uint32_t>(shape(gmem_layout));
    auto gmem_stride = to_array<uint64_t>(stride(gmem_layout));
    auto sbox_shape = to_array<uint32_t>(shape(sbox_layout));
    auto sbox_stride = to_array<uint32_t>(stride(sbox_layout));
    auto* payload = reinterpret_cast<TensorPayload *>(&tensorDesc_.payload);
    for (int i = 0; i < t_dim; ++i) {
      payload->DimSize[i] = gmem_shape[i] - 1;
      payload->ROITensorDimSize[i] = sbox_shape[i] - 1;
    }
    for (int i = 0; i < t_dim - 1; ++i) {
      // DimStride omits the innermost dimension (always contiguous stride-1).
      // The input layout stride includes that innermost stride at index 0,
      // so DimStride[i] maps to gmem_stride[i + 1].
      payload->DimStride[i] = gmem_stride[i + 1];
    }
    payload->ElementStride0 = sbox_stride[0] - 1;
    if constexpr (t_dim > 1) { payload->ElementStride1 = sbox_stride[1] - 1; }
    if constexpr (t_dim > 2) { payload->ElementStride2 = sbox_stride[2] - 1; }
    if constexpr (t_dim > 3) { payload->ElementStride3 = sbox_stride[3] - 1; }
    if constexpr (t_dim > 4) { payload->ElementStride4 = sbox_stride[4] - 1; }
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

  mutable TensorDescriptor<T> tensorDesc_;
  using AuxParams = AuxParams_;
  mutable AuxParams aux_params_;
  mutable uint64_t* tdesc_ptr_ { nullptr };

  CUTE_DEVICE void
  set_tensor_desc(uint64_t* tensor_desc) const {
    dupTensorPayload(tensor_desc, (uint64_t *)&tensorDesc_.payload);
    tdesc_ptr_ = tensor_desc;
  }

  CUTE_DEVICE void
  update_gmem_details_and_params(TensorDescriptor<T> tensor_desc, AuxParams aux_params) const {
    tensorDesc_ = tensor_desc;
    aux_params_ = aux_params;
  }

  template <class GmemLayout, class SBoxLayout>
  CUTE_DEVICE void
  update_gmem_details_and_params(
        TensorDescriptor<T> tensor_desc,
        AuxParams aux_params,
        GmemLayout const& gmem_layout,
        SBoxLayout const& sbox_layout
  ) const {
    static_assert(is_layout<GmemLayout>::value && is_layout<SBoxLayout>::value,
                  "GmemLayout and SBoxLayout must be valid CuTe layout types.");
    constexpr int t_dim = rank_v<GmemLayout>;
    static_assert(rank_v<SBoxLayout> == t_dim,
                  "GmemLayout and SBoxLayout must have equal rank.");
    // gmem_layout stride is expected to include the contiguous innermost stride at index 0.
    tensorDesc_ = tensor_desc;
    aux_params_ = aux_params;
    auto gmem_shape = to_array<uint32_t>(shape(gmem_layout));
    auto gmem_stride = to_array<uint64_t>(stride(gmem_layout));
    auto sbox_shape = to_array<uint32_t>(shape(sbox_layout));
    auto sbox_stride = to_array<uint32_t>(stride(sbox_layout));
    auto* payload = reinterpret_cast<TensorPayload *>(&tensorDesc_.payload);
    for (int i = 0; i < t_dim; ++i) {
      payload->DimSize[i] = gmem_shape[i] - 1;
      payload->ROITensorDimSize[i] = sbox_shape[i] - 1;
    }
    for (int i = 0; i < t_dim - 1; ++i) {
      // DimStride omits the innermost dimension (always contiguous stride-1).
      // The input layout stride includes that innermost stride at index 0,
      // so DimStride[i] maps to gmem_stride[i + 1].
      payload->DimStride[i] = gmem_stride[i + 1];
    }
    payload->ElementStride0 = sbox_stride[0] - 1;
    if constexpr (t_dim > 1) { payload->ElementStride1 = sbox_stride[1] - 1; }
    if constexpr (t_dim > 2) { payload->ElementStride2 = sbox_stride[2] - 1; }
    if constexpr (t_dim > 3) { payload->ElementStride3 = sbox_stride[3] - 1; }
    if constexpr (t_dim > 4) { payload->ElementStride4 = sbox_stride[4] - 1; }
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

  template <detail::CacheCtrl CC = detail::CacheCtrl::L2wb_L3uc>
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<XE4_ADMA_STORE_OP, T, NumBitsPerADMA, detail::CacheHint<CC>>
  with(uint64_t* abar_ptr, [[maybe_unused]] uint32_t const& multicast_mask = 0,
       detail::CacheHint<CC> = {}) const {
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
    return reinterpret_cast<TensorDescriptor<T> const*>(get<0>(opargs_));
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

  // 4-element opargs; ADMA_STORE_Unpack inserts slm_ptr + coord before the last element
  // Final call: (tdesc_ptr, gmem_ptr, mat_desc, slm_ptr, coord, abar_ptr) — 6 args
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

///////////////////////////////////////////////////////////////////////////////////////////////////

// Validate that ADMA box dimensions after cluster truncation still meet
// Core Matrix alignment requirements. Only checks data descriptors (Type1/Type2);
// SF Type3 is handled via SMEM padding at the layout level.
template <class InternalType>
CUTE_HOST_RTC
void validate_cm_alignment_after_truncation(
    const cute::array<uint16_t, 5>& smem_box_shape,
    uint32_t matrix_desc,
    uint32_t num_multicast)
{
  constexpr int elem_bits = cutlass::sizeof_bits<InternalType>::value;
  int desc_type = (matrix_desc >> 28) & 0x3;
  if (desc_type == MatrixDescriptor::Type3) return;

  cm_size_t cm = get_core_matrix_size(elem_bits, desc_type);
  int y_min = 0, x_min_bytes = 0;
  switch (cm) {
    case cm_size_t::cm_32x32B: y_min = 32; x_min_bytes = 32; break;
    case cm_size_t::cm_16x64B: y_min = 16; x_min_bytes = 64; break;
    case cm_size_t::cm_8x128B: y_min = 8;  x_min_bytes = 128; break;
    case cm_size_t::cm_4x256B: y_min = 4;  x_min_bytes = 256; break;
    default: break;
  }
  // For 6-bit elements, width_bytes*8/6 = 42 which is wrong because 6 doesn't
  // divide evenly into a byte. HW spec says D6 aligns same as D8 (32 elements)
  // since 6-bit elements are byte-padded in the core matrix.
  int x_min = x_min_bytes * 8 / ((elem_bits == 6) ? 8 : elem_bits);
  const char* type_name = (desc_type == 0) ? "Type1" : (desc_type == 1) ? "Type2" : "Type3";

  if (smem_box_shape[0] < static_cast<uint16_t>(x_min)) {
    fprintf(stderr, "ERROR: ADMA box X-dimension (dim0=%u) after cluster truncation "
            "violates Core Matrix alignment (min=%d, elem_bits=%d, desc_type=%d(%s), "
            "num_multicast=%u)\n",
            static_cast<unsigned int>(smem_box_shape[0]), x_min, elem_bits, desc_type, type_name, num_multicast);
    throw std::runtime_error("ADMA box X-dimension after cluster truncation "
                             "violates Core Matrix alignment");
  }
  if (smem_box_shape[1] < static_cast<uint16_t>(y_min)) {
    fprintf(stderr, "ERROR: ADMA box Y-dimension (dim1=%u) after cluster truncation "
            "violates Core Matrix alignment (min=%d, elem_bits=%d, desc_type=%d(%s), "
            "num_multicast=%u)\n",
            static_cast<unsigned int>(smem_box_shape[1]), y_min, elem_bits, desc_type, type_name, num_multicast);
    throw std::runtime_error("ADMA box Y-dimension after cluster truncation "
                             "violates Core Matrix alignment");
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Use a sidx2gmode to read through the GMEM tensor
//   and construct a TMA Descriptor for the resulting instruction
// At the same time, construct the Tma Tensor's Stride to generate
//   the TMA coordinates that the instruction consumes.
//
template <class InternalType,
          class GEngine, class GLayout,
          class TShape, class TStride>
CUTE_HOST_DEVICE
auto
make_adma_copy_desc_common(
    Tensor<GEngine,GLayout> const& gtensor,
    Layout<TShape,TStride>  const& adma_gbasis,
    uint32_t                       matrix_desc,
    uint32_t                       num_multicast)
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

  // Validate Core Matrix alignment after cluster truncation (A/B data only, skip SF Type3 as its handled by padding).
  if (num_multicast > 1) {
    validate_cm_alignment_after_truncation<InternalType>(smem_box_shape, matrix_desc, num_multicast);
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
      using EI = decltype(ei);
      [[maybe_unused]] auto j = find_if(tma_gmem_basis_stride, [&](auto tma_stride_j) {
        return any_of(tma_stride_j, [&](auto dj) { return dj == EI{}; });
      });
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
  cute::array<uint32_t, t_dim> sbox_shape;
  for_each(make_seq<t_dim>{}, [&](auto i) {sbox_shape[i] = smem_box_shape[i];});
  cute::array<uint32_t, t_dim> sbox_stride;
  for_each(make_seq<t_dim>{}, [&](auto i) {sbox_stride[i] = smem_box_stride[i];});

  tensor_desc.matrix_desc = matrix_desc;
  tensor_desc.g_pointer = sycl::address_space_cast<
    sycl::access::address_space::global_space,
    sycl::access::decorated::yes
  >(gmem_address).get();

  return cute::make_tuple(tensor_desc, AuxParams{gmem_tma_basis_stride}, adma_gbasis,
                          gmem_shape, gmem_stride, sbox_shape, sbox_stride);
}

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
  auto [tensor_desc, aux_params, common_gbasis, gmem_shape, gmem_stride, sbox_shape_u32, sbox_stride] =
    make_adma_copy_desc_common<InternalType>(gtensor, adma_gbasis, matrix_desc, num_multicast);
  constexpr int t_dim = decltype(rank(common_gbasis))::value;

  cute::array<uint16_t, t_dim> sbox_shape;
  for_each(make_seq<t_dim>{}, [&](auto i) {sbox_shape[i] = sbox_shape_u32[i];});

  fillTensorDescriptorDimSize((uint64_t *)&tensor_desc, gmem_shape);
  fillTensorDescriptorDimStride((uint64_t *)&tensor_desc, gmem_stride);
  fillTensorDescriptorROI((uint64_t *)&tensor_desc, sbox_shape);
  fillTensorDescriptorElementStride((uint64_t *)&tensor_desc, sbox_stride);

  return cute::make_tuple(tensor_desc, aux_params);
}

template <class InternalType,
          class GEngine, class GLayout,
          class TShape, class TStride>
CUTE_HOST_DEVICE
auto
make_adma_copy_desc_device(
    Tensor<GEngine,GLayout> const& gtensor,
    Layout<TShape,TStride>  const& adma_gbasis,
    uint32_t                       matrix_desc,
    uint32_t                       num_multicast)
{
  return make_adma_copy_desc_common<InternalType>(gtensor, adma_gbasis, matrix_desc, num_multicast);
}

template <class SLayout>
CUTE_HOST_DEVICE
auto
make_adma_matrix_descriptor(SLayout const& slayout, bool is_A_matrix = false) {
  auto slayout_2d = [&]() {
    if constexpr (decltype(rank(slayout))::value >= 3) {
      return layout<0>(slayout);
    } else {
      return coalesce(slayout);
    }
  }();

  auto strides = slayout_2d.stride();

  MatrixDescriptor matrix_desc{};
  if (is_A_matrix) {
    matrix_desc.Type = get<1>(strides) == 1 ?
      MatrixDescriptor::Type1 : MatrixDescriptor::Type2;
  } else {
    matrix_desc.Type = MatrixDescriptor::Type1;
  }

  matrix_desc.Pitch = get<1>(strides) == 1 ?
    get<0>(strides) >> 2 : get<1>(strides) >> 2;

  return matrix_desc.raw_;
}

template <class InternalType = void,
          class GEngine, class GLayout,
          class SLayout,
          class CtaVTile,
          class NumMulticast>
CUTE_DEVICE
auto
make_gmem_details_params(
    Tensor<GEngine, GLayout> const& gtensor,
    SLayout const& slayout,
    CtaVTile const& cta_v_tile,
    uint32_t matrix_desc,
    NumMulticast const& num_multicast)
{
  using AmmaType = conditional_t<is_same<void, InternalType>::value, typename GEngine::value_type, InternalType>;

  auto smem_layout = get_nonswizzle_portion(slayout);
  auto adma_gbasis = construct_tma_gbasis<AmmaType>(gtensor, smem_layout, cta_v_tile);

  return make_adma_copy_desc_device<AmmaType>(
      gtensor, adma_gbasis, matrix_desc, num_multicast);
}

template <class InternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class TiledMma,
          class TileShape,
          class ClusterLayout,
          class IsAMatrix>
CUTE_DEVICE
auto
make_gmem_details_params(
    CopyOp,
    Tensor<GEngine, GLayout> const& gtensor,
    SLayout const& slayout,
    TiledMma const& tiled_mma,
    TileShape const& tile_shape,
    ClusterLayout const& cluster_layout,
    IsAMatrix)
{
  constexpr bool is_A_matrix = bool(IsAMatrix::value);

  auto mma_tiler = [&]() {
    if constexpr (is_A_matrix) {
      return remove<1>(tile_shape);
    } else {
      return remove<0>(tile_shape);
    }
  }();

  auto g_tile = make_identity_layout(shape(gtensor)).compose(mma_tiler);
  auto cta_v_tile = [&]() {
    if constexpr (is_A_matrix) {
      return cute::layout<1>(tiled_mma.thrfrg_A(g_tile))(_, repeat<rank(g_tile)>(_));
    } else {
      return cute::layout<1>(tiled_mma.thrfrg_B(g_tile))(_, repeat<rank(g_tile)>(_));
    }
  }();
  auto matrix_desc = make_adma_matrix_descriptor(slayout, is_A_matrix);
  auto num_multicast = [&]() {
    if constexpr (is_same_v<CopyOp, XE4_ADMA_LOAD_MULTICAST>) {
      if constexpr (is_A_matrix) {
        return size<2>(cluster_layout);
      } else {
        return size<1>(cluster_layout);
      }
    } else {
      return Int<1>{};
    }
  }();

  return make_gmem_details_params<InternalType>(gtensor, slayout, cta_v_tile, matrix_desc, num_multicast);
}

template <class InternalType = void,
          class GEngine, class GLayout,
          class SmemLayoutOutput,
          class TileShape>
CUTE_DEVICE
auto
make_gmem_details_params(
    Tensor<GEngine, GLayout> const& gtensor,
    SmemLayoutOutput const& smem_layout_output,
    TileShape const& tile_shape)
{
  auto slayout = smem_layout_output(_, _, Int<0>{});
  auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(select<0, 1>(tile_shape));
  auto matrix_desc = make_adma_matrix_descriptor(slayout, false);
  return make_gmem_details_params<InternalType>(gtensor, slayout, cta_v_tile, matrix_desc, Int<1>{});
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
template <class AdmaCopy,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class TiledMma,
          class TileShape,
          class ClusterLayout,
          class IsAMatrix>
CUTE_DEVICE void
update_gmem_details_and_params(
    AdmaCopy const& adma_copy,
    CopyOp const& copy_op,
    Tensor<GEngine, GLayout> const& gtensor,
    SLayout const& slayout,
    TiledMma const& tiled_mma,
    TileShape const& tile_shape,
    ClusterLayout const& cluster_layout,
    IsAMatrix const& is_a_matrix) {
  auto [tma_details, aux_params, adma_gbasis, gmem_shape, gmem_stride, sbox_shape, sbox_stride] =
    detail::make_gmem_details_params(copy_op, gtensor, slayout, tiled_mma, tile_shape, cluster_layout, is_a_matrix);
  constexpr int t_dim = decltype(rank(adma_gbasis))::value;
  static_assert(tuple_size<decltype(gmem_stride)>::value == t_dim - 1,
                "gmem_stride tuple size must be t_dim-1 (excludes innermost contiguous dimension).");
  // ADMA DimStride stores strides starting from dimension-1; prepend dimension-0's contiguous stride.
  auto gmem_layout = make_layout(gmem_shape, prepend<t_dim>(gmem_stride, uint64_t{1}));
  auto sbox_layout = make_layout(sbox_shape, sbox_stride);
  adma_copy.update_gmem_details_and_params(tma_details, aux_params, gmem_layout, sbox_layout);
}

template <class AdmaCopy,
          class GEngine, class GLayout,
          class SLayout,
          class CtaVTile,
          class NumMulticast>
CUTE_DEVICE void
update_gmem_details_and_params(
    AdmaCopy const& adma_copy,
    Tensor<GEngine, GLayout> const& gtensor,
    SLayout const& slayout,
    CtaVTile const& cta_v_tile,
    uint32_t matrix_desc,
    NumMulticast const& num_multicast) {
  auto [tma_details, aux_params, adma_gbasis, gmem_shape, gmem_stride, sbox_shape, sbox_stride] =
    detail::make_gmem_details_params(gtensor, slayout, cta_v_tile, matrix_desc, num_multicast);
  constexpr int t_dim = decltype(rank(adma_gbasis))::value;
  static_assert(tuple_size<decltype(gmem_stride)>::value == t_dim - 1,
                "gmem_stride tuple size must be t_dim-1 (excludes innermost contiguous dimension).");
  // ADMA DimStride stores strides starting from dimension-1; prepend dimension-0's contiguous stride.
  auto gmem_layout = make_layout(gmem_shape, prepend<t_dim>(gmem_stride, uint64_t{1}));
  auto sbox_layout = make_layout(sbox_shape, sbox_stride);
  adma_copy.update_gmem_details_and_params(tma_details, aux_params, gmem_layout, sbox_layout);
}

template <class AdmaCopy,
          class GEngine, class GLayout,
          class SLayout,
          class TiledMma,
          class TileShape,
          class IsAMatrix>
CUTE_DEVICE void
update_gmem_details_and_params(
    AdmaCopy const& adma_copy,
    Tensor<GEngine, GLayout> const& gtensor,
    SLayout const& slayout,
    TiledMma const& tiled_mma,
    TileShape const& tile_shape,
    IsAMatrix const& is_a_matrix) {
  constexpr bool is_A_matrix = bool(IsAMatrix::value);

  auto mma_tiler = [&]() {
    if constexpr (is_A_matrix) {
      return remove<1>(tile_shape);
    } else {
      return remove<0>(tile_shape);
    }
  }();

  auto g_tile = make_identity_layout(shape(gtensor)).compose(mma_tiler);
  auto cta_v_tile = [&]() {
    if constexpr (is_A_matrix) {
      return cute::layout<1>(tiled_mma.thrfrg_A(g_tile))(_, repeat<rank(g_tile)>(_));
    } else {
      return cute::layout<1>(tiled_mma.thrfrg_B(g_tile))(_, repeat<rank(g_tile)>(_));
    }
  }();

  MatrixDescriptor matrix_desc{};
  matrix_desc.Type = MatrixDescriptor::Type3;
  if constexpr (is_A_matrix) {
    matrix_desc.Pitch = get<0>(tile_shape) >> 2;
  } else {
    matrix_desc.Pitch = get<1>(tile_shape) >> 2;
  }

  cute::update_gmem_details_and_params(
    adma_copy, gtensor, slayout, cta_v_tile, matrix_desc.raw_, cute::Int<1>{});
}

template <class AdmaCopy,
          class GEngine, class GLayout,
          class SmemLayoutOutput,
          class TileShape>
CUTE_DEVICE void
update_gmem_details_and_params(
    AdmaCopy const& adma_copy,
    Tensor<GEngine, GLayout> const& gtensor,
    SmemLayoutOutput const& smem_layout_output,
    TileShape const& tile_shape) {
  auto [tma_details, aux_params, adma_gbasis, gmem_shape, gmem_stride, sbox_shape, sbox_stride] =
    detail::make_gmem_details_params(gtensor, smem_layout_output, tile_shape);
  constexpr int t_dim = decltype(rank(adma_gbasis))::value;
  static_assert(tuple_size<decltype(gmem_stride)>::value == t_dim - 1,
                "gmem_stride tuple size must be t_dim-1 (excludes innermost contiguous dimension).");
  // ADMA DimStride stores strides starting from dimension-1; prepend dimension-0's contiguous stride.
  auto gmem_layout = make_layout(gmem_shape, prepend<t_dim>(gmem_stride, uint64_t{1}));
  auto sbox_layout = make_layout(sbox_shape, sbox_stride);
  adma_copy.update_gmem_details_and_params(tma_details, aux_params, gmem_layout, sbox_layout);
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
          // Flat 2D: coalesce per-mode so rank is preserved (avoids collapsing
          // contiguous MN-major layouts like (128,128):(1,128) into rank-1).
          return coalesce(slayout, Step<_1, _1>{});
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
          // Flat 2D: coalesce per-mode so rank is preserved (avoids collapsing
          // contiguous MN-major layouts like (128,128):(1,128) into rank-1).
          return coalesce(slayout, Step<_1, _1>{});
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
