#pragma once

#include "async_tensor_copy.hpp"

namespace cute {

// Forward declaration: XE4_ADMA_PREFETCH is defined later in this file.
// Needed here so XE4_ADMA_LOAD and XE4_ADMA_LOAD_MULTICAST can declare
// `using PREFETCH = XE4_ADMA_PREFETCH;` — the typedef that enables
// the generic cute::prefetch(Copy_Atom<...>) overload in prefetch.hpp
// to derive prefetch traits from any load atom via CopyOp::PREFETCH.
struct XE4_ADMA_PREFETCH;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_TENSOR_LOAD: Initiates a async tensor copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct XE4_ADMA_LOAD
{
  // PREFETCH typedef: enables cute::prefetch(load_atom, src) to construct
  // Copy_Traits<XE4_ADMA_PREFETCH> from Copy_Traits<XE4_ADMA_LOAD> automatically.
  // See prefetch.hpp line ~100: `using Prefetch_Traits = Copy_Traits<typename CopyOp::PREFETCH, ...>`
  using PREFETCH = XE4_ADMA_PREFETCH;

  template <typename DataType, size_t Dim,
            detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
            detail::FillMethod FM = detail::FillMethod::Zero>
  CUTE_HOST_DEVICE static void
  copy(
      uint64_t* tdesc_ptr, DataType const* gmem_ptr, const uint32_t mat_desc,
      uint64_t *abar_ptr, DataType* slm_ptr, const sycl::vec_t<int, Dim>& coord,
      detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}
  ) {
    MatrixDescriptor mat_desc_(mat_desc);
    mat_desc_.StartAddress = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(slm_space_cast(slm_ptr))) >> 9;

    detail::AsyncTensorGlobal2SLM<DataType>::
    Copy(mat_desc_, gmem_ptr, abar_ptr, reinterpret_cast<TensorPayload *>(tdesc_ptr), coord,
         detail::CacheHint<CC>{}, detail::FillMode<FM>{});
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_TENSOR_STORE : Initiates a async tensor copy from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct XE4_ADMA_STORE
{
  template <typename DataType, size_t Dim,
            detail::CacheCtrl CC = detail::CacheCtrl::L2wb_L3uc>
  CUTE_HOST_DEVICE static void
  copy(
      DataType* gmem_ptr, uint64_t* tdesc_ptr, const uint32_t mat_desc,
      uint64_t* abar_ptr, DataType* slm_ptr, const sycl::vec_t<int, Dim>& coord,
      detail::CacheHint<CC> = {}
  ) {
    MatrixDescriptor mat_desc_(mat_desc);
    mat_desc_.StartAddress = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(slm_space_cast(slm_ptr))) >> 9;

    // Use sizeof_bits in near future
    detail::AsyncTensorSLM2Global<sizeof(DataType) * 8>::
      Copy(gmem_ptr, mat_desc_, abar_ptr, reinterpret_cast<TensorPayload *>(tdesc_ptr), coord,
           detail::CacheHint<CC>{});
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_TENSOR_LOAD_MULTICAST: Initiates a async tensor copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct XE4_ADMA_LOAD_MULTICAST
{
  // Same PREFETCH typedef as XE4_ADMA_LOAD — multicast loads can also derive prefetch.
  using PREFETCH = XE4_ADMA_PREFETCH;

  template<typename DataType, size_t Dim,
           detail::CacheCtrl CC = detail::CacheCtrl::L2c_L3uc,
           detail::FillMethod FM = detail::FillMethod::Zero>
  CUTE_HOST_DEVICE static void
  copy(
      uint64_t* tdesc_ptr, DataType const* gmem_ptr, const uint32_t mat_desc,
      uint64_t* abar_ptr, uint32_t multicast_mask, DataType* slm_ptr,
      const sycl::vec_t<int, Dim>& coord,
      detail::CacheHint<CC> = {}, detail::FillMode<FM> = {}
  ) {
    MatrixDescriptor mat_desc_ (mat_desc);
    mat_desc_.StartAddress = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(slm_space_cast(slm_ptr))) >> 9;

    detail::AsyncTensorGlobal2SLM<DataType>::
      Copy(mat_desc_, gmem_ptr, abar_ptr, reinterpret_cast<TensorPayload *>(tdesc_ptr), coord, multicast_mask,
           detail::CacheHint<CC>{}, detail::FillMode<FM>{});
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// XE4_ADMA_RedBase — Compile-time constraint checker for async tensor reduction ops.
///
/// The XE4 async tensor reduce instructions (fred/ired) impose hardware constraints on
/// which (DataType, RedOp) combinations are valid. This base class uses static_assert
/// to catch invalid combinations at compile time:
///
///   IntType (ired):
///     - Bit-width must be 32 or 64
///     - Ops: Add, Smin, Smax, Umin, Umax, And, Or, Xor
///
///   FloatType (fred):
///     - Ops: Add, Min, Max
///     - Min/Max only supported for 16-bit types (half, bf16)
///     - float/double/tf32 only support Add
///
/// Inherited by XE4_ADMA_STORE_REDUCE to validate at instantiation time.
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, RedOp Rop>
struct XE4_ADMA_RedBase {

  // Classify T as IntType or FloatType for asm instruction selection (ired vs fred)
  static constexpr RedType RedDType = (std::is_same_v<T, int> ||
                                    std::is_same_v<T, uint32_t>)
                                 ? RedType::IntType
                                 : ((std::is_same_v<T, cutlass::tfloat32_t> ||
                                      std::is_same_v<T, float> ||
                                      std::is_same_v<T, double> ||
                                      std::is_same_v<T, ::sycl::ext::oneapi::bfloat16> ||
                                      std::is_same_v<T, cutlass::half_t> ||
                                      std::is_same_v<T, ::sycl::half>)
                                 ? RedType::FloatType
                                 : RedType::none);

  // Validate that the (T, Rop) combination is supported by the hardware
  static constexpr inline bool check_constraints() {

    static_assert(RedDType != RedType::none);
    constexpr unsigned int BitWidth =sizeof_bits_v<T>;
    if constexpr (RedDType ==RedType::IntType) {
      // ired: 32-bit or 64-bit integers only
      static_assert(cmp_values<BitWidth, 32, 64>());
      // ired: supported ops are Add, signed/unsigned min/max, and bitwise ops
      static_assert(cmp_values<static_cast<int>(Rop),
                               static_cast<int>(RedOp::Add), static_cast<int>(RedOp::Smin),
                               static_cast<int>(RedOp::Smax), static_cast<int>(RedOp::Umin),
                               static_cast<int>(RedOp::Umax), static_cast<int>(RedOp::And),
                               static_cast<int>(RedOp::Or), static_cast<int>(RedOp::Xor)>());
    }
    if constexpr (RedDType ==RedType::FloatType) {
      // fred: only Add, Min, Max supported
      static_assert(cmp_values<static_cast<int>(Rop),
                               static_cast<int>(RedOp::Add),
                               static_cast<int>(RedOp::Min),
                               static_cast<int>(RedOp::Max)>());
      // fred Min/Max only for 16-bit float types (half, bf16); 32-bit+ only supports Add
      constexpr bool half_type = (std::is_same_v<T, ::sycl::ext::oneapi::bfloat16> ||
                                  std::is_same_v<T, cutlass::half_t> ||
                                  std::is_same_v<T, ::sycl::half>);
      if constexpr (!half_type) {
        static_assert(Rop == RedOp::Add);
      }
    }
    return true;
  }
  static_assert(check_constraints());
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// XE4_ADMA_PREFETCH — CopyOp for fire-and-forget global memory prefetch.
///
/// Issues async_tensor_prefetch to warm L2 cache with tensor tile data.
/// No SLM destination, no barrier — the instruction is non-blocking and has no completion signal.
///
/// Directly executable: Copy_Traits<XE4_ADMA_PREFETCH> has inline copy_unpack (no .with() needed),
/// unlike LOAD/STORE which require .with(abar) to become executable.
///
/// Cache policy: L2c_L3uc (cache in L2, uncached in L3) — brings data close to the EU
/// without polluting L3.
///
/// Self-referencing PREFETCH typedef: allows cute::prefetch(prefetch_atom, src) to work
/// through the same generic path as load atoms (CopyOp::PREFETCH → self).
////////////////////////////////////////////////////////////////////////////////////////////////////
struct XE4_ADMA_PREFETCH
{
  using PREFETCH = XE4_ADMA_PREFETCH;

  // Simplified copy() signature: only needs tdesc + gmem_ptr + coord (no abar, no slm_ptr)
  template <typename DataType, size_t Dim>
  CUTE_HOST_DEVICE static void
  copy(uint64_t* tdesc_ptr, DataType const* gmem_ptr, const sycl::vec_t<int, Dim>& coord) {

    detail::AsyncTensorGlobalPrefetch<DataType, detail::CacheCtrl::L2c_L3uc>::
    Copy(gmem_ptr, reinterpret_cast<TensorPayload *>(tdesc_ptr), coord);
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// XE4_ADMA_STORE_REDUCE — CopyOp for SLM → gmem atomic reduction.
///
/// Atomically reduces SLM data into global memory using the specified reduction operation.
/// Inherits from XE4_ADMA_RedBase<T, Rop> which validates (T, Rop) at compile time.
///
/// Follows the same copy() signature order as XE4_ADMA_STORE (unified with ADMA_STORE_Unpack):
///   (tdesc_ptr, gmem_ptr, mat_desc, slm_ptr, coord, abar_ptr)
/// This allows it to reuse ADMA_STORE_Unpack for argument explosion in Copy_Traits.
///
/// Template params:
///   T     — element type (half, bf16, float, int, uint32_t, etc.)
///   Rop   — reduction operation (Add, Min, Max, Smin, Smax, Umin, Umax, And, Or, Xor)
///   BType — barrier type (Abarrier or Groupsync)
///
/// Cache policy: L2wb_L3wb (write-back at both levels — HW constraint for reduce ops).
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, RedOp Rop, BarrierType BType>
struct XE4_ADMA_STORE_REDUCE : public XE4_ADMA_RedBase<T, Rop>
{
  using Super = XE4_ADMA_RedBase<T, Rop>;
  template <size_t Dim>
  CUTE_HOST_DEVICE static void
  copy(
      uint64_t* tdesc_ptr, T const* gmem_ptr, const uint32_t mat_desc,
      T* slm_ptr, const sycl::vec_t<int, Dim>& coord, uint64_t *abar_ptr
  ) {
    // Patch the matrix descriptor's StartAddress with the SLM base address.
    // SLM addresses are shifted right by 9 bits per the hardware spec.
    MatrixDescriptor mat_desc_(mat_desc);
    mat_desc_.StartAddress = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(slm_space_cast(slm_ptr))) >> 9;

    // Dispatch to the asm wrapper: selects fred/ired specialization based on Super::RedDType,
    // and Abarrier vs Groupsync based on BType.
    detail::AsyncTensorReduce<Super::RedDType>::template Copy<detail::CacheCtrl::L2wb_L3wb,
                              Rop, BType>(mat_desc_, gmem_ptr, abar_ptr,
                              reinterpret_cast<TensorPayload *>(tdesc_ptr), coord);
  }
};


} // namespace cute
