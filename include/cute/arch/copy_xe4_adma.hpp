#pragma once

#include "async_tensor_copy.hpp"

namespace cute {

////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_TENSOR_LOAD: Initiates a async tensor copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct XE4_ADMA_LOAD
{
  template <typename DataType, size_t Dim>
  CUTE_HOST_DEVICE static void
  copy(
      uint64_t* tdesc_ptr, DataType const* gmem_ptr, const uint32_t mat_desc,
      uint64_t *abar_ptr, DataType* slm_ptr, const sycl::vec_t<int, Dim>& coord
  ) {
    MatrixDescriptor mat_desc_(mat_desc);
    mat_desc_.StartAddress = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(slm_space_cast(slm_ptr))) >> 9;

    detail::AsyncTensorGlobal2SLM<DataType, detail::CacheCtrl::L2c_L3uc, detail::FillMethod::Zero>::
    Copy(mat_desc_, gmem_ptr, abar_ptr, reinterpret_cast<TensorPayload *>(tdesc_ptr), coord);
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_TENSOR_STORE : Initiates a async tensor copy from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct XE4_ADMA_STORE
{
  template <typename DataType, size_t Dim>
  CUTE_HOST_DEVICE static void
  copy(
      DataType* gmem_ptr, uint64_t* tdesc_ptr, const uint32_t mat_desc,
      uint64_t* abar_ptr, DataType* slm_ptr, const sycl::vec_t<int, Dim>& coord
  ) {
    MatrixDescriptor mat_desc_(mat_desc);
    mat_desc_.StartAddress = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(slm_space_cast(slm_ptr))) >> 9;

    // Use sizeof_bits in near future
    detail::AsyncTensorSLM2Global<sizeof(DataType) * 8, detail::CacheCtrl::L2wb_L3uc>::
      Copy(gmem_ptr, mat_desc_, abar_ptr, reinterpret_cast<TensorPayload *>(tdesc_ptr), coord);
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_TENSOR_LOAD_MULTICAST: Initiates a async tensor copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct XE4_ADMA_LOAD_MULTICAST
{
  template<typename DataType, size_t Dim>
  CUTE_HOST_DEVICE static void
  copy(
      uint64_t* tdesc_ptr, DataType const* gmem_ptr, const uint32_t mat_desc,
      uint64_t* abar_ptr, uint32_t multicast_mask, DataType* slm_ptr,
      const sycl::vec_t<int, Dim>& coord
  ) {
    MatrixDescriptor mat_desc_ (mat_desc);
    mat_desc_.StartAddress = static_cast<uint32_t>(
        reinterpret_cast<uint64_t>(slm_space_cast(slm_ptr))) >> 9;

    detail::AsyncTensorGlobal2SLM<DataType, detail::CacheCtrl::L2c_L3uc, detail::FillMethod::Zero>::
      Copy(mat_desc_, gmem_ptr, abar_ptr, reinterpret_cast<TensorPayload *>(tdesc_ptr), coord, multicast_mask);
  }
};

} // namespace cute
