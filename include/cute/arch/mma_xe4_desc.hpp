#pragma once

#include <cute/arch/config.hpp>

#include <cute/arch/mma.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cute {

union MatrixDescriptor
{
  enum {Type1 = 0, Type2, Type3};
  constexpr MatrixDescriptor() noexcept : raw_(0) {}
  constexpr MatrixDescriptor(uint32_t desc) noexcept : raw_(desc) {}
  constexpr MatrixDescriptor(MatrixDescriptor const &desc) noexcept : raw_(desc.raw_) {}
  constexpr MatrixDescriptor(MatrixDescriptor && desc) noexcept : raw_(desc.raw_) {}
  constexpr MatrixDescriptor& operator=(MatrixDescriptor const& desc) noexcept {
    raw_ = desc.raw_;
    return *this;
  }
  constexpr MatrixDescriptor& operator=(MatrixDescriptor && desc) noexcept {
    raw_ = desc.raw_;
    return *this;
  }

  struct {
    unsigned StartAddress : 12, : 4;
    unsigned Pitch : 11, : 1;
    unsigned Type : 2;
  };

  uint32_t raw_;

  constexpr operator uint32_t() const noexcept { return raw_; }
};

static inline void print(MatrixDescriptor const& desc) {
#if defined(__SYCL_DEVICE_ONLY__)
//  using sycl::ext::oneapi::experimental::printf;
#endif
  printf("MatrixDescriptor: %#08x\n", desc.raw_);
  printf("  StartAddress: %#04x\n", desc.StartAddress);
  printf("  Pitch       : %#04x (%d)\n", desc.Pitch, desc.Pitch);
  printf("  Type        : %d\n", desc.Type);
}

union MMAControl {
  constexpr MMAControl() noexcept : raw_(0) {}
  constexpr MMAControl(uint64_t ctrl) noexcept : raw_(ctrl) {}
  constexpr MMAControl(MMAControl const &ctrl) noexcept : raw_(ctrl.raw_) {}
  constexpr MMAControl(MMAControl && ctrl) noexcept : raw_(ctrl.raw_) {}
  constexpr MMAControl& operator=(MMAControl const& ctrl) noexcept {
    raw_ = ctrl.raw_;
    return *this;
  }
  constexpr MMAControl& operator=(MMAControl && ctrl) noexcept {
    raw_ = ctrl.raw_;
    return *this;
  }

  struct {
    unsigned A_FlushToZero : 1;
    unsigned B_FlushToZero : 1;
    unsigned D_FlushToZero : 1;
    unsigned InvFlagEn : 1;
    unsigned OvflFlagEn : 1;
    unsigned SatDst : 1;
    unsigned SatSrc : 1;
    unsigned SatMxi : 1;
    unsigned NullC : 1;
    unsigned NegateAB : 1, : 6;
    unsigned SparsityType : 2, : 2;
    unsigned A_BlockScaleType : 3, : 1;
    unsigned B_BlockScaleType : 3;
  };

  uint64_t raw_;
  constexpr operator uint64_t() const noexcept { return raw_; }
};

//
// Tensor payload is in GRF however we use tensordesc pointer and tensordesc.fill
// to operate each lane (fields).
//
// Will be used for Universal Tensor descriptor, and change to Opaque later
//
union alignas(64) TensorPayload {
  struct {
    uint16_t ROITensorDimSize0;
    uint16_t ROITensorDimSize1;
    uint16_t ROITensorDimSize2;
    uint16_t ROITensorDimSize3;
    uint16_t ROITensorDimSize4;

    uint16_t ElementStride0:3;
    uint16_t ElementStride1:3;
    uint16_t ElementStride2:3;
    uint16_t ElementStride3:3;
    uint16_t ElementStride4:3, : 1;

    uint32_t DimSize0;
    uint32_t DimSize1;
    uint32_t DimSize2;
    uint32_t DimSize3;
    uint32_t DimSize4;

    uint64_t DimStride1;
    uint64_t DimStride2;
    uint64_t DimStride3;
    uint64_t DimStride4;
  };

  uint64_t raw_[8];
};

using TensorDescriptor = TensorPayload;

static_assert(sizeof(TensorPayload::raw_) == sizeof(TensorPayload));

template <uint32_t dim_num, typename dims_t>
inline void tensordesc_fill_dim_size(uint64_t* pTDesc, const dims_t &dim_sizes) {
  static_assert(dim_num > 0 && dim_num <= 5,
      "Unsupported dimention size for tensor descriptor dim_size");
  static_assert(sizeof(dims_t {}[0]) == sizeof(uint32_t), "Invalid dim size type");

#if defined(__SYCL_DEVICE_ONLY__)
  asm volatile ("tensordesc.fill.dim_size.32b [%0], 0, %1;" ::"r"(pTDesc), "r"(dim_sizes[0] -1));
  if constexpr (dim_num > 1)
    asm volatile ("tensordesc.fill.dim_size.32b [%0], 1, %1;" ::"r"(pTDesc), "r"(dim_sizes[1] -1));
  if constexpr (dim_num > 2)
    asm volatile ("tensordesc.fill.dim_size.32b [%0], 2, %1;" ::"r"(pTDesc), "r"(dim_sizes[1] -1));
  if constexpr (dim_num > 3)
    asm volatile ("tensordesc.fill.dim_size.32b [%0], 3, %1;" ::"r"(pTDesc), "r"(dim_sizes[1] -1));
  if constexpr (dim_num > 4)
    asm volatile ("tensordesc.fill.dim_size.32b [%0], 4, %1;" ::"r"(pTDesc), "r"(dim_sizes[1] -1));
  if constexpr (dim_num > 5)
    asm volatile ("tensordesc.fill.dim_size.32b [%0], 5, %1;" ::"r"(pTDesc), "r"(dim_sizes[1] -1));
#endif
}

template <uint32_t dim_num>
inline void tensordesc_set_dim_size(uint64_t* pTDesc, uint32_t dim_size) {
  static_assert(dim_num > 0 && dim_num <= 5,
      "Unsupported dimention size for tensor descriptor dim_size");

#if defined(__SYCL_DEVICE_ONLY__)
  if constexpr (dim_num == 1) {
    asm volatile("tensordesc.fill.dim_size.32b [%0], 0, %1;" ::"r"(pTDesc), "r"(dim_size -1));
  } else if constexpr (dim_num == 2) {
    asm volatile("tensordesc.fill.dim_size.32b [%0], 1, %1;" ::"r"(pTDesc), "r"(dim_size -1));
  } else if constexpr (dim_num == 3) {
    asm volatile("tensordesc.fill.dim_size.32b [%0], 2, %1;" ::"r"(pTDesc), "r"(dim_size -1));
  } else if constexpr (dim_num == 4) {
    asm volatile("tensordesc.fill.dim_size.32b [%0], 3, %1;" ::"r"(pTDesc), "r"(dim_size -1));
  } else if constexpr (dim_num == 5) {
    asm volatile("tensordesc.fill.dim_size.32b [%0], 4, %1;" ::"r"(pTDesc), "r"(dim_size -1));
  }
#endif
}

template <uint32_t dim_num, typename strides_t>
inline void tensordesc_fill_dim_stride(uint64_t* pTDesc, const strides_t &dim_strides) {
  static_assert(dim_num >= 2 && dim_num <= 5, "Unsupported dimention size fortensor descriptor dim_stride");
  static_assert(sizeof(strides_t {}[0]) == sizeof(uint64_t), "Invalid stride type");

#if defined(__SYCL_DEVICE_ONLY__)
  asm volatile("tensordesc.fill.dim_stride.64b [%0], 1, %1;" ::"r"(pTDesc), "r"(dim_strides[0]));
  if constexpr (dim_num > 2)
    asm volatile("tensordesc.fill.dim_stride.64b [%0], 2, %1;" ::"r"(pTDesc), "r"(dim_strides[1]));
  if constexpr (dim_num > 3)
    asm volatile("tensordesc.fill.dim_stride.64b [%0], 3, %1;" ::"r"(pTDesc), "r"(dim_strides[2]));
  if constexpr (dim_num > 4)
    asm volatile("tensordesc.fill.dim_stride.64b [%0], 4, %1;" ::"r"(pTDesc), "r"(dim_strides[3]));
#endif
}

template <uint32_t dim_num, typename dims_t>
inline void tensordesc_fill_roitensor_dim_size(uint64_t* pTDesc, const dims_t &roitensor_sizes) {
  static_assert(dim_num > 0 && dim_num <= 5, "Unsupported dimention size for tensor descriptor roitensor_size");
  static_assert(sizeof(dims_t {}[0]) == sizeof(uint32_t), "Invalid roitensor size type");

#if defined(__SYCL_DEVICE_ONLY__)
  asm volatile("tensordesc.fill.roitensor_dim_size.32b [%0], 0, %1;" ::"r"(pTDesc), "r"(roitensor_sizes[0] -1));
  if constexpr (dim_num > 1)
    asm volatile("tensordesc.fill.roitensor_dim_size.32b [%0], 1, %1;" ::"r"(pTDesc), "r"(roitensor_sizes[1] -1));
  if constexpr (dim_num > 2)
    asm volatile("tensordesc.fill.roitensor_dim_size.32b [%0], 2, %1;" ::"r"(pTDesc), "r"(roitensor_sizes[2] -1));
  if constexpr (dim_num > 3)
    asm volatile("tensordesc.fill.roitensor_dim_size.32b [%0], 3, %1;" ::"r"(pTDesc), "r"(roitensor_sizes[3] -1));
  if constexpr (dim_num > 4)
    asm volatile("tensordesc.fill.roitensor_dim_size.32b [%0], 4, %1;" ::"r"(pTDesc), "r"(roitensor_sizes[4] -1));
#endif
}

template <uint32_t dim_num, typename strides_t>
inline void tensordesc_fill_element_stride(uint64_t* pTDesc, const strides_t &element_stride) {
  static_assert(dim_num > 0 && dim_num <= 5, "Unsupported dimention size for tensor descriptor element_stride");
  static_assert(sizeof(strides_t {}[0]) == sizeof(uint32_t), "Invalid element stride type");

#if defined(__SYCL_DEVICE_ONLY__)
  asm volatile("tensordesc.fill.element_stride.32b [%0], 0, %1;" ::"r"(pTDesc), "r"(element_stride[0] -1));
  if constexpr (dim_num > 1)
    asm volatile("tensordesc.fill.element_stride.32b [%0], 1, %1;" ::"r"(pTDesc), "r"(element_stride[1] -1));
  if constexpr (dim_num > 2)
    asm volatile("tensordesc.fill.element_stride.32b [%0], 2, %1;" ::"r"(pTDesc), "r"(element_stride[2] -1));
  if constexpr (dim_num > 3)
    asm volatile("tensordesc.fill.element_stride.32b [%0], 3, %1;" ::"r"(pTDesc), "r"(element_stride[3] -1));
  if constexpr (dim_num > 4)
    asm volatile("tensordesc.fill.element_stride.32b [%0], 4, %1;" ::"r"(pTDesc), "r"(element_stride[4] -1));
#endif
}

//
// This structure is concept that'll never materialize through normal mean
// Abarrier object reside in internal memroy but operate through other means
//
union Abarrier {
  constexpr Abarrier() noexcept : raw_(0) {}
  constexpr Abarrier(uint64_t ctrl) noexcept : raw_(ctrl) {}
  constexpr Abarrier(Abarrier const &ctrl) noexcept : raw_(ctrl.raw_) {}
  constexpr Abarrier(Abarrier && ctrl) noexcept : raw_(ctrl.raw_) {}
  constexpr Abarrier& operator=(Abarrier const& ctrl) noexcept {
    raw_ = ctrl.raw_;
    return *this;
  }
  constexpr Abarrier& operator=(Abarrier && ctrl) noexcept {
    raw_ = ctrl.raw_;
    return *this;
  }
  struct {
    int64_t phase : 1, : 1;
    int64_t compl_func_flag : 2;
    int64_t pending_tx_ops : 28;
    int64_t expected_arrivals : 16;
    int64_t pending_arrivals : 16;
  };

  uint64_t raw_;
};

/* union AbarrierAddress {
  struct {
    unsigned resv0:2;
    unsigned barrierOffset:20;
    unsigned invalid:1;
    unsigned wg_rank:3;
  };

  uint32_t raw;
};*/

// Initialize barrier present in shared memory
CUTE_HOST_DEVICE
void
xe4_initialize_barrier(uint64_t& smem_barrier,                 // 64 bits user-manged barrier in smem
                   int thread_count = 1)                   // Thread count expected to arrive/wait on this barrier
{
#if defined(__SYCL_DEVICE_ONLY__)
  abarrier_init(&smem_barrier, thread_count);
#endif
}

// Set the number of bytes transfered per transaction and perform an arrive operation as well
CUTE_HOST_DEVICE
void
xe4_set_barrier_transaction_bytes(uint64_t& smem_barrier,      // 64 bits user-manged barrier in smem
                              uint32_t bytes)              // Number of bytes transfered by per TMA transaction
{
#if defined(__SYCL_DEVICE_ONLY__)
  abarrier_workgroup_arrive_expect_tx(&smem_barrier, bytes);
#endif
}

// Barrier wait
CUTE_HOST_DEVICE
void
xe4_wait_barrier(uint64_t& smem_barrier,                       // 64 bits user-manged barrier in smem
             int phase_bit)                                // Current phase bit the barrier waiting to flip
{
#if defined(__SYCL_DEVICE_ONLY__)
  abarrier_try_wait(&smem_barrier, phase_bit);
#endif
}

// Barrier arrive
CUTE_HOST_DEVICE
void
xe4_arrive_barrier(uint64_t& smem_barrier, int thread_count = 1)                      // 64 bits user-manged barrier in smem
{
#if defined(__SYCL_DEVICE_ONLY__)
  abarrier_workgroup_arrives(&smem_barrier, thread_count);
#endif
}



}
