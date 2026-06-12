#pragma once

#include <cute/arch/config.hpp>
#include <cute/arch/xe4_inline_pisa.hpp>
#include <cute/arch/mma.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cute {

namespace AMMA {
  enum class Major : uint8_t {
    K  = 0,
    MN = 1
  };
}

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
  MMAControl nullC() const noexcept {
    auto ret = *this;
    ret.NullC = 1;
    return ret;
  }
};

//
// Tensor payload is in GRF however we use tensordesc pointer and tensordesc.fill
// to operate each lane (fields).
//
// Will be used for Universal Tensor descriptor, and change to Opaque later
//
union alignas(64) TensorPayload {
  struct {
    uint16_t ROITensorDimSize[5];

    uint16_t ElementStride0:3;
    uint16_t ElementStride1:3;
    uint16_t ElementStride2:3;
    uint16_t ElementStride3:3;
    uint16_t ElementStride4:3, : 1;

    uint32_t DimSize[5];
    uint64_t DimStride[4]; // stride 1,2,3,4
  };

  uint64_t raw_[8];

};

static_assert(sizeof(TensorPayload::raw_) == sizeof(TensorPayload));

template <typename T>
struct TensorDescriptor {
  cute::array<uint64_t, 8> payload;
  uint32_t matrix_desc;
  const T *g_pointer;
};

static inline void dupTensorPayload(uint64_t* pTDesc, uint64_t* pSrc) {
#if defined(__SYCL_DEVICE_ONLY__)
  auto* pTPld = reinterpret_cast<TensorPayload *>(pSrc);
  uint32_t roi0 = (uint32_t)pTPld ->ROITensorDimSize[0];
  uint32_t roi1 = (uint32_t)pTPld ->ROITensorDimSize[1];
  uint32_t roi2 = (uint32_t)pTPld ->ROITensorDimSize[2];
  uint32_t roi3 = (uint32_t)pTPld ->ROITensorDimSize[3];
  uint32_t roi4 = (uint32_t)pTPld ->ROITensorDimSize[4];
  asm volatile(
      "tensordesc.fill.roitensor_dim_size.32b [%0], 0, %1;\n"
      "tensordesc.fill.roitensor_dim_size.32b [%0], 1, %2;\n"
      "tensordesc.fill.roitensor_dim_size.32b [%0], 2, %3;\n"
      "tensordesc.fill.roitensor_dim_size.32b [%0], 3, %4;\n"
      "tensordesc.fill.roitensor_dim_size.32b [%0], 4, %5;\n"
      ::"r"(pTDesc), "r"(roi0),
      "r"(roi1), "r"(roi2),
      "r"(roi3), "r"(roi4));

  uint32_t es0 = pTPld->ElementStride0;
  uint32_t es1 = pTPld->ElementStride1;
  uint32_t es2 = pTPld->ElementStride2;
  uint32_t es3 = pTPld->ElementStride3;
  uint32_t es4 = pTPld->ElementStride4;

  asm volatile(
      "tensordesc.fill.element_stride.32b [%0], 0, %1;\n"
      "tensordesc.fill.element_stride.32b [%0], 1, %2;\n"
      "tensordesc.fill.element_stride.32b [%0], 2, %3;\n"
      "tensordesc.fill.element_stride.32b [%0], 3, %4;\n"
      "tensordesc.fill.element_stride.32b [%0], 4, %5;\n"
      ::"r"(pTDesc), "r"((uint32_t)es0),
      "r"((uint32_t)es1), "r"((uint32_t)es2),
      "r"((uint32_t)es3), "r"((uint32_t)es4));

  uint32_t dimsize0 = (uint32_t)pTPld->DimSize[0];
  uint32_t dimsize1 = (uint32_t)pTPld->DimSize[1];
  uint32_t dimsize2 = (uint32_t)pTPld->DimSize[2];
  uint32_t dimsize3 = (uint32_t)pTPld->DimSize[3];
  uint32_t dimsize4 = (uint32_t)pTPld->DimSize[4];

  asm volatile (
      "tensordesc.fill.dim_size.32b [%0], 0, %1;\n"
      "tensordesc.fill.dim_size.32b [%0], 1, %2;\n"
      "tensordesc.fill.dim_size.32b [%0], 2, %3;\n"
      "tensordesc.fill.dim_size.32b [%0], 3, %4;\n"
      "tensordesc.fill.dim_size.32b [%0], 4, %5;\n"
      ::"r"(pTDesc), "r"(dimsize0), "r"(dimsize1),
      "r"(dimsize2), "r"(dimsize3), "r"(dimsize4));

  uint64_t dimstride1 = pTPld->DimStride[0];
  uint64_t dimstride2 = pTPld->DimStride[1];
  uint64_t dimstride3 = pTPld->DimStride[2];
  uint64_t dimstride4 = pTPld->DimStride[3];

  asm volatile(
      "tensordesc.fill.dim_stride.64b [%0], 1, %1;"
      "tensordesc.fill.dim_stride.64b [%0], 2, %2;"
      "tensordesc.fill.dim_stride.64b [%0], 3, %3;"
      "tensordesc.fill.dim_stride.64b [%0], 4, %4;"
      ::"r"(pTDesc), "r"(dimstride1), "r"(dimstride2),
      "r"(dimstride3), "r"(dimstride4));
#endif
}

template <size_t Dim>
static inline void fillTensorDescriptorDimSize(
    uint64_t* pTDesc, cute::array<uint32_t, Dim>& dimSize
) {
#if defined(__SYCL_DEVICE_ONLY__)
  asm volatile (
      "tensordesc.fill.dim_size.32b [%0], 0, %1;"
      ::"r"(pTDesc), "r"(dimSize[0] -1));
  if constexpr (Dim > 1)
    asm volatile (
        "tensordesc.fill.dim_size.32b [%0], 1, %1;"
        ::"r"(pTDesc), "r"(dimSize[1] -1));
  if constexpr (Dim > 2)
    asm volatile (
        "tensordesc.fill.dim_size.32b [%0], 2, %1;"
        ::"r"(pTDesc), "r"(dimSize[2] -1));
  if constexpr (Dim > 3)
    asm volatile (
        "tensordesc.fill.dim_size.32b [%0], 3, %1;"
        ::"r"(pTDesc), "r"(dimSize[3] -1));
  if constexpr (Dim > 4)
    asm volatile ("tensordesc.fill.dim_size.32b [%0], 4, %1;"
        ::"r"(pTDesc), "r"(dimSize[4] -1));
#else
  auto* pTPld = reinterpret_cast<TensorPayload *>(pTDesc);
  for (int i = 0; i < Dim; ++ i)
    pTPld->DimSize[i] = dimSize[i] -1;
#endif
}

// Be aware that stride 0 is data-type size.
template <size_t Dim>
static inline void fillTensorDescriptorDimStride(
    uint64_t *pTDesc, cute::array<uint64_t, Dim>& dimStride
){
#if defined(__SYCL_DEVICE_ONLY__)
  asm volatile("tensordesc.fill.dim_stride.64b [%0], 1, %1;"
      ::"r"(pTDesc), "r"(dimStride[0]));
  if constexpr (Dim > 1)
    asm volatile("tensordesc.fill.dim_stride.64b [%0], 2, %1;"
      ::"r"(pTDesc), "r"(dimStride[1]));
  if constexpr (Dim > 2)
    asm volatile("tensordesc.fill.dim_stride.64b [%0], 3, %1;"
      ::"r"(pTDesc), "r"(dimStride[2]));
  if constexpr (Dim > 3)
    asm volatile("tensordesc.fill.dim_stride.64b [%0], 4, %1;"
      ::"r"(pTDesc), "r"(dimStride[3]));
#else
  auto* pTPld = reinterpret_cast<TensorPayload *>(pTDesc);
  for (int i = 0; i < Dim; ++ i)
    pTPld->DimStride[i] = dimStride[i];
#endif
}

template <size_t Dim>
static inline void fillTensorDescriptorROI(
    uint64_t *pTDesc, cute::array<uint16_t, Dim> &roiSize
){
#if defined(__SYCL_DEVICE_ONLY__)
  asm volatile(
      "tensordesc.fill.roitensor_dim_size.32b [%0], 0, %1;"
      ::"r"(pTDesc), "r"(roiSize[0] -1));
  if (Dim > 1)
    asm volatile(
        "tensordesc.fill.roitensor_dim_size.32b [%0], 1, %1;"
        ::"r"(pTDesc), "r"(roiSize[1] -1));
  if (Dim > 2)
    asm volatile(
        "tensordesc.fill.roitensor_dim_size.32b [%0], 2, %1;"
        ::"r"(pTDesc), "r"(roiSize[2] -1));
  if (Dim > 3)
    asm volatile(
        "tensordesc.fill.roitensor_dim_size.32b [%0], 3, %1;"
        ::"r"(pTDesc), "r"(roiSize[3] -1));
  if (Dim > 4)
    asm volatile(
        "tensordesc.fill.roitensor_dim_size.32b [%0], 4, %1;"
        ::"r"(pTDesc), "r"(roiSize[4] -1));
#else
  auto* pTPld = reinterpret_cast<TensorPayload *>(pTDesc);
  for (int i = 0; i < Dim; ++ i)
    pTPld->ROITensorDimSize[i] = roiSize[i] -1;
#endif
}

template <size_t Dim>
static inline void fillTensorDescriptorElementStride(
    uint64_t *pTDesc,
    cute::array<uint32_t, Dim> &elemStride
) {
#if defined(__SYCL_DEVICE_ONLY__)
  asm volatile(
      "tensordesc.fill.element_stride.32b [%0], 0, %1;"
      ::"r"(pTDesc), "r"(elemStride[0] -1));
  if constexpr (Dim > 1)
    asm volatile(
        "tensordesc.fill.element_stride.32b [%0], 1, %1;"
        ::"r"(pTDesc), "r"(elemStride[1] -1));
  if constexpr (Dim > 2)
    asm volatile(
        "tensordesc.fill.element_stride.32b [%0], 2, %1;"
        ::"r"(pTDesc), "r"(elemStride[2] -1));
  if constexpr (Dim > 3)
    asm volatile(
        "tensordesc.fill.element_stride.32b [%0], 3, %1;"
        ::"r"(pTDesc), "r"(elemStride[3] -1));
  if constexpr (Dim > 4)
    asm volatile(
        "tensordesc.fill.element_stride.32b [%0], 4, %1;"
        ::"r"(pTDesc), "r"(elemStride[4] -1));
#else
  auto* pTPld = reinterpret_cast<TensorPayload *>(pTDesc);
  assert(elemStride[0] == 1);
  if (elemStride[0] != 1)
    std::cout<< "Warning! Element stride 0 must be 1" << std::endl;

  pTPld->ElementStride0 = 0;
  if (Dim > 1) pTPld->ElementStride1 = elemStride[1] -1;
  if (Dim > 2) pTPld->ElementStride2 = elemStride[2] -1;
  if (Dim > 3) pTPld->ElementStride3 = elemStride[3] -1;
  if (Dim > 4) pTPld->ElementStride4 = elemStride[4] -1;
#endif
}

//
// This structure is concept that'll never materialize through normal mean
// Abarrier object reside in internal memroy but operate through other means
//
union [[deprecated("use ClusterBarrier or ClusterTransactionBarrier APIs instead")]] Abarrier {
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

[[deprecated("use ClusterBarrier or ClusterTransactionBarrier APIs instead")]]
CUTE_HOST_DEVICE
void
xe4_initialize_barrier(uint64_t& smem_barrier,
                   int thread_count = 1)                   // Thread count expected to arrive/wait on this barrier
{
#if defined(__SYCL_DEVICE_ONLY__)
  abarrier_init(&smem_barrier, thread_count);
#endif
}

[[deprecated("use ClusterTransactionBarrier APIs instead")]]
CUTE_HOST_DEVICE
void
xe4_set_barrier_transaction_bytes(uint64_t& smem_barrier,
                              uint32_t bytes)              // Number of bytes transfered by per TMA transaction
{
#if defined(__SYCL_DEVICE_ONLY__)
  abarrier_workgroup_arrive_expect_tx(&smem_barrier, bytes);
#endif
}

[[deprecated("use ClusterBarrier or ClusterTransactionBarrier APIs instead")]]
CUTE_HOST_DEVICE
void
xe4_wait_barrier(uint64_t& smem_barrier,
             int phase_bit)                                // Current phase bit the barrier waiting to flip
{
#if defined(__SYCL_DEVICE_ONLY__)
  abarrier_try_wait(&smem_barrier, phase_bit);
#endif
}

[[deprecated("use ClusterBarrier APIs instead")]]
CUTE_HOST_DEVICE
void
xe4_arrive_barrier(uint64_t& smem_barrier, int thread_count = 1)
{
#if defined(__SYCL_DEVICE_ONLY__)
  abarrier_workgroup_arrives(&smem_barrier, thread_count);
#endif
}

}
