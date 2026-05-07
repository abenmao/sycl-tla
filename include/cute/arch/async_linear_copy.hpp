#pragma once

#include <cute/arch/async_tensor_copy.hpp>

namespace cute {
namespace detail {
struct AsyncLinearGlobal2SLM
{
  template <CacheCtrl CacheType = CacheCtrl::L2c_L3uc,
            BarrierType BarType = BarrierType::Abarrier>
  static inline void
  Copy(void* slm_ptr, void const* gmem_ptr, uint32_t copy_size, uint64_t const* abar_ptr, CacheHint<CacheType> = {})
  {
#if defined (__SYCL_DEVICE_ONLY__)
    if constexpr (BarType == BarrierType::Abarrier) {
      asm volatile (
        ("async_linear_copy.shared_workgroup.global"+_cc<CacheType>+".abarrier [%0], [%1], [%2], %3;")
        ::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(copy_size));
    } else {
      asm volatile (
        ("async_linear_copy.shared_workgroup.global"+_cc<CacheType>+".groupsync [%0], [%1], %2;")
        ::"r"(slm_ptr), "r"(gmem_ptr), "r"(copy_size));
    }
#endif
  }
};

struct AsyncLinearSLM2Global
{
  template <CacheCtrl CacheType = CacheCtrl::L2wb_L3uc,
            BarrierType BarType = BarrierType::Abarrier>
  static inline void
  Copy(void* gmem_ptr, void const* slm_ptr, uint32_t copy_size, uint64_t const* abar_ptr, CacheHint<CacheType> = {})
  {
#if defined (__SYCL_DEVICE_ONLY__)
    if constexpr (BarType == BarrierType::Abarrier) {
      asm volatile (
        ("async_linear_copy.global.shared_workgroup"+_cc<CacheType>+".abarrier [%0], [%1], [%2], %3;")
        ::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(copy_size));
    } else {
      asm volatile (
        ("async_linear_copy.global.shared_workgroup"+_cc<CacheType>+".groupsync [%0], [%1], %2;")
        ::"r"(gmem_ptr), "r"(slm_ptr), "r"(copy_size));
    }
#endif
  }
};

struct AsyncLinearMultiCastGlobal2SLM
{
  template <CacheCtrl CacheType = CacheCtrl::L2c_L3uc>
  static inline void
  Copy(void* slm_ptr, void const* gmem_ptr, uint32_t copy_size, uint64_t const* abar_ptr, uint32_t multicast_mask, CacheHint<CacheType> = {})
  {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
      ("async_linear_copy.shared_cluster.global"+_cc<CacheType>+".abarrier [%0], [%1], [%2], %3, %4;")
      ::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(copy_size), "r"(multicast_mask));
#endif
  }
};

struct AsyncLinearMultiCastLocal2RemoteSLM
{
  static inline void
  Copy(void* slm_ptr_dst, void const* slm_ptr_src, uint32_t copy_size, uint64_t const* abar_ptr, uint32_t multicast_mask)
  {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile("async_linear_copy.shared_cluster.shared_workgroup.abarrier [%0], [%1], [%2], %3, %4;"
        ::"r"(slm_ptr_dst), "r"(slm_ptr_src), "r"(abar_ptr), "r"(copy_size), "r"(multicast_mask));
#endif
  }
};

struct AsyncLinearCopyPrefetchFromGlobal
{
  // Cache control is hardcoded to L2-cached, L3-uncached (.L2c.L3uc).
  // This is the only cache policy supported by async_linear_prefetch per BSPEC.
  static inline void
  Prefetch(void* gmem_ptr, uint32_t size)
  {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile("async_linear_prefetch.L2c.L3uc.global [%0], %1;" ::"r"(gmem_ptr), "r"(size));
#endif
  }
};

}
}
