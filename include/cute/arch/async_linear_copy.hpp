#pragma once

#include <cute/arch/asm_helper.hpp>
#include <cute/arch/mma_xe4_desc.hpp>

namespace cute {
namespace detail {
struct AsyncLinearGlobal2SLM
{
  static inline void
  Copy(void* slm_ptr, void* gmem_ptr, uint32_t copy_size, uint64_t const* abar_ptr)
  {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile("async_linear_copy.shared_workgroup.global.L2c.L3uc.abarrier [%0], [%1], [%2], %3;" ::"r"(slm_ptr),
              "r"(gmem_ptr), "r"(abar_ptr), "r"(copy_size));
#endif
  }
};

struct AsyncLinearSLM2Global
{
  static inline void
  Copy(void* gmem_ptr, void* slm_ptr, uint32_t copy_size, uint64_t const* abar_ptr)
  {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile("async_linear_copy.global.shared_workgroup.L2wb.L3uc.abarrier [%0], [%1], [%2], %3;" ::"r"(gmem_ptr),
              "r"(slm_ptr), "r"(abar_ptr), "r"(copy_size));
#endif
  }
};

}
}
