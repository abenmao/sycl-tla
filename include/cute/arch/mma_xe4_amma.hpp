#pragma once

#include "xe4_inline_pisa.hpp"
#include <cute/arch/mma_xe4_desc.hpp>

#include <cute/arch/asm_helper.hpp>

namespace cute {

namespace AMMA {
// Enumerate barrier tracking combinations
enum class Tracking {
  None = 0, GroupSync, D, A, B, DA, DB, AB, DAB, GA, GB, GAB
};

template <Tracking Method> struct TrackMethod {
  constexpr static Tracking value = Method;
};
}

// Major enumeration
template <AMMA::Major> struct ammajor;

template <> struct ammajor<AMMA::Major::MN> {
  static constexpr fixstr::fixed_string value {".am"};
};

template <> struct ammajor<AMMA::Major::K> {
  static constexpr fixstr::fixed_string value {""};
};

template <cute::AMMA::Major> struct bkmajor;
template <> struct bkmajor<AMMA::Major::MN> {
  static constexpr fixstr::fixed_string value {""};
};

template <> struct bkmajor<AMMA::Major::K> {
  static constexpr fixstr::fixed_string value {".bk"};
};

template <cute::AMMA::Major major> constexpr auto _am = ammajor<major>::value;
template <cute::AMMA::Major major> constexpr auto _bk = bkmajor<major>::value;

// Async-MMA barriers track none
template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA {
  using DRegisters = void;
  using ARegisters = void;
  using BRegisters = void;
  using CRegisters = void;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d, uint32_t const& desc_a,
      uint32_t const& desc_b, uint32_t const& desc_c
  ) {
#if defined(__SYCL_DEVICE_ONLY__)
    if ( cute::elect_one_sync() ) {
        asm volatile (
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+_am<a_major>+_bk<b_major>+" %0, %1, %2, %3, %4;\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c));
    }
#endif
  }
};

template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_GROUPSYNC {
  using DRegisters = void;
  using ARegisters = void;
  using BRegisters = void;
  using CRegisters = void;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d, uint32_t const& desc_a,
      uint32_t const& desc_b, uint32_t const& desc_c
  ) {
#if defined(__SYCL_DEVICE_ONLY__)
    if ( cute::elect_one_sync() ) {
        asm volatile (
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+_am<a_major>+_bk<b_major>+".groupsync %0, %1, %2, %3, %4;\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c));
    }
#endif
  }
};

// Async-MMA barriers track D and B
template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_DB {
  using DRegisters = void;
  using ARegisters = void;
  using BRegisters = void;
  using CRegisters = void;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d, uint32_t const& desc_a,
      uint32_t const& desc_b, uint32_t const& desc_c,
      uint64_t* abar_d, uint64_t* abar_b
  ) {
#if defined(__SYCL_DEVICE_ONLY__)
    if ( cute::elect_one_sync() ) {
        asm volatile (
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+_am<a_major>+_bk<b_major>+".dtm.btm %0, %1, %2, %3, %4, [%5], [%6];\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(abar_d), "r"(abar_b));
    }
#endif
  }
};

// Async-MMA barriers track A and B
template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_AB {
  using DRegisters = void;
  using ARegisters = void;
  using BRegisters = void;
  using CRegisters = void;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d, uint32_t const& desc_a,
      uint32_t const& desc_b, uint32_t const& desc_c,
      uint64_t* abar_a, uint64_t* abar_b
  ) {
#if defined(__SYCL_DEVICE_ONLY__)
    if ( cute::elect_one_sync() ) {
        asm volatile (
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+_am<a_major>+_bk<b_major>+".atm.btm %0, %1, %2, %3, %4, [%5], [%6];\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(abar_a), "r"(abar_b));
    }
#endif
  }
};



// 32-bit accum, barriers track d, a, b
template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_DAB {
  using DRegisters = void;
  using ARegisters = void;
  using BRegisters = void;
  using CRegisters = void;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d, uint32_t const& desc_a,
      uint32_t const& desc_b, uint32_t const& desc_c,
      uint64_t* abar_d, uint64_t* abar_a, uint64_t* abar_b
  ) {
#if defined(__SYCL_DEVICE_ONLY__)
    if ( cute::elect_one_sync() ) {
        asm volatile (
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+_am<a_major>+_bk<b_major>+".dtm.atm.btm %0, %1, %2, %3, %4, [%5], [%6], [%7];\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(abar_d), "r"(abar_a), "r"(abar_b));
    }
#endif
  }
};

// 32-bit accum, barriers track a, b, cluster version
template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_AB_CLUSTER {
  using DRegisters = void;
  using ARegisters = void;
  using BRegisters = void;
  using CRegisters = void;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d, uint32_t const& desc_a,
      uint32_t const& desc_b, uint32_t const& desc_c,
      uint64_t* abar_a, uint64_t* abar_b,
      uint32_t a_mask, uint32_t b_mask
  ) {
#if defined(__SYCL_DEVICE_ONLY__)
    if ( cute::elect_one_sync() ) {
        asm volatile (
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+_am<a_major>+_bk<b_major>+".atmm.btmm %0, %1, %2, %3, %4, [%5], %7, [%6], %8;\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(abar_a), "r"(abar_b), "r"(a_mask), "r"(b_mask));
    }
#endif
  }
};

// 32-bit accum, barriers track d, a, b, cluster version
template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_DAB_CLUSTER {
  using DRegisters = void;
  using ARegisters = void;
  using BRegisters = void;
  using CRegisters = void;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d, uint32_t const& desc_a,
      uint32_t const& desc_b, uint32_t const& desc_c,
      uint64_t* abar_d, uint64_t* abar_a, uint64_t* abar_b,
      uint32_t a_mask, uint32_t b_mask
  ) {
#if defined(__SYCL_DEVICE_ONLY__)
    if ( cute::elect_one_sync() ) {
        asm volatile (
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+_am<a_major>+_bk<b_major>+".dtm.atmm.btmm %0, %1, %2, %3, %4, [%5], [%6], %8, [%7], %9;\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(abar_d), "r"(abar_a), "r"(abar_b), "r"(a_mask), "r"(b_mask));
    }
#endif
  }
};

} // namespace cute
