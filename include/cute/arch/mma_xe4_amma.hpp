#pragma once

#include "xe4_inline_pisa.hpp"
#include <cute/arch/mma_xe4_desc.hpp>
#include "cute/arch/cluster_xe4.hpp"
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

// Async-MMA barriers track D only
template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_D {
  using DRegisters = void;
  using ARegisters = void;
  using BRegisters = void;
  using CRegisters = void;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d, uint32_t const& desc_a,
      uint32_t const& desc_b, uint32_t const& desc_c,
      uint64_t* abar_d
  ) {
#if defined(__SYCL_DEVICE_ONLY__)
    if ( cute::elect_one_sync() ) {
        asm volatile (
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+_am<a_major>+_bk<b_major>+".dtm %0, %1, %2, %3, %4, [%5];\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(abar_d));
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

// ===============================================================================
// Block-Scaled AMMA Operations for FP4
//
// These variants add `.ascale.bscale` to the async_gmma instruction and accept
// scale factor (SF) MatrixDescriptors as additional operands.
//
// MMAControl bits encode the block scale configuration:
//   A_BlockScaleType bits 22:20
//   B_BlockScaleType bits 26:24
// (set by the caller before invoking fma)
//
// SF operands are Type-3 MatrixDescriptors pointing to SLM-resident SF data.
// ===============================================================================

// Block-scaled AMMA, barrier tracking: D only
template <class d_type, class a_type, class b_type, class c_type, class sf_type,
         int M, int N, int K, int VS, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_FP4FP8_D {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  static constexpr int SFVectorSize = VS;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb,
      uint64_t* abar_d)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if ( cute::elect_one_sync() ) {
        asm volatile (
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale.bscale"+_am<a_major>+_bk<b_major>+".dtm %0, %1, %2, %3, %4, %5, %6, [%7];\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(desc_sfa), "r"(desc_sfb), "r"(abar_d));
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb; (void)abar_d;
#endif
  }
};

// Block-scaled AMMA, barrier tracking: A and B
template <class d_type, class a_type, class b_type, class c_type, class sf_type,
         int M, int N, int K, int VS, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_FP4FP8_AB {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  static constexpr int SFVectorSize = VS;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb,
      uint64_t* abar_a, uint64_t* abar_b)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if ( cute::elect_one_sync() ) {
        asm volatile (
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale.bscale"+_am<a_major>+_bk<b_major>+".atm.btm %0, %1, %2, %3, %4, %5, %6, [%7], [%8];\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(desc_sfa), "r"(desc_sfb), "r"(abar_a), "r"(abar_b));
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb; (void)abar_a; (void)abar_b;
#endif
  }
};

// Block-scaled AMMA, barrier tracking: D, A, and B
template <class d_type, class a_type, class b_type, class c_type, class sf_type,
         int M, int N, int K, int VS, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_FP4FP8_DAB {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  static constexpr int SFVectorSize = VS;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb,
      uint64_t* abar_d, uint64_t* abar_a, uint64_t* abar_b)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if ( cute::elect_one_sync() ) {
        asm volatile (
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale.bscale"+_am<a_major>+_bk<b_major>+".dtm.atm.btm %0, %1, %2, %3, %4, %5, %6, [%7], [%8], [%9];\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(desc_sfa), "r"(desc_sfb), "r"(abar_d), "r"(abar_a), "r"(abar_b));
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb; (void)abar_d; (void)abar_a; (void)abar_b;
#endif
  }
};

// Block-scaled AMMA, no barrier tracking
template <class d_type, class a_type, class b_type, class c_type, class sf_type,
         int M, int N, int K, int VS, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_FP4FP8 {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];

  static constexpr int SFVectorSize = VS;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if ( cute::elect_one_sync() ) {
        asm volatile (
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale.bscale"+_am<a_major>+_bk<b_major>+" %0, %1, %2, %3, %4, %5, %6;\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(desc_sfa), "r"(desc_sfb));
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb;
#endif
  }
};

} // namespace cute