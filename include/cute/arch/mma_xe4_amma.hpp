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

// Controls which MX scale qualifiers appear in the block-scaled instruction.
// Both operands are MX (fp4/fp8), one operand is MX, or the other is MX.
//   Both  — ".ascale.bscale"  Both A and B use MX scaling
//   AOnly — ".ascale"         Only A uses MX scaling (B is bf16/fp16)
//   BOnly — ".bscale"         Only B uses MX scaling (A is bf16/fp16)
enum class BlockScaleMode { Both, AOnly, BOnly };
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

// 32-bit accum, barriers track d, b, cluster version
template <class d_type, class a_type, class b_type, class c_type,
  int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_DB_CLUSTER {
  using DRegisters = void;
  using ARegisters = void;
  using BRegisters = void;
  using CRegisters = void;

  CUTE_HOST_DEVICE static void fma(
    MMAControl const& ctrl,
    uint32_t const& desc_d, uint32_t const& desc_a,
    uint32_t const& desc_b, uint32_t const& desc_c,
    uint64_t* abar_d, uint64_t* abar_b,
    uint32_t b_mask
  ) {
#if defined(__SYCL_DEVICE_ONLY__)
    if (cute::elect_one_sync()) {
      asm volatile (
        ("async_gmma.m" + _s<M>+"n" + _s<N>+"k" + _s<K>+"." + _t<d_type>+"_" + _t<a_type>+"_" + _t<b_type>+"_" + _t<c_type>+_am<a_major>+_bk<b_major>+".dtm.btmm %0, %1, %2, %3, %4, [%5], [%6], %7;\n")
        ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
        "r"(abar_d), "r"(abar_b), "r"(b_mask));
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
// Block-Scaled AMMA instructions
//
// Block-scaled AMMA with 8 struct variants (4×2: tracking × scaling modes).
// Barrier tracking: none, _D, _AB, _DAB.
// Scaling modes: Both (A&B are MX), AOnly (A is MX), BOnly (B is MX).
// Mode=Both has explicit specializations to emit single asm instructions.
// ===============================================================================

// Block-scaled AMMA, no barrier tracking
// Primary template: AOnly / BOnly modes
template <class d_type, class a_type, class b_type, class c_type,
          class sf_a_type, class sf_b_type,
          int M, int N, int K, int VSA, int VSB,
          AMMA::Major a_major, AMMA::Major b_major,
          AMMA::BlockScaleMode Mode = AMMA::BlockScaleMode::Both>
struct XE4_AMMA_BlockScaled {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];
  static constexpr int SFVecSizeA = VSA;
  static constexpr int SFVecSizeB = VSB;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if (cute::elect_one_sync()) {
      if constexpr (Mode == AMMA::BlockScaleMode::AOnly) {
        asm volatile(
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale"+_am<a_major>+_bk<b_major>+" %0, %1, %2, %3, %4, %5;\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c), "r"(desc_sfa));
      } else {
        asm volatile(
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".bscale"+_am<a_major>+_bk<b_major>+" %0, %1, %2, %3, %4, %5;\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c), "r"(desc_sfb));
      }
    }
#endif
  }
};

// Mode=Both specialization: both A and B use MX scaling
template <class d_type, class a_type, class b_type, class c_type,
          class sf_a_type, class sf_b_type,
          int M, int N, int K, int VSA, int VSB,
          AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_BlockScaled<d_type, a_type, b_type, c_type,
          sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major,
          AMMA::BlockScaleMode::Both> {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];
  static constexpr int SFVecSizeA = VSA;
  static constexpr int SFVecSizeB = VSB;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if (cute::elect_one_sync()) {
      asm volatile(
        ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale.bscale"+_am<a_major>+_bk<b_major>+" %0, %1, %2, %3, %4, %5, %6;\n")
        ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c), "r"(desc_sfa), "r"(desc_sfb));
    }
#endif
  }
};

// Block-scaled AMMA, D-barrier tracking
template <class d_type, class a_type, class b_type, class c_type,
          class sf_a_type, class sf_b_type,
          int M, int N, int K, int VSA, int VSB,
          AMMA::Major a_major, AMMA::Major b_major,
          AMMA::BlockScaleMode Mode = AMMA::BlockScaleMode::Both>
struct XE4_AMMA_BlockScaled_D {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];
  static constexpr int SFVecSizeA = VSA;
  static constexpr int SFVecSizeB = VSB;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb,
      uint64_t* abar_d)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if (cute::elect_one_sync()) {
      if constexpr (Mode == AMMA::BlockScaleMode::AOnly) {
        asm volatile(
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale"+_am<a_major>+_bk<b_major>+".dtm %0, %1, %2, %3, %4, %5, [%6];\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c), "r"(desc_sfa), "r"(abar_d));
      } else {
        asm volatile(
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".bscale"+_am<a_major>+_bk<b_major>+".dtm %0, %1, %2, %3, %4, %5, [%6];\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c), "r"(desc_sfb), "r"(abar_d));
      }
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb; (void)abar_d;
#endif
  }
};

// Mode=Both specialization: both A and B use MX scaling
template <class d_type, class a_type, class b_type, class c_type,
          class sf_a_type, class sf_b_type,
          int M, int N, int K, int VSA, int VSB,
          AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_BlockScaled_D<d_type, a_type, b_type, c_type,
          sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major,
          AMMA::BlockScaleMode::Both> {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];
  static constexpr int SFVecSizeA = VSA;
  static constexpr int SFVecSizeB = VSB;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb,
      uint64_t* abar_d)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if (cute::elect_one_sync()) {
      asm volatile(
        ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale.bscale"+_am<a_major>+_bk<b_major>+".dtm %0, %1, %2, %3, %4, %5, %6, [%7];\n")
        ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c), "r"(desc_sfa), "r"(desc_sfb), "r"(abar_d));
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb; (void)abar_d;
#endif
  }
};

// Block-scaled AMMA, A+B-barrier tracking
template <class d_type, class a_type, class b_type, class c_type,
          class sf_a_type, class sf_b_type,
          int M, int N, int K, int VSA, int VSB,
          AMMA::Major a_major, AMMA::Major b_major,
          AMMA::BlockScaleMode Mode = AMMA::BlockScaleMode::Both>
struct XE4_AMMA_BlockScaled_AB {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];
  static constexpr int SFVecSizeA = VSA;
  static constexpr int SFVecSizeB = VSB;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb,
      uint64_t* abar_a, uint64_t* abar_b)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if (cute::elect_one_sync()) {
      if constexpr (Mode == AMMA::BlockScaleMode::AOnly) {
        asm volatile(
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale"+_am<a_major>+_bk<b_major>+".atm.btm %0, %1, %2, %3, %4, %5, [%6], [%7];\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c), "r"(desc_sfa), "r"(abar_a), "r"(abar_b));
      } else {
        asm volatile(
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".bscale"+_am<a_major>+_bk<b_major>+".atm.btm %0, %1, %2, %3, %4, %5, [%6], [%7];\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c), "r"(desc_sfb), "r"(abar_a), "r"(abar_b));
      }
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb; (void)abar_a; (void)abar_b;
#endif
  }
};

// Mode=Both specialization: both A and B use MX scaling
template <class d_type, class a_type, class b_type, class c_type,
          class sf_a_type, class sf_b_type,
          int M, int N, int K, int VSA, int VSB,
          AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_BlockScaled_AB<d_type, a_type, b_type, c_type,
          sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major,
          AMMA::BlockScaleMode::Both> {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];
  static constexpr int SFVecSizeA = VSA;
  static constexpr int SFVecSizeB = VSB;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb,
      uint64_t* abar_a, uint64_t* abar_b)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if (cute::elect_one_sync()) {
      asm volatile(
        ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale.bscale"+_am<a_major>+_bk<b_major>+".atm.btm %0, %1, %2, %3, %4, %5, %6, [%7], [%8];\n")
        ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c), "r"(desc_sfa), "r"(desc_sfb), "r"(abar_a), "r"(abar_b));
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb; (void)abar_a; (void)abar_b;
#endif
  }
};

// Block-scaled AMMA, D+A+B-barrier tracking
template <class d_type, class a_type, class b_type, class c_type,
          class sf_a_type, class sf_b_type,
          int M, int N, int K, int VSA, int VSB,
          AMMA::Major a_major, AMMA::Major b_major,
          AMMA::BlockScaleMode Mode = AMMA::BlockScaleMode::Both>
struct XE4_AMMA_BlockScaled_DAB {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];
  static constexpr int SFVecSizeA = VSA;
  static constexpr int SFVecSizeB = VSB;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb,
      uint64_t* abar_d, uint64_t* abar_a, uint64_t* abar_b)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if (cute::elect_one_sync()) {
      if constexpr (Mode == AMMA::BlockScaleMode::AOnly) {
        asm volatile(
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale"+_am<a_major>+_bk<b_major>+".dtm.atm.btm %0, %1, %2, %3, %4, %5, [%6], [%7], [%8];\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c), "r"(desc_sfa), "r"(abar_d), "r"(abar_a), "r"(abar_b));
      } else {
        asm volatile(
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".bscale"+_am<a_major>+_bk<b_major>+".dtm.atm.btm %0, %1, %2, %3, %4, %5, [%6], [%7], [%8];\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c), "r"(desc_sfb), "r"(abar_d), "r"(abar_a), "r"(abar_b));
      }
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb; (void)abar_d; (void)abar_a; (void)abar_b;
#endif
  }
};

// Mode=Both specialization: both A and B use MX scaling
template <class d_type, class a_type, class b_type, class c_type,
          class sf_a_type, class sf_b_type,
          int M, int N, int K, int VSA, int VSB,
          AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_BlockScaled_DAB<d_type, a_type, b_type, c_type,
          sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major,
          AMMA::BlockScaleMode::Both> {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];
  static constexpr int SFVecSizeA = VSA;
  static constexpr int SFVecSizeB = VSB;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb,
      uint64_t* abar_d, uint64_t* abar_a, uint64_t* abar_b)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if (cute::elect_one_sync()) {
      asm volatile(
        ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale.bscale"+_am<a_major>+_bk<b_major>+".dtm.atm.btm %0, %1, %2, %3, %4, %5, %6, [%7], [%8], [%9];\n")
        ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c), "r"(desc_sfa), "r"(desc_sfb), "r"(abar_d), "r"(abar_a), "r"(abar_b));
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb; (void)abar_d; (void)abar_a; (void)abar_b;
#endif
  }
};

// Block-scaled AMMA, A+B-barrier tracking, cluster version (multicast masks)
// Primary template: AOnly / BOnly modes
template <class d_type, class a_type, class b_type, class c_type,
          class sf_a_type, class sf_b_type,
          int M, int N, int K, int VSA, int VSB,
          AMMA::Major a_major, AMMA::Major b_major,
          AMMA::BlockScaleMode Mode = AMMA::BlockScaleMode::Both>
struct XE4_AMMA_BlockScaled_AB_CLUSTER {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];
  static constexpr int SFVecSizeA = VSA;
  static constexpr int SFVecSizeB = VSB;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb,
      uint64_t* abar_a, uint64_t* abar_b,
      uint32_t a_mask, uint32_t b_mask)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if (cute::elect_one_sync()) {
      if constexpr (Mode == AMMA::BlockScaleMode::AOnly) {
        asm volatile(
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale"+_am<a_major>+_bk<b_major>+".atmm.btmm %0, %1, %2, %3, %4, %5, [%6], %8, [%7], %9;\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(desc_sfa), "r"(abar_a), "r"(abar_b), "r"(a_mask), "r"(b_mask));
      } else {
        asm volatile(
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".bscale"+_am<a_major>+_bk<b_major>+".atmm.btmm %0, %1, %2, %3, %4, %5, [%6], %8, [%7], %9;\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(desc_sfb), "r"(abar_a), "r"(abar_b), "r"(a_mask), "r"(b_mask));
      }
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb; (void)abar_a; (void)abar_b; (void)a_mask; (void)b_mask;
#endif
  }
};

// Mode=Both specialization: both A and B use MX scaling
template <class d_type, class a_type, class b_type, class c_type,
          class sf_a_type, class sf_b_type,
          int M, int N, int K, int VSA, int VSB,
          AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_BlockScaled_AB_CLUSTER<d_type, a_type, b_type, c_type,
          sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major,
          AMMA::BlockScaleMode::Both> {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];
  static constexpr int SFVecSizeA = VSA;
  static constexpr int SFVecSizeB = VSB;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb,
      uint64_t* abar_a, uint64_t* abar_b,
      uint32_t a_mask, uint32_t b_mask)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if (cute::elect_one_sync()) {
      asm volatile(
        ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale.bscale"+_am<a_major>+_bk<b_major>+".atmm.btmm %0, %1, %2, %3, %4, %5, %6, [%7], %9, [%8], %10;\n")
        ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
        "r"(desc_sfa), "r"(desc_sfb), "r"(abar_a), "r"(abar_b), "r"(a_mask), "r"(b_mask));
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb; (void)abar_a; (void)abar_b; (void)a_mask; (void)b_mask;
#endif
  }
};

// Block-scaled AMMA, D+A+B-barrier tracking, cluster version (multicast masks)
// Primary template: AOnly / BOnly modes
template <class d_type, class a_type, class b_type, class c_type,
          class sf_a_type, class sf_b_type,
          int M, int N, int K, int VSA, int VSB,
          AMMA::Major a_major, AMMA::Major b_major,
          AMMA::BlockScaleMode Mode = AMMA::BlockScaleMode::Both>
struct XE4_AMMA_BlockScaled_DAB_CLUSTER {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];
  static constexpr int SFVecSizeA = VSA;
  static constexpr int SFVecSizeB = VSB;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb,
      uint64_t* abar_d, uint64_t* abar_a, uint64_t* abar_b,
      uint32_t a_mask, uint32_t b_mask)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if (cute::elect_one_sync()) {
      if constexpr (Mode == AMMA::BlockScaleMode::AOnly) {
        asm volatile(
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale"+_am<a_major>+_bk<b_major>+".dtm.atmm.btmm %0, %1, %2, %3, %4, %5, [%6], [%7], %9, [%8], %10;\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(desc_sfa), "r"(abar_d), "r"(abar_a), "r"(abar_b), "r"(a_mask), "r"(b_mask));
      } else {
        asm volatile(
          ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".bscale"+_am<a_major>+_bk<b_major>+".dtm.atmm.btmm %0, %1, %2, %3, %4, %5, [%6], [%7], %9, [%8], %10;\n")
          ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
          "r"(desc_sfb), "r"(abar_d), "r"(abar_a), "r"(abar_b), "r"(a_mask), "r"(b_mask));
      }
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb; (void)abar_d; (void)abar_a; (void)abar_b; (void)a_mask; (void)b_mask;
#endif
  }
};

// Mode=Both specialization: both A and B use MX scaling
template <class d_type, class a_type, class b_type, class c_type,
          class sf_a_type, class sf_b_type,
          int M, int N, int K, int VSA, int VSB,
          AMMA::Major a_major, AMMA::Major b_major>
struct XE4_AMMA_BlockScaled_DAB_CLUSTER<d_type, a_type, b_type, c_type,
          sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major,
          AMMA::BlockScaleMode::Both> {
  using DRegisters   = void;
  using ARegisters   = void;
  using BRegisters   = void;
  using CRegisters   = void;
  using SFARegisters = uint32_t[1];
  using SFBRegisters = uint32_t[1];
  static constexpr int SFVecSizeA = VSA;
  static constexpr int SFVecSizeB = VSB;

  CUTE_HOST_DEVICE static void fma(
      MMAControl const& ctrl,
      uint32_t const& desc_d,   uint32_t const& desc_a,
      uint32_t const& desc_b,   uint32_t const& desc_c,
      uint32_t const& desc_sfa, uint32_t const& desc_sfb,
      uint64_t* abar_d, uint64_t* abar_a, uint64_t* abar_b,
      uint32_t a_mask, uint32_t b_mask)
  {
#if defined(__SYCL_DEVICE_ONLY__)
    if (cute::elect_one_sync()) {
      asm volatile(
        ("async_gmma.m"+_s<M>+"n"+_s<N>+"k"+_s<K>+"."+_t<d_type>+"_"+_t<a_type>+"_"+_t<b_type>+"_"+_t<c_type>+".ascale.bscale"+_am<a_major>+_bk<b_major>+".dtm.atmm.btmm %0, %1, %2, %3, %4, %5, %6, [%7], [%8], %10, [%9], %11;\n")
        ::"r"(ctrl), "r"(desc_d), "r"(desc_a), "r"(desc_b), "r"(desc_c),
        "r"(desc_sfa), "r"(desc_sfb), "r"(abar_d), "r"(abar_a), "r"(abar_b), "r"(a_mask), "r"(b_mask));
    }
#else
    (void)ctrl; (void)desc_d; (void)desc_a; (void)desc_b; (void)desc_c;
    (void)desc_sfa; (void)desc_sfb; (void)abar_d; (void)abar_a; (void)abar_b; (void)a_mask; (void)b_mask;
#endif
  }
};

} // namespace cute