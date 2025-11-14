#!/usr/bin/env python3

# Usage:
#    python gen_mma.py --mma_type mma --shape 128x128x128 256x512x128 --dtype f32_f32_bf16_bf16 bf16_f32_bf16_bf16
#    python gen_mma.py --mma_type mma scale_mma --shape 256x512x128 --dtype f32_f32_bf16_bf16 bf16_f32_bf16_bf16


import argparse
import itertools
import subprocess
import os
import re


def SubstituteTemplate(template, values):
    text = template
    for key, val in values.items():
        placeholder = "${" + key + "}"
        text = text.replace(placeholder, val)
    return text


def as_pisa_type(dtype):
    converted_types = {
        "mxint8": "s8",
        "fp4ue8m0k32": "e2m1",
        "fp4ue8m0k16": "e2m1",
        "fp4ue5m3k32": "e2m1",
        "fp4ue5m3k16": "e2m1",
        "fp4ue4m3k16": "e2m1",
    }

    if dtype in converted_types:
        return converted_types[dtype]
    else:
        return dtype


def as_sycl_type(dtype):
    converted_types = {
        "f32": "float",
        "f16": "fp16",
        "s32": "int32_t",
        "s8": "int8_t",
        "e2m1": "fp4_e2m1",
        "e3m2": "fp6_e3m2",
        "s4": "int4_t",
        "fp4ue8m0k32": "fp4_ue8m0k32",
        "fp4ue8m0k16": "fp4_ue8m0k16",
        "fp4ue5m3k32": "fp4_ue5m3k32",
        "fp4ue5m3k16": "fp4_ue5m3k16",
        "fp4ue4m3k16": "fp4_ue4m3k16",
    }

    unchanged_types = ["bf16", "bf8", "hf8", "mxint8"]

    if dtype in converted_types:
        return converted_types[dtype]
    elif dtype in unchanged_types:
        return dtype
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def repr_layout(layout, is_a):
    return "" if layout == "row_major" else f".am" if is_a else f".bk"


class ProfileRegular:
    scenario = "mma"
    templates = {
        "head": """
template<typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K, mem_layout layout_a, mem_layout layout_b>
struct AsyncMMA {
  static_assert(false, "Could not find a async_gmma specialization.");
};
""",
        "body": """
template<>
struct AsyncMMA<${TD}, ${TC}, ${TA}, ${TB}, ${M}, ${N}, ${K}, mem_layout::${LayoutA}, mem_layout::${LayoutB}> {
  // non-cluster version
  template<typename mat_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, ctrl_t ctrl, abar_t abar_d) {
    INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.dtm %0, %1, %2, %3, %4, [%5];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(abar_d));
  }
  template<typename mat_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, ctrl_t ctrl, abar_t abar_a, abar_t abar_b) {
    INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.atm.btm %0, %1, %2, %3, %4, [%5], [%6];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(abar_a), "r"(abar_b));
  }
  template<typename mat_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, abar_t abar_b) {
    INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.dtm.atm.btm %0, %1, %2, %3, %4, [%5], [%6], [%7];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(abar_d), "r"(abar_a), "r"(abar_b));
  }

  // cluster version
  template<typename mat_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, ctrl_t ctrl, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
    INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.atmm.btmm %0, %1, %2, %3, %4, [%5], %6, [%7], %8;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
  }
  template<typename mat_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
    INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.dtm.atmm.btmm %0, %1, %2, %3, %4, [%5], [%6], %7, [%8], %9;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(abar_d), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
  }
};""",
        "foot": """
template<typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
  mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, typename mat_desc_t = uint32_t, typename abar_t = uint64_t*, typename ctrl_t = uint64_t>
inline void async_gmma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, ctrl_t ctrl, abar_t abar_d) {
  return AsyncMMA<TD, TC, TA, TB, M, N, K, layout_a, layout_b>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, ctrl, abar_d);
}

template<typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
  mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, typename mat_desc_t = uint32_t, typename abar_t = uint64_t*, typename ctrl_t = uint64_t>
inline void async_gmma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, ctrl_t ctrl, abar_t abar_a, abar_t abar_b) {
  return AsyncMMA<TD, TC, TA, TB, M, N, K, layout_a, layout_b>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, ctrl, abar_a, abar_b);
}

template<typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
  mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, typename mat_desc_t = uint32_t, typename abar_t = uint64_t*, typename ctrl_t = uint64_t>
inline void async_gmma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, abar_t abar_b) {
  return AsyncMMA<TD, TC, TA, TB, M, N, K, layout_a, layout_b>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, ctrl, abar_d, abar_a, abar_b);
}

template<typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
  mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, typename mat_desc_t = uint32_t, typename abar_t = uint64_t*, typename ctrl_t = uint64_t>
inline void async_gmma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, ctrl_t ctrl, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
  return AsyncMMA<TD, TC, TA, TB, M, N, K, layout_a, layout_b>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, ctrl, abar_a, mask_a, abar_b, mask_b);
}

template<typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
  mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, typename mat_desc_t = uint32_t, typename abar_t = uint64_t*, typename ctrl_t = uint64_t>
inline void async_gmma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
  return AsyncMMA<TD, TC, TA, TB, M, N, K, layout_a, layout_b>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, ctrl, abar_d, abar_a, mask_a, abar_b, mask_b);
}
""",
    }


class ProfileScale:
    scenario = "scale_mma"
    templates = {
        "head": """
template<typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K, mem_layout layout_a, mem_layout layout_b, bool a_scaling, bool b_scaling>
struct AsyncMMAScale {
  static_assert(false, "Could not find a scale async_gmma specialization.");
};
""",
        "body": """
template<bool a_scaling, bool b_scaling>
struct AsyncMMAScale<${TD}, ${TC}, ${TA}, ${TB}, ${M}, ${N}, ${K}, mem_layout::${LayoutA}, mem_layout::${LayoutB}, a_scaling, b_scaling> {
  // non-cluster version
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t meta_desc_a, meta_desc_t meta_desc_b, ctrl_t ctrl, abar_t abar_d) {
    if constexpr (a_scaling && b_scaling) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale.bscale${LA}${LB}.dtm %0, %1, %2, %3, %4, %5, %6, [%7];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_a), "r"(meta_desc_b), "r"(abar_d));
    } else if constexpr (a_scaling && (!b_scaling)) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale${LA}${LB}.dtm %0, %1, %2, %3, %4, %5, [%6];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_a), "r"(abar_d));
    } else if constexpr ((!a_scaling) && b_scaling) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.bscale${LA}${LB}.dtm %0, %1, %2, %3, %4, %5, [%6];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_b), "r"(abar_d));
    } else {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.dtm %0, %1, %2, %3, %4, [%5];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(abar_d));
    }
  }
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t meta_desc_a, meta_desc_t meta_desc_b, ctrl_t ctrl, abar_t abar_a, abar_t abar_b) {
    if constexpr (a_scaling && b_scaling) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale.bscale${LA}${LB}.atm.btm %0, %1, %2, %3, %4, %5, %6, [%7], [%8];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_a), "r"(meta_desc_b), "r"(abar_a), "r"(abar_b));
    } else if constexpr (a_scaling && (!b_scaling)) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale${LA}${LB}.atm.btm %0, %1, %2, %3, %4, %5, [%6], [%7];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_a), "r"(abar_a), "r"(abar_b));
    } else if constexpr ((!a_scaling) && b_scaling) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.bscale${LA}${LB}.atm.btm %0, %1, %2, %3, %4, %5, [%6], [%7];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_b), "r"(abar_a), "r"(abar_b));
    } else {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.atm.btm %0, %1, %2, %3, %4, [%5], [%6];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(abar_a), "r"(abar_b));
    }
  }
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t meta_desc_a, meta_desc_t meta_desc_b, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, abar_t abar_b) {
    if constexpr (a_scaling && b_scaling) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale.bscale${LA}${LB}.dtm.atm.btm %0, %1, %2, %3, %4, %5, %6, [%7], [%8], [%9];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_a), "r"(meta_desc_b), "r"(abar_d), "r"(abar_a), "r"(abar_b));
    } else if constexpr (a_scaling && (!b_scaling)) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale${LA}${LB}.dtm.atm.btm %0, %1, %2, %3, %4, %5, [%6], [%7], [%8];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_a), "r"(abar_d), "r"(abar_a), "r"(abar_b));
    } else if constexpr ((!a_scaling) && b_scaling) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.bscale${LA}${LB}.dtm.atm.btm %0, %1, %2, %3, %4, %5, [%6], [%7], [%8];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_b), "r"(abar_d), "r"(abar_a), "r"(abar_b));
    } else {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.dtm.atm.btm %0, %1, %2, %3, %4, [%5], [%6], [%7];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(abar_d), "r"(abar_a), "r"(abar_b));
    }
  }

  // cluster version
  // DxCxAxB; abar a/b with multicast
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t meta_desc_a, meta_desc_t meta_desc_b, ctrl_t ctrl, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
    if constexpr (a_scaling && b_scaling) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale.bscale${LA}${LB}.atmm.btmm %0, %1, %2, %3, %4, %5, %6, [%7], %8, [%9], %10;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_a), "r"(meta_desc_b), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    } else if constexpr (a_scaling && (!b_scaling)) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale${LA}${LB}.atmm.btmm %0, %1, %2, %3, %4, %5, [%6], %7, [%8], %9;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_a), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    } else if constexpr ((!a_scaling) && b_scaling) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.bscale${LA}${LB}.atmm.btmm %0, %1, %2, %3, %4, %5, [%6], %7, [%8], %9;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_b), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    } else {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.atmm.btmm %0, %1, %2, %3, %4, [%5], %6, [%7], %8;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    }
  }
  // DxCxAxB; abar_d, abar_a/b with multicast
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t meta_desc_a, meta_desc_t meta_desc_b, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
    if constexpr (a_scaling && b_scaling) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale.bscale${LA}${LB}.dtm.atmm.btmm %0, %1, %2, %3, %4, %5, %6, [%7], [%8], %9, [%10], %11;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_a), "r"(meta_desc_b), "r"(abar_d), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    } else if constexpr (a_scaling && (!b_scaling)) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale${LA}${LB}.dtm.atmm.btmm %0, %1, %2, %3, %4, %5, [%6], [%7], %8, [%9], %10;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_a), "r"(abar_d), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    } else if constexpr ((!a_scaling) && b_scaling) {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.bscale${LA}${LB}.dtm.atmm.btmm %0, %1, %2, %3, %4, %5, [%6], [%7], %8, [%9], %10;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc_b), "r"(abar_d), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    } else {
      INLINE_PISA("async_gmma.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.dtm.atmm.btmm %0, %1, %2, %3, %4, [%5], [%6], %7, [%8], %9;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(abar_d), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    }
  }
};""",
        "foot": """
template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, bool a_scaling = false, bool b_scaling = false, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename mxfp_meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, mxfp_meta_desc_t meta_desc_a, mxfp_meta_desc_t meta_desc_b, ctrl_t ctrl, abar_t abar_d) {
  return AsyncMMAScale<TD, TC, TA, TB, M, N, K, layout_a, layout_b, a_scaling, b_scaling>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, meta_desc_a, meta_desc_b, ctrl, abar_d);
}

template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, bool a_scaling = false, bool b_scaling = false, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename mxfp_meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, mxfp_meta_desc_t meta_desc_a, mxfp_meta_desc_t meta_desc_b, ctrl_t ctrl, abar_t abar_a, abar_t abar_b) {
  return AsyncMMAScale<TD, TC, TA, TB, M, N, K, layout_a, layout_b, a_scaling, b_scaling>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, meta_desc_a, meta_desc_b, ctrl, abar_a, abar_b);
}

template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, bool a_scaling = false, bool b_scaling = false, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename mxfp_meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, mxfp_meta_desc_t meta_desc_a, mxfp_meta_desc_t meta_desc_b, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, abar_t abar_b) {
  return AsyncMMAScale<TD, TC, TA, TB, M, N, K, layout_a, layout_b, a_scaling, b_scaling>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, meta_desc_a, meta_desc_b, ctrl, abar_d, abar_a, abar_b);
}

// mma scale cluster: DxCxAxB, abar_a/b with multicast
template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, bool a_scaling = false, bool b_scaling = false, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename mxfp_meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, mxfp_meta_desc_t meta_desc_a, mxfp_meta_desc_t meta_desc_b, ctrl_t ctrl, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
  return AsyncMMAScale<TD, TC, TA, TB, M, N, K, layout_a, layout_b, a_scaling, b_scaling>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, meta_desc_a, meta_desc_b, ctrl, abar_a, mask_a, abar_b, mask_b);
}

// mma scale cluster: DxCxAxB, abar_d abar_a/b with multicast
template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, bool a_scaling = false, bool b_scaling = false, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename mxfp_meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, mxfp_meta_desc_t meta_desc_a, mxfp_meta_desc_t meta_desc_b, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
  return AsyncMMAScale<TD, TC, TA, TB, M, N, K, layout_a, layout_b, a_scaling, b_scaling>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, meta_desc_a, meta_desc_b, ctrl, abar_d, abar_a, mask_a, abar_b, mask_b);
}
""",
    }


class ProfileSparsity:
    scenario = "sparsity_mma"
    templates = {
        "head": """
template<typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K, mem_layout layout_a, mem_layout layout_b>
struct AsyncMMASparsity {
  static_assert(false, "Could not find a sparsity async_gmma specialization.");
};
""",
        "body": """
template<>
struct AsyncMMASparsity<${TD}, ${TC}, ${TA}, ${TB}, ${M}, ${N}, ${K}, mem_layout::${LayoutA}, mem_layout::${LayoutB}> {
  // non-cluster version
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t meta_desc, ctrl_t ctrl, abar_t abar_d) {
    INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.dtm %0, %1, %2, %3, %4, %5, [%6];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc), "r"(abar_d));
  }
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t meta_desc, ctrl_t ctrl, abar_t abar_a, abar_t abar_b) {
    INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.atm.btm %0, %1, %2, %3, %4, %5, [%6], [%7];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc), "r"(abar_a), "r"(abar_b));
  }
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t meta_desc, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, abar_t abar_b) {
    INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.dtm.atm.btm %0, %1, %2, %3, %4, %5, [%6], [%7], [%8];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc), "r"(abar_d), "r"(abar_a), "r"(abar_b));
  }

  // cluster version
  // DxCxAxB; abar_a/b with multicast
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t meta_desc, ctrl_t ctrl, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
    INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.atmm.btmm %0, %1, %2, %3, %4, %5, [%6], %7, [%8], %9;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
  }
  // DxCxAxB; abar_d, abar_a/b with multicast
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t meta_desc, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
    INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.dtm.atmm.btmm %0, %1, %2, %3, %4, %5, [%6], [%7], %8, [%9], %10;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(meta_desc), "r"(abar_d), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
  }
};""",
        "foot": """template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename sparsity_meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma_s(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, sparsity_meta_desc_t meta_desc, ctrl_t ctrl, abar_t abar_d) {
  return AsyncMMASparsity<TD, TC, TA, TB, M, N, K, layout_a, layout_b>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, meta_desc, ctrl, abar_d);
}

template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename sparsity_meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma_s(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, sparsity_meta_desc_t meta_desc, ctrl_t ctrl, abar_t abar_a, abar_t abar_b) {
  return AsyncMMASparsity<TD, TC, TA, TB, M, N, K, layout_a, layout_b>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, meta_desc, ctrl, abar_a, abar_b);
}

template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename sparsity_meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma_s(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, sparsity_meta_desc_t meta_desc, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, abar_t abar_b) {
  return AsyncMMASparsity<TD, TC, TA, TB, M, N, K, layout_a, layout_b>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, meta_desc, ctrl, abar_d, abar_a, abar_b);
}

// mma sparsity cluster: DxCxAxB; abar_a/b with multicast
template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename sparsity_meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma_s(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, sparsity_meta_desc_t meta_desc, ctrl_t ctrl, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
  return AsyncMMASparsity<TD, TC, TA, TB, M, N, K, layout_a, layout_b>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, meta_desc, ctrl, abar_a, mask_a, abar_b, mask_b);
}

// mma sparsity cluster: DxCxAxB; abar_d, abar_a/b with multicast
template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename sparsity_meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma_s(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, sparsity_meta_desc_t meta_desc, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
  return AsyncMMASparsity<TD, TC, TA, TB, M, N, K, layout_a, layout_b>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, meta_desc, ctrl, abar_d, abar_a, mask_a, abar_b, mask_b);
}
""",
    }


class ProfileSparsityScale:
    scenario = "sparsity_scale_mma"
    templates = {
        "head": """
template<typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K, mem_layout layout_a, mem_layout layout_b, bool a_scaling, bool b_scaling>
struct AsyncMMASparsityScale {
  static_assert(false, "Could not find a sparsity mxfp async_gmma specialization.");
};
""",
        "body": """
template<bool a_scaling, bool b_scaling>
struct AsyncMMASparsityScale<${TD}, ${TC}, ${TA}, ${TB}, ${M}, ${N}, ${K}, mem_layout::${LayoutA}, mem_layout::${LayoutB}, a_scaling, b_scaling> {
  // non-cluster version
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t spars_desc, meta_desc_t mxfp_a_desc, meta_desc_t mxfp_b_desc, ctrl_t ctrl, abar_t abar_a, abar_t abar_b) {
    if constexpr (a_scaling && b_scaling) {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale.bscale${LA}${LB}.atm.btm %0, %1, %2, %3, %4, %5, %6, %7, [%8], [%9];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(mxfp_a_desc), "r"(mxfp_b_desc), "r"(spars_desc), "r"(abar_a), "r"(abar_b));
    } else if constexpr (a_scaling && (!b_scaling)) {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale${LA}${LB}.atm.btm %0, %1, %2, %3, %4, %5, %6, [%7], [%8];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(mxfp_a_desc), "r"(spars_desc), "r"(abar_a), "r"(abar_b));
    } else if constexpr ((!a_scaling) && b_scaling) {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.bscale${LA}${LB}.atm.btm %0, %1, %2, %3, %4, %5, %6, [%7], [%8];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(mxfp_b_desc), "r"(spars_desc), "r"(abar_a), "r"(abar_b));
    } else {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.atm.btm %0, %1, %2, %3, %4, %5, [%6], [%7];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(spars_desc), "r"(abar_a), "r"(abar_b));
    }
  }
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t spars_desc, meta_desc_t mxfp_a_desc, meta_desc_t mxfp_b_desc, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, abar_t abar_b) {
    if constexpr (a_scaling && b_scaling) {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale.bscale${LA}${LB}.dtm.atm.btm %0, %1, %2, %3, %4, %5, %6, %7, [%8], [%9], [%10];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(mxfp_a_desc), "r"(mxfp_b_desc), "r"(spars_desc), "r"(abar_d), "r"(abar_a), "r"(abar_b));
    } else if constexpr (a_scaling && (!b_scaling)) {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale${LA}${LB}.dtm.atm.btm %0, %1, %2, %3, %4, %5, %6, [%7], [%8], [%9];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(mxfp_a_desc), "r"(spars_desc), "r"(abar_d), "r"(abar_a), "r"(abar_b));
    } else if constexpr ((!a_scaling) && b_scaling) {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.bscale${LA}${LB}.dtm.atm.btm %0, %1, %2, %3, %4, %5, %6, [%7], [%8], [%9];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(mxfp_b_desc), "r"(spars_desc), "r"(abar_d), "r"(abar_a), "r"(abar_b));
    } else {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.dtm.atm.btm %0, %1, %2, %3, %4, %5, [%6], [%7], [%8];" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(spars_desc), "r"(abar_d), "r"(abar_a), "r"(abar_b));
    }
  }

  // cluster version
  // DxCxAxB; abar a/b with multicast
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t spars_desc, meta_desc_t mxfp_a_desc, meta_desc_t mxfp_b_desc, ctrl_t ctrl, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
    if constexpr (a_scaling && b_scaling) {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale.bscale${LA}${LB}.atmm.btmm %0, %1, %2, %3, %4, %5, %6, %7, [%8], %9, [%10], %11;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(mxfp_a_desc), "r"(mxfp_b_desc), "r"(spars_desc), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    } else if constexpr (a_scaling && (!b_scaling)) {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale${LA}${LB}.atmm.btmm %0, %1, %2, %3, %4, %5, %6, [%7], %8, [%9], %10;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(mxfp_a_desc), "r"(spars_desc), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    } else if constexpr ((!a_scaling) && b_scaling) {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.bscale${LA}${LB}.atmm.btmm %0, %1, %2, %3, %4, %5, %6, [%7], %8, [%9], %10;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(mxfp_b_desc), "r"(spars_desc), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    } else {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.atmm.btmm %0, %1, %2, %3, %4, %5, [%6], %7, [%8], %9;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(spars_desc), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    }
  }
  // DxCxAxB; abar_d, abar_a/b with multicast
  template<typename mat_desc_t, typename meta_desc_t, typename abar_t, typename ctrl_t>
  static void fma(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t spars_desc, meta_desc_t mxfp_a_desc, meta_desc_t mxfp_b_desc, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
    if constexpr (a_scaling && b_scaling) {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale.bscale${LA}${LB}.dtm.atmm.btmm %0, %1, %2, %3, %4, %5, %6, %7, [%8], [%9], %10, [%11], %12;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(mxfp_a_desc), "r"(mxfp_b_desc), "r"(spars_desc), "r"(abar_d), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    } else if constexpr (a_scaling && (!b_scaling)) {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.ascale${LA}${LB}.dtm.atmm.btmm %0, %1, %2, %3, %4, %5, %6, [%7], [%8], %9, [%10], %11;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(mxfp_a_desc), "r"(spars_desc), "r"(abar_d), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    } else if constexpr ((!a_scaling) && b_scaling) {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}.bscale${LA}${LB}.dtm.atmm.btmm %0, %1, %2, %3, %4, %5, %6, [%7], [%8], %9, [%10], %11;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(mxfp_b_desc), "r"(spars_desc), "r"(abar_d), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    } else {
      INLINE_PISA("async_gmma.s.m${M}n${N}k${K}.${D}_${A}_${B}_${C}${LA}${LB}.dtm.atmm.btmm %0, %1, %2, %3, %4, %5, [%6], [%7], %8, [%9], %10;" ::"r"(ctrl), "r"(mat_desc_d), "r"(mat_desc_a), "r"(mat_desc_b), "r"(mat_desc_c), "r"(spars_desc), "r"(abar_d), "r"(abar_a), "r"(mask_a), "r"(abar_b), "r"(mask_b));
    }
  }
};""",
        "foot": """
// mxfp sparsity non-cluster: DxCxAxB; abar_a/b with multicast
template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a, mem_layout layout_b, bool a_scaling, bool b_scaling, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma_s(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t spars_meta_desc, meta_desc_t mxfp_a_desc, meta_desc_t mxfp_b_desc, ctrl_t ctrl, abar_t abar_a, abar_t abar_b) {
  return AsyncMMASparsityScale<TD, TC, TA, TB, M, N, K, layout_a, layout_b, a_scaling, b_scaling>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, spars_meta_desc, mxfp_a_desc, mxfp_b_desc, ctrl, abar_a, abar_b);
}

// mxfp sparsity non-cluster: DxCxAxB; abar_d, abar_a/b with multicast
template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a, mem_layout layout_b, bool a_scaling, bool b_scaling, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma_s(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t spars_meta_desc, meta_desc_t mxfp_a_desc, meta_desc_t mxfp_b_desc, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, abar_t abar_b) {
  return AsyncMMASparsityScale<TD, TC, TA, TB, M, N, K, layout_a, layout_b, a_scaling, b_scaling>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, spars_meta_desc, mxfp_a_desc, mxfp_b_desc, ctrl, abar_d, abar_a, abar_b);
}


// mxfp sparsity cluster: DxCxAxB; abar_a/b with multicast
template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a, mem_layout layout_b, bool a_scaling, bool b_scaling, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma_s(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t spars_meta_desc, meta_desc_t mxfp_a_desc, meta_desc_t mxfp_b_desc, ctrl_t ctrl, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
  return AsyncMMASparsityScale<TD, TC, TA, TB, M, N, K, layout_a, layout_b, a_scaling, b_scaling>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, spars_meta_desc, mxfp_a_desc, mxfp_b_desc, ctrl, abar_a, mask_a, abar_b, mask_b);
}

// mxfp sparsity cluster: DxCxAxB; abar_d, abar_a/b with multicast
template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
mem_layout layout_a, mem_layout layout_b, bool a_scaling, bool b_scaling, typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma_s(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t spars_meta_desc, meta_desc_t mxfp_a_desc, meta_desc_t mxfp_b_desc, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
  return AsyncMMASparsityScale<TD, TC, TA, TB, M, N, K, layout_a, layout_b, a_scaling, b_scaling>::fma(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, spars_meta_desc, mxfp_a_desc, mxfp_b_desc, ctrl, abar_d, abar_a, mask_a, abar_b, mask_b);
}
""",
    }


class ProfileDispatch:
    templates = {
        "foot": """
// non-cluster gmma entry; abar_a/abar_b
template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
uint32_t is_sparsity, uint32_t is_scale_a, uint32_t is_scale_b,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major,
typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma_handler(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t spars_meta_desc, meta_desc_t mxfp_a_desc, meta_desc_t mxfp_b_desc, ctrl_t ctrl, abar_t abar_a, abar_t abar_b) {
  constexpr uint32_t is_mxfp = is_scale_a | is_scale_b;
  if constexpr (!is_mxfp && is_sparsity) {
    return async_gmma_s<TD, TC, TA, TB, M, N, K, layout_a, layout_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, spars_meta_desc, ctrl, abar_a, abar_b);
  } else if constexpr (is_mxfp && !is_sparsity) {
    return async_gmma<TD, TC, TA, TB, M, N, K, layout_a, layout_b, is_scale_a, is_scale_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, mxfp_a_desc, mxfp_b_desc, ctrl, abar_a, abar_b);
  } else if constexpr (is_mxfp && is_sparsity) {
    return async_gmma_s<TD, TC, TA, TB, M, N, K, layout_a, layout_b, is_scale_a, is_scale_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, spars_meta_desc, mxfp_a_desc, mxfp_b_desc, ctrl, abar_a, abar_b);
  } else {
    return async_gmma<TD, TC, TA, TB, M, N, K, layout_a, layout_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, ctrl, abar_a, abar_b);
  }
}

// non-cluster gmma entry; abar_d/abar_a/abar_b
template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
uint32_t is_sparsity, uint32_t is_scale_a, uint32_t is_scale_b,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major,
typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma_handler(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t spars_meta_desc, meta_desc_t mxfp_a_desc, meta_desc_t mxfp_b_desc, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, abar_t abar_b) {
  constexpr uint32_t is_mxfp = is_scale_a | is_scale_b;
  if constexpr (!is_mxfp && is_sparsity) {
    return async_gmma_s<TD, TC, TA, TB, M, N, K, layout_a, layout_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, spars_meta_desc, ctrl, abar_d, abar_a, abar_b);
  } else if constexpr (is_mxfp && !is_sparsity) {
    return async_gmma<TD, TC, TA, TB, M, N, K, layout_a, layout_b, is_scale_a, is_scale_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, mxfp_a_desc, mxfp_b_desc, ctrl, abar_d, abar_a, abar_b);
  } else if constexpr (is_mxfp && is_sparsity) {
    return async_gmma_s<TD, TC, TA, TB, M, N, K, layout_a, layout_b, is_scale_a, is_scale_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, spars_meta_desc, mxfp_a_desc, mxfp_b_desc, ctrl, abar_d, abar_a, abar_b);
  } else {
    return async_gmma<TD, TC, TA, TB, M, N, K, layout_a, layout_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, ctrl, abar_d, abar_a, abar_b);
  }
}

// cluster gmma entry; abar_a/abar_b
template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
uint32_t is_sparsity, uint32_t is_scale_a, uint32_t is_scale_b,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major,
typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma_handler(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t spars_meta_desc, meta_desc_t mxfp_a_desc, meta_desc_t mxfp_b_desc, ctrl_t ctrl, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
  constexpr uint32_t is_mxfp = is_scale_a | is_scale_b;
  if constexpr (!is_mxfp && is_sparsity) {
    return async_gmma_s<TD, TC, TA, TB, M, N, K, layout_a, layout_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, spars_meta_desc, ctrl, abar_a, mask_a, abar_b, mask_b);
  } else if constexpr (is_mxfp && !is_sparsity) {
    return async_gmma<TD, TC, TA, TB, M, N, K, layout_a, layout_b, is_scale_a, is_scale_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, mxfp_a_desc, mxfp_b_desc, ctrl, abar_a, mask_a, abar_b, mask_b);
  } else if constexpr (is_mxfp && is_sparsity) {
    return async_gmma_s<TD, TC, TA, TB, M, N, K, layout_a, layout_b, is_scale_a, is_scale_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, spars_meta_desc, mxfp_a_desc, mxfp_b_desc, ctrl, abar_a, mask_a, abar_b, mask_b);
  } else {
    return async_gmma<TD, TC, TA, TB, M, N, K, layout_a, layout_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, ctrl, abar_a, mask_a, abar_b, mask_b);
  }
}

// cluster gmma entry; abar_d/abar_a/abar_b
template <typename TD, typename TC, typename TA, typename TB, uint32_t M, uint32_t N, uint32_t K,
uint32_t is_sparsity, uint32_t is_scale_a, uint32_t is_scale_b,
mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major,
typename abar_t = uint64_t*, typename mat_desc_t = uint32_t, typename meta_desc_t = uint64_t, typename ctrl_t = uint64_t>
inline void async_gmma_handler(mat_desc_t mat_desc_d, mat_desc_t mat_desc_c, mat_desc_t mat_desc_a, mat_desc_t mat_desc_b, meta_desc_t spars_meta_desc, meta_desc_t mxfp_a_desc, meta_desc_t mxfp_b_desc, ctrl_t ctrl, abar_t abar_d, abar_t abar_a, uint32_t mask_a, abar_t abar_b, uint32_t mask_b) {
  constexpr uint32_t is_mxfp = is_scale_a | is_scale_b;
  if constexpr (!is_mxfp && is_sparsity) {
    return async_gmma_s<TD, TC, TA, TB, M, N, K, layout_a, layout_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, spars_meta_desc, ctrl, abar_d, abar_a, mask_a, abar_b, mask_b);
  } else if constexpr (is_mxfp && !is_sparsity) {
    return async_gmma<TD, TC, TA, TB, M, N, K, layout_a, layout_b, is_scale_a, is_scale_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, mxfp_a_desc, mxfp_b_desc, ctrl, abar_d, abar_a, mask_a, abar_b, mask_b);
  } else if constexpr (is_mxfp && is_sparsity) {
    return async_gmma_s<TD, TC, TA, TB, M, N, K, layout_a, layout_b, is_scale_a, is_scale_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, spars_meta_desc, mxfp_a_desc, mxfp_b_desc, ctrl, abar_d, abar_a, mask_a, abar_b, mask_b);
  } else {
    return async_gmma<TD, TC, TA, TB, M, N, K, layout_a, layout_b>(mat_desc_d, mat_desc_c, mat_desc_a, mat_desc_b, ctrl, abar_d, abar_a, mask_a, abar_b, mask_b);
  }
}

template <typename dtype_c, typename dtype_acc, typename dtype_a, typename dtype_b, uint32_t wg_m, uint32_t wg_n,
          uint32_t wg_k, uint32_t mma_m, uint32_t mma_k, mem_layout layout_a, mem_layout layout_b, bool is_sparsity,
          bool is_scale_a, bool is_scale_b, sparsity_repr_t sparse_repr>
constexpr std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                     uint32_t>
checkout_per_loop_stride() {
  auto round_up = [](const auto a, const auto b) { return (a + b - 1) / b * b; };

  constexpr bool is_trans_a = (layout_a == mem_layout::col_major);
  constexpr bool is_trans_b = (layout_b == mem_layout::col_major);
  constexpr uint32_t sparse_ratio = is_sparsity ? get_sparsity_ratio(sparse_repr) : 1;

  constexpr uint32_t rep_k = wg_k / mma_k;
  static_assert(wg_k % mma_k == 0, "wg_k must be times of mma_k");
  static_assert(is_valid_mma_k<dtype_a, dtype_b, mma_k>(), "mma_k should be valid for the given dtype_a and dtype_b");

  constexpr uint32_t rep_m = wg_m / mma_m;
  static_assert(wg_m % mma_m == 0, "wg_m must be times of mma_m");
  static_assert(is_valid_mma_m<mma_m>(), "mma_m should be valid");
  if constexpr (rep_m > 1) {
    static_assert(mma_m % TYPE3_CM_ALIGN_X == 0, "mma_m should be aligned to TYPE3_CM_ALIGN_X for metaA load consideration");
  }

  constexpr uint32_t mx_scale_elem_num_a = 32;
  constexpr uint32_t mx_scale_elem_num_b = 32 / sparse_ratio;

  constexpr uint32_t mma_k_a = mma_k * sparse_ratio;
  constexpr uint32_t mma_k_b = mma_k;
  constexpr uint32_t wg_k_a = wg_k * sparse_ratio;
  constexpr uint32_t wg_k_b = wg_k;

  constexpr uint32_t slm_stride_a_m = (is_trans_a ? TYPE1_CM_ELEM_Y * mma_m * sizeof_bits<dtype_a>() / BITS_PER_BYTE
                                                  : wg_k_a * mma_m * sizeof_bits<dtype_a>() / BITS_PER_BYTE);
  constexpr uint32_t slm_stride_a_k = (is_trans_a ? mma_k_a * wg_m * sizeof_bits<dtype_a>() / BITS_PER_BYTE
                                                  : TYPE1_CM_ELEM_Y * mma_k_a * sizeof_bits<dtype_a>() / BITS_PER_BYTE);

  constexpr uint32_t slm_stride_b_k = (is_trans_b ? TYPE1_CM_ELEM_Y * mma_k_b * sizeof_bits<dtype_b>() / BITS_PER_BYTE
                                                  : mma_k_b * wg_n * sizeof_bits<dtype_b>() / BITS_PER_BYTE);
  constexpr uint32_t slm_stride_acc_m = mma_m * wg_n * sizeof_bits<dtype_acc>() / BITS_PER_BYTE;
  constexpr uint32_t slm_stride_c_m = mma_m * wg_n * sizeof_bits<dtype_c>() / BITS_PER_BYTE;

  if constexpr (rep_k > 1) {
    static_assert(slm_stride_a_k >= SLM_BASE_BYTES_ALIGN && slm_stride_a_k % SLM_BASE_BYTES_ALIGN == 0,
                  "slm_stride_a_k must be aligned to SLM_BASE_BYTES_ALIGN");
    static_assert(slm_stride_a_m >= SLM_BASE_BYTES_ALIGN && slm_stride_a_m % SLM_BASE_BYTES_ALIGN == 0,
                  "slm_stride_a_m must be aligned to SLM_BASE_BYTES_ALIGN");
    static_assert(slm_stride_b_k >= SLM_BASE_BYTES_ALIGN && slm_stride_b_k % SLM_BASE_BYTES_ALIGN == 0,
                  "slm_stride_b_k must be aligned to SLM_BASE_BYTES_ALIGN");
    static_assert(slm_stride_acc_m >= SLM_BASE_BYTES_ALIGN && slm_stride_acc_m % SLM_BASE_BYTES_ALIGN == 0,
                  "slm_stride_acc_m must be aligned to SLM_BASE_BYTES_ALIGN");
    static_assert(slm_stride_c_m >= SLM_BASE_BYTES_ALIGN && slm_stride_c_m % SLM_BASE_BYTES_ALIGN == 0,
                  "slm_stride_c_m must be aligned to SLM_BASE_BYTES_ALIGN");
  }

  constexpr uint32_t padding_sparse_k = round_up(mma_k * sparse_ratio / BITS_PER_BYTE, TYPE2_CM_BYTE_Y);

  constexpr uint32_t padding_meta_k_a = round_up(mma_k * sparse_ratio / mx_scale_elem_num_a, TYPE3_CM_SIZE_Y);
  constexpr uint32_t padding_meta_k_b = round_up(mma_k / mx_scale_elem_num_b, TYPE3_CM_SIZE_Y);
  constexpr uint32_t padding_meta_n = round_up(wg_n, TYPE3_CM_ALIGN_X);
  constexpr uint32_t padding_meta_m = round_up(wg_m, TYPE3_CM_ALIGN_X);

  constexpr uint32_t slm_stride_meta_a_m = is_scale_a ? TYPE3_CM_SIZE_Y * mma_m : 0; // sizeof(meta_type) always 1B
  constexpr uint32_t slm_stride_meta_a_k = is_scale_a ? padding_meta_k_a * padding_meta_m : 0;

  constexpr uint32_t slm_stride_meta_b =
      is_scale_b ? padding_meta_k_b * padding_meta_n : 0; // sizeof(meta_type) always 1B
  constexpr uint32_t slm_stride_sparse = is_sparsity ? padding_sparse_k * wg_n : 0; // sizeof(sparse_type) always 1B

  constexpr uint32_t rounded_slm_stride_meta_a_m = round_up(slm_stride_meta_a_m, SLM_BASE_BYTES_ALIGN);
  constexpr uint32_t rounded_slm_stride_meta_a_k = round_up(slm_stride_meta_a_k, SLM_BASE_BYTES_ALIGN);
  constexpr uint32_t rounded_slm_stride_meta_b = round_up(slm_stride_meta_b, SLM_BASE_BYTES_ALIGN);
  constexpr uint32_t rounded_slm_stride_sparse = round_up(slm_stride_sparse, SLM_BASE_BYTES_ALIGN);

  auto shifted_slm_stride_c_m = slm_stride_c_m >> SLM_BASE_ADDR_OFFSET;
  auto shifted_slm_stride_acc_m = slm_stride_acc_m >> SLM_BASE_ADDR_OFFSET;
  auto shifted_slm_stride_a_m = slm_stride_a_m >> SLM_BASE_ADDR_OFFSET;
  auto shifted_slm_stride_a_k = slm_stride_a_k >> SLM_BASE_ADDR_OFFSET;
  auto shifted_slm_stride_b_k = slm_stride_b_k >> SLM_BASE_ADDR_OFFSET;
  auto shifted_slm_stride_meta_a_m = rounded_slm_stride_meta_a_m >> SLM_BASE_ADDR_OFFSET;
  auto shifted_slm_stride_meta_a_k = rounded_slm_stride_meta_a_k >> SLM_BASE_ADDR_OFFSET;
  auto shifted_slm_stride_meta_b = rounded_slm_stride_meta_b >> SLM_BASE_ADDR_OFFSET;
  auto shifted_slm_stride_sparse = rounded_slm_stride_sparse >> SLM_BASE_ADDR_OFFSET;

  return std::make_tuple(rep_m, rep_k, shifted_slm_stride_c_m, shifted_slm_stride_acc_m, shifted_slm_stride_a_m,
                         shifted_slm_stride_a_k, shifted_slm_stride_b_k, shifted_slm_stride_meta_a_m,
                         shifted_slm_stride_meta_a_k, shifted_slm_stride_meta_b, shifted_slm_stride_sparse);
}

// multi-mk non-cluster gmma entry; abar_a/abar_b
template <typename dtype_c, typename dtype_acc, typename dtype_a, typename dtype_b, uint32_t wg_m, uint32_t wg_n,
          uint32_t wg_k, uint32_t mma_m, uint32_t mma_k, bool is_sparsity = false, bool is_scale_a = false,
          bool is_scale_b = false, mem_layout layout_a = mem_layout::row_major,
          mem_layout layout_b = mem_layout::row_major, sparsity_repr_t sparse_repr = sparsity_repr_t::A4xB2,
          typename abar_ptr_t = uint64_t *, typename matrix_desc_t = uint32_t, typename ctrl_t = uint64_t>
ALWAYS_INLINE void multi_mk_gmma(matrix_desc_t mat_desc_c, matrix_desc_t mat_desc_acc, matrix_desc_t mat_desc_a,
                                 matrix_desc_t mat_desc_b, matrix_desc_t sparse_desc, matrix_desc_t meta_desc_a,
                                 matrix_desc_t meta_desc_b, ctrl_t mma_ctrl, abar_ptr_t abar_a, abar_ptr_t abar_b) {
  constexpr auto result_tuple =
      checkout_per_loop_stride<dtype_c, dtype_acc, dtype_a, dtype_b, wg_m, wg_n, wg_k, mma_m, mma_k, layout_a, layout_b,
                               is_sparsity, is_scale_a, is_scale_b, sparse_repr>();
  constexpr uint32_t rep_m = std::get<0>(result_tuple);
  constexpr uint32_t rep_k = std::get<1>(result_tuple);
  constexpr uint32_t slm_stride_c_m = std::get<2>(result_tuple);
  constexpr uint32_t slm_stride_acc_m = std::get<3>(result_tuple);
  constexpr uint32_t slm_stride_a_m = std::get<4>(result_tuple);
  constexpr uint32_t slm_stride_a_k = std::get<5>(result_tuple);
  constexpr uint32_t slm_stride_b_k = std::get<6>(result_tuple);
  constexpr uint32_t slm_stride_meta_a_m = std::get<7>(result_tuple);
  constexpr uint32_t slm_stride_meta_a_k = std::get<8>(result_tuple);
  constexpr uint32_t slm_stride_meta_b = std::get<9>(result_tuple);
  constexpr uint32_t slm_stride_sparse = std::get<10>(result_tuple);

#pragma unroll
  for (int i = 0; i < rep_k - 1; i++) {
    matrix_desc_t cur_mat_desc_b = mat_desc_b + slm_stride_b_k * i;
    matrix_desc_t cur_meta_desc_b = meta_desc_b + slm_stride_meta_b * i;
    matrix_desc_t cur_sparse_desc = sparse_desc + slm_stride_sparse * i;
#pragma unroll
    for (int j = 0; j < rep_m; j++) {
      matrix_desc_t cur_mat_desc_a = mat_desc_a + slm_stride_a_m * j + slm_stride_a_k * i;
      matrix_desc_t cur_meta_desc_a = meta_desc_a + slm_stride_meta_a_m * j + slm_stride_meta_a_k * i;
      matrix_desc_t cur_mat_desc_acc = mat_desc_acc + slm_stride_acc_m * j;
      async_gmma_handler<dtype_acc, dtype_acc, dtype_a, dtype_b, mma_m, wg_n, mma_k, is_sparsity, is_scale_a,
                         is_scale_b, layout_a, layout_b>(cur_mat_desc_acc, cur_mat_desc_acc, cur_mat_desc_a,
                                                         cur_mat_desc_b, cur_sparse_desc, cur_meta_desc_a,
                                                         cur_meta_desc_b, mma_ctrl, abar_a, abar_b);
    }
    mma_ctrl &= ~(uint64_t(1) << MMA_CTRL_NULL_C_OFFSET); // clean null_c
  }

#pragma unroll
  for (int j = 0; j < rep_m; j++) {
    matrix_desc_t cur_mat_desc_c = mat_desc_c + slm_stride_c_m * j;
    matrix_desc_t cur_mat_desc_acc = mat_desc_acc + slm_stride_acc_m * j;
    matrix_desc_t cur_mat_desc_a = mat_desc_a + slm_stride_a_m * j + slm_stride_a_k * (rep_k - 1);
    matrix_desc_t cur_meta_desc_a = meta_desc_a + slm_stride_meta_a_m * j + slm_stride_meta_a_k * (rep_k - 1);
    matrix_desc_t cur_mat_desc_b = mat_desc_b + slm_stride_b_k * (rep_k - 1);
    matrix_desc_t cur_meta_desc_b = meta_desc_b + slm_stride_meta_b * (rep_k - 1);
    matrix_desc_t cur_sparse_desc = sparse_desc + slm_stride_sparse * (rep_k - 1);
    async_gmma_handler<dtype_c, dtype_acc, dtype_a, dtype_b, mma_m, wg_n, mma_k, is_sparsity, is_scale_a, is_scale_b,
                       layout_a, layout_b>(cur_mat_desc_c, cur_mat_desc_acc, cur_mat_desc_a, cur_mat_desc_b,
                                           cur_sparse_desc, cur_meta_desc_a, cur_meta_desc_b, mma_ctrl, abar_a, abar_b);
  }
}

// multi-mk non-cluster gmma entry; abar_d/abar_a/abar_b
template <typename dtype_c, typename dtype_acc, typename dtype_a, typename dtype_b, uint32_t wg_m, uint32_t wg_n,
          uint32_t wg_k, uint32_t mma_m, uint32_t mma_k, bool is_sparsity = false, bool is_scale_a = false,
          bool is_scale_b = false, mem_layout layout_a = mem_layout::row_major,
          mem_layout layout_b = mem_layout::row_major, sparsity_repr_t sparse_repr = sparsity_repr_t::A4xB2,
          typename abar_ptr_t = uint64_t *, typename matrix_desc_t = uint32_t, typename ctrl_t = uint64_t>
ALWAYS_INLINE void multi_mk_gmma(matrix_desc_t mat_desc_c, matrix_desc_t mat_desc_acc, matrix_desc_t mat_desc_a,
                                 matrix_desc_t mat_desc_b, matrix_desc_t sparse_desc, matrix_desc_t meta_desc_a,
                                 matrix_desc_t meta_desc_b, ctrl_t mma_ctrl, abar_ptr_t abar_d, abar_ptr_t abar_a,
                                 abar_ptr_t abar_b) {
  constexpr auto result_tuple =
      checkout_per_loop_stride<dtype_c, dtype_acc, dtype_a, dtype_b, wg_m, wg_n, wg_k, mma_m, mma_k, layout_a, layout_b,
                               is_sparsity, is_scale_a, is_scale_b, sparse_repr>();
  constexpr uint32_t rep_m = std::get<0>(result_tuple);
  constexpr uint32_t rep_k = std::get<1>(result_tuple);
  constexpr uint32_t slm_stride_c_m = std::get<2>(result_tuple);
  constexpr uint32_t slm_stride_acc_m = std::get<3>(result_tuple);
  constexpr uint32_t slm_stride_a_m = std::get<4>(result_tuple);
  constexpr uint32_t slm_stride_a_k = std::get<5>(result_tuple);
  constexpr uint32_t slm_stride_b_k = std::get<6>(result_tuple);
  constexpr uint32_t slm_stride_meta_a_m = std::get<7>(result_tuple);
  constexpr uint32_t slm_stride_meta_a_k = std::get<8>(result_tuple);
  constexpr uint32_t slm_stride_meta_b = std::get<9>(result_tuple);
  constexpr uint32_t slm_stride_sparse = std::get<10>(result_tuple);

#pragma unroll
  for (int i = 0; i < rep_k - 1; i++) {
    matrix_desc_t cur_mat_desc_b = mat_desc_b + slm_stride_b_k * i;
    matrix_desc_t cur_meta_desc_b = meta_desc_b + slm_stride_meta_b * i;
    matrix_desc_t cur_sparse_desc = sparse_desc + slm_stride_sparse * i;
#pragma unroll
    for (int j = 0; j < rep_m; j++) {
      matrix_desc_t cur_mat_desc_a = mat_desc_a + slm_stride_a_m * j + slm_stride_a_k * i;
      matrix_desc_t cur_meta_desc_a = meta_desc_a + slm_stride_meta_a_m * j + slm_stride_meta_a_k * i;
      matrix_desc_t cur_mat_desc_acc = mat_desc_acc + slm_stride_acc_m * j;
      async_gmma_handler<dtype_acc, dtype_acc, dtype_a, dtype_b, mma_m, wg_n, mma_k, is_sparsity, is_scale_a,
                         is_scale_b, layout_a, layout_b>(cur_mat_desc_acc, cur_mat_desc_acc, cur_mat_desc_a,
                                                         cur_mat_desc_b, cur_sparse_desc, cur_meta_desc_a,
                                                         cur_meta_desc_b, mma_ctrl, abar_d, abar_a, abar_b);
    }
    mma_ctrl &= ~(uint64_t(1) << MMA_CTRL_NULL_C_OFFSET); // clean null_c
  }

#pragma unroll
  for (int j = 0; j < rep_m; j++) {
    matrix_desc_t cur_mat_desc_c = mat_desc_c + slm_stride_c_m * j;
    matrix_desc_t cur_mat_desc_acc = mat_desc_acc + slm_stride_acc_m * j;
    matrix_desc_t cur_mat_desc_a = mat_desc_a + slm_stride_a_m * j + slm_stride_a_k * (rep_k - 1);
    matrix_desc_t cur_meta_desc_a = meta_desc_a + slm_stride_meta_a_m * j + slm_stride_meta_a_k * (rep_k - 1);
    matrix_desc_t cur_mat_desc_b = mat_desc_b + slm_stride_b_k * (rep_k - 1);
    matrix_desc_t cur_meta_desc_b = meta_desc_b + slm_stride_meta_b * (rep_k - 1);
    matrix_desc_t cur_sparse_desc = sparse_desc + slm_stride_sparse * (rep_k - 1);
    async_gmma_handler<dtype_c, dtype_acc, dtype_a, dtype_b, mma_m, wg_n, mma_k, is_sparsity, is_scale_a, is_scale_b,
                       layout_a, layout_b>(cur_mat_desc_c, cur_mat_desc_acc, cur_mat_desc_a, cur_mat_desc_b,
                                           cur_sparse_desc, cur_meta_desc_a, cur_meta_desc_b, mma_ctrl, abar_d, abar_a,
                                           abar_b);
  }
}

// multi-mk cluster gmma entry; abar_a/abar_b
template <typename dtype_c, typename dtype_acc, typename dtype_a, typename dtype_b, uint32_t wg_m, uint32_t wg_n,
          uint32_t wg_k, uint32_t mma_m, uint32_t mma_k, bool is_sparsity = false, bool is_scale_a = false,
          bool is_scale_b = false, mem_layout layout_a = mem_layout::row_major,
          mem_layout layout_b = mem_layout::row_major, sparsity_repr_t sparse_repr = sparsity_repr_t::A4xB2,
          typename abar_ptr_t = uint64_t *, typename matrix_desc_t = uint32_t, typename ctrl_t = uint64_t>
ALWAYS_INLINE void multi_mk_gmma(matrix_desc_t mat_desc_c, matrix_desc_t mat_desc_acc, matrix_desc_t mat_desc_a,
                                 matrix_desc_t mat_desc_b, matrix_desc_t sparse_desc, matrix_desc_t meta_desc_a,
                                 matrix_desc_t meta_desc_b, ctrl_t mma_ctrl, abar_ptr_t abar_a, uint32_t mask_a,
                                 abar_ptr_t abar_b, uint32_t mask_b) {
  constexpr auto result_tuple =
      checkout_per_loop_stride<dtype_c, dtype_acc, dtype_a, dtype_b, wg_m, wg_n, wg_k, mma_m, mma_k, layout_a, layout_b,
                               is_sparsity, is_scale_a, is_scale_b, sparse_repr>();
  constexpr uint32_t rep_m = std::get<0>(result_tuple);
  constexpr uint32_t rep_k = std::get<1>(result_tuple);
  constexpr uint32_t slm_stride_c_m = std::get<2>(result_tuple);
  constexpr uint32_t slm_stride_acc_m = std::get<3>(result_tuple);
  constexpr uint32_t slm_stride_a_m = std::get<4>(result_tuple);
  constexpr uint32_t slm_stride_a_k = std::get<5>(result_tuple);
  constexpr uint32_t slm_stride_b_k = std::get<6>(result_tuple);
  constexpr uint32_t slm_stride_meta_a_m = std::get<7>(result_tuple);
  constexpr uint32_t slm_stride_meta_a_k = std::get<8>(result_tuple);
  constexpr uint32_t slm_stride_meta_b = std::get<9>(result_tuple);
  constexpr uint32_t slm_stride_sparse = std::get<10>(result_tuple);

#pragma unroll
  for (int i = 0; i < rep_k - 1; i++) {
    matrix_desc_t cur_mat_desc_b = mat_desc_b + slm_stride_b_k * i;
    matrix_desc_t cur_meta_desc_b = meta_desc_b + slm_stride_meta_b * i;
    matrix_desc_t cur_sparse_desc = sparse_desc + slm_stride_sparse * i;
#pragma unroll
    for (int j = 0; j < rep_m; j++) {
      matrix_desc_t cur_mat_desc_a = mat_desc_a + slm_stride_a_m * j + slm_stride_a_k * i;
      matrix_desc_t cur_meta_desc_a = meta_desc_a + slm_stride_meta_a_m * j + slm_stride_meta_a_k * i;
      matrix_desc_t cur_mat_desc_acc = mat_desc_acc + slm_stride_acc_m * j;
      async_gmma_handler<dtype_acc, dtype_acc, dtype_a, dtype_b, mma_m, wg_n, mma_k, is_sparsity, is_scale_a,
                         is_scale_b, layout_a, layout_b>(cur_mat_desc_acc, cur_mat_desc_acc, cur_mat_desc_a,
                                                         cur_mat_desc_b, cur_sparse_desc, cur_meta_desc_a,
                                                         cur_meta_desc_b, mma_ctrl, abar_a, mask_a, abar_b, mask_b);
    }
    mma_ctrl &= ~(uint64_t(1) << MMA_CTRL_NULL_C_OFFSET); // clean null_c
  }

#pragma unroll
  for (int j = 0; j < rep_m; j++) {
    matrix_desc_t cur_mat_desc_c = mat_desc_c + slm_stride_c_m * j;
    matrix_desc_t cur_mat_desc_acc = mat_desc_acc + slm_stride_acc_m * j;
    matrix_desc_t cur_mat_desc_a = mat_desc_a + slm_stride_a_m * j + slm_stride_a_k * (rep_k - 1);
    matrix_desc_t cur_meta_desc_a = meta_desc_a + slm_stride_meta_a_m * j + slm_stride_meta_a_k * (rep_k - 1);
    matrix_desc_t cur_mat_desc_b = mat_desc_b + slm_stride_b_k * (rep_k - 1);
    matrix_desc_t cur_meta_desc_b = meta_desc_b + slm_stride_meta_b * (rep_k - 1);
    matrix_desc_t cur_sparse_desc = sparse_desc + slm_stride_sparse * (rep_k - 1);
    async_gmma_handler<dtype_c, dtype_acc, dtype_a, dtype_b, mma_m, wg_n, mma_k, is_sparsity, is_scale_a, is_scale_b,
                       layout_a, layout_b>(cur_mat_desc_c, cur_mat_desc_acc, cur_mat_desc_a, cur_mat_desc_b,
                                           cur_sparse_desc, cur_meta_desc_a, cur_meta_desc_b, mma_ctrl, abar_a, mask_a,
                                           abar_b, mask_b);
  }
}

// multi-mk cluster gmma entry; abar_d/abar_a/abar_b
template <typename dtype_c, typename dtype_acc, typename dtype_a, typename dtype_b, uint32_t wg_m, uint32_t wg_n,
          uint32_t wg_k, uint32_t mma_m, uint32_t mma_k, bool is_sparsity = false, bool is_scale_a = false,
          bool is_scale_b = false, mem_layout layout_a = mem_layout::row_major,
          mem_layout layout_b = mem_layout::row_major, sparsity_repr_t sparse_repr = sparsity_repr_t::A4xB2,
          typename abar_ptr_t = uint64_t *, typename matrix_desc_t = uint32_t, typename ctrl_t = uint64_t>
ALWAYS_INLINE void multi_mk_gmma(matrix_desc_t mat_desc_c, matrix_desc_t mat_desc_acc, matrix_desc_t mat_desc_a,
                                 matrix_desc_t mat_desc_b, matrix_desc_t sparse_desc, matrix_desc_t meta_desc_a,
                                 matrix_desc_t meta_desc_b, ctrl_t mma_ctrl, abar_ptr_t abar_d, abar_ptr_t abar_a,
                                 uint32_t mask_a, abar_ptr_t abar_b, uint32_t mask_b) {
  constexpr auto result_tuple =
      checkout_per_loop_stride<dtype_c, dtype_acc, dtype_a, dtype_b, wg_m, wg_n, wg_k, mma_m, mma_k, layout_a, layout_b,
                               is_sparsity, is_scale_a, is_scale_b, sparse_repr>();
  constexpr uint32_t rep_m = std::get<0>(result_tuple);
  constexpr uint32_t rep_k = std::get<1>(result_tuple);
  constexpr uint32_t slm_stride_c_m = std::get<2>(result_tuple);
  constexpr uint32_t slm_stride_acc_m = std::get<3>(result_tuple);
  constexpr uint32_t slm_stride_a_m = std::get<4>(result_tuple);
  constexpr uint32_t slm_stride_a_k = std::get<5>(result_tuple);
  constexpr uint32_t slm_stride_b_k = std::get<6>(result_tuple);
  constexpr uint32_t slm_stride_meta_a_m = std::get<7>(result_tuple);
  constexpr uint32_t slm_stride_meta_a_k = std::get<8>(result_tuple);
  constexpr uint32_t slm_stride_meta_b = std::get<9>(result_tuple);
  constexpr uint32_t slm_stride_sparse = std::get<10>(result_tuple);

#pragma unroll
  for (int i = 0; i < rep_k - 1; i++) {
    matrix_desc_t cur_mat_desc_b = mat_desc_b + slm_stride_b_k * i;
    matrix_desc_t cur_meta_desc_b = meta_desc_b + slm_stride_meta_b * i;
    matrix_desc_t cur_sparse_desc = sparse_desc + slm_stride_sparse * i;
#pragma unroll
    for (int j = 0; j < rep_m; j++) {
      matrix_desc_t cur_mat_desc_a = mat_desc_a + slm_stride_a_m * j + slm_stride_a_k * i;
      matrix_desc_t cur_meta_desc_a = meta_desc_a + slm_stride_meta_a_m * j + slm_stride_meta_a_k * i;
      matrix_desc_t cur_mat_desc_acc = mat_desc_acc + slm_stride_acc_m * j;
      async_gmma_handler<dtype_acc, dtype_acc, dtype_a, dtype_b, mma_m, wg_n, mma_k, is_sparsity, is_scale_a,
                         is_scale_b, layout_a, layout_b>(
          cur_mat_desc_acc, cur_mat_desc_acc, cur_mat_desc_a, cur_mat_desc_b, cur_sparse_desc, cur_meta_desc_a,
          cur_meta_desc_b, mma_ctrl, abar_d, abar_a, mask_a, abar_b, mask_b);
    }
    mma_ctrl &= ~(uint64_t(1) << MMA_CTRL_NULL_C_OFFSET); // clean null_c
  }

#pragma unroll
  for (int j = 0; j < rep_m; j++) {
    matrix_desc_t cur_mat_desc_c = mat_desc_c + slm_stride_c_m * j;
    matrix_desc_t cur_mat_desc_acc = mat_desc_acc + slm_stride_acc_m * j;
    matrix_desc_t cur_mat_desc_a = mat_desc_a + slm_stride_a_m * j + slm_stride_a_k * (rep_k - 1);
    matrix_desc_t cur_meta_desc_a = meta_desc_a + slm_stride_meta_a_m * j + slm_stride_meta_a_k * (rep_k - 1);
    matrix_desc_t cur_mat_desc_b = mat_desc_b + slm_stride_b_k * (rep_k - 1);
    matrix_desc_t cur_meta_desc_b = meta_desc_b + slm_stride_meta_b * (rep_k - 1);
    matrix_desc_t cur_sparse_desc = sparse_desc + slm_stride_sparse * (rep_k - 1);
    async_gmma_handler<dtype_c, dtype_acc, dtype_a, dtype_b, mma_m, wg_n, mma_k, is_sparsity, is_scale_a, is_scale_b,
                       layout_a, layout_b>(cur_mat_desc_c, cur_mat_desc_acc, cur_mat_desc_a, cur_mat_desc_b,
                                           cur_sparse_desc, cur_meta_desc_a, cur_meta_desc_b, mma_ctrl, abar_d, abar_a,
                                           mask_a, abar_b, mask_b);
  }
}
"""
    }


# List of all public profile classes
PUBLIC_PROFILES = [ProfileRegular, ProfileScale, ProfileSparsity, ProfileSparsityScale]


class EmitMMA:
    """Responsible for emitting a Xe4 async mma template definition"""

    templates = {
        "file_header": """/*
 Generated by gen_mma.py - Do not edit.
*/

#pragma once

#include "util.hpp"

/* clang-format off */
""",
        "slash_line": """
///////////////////////////////////////////////////////////////////////////////////////////////////
""",
        "file_footer": """
/* clang-format on */
""",
    }

    def __init__(self, output_file):
        if output_file.endswith(".hpp") or output_file.endswith(".h"):
            self.output_file = output_file
        else:
            self.output_file = os.path.join(output_file, "async_gmma.hpp")

    def __enter__(self):
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)
        self.fwriter = open(self.output_file, "w")
        return self

    def emit(self, mma_types, dtype, shape, dispatch_func):
        profiles_enabled = [profile for profile in PUBLIC_PROFILES if profile.scenario in mma_types]

        for action in ["file_header", "slash_line"]:
            self.fwriter.write(EmitMMA.templates[action])

        for profile in profiles_enabled:
            self.fwriter.write(profile.templates["head"])

        self.fwriter.write(EmitMMA.templates["slash_line"])

        for profile in profiles_enabled:
            case_generator = self._create_generator(dtype, shape)
            for case in case_generator:
                self.fwriter.write(SubstituteTemplate(profile.templates["body"], case) + "\n")
            self.fwriter.write(EmitMMA.templates["slash_line"])

        for profile in profiles_enabled:
            self.fwriter.write(profile.templates["foot"])

        if dispatch_func:
            self.fwriter.write(ProfileDispatch.templates["foot"])

        self.fwriter.write(EmitMMA.templates["file_footer"])

    def _create_generator(self, dtype, shape):
        """Generate combinations of dtype, shape, and layouts.

        Args:
          dtype: List of dtype tuples
          shape: List of shape tuples

        Returns:
          A generator yielding values dictionary for each combination
        """
        major_available = ["row_major", "col_major"]
        combinations = list(itertools.product(dtype, shape, major_available, major_available))

        for dtype_, shape_, LayoutA_, LayoutB_ in combinations:
            values = {
                "M": shape_[0],
                "N": shape_[1],
                "K": shape_[2],
                "LayoutA": LayoutA_,
                "LayoutB": LayoutB_,
                "LA": repr_layout(LayoutA_, True),
                "LB": repr_layout(LayoutB_, False),
                "D": as_pisa_type(dtype_[0]),
                "C": as_pisa_type(dtype_[1]),
                "A": as_pisa_type(dtype_[-2]),
                "B": as_pisa_type(dtype_[-1]),
                "TD": as_sycl_type(dtype_[0]),
                "TC": as_sycl_type(dtype_[1]),
                "TA": as_sycl_type(dtype_[-2]),
                "TB": as_sycl_type(dtype_[-1]),
            }
            yield values

    def __exit__(self, exception_type, exception_value, traceback):
        self.fwriter.close()


################## define parser ##################


def parse_shape(value):
    pattern = re.compile(r"^\d+x\d+x\d+$")
    if not pattern.match(value):
        raise argparse.ArgumentTypeError(f"Invalid shape: {value}. Expected format: MxNxK (e.g., 32x32x32)")
    return tuple(value.split("x"))


def parse_dtype(value):
    pattern = re.compile(r"^[a-z0-9_]+$")
    if not pattern.match(value):
        raise argparse.ArgumentTypeError(
            f"Invalid dtype: {value}. Expected format: dtypeD_dtypeC_dtypeA_dtypeB (e.g., f32_f32_bf16_bf16)"
        )
    dtype_tuple = tuple(value.split("_"))
    if len(dtype_tuple) != 4:
        raise argparse.ArgumentTypeError(
            f"Invalid dtype: {value}. Expected format: dtypeD_dtypeC_dtypeA_dtypeB (e.g., f32_f32_bf16_bf16)"
        )
    return dtype_tuple


def define_parser():
    mma_types = [profile.scenario for profile in PUBLIC_PROFILES]
    parser = argparse.ArgumentParser(description="Generates xe4 async gmma code")
    parser.add_argument(
        "--mma_type",
        nargs="+",
        choices=mma_types,
        default=mma_types,
        help=f'Specify the MMA type(s). Available types: {", ".join(mma_types)}. Default is all types.',
    )
    parser.add_argument(
        "--shape",
        nargs="+",
        type=parse_shape,
        required=True,
        help="Specify the MNK shape as a list of MxNxK values, e.g., --shape 32x32x32 128x128x128",
    )
    parser.add_argument(
        "--dtype",
        nargs="+",
        type=parse_dtype,
        required=True,
        help="Specify the data types for A/B/C/D as a list of values, e.g., --dtype bf16_bf16_f32_f32. Available types: f32, bf16, f16",
    )
    parser.add_argument(
        "--dispatch_func",
        action="store_true",
        help="Enable generation of dispatch functions (default: off)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/async_gmma.hpp",
        help="Specify the output file path (default: ./output/async_gmma.hpp)",
    )
    return parser


if __name__ == "__main__":
    parser = define_parser()
    args = parser.parse_args()

    with EmitMMA(args.output) as emitter:
        emitter.emit(args.mma_type, args.dtype, args.shape, args.dispatch_func)

    print(f"Code generation completed. Check '{args.output}' for the output.")