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


def as_sycl_type(dtype):
    converted_types = {
        "f32": "float",
        "f16": "fp16",
        "s32": "int32_t",
        "s8": "int8_t",
        "e3m0": "fp4_e3m0",
    }

    unchanged_types = ["bf16", "bf8", "hf8"]

    if dtype in converted_types:
        return converted_types[dtype]
    elif dtype in unchanged_types:
        return dtype
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def repr_layout(layout, is_a):
    id_ = "a" if is_a else "b"
    return "" if layout == "row_major" else f".{id_}t"


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
                "D": dtype_[0],
                "C": dtype_[1],
                "A": dtype_[-2],
                "B": dtype_[-1],
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
