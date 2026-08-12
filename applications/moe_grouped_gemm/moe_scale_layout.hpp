/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Single source of truth for the MoE scale-surface geometry, shared by
           the device kernels and the host scale packer.

    Dependency-free on purpose: the device kernels (kernel/xe_moe_gemm_greedy.hpp,
    kernel/xe_moe_gemm_double_buffer.hpp) and the host packer (scale_surface_geom /
    pack_moe_scales in runner/moe_types.hpp) all include this, so the padding
    constants cannot drift between host allocation and kernel strides. A mismatch
    is silent: the kernel reads into the next expert's stripe rather than faulting.
*/

#pragma once

namespace cutlass::moe {

// Per-expert scale buffers are padded to this alignment for the 2D block-scale
// load surface. Power of two — the kernels use bitwise round-up.
constexpr int kScaleAlign = 64;

// Legacy spelling, kept so host call sites read naturally next to round_up_align.
constexpr int kBlockScaleAlign = kScaleAlign;

// TENSOR scale-path surface geometry (ScaleKind::Tensor). Fixed, NOT derived
// from the workgroup tile:
//   * height 2 — the BDPAS register-offset scheme requires Height=2; both K slots
//     hold the same duplicated scale value.
//   * one kScaleAlign-wide N stripe per expert — the 2D scale load walks the
//     surface in kScaleAlign-wide steps, covering the whole subgroup N extent as
//     long as SG_N <= kScaleAlign (static_assert'd in both kernels).
//
// Both values are 8-bit-only: the height follows from MMA_K =
// 256 / sizeof_bits(Element) = 32 against the tensor tiles' BLK_K of 64. The
// tensor-scale paths static_assert fp8, so a future 4-bit tensor config fails
// loudly rather than silently mis-striding.
constexpr int kTensorScaleK = 2;
constexpr int kTensorPaddedScaleN = kScaleAlign;

} // namespace cutlass::moe
