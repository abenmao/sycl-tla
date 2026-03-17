/* Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Unit tests for block-wise quantization via cute::quantize API.
*/

#include "cutlass/detail/layout.hpp"

#include <cmath>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/tensor_sg.hpp>
#include <cute/quantization/quantize.hpp>
#include <cute/algorithm/reorder.hpp>
#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>

#include "cutlass_unit_test.h"

#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>
#include <cutlass/numeric_types.h>

using namespace cute;
using namespace cutlass;
using namespace compat::experimental;

// ============================================================================
// Test Kernels
// ============================================================================

template<class...> class BlockWiseQuantizeKernelName;

// Kernel: loads src from global (round-robin), quantizes, stores dst & scale.
//
// Global memory layout (round-robin):
//   global[tid + v * sg_size]  ↔  register slot v of thread tid
//
// The TV layouts map (thread, value) → logical (M, N) coordinates.
template <int BlockSize,
          class SrcType, class DstType, class ScaleType,
          class SrcTVLayout, class DstTVLayout, class ScaleTVLayout>
void block_quantize_kernel(SrcType*  src_global,
                           DstType*  dst_global,
                           ScaleType* scale_global)
{
  const int tid = ThreadIdxX();

  // --- Source: global → register (round-robin) ---
  constexpr int src_total   = size(SrcTVLayout{});
  constexpr int src_per_thr = src_total / intel::sg_size;
  static_assert(src_total % intel::sg_size == 0);

  SrcType src_local[src_per_thr];
  for (int i = 0; i < src_per_thr; ++i)
    src_local[i] = src_global[tid + i * intel::sg_size];

  auto src_frag = make_tensor(make_rmem_ptr(src_local),
                               make_layout(Shape<Int<src_per_thr>>{}));
  auto src_sg   = make_subgroup_tensor(src_frag, SrcTVLayout{});

  // --- Destination: zero-initialized register fragment ---
  constexpr int dst_total   = size(DstTVLayout{});
  constexpr int dst_per_thr = dst_total / intel::sg_size;
  static_assert(dst_total % intel::sg_size == 0);

  DstType dst_local[dst_per_thr]{};
  auto dst_frag = make_tensor(make_rmem_ptr(dst_local),
                               make_layout(Shape<Int<dst_per_thr>>{}));
  auto dst_sg   = make_subgroup_tensor(dst_frag, DstTVLayout{});

  // --- Scale: zero-initialized register fragment ---
  constexpr int scale_total   = size(ScaleTVLayout{});
  constexpr int scale_per_thr = scale_total / intel::sg_size;
  static_assert(scale_total % intel::sg_size == 0);

  ScaleType scale_local[scale_per_thr]{};
  auto scale_frag = make_tensor(make_rmem_ptr(scale_local),
                                 make_layout(Shape<Int<scale_per_thr>>{}));
  auto scale_sg   = make_subgroup_tensor(scale_frag, ScaleTVLayout{});

  // --- Quantize ---
  quantize<BlockSize>(src_sg, dst_sg, scale_sg);

  // --- Store dst back to global (round-robin) ---
  for (int i = 0; i < dst_per_thr; ++i)
    dst_global[tid + i * intel::sg_size] = dst_local[i];

  // --- Store scale back to global (round-robin) ---
  for (int i = 0; i < scale_per_thr; ++i)
    scale_global[tid + i * intel::sg_size] = scale_local[i];
}

// Launch helper: allocates device memory, launches kernel, copies results back.
template <int BlockSize,
          class SrcType, class DstType, class ScaleType,
          class SrcTVLayout, class DstTVLayout, class ScaleTVLayout, int TestID>
void run_block_quantize_test(
    cutlass::host_vector<SrcType>&   host_src,
    cutlass::host_vector<DstType>&   host_dst,
    cutlass::host_vector<ScaleType>& host_scale)
{
  cutlass::device_vector<SrcType>   device_src   = host_src;
  cutlass::device_vector<DstType>   device_dst(size(DstTVLayout{}));
  cutlass::device_vector<ScaleType> device_scale(size(ScaleTVLayout{}));

  launch<block_quantize_kernel<BlockSize, SrcType, DstType, ScaleType,
                               SrcTVLayout, DstTVLayout, ScaleTVLayout>,
         BlockWiseQuantizeKernelName<SrcType, DstType, ScaleType, Int<TestID>>>(
      launch_policy{compat::dim3(1), compat::dim3(intel::sg_size),
                    kernel_properties{sycl_exp::sub_group_size<intel::sg_size>}},
      device_src.data(), device_dst.data(), device_scale.data());

  compat::wait_and_throw();
  host_dst   = device_dst;
  host_scale = device_scale;
}

// ============================================================================
// CPU Reference Implementation for Block-Wise Quantization
// ============================================================================
//
// Operates on row-major 2D logical data: element (m, n) at index m*N + n.
//
//   Blocks along N, scale shape (M, N/BlockSize), stored row-major
//
// Algorithm (matches the GPU implementation):
//   Phase 1: Compute per-block absolute max
//   Phase 2: scale = round_toward_zero(target_max / block_amax)
//   Phase 3: dst[i] = round_to_nearest(src[i] * scale)
//
template <int BlockSize, class SrcType, class DstType, class ScaleType>
void reference_block_wise_quantize(
    const std::vector<SrcType>& src,     // M*N elements, row-major
    std::vector<DstType>&       dst,     // M*N elements, row-major (output)
    std::vector<ScaleType>&     scales,  // scale factors (output)
    int M, int N)
{
  using FloatToScale  = cutlass::NumericConverter<ScaleType, float,
                         cutlass::FloatRoundStyle::round_toward_zero>;
  using FloatToDst    = cutlass::NumericConverter<DstType, float,
                         cutlass::FloatRoundStyle::round_to_nearest>;

  const float target_max = static_cast<float>(
      cutlass::platform::numeric_limits<DstType>::max());

  dst.resize(static_cast<size_t>(M * N));

  const int NumBlocks = N / BlockSize;
  scales.resize(static_cast<size_t>(M * NumBlocks));

  for (int m = 0; m < M; ++m) {
    for (int b = 0; b < NumBlocks; ++b) {
      // Phase 1: block_amax for row m, block b
      float block_amax = 0.0f;
      for (int k = 0; k < BlockSize; ++k) {
        int n = b * BlockSize + k;
        float val = static_cast<float>(src[static_cast<size_t>(m * N + n)]);
        block_amax = std::max(block_amax, std::abs(val));
      }

      // Phase 2: compute & store scale
      float s = (block_amax > 0.0f) ? (target_max / block_amax) : 0.0f;
      ScaleType scale = FloatToScale{}(s);
      scales[static_cast<size_t>(m * NumBlocks + b)] = scale;

      // Phase 3: quantize each element in the block
      for (int k = 0; k < BlockSize; ++k) {
        int n = b * BlockSize + k;
        float val    = static_cast<float>(src[static_cast<size_t>(m * N + n)]);
        float scaled = val * static_cast<float>(scale);
        dst[static_cast<size_t>(m * N + n)] = FloatToDst{}(scaled);
      }
    }
  }
}

// ============================================================================
// Conversion between row-major logical data and round-robin global memory
// ============================================================================
//
// GPU global memory uses round-robin layout:
//   global[tid + v * sg_size]  ↔  register slot v of thread tid
//
// A TV layout maps (tid, v) → (m, n).
// Logical row-major data indexes as: logical[m * N + n].
//
// CONSTRAINT: TV layouts must be rank 2 for tv(tid, v) to work.
//
// logical_to_roundrobin: fills a pre-allocated host_vector from row-major src
// roundrobin_to_logical: fills a row-major std::vector from round-robin host_vector
//
template <class TVLayout, class T>
void logical_to_roundrobin(const std::vector<T>& logical,  // M*N row-major
                           cutlass::host_vector<T>& global, // pre-allocated, size(TVLayout)
                           int N)
{
  constexpr auto tv = TVLayout{};
  constexpr int total = size(tv);
  constexpr int vals = total / intel::sg_size;

  for (int tid = 0; tid < intel::sg_size; ++tid) {
    for (int v = 0; v < vals; ++v) {
      auto coord = tv(tid, v);
      int m = get<0>(coord);
      int n = get<1>(coord);
      global[tid + v * intel::sg_size] = logical[static_cast<size_t>(m * N + n)];
    }
  }
}

template <class TVLayout, class T>
void roundrobin_to_logical(const cutlass::host_vector<T>& global, // size(TVLayout) round-robin
                           std::vector<T>& logical,               // M*N row-major (output)
                           int M, int N)
{
  constexpr auto tv = TVLayout{};
  constexpr int total = size(tv);
  constexpr int vals = total / intel::sg_size;
  logical.assign(static_cast<size_t>(M * N), T{});

  for (int tid = 0; tid < intel::sg_size; ++tid) {
    for (int v = 0; v < vals; ++v) {
      auto coord = tv(tid, v);
      int m = get<0>(coord);
      int n = get<1>(coord);
      logical[static_cast<size_t>(m * N + n)] = global[tid + v * intel::sg_size];
    }
  }
}

// ============================================================================
// Initialize source data (row-major logical)
// ============================================================================
template <class SrcType>
void initialize_source(std::vector<SrcType>& src, int M, int N) {
  src.resize(static_cast<size_t>(M * N));
  for (size_t i = 0; i < src.size(); ++i) {
    // Alternating sign pattern to exercise negative values
    float sign = (i % 3 == 0) ? -1.0f : 1.0f;
    float val  = sign * static_cast<float>(i % 17) * 0.5f;
    if constexpr (std::is_same_v<SrcType, cutlass::bfloat16_t>) {
      src[i] = cutlass::bfloat16_t(val);
    } else if constexpr (std::is_same_v<SrcType, cutlass::half_t>) {
      src[i] = cutlass::half_t(val);
    } else if constexpr (std::is_same_v<SrcType, float>) {
      src[i] = val;
    } else {
      CUTE_INVALID_CONTROL_PATH("Not Implemented");
    }
  }
}

// ============================================================================
// TV Layouts
// ============================================================================
//
// Design for src TV layouts:
//   - The row index m must be THREAD-INDEPENDENT (depends only on value v).
//   - Thread must map ONLY to dimension 1 (n).
//
// Design for scale TV layouts:
//   Scales are subgroup-uniform (block_amax computed via cross-lane reduce).
//   Thread stride is Int<0> (degenerate): all threads redundantly store
//   the full set of scale values.
//

// ---- Src/Dst TV Layout: M=16, N=32 ----
// Shape:  (Int<16>, (Int<16>, Int<2>))
// Stride: (ScaledBasis<Int<1>,1>, (ScaledBasis<Int<1>,0>, ScaledBasis<Int<16>,1>))
//
// Mapping: (t, (v0, v1)) → m = v0,  n = t + v1*16
//   - Thread t maps to n = t  (dim 1 only, thread-independent m) ✓
//   - v0 ∈ [0,16) → m,  v1 ∈ [0,2) → n offset {0, 16}
//   - coshape: (16, 32),  total: 512,  per_thr: 32
//
using DataTVLayout_16x32 = decltype(make_layout(
    make_shape(Int<16>{}, make_shape(Int<16>{}, Int<2>{})),
    make_stride(ScaledBasis<Int<1>,1>{},
                make_stride(ScaledBasis<Int<1>,0>{}, ScaledBasis<Int<16>,1>{}))
));

// ---- Scale TV Layout for (M=16, NumBlocks=1) ----
// Thread stride = 0 (degenerate, scales are subgroup-uniform).
// Shape:  (Int<16>, (Int<16>, Int<1>))
// Stride: (Int<0>, (ScaledBasis<Int<1>,0>, ScaledBasis<Int<1>,1>))
//
// Mapping: (t, (v0, v1)) → m = v0,  nb = v1 = 0
//   - coshape: (16, 1),  total: 256,  per_thr: 16
//
using ScaleTVLayout_16x1 = decltype(make_layout(
    make_shape(Int<16>{}, make_shape(Int<16>{}, Int<1>{})),
    make_stride(Int<0>{},
                make_stride(ScaledBasis<Int<1>,0>{}, ScaledBasis<Int<1>,1>{}))
));

// ---- Scale TV Layout for (M=16, NumBlocks=2) ----
// Thread stride = 0 (degenerate, scales are subgroup-uniform).
// Shape:  (Int<16>, (Int<16>, Int<2>))
// Stride: (Int<0>, (ScaledBasis<Int<1>,0>, ScaledBasis<Int<1>,1>))
//
// Mapping: (t, (v0, v1)) → m = v0,  nb = v1
//   - coshape: (16, 2),  total: 512,  per_thr: 32
//
using ScaleTVLayout_16x2 = decltype(make_layout(
    make_shape(Int<16>{}, make_shape(Int<16>{}, Int<2>{})),
    make_stride(Int<0>{},
                make_stride(ScaledBasis<Int<1>,0>{}, ScaledBasis<Int<1>,1>{}))
));

// ============================================================================
// Test struct: XeBlockQuantizeTest
//
// Template parameters:
//   BlockSize  — quantization block size
//   SrcType, DstType, ScaleType — element types
//   SrcTVLayout, DstTVLayout, ScaleTVLayout — TV layouts for the 3 tensors
//   TestID — unique ID for kernel name dedup
// ============================================================================
template <int BlockSize,
          class SrcType, class DstType, class ScaleType,
          class SrcTVLayout, class DstTVLayout, class ScaleTVLayout,
          int TestID>
struct XeBlockQuantizeTest {
  static void run() {
    // ---- Logical dimensions from src TV layout ----
    constexpr auto logical_shape = atuple_coshape(SrcTVLayout{});
    constexpr int M = get<0>(logical_shape);
    constexpr int N = get<1>(logical_shape);

    // ---- 1. Initialize row-major logical source data ----
    std::vector<SrcType> src_logical;
    initialize_source(src_logical, M, N);

    // ---- 2. CPU reference ----
    std::vector<DstType>   ref_dst;
    std::vector<ScaleType> ref_scales;
    reference_block_wise_quantize<BlockSize>(src_logical, ref_dst, ref_scales, M, N);

    // ---- 3. Convert logical src → round-robin for GPU ----
    constexpr int src_total = size(SrcTVLayout{});
    cutlass::host_vector<SrcType> rr_src(src_total);
    logical_to_roundrobin<SrcTVLayout>(src_logical, rr_src, N);

    // ---- 4. Run GPU kernel ----
    constexpr int dst_total   = size(DstTVLayout{});
    constexpr int scale_total = size(ScaleTVLayout{});
    cutlass::host_vector<DstType>   rr_dst(dst_total);
    cutlass::host_vector<ScaleType> rr_scale(scale_total);
    run_block_quantize_test<BlockSize, SrcType, DstType, ScaleType,
                            SrcTVLayout, DstTVLayout, ScaleTVLayout, TestID>(
        rr_src, rr_dst, rr_scale);

    // ---- 5. Convert round-robin GPU results → logical ----
    std::vector<DstType> gpu_dst_logical;
    roundrobin_to_logical<DstTVLayout>(rr_dst, gpu_dst_logical, M, N);

    // Compute scale dimensions from data shape + quantization parameters.
    // (Avoids atuple_coshape which collapses rank when a dimension is 1.)
    constexpr int ScaleM = M;
    constexpr int ScaleN = N / BlockSize;
    std::vector<ScaleType> gpu_scale_logical;
    roundrobin_to_logical<ScaleTVLayout>(rr_scale, gpu_scale_logical, ScaleM, ScaleN);

    // ---- 6. Compare dst element-by-element ----
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        size_t idx = static_cast<size_t>(m * N + n);
        EXPECT_NEAR(static_cast<float>(gpu_dst_logical[idx]),
                    static_cast<float>(ref_dst[idx]), 1e-4f)
            << "dst mismatch at (" << m << ", " << n << ")";
      }
    }

    // ---- 7. Compare scales element-by-element ----
    // Scale magnitudes can be large (target_max / block_amax), so use a
    // relative tolerance of 1e-6 (≈ a few float32 ULPs) instead of a fixed
    // absolute threshold.
    for (int s0 = 0; s0 < ScaleM; ++s0) {
      for (int s1 = 0; s1 < ScaleN; ++s1) {
        size_t idx = static_cast<size_t>(s0 * ScaleN + s1);
        float gpu_val = static_cast<float>(gpu_scale_logical[idx]);
        float ref_val = static_cast<float>(ref_scales[idx]);
        float tol = 1e-6f * std::max(std::abs(gpu_val), std::abs(ref_val));
        EXPECT_NEAR(gpu_val, ref_val, std::max(tol, 1e-6f))
            << "scale mismatch at (" << s0 << ", " << s1 << ")";
      }
    }
  }
};

// ============================================================================
// Test Cases (blocking along N)
// ============================================================================
// Src/Dst: M=16, N=32  (DataTVLayout_16x32)
// Uses same TV layout for dst as src (reorder is identity; reorder is tested separately).

// --- BlockSize=32 (single block per row, NumBlocks=1) ---
// Scale shape: (16, 1)

TEST(CuTe_Xe_BlockQuantize, bf16_to_e4m3_dim1_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::bfloat16_t, cutlass::float_e4m3_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x1, 0>::run();
}

TEST(CuTe_Xe_BlockQuantize, bf16_to_e5m2_dim1_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::bfloat16_t, cutlass::float_e5m2_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x1, 1>::run();
}

TEST(CuTe_Xe_BlockQuantize, half_to_e4m3_dim1_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::half_t, cutlass::float_e4m3_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x1, 2>::run();
}

TEST(CuTe_Xe_BlockQuantize, half_to_e5m2_dim1_bs32) {
  XeBlockQuantizeTest<32,
      cutlass::half_t, cutlass::float_e5m2_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x1, 3>::run();
}

// --- BlockSize=16 (two blocks per row, NumBlocks=2) ---
// Scale shape: (16, 2)

TEST(CuTe_Xe_BlockQuantize, bf16_to_e4m3_dim1_bs16) {
  XeBlockQuantizeTest<16,
      cutlass::bfloat16_t, cutlass::float_e4m3_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x2, 4>::run();
}

TEST(CuTe_Xe_BlockQuantize, bf16_to_e5m2_dim1_bs16) {
  XeBlockQuantizeTest<16,
      cutlass::bfloat16_t, cutlass::float_e5m2_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x2, 5>::run();
}

TEST(CuTe_Xe_BlockQuantize, half_to_e4m3_dim1_bs16) {
  XeBlockQuantizeTest<16,
      cutlass::half_t, cutlass::float_e4m3_t, float,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x2, 6>::run();
}

TEST(CuTe_Xe_BlockQuantize, half_to_e5m2_dim1_bs16) {
  XeBlockQuantizeTest<16,
      cutlass::half_t, cutlass::float_e5m2_t, float_ue8m0_t,
      DataTVLayout_16x32, DataTVLayout_16x32, ScaleTVLayout_16x2, 7>::run();
}
