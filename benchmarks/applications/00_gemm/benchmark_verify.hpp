/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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

#pragma once

#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/gemm_complex_mkl.h"
#include "cutlass/relatively_equal.h"
#include "cute/tensor.hpp"

#include <iostream>
#include <sstream>
#include <vector>
#include <cstdio>

using namespace cute;

namespace cutlass::benchmark {

// Verification mode enum
enum class VerifyMode {
  None = 0,           // No verification (skip both device and host)
  Device = 1,         // Device verification - uses reference::device::GemmComplex
  Host = 2            // Host verification - uses reference::host::GemmComplexMkl
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// verify_device — runs reference GEMM on device and compares.
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class ProblemShapeType, class ElementCompute,
          class ElementC, class ElementOutput, class ElementAccumulator, class ElementMMAVerify,
          class LayoutA, class LayoutB, class LayoutC, class LayoutD>
bool verify_device(
    const ProblemShapeType& problem_size,
    ElementCompute alpha, ElementCompute beta,
    DeviceAllocation<ElementMMAVerify>& block_A_dq,
    DeviceAllocation<ElementMMAVerify>& block_B_dq,
    DeviceAllocation<typename std::remove_const<ElementC>::type>& block_C,
    DeviceAllocation<typename std::remove_const<ElementOutput>::type>& block_D,
    DeviceAllocation<typename std::remove_const<ElementOutput>::type>& block_ref_D,
    std::ostream& label) {

  auto& M = cute::get<0>(problem_size);
  auto& N = cute::get<1>(problem_size);
  auto& K = cute::get<2>(problem_size);
  auto& L = cute::get<3>(problem_size);

  TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
  TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

  TensorRef ref_A(block_A_dq.get(), LayoutA::packed({M, K}));
  TensorRef ref_B(block_B_dq.get(), LayoutB::packed({K, N}));

  reference::device::GemmComplex(
          {M, N, K},
          alpha,
          ref_A,
          ComplexTransform::kNone,
          ref_B,
          ComplexTransform::kNone,
          beta,
          ref_C,
          ref_D,
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          N * K, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
  );

#if defined(CUTLASS_ENABLE_SYCL)
  compat::wait();
#else
  cudaDeviceSynchronize();
#endif

  compat::wait();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  ElementOutput const epsilon(1e-2f);
  ElementOutput const non_zero_floor(1e-4f);
  bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
    block_ref_D.get(), block_D.get(), block_D.size(), epsilon, non_zero_floor);

  if (!passed) {
    std::vector<ElementOutput> block_ref_D_host(block_ref_D.size());
    std::vector<ElementOutput> block_D_host(block_D.size());
    compat::memcpy(block_ref_D_host.data(), block_ref_D.get(), block_ref_D_host.size() * sizeof(ElementOutput));
    compat::memcpy(block_D_host.data(), block_D.get(), block_D_host.size() * sizeof(ElementOutput));
    for (int i = 0; i < block_D_host.size(); i++) {
      printf("i: %d , ref: %f, comp: %f\n", i, block_ref_D_host[i], block_D_host[i]);
    }
  }

  label << "verify_status=" << (passed ? "passed" : "failed")
        << "_with_device_ref_impl_gemm_complex ";
  return passed;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// verify_host — runs reference GEMM on host and compares.
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class ProblemShapeType, class ElementCompute,
          class ElementC, class ElementOutput, class ElementAccumulator, class ElementMMAVerify,
          class LayoutA, class LayoutB, class LayoutC, class LayoutD>
bool verify_host(
    const ProblemShapeType& problem_size,
    ElementCompute alpha, ElementCompute beta,
    DeviceAllocation<ElementMMAVerify>& block_A_dq,
    DeviceAllocation<ElementMMAVerify>& block_B_dq,
    DeviceAllocation<ElementC>& block_C,
    DeviceAllocation<ElementOutput>& block_D,
    std::ostream& label) {

  auto& M = cute::get<0>(problem_size);
  auto& N = cute::get<1>(problem_size);
  auto& K = cute::get<2>(problem_size);
  auto& L = cute::get<3>(problem_size);

  std::vector<ElementMMAVerify> host_A(block_A_dq.size());
  std::vector<ElementMMAVerify> host_B(block_B_dq.size());
  std::vector<ElementC> host_C(block_C.size());
  std::vector<ElementOutput> host_D(block_D.size());

  cutlass::device_memory::copy_to_host(host_A.data(), block_A_dq.get(), block_A_dq.size());
  cutlass::device_memory::copy_to_host(host_B.data(), block_B_dq.get(), block_B_dq.size());
  cutlass::device_memory::copy_to_host(host_C.data(), block_C.get(), block_C.size());
  cutlass::device_memory::copy_to_host(host_D.data(), block_D.get(), block_D.size());
  compat::wait();

  std::vector<ElementOutput> host_ref_D(block_D.size());

  TensorRef ref_A(host_A.data(), LayoutA::packed({M, K}));
  TensorRef ref_B(host_B.data(), LayoutB::packed({K, N}));
  TensorRef ref_C(host_C.data(), LayoutC::packed({M, N}));
  TensorRef ref_D(host_ref_D.data(), LayoutD::packed({M, N}));

  bool used_mkl = reference::host::GemmComplexMkl(
          {M, N, K},
          alpha,
          ref_A,
          ComplexTransform::kNone,
          ref_B,
          ComplexTransform::kNone,
          beta,
          ref_C,
          ref_D,
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          N * K, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
  );

  // Match the tolerances used by the device verification path.
  ElementOutput const epsilon(1e-2f);
  ElementOutput const non_zero_floor(1e-4f);
  bool passed = true;
  for (std::size_t i = 0; i < host_D.size(); ++i) {
    if (!cutlass::relatively_equal(host_ref_D[i], host_D[i], epsilon, non_zero_floor)) {
      passed = false;
      printf("i: %zu , ref: %f, comp: %f\n", i, float(host_ref_D[i]), float(host_D[i]));
    }
  }

  label << "verify_status=" << (passed ? "passed" : "failed")
        << "_with_host_ref_impl_" << (used_mkl ? "mkl " : "gemm_complex ");
  return passed;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// run_verify — verifies the GEMM result according to mode.
/// Returns true if verification passed or was skipped.
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class ProblemShapeType, class ElementCompute,
          class ElementC, class ElementOutput, class ElementAccumulator, class ElementMMAVerify,
          class LayoutA, class LayoutB, class LayoutC, class LayoutD>
bool run_verify(
    VerifyMode mode,
    const ProblemShapeType& problem_size,
    ElementCompute alpha, ElementCompute beta,
    DeviceAllocation<ElementMMAVerify>& block_A_dq,
    DeviceAllocation<ElementMMAVerify>& block_B_dq,
    DeviceAllocation<ElementC>& block_C,
    DeviceAllocation<ElementOutput>& block_D,
    DeviceAllocation<ElementOutput>& block_ref_D,
    std::ostream& label) {

  if (mode == VerifyMode::None) return true;

  if (mode == VerifyMode::Host) {
    return verify_host<ProblemShapeType, ElementCompute, ElementC, ElementOutput,
                       ElementAccumulator, ElementMMAVerify, LayoutA, LayoutB, LayoutC, LayoutD>(
        problem_size, alpha, beta, block_A_dq, block_B_dq, block_C, block_D, label);
  } else {
    return verify_device<ProblemShapeType, ElementCompute, ElementC, ElementOutput,
                         ElementAccumulator, ElementMMAVerify, LayoutA, LayoutB, LayoutC, LayoutD>(
        problem_size, alpha, beta, block_A_dq, block_B_dq, block_C, block_D, block_ref_D, label);
  }
}

} // namespace cutlass::benchmark
