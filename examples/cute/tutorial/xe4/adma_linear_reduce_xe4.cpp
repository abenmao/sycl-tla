/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation. All rights reserved.
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

//
// ADMA Linear Reduce Tutorial (Xe4)
//
// Tests all 26 Bspec (T, RedOp) configurations for async_linear_fred / async_linear_ired.
// Each config: ADMA linear copy (G2S) then ADMA linear reduce (S2G, atomic).
// Multiple workgroups reduce into the SAME output region in GMEM.
//

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
#include <iostream>

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gpu_generics.h>
#include <cute/atom/copy_traits_xe4_adma.hpp>
#include <cute/arch/xe4_util.hpp>
#include "cutlass/arch/barrier.h"

using namespace cute;
using sycl::ext::oneapi::this_work_item::get_nd_item;

// ---------------------------------------------------------------------------
// Name helpers
// ---------------------------------------------------------------------------
template <RedOp R> constexpr const char* op_name() {
  if constexpr (R == RedOp::Add)     return "Add";     if constexpr (R == RedOp::Min)     return "Min";
  if constexpr (R == RedOp::Max)     return "Max";     if constexpr (R == RedOp::Smin)    return "Smin";
  if constexpr (R == RedOp::Smax)    return "Smax";    if constexpr (R == RedOp::Umin)    return "Umin";
  if constexpr (R == RedOp::Umax)    return "Umax";    if constexpr (R == RedOp::And)     return "And";
  if constexpr (R == RedOp::Or)      return "Or";      if constexpr (R == RedOp::Xor)     return "Xor";
  if constexpr (R == RedOp::Incwrap) return "Incwrap"; if constexpr (R == RedOp::Decwrap) return "Decwrap";
  return "?";
}
template <typename T> constexpr const char* dtype_name() {
  if constexpr (std::is_same_v<T, fp16>)               return "fp16";
  if constexpr (std::is_same_v<T, bf16>)               return "bf16";
  if constexpr (std::is_same_v<T, float>)              return "float";
  if constexpr (std::is_same_v<T, double>)             return "double";
  if constexpr (std::is_same_v<T, int>)                return "int32";
  if constexpr (std::is_same_v<T, uint32_t>)           return "uint32";
  if constexpr (std::is_same_v<T, long>)               return "int64";
  if constexpr (std::is_same_v<T, unsigned long>)      return "uint64";
  return "?";
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
template <typename T, RedOp Rop>
T cpu_reduce(T acc, T val) {
  if constexpr (std::is_integral_v<T>) {
    if constexpr (Rop == RedOp::Add)     return acc + val;
    if constexpr (Rop == RedOp::Smin || Rop == RedOp::Umin) return acc < val ? acc : val;
    if constexpr (Rop == RedOp::Smax || Rop == RedOp::Umax) return acc > val ? acc : val;
    if constexpr (Rop == RedOp::And)     return acc & val;
    if constexpr (Rop == RedOp::Or)      return acc | val;
    if constexpr (Rop == RedOp::Xor)     return acc ^ val;
    if constexpr (Rop == RedOp::Incwrap) return (acc >= val) ? T(0) : T(acc + 1);
    if constexpr (Rop == RedOp::Decwrap) return ((acc == T(0)) || (acc > val)) ? val : T(acc - 1);
  } else {
    if constexpr (Rop == RedOp::Add) {
      if constexpr (sizeof(T) > 2) return acc + val;
      else return static_cast<T>(static_cast<float>(acc) + static_cast<float>(val));
    }
    if constexpr (Rop == RedOp::Min)
      return (static_cast<double>(acc) <= static_cast<double>(val)) ? acc : val;
    if constexpr (Rop == RedOp::Max)
      return (static_cast<double>(acc) >= static_cast<double>(val)) ? acc : val;
  }
}

template <typename T, RedOp Rop>
T identity_value() {
  if constexpr (Rop == RedOp::Add || Rop == RedOp::Or || Rop == RedOp::Xor ||
                Rop == RedOp::Umax || Rop == RedOp::Incwrap || Rop == RedOp::Decwrap)
    return T(0);
  if constexpr (Rop == RedOp::And)   return static_cast<T>(~T(0));
  if constexpr (Rop == RedOp::Min  || Rop == RedOp::Smin || Rop == RedOp::Umin)
    return std::is_integral_v<T> ? std::numeric_limits<T>::max() : static_cast<T>(65504.0f);
  if constexpr (Rop == RedOp::Max  || Rop == RedOp::Smax)
    return std::is_integral_v<T> ? std::numeric_limits<T>::lowest() : static_cast<T>(-65504.0f);
}

// ---------------------------------------------------------------------------
// Device kernel
// ---------------------------------------------------------------------------
constexpr static size_t SmemAlignment = 256;

template <class ElementA, class SmemLayoutA>
struct SharedStorage : cute::aligned_struct<SmemAlignment, _0> {
  cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>, SmemAlignment> smem_A;
};

template <class ProblemShape, int N,
          class T, class SmemLayoutA, class LoadAtom, class ReduceAtom>
void adma_linear_reduce_kernel(ProblemShape problemSize,
    T const* A, SmemLayoutA, LoadAtom adma_load, ReduceAtom adma_reduce,
    T* C, sycl::nd_item<3> item)
{
  using Smem = SharedStorage<T, SmemLayoutA>;
  auto ptr = alloc_slm_buffer<uint8_t, sizeof(Smem)>(item.get_group());
  Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<Smem*>(ptr)->smem_A.begin()), SmemLayoutA{});

  auto gA = coalesce(local_tile(
      make_tensor(make_gmem_ptr(A), make_layout(make_shape(problemSize))),
      make_shape(Int<N>{}), make_coord(BlockIdxX()), Step<_1>{}));
  auto gC = coalesce(make_tensor(make_gmem_ptr(C), make_layout(make_shape(Int<N>{}))));

  constexpr int dma_bytes = N * sizeof(T);
  uint32_t leader = cute::elect_one_sync();
  uint32_t sg     = get_sg_id();

  auto load_bar   = allocate_abar<0>();
  auto reduce_bar = allocate_abar<1>();

  if (leader && sg == 0)      xe4_initialize_barrier(load_bar[0], 1);
  else if (leader && sg == 1) xe4_initialize_barrier(reduce_bar[0], 1);
  sycl::group_barrier(get_nd_item<1>().get_group());

  if (sg == 0 && leader) {
    xe4_set_barrier_transaction_bytes(load_bar[0], dma_bytes);
    copy(adma_load.with(&load_bar[0]), gA, coalesce(sA));
  }
  // SG1: wait for load (via abarrier), then reduce SLM → GMEM
  if (sg == 1 && leader) {
    xe4_wait_barrier(load_bar[0], 0);
    xe4_set_barrier_transaction_bytes(reduce_bar[0], dma_bytes);
    copy(adma_reduce.with(&reduce_bar[0]), coalesce(sA), gC);
    xe4_wait_barrier(reduce_bar[0], 0);
  }
}

// ---------------------------------------------------------------------------
// run_config: allocate, launch, verify one (T, RedOp) combination
// ---------------------------------------------------------------------------
template <typename T, RedOp Rop>
bool run_config(sycl::queue& queue) {
  constexpr int N         = 65536 / (int)sizeof(T);
  constexpr bool order_dep = (Rop == RedOp::Incwrap || Rop == RedOp::Decwrap);
  // Incwrap/Decwrap are order-dependent: result depends on the sequence of
  // reductions applied. Multiple WGs reduce concurrently with no guaranteed
  // ordering, making the final value non-deterministic. Restrict to 1 WG so
  // the CPU reference can reproduce the exact reduction sequence.
  constexpr int n_tiles   = order_dep ? 1 : 8;
  constexpr int totalSize = n_tiles * N;

  printf("  [%-6s x %-7s] %d WGs -> %5d elems ... ", dtype_name<T>(), op_name<Rop>(), n_tiles, N);
  fflush(stdout);

  T* A = sycl::malloc_shared<T>(totalSize, queue);
  T* C = sycl::malloc_shared<T>(N, queue);
  std::vector<T> C_ref(N);

  // Fill source with random data
  for (int i = 0; i < totalSize; i++) {
    if constexpr (std::is_integral_v<T>) {
      int r = (rand() % 21) - 10;
      A[i] = static_cast<T>(std::is_unsigned_v<T> ? std::abs(r) : r);
      if (order_dep && A[i] < T(1)) A[i] = T(1);
    } else {
      A[i] = static_cast<T>(-1.0f + 2.0f * (float(rand()) / float(RAND_MAX)));
      if (order_dep) A[i] = static_cast<T>(std::fabs(static_cast<float>(A[i])) + 0.01f);
    }
  }

  // CPU reference
  T init = identity_value<T, Rop>();
  for (int i = 0; i < N; i++) { C[i] = init; C_ref[i] = init; }
  for (int t = 0; t < n_tiles; t++)
    for (int i = 0; i < N; i++)
      C_ref[i] = cpu_reduce<T, Rop>(C_ref[i], A[t * N + i]);

  // Build atoms and launch
  using SmemLayout = decltype(make_layout(Int<N>{}, Int<1>{}));
  constexpr auto Bytes = Int<N * (int)sizeof(T)>{};
  // Load & Reduce Atoms
  auto load_atom   = make_tiled_copy(Copy_Atom<Copy_Traits<XE4_ADMA_LINEAR_LOAD<>,
                         decltype(Bytes)>, T>{}, Layout<_1>{}, Layout<Int<N>>{});
  auto reduce_atom = make_tiled_copy(Copy_Atom<Copy_Traits<XE4_ADMA_LINEAR_REDUCE<T, Rop,
                         BarrierType::Abarrier>, decltype(Bytes)>, T>{}, Layout<_1>{}, Layout<Int<N>>{});

  namespace syclexp = sycl::ext::oneapi::experimental;
  syclexp::properties Props{syclexp::work_groups_per_cluster<3>(sycl::range<3>(1, 1, 1))};
  sycl::range<3> local(1, 2, 32);

  syclexp::submit_with_event(queue, [&](sycl::handler &h) {
    syclexp::nd_launch(h, syclexp::launch_config(
        sycl::nd_range<3>(sycl::range<3>(1, 1, n_tiles) * local, local), Props),
      [=](sycl::nd_item<3> item) ALWAYS_INLINE {
        adma_linear_reduce_kernel<decltype(totalSize), N, T, SmemLayout,
            decltype(load_atom), decltype(reduce_atom)>(
            totalSize, A, SmemLayout{}, load_atom, reduce_atom, C, item);
    });
  }).wait();

  // Verify
  int errs = 0;
  for (int i = 0; i < N; i++) {
    bool ok;
    if constexpr (std::is_integral_v<T>) ok = (C[i] == C_ref[i]);
    else {
      double ref = static_cast<double>(C_ref[i]), got = static_cast<double>(C[i]);
      double tol = std::fabs(ref) * (sizeof(T) <= 2 ? 0.05 : 0.001) + (sizeof(T) <= 2 ? 0.1 : 1e-5);
      ok = (std::fabs(ref - got) <= tol);
    }
    if (!ok && errs++ < 5) {
      if constexpr (std::is_integral_v<T>)
        printf("\n    Mismatch[%d]: ref=%lld got=%lld", i, (long long)C_ref[i], (long long)C[i]);
      else
        printf("\n    Mismatch[%d]: ref=%f got=%f", i, static_cast<double>(C_ref[i]), static_cast<double>(C[i]));
    }
  }
  sycl::free(A, queue);  sycl::free(C, queue);
  if (errs == 0) { printf("PASSED\n"); } else { printf("\n    FAILED (%d errors)\n", errs); }
  return errs == 0;
}

// Run all ops for a given type in one call
template <typename T, RedOp... Ops>
bool run_type(sycl::queue& q) { return (run_config<T, Ops>(q) & ...); }

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int, char**)
{
  srand(42);
  sycl::queue queue{sycl::gpu_selector_v};
  std::cout << "Running on device: "
            << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

  bool pass = true;
  printf("\n=== fred (float reduction) -- 8 configs ===\n");
  pass &= run_type<fp16,   RedOp::Add, RedOp::Min, RedOp::Max>(queue);
  pass &= run_type<bf16,   RedOp::Add, RedOp::Min, RedOp::Max>(queue);
  pass &= run_config<float,  RedOp::Add>(queue);
  pass &= run_config<double, RedOp::Add>(queue);

  printf("\n=== ired .32b (integer 32-bit reduction) -- 10 configs ===\n");
  pass &= run_type<int, RedOp::Add, RedOp::Smin, RedOp::Smax,
                        RedOp::And, RedOp::Or, RedOp::Xor,
                        RedOp::Incwrap, RedOp::Decwrap>(queue);
  pass &= run_type<uint32_t, RedOp::Umin, RedOp::Umax>(queue);

  printf("\n=== ired .64b (integer 64-bit reduction) -- 8 configs ===\n");
  pass &= run_type<long, RedOp::Add, RedOp::Smin, RedOp::Smax,
                         RedOp::And, RedOp::Or, RedOp::Xor>(queue);
  pass &= run_type<unsigned long, RedOp::Umin, RedOp::Umax>(queue);

  printf("\n=== Overall: %s ===\n", pass ? "ALL 26 PASSED" : "SOME FAILED");
  return pass ? 0 : 1;
}
