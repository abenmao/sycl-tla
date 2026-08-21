/***************************************************************************************************
* Copyright (C) 2025 Intel Corporation, All rights reserved.
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

// Sub-byte (FP4/e2m1) transposed 2D-block-load unit test.
//
// A small GEMM with layoutB = 'C' loads the B operand transposed, lowering to
// `load_block2d.ugm.d32t` - the same transposed load path exercised by the FMHA
// K load. The bf16 case is the correct-on-HW reference; the e2m1 case issues the
// identical d32t instruction over sub-byte packed data. Both must produce
// numerically correct results.

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/platform/platform.h"
#include "cutlass/util/sycl_event_manager.hpp"

#include "cutlass_unit_test.h"

#if defined(__clang__)
  #pragma clang diagnostic ignored "-Wpass-failed"
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace cute;
using namespace cutlass;

#if (IGC_VERSION_MAJOR > 2) || (IGC_VERSION_MAJOR == 2 && IGC_VERSION_MINOR >= 18)

// Variant of make_signed_t that works for both integer and floating point types.
template <typename T>
auto ensure_signed_helper_t() {
  if constexpr (cute::is_unsigned_v<T>)
    return cute::make_signed_t<T>{};
  else
    return T{};
}

template <typename T>
using ensure_signed_t = decltype(ensure_signed_helper_t<T>());

template <typename T>
T random_value()
{
  using Limits = cutlass::platform::numeric_limits<T>;

  static std::vector<T> saved;
  static constexpr size_t nsave = 65537;
  static size_t idx = 0;

  if (saved.empty()) {
    float range = Limits::is_integer ? 10.f : 1.f;
    float v_min = cute::max(-range, float(Limits::lowest()));
    float v_max = cute::min(+range, float(Limits::max()));

    saved.resize(nsave);
    for (auto &x: saved)
      x = T(v_min + (v_max - v_min) * (float(rand()) / float(RAND_MAX)));
  }

  auto v = saved[idx++];
  if (idx >= nsave) idx -= nsave;

  return v;
}

template <typename InTensor>
void random_fill(InTensor &X)
{
  using T = typename InTensor::element_type;
  for (int i = 0; i < size(X); i++)
    X(i) = random_value<T>();
}

template <typename InTensor>
void zero_fill(InTensor &X)
{
  using T = typename InTensor::element_type;
  for (int i = 0; i < size(X); i++)
    X(i) = T(0);
}

// Pack sub-byte types in a gmem tensor.
// On input, the backing array holds one sub-byte value per byte.
// On exit, the backing array contains packed values.
template <typename InTensor>
void subbyte_pack(InTensor &X)
{
  using T = typename InTensor::element_type;

  if constexpr (sizeof_bits_v<T> % 8 != 0) {
    static_assert(sizeof_bits_v<T> == 4, "Unsupported sub-byte data size");

    auto ptr = recast_ptr<uint8_t>(&*X.data());
    auto bytes = X.size();

    for (size_t i = 0; i < bytes/2; i++)
        ptr[i] = ptr[2*i] | (ptr[2*i + 1] << 4);
    if (bytes & 1)
        ptr[bytes >> 1] = ptr[bytes - 1];
  }
}

template <typename T, char LayoutKind>
auto make_shared_usm_tensor(sycl::queue &Q, int r, int c)
{
  auto ptr = make_gmem_ptr(sycl::malloc_shared<T>(r*c, Q));
  auto shape = make_shape(r, c);
  if constexpr (LayoutKind == 'C')
    return make_tensor(ptr, make_layout(shape, make_stride(_1{}, r)));
  else
    return make_tensor(ptr, make_layout(shape, make_stride(c, _1{})));
}

template <typename InTensor>
void free_usm_tensor(InTensor &X, sycl::queue &Q)
{
  sycl::free(&*X.data(), Q);
}

template <class ATensor, class BTensor, class CTensor,
          class TiledMMA>
void
gemm_device(ATensor   const& A,         // (M,K)
            BTensor   const& B,         // (N,K)
            CTensor        & C,         // (M,N)
            TiledMMA const & mma)
{
  // -----
  // Setup
  // -----

  /* Get workgroup and local IDs */
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto wg_n = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));

  /* Create proxy coordinate tensors for each global tensor */
  Tensor cA = make_identity_tensor(A.shape());   // (M,K)
  Tensor cB = make_identity_tensor(B.shape());   // (N,K)
  Tensor cC = make_identity_tensor(C.shape());   // (M,N)

  /* Split GEMM into workgroup tiles, and identify our workgroup's tile (wg_coord) */
  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m,_));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n,_));  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1,_1, X>{});       // (BLK_M,BLK_N)

  /* Create block 2D TiledCopies */
  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_D(mma, C);

  /* Slice TiledCopy/TiledMMA operations to thread (work-item) level */
  auto thr_mma    =    mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  /* Register fragments for MMA */
  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));

  /* Register fragments for copies */
  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

  /* Partition global tensor (proxies) for copies */
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  /* Partition C */
  Tensor tCrC = partition_fragment_C(mma, select<0,1>(wg_tile));
  Tensor tCgC = thr_mma.partition_C(gC);    /* also matches copy_c's source layout */

  /* Create prefetch TiledCopy instances */
  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  /* Partition global tensor (proxies) for prefetch */
  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  /* Prefetch distance, in units of k tiles */
  const int prefetch_dist = 3;

  // ------
  // Kernel
  // ------

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  /* Clear the accumulators */
  clear(tCrC);

  /* Warm up loops with prefetch to L1 */
  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
  }

  /* Main loop */
  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    /* Split barrier keeping threads loosely together */
    barrier_arrive(barrier_scope);

    /* Copy A/B from global memory (ideally L1 cache) to registers */
    copy(copy_a, tAgA(_,_,_,k_tile), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

    /* Prefetch A/B tiles to L1 */
    prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));

    /* Shuffle data from copy fragments to MMA fragments */
    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    /* Accumulate C += A * B */
    gemm(mma, tCrA, tCrB, tCrC);

    /* Other half of split barrier */
    barrier_wait(barrier_scope);
  }

  /* Write C to global memory */
  copy(copy_c, tCrC, tCgC);
}

template <typename TA, typename TB, typename TC>
auto
choose_mma_op()
{
  if constexpr (is_complete_v<XE_DPAS_TT<8, TC, TA, TB>>)
    return XE_DPAS_TT<8, TC, TA, TB>{};
  else if constexpr (is_same_v<TA, cute::bfloat16_t>)
    return XE_DPAS_TT<8, float, cute::bfloat16_t>{};
  else  /* Use f16 by default as upconversion sequences are typically faster */
    return XE_DPAS_TT<8, float, cute::half_t>{};
}

template <class ATensor, class BTensor, class CTensor>
auto
choose_tiled_mma(ATensor const& A, BTensor const& B, CTensor const&)
{
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  using TC = typename CTensor::element_type;

  auto op = choose_mma_op<TA,TB,TC>();

  constexpr bool byte = (cute::max(sizeof_bits_v<TA>, sizeof_bits_v<TB>) <= 8);
  constexpr bool a_t = is_constant_v<1, decltype(stride<0>(A))>;
  constexpr bool b_n = is_constant_v<1, decltype(stride<0>(B))>;

  constexpr bool use_1x_dpas_per_k = a_t                                  // Use one DPAS in k dimension for A^T case
                                  || (byte && b_n);                       //  pending compiler improvements (also int8 B^N).
  constexpr bool use_4x8_sg = ((sizeof_bits_v<TB> < sizeof_bits_v<TA>)    // Use smaller B loads for expensive reorders.
                                  && !(is_same_v<TB, cute::float_e5m2_t>))
                           || (b_n && sizeof_bits_v<TB> < 8);

  using _K = conditional_t<use_1x_dpas_per_k,
                           C<op.K>, C<op.K*2>>;

  using WGTile = Shape<_256, _256, _K>;                               // 256x256 WG tile size
  using SGLayout8x4 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;  // 8x4 SG tiling, n-major
  using SGLayout4x8 = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;  // 4x8 SG tiling, n-major
  using SGLayout = conditional_t<use_4x8_sg, SGLayout4x8, SGLayout8x4>;

  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;

  return MMA{};
}

template <class, class, class, char, char> class GemmCuteName;
template <class ATensor, class BTensor, class CTensor, typename TA, typename TB, typename TC, char layoutA, char layoutB>
void
gemm_cute(sycl::queue &Q,
          ATensor   const& A,         // (M,K)
          BTensor   const& B,         // (N,K)
          CTensor        & C)         // (M,N)
{
  auto mma = choose_tiled_mma(A, B, C);

  sycl::range<2> local = {size(mma), 1};
  sycl::range<2> global = {local[0] * ceil_div(shape<0>(B), get<1>(mma.tile_mnk())),
                           local[1] * ceil_div(shape<0>(A), get<0>(mma.tile_mnk()))};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props {
    syclex::sub_group_size<16>,
    intelex::grf_size<256>
  };

  auto event = Q.parallel_for<GemmCuteName<TA, TB, TC, layoutA, layoutB>>(sycl::nd_range<2>(global, local), kernel_props,
    [=](auto) {
      gemm_device(A, B, C, mma);
    }
  );

  EventManager::getInstance().addEvent(event);
}

template <class...> class GemmVerifyKernelName;
template <class ATensor, class BTensor, class CTensor>
bool
gemm_verify(sycl::queue &Q,
            ATensor const& A,         // (M,K)
            BTensor const& B,         // (N,K)
            CTensor const& C)         // (M,N)
{
  int m = size<0>(A);
  int n = size<0>(B);
  int k = size<1>(A);

  auto ok = sycl::malloc_shared<bool>(1, Q);
  *ok = true;

  Q.parallel_for<GemmVerifyKernelName<ATensor, BTensor, CTensor>>(sycl::range<2>(m, n), [=](sycl::item<2> id) {
    int i = id[0], j = id[1];

    using AccType = typename CTensor::element_type;
    using SignedAccType = ensure_signed_t<AccType>;

    auto c = AccType(0);
    for (int h = 0; h < k; h++) {
      c += AccType(A(i,h)) * AccType(B(j,h));
    }

    auto tol = AccType(1e-5f * k);
    if (std::abs(SignedAccType(c - AccType(C(i,j)))) > tol) {
      *ok = false;
    }
  }).wait();

  bool read_ok = *ok;

  sycl::free(ok, Q);

  return read_ok;
}

template <typename TA, typename TB, typename TC,
          char layoutA = 'R', char layoutB = 'C'>
bool
run_subbyte_gemm(sycl::queue &Q, int m, int n, int k)
{
  // Transpose B to match CuTe conventions (layoutB='C' -> transposed B load).
  constexpr char tlayoutB = layoutB ^ ('R' ^ 'C');

  auto A = make_shared_usm_tensor<TA,  layoutA>(Q, m, k);
  auto B = make_shared_usm_tensor<TB, tlayoutB>(Q, n, k);
  auto C = make_shared_usm_tensor<TC,      'R'>(Q, m, n);

  random_fill(A);
  random_fill(B);
  zero_fill(C);

  auto A_ref = make_shared_usm_tensor<float,  layoutA>(Q, m, k);
  auto B_ref = make_shared_usm_tensor<float, tlayoutB>(Q, n, k);

  copy(A, A_ref);
  copy(B, B_ref);

  subbyte_pack(A);
  subbyte_pack(B);

  gemm_cute<decltype(A), decltype(B), decltype(C), TA, TB, TC, layoutA, layoutB>(Q, A, B, C);
  Q.wait_and_throw();

  bool ok = gemm_verify(Q, A_ref, B_ref, C);

  free_usm_tensor(A, Q);
  free_usm_tensor(B, Q);
  free_usm_tensor(C, Q);
  free_usm_tensor(A_ref, Q);
  free_usm_tensor(B_ref, Q);

  return ok;
}

// Reference: bf16 transposed-B load (d32t) - correct on HW.
TEST(CuTe_Xe, XE_BLOCK2D_SUBBYTE_bf16_transposed) {
  sycl::queue Q = compat::get_default_queue();
  EXPECT_TRUE((run_subbyte_gemm<bfloat16_t, bfloat16_t, float, 'R', 'C'>(Q, 256, 256, 256)));
}

#if defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35)
// Sub-byte FP4 (e2m1) transposed-B load: identical d32t instruction, sub-byte packing.
TEST(CuTe_Xe, XE_BLOCK2D_SUBBYTE_e2m1_transposed) {
  sycl::queue Q = compat::get_default_queue();
  EXPECT_TRUE((run_subbyte_gemm<float_e2m1_t, float_e2m1_t, float, 'R', 'C'>(Q, 256, 256, 256)));
}
#else
TEST(CuTe_Xe, XE_BLOCK2D_SUBBYTE_e2m1_transposed) {
  GTEST_SKIP() << "float_e2m1_t case requires SYCL_INTEL_TARGET==35 (CRI) build. skipped";
}
#endif

#else

TEST(CuTe_Xe, XE_BLOCK2D_SUBBYTE_SKIPPED) {
  GTEST_SKIP() << "Sub-byte transposed 2D-block-load tests require IGC version 2.18 or higher. skipped";
}

#endif
