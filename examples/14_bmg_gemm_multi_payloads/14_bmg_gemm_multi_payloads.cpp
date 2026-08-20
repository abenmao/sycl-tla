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
/*! \file
    \brief CUTLASS Intel XE GEMM example using the multi-payload Xe Block2D APIs.

    This example constructs and executes a CuTe GEMM kernel on Intel hardware whose mainloop
    issues all of its global memory traffic through the multi-payload Xe Block2D API, and verifies
    its correctness with a reference implementation (cutlass::reference::device::GemmComplex). The
    example also provides a performance measurement for the GEMM in TFLOPS.

    Xe 2D block messages are issued from an "address payload": a small register-resident descriptor
    holding the base pointer, the surface width/height/pitch, and the block's x/y offset.

    The classic CuTe entry points (`cute::copy` / `cute::prefetch` on a block 2D TiledCopy) keep a
    single payload per TiledCopy and rewrite its x/y offset immediately before every message. That
    is convenient, but it serializes the messages: each send must wait for the previous send to read
    the shared payload register before the offset can be overwritten.

    The multi-payload API instead materializes one payload per copy atom, up front:

      prepare_payloads(tiled_copy, src_coord_tensor)
          -> Xe2DPreparedPayloads<N, BaseT>, one payload per atom
      copy(tiled_copy, prepared, dst_fragment)
          -> issues all N loads back-to-back, with no address setup in between
      prefetch(tiled_copy, prepared)
          -> the same, for prefetch messages
      prepared += delta
          -> advances every payload by `delta`, expressed in the copy's tensor coordinate space

    The step is expressed as a coordinate because the mode being
    walked is not always the same block 2D dimension. This example is deliberately fixed to
    row-major operands so a single kernel exercises both directions: A is (M,K) with K contiguous,
    so a K step moves the block 2D x offset, while B is viewed as (N,K) with N contiguous, so the
    same K step moves the y offset. The copy traits resolve which is which, so the caller only has
    to say "advance by one k tile".

    The C store still uses `cute::copy`: the multi-payload API covers loads and prefetches only, and
    the epilogue writes each accumulator tile exactly once, so there is nothing to amortize.

    The shapes of the A and B matrices are defined at runtime by `options.m`, `.n` and `.k`. The tile
    shape, which defines how much work is executed by a single work-group, is defined at compile
    time by:
    ```
      using TileShape = Shape<_256, _256, _32>;
    ```
    That is, each work-group processes a tile of M=256, N=256, and iterates over `options.k` in
    blocks of K=32.

    To build & run this example (from your build dir):

      $ ninja 14_bmg_gemm_multi_payloads
      $ ./examples/14_bmg_gemm_multi_payloads/14_bmg_gemm_multi_payloads

    Call with `--help` for information about available options
*/

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "sycl_common.hpp"
#include "helper.h"

// The kernel is launched with sub-group size / GRF size properties, which still requires the
// deprecated parallel_for overload.
#if defined(__clang__)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, iterations, verify;
  // The `verify` controls whether verification will be executed, which is true
  // by default. Users can skip the verification step by specifying it with 0.

  Options():
    help(false),
    error(false),
    m(5120), n(4096), k(4096), iterations(20), verify(1)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "BMG GEMM Multi-Payload Example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n\n"
      << "  --iterations=<int>          Iterations\n\n"
      << "  --verify=<int>              Specify whether to verify.\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Number of k tiles the prefetch runs ahead of the loads.
static constexpr int PrefetchDistance = 3;

/// Perform a workgroup-scoped matrix multiply-accumulate driven by prepared block 2D payloads.
template <class ATensor, class BTensor, class CTensor, class TiledMma>
CUTLASS_DEVICE void
gemm_multi_payload_device(ATensor  const& A,     // (M,K)
                          BTensor  const& B,     // (N,K)
                          CTensor       & C,     // (M,N)
                          TiledMma const& mma)
{
  //
  // Setup
  //

  auto item     = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m     = int(item.get_group(1));
  auto wg_n     = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));

  /* Create proxy coordinate tensors for each global tensor */
  Tensor cA = make_identity_tensor(A.shape());   // (M,K)
  Tensor cB = make_identity_tensor(B.shape());   // (N,K)
  Tensor cC = make_identity_tensor(C.shape());   // (M,N)

  /* Split the GEMM into workgroup tiles and identify this workgroup's tile */
  auto wg_tile  = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m,_));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n,_));  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1,_1, X>{});       // (BLK_M,BLK_N)

  /* Create block 2D TiledCopies */
  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_D(mma, C);

  /* Slice TiledCopy/TiledMMA operations down to work-item level */
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
  Tensor tCgC = thr_mma.partition_C(gC);

  /* Create prefetch TiledCopy instances */
  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  /* Partition global tensor (proxies) for prefetch */
  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  //
  // Payload preparation
  //

  /* One k tile is a (0, BLK_K) step in both A's (M,K) and B's (N,K) coordinate space. Whether that
     lands on the block 2D x or y offset is resolved from each copy's traits. */
  static constexpr auto SG_K = get<2>(wg_tile);

  /* Build one payload per copy atom. The source coordinate tensor fixes how many atoms there are. */
  auto prepared_a  = prepare_payloads(copy_a, tAgA(_,_,_,0));
  auto prepared_b  = prepare_payloads(copy_b, tBgB(_,_,_,0));

  /* Prefetches have no destination, so only the coordinate tensor is needed. */
  auto prepared_pa = prepare_payloads(prefetch_a, pAgA(_,_,_,0));
  auto prepared_pb = prepare_payloads(prefetch_b, pBgB(_,_,_,0));

  //
  // Mainloop
  //

  constexpr SPIRVScope barrier_scope = ScopeWorkgroup;

  int const k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));

  /* Clear the accumulators */
  clear(tCrC);

  /* Warm up loops with prefetch to L1 */
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < PrefetchDistance; i++, prepared_pa += SG_K, prepared_pb += SG_K) {
    prefetch(prefetch_a, prepared_pa);
    prefetch(prefetch_b, prepared_pb);
  }

  /* Main loop */
  for (int k_tile = 0; k_tile < k_tile_count; k_tile++,
       prepared_a += SG_K, prepared_b += SG_K,
       prepared_pa += SG_K, prepared_pb += SG_K) {
    /* Split barrier keeping threads loosely together */
    barrier_arrive(barrier_scope);

    /* Copy A/B from global memory (ideally L1 cache) to registers. All loads for an operand issue
       back-to-back: each one owns a payload, so no address setup separates them. */
    copy(copy_a, prepared_a, tArA);
    copy(copy_b, prepared_b, tBrB);

    /* Prefetch A/B tiles to L1 */
    prefetch(prefetch_a, prepared_pa);
    prefetch(prefetch_b, prepared_pb);

    /* Shuffle data from copy fragments to MMA fragments */
    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    /* Accumulate C += A * B */
    cute::gemm(mma, tCrA, tCrB, tCrC);

    /* Other half of split barrier */
    barrier_wait(barrier_scope);
  }

  /* Write C to global memory */
  copy(copy_c, tCrC, tCgC);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

struct ExampleRunner {

  // The code section below describes datatype for input, output matrices and computation between
  // elements in input matrices.
  using ElementAccumulator = float;      // <- data type of accumulator
  using ElementA = bfloat16_t;           // <- data type of elements in input matrix A
  using ElementB = bfloat16_t;           // <- data type of elements in input matrix B
  using ElementOutput = float;           // <- data type of elements in output matrix D

  // The mainloop walks K along the block 2D x offset for A and along the y offset for B, which
  // requires A to be K-major and B to be N-major.
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;

  // Workgroup-level tile
  using TileShape = Shape<_256, _256, _32>;

  // A TiledMMA struct defines a tiling of an MMA atom over M, N and K, combining both additional
  // hardware (sub-groups for Intel Xe) and iterations by each sub-group.
  using TiledMma = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>,
                                           Layout<TileShape>,
                                           Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  static constexpr int SubgroupSize = 16;

  //
  // Data members
  //

  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementOutput> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D; // Reference GEMM result for verification

  //
  // Methods
  //

  bool verify(const Options& options) {
    int M = options.m, N = options.n, K = options.k;

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutC::packed({M, N}));

    // This example has no epilogue, so the reference is a plain D = A * B.
    cutlass::reference::device::GemmComplex(
          {M, N, K},
          ElementAccumulator(1),
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          ElementAccumulator(0),
          ref_C,
          ref_D,
          ElementAccumulator(0)
        );

    // CUTLASS on SYCL uses the compatibility library compat for e.g. default in-order queue
    compat::wait();

    // Check if output from the CuTe kernel and the reference kernel are equal or not
    ElementOutput const epsilon(1e-2f);
    ElementOutput const non_zero_floor(1e-4f);
    return cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D.get(), block_D.size(), epsilon, non_zero_floor);
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const Options& options) {
    int M = options.m, N = options.n, K = options.k;

    block_A.reset(static_cast<std::size_t>(M) * K);
    block_B.reset(static_cast<std::size_t>(K) * N);
    block_C.reset(static_cast<std::size_t>(M) * N);
    block_D.reset(static_cast<std::size_t>(M) * N);
    block_ref_D.reset(static_cast<std::size_t>(M) * N);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
  }

  /// Build the views the mainloop expects: A is (M,K), B is (N,K) and D is (M,N).
  auto make_tensors(const Options& options) {
    int M = options.m, N = options.n, K = options.k;

    auto mA = make_tensor(make_gmem_ptr(block_A.get()),
                          make_layout(make_shape(M, K), make_stride(K, _1{})));
    auto mB = make_tensor(make_gmem_ptr(block_B.get()),
                          make_layout(make_shape(N, K), make_stride(_1{}, N)));
    auto mD = make_tensor(make_gmem_ptr(block_D.get()),
                          make_layout(make_shape(M, N), make_stride(N, _1{})));

    return cute::make_tuple(mA, mB, mD);
  }

  template <class ATensor, class BTensor, class DTensor>
  void launch(sycl::queue& Q, ATensor const& mA, BTensor const& mB, DTensor const& mD) {
    TiledMma mma;
    auto wg_tile = mma.tile_mnk();

    sycl::range<2> local  = {size(mma), 1};
    sycl::range<2> global = {local[0] * ceil_div(size<0>(mB), get<1>(wg_tile)),
                             local[1] * ceil_div(size<0>(mA), get<0>(wg_tile))};

    namespace syclex  = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    syclex::properties kernel_props {
      syclex::sub_group_size<SubgroupSize>,
      intelex::grf_size<256>
    };

    auto event = Q.parallel_for(sycl::nd_range<2>(global, local), kernel_props,
      [=](auto) {
        auto D = mD;
        gemm_multi_payload_device(mA, mB, D, TiledMma{});
      }
    );

    EventManager::getInstance().addEvent(event);
  }

  cutlass::Status run(const Options& options) {
    initialize(options);

    sycl::queue Q = compat::get_default_queue();

    auto [mA, mB, mD] = make_tensors(options);

    // Run the GEMM
    launch(Q, mA, mB, mD);

    compat::wait();

    if (options.verify != 0) {
      // Verify that the result is correct
      bool passed = verify(options);
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

      if (!passed) return cutlass::Status::kErrorInternal;
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        launch(Q, mA, mB, mD);
      }
      compat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tflops = (2.0 * options.m * options.n * options.k) * 1e-12;
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
      printf("Cutlass GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);
    }

    return cutlass::Status::kSuccess;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char** argv)
{
  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  //
  // Run example
  //

  ExampleRunner runner;

  CUTLASS_CHECK(runner.run(options));

  return 0;
}
