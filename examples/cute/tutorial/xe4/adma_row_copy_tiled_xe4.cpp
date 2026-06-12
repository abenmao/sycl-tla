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
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <cmath>

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gpu_generics.h>
#include <cute/atom/copy_traits_xe4_adma.hpp>
#include <cute/arch/xe4_util.hpp>
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;
using bfloat16 = sycl::ext::oneapi::bfloat16;

constexpr static size_t   kSmemAlignment    = 256;
constexpr static uint32_t WARP_SIZE = 32;
constexpr static uint32_t NUM_COLS_PER_COPY = 32;

// Matrix-descriptor parameters (palladium test: type1, 32x32B core tile).
constexpr static slm_matrix_type kCmType  = slm_matrix_type::type1;
constexpr static cm_size_t       kCmSize  = cm_size_t::cm_32x32B;

// DType-dependent matrix-descriptor / row-copy sizes.
template <class T>
constexpr uint32_t kCmSizeX_v      = get_width_in_bytes<kCmSize>() / uint32_t(sizeof(T));
template <class T>
constexpr uint32_t kAlignX_v       = kCmSizeX_v<T>;
template <class T>
constexpr uint32_t kMatrixStride_v =
    ((NUM_COLS_PER_COPY + kAlignX_v<T> - 1) / kAlignX_v<T>) * kAlignX_v<T>;

// Hardware row-copy size, in bytes (one row per lane).
template <class T>
constexpr uint32_t kRowSizeBytes_v = NUM_COLS_PER_COPY * uint32_t(sizeof(T));

// Per-WG row-size variants (COL_MULT widens the row each thread copies).
template <class T, uint32_t COL_MULT>
constexpr uint32_t kRowSizeBytesMult_v = NUM_COLS_PER_COPY * COL_MULT * uint32_t(sizeof(T));

template <class T, uint32_t COL_MULT>
constexpr uint32_t kMatrixStrideMult_v =
    ((NUM_COLS_PER_COPY * COL_MULT + kAlignX_v<T> - 1) / kAlignX_v<T>) * kAlignX_v<T>;

template <class T, uint32_t NRows, uint32_t NCols>
struct SharedStorage : cute::aligned_struct<kSmemAlignment, _0> {
  cute::array_aligned<T, NRows * NCols, kSmemAlignment> smem_A;
};

#define PRINT(x) print(#x ": "); print(x); print("\n");

// ADMA Tiled Row Copy (A32S)
// Algorithm: (M x K) DTYPE tensor (row-major) is split into (WG_SIZE x 1) blocks
// in workgroup-space, where each block copies a (NUM_WARPS * WARP_SIZE x NUM_COLS_PER_COPY)
// using Row-copy instructions
////////////////////////////////////////////////////////////////////////////////////////////////////
/// Device function — one workgroup copies a tile of A into SLM
/// then back out to C, using the TILED .a32s instructions.
///  * BlockIdx (wg_y, wg_x) selects the tile.
///  * Each lane (local_id_y) copies one row of the tile.
///  * gmem and smem tile tensors are built with `local_tile` from full-tensor views.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T, class TensorMA, class TensorMC, class TiledCopyLoad, class TiledCopyStore,
          class OffsetT = int32_t,
          uint32_t NUM_WARPS_K = 1, uint32_t COL_MULT_K = 1,
          bool PATTERN_ROW_COPY = true>
SYCL_EXTERNAL ALWAYS_INLINE void
adma_row_copy_tiled_device(sycl::nd_item<2> item,
                           TensorMA       mA,
                           TensorMC       mC,
                           TiledCopyLoad  adma_row_copy_tiled_load,
                           TiledCopyStore adma_row_copy_tiled_store)
{
  constexpr uint32_t kWgRows = WARP_SIZE * NUM_WARPS_K;
  constexpr uint32_t kWgCols = NUM_COLS_PER_COPY * COL_MULT_K;
  using SharedStorageType = SharedStorage<T, kWgRows, kWgCols>;

  auto     group   = item.get_group();
  auto     sg      = item.get_sub_group();
  uint32_t sg_id   = sg.get_group_id()[0];
  uint32_t sg_size = sg.get_local_range()[0];

  uint32_t wg_y       = item.get_group(0);
  uint32_t wg_x       = item.get_group(1);
  uint32_t local_id_y = item.get_local_id(0);
  uint32_t local_id_x = item.get_local_id(1);

  bool is_leader = (local_id_y == 0) && (local_id_x == 0);

  // ---- SLM staging buffer ----
  auto ptr_u8   = alloc_slm_buffer<uint8_t, sizeof(SharedStorageType)>(group);
  auto& smem    = *reinterpret_cast<SharedStorageType*>(ptr_u8);
  T*    slm_ptr = smem.smem_A.begin();

  // ---- Barriers (one for load, one for store) ----
  auto& load_abar  = cutlass::arch::allocate_cluster_tx_barrier();
  auto& store_abar = cutlass::arch::allocate_cluster_tx_barrier();

  // ---- Matrix descriptor over the SLM tile ----
  matrix_desc_t mat_slm_desc(slm_space_cast(slm_ptr), kMatrixStrideMult_v<T, COL_MULT_K>, kCmType);
  mat_slm_desc = mat_slm_desc + sg_id * sg_size * kWgCols * uint32_t(sizeof(T));

  // ---- Per-block GMEM / SMEM tiles ----
  // SmemLayout: a (kWgRows x kWgCols) row-major tile.
  auto sLayout = make_layout(make_shape (Int<kWgRows>{}, Int<kWgCols>{}),
                             make_stride(Int<kWgCols>{}, _1{}));
  Tensor sTile = make_tensor(make_smem_ptr(slm_ptr), sLayout);

  // Block-tile view of the full GMEM tensor.
  auto blk_shape = make_shape(Int<kWgRows>{}, Int<kWgCols>{});
  auto blk_coord = make_coord(wg_y, wg_x);
  Tensor gTileA = local_tile(mA, blk_shape, blk_coord);   // (NRows, NCols)
  Tensor gTileC = local_tile(mC, blk_shape, blk_coord);   // (NRows, NCols)

  auto thread_idx     = item.get_local_linear_id();

  // ---- Initialize barriers + load-phase transaction bytes ----
  constexpr uint32_t kTileBytes = kWgRows * kWgCols * sizeof(T);
  if (is_leader) {
    load_abar.init(1);
    store_abar.init(1);
    load_abar.arrive_and_expect_tx(kTileBytes);
  }
  sycl::group_barrier(group);

  uint32_t phase_bit = 0;

  if constexpr (!PATTERN_ROW_COPY) {
    // ---- Scalar (per-lane row) path ----
    Tensor gRowA = gTileA(local_id_y, _);
    Tensor gRowC = gTileC(local_id_y, _);
    Tensor sRow  = sTile (local_id_y, _);

    T const* mA_base     = raw_pointer_cast(mA.data());
    T const* gRowA_base  = mA_base;
    OffsetT  load_offset = OffsetT(int32_t(raw_pointer_cast(gRowA.data()) - mA_base) * sizeof(T));
    auto load_with = adma_row_copy_tiled_load.with(mat_slm_desc.get(),
                                    const_cast<T*>(gRowA_base),
                                    load_offset,
                                    kRowSizeBytesMult_v<T, COL_MULT_K>,
                                    reinterpret_cast<uint64_t*>(&load_abar));
    copy(load_with, gRowA, sRow);

    load_abar.try_wait(phase_bit);
    sycl::group_barrier(group);
    if (is_leader) {
      store_abar.arrive_and_expect_tx(kTileBytes);
    }
    sycl::group_barrier(group);

    T*       mC_base      = raw_pointer_cast(mC.data());
    T*       gRowC_base   = mC_base;
    OffsetT  store_offset = OffsetT(int32_t(raw_pointer_cast(gRowC.data()) - mC_base) * sizeof(T));
    auto store_with = adma_row_copy_tiled_store.with(mat_slm_desc.get(),
                                      gRowC_base,
                                      store_offset,
                                      kRowSizeBytesMult_v<T, COL_MULT_K>,
                                      reinterpret_cast<uint64_t*>(&store_abar));
    copy(store_with, sRow, gRowC);
  } else {
    // ---- Threaded (TiledCopy partition_S/partition_D) path ----
    auto thr_copy_load  = adma_row_copy_tiled_load.get_slice(thread_idx);
    auto thr_copy_store = adma_row_copy_tiled_store.get_slice(thread_idx);

    Tensor tGgA = thr_copy_load .partition_S(gTileA);
    Tensor tGsA = thr_copy_load .partition_D(sTile);
    Tensor tGsC = thr_copy_store.partition_D(sTile);
    Tensor tGgC = thr_copy_store.partition_S(gTileC);

    T const* mA_base     = raw_pointer_cast(mA.data());
    T const* gRowA_base  = mA_base;
    OffsetT  load_offset = OffsetT(int32_t(&tGgA(0) - mA_base) * int32_t(sizeof(T)));
    auto load_with = adma_row_copy_tiled_load.with(mat_slm_desc.get(),
                                          const_cast<T*>(gRowA_base),
                                          load_offset,
                                          kRowSizeBytesMult_v<T, COL_MULT_K>,
                                          reinterpret_cast<uint64_t*>(&load_abar));
    copy(load_with, tGgA, tGsA);

    load_abar.try_wait(phase_bit);
    sycl::group_barrier(group);
    if (is_leader) {
      store_abar.arrive_and_expect_tx(kTileBytes);
    }
    sycl::group_barrier(group);

    T*       mC_base      = raw_pointer_cast(mC.data());
    T*       gRowC_base   = mC_base;
    OffsetT  store_offset = OffsetT(int32_t(&tGgC(0) - mC_base) * int32_t(sizeof(T)));
    auto store_with = adma_row_copy_tiled_store.with(mat_slm_desc.get(),
                                            gRowC_base,
                                            store_offset,
                                            kRowSizeBytesMult_v<T, COL_MULT_K>,
                                            reinterpret_cast<uint64_t*>(&store_abar));
    copy(store_with, tGsC, tGgC);
  }

  store_abar.try_wait(phase_bit);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Host helper — builds traits/atoms and launches the device kernel.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T, class OffsetT = int32_t,
          uint32_t NUM_WARPS_K = 1, uint32_t COL_MULT_K = 1,
          bool PATTERN_ROW_COPY = true>
void
adma_row_copy_tiled_A_k_major(uint32_t M, uint32_t K,
                              T const*       d_input,
                              T*             d_output,
                              sycl::queue&   queue)
{
  // Per-WG tile geometry mirrors the kernel.
  constexpr uint32_t kWgRows = WARP_SIZE * NUM_WARPS_K;
  constexpr uint32_t kWgCols = NUM_COLS_PER_COPY * COL_MULT_K;

  static_assert(kRowSizeBytesMult_v<T, COL_MULT_K> >= 16 && kRowSizeBytesMult_v<T, COL_MULT_K> <= 2048,
                "RowSize must be in [16, 2048]");
  static_assert((kRowSizeBytesMult_v<T, COL_MULT_K> & (kRowSizeBytesMult_v<T, COL_MULT_K> - 1)) == 0,
                "RowSize must be a power of 2");


  // Full row-major (M, K) gmem tensors. Stride = (K, 1).
  auto gShape  = make_shape(M, K);
  auto gStride = make_stride(K, uint32_t(1));
  Tensor mA = make_tensor(make_gmem_ptr(d_input),  make_layout(gShape, gStride));
  Tensor mC = make_tensor(make_gmem_ptr(d_output), make_layout(gShape, gStride));

  // -------------------- Traits + Atoms --------------------
  using LoadOp     = cute::XE4_ADMA_ROW_COPY_TILED_LOAD;
  using StoreOp    = cute::XE4_ADMA_ROW_COPY_TILED_STORE;
  using LoadTraits = cute::Copy_Traits<LoadOp,  T,
                                       cute::Int<kRowSizeBytesMult_v<T, COL_MULT_K>>,
                                       cute::Int<kRowSizeBytesMult_v<T, COL_MULT_K>>,
                                       cute::C<PATTERN_ROW_COPY>>;
  using StoreTraits= cute::Copy_Traits<StoreOp, T,
                                       cute::Int<kRowSizeBytesMult_v<T, COL_MULT_K>>,
                                       cute::Int<kRowSizeBytesMult_v<T, COL_MULT_K>>,
                                       cute::C<PATTERN_ROW_COPY>>;

  Copy_Atom<LoadTraits,  T> load_atom {};
  Copy_Atom<StoreTraits, T> store_atom{};

  // -------------------- TiledCopy --------------------
  // Thread layout: WG_SIZE threads in the row dim, 1 in the col dim.
  // Value  layout: each thread copies (1 x NUM_COLS_PER_COPY) elements (one row).

  auto adma_row_copy_tiled_load = make_tiled_copy(
      load_atom,
      Layout<Shape<Int<kWgRows>, _1>>{},
      Layout<Shape<_1, Int<kWgCols>>>{});
  auto adma_row_copy_tiled_store = make_tiled_copy(
      store_atom,
      Layout<Shape<Int<kWgRows>, _1>>{},
      Layout<Shape<_1, Int<kWgCols>>>{});

#if 0
  PRINT(adma_row_copy_tiled_load);
  PRINT(adma_row_copy_tiled_store);
#endif

  // -------------------- Launch configuration --------------------
  // Each workgroup covers one (kWgRows x kWgCols) tile.
  // WG-grid: (row_wgs, col_wgs); local range: (kWgRows, 1).
  uint32_t row_wgs = (M + kWgRows - 1) / kWgRows;
  uint32_t col_wgs = (K + kWgCols - 1) / kWgCols;

  sycl::range<2> local_range (kWgRows, 1);
  sycl::range<2> global_range(row_wgs * kWgRows, col_wgs * 1);
  sycl::nd_range<2> nd_rng(global_range, local_range);

  std::cout << "global_range = (" << global_range.get(0) << ", " << global_range.get(1) << ")\n"
            << "local_range  = (" << local_range.get(0)  << ", " << local_range.get(1)  << ")"
            << std::endl;
  
  queue.submit([&](sycl::handler& h) {
    h.parallel_for(nd_rng, [=](sycl::nd_item<2> item) ALWAYS_INLINE {
      adma_row_copy_tiled_device<T, decltype(mA), decltype(mC),
                                 decltype(adma_row_copy_tiled_load),
                                 decltype(adma_row_copy_tiled_store),
                                 OffsetT,
                                 NUM_WARPS_K, COL_MULT_K,
                                 PATTERN_ROW_COPY>(
          item, mA, mC,
          adma_row_copy_tiled_load,
          adma_row_copy_tiled_store);
    });
  }).wait();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Templated test runner — runs the ADMA row-copy test for a given DType.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T, class OffsetT = int32_t,
          uint32_t NUM_WARPS_K = 1, uint32_t COL_MULT_K = 1,
          bool PATTERN_ROW_COPY = true>
int
run_adma_row_copy_tiled_test(uint32_t M, uint32_t K,
                             char const* dtype_name,
                             char const* mode_name = "A32S")
{
  uint32_t in_strides [2] = { K, 1 };  // (K, 1) row-major
  uint32_t out_strides[2] = { K, 1 };

  uint32_t input_size  = M * in_strides [0];
  uint32_t output_size = M * out_strides[0];

  std::cout << "Running ADMA Row Copy TILED (" << mode_name << ", dtype=" << dtype_name
            << ", NUM_WARPS_K=" << NUM_WARPS_K
            << ", COL_MULT_K="  << COL_MULT_K
            << ", PATTERN_ROW_COPY=" << PATTERN_ROW_COPY
            << ") for shape (" << M << " , " << K << ")" << std::endl;
  T* h_input  = new T[input_size];
  T* h_output = new T[output_size];
  T* h_dst    = new T[output_size];

  for (uint32_t i = 0; i < input_size; ++i) {
    h_input[i] = T(-1.0f);
  }
  std::memset(h_output, 0, output_size * sizeof(T));

  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < K; ++j) {
      uint32_t in_idx  = i * in_strides [0] + j * in_strides [1];
      uint32_t out_idx = i * out_strides[0] + j * out_strides[1];
      h_input [in_idx]  = T(float(in_idx));
      h_output[out_idx] = h_input[in_idx];   // expected result
    }
  }

  sycl::queue queue(sycl::default_selector_v);
  std::cout << "Using device: "
            << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

  auto d_input  = sycl::malloc_device<T>(input_size,  queue);
  auto d_output = sycl::malloc_device<T>(output_size, queue);

  queue.memcpy(d_input, h_input, input_size * sizeof(T)).wait();
  queue.memset(d_output, 0,      output_size * sizeof(T)).wait();

  // -------------------- Launch via host helper --------------------
  adma_row_copy_tiled_A_k_major<T, OffsetT, NUM_WARPS_K, COL_MULT_K, PATTERN_ROW_COPY>(M, K, d_input, d_output, queue);

  queue.memcpy(h_dst, d_output, output_size * sizeof(T)).wait();

  // -------------------- Verify --------------------
  uint32_t err_cnt = 0;
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < K; ++j) {
      uint32_t idx = i * out_strides[0] + j * out_strides[1];
      using CmpT = std::conditional_t<std::is_integral_v<T>, int32_t, float>;
      if (CmpT(h_dst[idx]) != CmpT(h_output[idx])) {
        if (err_cnt < 10) {
          std::cout << "Verification failed at (" << i << ", " << j << "): expected "
                    << CmpT(h_output[idx]) << ", got " << CmpT(h_dst[idx]) << std::endl;
        }
        ++err_cnt;
      }
    }
  }

  if (err_cnt == 0) std::cout << "\nADMA Row Copy TILED (" << mode_name << ", " << dtype_name << "): PASSED" << std::endl;
  else              std::cout << "\nADMA Row Copy TILED (" << mode_name << ", " << dtype_name << "): FAILED ("
                              << err_cnt << " mismatches)" << std::endl;

  sycl::free(d_input,  queue);
  sycl::free(d_output, queue);
  delete[] h_input;
  delete[] h_output;
  delete[] h_dst;

  return err_cnt == 0 ? 0 : 1;
}

template <uint32_t NUM_ROW_WG, uint32_t MULTIPLE_OF_COL_SIZE_32,
          uint32_t NUM_WARPS = 1, uint32_t NUM_COL_WG = 1,
          bool PATTERN_ROW_COPY = true>
int run_all_dtype_mode_combos_for_shape()
{
#if 0
  static_assert(NUM_ROW_WG >= 1 && NUM_ROW_WG <= 16 &&
                (NUM_ROW_WG & (NUM_ROW_WG - 1)) == 0,
                "NUM_ROW_WG must be a power of 2 in [1, 16]");
  static_assert(MULTIPLE_OF_COL_SIZE_32 >= 1 && MULTIPLE_OF_COL_SIZE_32 <= 4 &&
                (MULTIPLE_OF_COL_SIZE_32 & (MULTIPLE_OF_COL_SIZE_32 - 1)) == 0,
                "MULTIPLE_OF_COL_SIZE_32 must be a power of 2 in [1, 4]");
  static_assert(NUM_WARPS >= 1 && NUM_WARPS <= 16 &&
                (NUM_WARPS & (NUM_WARPS - 1)) == 0,
                "NUM_WARPS must be a power of 2 in [1, 16]");
  static_assert(NUM_COL_WG >= 1 && NUM_COL_WG <= 16 &&
                (NUM_COL_WG & (NUM_COL_WG - 1)) == 0,
                "NUM_COL_WG must be a power of 2 in [1, 16]");
#endif

  uint32_t M = NUM_ROW_WG              * WARP_SIZE * NUM_WARPS;
  uint32_t K = MULTIPLE_OF_COL_SIZE_32 * NUM_COLS_PER_COPY * NUM_COL_WG;

  std::cout << "\n================================================================\n"
            << " Shape sweep: NUM_ROW_WG=" << NUM_ROW_WG
            << ", MULTIPLE_OF_COL_SIZE_32=" << MULTIPLE_OF_COL_SIZE_32
            << ", NUM_WARPS=" << NUM_WARPS
            << ", NUM_COL_WG=" << NUM_COL_WG
            << ", PATTERN_ROW_COPY=" << PATTERN_ROW_COPY
            << "  =>  (M=" << M << ", K=" << K << ")\n"
            << "================================================================" << std::endl;

  int fails = 0;

  // Per-WG tile = (WARP_SIZE * NUM_WARPS) x (NUM_COLS_PER_COPY * MULTIPLE_OF_COL_SIZE_32).
  // WG_SIZE = WARP_SIZE * NUM_WARPS (one lane per row).
  // ---- A32S (int32_t offset) ----
  fails += run_adma_row_copy_tiled_test<bfloat16, int32_t,  NUM_WARPS, MULTIPLE_OF_COL_SIZE_32, PATTERN_ROW_COPY>(M, K, "bf16",  "A32S");
  fails += run_adma_row_copy_tiled_test<float,    int32_t,  NUM_WARPS, MULTIPLE_OF_COL_SIZE_32, PATTERN_ROW_COPY>(M, K, "float", "A32S");
  fails += run_adma_row_copy_tiled_test<int8_t,   int32_t,  NUM_WARPS, MULTIPLE_OF_COL_SIZE_32, PATTERN_ROW_COPY>(M, K, "int8",  "A32S");

  // ---- A32U (uint32_t offset) ----
  fails += run_adma_row_copy_tiled_test<bfloat16, uint32_t, NUM_WARPS, MULTIPLE_OF_COL_SIZE_32, PATTERN_ROW_COPY>(M, K, "bf16",  "A32U");
  fails += run_adma_row_copy_tiled_test<float,    uint32_t, NUM_WARPS, MULTIPLE_OF_COL_SIZE_32, PATTERN_ROW_COPY>(M, K, "float", "A32U");
  fails += run_adma_row_copy_tiled_test<int8_t,   uint32_t, NUM_WARPS, MULTIPLE_OF_COL_SIZE_32, PATTERN_ROW_COPY>(M, K, "int8",  "A32U");

  return fails;
}

template <bool PATTERN_ROW_COPY>
int run_full_sweep()
{
  std::cout << "\n################################################################\n"
            << "##  FULL SWEEP with PATTERN_ROW_COPY = " << PATTERN_ROW_COPY
            << "  (" << (PATTERN_ROW_COPY ? "threaded TiledCopy path" : "scalar per-lane path") << ")\n"
            << "################################################################" << std::endl;

  int fails = 0;
  // ---- Full sweep: NUM_ROW_WG ∈ {1,2,4,8,16} × MULTIPLE_OF_COL_SIZE_32 ∈ {1,2,4} ----
  //  Template args: <NUM_ROW_WG, MULTIPLE_OF_COL_SIZE_32, NUM_WARPS, NUM_COL_WG, PATTERN_ROW_COPY>
#if 0 // reducing the set of tests running on CI
  fails += run_all_dtype_mode_combos_for_shape<1,  1, 1, 1, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<1,  2, 1, 1, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<1,  4, 1, 1, PATTERN_ROW_COPY>();

  fails += run_all_dtype_mode_combos_for_shape<2,  1, 1, 1, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<2,  2, 1, 1, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<2,  4, 1, 1, PATTERN_ROW_COPY>();

  fails += run_all_dtype_mode_combos_for_shape<4,  1, 1, 1, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<4,  2, 1, 1, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<4,  4, 1, 1, PATTERN_ROW_COPY>();

  fails += run_all_dtype_mode_combos_for_shape<8,  1, 1, 1, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<8,  1, 1, 2, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<8,  1, 1, 4, PATTERN_ROW_COPY>();

  fails += run_all_dtype_mode_combos_for_shape<16, 1, 1, 1, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<16, 1, 1, 2, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<16, 1, 1, 4, PATTERN_ROW_COPY>();
#endif
  fails += run_all_dtype_mode_combos_for_shape<1, 1, 2, 1, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<1, 2, 4, 2, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<2, 1, 2, 2, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<2, 4, 4, 4, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<4, 8, 4, 4, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<5, 8, 4, 7, PATTERN_ROW_COPY>();
  fails += run_all_dtype_mode_combos_for_shape<8, 4, 4, 3, PATTERN_ROW_COPY>();

  return fails;
}

template <bool PATTERN_ROW_COPY>
int run_cli_single_shape(uint32_t M, uint32_t K)
{
  std::cout << "\n################################################################\n"
            << "##  Single-shape (CLI) run with PATTERN_ROW_COPY = " << PATTERN_ROW_COPY
            << "  (M=" << M << ", K=" << K << ")\n"
            << "################################################################" << std::endl;

  int fails = 0;
  fails += run_adma_row_copy_tiled_test<bfloat16, int32_t,  1, 1, PATTERN_ROW_COPY>(M, K, "bf16",  "A32S");
  fails += run_adma_row_copy_tiled_test<float,    int32_t,  1, 1, PATTERN_ROW_COPY>(M, K, "float", "A32S");
  fails += run_adma_row_copy_tiled_test<int8_t,   int32_t,  1, 1, PATTERN_ROW_COPY>(M, K, "int8",  "A32S");
  fails += run_adma_row_copy_tiled_test<bfloat16, uint32_t, 1, 1, PATTERN_ROW_COPY>(M, K, "bf16",  "A32U");
  fails += run_adma_row_copy_tiled_test<float,    uint32_t, 1, 1, PATTERN_ROW_COPY>(M, K, "float", "A32U");
  fails += run_adma_row_copy_tiled_test<int8_t,   uint32_t, 1, 1, PATTERN_ROW_COPY>(M, K, "int8",  "A32U");
  return fails;
}

int main(int argc, char** argv)
{
  int fails = 0;

  if (argc > 2) {
    // ---- CLI-driven single-shape mode (run for both PATTERN_ROW_COPY values) ----
    uint32_t M = std::atoi(argv[1]);
    uint32_t K = std::atoi(argv[2]);

    fails += run_cli_single_shape<false>(M, K);
    fails += run_cli_single_shape<true >(M, K);
  } else {
    // ---- Full sweep, run twice: once per PATTERN_ROW_COPY value ----
    fails += run_full_sweep<false>();
    fails += run_full_sweep<true >();
  }

  if (fails == 0) std::cout << "\n✅ All ADMA Row Copy TILED tests PASSED" << std::endl;
  else            std::cout << "\n❌ " << fails << " ADMA Row Copy TILED test(s) FAILED" << std::endl;

  return fails == 0 ? 0 : 1;
}
