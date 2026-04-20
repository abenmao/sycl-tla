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
#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gpu_generics.h>
#include <cute/atom/copy_traits_xe4_adma.hpp>
#include <cute/arch/xe4_util.hpp>
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "../../../common/sycl_cute_common.hpp"

using namespace cute;
using sycl::ext::oneapi::this_work_item::get_nd_item;

constexpr static size_t kSmemAlignment = 256;
constexpr static size_t kNumThreadsPerWarp = 32; // Number of lanes.

template <class ElementA, class SmemLayoutA>
struct SharedStorage : cute::aligned_struct<kSmemAlignment, _0> {
  cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>, kSmemAlignment> smem_A;
};

inline void xe4_syncthreads(sycl::nd_item<3>& item) {
  sycl::group_barrier(item.get_group());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Kernel function — one row per lane. One full CTA spans exactly 32 rows (one per lane).
///
///   1. Problem size (bytes) = prob_size * sizeof(TA).
///   2. CTA size is fixed: kCtaBytes = 32 * RowSize. num_ctas = ceil(problem_bytes / kCtaBytes).
///      The last CTA may be partial.
///   3. In the last (partial) CTA, cta_bytes / RowSize = full-row lanes, any remainder in
///      [0, RowSize) makes the next lane partial, and the rest are inactive.
///
/// Every active lane's gmem offset is (cta_base_bytes + lane_id * RowSize), always
/// RowSize-aligned.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <AddressingMode Mode, uint32_t RowSize, typename T,
          typename LoadOp, typename StoreOp,
          typename SmemLayout,
          detail::CacheCtrl LoadCC = detail::CacheCtrl::L2c_L3uc,
          detail::FillMethod LoadFM = detail::FillMethod::Zero,
          detail::CacheCtrl StoreCC = detail::CacheCtrl::L2wb_L3uc>
SYCL_EXTERNAL ALWAYS_INLINE void adma_row_per_lane_kernel(
    sycl::nd_item<3> item,
    T const* src_ptr,
    T*       dst_ptr,
    uint32_t prob_size,
    SmemLayout sL)
{
  // Bit-based math so sub-byte types (e.g., fp4 = 4 bits) compute element counts correctly.
  constexpr uint32_t kTBits = cute::sizeof_bits_v<T>;
  static_assert((RowSize * 8) % kTBits == 0, "RowSize (in bits) must be a multiple of element bits");

  constexpr uint32_t kRowElements  = (RowSize * 8) / kTBits;
  constexpr uint32_t kCtaBytes     = kNumThreadsPerWarp * RowSize;
  constexpr uint32_t kCopyElements = kNumThreadsPerWarp * kRowElements;

  // 1. Problem in bytes. For sub-byte T, prob_size*bits may round up.
  uint32_t total_bytes = (prob_size * kTBits + 7u) / 8u;

  // 2. This CTA's byte span. Last CTA may be partial.
  uint32_t cta_base_bytes = BlockIdxX() * kCtaBytes;
  uint32_t cta_bytes = kCtaBytes;

  if (BlockIdxX() == GridDimX() - 1){
    cta_bytes = total_bytes - cta_base_bytes;
  }

  // 3. Divide CTA bytes by RowSize -> full-row lanes + optional partial lane.
  uint32_t full_rows    = cta_bytes / RowSize;
  uint32_t tail_bytes   = cta_bytes % RowSize;
  uint32_t active_lanes = full_rows + (tail_bytes > 0 ? 1u : 0u);

  uint32_t warp_idx = get_sg_id();
  auto sg = item.get_sub_group();
  uint32_t lane_id = sg.get_local_id();

  // 4. Per-lane role.
  bool     lane_active       = (lane_id < active_lanes);
  bool     is_partial_lane   = (tail_bytes > 0) && (lane_id == full_rows);
  uint32_t lane_bytes        = is_partial_lane ? tail_bytes : uint32_t(RowSize);
  uint32_t lane_elem_offset  = BlockIdxX() * kCopyElements + lane_id * kRowElements;
  uint32_t barrier_txn_bytes = active_lanes * uint32_t(RowSize);

  using OffsetType = std::conditional_t<Mode == AddressingMode::A64, uint64_t,
                     std::conditional_t<Mode == AddressingMode::A32S, int32_t, uint32_t>>;
  using SharedStorageType = SharedStorage<T, SmemLayout>;
  auto ptr = alloc_slm_buffer<uint8_t, sizeof(SharedStorageType)>(item.get_group());
  auto& smem = *reinterpret_cast<SharedStorageType*>(ptr);

  Tensor sL_local = make_tensor(make_smem_ptr(smem.smem_A.begin()), sL);

  using GmemTiledCopyLoad  = cute::Copy_Traits<LoadOp,  T, cute::Int<RowSize>, cute::Int<RowSize>>;
  using GmemTiledCopyStore = cute::Copy_Traits<StoreOp, T, cute::Int<RowSize>, cute::Int<RowSize>>;
  auto adma_load  = make_tiled_copy(Copy_Atom<GmemTiledCopyLoad,  T>{}, Layout<_1>{}, Layout<Int<kRowElements>>{});
  auto adma_store = make_tiled_copy(Copy_Atom<GmemTiledCopyStore, T>{}, Layout<_1>{}, Layout<Int<kRowElements>>{});

  uint32_t elect_one_thr = cute::elect_one_sync();
  auto load_abar  = allocate_abar<0>();
  auto store_abar = allocate_abar<1>();

  if (warp_idx == 0 && elect_one_thr) {
    xe4_initialize_barrier(load_abar[0], 1);
  } else if (warp_idx == 1 && elect_one_thr) {
    xe4_initialize_barrier(store_abar[0], 1);
  }
  sycl::group_barrier(sg);

  auto mSrc = make_tensor(make_gmem_ptr(src_ptr), make_layout(make_shape(prob_size)));
  auto mDst = make_tensor(make_gmem_ptr(dst_ptr), make_layout(make_shape(prob_size)));

  constexpr auto lane_tile_size = Int<kRowElements>{};
  auto lane_coord = make_coord(lane_id);

  auto tGsrc  = coalesce(local_tile(mSrc,     make_shape(lane_tile_size), lane_coord, Step<_1>{}));
  auto tGdst  = coalesce(local_tile(mDst,     make_shape(lane_tile_size), lane_coord, Step<_1>{}));
  auto tSmem  = coalesce(local_tile(sL_local, make_shape(lane_tile_size), lane_coord, Step<_1>{}));

  // Byte offset for this lane's row. For sub-byte types, sizeof(T) is the storage
  // size (e.g. fp4 has sizeof==1 but packs 2 elements/byte), so use bit-based math.
  uint32_t lane_byte_offset = (lane_elem_offset * kTBits) / 8u;
  auto     src_ptr_u8       = reinterpret_cast<uint8_t const*>(src_ptr);
  auto     dst_ptr_u8       = reinterpret_cast<uint8_t*>(dst_ptr);

  // WARP 0: LOAD. Partial lane uses size < RowSize; others use size = RowSize.
  if (warp_idx == 0) {
    if (elect_one_thr) {
      xe4_set_barrier_transaction_bytes(load_abar[0], barrier_txn_bytes);
    }
    if (lane_active) {
      if constexpr (std::is_same_v<OffsetType, uint64_t>) {
        uint64_t gmem_load_addr = reinterpret_cast<uint64_t>(src_ptr_u8 + lane_byte_offset);
        auto load_atom = adma_load.with(gmem_load_addr, lane_bytes, &load_abar[0],
                                        detail::CacheHint<LoadCC>{},
                                        detail::FillMode<LoadFM>{});
        copy(load_atom, tGsrc, tSmem);
      } else {
        OffsetType byte_offset = static_cast<OffsetType>(lane_byte_offset);
        auto load_atom = adma_load.with(const_cast<T*>(src_ptr), byte_offset, lane_bytes, &load_abar[0],
                                        detail::CacheHint<LoadCC>{},
                                        detail::FillMode<LoadFM>{});
        copy(load_atom, tGsrc, tSmem);
      }
    }
  }

  xe4_syncthreads(item);

  // WARP 1: STORE — mirrors LOAD.
  if (warp_idx == 1) {
    xe4_wait_barrier(load_abar[0], 0);
    if (elect_one_thr) {
      xe4_set_barrier_transaction_bytes(store_abar[0], barrier_txn_bytes);
    }
    if (lane_active) {
      if constexpr (std::is_same_v<OffsetType, uint64_t>) {
        uint64_t gmem_store_addr = reinterpret_cast<uint64_t>(dst_ptr_u8 + lane_byte_offset);
        auto store_atom = adma_store.with(gmem_store_addr, lane_bytes, &store_abar[0],
                                          detail::CacheHint<StoreCC>{});
        copy(store_atom, tSmem, tGdst);
      } else {
        OffsetType byte_offset = static_cast<OffsetType>(lane_byte_offset);
        auto store_atom = adma_store.with(dst_ptr, byte_offset, lane_bytes, &store_abar[0],
                                          detail::CacheHint<StoreCC>{});
        copy(store_atom, tSmem, tGdst);
      }
    }
    xe4_wait_barrier(store_abar[0], 0);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Test Runner - Row-per-lane kernel (active_lanes = CTA_bytes / RowSize; tail lane size<RowSize)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <AddressingMode Mode, typename LoadOp, typename StoreOp, uint32_t RowSize, typename T,
          detail::CacheCtrl LoadCC = detail::CacheCtrl::L2c_L3uc,
          detail::FillMethod LoadFM = detail::FillMethod::Zero,
          detail::CacheCtrl StoreCC = detail::CacheCtrl::L2wb_L3uc>
void run_test_row_per_lane(uint32_t prob_size, sycl::queue& queue, const std::string& test_name)
{
  constexpr uint32_t kNumControlWarps = 2;
  constexpr uint32_t kTBits           = cute::sizeof_bits_v<T>;
  constexpr bool     kIsSubByte       = (kTBits < 8);
  constexpr uint32_t kRowElements     = (RowSize * 8) / kTBits;
  constexpr uint32_t kCopyElements    = kNumThreadsPerWarp * kRowElements;   // 32 rows per CTA
  constexpr uint32_t kCtaBytes        = kNumThreadsPerWarp * RowSize;

  using SmemLayout = Layout<Shape<Int<kCopyElements>>, Stride<Int<1>>>;
  SmemLayout sL{};

  auto src     = make_shared_usm_tensor<T, 'R'>(queue, 1, prob_size);
  auto dst     = make_shared_usm_tensor<T, 'R'>(queue, 1, prob_size);
  auto src_ref = make_shared_usm_tensor<T, 'R'>(queue, 1, prob_size);

  random_fill(src);
  zero_fill(dst);
  copy(src, src_ref);
  subbyte_pack(src);

  T const* src_ptr = &*src.data();
  T*       dst_ptr = &*dst.data();

  using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;
  auto [cluster_size_y, cluster_size_x, cluster_size_z] = ClusterShape_MNK{};
  sycl::range<3> cluster_size(cluster_size_z, cluster_size_y, cluster_size_x);
  namespace syclexp = sycl::ext::oneapi::experimental;
  syclexp::properties props {syclexp::work_groups_per_cluster<3>(cluster_size)};

  uint32_t total_bytes = (prob_size * kTBits + 7u) / 8u;
  uint32_t num_ctas    = (total_bytes + kCtaBytes - 1) / kCtaBytes;            // last CTA may be partial

  sycl::range<3> local_range(1, kNumControlWarps, kNumThreadsPerWarp);
  sycl::range<3> group_range(1, 1, num_ctas);
  sycl::nd_range<3> range(group_range * local_range, local_range);

  std::cout << "[Row-per-lane] prob_size=" << prob_size
            << "  RowSize=" << RowSize << "B"
            << "  cta_bytes=" << kCtaBytes << "B (32 rows/CTA)"
            << "  total_bytes=" << total_bytes << "B"
            << "  num_ctas=" << num_ctas << std::endl;

  auto launch_cfg = syclexp::launch_config(range, props);
  syclexp::submit_with_event(queue, [&](sycl::handler &handler) {
    syclexp::nd_launch(handler, launch_cfg, [=](sycl::nd_item<3> item) ALWAYS_INLINE {
      adma_row_per_lane_kernel<Mode, RowSize, T,
                               LoadOp, StoreOp, SmemLayout, LoadCC, LoadFM, StoreCC>(
          item, src_ptr, dst_ptr, prob_size, sL);
    });
  }).wait();

  uint32_t err_cnt = 0;

  if constexpr (kIsSubByte) {
    // Sub-byte (fp4) elements are packed two per byte; compare packed bytes.
    subbyte_pack(src_ref);
    auto* ref_bytes = reinterpret_cast<uint8_t const*>(&*src_ref.data());
    auto* dst_bytes = reinterpret_cast<uint8_t const*>(&*dst.data());
    for (uint32_t i = 0; i < total_bytes; i++) {
      if (ref_bytes[i] != dst_bytes[i]) {
        if (err_cnt < 10) {
          printf("  [%s] Packed-byte mismatch at byte %u: ref=0x%02x, dst=0x%02x\n",
                 test_name.c_str(), i, ref_bytes[i], dst_bytes[i]);
        }
        err_cnt++;
      }
    }
  } else {
    T* src_ref_ptr    = &*src_ref.data();
    T* dst_check_ptr  = &*dst.data();
    for (uint32_t i = 0; i < prob_size; i++) {
      if (src_ref_ptr[i] != dst_check_ptr[i]) {
        if (err_cnt < 10) {
          printf("  [%s] Mismatch at index %u: src_ref = %f, dst = %f\n", test_name.c_str(), i,
                 static_cast<float>(src_ref_ptr[i]), static_cast<float>(dst_check_ptr[i]));
        }
        err_cnt++;
      }
    }
  }

  if (err_cnt == 0) {
    printf("✅ [%s] PASSED\n", test_name.c_str());
  } else {
    printf("❌ [%s] FAILED - %d mismatches\n", test_name.c_str(), err_cnt);
    throw std::runtime_error(test_name + " verification failed!");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Test Suite Runner
////////////////////////////////////////////////////////////////////////////////////////////////////

template <AddressingMode Mode>
void run_test_suite(sycl::queue& queue, const std::string& mode_name, uint32_t& test_offset) {
  try {
    // -------- Row-per-lane kernel test suite --------
    // CTA is fixed at 32 rows: kCtaBytes = 32 * RowSize.

    // A. Single CTA, all 32 lanes full. fp16 RowSize=128 -> kCtaBytes=4096=2048 fp16. 1 CTA.
    std::cout << "\nTest " << (test_offset++) << ": " << mode_name << " | fp16 | row-per-lane | RowSize=128, 1 CTA, 32 full lanes" << std::endl;
    run_test_row_per_lane<Mode, cute::XE4_ADMA_ROW_COPY_LINEAR_LOAD, cute::XE4_ADMA_ROW_COPY_LINEAR_STORE, 128, fp16>(
      2048, queue, mode_name + "/fp16/row_per_lane_RS128");

    // B. Multi-CTA, int8. RowSize=32 -> kCtaBytes=1024=1024 int8. prob_size=2048 -> 2 full CTAs.
    std::cout << "\nTest " << (test_offset++) << ": " << mode_name << " | int8 | row-per-lane | RowSize=32, 2 full CTAs" << std::endl;
    run_test_row_per_lane<Mode, cute::XE4_ADMA_ROW_COPY_LINEAR_LOAD, cute::XE4_ADMA_ROW_COPY_LINEAR_STORE, 32, int8_t>(
      2048, queue, mode_name + "/int8/row_per_lane_RS32");

    // C. Single partial CTA, large RowSize. fp16 RowSize=1024 -> kCtaBytes=32768.
    // prob_size=2048 fp16 = 4096 B -> 1 CTA: 4 full lanes + 28 inactive.
    std::cout << "\nTest " << (test_offset++) << ": " << mode_name << " | fp16 | row-per-lane | RowSize=1024, 1 CTA: 4 full + 28 inactive" << std::endl;
    run_test_row_per_lane<Mode, cute::XE4_ADMA_ROW_COPY_LINEAR_LOAD, cute::XE4_ADMA_ROW_COPY_LINEAR_STORE, 1024, fp16>(
      2048, queue, mode_name + "/fp16/row_per_lane_RS1024");

    // D. Min-aligned RowSize=16, single full CTA. kCtaBytes=512=256 fp16. prob_size=256 -> 1 CTA.
    std::cout << "\nTest " << (test_offset++) << ": " << mode_name << " | fp16 | row-per-lane | RowSize=16 (min aligned), 1 full CTA" << std::endl;
    run_test_row_per_lane<Mode, cute::XE4_ADMA_ROW_COPY_LINEAR_LOAD, cute::XE4_ADMA_ROW_COPY_LINEAR_STORE, 16, fp16>(
      256, queue, mode_name + "/fp16/row_per_lane_RS16");

    // E. Very large RowSize, tiny problem. fp16 RowSize=2048, prob_size=256 = 512 B.
    // 1 CTA: cta_bytes=512 -> 0 full + 1 partial lane (size=512 < 2048) + 31 inactive.
    std::cout << "\nTest " << (test_offset++) << ": " << mode_name << " | fp16 | row-per-lane | RowSize=2048, 1 partial lane (size=512B<2048B), 31 inactive" << std::endl;
    run_test_row_per_lane<Mode, cute::XE4_ADMA_ROW_COPY_LINEAR_LOAD, cute::XE4_ADMA_ROW_COPY_LINEAR_STORE, 2048, fp16>(
      256, queue, mode_name + "/fp16/row_per_lane_RS2048_tiny");

    // F. Non-default cache policy (same shape as E). Load:L2uc_L3c, Store:L2wb_L3wb.
    std::cout << "\nTest " << (test_offset++) << ": " << mode_name << " | fp16 | row-per-lane | Load:L2uc_L3c, Store:L2wb_L3wb" << std::endl;
    run_test_row_per_lane<Mode, cute::XE4_ADMA_ROW_COPY_LINEAR_LOAD, cute::XE4_ADMA_ROW_COPY_LINEAR_STORE, 2048, fp16,
             detail::CacheCtrl::L2uc_L3c, detail::FillMethod::Zero, detail::CacheCtrl::L2wb_L3wb>(
      256, queue, mode_name + "/fp16/row_per_lane_l2uc_l3c");

    // G. Multi-CTA with mixed partial in last CTA. prob_size=23424 fp16 = 46848 B -> 3 CTAs.
    // CTA 0, CTA 1: full (32 lanes each). CTA 2: 27 full + 1 partial (256B) + 4 inactive.
    std::cout << "\nTest " << (test_offset++) << ": " << mode_name << " | fp16 | row-per-lane | 3 CTAs: 2 full + 1 mixed partial CTA" << std::endl;
    run_test_row_per_lane<Mode, cute::XE4_ADMA_ROW_COPY_LINEAR_LOAD, cute::XE4_ADMA_ROW_COPY_LINEAR_STORE, 512, fp16>(
      23424, queue, mode_name + "/fp16/row_per_lane_3ctas_mixed");

    // H. 2 CTAs, last CTA has 4 full + 1 partial (96B) + 27 inactive.
    // fp16 RowSize=256, prob_size=4656 fp16 = 9312 B. kCtaBytes=8192.
    std::cout << "\nTest " << (test_offset++) << ": " << mode_name << " | fp16 | row-per-lane | 2 CTAs, last CTA 4 full + 1 partial (96B<256B) + 27 inactive" << std::endl;
    run_test_row_per_lane<Mode, cute::XE4_ADMA_ROW_COPY_LINEAR_LOAD, cute::XE4_ADMA_ROW_COPY_LINEAR_STORE, 256, fp16>(
      4656, queue, mode_name + "/fp16/row_per_lane_partial_cta_mixed");

    // I. 2 CTAs, last CTA has a single partial lane only.
    // fp16 RowSize=256, prob_size=4144 fp16 = 8288 B -> CTA 0 full, CTA 1 cta_bytes=96 -> 1 partial + 31 inactive.
    std::cout << "\nTest " << (test_offset++) << ": " << mode_name << " | fp16 | row-per-lane | 2 CTAs, last CTA single partial (96B<256B) + 31 inactive" << std::endl;
    run_test_row_per_lane<Mode, cute::XE4_ADMA_ROW_COPY_LINEAR_LOAD, cute::XE4_ADMA_ROW_COPY_LINEAR_STORE, 256, fp16>(
      4144, queue, mode_name + "/fp16/row_per_lane_single_partial");

    // J. FillMethod::Nan on partial-lane CTA. Same shape as H, FM=Nan (non-default).
    std::cout << "\nTest " << (test_offset++) << ": " << mode_name << " | fp16 | row-per-lane | FM:Nan  | last-CTA partial (96B<256B)" << std::endl;
    run_test_row_per_lane<Mode, cute::XE4_ADMA_ROW_COPY_LINEAR_LOAD, cute::XE4_ADMA_ROW_COPY_LINEAR_STORE, 256, fp16,
             detail::CacheCtrl::L2c_L3uc, detail::FillMethod::Nan, detail::CacheCtrl::L2wb_L3uc>(
      4656, queue, mode_name + "/fp16/row_per_lane_FMNan");

    // K. fp4 (e2m1, 4-bit packed). RowSize=16B holds 32 fp4 elements/row.
    // kCtaBytes = 32*16 = 512B = 1024 fp4. prob_size=1024 -> 1 full CTA.
    std::cout << "\nTest " << (test_offset++) << ": " << mode_name << " | fp4 | row-per-lane | RowSize=16, 1 full CTA (packed)" << std::endl;
    run_test_row_per_lane<Mode, cute::XE4_ADMA_ROW_COPY_LINEAR_LOAD, cute::XE4_ADMA_ROW_COPY_LINEAR_STORE, 16, cute::float_e2m1_t>(
      1024, queue, mode_name + "/fp4/row_per_lane_RS16_packed");

    // L. fp4 multi-CTA. RowSize=32B -> 64 fp4/row, kCtaBytes=1024B=2048 fp4.
    // prob_size=4096 fp4 = 2048 B -> 2 full CTAs.
    std::cout << "\nTest " << (test_offset++) << ": " << mode_name << " | fp4 | row-per-lane | RowSize=32, 2 full CTAs (packed)" << std::endl;
    run_test_row_per_lane<Mode, cute::XE4_ADMA_ROW_COPY_LINEAR_LOAD, cute::XE4_ADMA_ROW_COPY_LINEAR_STORE, 32, cute::float_e2m1_t>(
      4096, queue, mode_name + "/fp4/row_per_lane_RS32_packed");
  } catch (const std::exception& e) {
    std::cout << "\n❌ Test suite failed for " << mode_name << ": " << e.what() << std::endl;
    throw;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Main
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  sycl::queue queue{sycl::gpu_selector_v};

  std::cout << "\n============================================" << std::endl;
  std::cout << "ADMA Row Copy - All Addressing Modes" << std::endl;
  std::cout << "============================================" << std::endl;
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "============================================\n" << std::endl;

  try {
    // .a64 mode
    std::cout << "\n╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║   .a64 Mode (uint64_t)                 ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝\n" << std::endl;
    uint32_t total_tests = 1;
    run_test_suite<AddressingMode::A64>(queue, "A64", total_tests);

    // .a32u mode
    std::cout << "\n╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║   .a32u Mode (uint32_t)                ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝\n" << std::endl;
    run_test_suite<AddressingMode::A32U>(queue, "A32U", total_tests);

    // .a32s mode
    std::cout << "\n╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║   .a32s Mode (int32_t)                 ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝\n" << std::endl;
    run_test_suite<AddressingMode::A32S>(queue, "A32S", total_tests);

    std::cout << "\n============================================" << std::endl;
    std::cout << "✅ ALL TESTS PASSED!" << std::endl;
    std::cout << "   .a64 Mode" << std::endl;
    std::cout << "   .a32u Mode" << std::endl;
    std::cout << "   .a32s Mode" << std::endl;
    std::cout << "============================================\n" << std::endl;

  } catch (const std::exception& e) {
    std::cout << "\n============================================" << std::endl;
    std::cout << "❌ TEST SUITE FAILED" << std::endl;
    std::cout << "Error: " << e.what() << std::endl;
    std::cout << "============================================\n" << std::endl;
    return 1;
  }

  return 0;
}
