/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation. All rights reserved.
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
// GEMM with Intra-WG Sub-Group Split-K + Async Reduce (Xe4 CuTe Tutorial)
//
// Companion to gemm_async_reduce_xe4.cpp (cross-WG SplitK baseline) and
// gemm_pipe_async_reduce_xe4.cpp (D-pipelined intra-mainloop reduce).
//
// This example demonstrates **intra-WG SplitK** — the K dimension is split
// across SUB-GROUPS within a single workgroup, replacing the baseline's
// cross-WG SplitK (which used multiple WGs per (m,n) tile).
//
// Architecture: 3-Subgroup warp specialization
//   SG0 (Producer):       ADMA loads A,B for the FULL K range
//   SG1 (Consumer-even):  MMA over K-tiles {0, 2, 4, …} → sD[0]
//   SG2 (Consumer-odd):   MMA over K-tiles {1, 3, 5, …} → sD[1]
//   Epilogue:             SG1 STORE_REDUCEs sD[0]; SG2 STORE_REDUCEs sD[1].
//                         Atomicity in GMEM merges the two partials.
//
// Key differences vs cross-WG SplitK:
//   - One WG per (m,n) (vs num_k_splits WGs)
//   - A,B are loaded ONCE per K-tile and SHARED across both consumer SGs
//     (each consumer reads its own subset from the same K-pipe stages)
//   - Each consumer accumulates with carry (mma_ctrl=0x100 first iter, 0x000 after)
//     so its SLM slot holds the in-WG partial sum over its K-slice
//   - GMEM atomic ops per (m,n) = 2 (one per SG slot) instead of num_k_splits
//
// Trade-offs:
//   + Determinism: each SG's K-slice sum is fixed-order (vs cross-WG atomic order)
//   + Fewer WGs to schedule, less GMEM atomic contention
//   - Different launch geometry (3 SGs/WG, not 2)
//   - Both consumer SGs share the systolic array hardware; whether their MMAs
//     run concurrently is a hardware microarchitecture question
//

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gpu_generics.h>

#include <cute/arch/mma_xe4_amma.hpp>
#include <cute/arch/mma_xe4.hpp>
#include <cute/atom/mma_traits_xe4_amma.hpp>
#include <cute/atom/copy_traits_xe4_adma.hpp>
#include <cute/arch/xe4_util.hpp>

#include "../../../common/sycl_cute_common.hpp"
#include "../../../xe4/utils/validation.hpp"

using namespace cute;
using sycl::ext::oneapi::this_work_item::get_nd_item;

constexpr static size_t SmemAlignment = 256;

// Per-CTA SLM layout. SmemLayoutAcc is rank-3 — the outer mode is K_SPLIT_INTRA
// (number of intra-WG K-consumer SGs), so smem_D holds K_SPLIT_INTRA distinct
// per-K-slice partial accumulators that get merged via STORE_REDUCE in GMEM.
template <class ElementA,
          class ElementB,
          class ElementD,
          class SmemLayoutA,
          class SmemLayoutB,
          class SmemLayoutAcc>
struct SharedStorage : cute::aligned_struct<SmemAlignment, _0>
{
  cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>, SmemAlignment> smem_A;
  cute::array_aligned<ElementB, cute::cosize_v<SmemLayoutB>, SmemAlignment> smem_B;
  // smem_D: K_SPLIT_INTRA per-K-slice accumulator slots.
  cute::array_aligned<ElementD, cute::cosize_v<SmemLayoutAcc>, SmemAlignment> smem_D;
};

inline void xe4_syncthreads() {
  auto group = get_nd_item<1>().get_group();
  sycl::group_barrier(group);
}

////////////////////////////////////////////////////////////////////////////////
// Device kernel: Split-K GEMM with ADMA load/store atoms and AMMA compute
////////////////////////////////////////////////////////////////////////////////
template <class ProblemShape, class CtaTiler, class TileShape,
          class TA, class SmemLayoutA, class SmemLayoutAcc, class ADMA_A,
          class TB, class SmemLayoutB, class ADMA_B, class ADMA_Reduce,
          class TD, class TiledMma>
void
gemm_reduce_device(ProblemShape shape_MNK, CtaTiler cta_tiler, TileShape tile_shape,
            TA const* A, SmemLayoutA sA_layout, SmemLayoutAcc sD_layout, ADMA_A adma_load_a,
            TB const* B, SmemLayoutB sB_layout, ADMA_B adma_load_b,
            ADMA_Reduce adma_reduce_d,
            TD* D,
            int k_tile_start, int k_tiles_this_wg,
            sycl::nd_item<3> item)
{
  static_assert(rank(shape_MNK) == 3);
  static_assert(rank(cta_tiler) == 3);
  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);
  static_assert(is_static<SmemLayoutAcc>::value);

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 1: Allocate hardware tensor descriptors and bind them to ADMA atoms.
  // ──────────────────────────────────────────────────────────────────────────
  auto tdesc_a = allocate_tdesc<0>();
  auto tdesc_b = allocate_tdesc<1>();
  auto tdesc_d = allocate_tdesc<2>();

  adma_load_a.set_tensor_desc(tdesc_a);
  adma_load_b.set_tensor_desc(tdesc_b);
  adma_reduce_d.set_tensor_desc(tdesc_d);

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 2: Allocate SLM. smem_D holds K_SPLIT_INTRA per-K-slice accumulators.
  // ──────────────────────────────────────────────────────────────────────────
  using SharedStorageType = SharedStorage<TA, TB, TD, SmemLayoutA, SmemLayoutB, SmemLayoutAcc>;
  auto ptr = alloc_slm_buffer<uint8_t, sizeof(SharedStorageType)>(item.get_group());
  auto& smem = *reinterpret_cast<SharedStorageType*>(ptr);

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.begin()), SmemLayoutA{});   // (BLK_M,BLK_K,K_PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.begin()), SmemLayoutB{});   // (BLK_N,BLK_K,K_PIPE)
  Tensor sD = make_tensor(make_smem_ptr(smem.smem_D.begin()), SmemLayoutAcc{}); // (BLK_M,BLK_N,K_SPLIT_INTRA)

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 3: TMA tensor views, tiled for this CTA. (One WG per (m,n) — no SplitK
  // index in cta_coord. K is kept free via the trailing _.)
  // ──────────────────────────────────────────────────────────────────────────
  auto [M, N, K] = shape_MNK;
  auto mA = adma_load_a.get_tma_tensor(make_shape(M, K));
  auto mB = adma_load_b.get_tma_tensor(make_shape(N, K));
  auto mD = adma_reduce_d.get_tma_tensor(make_shape(M, N));

  auto cta_coord = make_coord(BlockIdxX(), BlockIdxY(), _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});   // (BLK_M, BLK_K, k_tiles)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});  // (BLK_N, BLK_K, k_tiles)
  Tensor gD = local_tile(mD, cta_tiler, cta_coord, Step<_1, _1, X>{});   // (BLK_M, BLK_N)

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 4: TMA-partition GMEM and SLM views.
  // ──────────────────────────────────────────────────────────────────────────
  auto [tAgA, tAsA] = tma_partition(adma_load_a,
                                     group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto [tBgB, tBsB] = tma_partition(adma_load_b,
                                     group_modes<0,2>(sB), group_modes<0,2>(gB));

  constexpr int K_PIPE_MAX     = decltype(size<1>(tAsA))::value;
  constexpr int K_SPLIT_INTRA  = decltype(size<2>(SmemLayoutAcc{}))::value;

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 5: Per-stage DMA byte counts.
  // ──────────────────────────────────────────────────────────────────────────
  constexpr int dma_transaction_bytesA = (cosize(SmemLayoutA{}) * sizeof(TA)) / K_PIPE_MAX;
  constexpr int dma_transaction_bytesB = (cosize(SmemLayoutB{}) * sizeof(TB)) / K_PIPE_MAX;
  constexpr int dma_transaction_bytesD = (cosize(SmemLayoutAcc{}) * sizeof(TD)) / K_SPLIT_INTRA;

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t warp_idx = get_sg_id();

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 6: Allocate and initialize abarriers.
  //
  //   load_a_abar[K_PIPE_MAX]: GMEM→SLM A done (signaled by ADMA load)
  //   load_b_abar[K_PIPE_MAX]: GMEM→SLM B done
  //   mma_abar   [K_PIPE_MAX]: AB-track per K-pipe stage. Each consumer SG that
  //                            reads slot p signals 2 bytes; if BOTH consumers
  //                            read this stage (same K-tile), tx_bytes = 4.
  //                            But here each K-tile is read by exactly ONE
  //                            consumer (even/odd partition), so tx_bytes = 2.
  //   reduce_abar[K_SPLIT_INTRA]: STORE_REDUCE completion, one per slot.
  // ──────────────────────────────────────────────────────────────────────────
  auto load_a_abar  = allocate_abar<0, K_PIPE_MAX>();
  auto load_b_abar  = allocate_abar<1, K_PIPE_MAX>();
  auto mma_abar     = allocate_abar<2, K_PIPE_MAX>();
  auto reduce_abar  = allocate_abar<3, K_SPLIT_INTRA>();

  if (elect_one_thr && warp_idx == 0) {
    // SG0 (producer) initializes load barriers and reduce-completion barriers.
    for (int i = 0; i < K_PIPE_MAX; ++i) {
      xe4_initialize_barrier(load_a_abar[i], 1);
      xe4_initialize_barrier(load_b_abar[i], 1);
    }
  } else if (elect_one_thr && warp_idx == 1) {
    // SG1 initializes mma_abar (both consumers signal it) and its reduce slot.
    for (int i = 0; i < K_PIPE_MAX; ++i) {
      xe4_initialize_barrier(mma_abar[i], 1);
    }
    for (int i = 0; i < K_SPLIT_INTRA; ++i) {
      xe4_initialize_barrier(reduce_abar[i], 1);
    }
  }
  xe4_syncthreads();

  uint32_t dummy_mask = 0;

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 7: MMA partition. tCsD is rank-4 (MMA, MMA_M, MMA_N, K_SPLIT_INTRA).
  // Each consumer indexes its own slot via tCrD(_,_,_,my_slot).
  // ──────────────────────────────────────────────────────────────────────────
  TiledMma mma{};
  ThrMMA thr_mma = mma.get_thread_slice(ThreadIdxX());
  Tensor tCsA = thr_mma.partition_A(sA);   // (MMA,MMA_M,MMA_K,K_PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);   // (MMA,MMA_N,MMA_K,K_PIPE)
  Tensor tCsD = thr_mma.partition_C(sD);   // (MMA,MMA_M,MMA_N,K_SPLIT_INTRA)
  Tensor tCgD = thr_mma.partition_C(gD);   // (MMA,MMA_M,MMA_N) — single GMEM target

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrD = thr_mma.make_fragment_C(tCsD);

  xe4_syncthreads();

  // K-pipeline state — both consumers share the SAME K-pipe slot for any
  // given K-tile (the producer loads K-tile k into slot k % K_PIPE_MAX, and
  // whichever consumer owns k reads from that slot).
  int write_pipe = 0;
  uint32_t write_phase = 0;
  int read_pipe = 0;
  uint32_t read_phase = 0;

  // ════════════════════════════════════════════════════════════════════════
  // STEP 8: PRODUCER (SG0) — load all K-tiles, with AB-track back-pressure.
  // Identical to the baseline producer; it doesn't care which consumer will
  // read each K-tile.
  // ════════════════════════════════════════════════════════════════════════
  if (warp_idx == 0) {
    if (elect_one_thr) {
      int k_tile = k_tile_start;
      int prologue_count = (k_tiles_this_wg < K_PIPE_MAX) ? k_tiles_this_wg : K_PIPE_MAX;
      for (int pipe = 0; pipe < prologue_count; ++pipe) {
        xe4_set_barrier_transaction_bytes(load_a_abar[pipe], dma_transaction_bytesA);
        xe4_set_barrier_transaction_bytes(load_b_abar[pipe], dma_transaction_bytesB);
        copy(adma_load_a.with(&load_a_abar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
        copy(adma_load_b.with(&load_b_abar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
        ++k_tile;
      }
      for (int remaining = prologue_count; remaining < k_tiles_this_wg; ++remaining) {
        xe4_wait_barrier(mma_abar[write_pipe], write_phase);
        xe4_set_barrier_transaction_bytes(load_a_abar[write_pipe], dma_transaction_bytesA);
        xe4_set_barrier_transaction_bytes(load_b_abar[write_pipe], dma_transaction_bytesB);
        copy(adma_load_a.with(&load_a_abar[write_pipe]), tAgA(_, k_tile), tAsA(_, write_pipe));
        copy(adma_load_b.with(&load_b_abar[write_pipe]), tBgB(_, k_tile), tBsB(_, write_pipe));
        ++k_tile;
        if (++write_pipe == K_PIPE_MAX) { write_pipe = 0; write_phase ^= 1; }
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }
  // ════════════════════════════════════════════════════════════════════════
  // STEP 9: CONSUMERS (SG1, SG2) — each owns half the K-tiles.
  //
  // SG1 (warp_idx=1) owns K-tiles {0, 2, 4, …} → accumulates into sD[0]
  // SG2 (warp_idx=2) owns K-tiles {1, 3, 5, …} → accumulates into sD[1]
  //
  // Within each consumer:
  //   - mma_ctrl = 0x100 (NullC=1) on the FIRST iter of its slice → D = A·B
  //   - mma_ctrl = 0x000 thereafter → D += A·B (with-carry accumulation)
  //   - On the LAST iter, switches to D-track so the epilogue can wait for
  //     accumulator-written-to-SLM before issuing STORE_REDUCE.
  //
  // The consumers walk the K-pipe state IN LOCKSTEP — they use the same
  // read_pipe/read_phase counters incremented by every K-tile, and each
  // consumer waits on load_a/b_abar only for the K-tiles it owns. The other
  // consumer's K-tiles still need their slot freed via mma_abar AB-track.
  // ════════════════════════════════════════════════════════════════════════
  else if (warp_idx == 1 || warp_idx == 2) {
    if (elect_one_thr) {
      const int my_sg     = warp_idx - 1;          // 0 or 1
      const int my_slot   = my_sg;                 // sD[my_slot] is this SG's accumulator
      // Count of K-tiles this consumer owns:
      //   even k's: ceil(T/2) = (T+1)/2
      //   odd  k's: floor(T/2) = T/2
      const int my_count  = (my_sg == 0) ? (k_tiles_this_wg + 1) / 2
                                         : (k_tiles_this_wg) / 2;
      int my_iter = 0;
      uint64_t mma_ctrl_local = 0x100;             // NullC on first MMA into this slot

      // Walk the global K-tile index. Each iteration picks one K-tile; only
      // the consumer that owns it issues an MMA; the other consumer skips
      // (but still advances read_pipe so they stay in lockstep with the
      // shared K-pipeline barrier slots).
      for (int k_iter = 0; k_iter < k_tiles_this_wg; ++k_iter) {
        bool mine = (k_iter % K_SPLIT_INTRA) == my_sg;
        if (mine) {
          xe4_wait_barrier(load_a_abar[read_pipe], read_phase);
          xe4_wait_barrier(load_b_abar[read_pipe], read_phase);
          bool last_for_me = (my_iter + 1 == my_count);
          if (!last_for_me) {
            // AB-track: signal mma_abar so producer can refill this K-pipe slot.
            xe4_set_barrier_transaction_bytes(mma_abar[read_pipe], 2);
            auto new_mma = mma.with(AMMA::TrackMethod<AMMA::Tracking::AB>{}, mma_ctrl_local,
                                    &mma_abar[read_pipe], &mma_abar[read_pipe],
                                    dummy_mask, dummy_mask);
            cute::gemm(new_mma,
                       tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe),
                       tCrD(_,_,_,my_slot));
          } else {
            // Last MMA for this consumer → use D-track, signal reduce_abar[my_slot]
            xe4_set_barrier_transaction_bytes(mma_abar[read_pipe], 1);
            xe4_set_barrier_transaction_bytes(reduce_abar[my_slot], 1);
            auto new_mma = mma.with(AMMA::TrackMethod<AMMA::Tracking::DAB>{}, mma_ctrl_local,
                                    &reduce_abar[my_slot],
                                    &mma_abar[read_pipe], &mma_abar[read_pipe],
                                    dummy_mask, dummy_mask);
            cute::gemm(new_mma,
                       tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe),
                       tCrD(_,_,_,my_slot));
          }
          mma_ctrl_local = 0x000;                  // accumulate from now on
          ++my_iter;
        }
        // Advance shared K-pipe state regardless of ownership.
        if (++read_pipe == K_PIPE_MAX) { read_pipe = 0; read_phase ^= 1; }
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }

  xe4_syncthreads();

  // ════════════════════════════════════════════════════════════════════════
  // STEP 10: EPILOGUE — each consumer SG fires STORE_REDUCE for its own slot.
  // Both reduces target the SAME GMEM tile D[m,n]; atomicity in the GMEM
  // reduce instruction merges them. (Two atomic adds per (m,n) — one from
  // SG1's slot, one from SG2's slot — into the zero-initialized D.)
  // ════════════════════════════════════════════════════════════════════════
  if (warp_idx == 1 || warp_idx == 2) {
    if (elect_one_thr) {
      const int my_slot = warp_idx - 1;
      xe4_wait_barrier(reduce_abar[my_slot], 0);
      xe4_set_barrier_transaction_bytes(reduce_abar[my_slot], dma_transaction_bytesD);
      copy(adma_reduce_d.with(&reduce_abar[my_slot]),
           tCsD(_,_,_,my_slot), tCgD);
      xe4_wait_barrier(reduce_abar[my_slot], 1);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Host setup and launch
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  sycl::queue queue{sycl::gpu_selector_v};
  std::cout << "Running on device: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  int m = 256;
  if (argc >= 2) sscanf(argv[1], "%d", &m);
  int n = 256;
  if (argc >= 3) sscanf(argv[2], "%d", &n);
  int k = 256;
  if (argc >= 4) sscanf(argv[3], "%d", &k);

  using TA = fp16;
  using TB = fp16;
  using TD = float;

  auto prob_shape = make_shape(m, n, k);

  auto A = make_shared_usm_tensor<TA, 'R'>(queue, m, k);
  auto B = make_shared_usm_tensor<TB, 'R'>(queue, n, k);
  auto D = make_shared_usm_tensor<TD, 'R'>(queue, m, n);

  random_fill(A);
  random_fill(B);
  zero_fill(D);

  // Keep a copy of A/B before subbyte_pack for verification
  auto A_ref = make_shared_usm_tensor<TA, 'R'>(queue, m, k);
  auto B_ref = make_shared_usm_tensor<TB, 'R'>(queue, n, k);
  copy(A, A_ref);
  copy(B, B_ref);

  subbyte_pack(A);
  subbyte_pack(B);

  TA const* A_ptr = &*A.data();
  TB const* B_ptr = &*B.data();
  TD* D_ptr = &*D.data();

  auto dA = A.stride();
  auto dB = B.stride();
  auto dD = D.stride();

  Tensor mA = make_tensor(make_gmem_ptr(A_ptr), make_layout(make_shape(m, k), dA));
  Tensor mB = make_tensor(make_gmem_ptr(B_ptr), make_layout(make_shape(n, k), dB));
  Tensor mD = make_tensor(make_gmem_ptr(D_ptr), make_layout(make_shape(m, n), dD));

  // Tile and cluster shapes
  using TileShape_MNK = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;
  constexpr auto bM = get<0>(TileShape_MNK{});
  constexpr auto bN = get<1>(TileShape_MNK{});
  constexpr auto bK = get<2>(TileShape_MNK{});
  auto cta_tiler = make_shape(bM, bN, bK);
  constexpr auto bP   = Int<3>{};  // K-load pipeline depth
  constexpr auto bSI  = Int<2>{};  // K_SPLIT_INTRA — number of intra-WG K-consumer SGs

  static constexpr auto majorA = cute::AMMA::Major::K;
  static constexpr auto majorB = cute::AMMA::Major::K;

  using ElementAccumulator = float;

  // MMA atom
  using TiledMma = decltype(cute::make_tiled_mma(
    cute::AMMA::ss_op_selector<
      ElementAccumulator, TA, TB, TD,
      decltype(cute::product_each(TileShape_MNK{})),
      ClusterShape_MNK, majorA, majorB>()
  ));

  // SMEM layouts (static)
  using SmemLayoutAtomA = decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenRowMajor{}));
  using SmemLayoutAtomB = decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenRowMajor{}));
  using SmemLayoutAtomD = decltype(make_layout(cute::select<0, 1>(TileShape_MNK{}), GenRowMajor{}));

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
      Step<_2, _1, _3>{}));

  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
      Step<_2, _1, _3>{}));

  // SmemLayoutAcc is K_SPLIT_INTRA-deep: one accumulator slot per consumer SG.
  using SmemLayoutAcc = decltype(tile_to_shape(
      SmemLayoutAtomD{},
      make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), bSI),
      Step<_2, _1, _3>{}));

  SmemLayoutA sA_layout{};
  SmemLayoutB sB_layout{};
  SmemLayoutAcc sD_layout{};

  // Cluster layout for non-cluster case
  auto cluster_layout_vmnk = tiled_divide(make_layout(ClusterShape_MNK{}),
                                           make_tile(typename TiledMma::AtomThrID{}));

  // ADMA Load atoms for A and B
  using GmemTiledCopyA = cute::XE4_ADMA_LOAD;
  using GmemTiledCopyB = cute::XE4_ADMA_LOAD;

  auto adma_load_a = make_adma_atom_A_xe4(
      GmemTiledCopyA{},
      mA,
      SmemLayoutA{}(_, _, cute::Int<0>{}),
      TileShape_MNK{},
      TiledMma{},
      cluster_layout_vmnk);

  auto adma_load_b = make_adma_atom_B_xe4(
      GmemTiledCopyB{},
      mB,
      SmemLayoutB{}(_, _, cute::Int<0>{}),
      TileShape_MNK{},
      TiledMma{},
      cluster_layout_vmnk);

  // ADMA STORE_REDUCE atom for D (atomic float add). Built from a single-slot
  // slice of the K_SPLIT_INTRA-deep SmemLayoutAcc — each STORE_REDUCE call
  // moves one bM×bN slot to GMEM; the per-call source picks the slot via
  // tCsD(_,_,_,my_slot).
  using GmemTiledCopyD = cute::XE4_ADMA_STORE_REDUCE<TD, cute::RedOp::Add, cute::BarrierType::Abarrier>;
  auto cta_tiler_mn = make_shape(bM, bN);
  auto adma_reduce_d = make_adma_copy<TD>(GmemTiledCopyD{}, mD,
                                          SmemLayoutAcc{}(_, _, cute::Int<0>{}),
                                          cta_tiler_mn, Int<1>{});

  // Intra-WG SplitK: one WG per (m,n) sweeps ALL K-tiles. Inside the WG, two
  // consumer SGs partition K-tiles even/odd. No cross-WG SplitK needed.
  int total_k_tiles = k / (int)bK;
  int num_k_splits = 1;
  int k_tiles_per_wg = total_k_tiles;

  // Launch configuration. NumWarps = 3: SG0 producer + SG1/SG2 consumers.
  constexpr int NumWarps = 3;
  constexpr int NumThreadsPerWarp = 32;
  auto num_groups_mn = ceil_div(prob_shape, TileShape_MNK{});

  namespace syclexp = sycl::ext::oneapi::experimental;
  auto [cluster_size_y, cluster_size_x, cluster_size_z] = ClusterShape_MNK{};
  sycl::range<3> clusterSize(cluster_size_z, cluster_size_y, cluster_size_x);
  syclexp::properties Props{syclexp::work_groups_per_cluster<3>(clusterSize)};

  sycl::range<3> local_range(1, NumWarps, NumThreadsPerWarp);
  sycl::range<3> group_range(num_k_splits,
                             get<1>(num_groups_mn), get<0>(num_groups_mn));
  sycl::nd_range<3> Range(group_range * local_range, local_range);

  std::cout << "Running GEMM with intra-WG SG-SplitK + ADMA STORE_REDUCE..." << std::endl;
  std::cout << "  Problem: " << m << "x" << n << "x" << k << std::endl;
  std::cout << "  Tile: " << bM << "x" << bN << "x" << bK << std::endl;
  std::cout << "  Sub-groups: " << NumWarps << " (1 producer + " << bSI << " K-consumers)" << std::endl;
  std::cout << "  K-splits (cross-WG): " << num_k_splits << "  (tiles/WG: " << k_tiles_per_wg << ")" << std::endl;

  auto launch_cfg = syclexp::launch_config(Range, Props);
  syclexp::submit_with_event(queue, [&](sycl::handler &handler) {
    syclexp::nd_launch(handler, launch_cfg, [=](sycl::nd_item<3> item) ALWAYS_INLINE {
      int k_split_id = item.get_group(0);
      int k_tile_start = k_split_id * k_tiles_per_wg;

      gemm_reduce_device<decltype(prob_shape), decltype(cta_tiler), decltype(TileShape_MNK{}),
                  TA, decltype(sA_layout), decltype(sD_layout), decltype(adma_load_a),
                  TB, decltype(sB_layout), decltype(adma_load_b), decltype(adma_reduce_d),
                  TD, TiledMma>(
                  prob_shape, cta_tiler, TileShape_MNK{},
                  A_ptr, sA_layout, sD_layout, adma_load_a,
                  B_ptr, sB_layout, adma_load_b, adma_reduce_d,
                  D_ptr,
                  k_tile_start, k_tiles_per_wg, item);
    });
  }).wait();

  // Verification using validate_gemm_result (MKL GEMM reference).
  std::cout << "=======TEST RESULT========" << std::endl;

  TA* A_ref_ptr = &*A_ref.data();
  TB* B_ref_ptr = &*B_ref.data();

  // A is M×K row_major; B is N×K row_major (i.e. B^T is K×N row_major), so
  // pass col_major for B to make MKL transpose it before multiplying.
  int err_cnt = validate_gemm_result(A_ref_ptr, B_ref_ptr, D_ptr, m, n, k,
                                     mem_layout::row_major, mem_layout::col_major);
  printf("Verification: %s\n", (err_cnt == 0) ? "PASSED" : "FAILED");

  return err_cnt == 0 ? 0 : 1;
}
