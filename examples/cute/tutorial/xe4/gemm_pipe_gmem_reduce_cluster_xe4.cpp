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

// Split-K GEMM with Cluster Multicast + D-Pipelined Intra-Mainloop Async Reduce
//
// Companion to gemm_async_reduce_cluster_xe4.cpp (the epilogue-style baseline).
// Both files use ClusterShape<4,1,1> for B-multicast and cross-WG SplitK; this
// file additionally pipelines the STORE_REDUCE inside the mainloop so the
// reduce DMA overlaps with subsequent MMAs (the cumulative K-sum lives in
// GMEM, not SLM).
//
// D = A × B^T (TN layout) combining TWO orthogonal mechanisms:
//   1. Cross-WG SplitK: K divided into num_k_splits slices, multiple WGs
//      target the same (m,n) tile. Atomicity from STORE_REDUCE.
//   2. Intra-WG D-pipeline (NEW): each K-tile MMA produces a per-K-tile
//      partial in sD[d_pipe]; STORE_REDUCE fires per K-tile while later
//      MMAs run. The single epilogue STORE_REDUCE goes away — cumulative
//      sum lives in GMEM, not SLM.
//   - ClusterShape<4,1,1>: B multicast across 4 M-CTAs (unchanged)
//   - Warp-specialized: SG0=producer(DMA loads + reduce drain), SG1=consumer(MMA)
//   - Triple-buffered K-pipeline (K_PIPE_MAX=3), D-pipeline depth D_PIPE=2
//
// Algorithm at a glance (one cluster of 4 CTAs, each owns one M-tile):
//
//   Step 0 (host):  D is zero-initialized in GMEM.
//   Step 1 (SG0,every CTA): Issue K_PIPE_MAX cooperative loads (B uses
//                           cluster mcast; A is self-only). Refill in
//                           steady state via mma_abar back-pressure.
//   Step 2 (SG1,every CTA): For r = 0..T-1:
//                             - wait load_a/b (mcast B has populated this CTA's SLM)
//                             - if recycling sD slot, wait d_done_abar[r % D_PIPE]
//                             - issue MMA(NullC=1, DAB-track, mcast masks):
//                                 sD[r % D_PIPE] = A·B[k=r]^T  (per-K-tile partial)
//                             - DAB signals: d_ready_abar (D done) and mma_abar (AB done)
//   Step 3 (SG0,every CTA): For r = 0..T-1, after Phase A:
//                             - wait d_ready_abar[r % D_PIPE]
//                             - STORE_REDUCE(sD[r % D_PIPE]) → D_gmem[m,n] (atomic add)
//                             - d_done_abar[d_pipe] tracks DMA completion
//   Step 4 (SG0): Drain — wait the *last* d_done_abar phase per slot.
//   Step 5: cluster barrier on entry (cluster_sync) ensures all peer SLMs
//           are allocated before any multicast write fires.
//
// Multiple K-splits compose naturally: each cluster-WG runs the above pipeline
// independently, and T atomic STORE_REDUCEs from each WG land safely into the
// shared D[m,n]. Total atomics per (m,n) = num_k_splits × k_tiles_per_wg.

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
#include <cute/arch/cluster_xe4.hpp>

#include "../../../common/sycl_cute_common.hpp"
#include "../../../xe4/utils/validation.hpp"

using namespace cute;
using sycl::ext::oneapi::this_work_item::get_nd_item;

constexpr static size_t SmemAlignment = 256;

// Per-CTA SLM layout. SmemLayoutAcc is rank-3 (BLK_M, BLK_N, D_PIPE), so smem_D
// holds D_PIPE separate per-K-tile partial accumulator slots — that's the key
// SLM change vs. the epilogue-style baseline (which uses a single accumulator).
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
  // smem_D: D_PIPE per-K-tile partial accumulator slots (vs. 1 in the baseline).
  cute::array_aligned<ElementD, cute::cosize_v<SmemLayoutAcc>, SmemAlignment> smem_D;
};

inline void xe4_syncthreads() {
  auto group = get_nd_item<1>().get_group();
  sycl::group_barrier(group);
}

////////////////////////////////////////////////////////////////////////////////
// DEVICE KERNEL
////////////////////////////////////////////////////////////////////////////////
template <class ProblemShape, class CtaTiler, class TileShape, class ClusterShape,
          class TA, class SmemLayoutA, class SmemLayoutAcc, class ADMA_A,
          class TB, class SmemLayoutB, class ADMA_B, class ADMA_Reduce,
          class TD, class TiledMma>
void
gemm_reduce_device(ProblemShape shape_MNK, CtaTiler cta_tiler, TileShape tile_shape,
            ClusterShape cluster_shape,
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

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 1: Allocate hardware tensor descriptors and bind them to ADMA atoms.
  //
  // The DMA engine reads via descriptor slots; set_tensor_desc copies the
  // CPU-prepared payload into the device descriptor file. The reduce
  // descriptor (tdesc_d) is reused across ALL T STORE_REDUCE issues — that's
  // what makes the per-K-tile reduce loop cheap.
  // ──────────────────────────────────────────────────────────────────────────
  auto tdesc_a = allocate_tdesc<0>();
  auto tdesc_b = allocate_tdesc<1>();
  auto tdesc_d = allocate_tdesc<2>();

  adma_load_a.set_tensor_desc(tdesc_a);
  adma_load_b.set_tensor_desc(tdesc_b);
  adma_reduce_d.set_tensor_desc(tdesc_d);

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 2: Allocate workgroup-local SLM and wrap as CuTe tensors.
  //
  // smem_A/B are triple-buffered for the K-pipeline (K_PIPE_MAX stages).
  // smem_D is the new D-pipeline buffer (D_PIPE slots), one bM×bN float
  // accumulator per in-flight per-K-tile partial.
  // ──────────────────────────────────────────────────────────────────────────
  using SharedStorageType = SharedStorage<TA, TB, TD, SmemLayoutA, SmemLayoutB, SmemLayoutAcc>;
  auto ptr = alloc_slm_buffer<uint8_t, sizeof(SharedStorageType)>(item.get_group());
  auto& smem = *reinterpret_cast<SharedStorageType*>(ptr);

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.begin()), SmemLayoutA{});   // (BLK_M, BLK_K, K_PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.begin()), SmemLayoutB{});   // (BLK_N, BLK_K, K_PIPE)
  Tensor sD = make_tensor(make_smem_ptr(smem.smem_D.begin()), SmemLayoutAcc{}); // (BLK_M, BLK_N, D_PIPE)

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 3: Logical TMA tensor views, tiled for this CTA.
  //
  // - get_tma_tensor: a coordinate-space view of the entire global tensor.
  // - local_tile: slice out this CTA's region. The trailing `_` in cta_coord
  //   keeps the K dimension free so we can iterate over k_tile in the kernel.
  // ──────────────────────────────────────────────────────────────────────────
  auto [M, N, K] = shape_MNK;
  auto mA = adma_load_a.get_tma_tensor(make_shape(M, K));
  auto mB = adma_load_b.get_tma_tensor(make_shape(N, K));
  auto mD = adma_reduce_d.get_tma_tensor(make_shape(M, N));

  auto cta_coord = make_coord(BlockIdxY(), BlockIdxX(), _);

  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});   // (BLK_M, BLK_K, k_tiles)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});  // (BLK_N, BLK_K, k_tiles)
  Tensor gD = local_tile(mD, cta_tiler, cta_coord, Step<_1, _1, X>{});   // (BLK_M, BLK_N)

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 4: Cluster-aware TMA partition for cooperative B multicast.
  //
  // ClusterShape<4,1,1>: 4 CTAs along the M-dim, 1 along N, 1 along K.
  //   - A is loaded independently per CTA (cluster_layout's N-dim=1).
  //   - B is split 4 ways across the cluster M-dim; each CTA loads 1/4 from
  //     GMEM and the multicast network copies it into ALL 4 peer SLMs.
  //
  // cluster_layout_vmnk maps (V, M, N, K) cluster coords → CTA rank.
  // block_rank = this CTA's linear rank (0..3 for <4,1,1>) computed from its
  //              SYCL workgroup id within the cluster.
  // ──────────────────────────────────────────────────────────────────────────
  TiledMma mma{};
  auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                          make_tile(typename TiledMma::AtomThrID{}));

  auto cluster_layout_mn = make_layout(select<0,1>(cluster_shape),
                                       make_stride(_1{}, get<0>(cluster_shape)));
  uint32_t block_rank = cluster_layout_mn(make_coord(get_cluster_wgid<1>(), get_cluster_wgid<0>()));
  auto cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(block_rank);

  // tma_partition picks each CTA's slice of A/B given its cluster rank.
  // group_modes<0,2> flattens (BLK, BLK, PIPE/k_tiles) → (tile_elems, stages)
  // before partitioning.
  //
  // For A: project along N-cluster (size=1) → no split, each CTA loads full A.
  // For B: project along M-cluster (size=4) → 4-way split + multicast assembles.
  auto [tAgA, tAsA] = tma_partition(adma_load_a,
                                    get<2>(cta_coord_vmnk), make_layout(size<2>(cluster_layout_vmnk)),
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));

  auto [tBgB, tBsB] = tma_partition(adma_load_b,
                                    get<1>(cta_coord_vmnk), make_layout(size<1>(cluster_layout_vmnk)),
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));

  // K-pipeline depth (from the SLM tensor's pipe-stage count).
  constexpr int K_PIPE_MAX = decltype(size<1>(tAsA))::value;
  // D-pipeline depth (from rank-3 SmemLayoutAcc's mode-2). 2 is the typical sweet spot.
  constexpr int D_PIPE = decltype(size<2>(SmemLayoutAcc{}))::value;

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 5: Compute byte counts + multicast masks.
  //
  // - load_a/b transaction bytes: per-stage size of one A or B tile in SLM.
  // - reduce_d transaction bytes: per-slot size of one bM×bN float accumulator.
  // - mcast_mask_a = 1 << cluster_wgid_m (self-only — A is unique per CTA).
  // - mcast_mask_b = bitmask of all M-row CTAs (all want the same B tile).
  //
  // STORE_REDUCE never multicasts (enforced inside is_xe4_adma_store_reduce_v
  // when constructing make_adma_atom_*). So d_done_abar's tx_bytes is per-slot,
  // not multiplied by cluster size.
  // ──────────────────────────────────────────────────────────────────────────
  constexpr int dma_transaction_bytesA = (cosize(SmemLayoutA{}) * sizeof(TA)) / K_PIPE_MAX;
  constexpr int dma_transaction_bytesB = (cosize(SmemLayoutB{}) * sizeof(TB)) / K_PIPE_MAX;
  constexpr int dma_transaction_bytesD = (cosize(SmemLayoutAcc{}) * sizeof(TD)) / D_PIPE;

  auto [cluster_size_m, cluster_size_n, cluster_size_k] = cluster_shape;
  uint32_t cluster_wgid_m = get_cluster_wgid<1>();

  // A: self-only delivery — each CTA owns a different M-slice of A.
  uint32_t mcast_mask_a = 1u << cluster_wgid_m;
  // B: multicast to all cluster M-CTAs — they all process the same N×K tile.
  uint32_t mcast_mask_b = 0;
  for (uint32_t i = 0; i < (uint32_t)cluster_size_m; ++i) {
    mcast_mask_b |= (1u << i);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 6: Elect a single lane per sub-group, identify SG role.
  //
  // ADMA / AMMA hardware engines are per-SG, not per-lane; only one lane per
  // SG should issue commands. SG0 = producer (DMAs), SG1 = consumer (MMA).
  // ──────────────────────────────────────────────────────────────────────────
  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t warp_idx = get_sg_id();

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 7: Allocate and initialize abarrier groups (intra-loop reduce protocol).
  //
  // Five groups in this kernel — three for the K-pipeline and two new groups
  // for the D-pipeline:
  //
  //   load_a_abar [K_PIPE_MAX]: GMEM→SLM A done   (signaled by ADMA load A)
  //   load_b_abar [K_PIPE_MAX]: GMEM→SLM B done   (signaled by ADMA load B,
  //                                                 cluster mcast → 1 byte each
  //                                                 from each contributing CTA)
  //   mma_abar    [K_PIPE_MAX]: AB-track          (signaled by MMA after reading
  //                                                 A,B; tells SG0 the K-pipe slot
  //                                                 is free to refill)
  //   d_ready_abar[D_PIPE]    : D-track per slot  (NEW; signaled by MMA when
  //                                                 sD[d_pipe] is written; SG0's
  //                                                 reduce loop waits on this)
  //   d_done_abar [D_PIPE]    : DMA done per slot (NEW; signaled by STORE_REDUCE
  //                                                 completion; SG1 waits before
  //                                                 reusing slot, SG0 drains)
  //
  // Initialization rule: each SG initializes the barriers IT will signal.
  // SG0 owns DMA → load_*_abar + d_done_abar.
  // SG1 owns MMA → mma_abar + d_ready_abar.
  // ──────────────────────────────────────────────────────────────────────────
  auto load_a_abar  = allocate_abar<0, K_PIPE_MAX>();
  auto load_b_abar  = allocate_abar<1, K_PIPE_MAX>();
  auto mma_abar     = allocate_abar<2, K_PIPE_MAX>();
  auto d_ready_abar = allocate_abar<3, D_PIPE>();
  auto d_done_abar  = allocate_abar<4, D_PIPE>();

  if (elect_one_thr && warp_idx == 0) {
    // SG0 (producer) initializes the barriers its DMAs will signal.
    for (int i = 0; i < K_PIPE_MAX; ++i) {
      xe4_initialize_barrier(load_a_abar[i], 1);
      xe4_initialize_barrier(load_b_abar[i], 1);
    }
    for (int i = 0; i < D_PIPE; ++i) {
      xe4_initialize_barrier(d_done_abar[i], 1);
    }
  } else if (elect_one_thr && warp_idx == 1) {
    // SG1 (consumer) initializes the barriers its MMA will signal.
    for (int i = 0; i < K_PIPE_MAX; ++i) {
      xe4_initialize_barrier(mma_abar[i], 1);
    }
    for (int i = 0; i < D_PIPE; ++i) {
      xe4_initialize_barrier(d_ready_abar[i], 1);
    }
  }

  xe4_syncthreads();

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 8: Cluster sync — required before any multicast write.
  //
  // The B multicast writes into peer CTAs' SLM. cluster_sync() guarantees ALL
  // peer CTAs in the cluster have allocated SLM and initialized barriers
  // before any DMA fires. Skipped at compile time when ClusterShape size = 1
  // (no peers to write to).
  // ──────────────────────────────────────────────────────────────────────────
  if constexpr (size(ClusterShape{}) > 1) {
    cluster_sync();
  }

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 9: MMA partition and fragment setup.
  //
  // tCsA/tCsB live in SLM (descriptors). tCsD is RANK-4: (MMA, MMA_M, MMA_N,
  // D_PIPE) — picks the d_pipe slot at MMA-issue time. tCgD is rank-3 (single
  // GMEM target — STORE_REDUCE writes the same D[m,n] every iter).
  //
  // No clear(tCsD) here, unlike the baseline! The first MMA into each slot
  // uses NullC=1 (mma_ctrl=0x100) which writes D=A·B without summing C, so
  // any prior junk in sD is harmlessly overwritten.
  // ──────────────────────────────────────────────────────────────────────────
  ThrMMA thr_mma = mma.get_thread_slice(ThreadIdxX());

  Tensor tCsA = thr_mma.partition_A(sA);   // (MMA, MMA_M, MMA_K, K_PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);   // (MMA, MMA_N, MMA_K, K_PIPE)
  Tensor tCsD = thr_mma.partition_C(sD);   // (MMA, MMA_M, MMA_N, D_PIPE) per-slot
  Tensor tCgD = thr_mma.partition_C(gD);   // (MMA, MMA_M, MMA_N) single GMEM target

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrD = thr_mma.make_fragment_C(tCsD);

  xe4_syncthreads();

  // MMAControl(0x100) sets NullC=1 (bit 8 of MMAControl, see mma_xe4_desc.hpp:75)
  // → MMA computes D = A·B (no carry from C). Used on EVERY iter so each MMA
  // produces a fresh per-K-tile partial in sD[d_pipe]. The cumulative K-sum is
  // maintained in GMEM by repeated atomic STORE_REDUCEs.
  uint64_t mma_ctrl = 0x100;

  // ──────────────────────────────────────────────────────────────────────────
  // STEP 10: K-pipeline state (for the load/MMA back-pressure pair).
  // The D-pipeline state is held inside the producer's Phase B and consumer
  // loop locals because they don't share a global counter.
  // ──────────────────────────────────────────────────────────────────────────
  int write_pipe = 0;
  uint32_t write_phase = 0;
  int read_pipe = 0;
  uint32_t read_phase = 0;

  // ════════════════════════════════════════════════════════════════════════
  // STEP 11: PRODUCER (SG0) — three phases in sequence:
  //   Phase A:  K-load loop (cooperative cluster loads, including B multicast)
  //   Phase B:  Reduce loop (one STORE_REDUCE per K-tile, each in flight)
  //   Phase C:  Drain wait  (block until all reduce DMAs have committed to GMEM)
  //
  // Phase A keeps the K-pipeline full so SG1 doesn't starve. Phase B walks the
  // D-pipe slots in issue order, waits for SG1's D-track signal that
  // sD[d_pipe] holds a fresh partial, then asynchronously fires STORE_REDUCE.
  // Successive STORE_REDUCEs run in parallel with SG1's later MMAs — this is
  // where the latency hiding lives.
  // ════════════════════════════════════════════════════════════════════════
  if (warp_idx == 0) {
    if (elect_one_thr) {
      // ───── Phase A: K-load loop ─────
      // A1. Prologue — fire K_PIPE_MAX cooperative loads. Each load_a uses
      //     mcast_mask_a (self-only); each load_b uses mcast_mask_b (all peer
      //     CTAs of this cluster receive 1/cluster_size_m of the data).
      int k_tile = k_tile_start;
      int prologue_count = (k_tiles_this_wg < K_PIPE_MAX) ? k_tiles_this_wg : K_PIPE_MAX;

      for (int pipe = 0; pipe < prologue_count; ++pipe) {
        xe4_set_barrier_transaction_bytes(load_a_abar[pipe], dma_transaction_bytesA);
        xe4_set_barrier_transaction_bytes(load_b_abar[pipe], dma_transaction_bytesB);
        copy(adma_load_a.with(&load_a_abar[pipe], mcast_mask_a), tAgA(_, k_tile), tAsA(_, pipe));
        copy(adma_load_b.with(&load_b_abar[pipe], mcast_mask_b), tBgB(_, k_tile), tBsB(_, pipe));
        ++k_tile;
      }

      // A2. Steady state — wait for AB-track on K-pipe slot, refill it.
      //     Back-pressure prevents the producer from overrunning the consumer.
      for (int remaining = prologue_count; remaining < k_tiles_this_wg; ++remaining) {
        xe4_wait_barrier(mma_abar[write_pipe], write_phase);
        xe4_set_barrier_transaction_bytes(load_a_abar[write_pipe], dma_transaction_bytesA);
        xe4_set_barrier_transaction_bytes(load_b_abar[write_pipe], dma_transaction_bytesB);
        copy(adma_load_a.with(&load_a_abar[write_pipe], mcast_mask_a), tAgA(_, k_tile), tAsA(_, write_pipe));
        copy(adma_load_b.with(&load_b_abar[write_pipe], mcast_mask_b), tBgB(_, k_tile), tBsB(_, write_pipe));
        ++k_tile;
        // Phase alternates on K-pipe wrap so adjacent flips are distinguishable.
        if (++write_pipe == K_PIPE_MAX) { write_pipe = 0; write_phase ^= 1; }
      }

      // ───── Phase B: reduce loop ─────
      // For each K-tile r:
      //   (a) wait for SG1's D-track signal on sD[r mod D_PIPE]
      //   (b) program d_done_abar[d_pipe] with the expected DMA byte count
      //   (c) fire STORE_REDUCE: sD[d_pipe] → GMEM D[m,n] (atomic add)
      // The DMA is asynchronous — SG0 immediately moves on, the consumer's next
      // MMA runs in parallel.
      //
      // Phase mechanics for slot reuse (k_tiles_this_wg > D_PIPE):
      //   d_ready_abar[s] flips once per MMA into slot s. When iter r is about
      //   to wait, the barrier has flipped `cycle` times (count of completed
      //   full passes through D_PIPE slots). Wait phase = cycle & 1.
      {
        int d_pipe = 0;
        int cycle  = 0;
        for (int r = 0; r < k_tiles_this_wg; ++r) {
          uint32_t d_ready_phase = (uint32_t)(cycle & 1);
          xe4_wait_barrier(d_ready_abar[d_pipe], d_ready_phase);
          xe4_set_barrier_transaction_bytes(d_done_abar[d_pipe], dma_transaction_bytesD);
          copy(adma_reduce_d.with(&d_done_abar[d_pipe]), tCsD(_,_,_,d_pipe), tCgD);
          if (++d_pipe == D_PIPE) { d_pipe = 0; ++cycle; }
        }
      }

      // ───── Phase C: drain — wait for all outstanding reduce DMAs ─────
      // Per-slot drain phase:
      //   N_s = floor(T / D_PIPE) + (1 if s < T mod D_PIPE else 0)
      //   The last issuance flips d_done_abar[s] from phase (N_s-1) & 1.
      //   Skip s with N_s == 0 (slot never used when T < D_PIPE).
      {
        int q   = k_tiles_this_wg / D_PIPE;
        int rem = k_tiles_this_wg % D_PIPE;
        for (int s = 0; s < D_PIPE; ++s) {
          int N_s = q + (s < rem ? 1 : 0);
          if (N_s == 0) continue;
          uint32_t drain_phase = (uint32_t)((N_s - 1) & 1);
          xe4_wait_barrier(d_done_abar[s], drain_phase);
        }
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }

  // ════════════════════════════════════════════════════════════════════════
  // STEP 12: CONSUMER (SG1) — single MMA loop, DAB-tracked every iter.
  //
  // For each K-tile k_iter:
  //   1. Wait A,B in K-pipe slot read_pipe.
  //   2. If recycling a D-pipe slot (k_iter >= D_PIPE), wait for the prior
  //      STORE_REDUCE to drain so the slot is safe to overwrite.
  //   3. Program tx_bytes for the barriers the MMA will signal:
  //        mma_abar[read_pipe]:   2 bytes (A-track + B-track)
  //        d_ready_abar[d_pipe]:  1 byte  (D-track)
  //   4. Issue MMA with NullC=1 + DAB tracking + cluster mcast masks.
  //        sD[d_pipe] = sA[read_pipe] · sB[read_pipe]^T  (per-K-tile partial)
  //   5. Advance K-pipe and D-pipe state.
  //
  // The cluster MMA atom (XE4_AMMA_AB_CLUSTER) supports Tracking::AB and
  // Tracking::DAB only. DAB is exactly what the intra-loop pipeline needs —
  // it signals all three barriers in one call.
  // ════════════════════════════════════════════════════════════════════════
  else if (warp_idx == 1) {
    if (elect_one_thr) {
      int d_pipe  = 0;
      int d_cycle = 0;
      for (int k_iter = 0; k_iter < k_tiles_this_wg; ++k_iter) {
        // Step 12.1 — wait for A,B in SLM
        xe4_wait_barrier(load_a_abar[read_pipe], read_phase);
        xe4_wait_barrier(load_b_abar[read_pipe], read_phase);

        // Step 12.2 — slot reuse: wait for prior STORE_REDUCE on this slot
        // to drain before overwriting sD[d_pipe].
        if (k_iter >= D_PIPE) {
          uint32_t d_done_phase = (uint32_t)((d_cycle - 1) & 1);
          xe4_wait_barrier(d_done_abar[d_pipe], d_done_phase);
        }

        // Step 12.3 — program tx_bytes for the barriers the MMA will signal.
        xe4_set_barrier_transaction_bytes(mma_abar[read_pipe], 2);
        xe4_set_barrier_transaction_bytes(d_ready_abar[d_pipe], 1);

        // Step 12.4 — issue MMA. DAB tracking signals THREE barriers:
        //   D-barrier (1B) → d_ready_abar[d_pipe] : sD slot ready for reduce
        //   A-barrier (1B) → mma_abar[read_pipe]  : K-pipe slot freed
        //   B-barrier (1B) → mma_abar[read_pipe]  : K-pipe slot freed
        // Cluster mcast masks tell the MMA which peer barriers to also signal
        // (for cluster-wide AB-track coordination).
        auto new_mma = mma.with(AMMA::TrackMethod<AMMA::Tracking::DAB>{}, mma_ctrl,
                                &d_ready_abar[d_pipe],                          // D
                                &mma_abar[read_pipe], &mma_abar[read_pipe],     // A, B
                                mcast_mask_a, mcast_mask_b);
        cute::gemm(new_mma,
                   tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe),
                   tCrD(_,_,_,d_pipe));   // ← write to slot d_pipe

        // Step 12.5 — advance state machines.
        if (++read_pipe == K_PIPE_MAX) { read_pipe = 0; read_phase ^= 1; }
        if (++d_pipe   == D_PIPE)      { d_pipe    = 0; ++d_cycle;       }
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }

  // STEP 13: WG-wide sync. Phase C above already drained all DMAs; this just
  // rejoins both SGs before the kernel function returns.
  xe4_syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// HOST SETUP AND LAUNCH
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  sycl::queue queue{sycl::gpu_selector_v};
  std::cout << "Running on device: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  int m = 512;
  if (argc >= 2) sscanf(argv[1], "%d", &m);
  int n = 128;
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
  // D must be zero: STORE_REDUCE accumulates into existing value
  zero_fill(D);

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

  // ──────────────────────────────────────────────────────────────────────────
  // HOST STEP 1: Tile, cluster, and pipeline configuration.
  //
  // - TileShape_MNK = 128×128×128 per-WG output tile / K-tile width.
  // - ClusterShape_MNK = 4×1×1: 4 CTAs along M, sharing the same N×K tile of B
  //   via multicast (the entire reason for the cluster).
  // - bP=3:   K-load triple-buffer (producer up to 3 K-tiles ahead of consumer).
  // - bDP=2:  D-pipe double-buffer (consumer 1 K-tile ahead of reducer).
  // ──────────────────────────────────────────────────────────────────────────
  using TileShape_MNK = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape_MNK = cute::Shape<cute::_4, cute::_1, cute::_1>;
  constexpr auto bM = get<0>(TileShape_MNK{});
  constexpr auto bN = get<1>(TileShape_MNK{});
  constexpr auto bK = get<2>(TileShape_MNK{});
  auto cta_tiler = make_shape(bM, bN, bK);
  constexpr auto bP  = Int<3>{};  // K-load pipeline depth
  constexpr auto bDP = Int<2>{};  // D (accumulator) pipeline depth

  static constexpr auto majorA = cute::AMMA::Major::K;
  static constexpr auto majorB = cute::AMMA::Major::K;

  using ElementAccumulator = float;

  // MMA atom: ss_op_selector picks AMMA_AB_CLUSTER variant for ClusterShape<4,1,1>
  using TiledMma = decltype(cute::make_tiled_mma(
    cute::AMMA::ss_op_selector<
      ElementAccumulator, TA, TB, TD,
      decltype(cute::product_each(TileShape_MNK{})),
      ClusterShape_MNK, majorA, majorB>()
  ));

  // ──────────────────────────────────────────────────────────────────────────
  // HOST STEP 2: SLM layouts.
  //
  // - SmemLayoutA: bM × bK × bP   (K-pipeline triple-buffer for A)
  // - SmemLayoutB: bN × bK × bP   (K-pipeline triple-buffer for B)
  // - SmemLayoutAcc: bM × bN × bDP   (D-pipeline buffer — one float slot per
  //                                   in-flight per-K-tile partial)
  // Step<_2, _1, _3> assigns stride priority: K (or N for D) is innermost,
  // outer modes are stride-displaced. Each pipeline slot is contiguous.
  // ──────────────────────────────────────────────────────────────────────────
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

  // SmemLayoutAcc is D_PIPE-deep so MMA-write and STORE_REDUCE can pipeline.
  using SmemLayoutAcc = decltype(tile_to_shape(
      SmemLayoutAtomD{},
      make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), bDP),
      Step<_2, _1, _3>{}));

  SmemLayoutA sA_layout{};
  SmemLayoutB sB_layout{};
  SmemLayoutAcc sD_layout{};

  // ──────────────────────────────────────────────────────────────────────────
  // HOST STEP 3: Build CuTe atoms (ADMA load × A/B with cluster mcast,
  // ADMA STORE_REDUCE × D).
  //
  // Both A and B use XE4_ADMA_LOAD_MULTICAST (the cluster-aware load atom);
  // mcast_mask_a / mcast_mask_b at issue time control which peer SLMs receive
  // each transfer.
  // ──────────────────────────────────────────────────────────────────────────
  auto cluster_layout_vmnk = tiled_divide(make_layout(ClusterShape_MNK{}),
                                           make_tile(typename TiledMma::AtomThrID{}));

  using GmemTiledCopyA = cute::XE4_ADMA_LOAD_MULTICAST;
  using GmemTiledCopyB = cute::XE4_ADMA_LOAD_MULTICAST;

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

  // STORE_REDUCE: atomically adds partial results from SLM to GMEM. Pass a
  // single-slot slice of the D_PIPE-deep SmemLayoutAcc so the atom describes
  // one bM×bN tile transfer (the per-call source picks d_pipe at issue time).
  using GmemTiledCopyD = cute::XE4_ADMA_STORE_REDUCE<TD, cute::RedOp::Add, cute::BarrierType::Abarrier>;
  auto cta_tiler_mn = make_shape(bM, bN);
  auto adma_reduce_d = make_adma_copy<TD>(GmemTiledCopyD{}, mD,
                                          SmemLayoutAcc{}(_, _, cute::Int<0>{}),
                                          cta_tiler_mn, Int<1>{});

  // ──────────────────────────────────────────────────────────────────────────
  // HOST STEP 4: K-decomposition — Cross-WG SplitK + intra-loop D-pipeline.
  //
  // Cross-WG SplitK and intra-loop reduce are ORTHOGONAL: cross-WG SplitK
  // divides K across multiple WGs targeting the same (m,n); within each WG,
  // every K-tile fires its own STORE_REDUCE.
  //
  // Default: 2 K-splits, k_tiles_per_wg = total/2. Configurations:
  //   K=256:  total_k_tiles=2, splits=2, k_tiles_per_wg=1 — intra-loop pipeline
  //                                                          degenerates to 1
  //                                                          STORE_REDUCE per WG
  //                                                          (still correct).
  //   K=512:  total_k_tiles=4, splits=2, k_tiles_per_wg=2 — D_PIPE=2 fully used,
  //                                                          no slot reuse.
  //   K=1024: total_k_tiles=8, splits=2, k_tiles_per_wg=4 — slot reuse 2× per
  //                                                          slot (full phase
  //                                                          tracking exercised).
  //
  // Total atomic STORE_REDUCEs landing in each D[m,n]:
  //   = num_k_splits × k_tiles_per_wg = total_k_tiles
  // All atomic, all hardware-safe regardless of order.
  // ──────────────────────────────────────────────────────────────────────────
  int total_k_tiles = k / (int)bK;
  int num_k_splits  = (total_k_tiles >= 2) ? 2 : 1;
  int k_tiles_per_wg = total_k_tiles / num_k_splits;

  // ──────────────────────────────────────────────────────────────────────────
  // HOST STEP 5: Launch configuration.
  //
  // Each WG: 64 threads = 2 SGs × 32 lanes. SG0=producer, SG1=consumer.
  // Grid dim-0 = num_k_splits — different K-splits target the same (m,n) via
  // atomic STORE_REDUCE. Cluster size = (1,4,1): 4 CTAs along M cooperate via
  // multicast on the same N×K tile of B.
  // ──────────────────────────────────────────────────────────────────────────
  constexpr int NumWarps = 2;
  constexpr int NumThreadsPerWarp = 32;
  auto num_groups_mn = ceil_div(prob_shape, TileShape_MNK{});

  namespace syclexp = sycl::ext::oneapi::experimental;
  auto [cluster_size_m, cluster_size_n, cluster_size_k] = ClusterShape_MNK{};
  sycl::range<3> clusterSize(cluster_size_k, cluster_size_m, cluster_size_n);
  syclexp::properties Props{syclexp::work_groups_per_cluster<3>(clusterSize)};

  sycl::range<3> local_range(1, NumWarps, NumThreadsPerWarp);
  sycl::range<3> group_range(num_k_splits,
                             get<0>(num_groups_mn), get<1>(num_groups_mn));
  sycl::nd_range<3> Range(group_range * local_range, local_range);

  std::cout << "Running Split-K GEMM with Cluster<4,1,1> + intra-mainloop ADMA STORE_REDUCE..." << std::endl;
  std::cout << "  Problem: " << m << "x" << n << "x" << k << std::endl;
  std::cout << "  Tile: " << bM << "x" << bN << "x" << bK << std::endl;
  std::cout << "  K-pipe: " << bP << "  D-pipe: " << bDP << std::endl;
  std::cout << "  Cluster: " << cluster_size_m << "x" << cluster_size_n << "x" << cluster_size_k << std::endl;
  std::cout << "  K-splits: " << num_k_splits << " (tiles/WG: " << k_tiles_per_wg << ")" << std::endl;

  auto launch_cfg = syclexp::launch_config(Range, Props);
  syclexp::submit_with_event(queue, [&](sycl::handler &handler) {
    syclexp::nd_launch(handler, launch_cfg, [=](sycl::nd_item<3> item) ALWAYS_INLINE {
      int k_split_id = item.get_group(0);
      int k_tile_start = k_split_id * k_tiles_per_wg;

      gemm_reduce_device<decltype(prob_shape), decltype(cta_tiler), decltype(TileShape_MNK{}),
                  decltype(ClusterShape_MNK{}),
                  TA, decltype(sA_layout), decltype(sD_layout), decltype(adma_load_a),
                  TB, decltype(sB_layout), decltype(adma_load_b), decltype(adma_reduce_d),
                  TD, TiledMma>(
                  prob_shape, cta_tiler, TileShape_MNK{}, ClusterShape_MNK{},
                  A_ptr, sA_layout, sD_layout, adma_load_a,
                  B_ptr, sB_layout, adma_load_b, adma_reduce_d,
                  D_ptr,
                  k_tile_start, k_tiles_per_wg, item);
    });
  }).wait();

  // ──────────────────────────────────────────────────────────────────────────
  // HOST STEP 6: Verification against MKL GEMM reference.
  //
  // The GPU does (num_k_splits × k_tiles_per_wg) atomic adds per (m,n) in
  // non-deterministic order; MKL gives a deterministic single-thread sum.
  // fp32 atomic-add is not associative, so validate_gemm_result uses an
  // appropriate tolerance for fp16 inputs / float accumulation.
  // ──────────────────────────────────────────────────────────────────────────
  std::cout << "=======TEST RESULT========" << std::endl;

  TA* A_ref_ptr = &*A_ref.data();
  TB* B_ref_ptr = &*B_ref.data();

  // A is M×K row-major; B is N×K row-major → pass col_major so MKL transposes
  int err_cnt = validate_gemm_result(A_ref_ptr, B_ref_ptr, D_ptr, m, n, k,
                                     mem_layout::row_major, mem_layout::col_major);
  printf("Verification: %s\n", (err_cnt == 0) ? "PASSED" : "FAILED");

  return err_cnt == 0 ? 0 : 1;
}
