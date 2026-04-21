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

#pragma once

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// XE4 Block-Scaled GEMM with Cluster Multicast — Parameterized Test Infrastructure
//
// This header extends amma_adma_xe4_blockscaled_gemm_base.hpp with cluster
// multicast support.  With ClusterShape > (1,1,1), multiple CTAs cooperate:
//   - A tiles + SFA are multicast across N-CTAs sharing the same M-row
//   - B tiles + SFB are multicast across M-CTAs sharing the same N-column
//
// Key differences from the non-cluster base:
//   1. ClusterShape_MNK from Config (e.g. Shape<_2,_2,_1>)
//   2. XE4_ADMA_LOAD_MULTICAST atoms for data and scale factors
//   3. XE4_AMMA_AB_CLUSTER MMA atom (via bs_op_selector with cluster > 1)
//   4. Multicast masks computed per-CTA for A/SFA and B/SFB loads
//   5. Cluster-aware tma_partition with cta_coord/cta_layout projection
//   6. cluster_sync() barriers for cross-work-group synchronization
//   7. MMA .with() passes multicast masks for AB and DAB tracking
//   8. SYCL dimension ordering: (K, M, N) — N-fast hardware linearization
//   9. Config::EnableCooperativeSF (default false): when enabled, SF tiles
//      are loaded cooperatively across the cluster (same as SM100), with each
//      CTA loading 1/N-th of the SF tile via 5-arg tma_partition. When
//      disabled, each CTA independently loads the full SF tile using a
//      trivial (1,1,1) cluster layout for the SF ADMA atoms.
//
// Driver files supply a Config struct specifying all the same members as
// the non-cluster base, plus ClusterShape_MNK.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <type_traits>

#include <sycl/sycl.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gpu_generics.h>

#include "cute/arch/mma_xe4_amma.hpp"
#include <cute/arch/mma_xe4.hpp>
#include <cute/atom/mma_traits_xe4_amma.hpp>
#include <cute/atom/copy_traits_xe4_adma.hpp>
#include <cute/arch/xe4_util.hpp>
#include <cute/arch/cluster_xe4.hpp>
#include "cutlass/arch/barrier.h"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/pipeline/pipeline.hpp"

#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>

#include <cutlass/float_subbyte.h>

#include "../../../common/sycl_cute_common.hpp"
#include "../../../xe4/utils/validation.hpp"
#include "cutlass/detail/xe4_blockscaled_layout.hpp"

namespace xe4_blockscaled_gemm_cluster {

using namespace cute;
using sycl::ext::oneapi::this_work_item::get_nd_item;
using cutlass::detail::Xe4BlockScaledConfig;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Constants
//
///////////////////////////////////////////////////////////////////////////////////////////////////

constexpr static size_t SmemAlignment = 512;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SMEM tensor construction helper — sub-byte vs regular pointer dispatch
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename Storage, typename Layout>
auto make_data_smem_tensor(Storage& storage, Layout layout) {
  if constexpr (cute::sizeof_bits_v<T> < 8) {
    return make_tensor(make_smem_ptr(subbyte_iterator<T>(storage.data())), layout);
  } else {
    return make_tensor(make_smem_ptr(reinterpret_cast<T*>(storage.begin())), layout);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// BlockScaledSharedStorage: SMEM buffers for A, B, C, SFA, SFB
//
// SF allocation uses pipe_stride * num_stages (not cosize) to include
// padding room for the last pipeline stage's cm_8x32B overflow.
// When sf_bK_actual < 8, the padded pipe stride is larger than the atom's
// non-zero extent, so cosize alone would miss the last stage's padding gap.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

// Padded SF SMEM allocation: pipe_stride * pipe_count.
// SF layouts are rank-4: (atom_modes..., pipe). The pipe dimension (index 3)
// has stride = SmemSizeSingleBuffer and size = PipelineStages.
template <class SmemLayoutSF>
inline constexpr int padded_sf_smem_v =
    decltype(cute::stride<3>(SmemLayoutSF{}) * cute::size<3>(SmemLayoutSF{}))::value;

template <class ElementA_,  class ElementB_,  class ElementC_,  class ElementSF_, class ElementAcc_,
          class SmemLayoutA,  class SmemLayoutB,  class SmemLayoutC,
          class SmemLayoutSFA, class SmemLayoutSFB,
          bool SameCD = std::is_same_v<ElementC_, ElementAcc_>>
struct BlockScaledSharedStorage;

// Specialization when ElementC != ElementAcc: separate smem_D buffer
template <class ElementA_,  class ElementB_,  class ElementC_,  class ElementSF_, class ElementAcc_,
          class SmemLayoutA,  class SmemLayoutB,  class SmemLayoutC,
          class SmemLayoutSFA, class SmemLayoutSFB>
struct BlockScaledSharedStorage<ElementA_, ElementB_, ElementC_, ElementSF_, ElementAcc_,
                                SmemLayoutA, SmemLayoutB, SmemLayoutC,
                                SmemLayoutSFA, SmemLayoutSFB, false>
  : cute::aligned_struct<SmemAlignment, _0>
{
  cute::array_aligned<uint8_t,
     (cute::cosize_v<SmemLayoutA> * cutlass::sizeof_bits<ElementA_>::value + 7) / 8,
     SmemAlignment> smem_A;
  cute::array_aligned<uint8_t,
     (cute::cosize_v<SmemLayoutB> * cutlass::sizeof_bits<ElementB_>::value + 7) / 8,
     SmemAlignment> smem_B;
  cute::array_aligned<ElementAcc_,  cute::cosize_v<SmemLayoutC>,   SmemAlignment> smem_C;
  cute::array_aligned<ElementC_,  cute::cosize_v<SmemLayoutC>,   SmemAlignment> smem_D;
  cute::array_aligned<ElementSF_, padded_sf_smem_v<SmemLayoutSFA>, SmemAlignment> smem_SFA;
  cute::array_aligned<ElementSF_, padded_sf_smem_v<SmemLayoutSFB>, SmemAlignment> smem_SFB;
};

// Specialization when ElementC == ElementAcc: D aliases C
template <class ElementA_,  class ElementB_,  class ElementC_,  class ElementSF_, class ElementAcc_,
          class SmemLayoutA,  class SmemLayoutB,  class SmemLayoutC,
          class SmemLayoutSFA, class SmemLayoutSFB>
struct BlockScaledSharedStorage<ElementA_, ElementB_, ElementC_, ElementSF_, ElementAcc_,
                                SmemLayoutA, SmemLayoutB, SmemLayoutC,
                                SmemLayoutSFA, SmemLayoutSFB, true>
  : cute::aligned_struct<SmemAlignment, _0>
{
  cute::array_aligned<uint8_t,
     (cute::cosize_v<SmemLayoutA> * cutlass::sizeof_bits<ElementA_>::value + 7) / 8,
     SmemAlignment> smem_A;
  cute::array_aligned<uint8_t,
     (cute::cosize_v<SmemLayoutB> * cutlass::sizeof_bits<ElementB_>::value + 7) / 8,
     SmemAlignment> smem_B;
  cute::array_aligned<ElementAcc_,  cute::cosize_v<SmemLayoutC>,   SmemAlignment> smem_C;
  cute::array_aligned<ElementSF_, padded_sf_smem_v<SmemLayoutSFA>, SmemAlignment> smem_SFA;
  cute::array_aligned<ElementSF_, padded_sf_smem_v<SmemLayoutSFB>, SmemAlignment> smem_SFB;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Helper Functions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

inline void xe4_syncthreads() {
  auto group = get_nd_item<3>().get_group();
  sycl::group_barrier(group);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Compute multicast masks for 2D cluster arrangement.
//
// SYCL cluster range is (K, M, N) so:
//   get_cluster_wgid<0>() = X = dim-2 = N position within cluster
//   get_cluster_wgid<1>() = Y = dim-1 = M position within cluster
//
// Hardware linearization: rank = M_pos * N_size + N_pos (N-fast)
//
// A (shared across N at same M): mask_a = N_size consecutive bits at M_pos * N_size
// B (shared across M at same N): mask_b = bits at N_pos, N_pos+N_size, ...
//
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class ClusterShape_MNK>
CUTE_DEVICE auto
compute_cluster_masks(ClusterShape_MNK cluster_shape)
{
  auto [cluster_size_m, cluster_size_n, cluster_size_k] = cluster_shape;

  // SYCL cluster range = (K, M, N): dim-2(X)=N, dim-1(Y)=M
  uint32_t cluster_wgid_n = get_cluster_wgid<0>();  // N position (SYCL X / dim-2)
  uint32_t cluster_wgid_m = get_cluster_wgid<1>();  // M position (SYCL Y / dim-1)

  uint32_t cluster_mask_a = 0;
  uint32_t cluster_mask_b = 0;

  uint32_t coop_num_a = cluster_size_n;   // CTAs cooperating on A = along N
  uint32_t coop_num_b = cluster_size_m;   // CTAs cooperating on B = along M

  // A: N_size consecutive bits starting at M_pos * N_size
  cluster_mask_a = ((1u << coop_num_a) - 1) << (cluster_wgid_m * coop_num_a);

  // B: bits at positions N_pos, N_pos + N_size, N_pos + 2*N_size, ...
  uint32_t cluster_mask_b_base = 1u << cluster_wgid_n;
  #pragma unroll
  for (uint32_t i = 0; i < coop_num_b; i++) {
    cluster_mask_b |= cluster_mask_b_base << (i * coop_num_a);
  }

  return make_tuple(cluster_mask_a, cluster_mask_b);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Device Kernel: Block-Scaled GEMM with AMMA + ADMA Pipelining + Cluster Multicast
//
// Barrier Allocation (6 abar slots: 0-5):
//   abar 0 : load_a_abar   — Tracks A data load completion
//   abar 1 : load_sfa_abar — Tracks SFA load completion
//   abar 2 : load_b_abar   — Tracks B data load completion
//   abar 3 : load_sfb_abar — Tracks SFB load completion
//   abar 4 : mma_abar      — Tracks MMA AB completion (consumer → producer signal)
//   abar 5 : store_c_abar  — Tracks C store + MMA D completion
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Config, class ClusterShape,
          class ProblemShape, class CtaTiler, class TileShape,
          class SmemLayoutA, class SmemLayoutC, class ADMA_A,
          class SmemLayoutB, class ADMA_B, class ADMA_C,
          class CStride,
          class SmemLayoutSFA, class SmemLayoutSFB,
          class ADMA_SFA, class ADMA_SFB,
          class TiledMma,
          class Alpha, class Beta>
void
gemm_device_blockscaled_cluster(ProblemShape shape_MNK, CtaTiler cta_tiler, TileShape tile_shape,
                                ClusterShape cluster_shape,
                                typename Config::ElementA const* A,
                                SmemLayoutA sA_layout, SmemLayoutC sC_layout, ADMA_A adma_load_a,
                                typename Config::ElementB const* B,
                                SmemLayoutB sB_layout, ADMA_B adma_load_b, ADMA_C adma_store_c,
                                typename Config::ElementC* C, CStride dC,
                                typename Config::ElementSF const* SFA, SmemLayoutSFA sSFA_layout, ADMA_SFA adma_load_sfa,
                                typename Config::ElementSF const* SFB, SmemLayoutSFB sSFB_layout, ADMA_SFB adma_load_sfb,
                                Alpha alpha, Beta beta,
                                sycl::nd_item<3> item)
{
  using ElementA  = typename Config::ElementA;
  using ElementB  = typename Config::ElementB;
  using ElementC  = typename Config::ElementC;
  using ElementSF = typename Config::ElementSF;
  using ElementAcc = typename Config::ElementAcc;
  static constexpr int SFVecSize = Config::SFVecSize;

  // Preconditions
  static_assert(rank(ProblemShape{}) == 3);
  static_assert(rank(CtaTiler{}) == 3);

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  static_assert(size<0>(SmemLayoutA{}) == size<0>(CtaTiler{}));  // BLK_M
  static_assert(size<0>(SmemLayoutB{}) == size<1>(CtaTiler{}));  // BLK_N
  static_assert(size<1>(SmemLayoutA{}) == size<2>(CtaTiler{}));  // BLK_K
  static_assert(size<1>(SmemLayoutB{}) == size<2>(CtaTiler{}));  // BLK_K

  // ---- Allocate tensor descriptors for ADMA Copy ----
  auto tdesc_a   = allocate_tdesc<0>();
  auto tdesc_b   = allocate_tdesc<1>();
  auto tdesc_c   = allocate_tdesc<2>();
  auto tdesc_sfa = allocate_tdesc<3>();
  auto tdesc_sfb = allocate_tdesc<4>();

  adma_load_a.set_tensor_desc(tdesc_a);
  adma_load_b.set_tensor_desc(tdesc_b);
  adma_store_c.set_tensor_desc(tdesc_c);
  adma_load_sfa.set_tensor_desc(tdesc_sfa);
  adma_load_sfb.set_tensor_desc(tdesc_sfb);

  // ---- Allocate SLM and create SMEM tensors ----
  using SharedStorageType = BlockScaledSharedStorage<
    ElementA, ElementB, ElementC, ElementSF, ElementAcc,
    SmemLayoutA, SmemLayoutB, SmemLayoutC, SmemLayoutSFA, SmemLayoutSFB>;

  auto ptr = alloc_slm_buffer<uint8_t, sizeof(SharedStorageType)>(item.get_group());
  auto& smem = *reinterpret_cast<SharedStorageType*>(ptr);

  Tensor sA = make_data_smem_tensor<ElementA>(smem.smem_A, SmemLayoutA{});
  Tensor sB = make_data_smem_tensor<ElementB>(smem.smem_B, SmemLayoutB{});
  Tensor sC = make_tensor(make_smem_ptr(smem.smem_C.begin()), SmemLayoutC{});

  [[maybe_unused]] auto sD_tensor = [&]() {
    if constexpr (std::is_same_v<ElementC, ElementAcc>) {
      return make_tensor(make_smem_ptr(reinterpret_cast<ElementC*>(smem.smem_C.begin())), SmemLayoutC{});
    } else {
      return make_tensor(make_smem_ptr(smem.smem_D.begin()), SmemLayoutC{});
    }
  }();
  auto sD = sD_tensor;

  Tensor sSFA = make_tensor(make_smem_ptr(smem.smem_SFA.begin()), SmemLayoutSFA{});
  Tensor sSFB = make_tensor(make_smem_ptr(smem.smem_SFB.begin()), SmemLayoutSFB{});

  // ---- Create global memory tensor views ----
  auto [M, N, K] = shape_MNK;
  auto mA   = adma_load_a.get_tma_tensor(make_shape(M, K));
  auto mB   = adma_load_b.get_tma_tensor(make_shape(N, K));
  auto mC   = adma_store_c.get_tma_tensor(make_shape(M, N));

  constexpr int SFVecSizeK = SFVecSize;
  auto mSFA = adma_load_sfa.get_tma_tensor(
      make_shape(make_shape(Int<1>{}, M), make_shape(Int<SFVecSizeK>{}, K / SFVecSizeK)));
  auto mSFB = adma_load_sfb.get_tma_tensor(
      make_shape(make_shape(Int<1>{}, N), make_shape(Int<SFVecSizeK>{}, K / SFVecSizeK)));

  // ---- Cluster-aware tile partitioning ----
  // SYCL range (K, M, N): dim-1(Y)=M, dim-2(X)=N
  //   BlockIdxY() = get_group(1) = M tile index (global)
  //   BlockIdxX() = get_group(2) = N tile index (global)
  auto cta_coord = make_coord(BlockIdxY(), BlockIdxX(), _);

  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1,  X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1,  X>{});

  Tensor gSFA = local_tile(mSFA, cta_tiler, cta_coord, Step<_1,  X, _1>{});
  Tensor gSFB = local_tile(mSFB, cta_tiler, cta_coord, Step< X, _1, _1>{});

  // ---- Cluster-aware TMA partition (cooperative loading for A/B) ----
  //
  // With cluster > 1, each CTA loads a portion of A/B cooperatively.
  // tma_partition uses the CTA's position within the cluster to determine
  // which portion of shared memory this CTA is responsible for.
  //   A: projected along N-modes (N-CTAs cooperatively load A)
  //   B: projected along M-modes (M-CTAs cooperatively load B)
  //   SFA/SFB: cooperative or non-cooperative based on Config::EnableCooperativeSF
  TiledMma mma{};
  auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));

  // Block rank within the cluster: consistent with SYCL N-fast linearization
  // cluster_layout_mn: shape=(M,N), stride=(1,M_size) → M-fast CuTe ordering
  auto cluster_layout_mn = make_layout(select<0,1>(cluster_shape),
                                       make_stride(_1{}, get<0>(cluster_shape)));
  // With SYCL range (K,M,N): get_cluster_wgid<1>()=M_pos(Y), get_cluster_wgid<0>()=N_pos(X)
  uint32_t block_rank = cluster_layout_mn(make_coord(get_cluster_wgid<1>(), get_cluster_wgid<0>()));
  auto cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(block_rank);

  // Partition A: project along N-modes (CTAs along N share the same A tile)
  auto [tAgA, tAsA] = tma_partition(adma_load_a,
                                    get<2>(cta_coord_vmnk), make_layout(size<2>(cluster_layout_vmnk)),
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));

  // Partition B: project along M-modes (CTAs along M share the same B tile)
  auto [tBgB, tBsB] = tma_partition(adma_load_b,
                                    get<1>(cta_coord_vmnk), make_layout(size<1>(cluster_layout_vmnk)),
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));

  // SF partitions: cooperative or non-cooperative based on Config::EnableCooperativeSF.
  static constexpr bool EnableCoopSF = Config::EnableCooperativeSF;

  auto [tSFAgSFA, tSFAsSFA] = [&]() {
    if constexpr (EnableCoopSF) {
      // SF cooperative via standard tma_partition (same as SM100).
      // The ADMA box is truncated by num_multicast in make_adma_copy_desc, so each
      // CTA loads 1/N-th of the SF tile.  tma_partition splits the logical TMA box
      // and computes distinct SMEM offsets + GMEM coordinates for each CTA.
      // SFA projects along N-modes (same as data A).
      return tma_partition(adma_load_sfa,
                           get<2>(cta_coord_vmnk), make_layout(size<2>(cluster_layout_vmnk)),
                           group_modes<0,3>(sSFA), group_modes<0,2>(gSFA));
    } else {
      // Non-cooperative (3-arg tma_partition).
      // Each CTA independently loads the full SF tile.
      return tma_partition(adma_load_sfa,
                           group_modes<0,3>(sSFA), group_modes<0,2>(gSFA));
    }
  }();

  auto [tSFBgSFB, tSFBsSFB] = [&]() {
    if constexpr (EnableCoopSF) {
      // SFB projects along M-modes (same as data B).
      return tma_partition(adma_load_sfb,
                           get<1>(cta_coord_vmnk), make_layout(size<1>(cluster_layout_vmnk)),
                           group_modes<0,3>(sSFB), group_modes<0,2>(gSFB));
    } else {
      // Non-cooperative (3-arg tma_partition).
      // Each CTA independently loads the full SF tile.
      return tma_partition(adma_load_sfb,
                           group_modes<0,3>(sSFB), group_modes<0,2>(gSFB));
    }
  }();

  // ---- Pipeline configuration ----
  auto K_PIPE_MAX = size<1>(tAsA);
  int  K_TILE_MAX = size<1>(tAgA);
  int  k_tile = 0;

  constexpr int dma_bytes_A   = (cosize(SmemLayoutA{}) * sizeof_bits_v<ElementA> / 8) / decltype(K_PIPE_MAX)::value;
  constexpr int dma_bytes_B   = (cosize(SmemLayoutB{}) * sizeof_bits_v<ElementB> / 8) / decltype(K_PIPE_MAX)::value;
  constexpr int dma_bytes_C   = cosize(SmemLayoutC{}) * sizeof(ElementC);
  // SF barrier bytes use the actual ADMA transfer size (bMN * bK/SFVecSize), not the
  // padded SmemLayout cosize.  When sf_bK_actual < 8, the padded layout has gaps between
  // pipeline stages; those gaps are not part of the DMA transfer.
  constexpr int dma_sf_bK_actual = size<2>(CtaTiler{}) / SFVecSize;
  constexpr int dma_bytes_SFA = size<0>(CtaTiler{}) * dma_sf_bK_actual * static_cast<int>(sizeof(ElementSF));
  constexpr int dma_bytes_SFB = size<1>(CtaTiler{}) * dma_sf_bK_actual * static_cast<int>(sizeof(ElementSF));

  // ---- Compute multicast masks based on 2D cluster position ----
  auto [mcast_mask_a, mcast_mask_b] = compute_cluster_masks(cluster_shape);

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t warp_idx      = get_sg_id();

  // ---- Allocate asynchronous barriers ----
  auto load_a_abar   = allocate_abar<0, K_PIPE_MAX>();
  auto load_sfa_abar = allocate_abar<1, K_PIPE_MAX>();
  auto load_b_abar   = allocate_abar<2, K_PIPE_MAX>();
  auto load_sfb_abar = allocate_abar<3, K_PIPE_MAX>();
  auto mma_abar      = allocate_abar<4, K_PIPE_MAX>();
  auto store_c_abar  = allocate_abar<5>();

  if (elect_one_thr && warp_idx == 0) {
    for (int i = 0; i < K_PIPE_MAX; ++i) {
      xe4_initialize_barrier(load_a_abar[i], 1);
      xe4_initialize_barrier(load_b_abar[i], 1);
      xe4_initialize_barrier(load_sfa_abar[i], 1);
      xe4_initialize_barrier(load_sfb_abar[i], 1);
    }
  } else if (elect_one_thr && warp_idx == 1) {
     for (int i = 0; i < K_PIPE_MAX; ++i) {
      xe4_initialize_barrier(mma_abar[i], 1);
    }
    xe4_initialize_barrier(store_c_abar[0], 1);
  }
  xe4_syncthreads();

  int store_c_phase_bit = 0;

  // ---- Partition for MMA ----
  ThrMMA thr_mma = mma.get_thread_slice(ThreadIdxX());
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCsC = thr_mma.partition_C(sC);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCsC);

  Tensor tCsD = thr_mma.partition_C(sD);
  Tensor tCrD = thr_mma.make_fragment_C(tCsD);

  // SF descriptor tensors
  Tensor tCrSFA = make_tensor<AMMA::smem_sf_desc>(sSFA);
  Tensor tCrSFB = make_tensor<AMMA::smem_sf_desc>(sSFB);

  // Clear accumulators in SLM
  clear(tCsC);

  xe4_syncthreads();

  // ---- Cluster sync before pipeline ----
  // All CTAs in the cluster must have SLM initialized before multicast DMA.
  if constexpr (size(ClusterShape{}) > 1) {
    cluster_sync();
  }

  // Pipeline state
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();

  // MMA control:
  //   NullC=1 for first iteration (bypass C read).
  //   BlockScaleType from Config.
  MMAControl mma_ctrl{};
  mma_ctrl.NullC = 1;
  mma_ctrl.A_BlockScaleType = Config::BlockScaleType;
  mma_ctrl.B_BlockScaleType = Config::BlockScaleType;

  // ==================================================================
  // Warp 0: ADMA Producer — Load A, B, SFA, SFB with multicast
  //   A + SFA loads use mcast_mask_a (delivers to all CTAs in same M-row)
  //   B + SFB loads use mcast_mask_b (delivers to all CTAs in same N-column)
  // ==================================================================
  if (warp_idx == 0)
  {
    if (elect_one_thr)
    {
      // ---- Prologue: Fill pipeline stages ----
      CUTLASS_PRAGMA_UNROLL
      for (int pipe = 0; pipe < K_PIPE_MAX && k_tile < K_TILE_MAX; ++pipe)
      {
        xe4_set_barrier_transaction_bytes(load_a_abar[pipe], dma_bytes_A);
        xe4_set_barrier_transaction_bytes(load_b_abar[pipe], dma_bytes_B);
        xe4_set_barrier_transaction_bytes(load_sfa_abar[pipe], dma_bytes_SFA);
        xe4_set_barrier_transaction_bytes(load_sfb_abar[pipe], dma_bytes_SFB);

        copy(adma_load_a.with(&load_a_abar[pipe], mcast_mask_a),       tAgA(_, k_tile),     tAsA(_, pipe));
        copy(adma_load_sfa.with(&load_sfa_abar[pipe], mcast_mask_a),   tSFAgSFA(_, k_tile), tSFAsSFA(_, pipe));
        copy(adma_load_b.with(&load_b_abar[pipe], mcast_mask_b),       tBgB(_, k_tile),     tBsB(_, pipe));
        copy(adma_load_sfb.with(&load_sfb_abar[pipe], mcast_mask_b),   tSFBgSFB(_, k_tile), tSFBsSFB(_, pipe));
        ++k_tile;
      }

      // ---- Mainloop: Wait for MMA to consume, then refill freed stage ----
      for (; k_tile < K_TILE_MAX; ++k_tile)
      {
        int pipe = write_state.index();

        xe4_wait_barrier(mma_abar[pipe], write_state.phase());

        xe4_set_barrier_transaction_bytes(load_a_abar[pipe], dma_bytes_A);
        xe4_set_barrier_transaction_bytes(load_b_abar[pipe], dma_bytes_B);
        xe4_set_barrier_transaction_bytes(load_sfa_abar[pipe], dma_bytes_SFA);
        xe4_set_barrier_transaction_bytes(load_sfb_abar[pipe], dma_bytes_SFB);

        copy(adma_load_a.with(&load_a_abar[pipe], mcast_mask_a),       tAgA(_, k_tile),     tAsA(_, pipe));
        copy(adma_load_sfa.with(&load_sfa_abar[pipe], mcast_mask_a),   tSFAgSFA(_, k_tile), tSFAsSFA(_, pipe));
        copy(adma_load_b.with(&load_b_abar[pipe], mcast_mask_b),       tBgB(_, k_tile),     tBsB(_, pipe));
        copy(adma_load_sfb.with(&load_sfb_abar[pipe], mcast_mask_b),   tSFBgSFB(_, k_tile), tSFBsSFB(_, pipe));

        ++write_state;
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }
  // ==================================================================
  // Warp 1: AMMA Consumer — Execute block-scaled MMA with cluster masks
  // ==================================================================
  else if (warp_idx == 1)
  {
    if (elect_one_thr)
    {
      // Mainloop: Wait for loads, execute MMA, signal producer
      // Unroll the K mode manually so we can set mma_ctrl and barrier tracking
      for (int k_tile_next = 0; k_tile_next < K_TILE_MAX; ++k_tile_next)
      {
        int read_pipe = read_state.index();

        xe4_wait_barrier(load_a_abar[read_pipe], read_state.phase());
        xe4_wait_barrier(load_b_abar[read_pipe], read_state.phase());
        xe4_wait_barrier(load_sfa_abar[read_pipe], read_state.phase());
        xe4_wait_barrier(load_sfb_abar[read_pipe], read_state.phase());

        // Set barrier expected bytes ONCE for all k_blocks in this pipeline stage.
        // All k_blocks signal AB (2 bytes each) to mma_abar.
        // On the last tile, last k_block additionally signals D (1 byte) to store_c_abar.
        int num_k = int(size<2>(tCrA));
        bool is_last_tile = (k_tile_next == K_TILE_MAX - 1);
        xe4_set_barrier_transaction_bytes(mma_abar[read_pipe], 2 * num_k);
        if (is_last_tile) {
          xe4_set_barrier_transaction_bytes(store_c_abar[0], 1);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < num_k; ++k_block) {
          bool is_last_iter = is_last_tile && (k_block == num_k - 1);

          if (is_last_iter) {
            // Last k_block of last tile: Track D+A+B completion for the store barrier
            if constexpr (std::is_same_v<ElementC, ElementAcc>) {
              // Same type: no dtype change, 3-operand DAB tracking
              auto new_mma = mma.with(
                ElementAcc{},
                AMMA::TrackMethod<AMMA::Tracking::DAB>{},
                mma_ctrl,
                tCrSFA(0, 0, k_block, read_pipe), tCrSFB(0, 0, k_block, read_pipe),
                &store_c_abar[0],                                             // D → dedicated epilogue barrier
                &mma_abar[read_pipe], &mma_abar[read_pipe],                  // A,B → pipeline slot barrier
                mcast_mask_a, mcast_mask_b);
              cute::gemm(new_mma, tCrA(_,_,k_block,read_pipe), tCrB(_,_,k_block,read_pipe), tCrC);
            } else {
              // Dtype change: D(FP16/BF16) to separate buffer, 4-operand DAB tracking
              auto new_mma = mma.with(
                ElementC{},
                AMMA::TrackMethod<AMMA::Tracking::DAB>{},
                mma_ctrl,
                tCrSFA(0, 0, k_block, read_pipe), tCrSFB(0, 0, k_block, read_pipe),
                &store_c_abar[0],                                             // D → dedicated epilogue barrier
                &mma_abar[read_pipe], &mma_abar[read_pipe],                  // A,B → pipeline slot barrier
                mcast_mask_a, mcast_mask_b);
              cute::gemm(new_mma, tCrD, tCrA(_,_,k_block,read_pipe), tCrB(_,_,k_block,read_pipe), tCrC);
            }
          } else {
            // Non-last k_blocks: Track AB consumption for pipeline feedback
            auto new_mma = mma.with(
              ElementAcc{},
              AMMA::TrackMethod<AMMA::Tracking::AB>{},
              mma_ctrl,
              tCrSFA(0, 0, k_block, read_pipe), tCrSFB(0, 0, k_block, read_pipe),
              &mma_abar[read_pipe], &mma_abar[read_pipe],
              mcast_mask_a, mcast_mask_b);
            cute::gemm(new_mma, tCrA(_,_,k_block,read_pipe), tCrB(_,_,k_block,read_pipe), tCrC);
          }
          // After first fma, switch to accumulate mode: NullC=0 reads AMMA-written D from SLM.
          mma_ctrl.NullC = 0;
        }
        ++read_state;
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }

  xe4_syncthreads();

  // ==================================================================
  // Epilogue: Store C from SLM → GMEM via ADMA
  // ==================================================================
  if (warp_idx == 0)
  {
    if (elect_one_thr)
    {
      // Wait for MMA's D-completion signal on store_c_abar (phase 0 → signaled by last DAB)
      xe4_wait_barrier(store_c_abar[0], 0);
      xe4_set_barrier_transaction_bytes(store_c_abar[0], dma_bytes_C);
      if constexpr (std::is_same_v<ElementC, ElementAcc>) {
        copy(adma_store_c.with(&store_c_abar[0]), tCsC, tCgC);
      } else {
        copy(adma_store_c.with(&store_c_abar[0]), tCsD, tCgC);
      }
      xe4_wait_barrier(store_c_abar[0], store_c_phase_bit);
      store_c_phase_bit ^= 1;
    }
  }

  xe4_syncthreads();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host Setup: TN GEMM with Block Scaling + Cluster (parameterized on Config)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Config,
          class TensorA, class TensorB, class TensorC,
          class TensorSFA, class TensorSFB,
          class Alpha, class Beta>
void
gemm_tn_blockscaled_cluster(int m, int n, int k,
                            Alpha alpha,
                            TensorA const& A,
                            TensorB const& B,
                            Beta beta,
                            TensorC& C,
                            TensorSFA const& SFA,
                            TensorSFB const& SFB,
                            sycl::queue& queue)
{
  using ElementA  = typename Config::ElementA;
  using ElementB  = typename Config::ElementB;
  using ElementC  = typename Config::ElementC;
  using ElementSF = typename Config::ElementSF;
  using TiledMma  = typename Config::TiledMma;
  using TileShape_MNK = typename Config::TileShape_MNK;
  using ClusterShape_MNK = typename Config::ClusterShape_MNK;
  static constexpr int SFVecSize       = Config::SFVecSize;
  static constexpr int PipelineStages  = Config::PipelineStages;
  static constexpr auto majorA = Config::MajorA;
  static constexpr auto majorB = Config::MajorB;

  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  ElementA  const* A_ptr   = &*A.data();
  ElementB  const* B_ptr   = &*B.data();
  auto C_ptr   = &*C.data();
  auto SFA_ptr = &*SFA.data();
  auto SFB_ptr = &*SFB.data();

  auto dA = A.stride();
  auto dB = B.stride();
  auto dC = C.stride();

  Tensor mA = make_tensor(make_gmem_ptr(A_ptr), make_layout(make_shape(M, K), dA));
  Tensor mB = make_tensor(make_gmem_ptr(B_ptr), make_layout(make_shape(N, K), dB));
  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), make_layout(make_shape(M, N), dC));

  using BlkScaledConfig = Xe4BlockScaledConfig<SFVecSize>;
  auto layout_SFA = BlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K));
  auto layout_SFB = BlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K));

  Tensor mSFA = make_tensor(make_gmem_ptr(SFA_ptr), layout_SFA);
  Tensor mSFB = make_tensor(make_gmem_ptr(SFB_ptr), layout_SFB);

  // ---- Tile and cluster shapes ----
  constexpr auto bM = get<0>(TileShape_MNK{});
  constexpr auto bN = get<1>(TileShape_MNK{});
  constexpr auto bK = get<2>(TileShape_MNK{});
  auto cta_tiler = make_shape(bM, bN, bK);
  constexpr auto bP = Int<PipelineStages>{};

  // ---- SMEM layouts ----
  using SmemLayoutAtomA =
    cute::conditional_t<
      majorA == cute::AMMA::Major::K,
      decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenRowMajor{})),
      decltype(make_layout(cute::select<0, 2>(TileShape_MNK{}), GenColMajor{}))
    >;
  using SmemLayoutAtomB =
    cute::conditional_t<
      majorB == cute::AMMA::Major::K,
      decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenRowMajor{})),
      decltype(make_layout(cute::select<1, 2>(TileShape_MNK{}), GenColMajor{}))
    >;

  using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
    cute::conditional_t<majorA == cute::AMMA::Major::K, Step<_2, _1, _3>, Step<_1, _2, _3>>{}));

  using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{},
    make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), bP),
    cute::conditional_t<majorB == cute::AMMA::Major::K, Step<_2, _1, _3>, Step<_1, _2, _3>>{}));

  using SmemLayoutAtomC =
    cute::conditional_t<
      majorA == cute::AMMA::Major::K,
      decltype(make_layout(cute::select<0, 1>(TileShape_MNK{}), GenRowMajor{})),
      decltype(make_layout(cute::select<0, 1>(TileShape_MNK{}), GenColMajor{}))
    >;

  using SmemLayoutC = decltype(tile_to_shape(
    SmemLayoutAtomC{},
    make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
    Step<_2, _1>{}));

  // ---- SMEM layouts for scale factors ----
  TiledMma tiled_mma_inst{};
  auto xe4_sfA_layout_atom = BlkScaledConfig::deduce_smem_layoutSFA(tiled_mma_inst, TileShape_MNK{});
  auto xe4_sfB_layout_atom = BlkScaledConfig::deduce_smem_layoutSFB(tiled_mma_inst, TileShape_MNK{});

  // SF padding for cm_8x32B core-matrix alignment.
  // The ADMA 2D-block-copy always writes in cm_8x32B units (8 rows minimum).
  // When sf_bK_actual < 8, overflow rows would land in the next pipeline stage's
  // SF region.  Padding the pipe stride to sf_bK = max(sf_bK_actual, 8) creates
  // a gap between stages so overflow lands in zero-initialized padding, not live data.
  static constexpr int sf_bK_actual = get<2>(TileShape_MNK{}) / SFVecSize;
  static constexpr int sf_bK = (sf_bK_actual < 8) ? 8 : sf_bK_actual;
  static constexpr int SmemSizeSingleBufferSFA = get<0>(TileShape_MNK{}) * sf_bK;
  static constexpr int SmemSizeSingleBufferSFB = get<1>(TileShape_MNK{}) * sf_bK;

  using SmemLayoutSFA = decltype(make_layout(
    append(shape(decltype(xe4_sfA_layout_atom){}), bP),
    append(stride(decltype(xe4_sfA_layout_atom){}), Int<SmemSizeSingleBufferSFA>{})
  ));

  using SmemLayoutSFB = decltype(make_layout(
    append(shape(decltype(xe4_sfB_layout_atom){}), bP),
    append(stride(decltype(xe4_sfB_layout_atom){}), Int<SmemSizeSingleBufferSFB>{})
  ));

  SmemLayoutA   sA{};
  SmemLayoutB   sB{};
  SmemLayoutC   sC{};
  SmemLayoutSFA sSFA{};
  SmemLayoutSFB sSFB{};

  // ---- ADMA copy atoms: multicast when cluster > 1 ----
  using GmemTiledCopyA =
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::XE4_ADMA_LOAD, cute::XE4_ADMA_LOAD_MULTICAST>;
  using GmemTiledCopyB =
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::XE4_ADMA_LOAD, cute::XE4_ADMA_LOAD_MULTICAST>;
  using GmemTiledCopySF =
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::XE4_ADMA_LOAD, cute::XE4_ADMA_LOAD_MULTICAST>;
  using GmemTiledCopyC = cute::XE4_ADMA_STORE;

  TiledMma tiled_mma{};
  // Cluster layout for TMA partition projection — cooperative loading.
  // Matches xe4_mma_warpspecialized.hpp: ADMA atoms are created with the real
  // cluster layout so num_multicast > 1, enabling cooperative tile loading.
  auto cluster_layout_vmnk = tiled_divide(
    make_layout(ClusterShape_MNK{}),
    make_tile(typename TiledMma::AtomThrID{}));

  auto adma_load_a = make_adma_atom_A_xe4(
    GmemTiledCopyA{}, mA,
    SmemLayoutA{}(_, _, cute::Int<0>{}),
    TileShape_MNK{}, TiledMma{}, cluster_layout_vmnk);

  auto adma_load_b = make_adma_atom_B_xe4(
    GmemTiledCopyB{}, mB,
    SmemLayoutB{}(_, _, cute::Int<0>{}),
    TileShape_MNK{}, TiledMma{}, cluster_layout_vmnk);

  auto cta_tiler_mn = make_shape(bM, bN);
  auto adma_store_c = make_adma_copy<ElementC>(GmemTiledCopyC{}, mC, SmemLayoutC{}, cta_tiler_mn, Int<1>{});

  // SF ADMA atoms: select cluster layout based on EnableCooperativeSF.
  //
  // When cooperative SF loading is DISABLED (default):
  //   SF ADMA atoms use trivial (1,1,1) cluster layout (num_multicast=1).
  //   The SF SMEM layouts contain stride-0 broadcast modes (from SFVecSize) that
  //   cause the cooperative tma_partition split to happen along K-blocks instead
  //   of along M/N. This results in each CTA loading only half the K-blocks of
  //   SF, producing incorrect results (K/2 values). Since SF data is much smaller
  //   than A/B data, the bandwidth savings from cooperative SF loading are minimal.
  //   Each CTA independently loads the full SF tile; the multicast mask in .with()
  //   still delivers SF data to peer SLMs for cluster MMA consumption.
  //
  // When cooperative SF loading is ENABLED:
  //   SF ADMA atoms use the real cluster layout (same as SM100 block-scaled GEMM).
  //   The ADMA box is truncated by num_multicast in make_adma_copy_desc, so each
  //   CTA loads 1/N-th of the SF tile. tma_partition splits the logical TMA box
  //   and computes distinct SMEM offsets + GMEM coordinates for each CTA.
  //   SFA projects along N-modes (same as data A), SFB along M-modes (same as data B).
  static constexpr bool EnableCoopSF = Config::EnableCooperativeSF;

  auto sf_cluster_layout_vmnk = [&]() {
    if constexpr (EnableCoopSF) {
      return cluster_layout_vmnk;
    } else {
      using TrivialCluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
      return tiled_divide(
        make_layout(TrivialCluster{}),
        make_tile(typename TiledMma::AtomThrID{}));
    }
  }();

  auto adma_load_sfa = make_adma_atom_A_xe4(
    GmemTiledCopySF{}, mSFA,
    SmemLayoutSFA{}(_, _, _, cute::Int<0>{}),
    TileShape_MNK{}, TiledMma{}, sf_cluster_layout_vmnk);

  auto adma_load_sfb = make_adma_atom_B_xe4(
    GmemTiledCopySF{}, mSFB,
    SmemLayoutSFB{}(_, _, _, cute::Int<0>{}),
    TileShape_MNK{}, TiledMma{}, sf_cluster_layout_vmnk);

  // ---- Launch configuration ----
  // SYCL dimension mapping: dim-0=K, dim-1=M, dim-2=N (N-fast)
  constexpr int NumControlWarps = 2;
  constexpr int NumThreadsPerWarp = 32;

  namespace syclexp = sycl::ext::oneapi::experimental;
  auto [cluster_size_m, cluster_size_n, cluster_size_k] = ClusterShape_MNK{};
  sycl::range<3> clusterSize(cluster_size_k, cluster_size_m, cluster_size_n);
  syclexp::properties Props{syclexp::work_groups_per_cluster<3>(clusterSize)};

  auto num_groups = ceil_div(prob_shape, TileShape_MNK{});
  sycl::range<3> local_range(1, NumControlWarps, NumThreadsPerWarp);
  // group_range: (Z=1, Y=M_tiles, X=N_tiles) matching cluster dim mapping
  sycl::range<3> group_range(1, get<0>(num_groups), get<1>(num_groups));
  sycl::nd_range<3> Range(group_range * local_range, local_range);

  // ---- Launch kernel ----
  auto launch_cfg = syclexp::launch_config(Range, Props);
  syclexp::submit_with_event(queue, [&](sycl::handler& handler) {
    syclexp::nd_launch(handler, launch_cfg, [=](sycl::nd_item<3> item) ALWAYS_INLINE {
      gemm_device_blockscaled_cluster<Config, decltype(ClusterShape_MNK{}),
                       decltype(prob_shape), decltype(cta_tiler), decltype(TileShape_MNK{}),
                       decltype(sA), decltype(sC), decltype(adma_load_a),
                       decltype(sB), decltype(adma_load_b), decltype(adma_store_c),
                       decltype(dC),
                       decltype(sSFA), decltype(sSFB),
                       decltype(adma_load_sfa), decltype(adma_load_sfb),
                       TiledMma,
                       Alpha, Beta>(
                       prob_shape, cta_tiler, TileShape_MNK{}, ClusterShape_MNK{},
                       A_ptr, sA, sC, adma_load_a,
                       B_ptr, sB, adma_load_b, adma_store_c,
                       C_ptr, dC,
                       SFA_ptr, sSFA, adma_load_sfa,
                       SFB_ptr, sSFB, adma_load_sfb,
                       alpha, beta,
                       item);
    });
  }).wait();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// GEMM dispatch
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Config,
          class TensorA, class TensorB, class TensorC,
          class TensorSFA, class TensorSFB,
          class Alpha, class Beta>
void
gemm_dispatch(char transA, char transB, int m, int n, int k,
              Alpha alpha,
              TensorA const& A, TensorB const& B,
              Beta beta,
              TensorC& C,
              TensorSFA const& SFA, TensorSFB const& SFB,
              sycl::queue& queue)
{
  if (transA == 'T' && transB == 'N') {
    return gemm_tn_blockscaled_cluster<Config>(m, n, k, alpha, A, B, beta, C, SFA, SFB, queue);
  }
  assert(false && "Not implemented");
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Entry Point: run_blockscaled_gemm_cluster<Config>(m, n, k, transA, transB)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Config>
int run_blockscaled_gemm_cluster(int m, int n, int k, char transA, char transB)
{
  using ElementA  = typename Config::ElementA;
  using ElementB  = typename Config::ElementB;
  using ElementC  = typename Config::ElementC;
  using ElementSF = typename Config::ElementSF;
  static constexpr int SFVecSize = Config::SFVecSize;
  using TileShape_MNK = typename Config::TileShape_MNK;
  using ClusterShape_MNK = typename Config::ClusterShape_MNK;

  sycl::queue queue{sycl::gpu_selector_v};

  std::cout << "Running on device: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  std::cout << "=== XE4 Block-Scaled GEMM with Cluster Multicast ===" << std::endl;
  std::cout << "  A type: " << type_str<ElementA>()
            << "  B type: " << type_str<ElementB>()
            << "  C type: " << type_str<ElementC>()
            << "  SF type: " << type_str<ElementSF>()
            << "  SFVecSize: " << SFVecSize << std::endl;

  auto [csm, csn, csk] = ClusterShape_MNK{};
  std::cout << "  ClusterShape: <" << csm << ", " << csn << ", " << csk << ">" << std::endl;
  auto [tm, tn, tk] = TileShape_MNK{};
  std::cout << "  TileShape:    <" << tm << ", " << tn << ", " << tk << ">" << std::endl;
  std::cout << "  Coop SF load: " << (Config::EnableCooperativeSF ? "enabled" : "disabled") << std::endl;
  std::cout << "  Problem: M=" << m << " N=" << n << " K=" << k
            << "  transA=" << transA << " transB=" << transB << std::endl;

  using TI = float;
  TI alpha = TI(1.0f);
  TI beta  = TI(0.0f);

  // ---- Allocate data tensors ----
  auto A = make_shared_usm_tensor<ElementA, 'R'>(queue, m, k);
  auto B = make_shared_usm_tensor<ElementB, 'R'>(queue, n, k);
  auto C = make_shared_usm_tensor<ElementC, 'R'>(queue, m, n);

  // ---- Allocate scale factor tensors ----
  if (k % SFVecSize != 0) {
    std::cerr << "Error: K dimension (" << k
              << ") must be a multiple of SFVecSize (" << SFVecSize
              << ") for block-scaled GEMM." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int sf_k = k / SFVecSize;
  auto SFA = make_shared_usm_tensor<ElementSF, 'C'>(queue, m, sf_k);
  auto SFB = make_shared_usm_tensor<ElementSF, 'C'>(queue, n, sf_k);

  // ---- Initialize tensors ----
  constexpr uint64_t seed_base = 42;
  random_fill_data(A,   seed_base + 2022);
  random_fill_data(B,   seed_base + 2021);
  random_fill_sf(SFA,   seed_base + 2024);
  random_fill_sf(SFB,   seed_base + 2025);
  zero_fill(C);

  // ---- Reference tensors for validation ----
  auto A_ref = make_shared_usm_tensor<ElementA, 'R'>(queue, m, k);
  auto B_ref = make_shared_usm_tensor<ElementB, 'R'>(queue, n, k);
  copy(A, A_ref);
  copy(B, B_ref);

  subbyte_pack(A);
  subbyte_pack(B);

  // ---- Run Block-Scaled GEMM with Cluster ----
  gemm_dispatch<Config>(transA, transB, m, n, k, alpha, A, B, beta, C, SFA, SFB, queue);
  queue.wait_and_throw();

  // ---- Validate ----
  mem_layout layout_a = mem_layout::row_major;
  mem_layout layout_b = mem_layout::col_major;

  ElementA*  A_ref_ptr = &*A_ref.data();
  ElementB*  B_ref_ptr = &*B_ref.data();
  ElementC*  C_ptr     = &*C.data();
  ElementSF* SFA_ptr   = &*SFA.data();
  ElementSF* SFB_ptr   = &*SFB.data();

  int err_cnt = validate_mxfp_gemm_result<ElementA, ElementB, ElementC, ElementSF, float>(
        A_ref_ptr, B_ref_ptr, C_ptr,
        m, n, k,
        true, true,
        SFA_ptr, SFB_ptr,
        layout_a, layout_b,
        false,
        tolerance<ElementC>{},
        SFVecSize);

  printf("Verification: %s\n", (err_cnt == 0) ? "PASSED" : "FAILED");

  if (err_cnt != 0) {
    throw std::runtime_error("Block-Scaled GEMM with Cluster verification failed!");
  }

  return 0;
}

/// Convenience wrapper using compile-time defaults from Config.
template <class Config>
int run_blockscaled_gemm_cluster_defaults()
{
  return run_blockscaled_gemm_cluster<Config>(
    Config::DefaultM, Config::DefaultN, Config::DefaultK,
    Config::DefaultTransA, Config::DefaultTransB);
}

} // namespace xe4_blockscaled_gemm_cluster