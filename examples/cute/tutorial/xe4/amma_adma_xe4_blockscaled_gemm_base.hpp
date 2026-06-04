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
// XE4 Block-Scaled GEMM Base — Parameterized Test Infrastructure
//
// This header provides a reusable, config-driven test harness for block-scaled
// GEMM on XE4 using AMMA + ADMA pipelining.
//
// Driver files supply a Config struct specifying:
//   - Element types (A, B, C, D, SF)
//   - Scale factor block size (SFVecSize)
//   - Tile shape (TileShape_MNK)
//   - Pipeline stages
//   - MMA atom (TiledMma)
//   - MMAControl BlockScaleType encoding
//   - Default problem shape and transpositions
//
// Entry point: run_blockscaled_gemm<Config>(m, n, k, transA, transB, decoupleSFLoad)
// Convenience:  run_blockscaled_gemm_defaults<Config>()  — uses Config defaults
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
#include "cutlass/arch/barrier.h"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/pipeline/pipeline.hpp"

#include <cute/layout.hpp>
#include <cute/atom/mma_atom.hpp>

#include <cutlass/float_subbyte.h>

#include "../../../common/sycl_cute_common.hpp"
#include "../../../xe4/utils/validation.hpp"
#include "cutlass/detail/xe4_blockscaled_layout.hpp"

namespace xe4_blockscaled_gemm {

using namespace cute;
using sycl::ext::oneapi::this_work_item::get_nd_item;
using cutlass::detail::Xe4BlockScaledConfig;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Constants
//
///////////////////////////////////////////////////////////////////////////////////////////////////

constexpr static size_t SmemAlignment = 512;  // Minimum 512-byte alignment recommended by HW spec

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
    // Cast from uint8_t* to T* so the SMEM pointer carries the correct element
    // type.  Without this, ADMA copy deduces conflicting DataType between the
    // typed GMEM pointer (e.g. float_e4m3_t*) and the raw uint8_t* SLM pointer.
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

// Specialization when ElementC != ElementAcc (e.g. FP16/BF16 output, FP32 accumulator):
// Separate smem_D buffer required because the last MMA iteration performs a dtype change,
// writing FP16/BF16 to smem_D while simultaneously reading FP32 from smem_C.
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

  // Accumulator buffer (FP32) — used by AMMA mainloop for D = A*B + C accumulation
  cute::array_aligned<ElementAcc_,  cute::cosize_v<SmemLayoutC>,   SmemAlignment> smem_C;

  // D output buffer (FP16/BF16) — separate from accumulator.
  // On the last MMA iteration, the dtype change writes FP16/BF16 result here,
  // while C (FP32 accumulator) is read from smem_C. Avoids C/D overlap error.
  cute::array_aligned<ElementC_,  cute::cosize_v<SmemLayoutC>,   SmemAlignment> smem_D;

  // Scale factor buffers
  cute::array_aligned<ElementSF_, padded_sf_smem_v<SmemLayoutSFA>, SmemAlignment> smem_SFA;
  cute::array_aligned<ElementSF_, padded_sf_smem_v<SmemLayoutSFB>, SmemAlignment> smem_SFB;
};

// Specialization when ElementC == ElementAcc (e.g. both FP32):
// No dtype change needed on the last MMA iteration — D is written in-place to smem_C.
// Saves one full C-sized SLM buffer and avoids the extra dtype-change write.
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

  // Accumulator buffer (FP32) — also serves as D output buffer since types match.
  cute::array_aligned<ElementAcc_,  cute::cosize_v<SmemLayoutC>,   SmemAlignment> smem_C;

  // Scale factor buffers
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
// Device Kernel: Block-Scaled GEMM with AMMA + ADMA Pipelining
//
// Barrier Allocation (6 abar slots, 0-5):
//   abar 0 : load_a_abar   — Tracks A data load (K_PIPE_MAX slots)
//   abar 1 : load_sfa_abar — Tracks SFA load    (1 slot — wide atom fills all stages)
//   abar 2 : load_b_abar   — Tracks B data load (K_PIPE_MAX slots)
//   abar 3 : load_sfb_abar — Tracks SFB load    (1 slot — wide atom fills all stages)
//   abar 4 : mma_abar      — Tracks MMA completion (consumer → producer signal)
//   abar 5 : store_c_abar  — Tracks C store completion
//
// Tensor Descriptor Allocation (8 tdesc slots: 0-7):
//   tdesc 0 : A data       tdesc 3 : SFA scale factors
//   tdesc 1 : B data       tdesc 4 : SFB scale factors
//   tdesc 2 : C output     tdesc 5-7 : reserved
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Config,
          class ProblemShape, class CtaTilerAB, class CtaTilerSF, class TileShape,
          class SmemLayoutA, class SmemLayoutC, class ADMA_A,
          class SmemLayoutB, class ADMA_B, class ADMA_C,
          class CStride,
          class SmemLayoutSFA, class SmemLayoutSFB,
          class ADMA_SFA, class ADMA_SFB,
          class TiledMma,
          class Alpha, class Beta,
          bool LoadSfAllAtOnce>
void
gemm_device_blockscaled(ProblemShape shape_MNK, CtaTilerAB cta_tiler_AB, CtaTilerSF cta_tiler_SF, TileShape tile_shape,
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
  constexpr auto bP = Int<Config::PipelineStages>{};
  static constexpr int SFVecSize = Config::SFVecSize;

  // Preconditions
  static_assert(rank(ProblemShape{}) == 3);
  static_assert(rank(CtaTilerAB{}) == 3);
  static_assert(rank(CtaTilerSF{}) == 3);

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  // Flat 3-mode layout: (BLK_MN, BLK_K, PIPE)
  static_assert(size<0>(SmemLayoutA{}) == size<0>(CtaTilerAB{}));  // BLK_M
  static_assert(size<0>(SmemLayoutB{}) == size<1>(CtaTilerAB{}));  // BLK_N
  static_assert(size<1>(SmemLayoutA{}) == size<2>(CtaTilerAB{}));  // BLK_K
  static_assert(size<1>(SmemLayoutB{}) == size<2>(CtaTilerAB{}));  // BLK_K

  // ---- Allocate tensor descriptors for ADMA Copy ----
  auto tdesc_a   = allocate_tdesc<0>();    // A data
  auto tdesc_b   = allocate_tdesc<1>();    // B data
  auto tdesc_c   = allocate_tdesc<2>();    // C output
  auto tdesc_sfa = allocate_tdesc<3>();    // SFA scale factors
  auto tdesc_sfb = allocate_tdesc<4>();    // SFB scale factors

  // Set tensor descriptors for ADMA copy atoms
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

  // SMEM tensors for data operands — flat 3-mode layout (BLK_MN, BLK_K, PIPE)
  Tensor sA = make_data_smem_tensor<ElementA>(smem.smem_A, SmemLayoutA{});     // (BLK_M, BLK_K, PIPE)
  Tensor sB = make_data_smem_tensor<ElementB>(smem.smem_B, SmemLayoutB{});     // (BLK_N, BLK_K, PIPE)
  Tensor sC = make_tensor(make_smem_ptr(smem.smem_C.begin()), SmemLayoutC{});  // (BLK_M, BLK_N) — FP32 accumulator

  // When ElementC == ElementAcc (both FP32), D aliases C — no separate buffer needed.
  // When ElementC != ElementAcc (FP16/BF16 output), D uses a separate smem_D buffer
  // because the last MMA iteration performs a dtype change, writing to D while reading C.
  [[maybe_unused]] auto sD_tensor = [&]() {
    if constexpr (std::is_same_v<ElementC, ElementAcc>) {
      return make_tensor(make_smem_ptr(reinterpret_cast<ElementC*>(smem.smem_C.begin())), SmemLayoutC{});
    } else {
      return make_tensor(make_smem_ptr(smem.smem_D.begin()), SmemLayoutC{});
    }
  }();
  auto sD = sD_tensor;  // (BLK_M, BLK_N)

  // SMEM tensors for scale factors
  Tensor sSFA = make_tensor(make_smem_ptr(smem.smem_SFA.begin()), SmemLayoutSFA{}); // (sf_block, _, blk, PIPE)
  Tensor sSFB = make_tensor(make_smem_ptr(smem.smem_SFB.begin()), SmemLayoutSFB{}); // (sf_block, _, blk, PIPE)

  // ---- Create global memory tensor views ----
  auto [M, N, K] = shape_MNK;
  auto mA   = adma_load_a.get_tma_tensor(make_shape(M, K));
  auto mB   = adma_load_b.get_tma_tensor(make_shape(N, K));
  auto mC   = adma_store_c.get_tma_tensor(make_shape(M, N));

  // SF GMEM tensors use a hierarchical shape matching tile_atom_to_shape.
  // The ADMA atom receives the hierarchical SF GMEM tensor, producing a hierarchical
  // g_stride_. get_tma_tensor must use a congruent shape: ((1, MN), (SFVecSize, K/SFVecSize)).
  constexpr int SFVecSizeK = SFVecSize;
  auto mSFA = adma_load_sfa.get_tma_tensor(
      make_shape(make_shape(Int<1>{}, M), make_shape(Int<SFVecSizeK>{}, K / SFVecSizeK)));
  auto mSFB = adma_load_sfb.get_tma_tensor(
      make_shape(make_shape(Int<1>{}, N), make_shape(Int<SFVecSizeK>{}, K / SFVecSizeK)));

  // ---- Tile global tensors for this CTA ----
  auto cta_coord = make_coord(BlockIdxX(), BlockIdxY(), _);

  Tensor gA = local_tile(mA, cta_tiler_AB, cta_coord, Step<_1,  X, _1>{});          // (BLK_M, BLK_K, k_tiles)
  Tensor gB = local_tile(mB, cta_tiler_AB, cta_coord, Step< X, _1, _1>{});          // (BLK_N, BLK_K, k_tiles)
  Tensor gC = local_tile(mC, cta_tiler_AB, cta_coord, Step<_1, _1,  X>{});          // (BLK_M, BLK_N)

  Tensor gSFA = local_tile(mSFA, cta_tiler_SF, cta_coord, Step<_1,  X, _1>{});   // (bM_hier, bK_hier, k_tiles)
  Tensor gSFB = local_tile(mSFB, cta_tiler_SF, cta_coord, Step< X, _1, _1>{});   // (bN_hier, bK_hier, k_tiles)

  // Partition for ADMA copy
  auto [tAgA, tAsA] = tma_partition(adma_load_a,
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto [tBgB, tBsB] = tma_partition(adma_load_b,
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));

  // SF ADMA uses hierarchical rank-4 SMEM layouts with stride-0 broadcast modes.
  // group_modes<0,3> groups (sf_block, _1, blk) into DATA, keeping PIPE as the iteration mode --> used in stage-by-stage SF loading
  // group_modes<0,4> groups (sf_block, _1, blk) into DATA, loading all stages at once.
  constexpr int number_of_modes_to_group = LoadSfAllAtOnce ? 4 : 3;
  auto [tSFAgSFA, tSFAsSFA] = tma_partition(adma_load_sfa,
                                            group_modes<0,number_of_modes_to_group>(sSFA), group_modes<0,2>(gSFA));
  auto [tSFBgSFB, tSFBsSFB] = tma_partition(adma_load_sfb,
                                            group_modes<0,number_of_modes_to_group>(sSFB), group_modes<0,2>(gSFB));

  // ---- Pipeline configuration ----
  int  K_TILE_MAX = size<1>(tAgA);        // Total number of K tiles
  int  K_GROUP_MAX = size<1>(tSFAgSFA);   // SF groups = K_TILE_MAX / bP
  int  k_tile = 0;
  int  k_group = 0;                       // SF k-group cursor

  // ADMA transaction bytes per pipeline stage. SF barrier bytes use the actual ADMA
  // transfer size (bMN * bK/SFVecSize), not the padded SmemLayout cosize. When
  // sf_bK_actual < 8, the padded layout has gaps between pipeline stages that are
  // not part of the DMA transfer.
  constexpr int dma_bytes_A        = (cosize(SmemLayoutA{}) * sizeof_bits_v<ElementA> / 8) / decltype(bP)::value;
  constexpr int dma_bytes_B        = (cosize(SmemLayoutB{}) * sizeof_bits_v<ElementB> / 8) / decltype(bP)::value;
  constexpr int dma_bytes_C        = cosize(SmemLayoutC{}) * sizeof(ElementC);
  constexpr int dma_sf_bK_actual   = size<2>(CtaTilerAB{}) / SFVecSize;
  // SF bytes per pipeline stage — used with stage-by-stage copy atom.
  constexpr int dma_bytes_SFA      = size<0>(CtaTilerAB{}) * dma_sf_bK_actual * static_cast<int>(sizeof(ElementSF));
  constexpr int dma_bytes_SFB      = size<1>(CtaTilerAB{}) * dma_sf_bK_actual * static_cast<int>(sizeof(ElementSF));
  // Total SF bytes for all bP pipeline stages — used with all-stages-at-once copy atom.
  constexpr int dma_bytes_SFA_wide = dma_bytes_SFA * decltype(bP)::value;
  constexpr int dma_bytes_SFB_wide = dma_bytes_SFB * decltype(bP)::value;
  constexpr int SF_barriers_needed = LoadSfAllAtOnce ? 1 : bP;

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t warp_idx      = get_sg_id();

  // ---- Allocate asynchronous barriers ----
  auto load_a_abar  = allocate_abar<0, bP>();   // A data
  auto load_sfa_abar = allocate_abar<1, SF_barriers_needed>();  // SFA scale factors
  auto load_b_abar  = allocate_abar<2, bP>();   // B data
  auto load_sfb_abar = allocate_abar<3, SF_barriers_needed>();  // SFB scale factors
  auto mma_abar     = allocate_abar<4, bP>();   // MMA completion
  auto store_c_abar = allocate_abar<5>();               // C store completion

  // Initialize barriers
  if (elect_one_thr && warp_idx == 0) {
    for (int i = 0; i < bP; ++i) {
      xe4_initialize_barrier(load_a_abar[i], 1);
      xe4_initialize_barrier(load_b_abar[i], 1);
    }

    if constexpr (LoadSfAllAtOnce) {
      xe4_initialize_barrier(load_sfa_abar[0], 1);
      xe4_initialize_barrier(load_sfb_abar[0], 1);
    } else {
      for (int i = 0; i < bP; ++i) {
        xe4_initialize_barrier(load_sfa_abar[i], 1);
        xe4_initialize_barrier(load_sfb_abar[i], 1);
      }
    }
  } else if (elect_one_thr && warp_idx == 1) {
    for (int i = 0; i < bP; ++i) {
      xe4_initialize_barrier(mma_abar[i], 1);
    }
    xe4_initialize_barrier(store_c_abar[0], 1);
  }
  xe4_syncthreads();

  int store_c_phase_bit = 0;

  // ---- Partition for MMA ----
  TiledMma mma{};
  ThrMMA thr_mma = mma.get_thread_slice(ThreadIdxX());
  Tensor tCsA = thr_mma.partition_A(sA);                                          // (MMA, MMA_M, MMA_K, PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);                                          // (MMA, MMA_N, MMA_K, PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCsC = thr_mma.partition_C(sC);                                          // (MMA, MMA_M, MMA_N)

  // Allocate MMA fragments (SMEM descriptors)
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCsC);

  // D output partition — when ElementC == ElementAcc, D aliases C (same buffer).
  // When ElementC != ElementAcc, D uses a separate SLM buffer for dtype-changed output.
  Tensor tCsD = thr_mma.partition_C(sD);                                          // (MMA, MMA_M, MMA_N)
  Tensor tCrD = thr_mma.make_fragment_C(tCsD);

  // Create SF descriptor tensors via MakeTensor<AMMA::smem_sf_desc>.
  // This builds a DescriptorIterator-based tensor from the rank-4 SF SMEM layout.
  // Indexing by pipeline stage yields a MatrixDescriptor (implicitly convertible to
  // uint32_t) with the correct Type-3 StartAddress and Pitch for that stage.
  Tensor tCrSFA = make_tensor<AMMA::smem_sf_desc>(sSFA);   // (_1, _1, (blk_MN, blk_K), PIPE)
  Tensor tCrSFB = make_tensor<AMMA::smem_sf_desc>(sSFB);   // (_1, _1, (blk_MN, blk_K), PIPE)

  // Clear accumulators in SLM
  clear(tCsC);
  xe4_syncthreads();

  // ==================================================================
  // Warp-Specialized Pipeline: ADMA Producer + AMMA Consumer
  //
  // Warp 0 (ADMA): Loads A, B, SFA, SFB into SLM; stores C to GMEM
  // Warp 1 (AMMA): Executes block-scaled MMA with SF descriptors
  //
  // Pipeline: bP stages, PipelineState for read/write tracking
  //   Prologue → Mainloop → Drain (producer)
  //   Mainloop → Last iteration (consumer)
  //   Epilogue: C store (producer)
  // ==================================================================

  // Pipeline state
  auto write_state = cutlass::PipelineState<bP>();
  auto read_state  = cutlass::PipelineState<bP>();

  // MMA control register:
  //   NullC=1 for the first iteration: the AMMA engine CANNOT read EU-written SLM
  //   for the C operand (causes systolic deadlock). NullC=1 bypasses the C read,
  //   computing D = A*B. After the first AMMA writes D to SLM, subsequent iterations
  //   use NullC=0 to read AMMA-written SLM and accumulate: D = A*B + C.
  //
  //   BlockScaleType from Config.
  MMAControl mma_ctrl{};
  mma_ctrl.NullC = 1;               // First iteration: bypass C read (D = A*B)
  mma_ctrl.A_BlockScaleType = Config::BlockScaleType;
  mma_ctrl.B_BlockScaleType = Config::BlockScaleType;

  // ==================================================================
  // Warp 0: ADMA Producer — Load A, B, SFA, SFB into SLM
  // ==================================================================
  if (warp_idx == 0)
  {
    if (elect_one_thr)
    {
      // ---- Prologue: Fill pipeline stages with ADMA loads ----
      CUTLASS_PRAGMA_UNROLL
      for (int pipe = 0; pipe < bP && k_tile < K_TILE_MAX; ++k_tile, ++pipe)
      {
        xe4_set_barrier_transaction_bytes(load_a_abar[pipe], dma_bytes_A);
        xe4_set_barrier_transaction_bytes(load_b_abar[pipe], dma_bytes_B);
        // Load A data
        copy(adma_load_a.with(&load_a_abar[pipe]), tAgA(_, k_tile), tAsA(_, pipe));
        // Load B data
        copy(adma_load_b.with(&load_b_abar[pipe]), tBgB(_, k_tile), tBsB(_, pipe));
	
        if constexpr (LoadSfAllAtOnce) {
          if (pipe == 0 && k_tile == 0) {
            xe4_set_barrier_transaction_bytes(load_sfa_abar[0], dma_bytes_SFA_wide);
            xe4_set_barrier_transaction_bytes(load_sfb_abar[0], dma_bytes_SFB_wide);
            // Load A's SFA scale factors
            copy(adma_load_sfa.with(&load_sfa_abar[0]), tSFAgSFA(_, k_group), tSFAsSFA(_));
            // Load B's SFB scale factors
            copy(adma_load_sfb.with(&load_sfb_abar[0]), tSFBgSFB(_, k_group), tSFBsSFB(_));
            ++k_group;
          }
        } else {
          xe4_set_barrier_transaction_bytes(load_sfa_abar[pipe], dma_bytes_SFA);
          xe4_set_barrier_transaction_bytes(load_sfb_abar[pipe], dma_bytes_SFB);
          // Load A's SFA scale factors
          copy(adma_load_sfa.with(&load_sfa_abar[pipe]),   tSFAgSFA(_, k_tile), tSFAsSFA(_, pipe));
          // Load B's SFB scale factors
          copy(adma_load_sfb.with(&load_sfb_abar[pipe]),   tSFBgSFB(_, k_tile), tSFBsSFB(_, pipe));
        }
      }

      int used_ab_count = 0;
      // ---- Mainloop: Wait for MMA to consume, then refill freed stage ----
      for (; k_tile < K_TILE_MAX; ++k_tile)
      {
        int pipe = write_state.index();

        // Wait for MMA consumer to signal completion of this stage
        xe4_wait_barrier(mma_abar[pipe], write_state.phase());

        // Re-issue new loads into freed stage
        xe4_set_barrier_transaction_bytes(load_a_abar[pipe], dma_bytes_A);
        xe4_set_barrier_transaction_bytes(load_b_abar[pipe], dma_bytes_B);

        copy(adma_load_a.with(&load_a_abar[pipe]),       tAgA(_, k_tile),     tAsA(_, pipe));
        copy(adma_load_b.with(&load_b_abar[pipe]),       tBgB(_, k_tile),     tBsB(_, pipe));

        if constexpr (LoadSfAllAtOnce) {
          ++used_ab_count;
          if (used_ab_count == int(bP) && k_group < K_GROUP_MAX) {
            xe4_set_barrier_transaction_bytes(load_sfa_abar[0], dma_bytes_SFA_wide);
            xe4_set_barrier_transaction_bytes(load_sfb_abar[0], dma_bytes_SFB_wide);
            copy(adma_load_sfa.with(&load_sfa_abar[0]), tSFAgSFA(_, k_group), tSFAsSFA(_));
            copy(adma_load_sfb.with(&load_sfb_abar[0]), tSFBgSFB(_, k_group), tSFBsSFB(_));
            ++k_group;
            used_ab_count = 0;
          }
        } else {
          xe4_set_barrier_transaction_bytes(load_sfa_abar[pipe], dma_bytes_SFA);
          xe4_set_barrier_transaction_bytes(load_sfb_abar[pipe], dma_bytes_SFB);
          copy(adma_load_sfa.with(&load_sfa_abar[pipe]), tSFAgSFA(_, k_tile), tSFAsSFA(_, pipe));
          copy(adma_load_sfb.with(&load_sfb_abar[pipe]), tSFBgSFB(_, k_tile), tSFBsSFB(_, pipe));
        }

        ++write_state;
      }
    }
    sycl::group_barrier(item.get_sub_group());
  }
  // ==================================================================
  // Warp 1: AMMA Consumer — Execute block-scaled MMA
  // ==================================================================
  else if (warp_idx == 1)
  {
    if (elect_one_thr)
    {
      int read_state_phase_sf = 0;

      // Mainloop: Wait for loads, execute MMA, signal producer
      // Unroll the K mode manually so we can set mma_ctrl and barrier tracking
      for (int k_tile_next = 0; k_tile_next < K_TILE_MAX; ++k_tile_next)
      {
        int read_pipe = read_state.index();

        // Wait for A and B data loads for this pipeline stage to complete
        xe4_wait_barrier(load_a_abar[read_pipe], read_state.phase());
        xe4_wait_barrier(load_b_abar[read_pipe], read_state.phase());
	
	      if constexpr (LoadSfAllAtOnce) {
	        // SF barrier: wait once at the start of each K_PIPE_MAX group.
          // The wide atom fills all pipeline stages in one DMA, so a single
          // barrier covers the entire group.
          if (k_tile_next % int(bP) == 0) {
            xe4_wait_barrier(load_sfa_abar[0], read_state_phase_sf);
            xe4_wait_barrier(load_sfb_abar[0], read_state_phase_sf);
            read_state_phase_sf ^= 1;
          }
	      } else {
	        // Wait for SFA and SFB data loads for this pipeline stage to complete
          xe4_wait_barrier(load_sfa_abar[read_pipe], read_state.phase());
          xe4_wait_barrier(load_sfb_abar[read_pipe], read_state.phase());
	      }

        // Set barrier expected bytes ONCE for all k_blocks in this pipeline stage.
        // Non-last tiles: all k_blocks track AB (2 bytes each).
        // Last tile: (N-1) k_blocks track AB + last k_block tracks D (1 byte).
        int num_k = int(size<2>(tCrA));
        bool is_last_tile = (k_tile_next == K_TILE_MAX - 1);
        xe4_set_barrier_transaction_bytes(mma_abar[read_pipe],
            is_last_tile ? 2 * (num_k - 1) + 1 : 2 * num_k);

        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < num_k; ++k_block) {
          bool is_last_iter = is_last_tile && (k_block == num_k - 1);

          if (is_last_iter) {
            // Last k_block of last tile: Track D completion for the store barrier
            if constexpr (std::is_same_v<ElementC, ElementAcc>) {
              // ElementC == ElementAcc (both FP32): no dtype change needed.
              // Use 3-operand gemm with D-tracking — result stays in smem_C (which D aliases).
              auto new_mma = mma.with(
                ElementAcc{},
                AMMA::TrackMethod<AMMA::Tracking::D>{},
                mma_ctrl,
                tCrSFA(0, 0, k_block, read_pipe), tCrSFB(0, 0, k_block, read_pipe),
                &mma_abar[read_pipe]);
              cute::gemm(new_mma, tCrA(_,_,k_block,read_pipe), tCrB(_,_,k_block,read_pipe), tCrC);
            } else {
              // ElementC != ElementAcc (e.g. FP16/BF16 output, FP32 accumulator):
              // 4-operand gemm with dtype change — D(FP16/BF16) written to separate smem_D buffer
              // while reading C(FP32) from smem_C.
              auto new_mma = mma.with(
                ElementC{},
                AMMA::TrackMethod<AMMA::Tracking::D>{},
                mma_ctrl,
                tCrSFA(0, 0, k_block, read_pipe), tCrSFB(0, 0, k_block, read_pipe),
                &mma_abar[read_pipe]);
              cute::gemm(new_mma, tCrD, tCrA(_,_,k_block,read_pipe), tCrB(_,_,k_block,read_pipe), tCrC);
            }
          } else {
            // Non-last k_blocks: Track AB consumption for pipeline feedback
            auto new_mma = mma.with(
              ElementAcc{},
              AMMA::TrackMethod<AMMA::Tracking::AB>{},
              mma_ctrl,
              tCrSFA(0, 0, k_block, read_pipe), tCrSFB(0, 0, k_block, read_pipe),
              &mma_abar[read_pipe], &mma_abar[read_pipe]);
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
      // Wait for MMA to finish writing the final result.
      // read_state is per-work-item and was never modified in warp 0 (only warp 1
      // increments it). Compute the correct pipe index and phase that the consumer's
      // last D-tracking AMMA signals: after K_TILE_MAX-1 mainloop increments.
      int last_k_tile  = K_TILE_MAX - 1;
      int last_pipe    = last_k_tile % int(bP);
      int last_phase   = (last_k_tile / int(bP)) % 2;
      xe4_wait_barrier(mma_abar[last_pipe], last_phase);
      xe4_set_barrier_transaction_bytes(store_c_abar[0], dma_bytes_C);
      // When ElementC == ElementAcc, D aliases C — store from smem_C directly.
      // When types differ, store from the separate smem_D buffer.
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
// Host Setup: TN GEMM with Block Scaling (parameterized on Config)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Config,
          class TensorA, class TensorB, class TensorC,
          class TensorSFA, class TensorSFB,
          class Alpha, class Beta>
void
gemm_tn_blockscaled(int m, int n, int k,
                    Alpha alpha,
                    TensorA const& A,
                    TensorB const& B,
                    Beta beta,
                    TensorC& C,
                    TensorSFA const& SFA,
                    TensorSFB const& SFB,
                    bool decoupleSFLoad,
                    sycl::queue& queue)
{
  using ElementA  = typename Config::ElementA;
  using ElementB  = typename Config::ElementB;
  using ElementC  = typename Config::ElementC;
  using ElementSF = typename Config::ElementSF;
  using TiledMma  = typename Config::TiledMma;
  using TileShape_MNK = typename Config::TileShape_MNK;
  static constexpr int SFVecSize       = Config::SFVecSize;
  static constexpr int PipelineStages  = Config::PipelineStages;
  static constexpr auto majorA = Config::MajorA;
  static constexpr auto majorB = Config::MajorB;

  constexpr auto bM = get<0>(TileShape_MNK{});
  constexpr auto bN = get<1>(TileShape_MNK{});
  constexpr auto bK = get<2>(TileShape_MNK{});
  constexpr auto bP = Int<PipelineStages>{};

  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  ElementA  const* A_ptr   = &*A.data();
  ElementB  const* B_ptr   = &*B.data();
  auto C_ptr   = &*C.data();

  auto dA = A.stride();
  auto dB = B.stride();
  auto dC = C.stride();

  Tensor mA = make_tensor(make_gmem_ptr(A_ptr), make_layout(make_shape(M, K), dA));
  Tensor mB = make_tensor(make_gmem_ptr(B_ptr), make_layout(make_shape(N, K), dB));
  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), make_layout(make_shape(M, N), dC));

  // ---- Tile and cluster shapes ----
  using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;

  // ---- SMEM layouts derived from tile shape and majors ----
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

  SmemLayoutA   sA{};
  SmemLayoutB   sB{};
  SmemLayoutC   sC{};

  // ---- ADMA copy atoms for data ----
  using GmemTiledCopyA =
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::XE4_ADMA_LOAD, cute::XE4_ADMA_LOAD_MULTICAST>;
  using GmemTiledCopyB =
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::XE4_ADMA_LOAD, cute::XE4_ADMA_LOAD_MULTICAST>;
  using GmemTiledCopyC = cute::XE4_ADMA_STORE;

  TiledMma tiled_mma{};
  auto cluster_layout_vmnk = tiled_divide(
    make_layout(ClusterShape_MNK{}),
    make_tile(typename TiledMma::AtomThrID{}));

  // ADMA load atoms for A and B data
  auto adma_load_a = make_adma_atom_A_xe4(
    GmemTiledCopyA{}, mA,
    SmemLayoutA{}(_, _, cute::Int<0>{}),
    TileShape_MNK{}, TiledMma{}, cluster_layout_vmnk);

  auto adma_load_b = make_adma_atom_B_xe4(
    GmemTiledCopyB{}, mB,
    SmemLayoutB{}(_, _, cute::Int<0>{}),
    TileShape_MNK{}, TiledMma{}, cluster_layout_vmnk);

  // ADMA store atom for C
  auto cta_tiler_mn = make_shape(bM, bN);
  auto adma_store_c = make_adma_copy<ElementC>(GmemTiledCopyC{}, mC, SmemLayoutC{}, cta_tiler_mn, Int<1>{});

  using BlkScaledConfig = Xe4BlockScaledConfig<SFVecSize>;
  auto layout_SFA = BlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K));
  auto layout_SFB = BlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K));

  auto SFA_ptr = &*SFA.data();
  auto SFB_ptr = &*SFB.data();
  // Hierarchical GMEM layout from tile_atom_to_shape.
  // The ADMA atom creator internally flattens this via filter_zeros+coalesce
  // for TMA descriptor construction, producing a flat g_stride_.
  Tensor mSFA = make_tensor(make_gmem_ptr(SFA_ptr), layout_SFA);
  Tensor mSFB = make_tensor(make_gmem_ptr(SFB_ptr), layout_SFB);

  // ---- SMEM layouts for scale factors ----
  // Rank-4 layout ((mnBlock, kBlock), _1, (blk_MN, blk_K)) deduced from TiledMma
  // and TileShape_MNK to match the cm_8x32B Type3 matrix descriptor requirements.
  TiledMma tiled_mma_inst{};
  auto xe4_sfA_layout_atom = BlkScaledConfig::deduce_smem_layoutSFA(tiled_mma_inst, TileShape_MNK{});
  auto xe4_sfB_layout_atom = BlkScaledConfig::deduce_smem_layoutSFB(tiled_mma_inst, TileShape_MNK{});

  // SF padding for cm_8x32B core-matrix alignment.
  // The ADMA 2D-block-copy writes in cm_8x32B units (8 rows minimum).
  // When sf_bK_actual < 8, overflow rows land in the next pipeline stage's SF region.
  // Padding the pipe stride to sf_bK = max(sf_bK_actual, 8) isolates each stage.
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

  SmemLayoutSFA sSFA{};
  SmemLayoutSFB sSFB{};

  using GmemTiledCopySF =
    cute::conditional_t<
      size(ClusterShape_MNK{}) == 1,
      cute::XE4_ADMA_LOAD, cute::XE4_ADMA_LOAD_MULTICAST>;

  // K = bK --> atom footprint spans one stage
  auto cta_tiler_AB = make_shape(bM, bN, bK);

  // C++17 tag-dispatch: pass std::true_type{} (all-stages-at-once) or std::false_type{} (per-stage)

  // ---- CTA Tiler for scale factors ----
  auto make_cta_tiler_SF = [&](auto wide_tag) {
    constexpr bool Wide = decltype(wide_tag)::value;
    if constexpr (Wide) {
      return make_shape(bM, bN, bK * bP);  // K widened by bP — one copy() loads all stages
    } else {
      return make_shape(bM, bN, bK);       // atom footprint spans one stage
    }
  };

  // ---- ADMA copy atoms for scale factors ----
  auto make_adma_sf_A = [&](auto wide_tag) {
    constexpr bool Wide = decltype(wide_tag)::value;
    if constexpr (Wide) {
      return make_adma_atom_A_xe4(
        GmemTiledCopySF{}, mSFA,
        SmemLayoutSFA{},                   // full multi-stage layout (all bP pipeline stages)
        make_cta_tiler_SF(wide_tag),
        TiledMma{}, cluster_layout_vmnk);
    } else {
      return make_adma_atom_A_xe4(
        GmemTiledCopySF{}, mSFA,
        decltype(xe4_sfA_layout_atom){},
        TileShape_MNK{},
        TiledMma{}, cluster_layout_vmnk);
    }
  };

  auto make_adma_sf_B = [&](auto wide_tag) {
    constexpr bool Wide = decltype(wide_tag)::value;
    if constexpr (Wide) {
      return make_adma_atom_B_xe4(
        GmemTiledCopySF{}, mSFB,
        SmemLayoutSFB{},                   // full multi-stage layout (all bP pipeline stages)
        make_cta_tiler_SF(wide_tag),
        TiledMma{}, cluster_layout_vmnk);
    } else {
      return make_adma_atom_B_xe4(
        GmemTiledCopySF{}, mSFB,
        decltype(xe4_sfB_layout_atom){},
        TileShape_MNK{},
        TiledMma{}, cluster_layout_vmnk);
    }
  };

  // ---- Launch configuration ----
  constexpr int NumControlWarps = 2;   // Sub-group 0 = ADMA, Sub-group 1 = AMMA
  constexpr int NumThreadsPerWarp = 32;

  namespace syclexp = sycl::ext::oneapi::experimental;
  auto [cluster_size_y, cluster_size_x, cluster_size_z] = ClusterShape_MNK{};
  sycl::range<3> clusterSize(cluster_size_z, cluster_size_y, cluster_size_x);
  syclexp::properties Props{syclexp::work_groups_per_cluster<3>(clusterSize)};

  auto num_groups = ceil_div(prob_shape, TileShape_MNK{});
  sycl::range<3> local_range(1, NumControlWarps, NumThreadsPerWarp);
  sycl::range<3> group_range(1, get<1>(num_groups), get<0>(num_groups));
  sycl::nd_range<3> Range(group_range * local_range, local_range);

  if (decoupleSFLoad) {
    if (sf_bK_actual < 8) {
      std::cout << "[GEMM-MX] Warning: K length of SF block i.e. sf_bK_actual (" << sf_bK_actual
                << ") is less than 8, which is not supported in current all-stage SF loading.\n";
      std::cout << "[GEMM-MX] Falling back to per-stage SF loading with ADMA. \n";
      decoupleSFLoad = false;
    }
    if (K % (int(bK) * int(bP)) != 0) {
      std::cout << "[GEMM-MX] Warning: K (" << K << ") must be divisible by bK * PipelineStages ("
                << int(bK) << " * " << int(bP) << " = " << int(bK) * int(bP)
                << ") for single-DMA SF loading for all pipeline stages.\n";
      std::cout << "[GEMM-MX] Falling back to per-stage SF loading with ADMA. \n";
      decoupleSFLoad = false;
    }
  }

  // ---- Launch kernel ----
  auto launch_cfg = syclexp::launch_config(Range, Props);

  // Single compile-time-parameterized launch lambda — Wide=false: per-stage SF load,
  // Wide=true: all-stages-at-once SF load. Selected at runtime by decoupleSFLoad.
  auto do_launch = [&](auto wide_tag) {
    constexpr bool Wide = decltype(wide_tag)::value;
    auto adma_load_sfa = make_adma_sf_A(wide_tag);
    auto adma_load_sfb = make_adma_sf_B(wide_tag);
    auto cta_tiler_SF  = make_cta_tiler_SF(wide_tag);
    syclexp::submit_with_event(queue, [&](sycl::handler& handler) {
      syclexp::nd_launch(handler, launch_cfg, [=](sycl::nd_item<3> item) ALWAYS_INLINE {
        gemm_device_blockscaled<Config,
                         decltype(prob_shape),
                         decltype(cta_tiler_AB), decltype(cta_tiler_SF), decltype(TileShape_MNK{}),
                         decltype(sA), decltype(sC), decltype(adma_load_a),
                         decltype(sB), decltype(adma_load_b), decltype(adma_store_c),
                         decltype(dC),
                         decltype(sSFA), decltype(sSFB),
                         decltype(adma_load_sfa), decltype(adma_load_sfb),
                         TiledMma,
                         Alpha, Beta,
                         Wide>(
                         prob_shape, cta_tiler_AB, cta_tiler_SF, TileShape_MNK{},
                         A_ptr, sA, sC, adma_load_a,
                         B_ptr, sB, adma_load_b, adma_store_c,
                         C_ptr, dC,
                         SFA_ptr, sSFA, adma_load_sfa,
                         SFB_ptr, sSFB, adma_load_sfb,
                         alpha, beta,
                         item);
      });
    }).wait();
  };

  if (decoupleSFLoad) {
    std::cout << "[GEMM-MX] launch: decoupleSFLoad: true\n";
    do_launch(std::true_type{});
  } else {
    std::cout << "[GEMM-MX] launch: decoupleSFLoad: false\n";
    do_launch(std::false_type{});
  }
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
              bool decoupleSFLoad,
              sycl::queue& queue)
{
  if (transA == 'T' && transB == 'N') {
    return gemm_tn_blockscaled<Config>(m, n, k, alpha, A, B, beta, C, SFA, SFB, decoupleSFLoad, queue);
  }
  assert(false && "Not implemented");
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Entry Point: run_blockscaled_gemm<Config>(m, n, k, transA, transB, decoupleSFLoad)
//
// Allocates tensors, initializes data, runs the GEMM, and validates.
// Runtime problem shape and transpositions are passed as parameters.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Config>
int run_blockscaled_gemm(int m, int n, int k, char transA, char transB, bool decoupleSFLoad)
{
  using ElementA  = typename Config::ElementA;
  using ElementB  = typename Config::ElementB;
  using ElementC  = typename Config::ElementC;
  using ElementSF = typename Config::ElementSF;
  static constexpr int SFVecSize = Config::SFVecSize;
  using TileShape_MNK = typename Config::TileShape_MNK;

  sycl::queue queue{sycl::gpu_selector_v};

  std::cout << "Running on device: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  std::cout << "=== XE4 Block-Scaled GEMM with AMMA + ADMA Pipelining ===" << std::endl;
  std::cout << "  A type: " << type_str<ElementA>()
            << "  B type: " << type_str<ElementB>()
            << "  C type: " << type_str<ElementC>()
            << "  SF type: " << type_str<ElementSF>()
            << "  SFVecSize: " << SFVecSize << std::endl;
  auto [tm, tn, tk] = TileShape_MNK{};
  std::cout << "  TileShape:    <" << tm << ", " << tn << ", " << tk << ">" << std::endl;
  std::cout << "  Problem: M=" << m << " N=" << n << " K=" << k
            << "  transA=" << transA << " transB=" << transB << std::endl;

  using TI = float;
  TI alpha = TI(1.0f);
  TI beta  = TI(0.0f);

  // ---- Allocate data tensors ----
  auto A = make_shared_usm_tensor<ElementA, 'R'>(queue, m, k);
  auto B = make_shared_usm_tensor<ElementB, 'R'>(queue, n, k);
  auto C = make_shared_usm_tensor<ElementC, 'R'>(queue, m, n);

  // ---- Allocate scale factor tensors (column-major: M/N-contiguous) ----
   if (k % SFVecSize != 0) {
    std::cerr << "Error: K dimension (" << k
              << ") must be a multiple of SFVecSize (" << SFVecSize
              << ") for block-scaled GEMM. "
              << "Adjust K or SFVecSize so that K % SFVecSize == 0."
              << std::endl;
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

  // Pack sub-byte elements if needed (no-op for >= 8-bit types)
  subbyte_pack(A);
  subbyte_pack(B);

  // ---- Run Block-Scaled GEMM ----
  gemm_dispatch<Config>(transA, transB, m, n, k, alpha, A, B, beta, C, SFA, SFB, decoupleSFLoad, queue);
  queue.wait_and_throw();

 // ---- Validate against block-scaled reference ----
  // validate_blockscaled_gemm_result applies SF to A/B before computing the golden reference:
  //   golden = (A_ref * diag(SFA)) x (B_ref * diag(SFB))
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
    throw std::runtime_error("Block-Scaled GEMM verification failed!");
  }

  return 0;
}

/// Convenience wrapper: runs GEMM with compile-time defaults from Config.
/// Does NOT parse command-line arguments; use the (m,n,k,transA,transB) overload
/// for runtime parameters.
template <class Config>
int run_blockscaled_gemm_defaults()
{
  return run_blockscaled_gemm<Config>(
    Config::DefaultM, Config::DefaultN, Config::DefaultK,
    Config::DefaultTransA, Config::DefaultTransB, true /*decoupleSFLoad*/);
}

} // namespace xe4_blockscaled_gemm
