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

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// XE4 Block-Scaled GEMM with Cluster Multicast — Generic CLI Test Driver
//
// Supports multiple block-scaled data formats with cluster multicast:
//   NVFP4, NVFP4+, MXFP4, MXFP8
//
// CLI Arguments (all positional, all optional):
//   argv[1]  : M            (default: 512)
//   argv[2]  : N            (default: 1024)
//   argv[3]  : K            (default: 2048)
//   argv[4]  : transA       (default: 'T')
//   argv[5]  : transB       (default: 'N')
//   argv[6]  : input_type   (default: "NVFP4", valid: NVFP4/NVFP4+/MXFP4/MXFP8/ALL)
//   argv[7]  : coop_sf      (default: 0, valid: 0=disabled, 1=enabled)
//   argv[8]  : SFVecSize    (default: 16, valid: 16 or 32; ignored when input_type=ALL)
//   argv[9]  : output_type  (default: "FP32",  valid: FP32/FP16/BF16; ignored when input_type=ALL)
//   argv[10] : cluster_m    (default: 2; ignored when input_type=ALL)
//   argv[11] : cluster_n    (default: 2; ignored when input_type=ALL)
//
// Element type mapping:
//   NVFP4  : A/B = float_e2m1_t (4-bit), SF = float_ue4m3_t, BlockScaleType = 5
//   NVFP4+ : A/B = float_e2m1_t (4-bit), SF = float_ue5m3_t, BlockScaleType = 2 (SFVecSize 32) or 3 (SFVecSize 16)
//   MXFP4  : A/B = float_e2m1_t (4-bit), SF = float_ue8m0_t, BlockScaleType = 0 (SFVecSize 32) or 1 (SFVecSize 16)
//   MXFP8  : A/B = float_e4m3_t (8-bit), SF = float_ue8m0_t, BlockScaleType = 0 (SFVecSize 32)
//
// All GEMM infrastructure (device kernel, host setup, dispatch, validation)
// lives in amma_adma_xe4_blockscaled_gemm_cluster_base.hpp.
// Default (cooperative SF disabled):
//./blockscaled_gemm_cluster 512 512 512 T N NVFP4 0 16 FP32 2 2
//
// Cooperative SF enabled:
//./blockscaled_gemm_cluster 512 512 512 T N NVFP4 1 16 FP32 2 2
//
// ALL configs (non-cooperative):
//./blockscaled_gemm_cluster 512 512 512 T N ALL
//
// ALL configs (cooperative):
//./blockscaled_gemm_cluster 512 512 512 T N ALL 1
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "amma_adma_xe4_blockscaled_gemm_cluster_base.hpp"
#include <string>
#include <type_traits>
using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Generic Config struct — parameterized on element types, tile shape, cluster shape
//
// Provides all members required by the cluster base header's GEMM infrastructure.
// Extends GenericBlockScaledConfig with ClusterShape_MNK.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class ElementA_, class ElementB_, class ElementC_, class ElementSF_,
          int SFVecSize_, int BlockScaleType_,
          int TileM_, int TileN_, int TileK_,
          int ClusterM_, int ClusterN_,
          bool CoopSF_ = false>
struct GenericBlockScaledClusterConfig {

  // ---- Element types ----
  using ElementA  = ElementA_;
  using ElementB  = ElementB_;
  using ElementC  = ElementC_;
  using ElementSF = ElementSF_;
  using ElementAcc = float;

  // ---- Scale factor block size ----
  static constexpr int SFVecSize = SFVecSize_;

  // ---- Tile shape ----
  using TileShape_MNK = cute::Shape<cute::Int<TileM_>, cute::Int<TileN_>, cute::Int<TileK_>>;

  // ---- Pipeline stages ----
  static constexpr int PipelineStages = 4;

  // ---- Layout majors for AMMA ----
  static constexpr auto MajorA = cute::AMMA::Major::K;
  static constexpr auto MajorB = cute::AMMA::Major::K;

  // ---- Cluster shape ----
  using ClusterShape_MNK = cute::Shape<cute::Int<ClusterM_>, cute::Int<ClusterN_>, cute::_1>;

  // ---- Cooperative SF loading (default: disabled, uses trivial cluster layout for SF) ----
  // When true, SF tiles are loaded cooperatively across the cluster (same as SM100),
  // using the real cluster layout so each CTA loads 1/N-th of the SF tile.
  // When false (default), each CTA independently loads the full SF tile using
  // a trivial (1,1,1) cluster layout for the SF ADMA atoms.
  static constexpr bool EnableCooperativeSF = CoopSF_;

  // Cooperative SF loading requires that after ADMA box truncation by the cluster,
  // each CTA's SF K-dimension is a multiple of 8 (the cm_8x32B core-matrix row count).
  // The ADMA 2D-block-copy always writes in cm_8x32B core-matrix units (8 rows x 32 bytes
  // minimum). With cooperative loading, the SF tile is split across cluster CTAs — each
  // CTA's ADMA box covers only (TileK / SFVecSize / max(ClusterM, ClusterN)) rows.
  // If this is not a multiple of 8 (e.g. 12), the last 8-row core-matrix write overflows
  // into a peer CTA's SF region within the SAME pipeline stage, corrupting its scale
  // factor data. Unlike non-cooperative mode where padding spaces out pipeline stages to
  // absorb inter-stage overflow, intra-stage collision between CTAs cannot be fixed by
  // padding because both CTAs write to the same stage simultaneously.
  // Requirements:
  //   1. (TileK / SFVecSize) must be divisible by max(ClusterM, ClusterN) so that
  //      each CTA gets an equal share of SF rows (avoids silent truncation by integer division).
  //   2. The resulting per-CTA K_sf must be >= 8 and a multiple of 8.
  // Required minimum TileK: VS=16 → 256, VS=32 → 512 (for max cluster dim = 2).
  static constexpr int MaxClusterDim_ = (ClusterM_ > ClusterN_) ? ClusterM_ : ClusterN_;
  static constexpr int TileK_sf_ = TileK_ / SFVecSize_;
  static_assert(!EnableCooperativeSF ||
      (TileK_sf_ % MaxClusterDim_ == 0 &&
       (TileK_sf_ / MaxClusterDim_ >= 8) &&
       (TileK_sf_ / MaxClusterDim_) % 8 == 0),
      "Cooperative SF loading requires (TileK / SFVecSize) to be divisible by "
      "max(ClusterM, ClusterN), and the per-CTA K_sf to be a multiple of 8 "
      "(cm_8x32B core-matrix alignment after ADMA box truncation by cluster). "
      "Increase TileK: VS=16 needs >=256, VS=32 needs >=512.");

  // ---- MMA atom ----
  // Use bs_op_selector to automatically select the correct block-scaled MMA op.
  // With cluster > 1, this selects XE4_AMMA_AB_CLUSTER variant.
  // EnableCooperativeSF is passed so that for VS=32 cooperative, atom K=512
  // (Tile_K directly) instead of gcd(512, 768)=256 which is insufficient.
  using TiledMma = decltype(cute::make_tiled_mma(
    cute::AMMA::bs_op_selector<
      ElementAcc,                            // d_type
      ElementA_,                            // a_type
      ElementB_,                            // b_type
      ElementAcc,                            // c_type
      ElementSF_,                           // sf_a_type
      ElementSF_,                           // sf_b_type (same as A for symmetric configs)
      SFVecSize_,                           // VSA: scale factor vector size
      SFVecSize_,                           // VSB: scale factor vector size (same as A)
      TileShape_MNK,                       // Tile shape — selector computes MMA dims via gcd
      ClusterShape_MNK,                    // Cluster shape
      MajorA, MajorB,                      // A and B layout majors
      EnableCooperativeSF>()               // cooperative SF flag for atom K selection
  ));

  // ---- MMAControl BlockScaleType encoding ----
  static constexpr int BlockScaleType = BlockScaleType_;

  // ---- Default problem shape (used by backward-compat wrapper) ----
  static constexpr int  DefaultM      = 512;
  static constexpr int  DefaultN      = 1024;
  static constexpr int  DefaultK      = 2048;
  static constexpr char DefaultTransA = 'T';
  static constexpr char DefaultTransB = 'N';
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Compile-time BlockScaleType encoding map
//
// Maps {ElementSF, SFVecSize} -> MMAControl BlockScaleType value:
//   ue4m3 + 16 -> 5 (ue4m3k16)
//   ue5m3 + 16 -> 3 (ue5m3k16)
//   ue5m3 + 32 -> 2 (ue5m3k32)
//   ue8m0 + 16 -> 1 (ue8m0k16)
//   ue8m0 + 32 -> 0 (ue8m0k32)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class ElementSF, int SFVecSize>
struct BlockScaleTypeMap { static constexpr int value = 0; };

template <> struct BlockScaleTypeMap<cutlass::float_ue4m3_t, 16> { static constexpr int value = 5; };
template <> struct BlockScaleTypeMap<cutlass::float_ue5m3_t, 16> { static constexpr int value = 3; };
template <> struct BlockScaleTypeMap<cutlass::float_ue5m3_t, 32> { static constexpr int value = 2; };
template <> struct BlockScaleTypeMap<cutlass::float_ue8m0_t, 16> { static constexpr int value = 1; };
template <> struct BlockScaleTypeMap<cutlass::float_ue8m0_t, 32> { static constexpr int value = 0; };


// Helper template function: given compile-time element types, SFVecSize,
// BlockScaleType, tile shape, and cluster shape, resolve runtime output_type
// and run GEMM with cluster.
// Uses if-constexpr to skip invalid combos (e.g. FP8 + SFVecSize=16).
template <class ElementA, class ElementB, class ElementSF,
          int SVS, int BST, int TM, int TN, int TK, int CM, int CN, bool CoopSF>
int run_gemm_config(const std::string& output_type,
                    int m, int n, int k, char transA, char transB)
{
  // Guard: FP8 types (float_e4m3_t) only support SFVecSize=32.
  // Prevent instantiation of invalid ADMA copy atoms at compile-time.
  if constexpr (std::is_same_v<ElementA, cutlass::float_e4m3_t> && SVS != 32) {
    std::cerr << "Error: FP8 does not support SFVecSize=" << SVS << std::endl;
    return 1;
  } else {
    if (output_type == "FP32") {
      using Cfg = GenericBlockScaledClusterConfig<ElementA, ElementB, float, ElementSF, SVS, BST, TM, TN, TK, CM, CN, CoopSF>;
      return xe4_blockscaled_gemm_cluster::run_blockscaled_gemm_cluster<Cfg>(m, n, k, transA, transB);
    }
    if (output_type == "FP16") {
      using Cfg = GenericBlockScaledClusterConfig<ElementA, ElementB, sycl::half, ElementSF, SVS, BST, TM, TN, TK, CM, CN, CoopSF>;
      return xe4_blockscaled_gemm_cluster::run_blockscaled_gemm_cluster<Cfg>(m, n, k, transA, transB);
    }
    if (output_type == "BF16") {
      using Cfg = GenericBlockScaledClusterConfig<ElementA, ElementB, sycl::ext::oneapi::bfloat16, ElementSF, SVS, BST, TM, TN, TK, CM, CN, CoopSF>;
      return xe4_blockscaled_gemm_cluster::run_blockscaled_gemm_cluster<Cfg>(m, n, k, transA, transB);
    }
    std::cerr << "Error: Unsupported output type: " << output_type << std::endl;
    return 1;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Run all valid {input_type, SFVecSize, output_type, cluster_m, cluster_n} combos
// in a single process. This keeps the SYCL runtime/simulator alive across all configs.
// When coop_sf=false: Non-cooperative SF, TileK=256 for all VS.
// When coop_sf=true:  Cooperative SF, TileK=256 for VS=16, TileK=512 for VS=32.
//                     MXFP8 is skipped (max atom K=256 < required 512).
int run_all_configs(int m, int n, int k, char transA, char transB, bool coop_sf)
{
  using namespace cutlass;

  struct ConfigDesc {
    const char* input_type;
    int sfvec;
    const char* output_type;
    int cm;
    int cn;
  };

  // All valid combos: {input_type x valid SFVecSizes x output_types x cluster_shapes}
  // Cluster shapes: (2,1), (1,2), (2,2)
  // NVFP4: VS=16 only,  NVFP4+/MXFP4: VS=16 and 32,  MXFP8: VS=32 only (non-coop only)
  const ConfigDesc configs[] = {
    // NVFP4: SFVecSize=16 only
    {"NVFP4", 16, "FP32", 2, 1}, {"NVFP4", 16, "FP16", 2, 1}, {"NVFP4", 16, "BF16", 2, 1},
    {"NVFP4", 16, "FP32", 1, 2}, {"NVFP4", 16, "FP16", 1, 2}, {"NVFP4", 16, "BF16", 1, 2},
    {"NVFP4", 16, "FP32", 2, 2}, {"NVFP4", 16, "FP16", 2, 2}, {"NVFP4", 16, "BF16", 2, 2},
    // NVFP4+: SFVecSize=16
    {"NVFP4+", 16, "FP32", 2, 1}, {"NVFP4+", 16, "FP16", 2, 1}, {"NVFP4+", 16, "BF16", 2, 1},
    {"NVFP4+", 16, "FP32", 1, 2}, {"NVFP4+", 16, "FP16", 1, 2}, {"NVFP4+", 16, "BF16", 1, 2},
    {"NVFP4+", 16, "FP32", 2, 2}, {"NVFP4+", 16, "FP16", 2, 2}, {"NVFP4+", 16, "BF16", 2, 2},
    // NVFP4+: SFVecSize=32
    {"NVFP4+", 32, "FP32", 2, 1}, {"NVFP4+", 32, "FP16", 2, 1}, {"NVFP4+", 32, "BF16", 2, 1},
    {"NVFP4+", 32, "FP32", 1, 2}, {"NVFP4+", 32, "FP16", 1, 2}, {"NVFP4+", 32, "BF16", 1, 2},
    {"NVFP4+", 32, "FP32", 2, 2}, {"NVFP4+", 32, "FP16", 2, 2}, {"NVFP4+", 32, "BF16", 2, 2},
    // MXFP4: SFVecSize=16
    {"MXFP4", 16, "FP32", 2, 1}, {"MXFP4", 16, "FP16", 2, 1}, {"MXFP4", 16, "BF16", 2, 1},
    {"MXFP4", 16, "FP32", 1, 2}, {"MXFP4", 16, "FP16", 1, 2}, {"MXFP4", 16, "BF16", 1, 2},
    {"MXFP4", 16, "FP32", 2, 2}, {"MXFP4", 16, "FP16", 2, 2}, {"MXFP4", 16, "BF16", 2, 2},
    // MXFP4: SFVecSize=32
    {"MXFP4", 32, "FP32", 2, 1}, {"MXFP4", 32, "FP16", 2, 1}, {"MXFP4", 32, "BF16", 2, 1},
    {"MXFP4", 32, "FP32", 1, 2}, {"MXFP4", 32, "FP16", 1, 2}, {"MXFP4", 32, "BF16", 1, 2},
    {"MXFP4", 32, "FP32", 2, 2}, {"MXFP4", 32, "FP16", 2, 2}, {"MXFP4", 32, "BF16", 2, 2},
    // MXFP8: SFVecSize=32 only (non-cooperative only; skipped when coop_sf=true)
    {"MXFP8", 32, "FP32", 2, 1}, {"MXFP8", 32, "FP16", 2, 1}, {"MXFP8", 32, "BF16", 2, 1},
    {"MXFP8", 32, "FP32", 1, 2}, {"MXFP8", 32, "FP16", 1, 2}, {"MXFP8", 32, "BF16", 1, 2},
    {"MXFP8", 32, "FP32", 2, 2}, {"MXFP8", 32, "FP16", 2, 2}, {"MXFP8", 32, "BF16", 2, 2},
  };

  int total = 0, passed = 0, failed = 0, skipped = 0;
  for (const auto& cfg : configs) {
    std::string it(cfg.input_type);
    std::string ot(cfg.output_type);

    // Skip MXFP8 in cooperative mode (max atom K=256 < required 512)
    if (coop_sf && it == "MXFP8") {
      ++skipped;
      continue;
    }

    ++total;
    std::cout << "\n======================================================================"
              << "\n[" << total << "] " << it << " SFVecSize=" << cfg.sfvec << " output=" << ot
              << " cluster=<" << cfg.cm << "," << cfg.cn << ",1>"
              << " coop_sf=" << (coop_sf ? 1 : 0)
              << "  M=" << m << " N=" << n << " K=" << k
              << " transA=" << transA << " transB=" << transB
              << "\n======================================================================\n";

    int rc = 1;
    try {
      if (coop_sf) {
        // Cooperative SF: VS=16 -> TileK=256, VS=32 -> TileK=512
#define DISPATCH_COOP_K16(cm_, cn_, ElemA, ElemB, ElemSF)                          \
        if (cfg.cm == cm_ && cfg.cn == cn_) {                                       \
          rc = run_gemm_config<ElemA, ElemB, ElemSF,                                \
                16, BlockScaleTypeMap<ElemSF, 16>::value,                            \
                128, 256, 256, cm_, cn_, true>(ot, m, n, k, transA, transB);        \
        }
#define DISPATCH_COOP_K32(cm_, cn_, ElemA, ElemB, ElemSF)                          \
        if (cfg.cm == cm_ && cfg.cn == cn_) {                                       \
          rc = run_gemm_config<ElemA, ElemB, ElemSF,                                \
                32, BlockScaleTypeMap<ElemSF, 32>::value,                            \
                128, 256, 512, cm_, cn_, true>(ot, m, n, k, transA, transB);        \
        }

        if (it == "NVFP4") {
          DISPATCH_COOP_K16(2, 1, float_e2m1_t, float_e2m1_t, float_ue4m3_t)
          DISPATCH_COOP_K16(1, 2, float_e2m1_t, float_e2m1_t, float_ue4m3_t)
          DISPATCH_COOP_K16(2, 2, float_e2m1_t, float_e2m1_t, float_ue4m3_t)
        } else if (it == "NVFP4+" && cfg.sfvec == 16) {
          DISPATCH_COOP_K16(2, 1, float_e2m1_t, float_e2m1_t, float_ue5m3_t)
          DISPATCH_COOP_K16(1, 2, float_e2m1_t, float_e2m1_t, float_ue5m3_t)
          DISPATCH_COOP_K16(2, 2, float_e2m1_t, float_e2m1_t, float_ue5m3_t)
        } else if (it == "NVFP4+" && cfg.sfvec == 32) {
          DISPATCH_COOP_K32(2, 1, float_e2m1_t, float_e2m1_t, float_ue5m3_t)
          DISPATCH_COOP_K32(1, 2, float_e2m1_t, float_e2m1_t, float_ue5m3_t)
          DISPATCH_COOP_K32(2, 2, float_e2m1_t, float_e2m1_t, float_ue5m3_t)
        } else if (it == "MXFP4" && cfg.sfvec == 16) {
          DISPATCH_COOP_K16(2, 1, float_e2m1_t, float_e2m1_t, float_ue8m0_t)
          DISPATCH_COOP_K16(1, 2, float_e2m1_t, float_e2m1_t, float_ue8m0_t)
          DISPATCH_COOP_K16(2, 2, float_e2m1_t, float_e2m1_t, float_ue8m0_t)
        } else if (it == "MXFP4" && cfg.sfvec == 32) {
          DISPATCH_COOP_K32(2, 1, float_e2m1_t, float_e2m1_t, float_ue8m0_t)
          DISPATCH_COOP_K32(1, 2, float_e2m1_t, float_e2m1_t, float_ue8m0_t)
          DISPATCH_COOP_K32(2, 2, float_e2m1_t, float_e2m1_t, float_ue8m0_t)
        }
#undef DISPATCH_COOP_K16
#undef DISPATCH_COOP_K32
      } else {
        // Non-cooperative: TileK=256 for all VS
#define DISPATCH_NONCOOP(cm_, cn_, ElemA, ElemB, ElemSF, svs)                      \
        if (cfg.cm == cm_ && cfg.cn == cn_) {                                       \
          rc = run_gemm_config<ElemA, ElemB, ElemSF,                                \
                svs, BlockScaleTypeMap<ElemSF, svs>::value,                          \
                128, 256, 256, cm_, cn_, false>(ot, m, n, k, transA, transB);          \
        }

        if (it == "NVFP4") {
          DISPATCH_NONCOOP(2, 1, float_e2m1_t, float_e2m1_t, float_ue4m3_t, 16)
          DISPATCH_NONCOOP(1, 2, float_e2m1_t, float_e2m1_t, float_ue4m3_t, 16)
          DISPATCH_NONCOOP(2, 2, float_e2m1_t, float_e2m1_t, float_ue4m3_t, 16)
        } else if (it == "NVFP4+" && cfg.sfvec == 16) {
          DISPATCH_NONCOOP(2, 1, float_e2m1_t, float_e2m1_t, float_ue5m3_t, 16)
          DISPATCH_NONCOOP(1, 2, float_e2m1_t, float_e2m1_t, float_ue5m3_t, 16)
          DISPATCH_NONCOOP(2, 2, float_e2m1_t, float_e2m1_t, float_ue5m3_t, 16)
        } else if (it == "NVFP4+" && cfg.sfvec == 32) {
          DISPATCH_NONCOOP(2, 1, float_e2m1_t, float_e2m1_t, float_ue5m3_t, 32)
          DISPATCH_NONCOOP(1, 2, float_e2m1_t, float_e2m1_t, float_ue5m3_t, 32)
          DISPATCH_NONCOOP(2, 2, float_e2m1_t, float_e2m1_t, float_ue5m3_t, 32)
        } else if (it == "MXFP4" && cfg.sfvec == 16) {
          DISPATCH_NONCOOP(2, 1, float_e2m1_t, float_e2m1_t, float_ue8m0_t, 16)
          DISPATCH_NONCOOP(1, 2, float_e2m1_t, float_e2m1_t, float_ue8m0_t, 16)
          DISPATCH_NONCOOP(2, 2, float_e2m1_t, float_e2m1_t, float_ue8m0_t, 16)
        } else if (it == "MXFP4" && cfg.sfvec == 32) {
          DISPATCH_NONCOOP(2, 1, float_e2m1_t, float_e2m1_t, float_ue8m0_t, 32)
          DISPATCH_NONCOOP(1, 2, float_e2m1_t, float_e2m1_t, float_ue8m0_t, 32)
          DISPATCH_NONCOOP(2, 2, float_e2m1_t, float_e2m1_t, float_ue8m0_t, 32)
        } else if (it == "MXFP8") {
          DISPATCH_NONCOOP(2, 1, float_e4m3_t, float_e4m3_t, float_ue8m0_t, 32)
          DISPATCH_NONCOOP(1, 2, float_e4m3_t, float_e4m3_t, float_ue8m0_t, 32)
          DISPATCH_NONCOOP(2, 2, float_e4m3_t, float_e4m3_t, float_ue8m0_t, 32)
        }
#undef DISPATCH_NONCOOP
      }
    } catch (const std::exception& e) {
      std::cerr << "EXCEPTION: " << e.what() << std::endl;
      rc = 1;
    }

    if (rc == 0) { ++passed; std::cout << ">>> PASSED\n"; }
    else         { ++failed; std::cout << ">>> FAILED\n"; }
  }

  std::cout << "\n======================================================================"
            << "\nSUMMARY: " << passed << "/" << total << " passed, " << failed << " failed"
            << (skipped > 0 ? ", " + std::to_string(skipped) + " skipped" : "")
            << " (coop_sf=" << (coop_sf ? 1 : 0) << ")"
            << "\n======================================================================\n";
  return failed;
}

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog
            << " [M] [N] [K] [transA] [transB] [input_type] [coop_sf]"
            << " [SFVecSize] [output_type] [cluster_m] [cluster_n]\n"
            << "\n"
            << "  M, N, K       : Problem dimensions (default: 512 1024 2048)\n"
            << "  transA/transB : Transpose flags, 'T' or 'N' (default: T N)\n"
            << "  input_type    : NVFP4, NVFP4+, MXFP4, MXFP8, ALL (default: NVFP4)\n"
            << "                  ALL runs all valid combos in a single process\n"
            << "                  (remaining args except coop_sf ignored when ALL)\n"
            << "  coop_sf       : Cooperative SF loading, 0=disabled 1=enabled (default: 0)\n"
            << "                  coop_sf=1 NOT supported for MXFP8 (max atom K=256)\n"
            << "  SFVecSize     : Scale factor block size, 16 or 32 (default: 16)\n"
            << "  output_type   : FP32, FP16, BF16 (default: FP32)\n"
            << "  cluster_m     : Cluster size along M, 1 or 2 (default: 2)\n"
            << "  cluster_n     : Cluster size along N, 1 or 2 (default: 2)\n"
            << "\n"
            << "  TileK selection (automatic based on coop_sf and SFVecSize):\n"
            << "    Non-cooperative (default): TileK=256 for all VS (padding handles cm_8x32B)\n"
            << "    Cooperative (coop_sf=1):   VS=16 -> TileK=256, VS=32 -> TileK=512\n"
            << "\n"
            << "  BlockScaleType (auto-selected from input_type + SFVecSize):\n"
            << "    NVFP4              + SFVecSize=16 -> ue4m3k16 (type 5)\n"
            << "    NVFP4+             + SFVecSize=16 -> ue5m3k16 (type 3) / ue5m3k32 (type 2)\n"
            << "    MXFP4              + SFVecSize=16 -> ue8m0k16 (type 1) / ue8m0k32 (type 0)\n"
            << "    MXFP4/MXFP8        + SFVecSize=32 -> ue8m0k32 (type 0)\n"
            << "\n"
            << "Examples:\n"
            << "  # Single config: NVFP4+ with SFVecSize=16, FP16 output, cluster <2,1,1>\n"
            << "  " << prog << " 512 512 512 T N NVFP4+ 0 16 FP16 2 1\n"
            << "\n"
            << "  # Single config: MXFP4 with SFVecSize=32, FP32 output, cluster <2,2,1>\n"
            << "  " << prog << " 512 512 512 T N MXFP4 0 32 FP32 2 2\n"
            << "\n"
            << "  # Single config with cooperative SF: NVFP4 cluster <1,2,1>\n"
            << "  " << prog << " 512 512 512 T N NVFP4 1 16 FP32 1 2\n"
            << "\n"
            << "  # Run ALL configs non-cooperative (keeps simulator alive):\n"
            << "  " << prog << " 512 512 512 T N ALL\n"
            << "\n"
            << "  # Run ALL configs with cooperative SF (MXFP8 auto-skipped):\n"
            << "  " << prog << " 512 512 512 T N ALL 1\n"
            << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  // ---- Parse command-line arguments ----

  int m = 512;
  if (argc >= 2) {
    if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
      print_usage(argv[0]);
      return 0;
    }
    sscanf(argv[1], "%d", &m);
  }
  int n = 512;
  if (argc >= 3) sscanf(argv[2], "%d", &n);
  int k = 512;
  if (argc >= 4) sscanf(argv[3], "%d", &k);

  char transA = 'T';
  if (argc >= 5) sscanf(argv[4], "%c", &transA);
  char transB = 'N';
  if (argc >= 6) sscanf(argv[5], "%c", &transB);

  std::string input_type = "NVFP4";
  if (argc >= 7) input_type = argv[6];

  // ---- Validate input datatype: must be NVFP4, NVFP4+, MXFP4, MXFP8, or ALL ----
  if (input_type != "NVFP4"  && input_type != "NVFP4+" && input_type != "MXFP4" && input_type != "MXFP8" && input_type != "ALL") {
    std::cerr << "Error: input_type must be one of: NVFP4, NVFP4+, MXFP4, MXFP8, ALL. Got: "
              << input_type << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  int coop_sf = 0;
  if (argc >= 8) sscanf(argv[7], "%d", &coop_sf);

  // ---- Validate coop_sf: must be 0 or 1 ----
  if (coop_sf != 0 && coop_sf != 1) {
    std::cerr << "Error: coop_sf must be 0 (disabled) or 1 (enabled). Got: " << coop_sf << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  // ---- ALL mode: run all valid configs in a single process ----
  // When ALL is specified, remaining args (SFVecSize, output_type, cluster shape) are ignored.
  if (input_type == "ALL") {
    return run_all_configs(m, n, k, transA, transB, coop_sf != 0);
  }

  int SFVecSize = 16;
  if (argc >= 9) sscanf(argv[8], "%d", &SFVecSize);

  std::string output_type = "FP32";
  if (argc >= 10) output_type = argv[9];

  int cluster_m = 2;
  if (argc >= 11) sscanf(argv[10], "%d", &cluster_m);
  int cluster_n = 2;
  if (argc >= 12) sscanf(argv[11], "%d", &cluster_n);

  // ---- Validate SFVecSize: must be 16 or 32 ----
  if (SFVecSize != 16 && SFVecSize != 32) {
    std::cerr << "Error: SFVecSize must be 16 or 32. Got: " << SFVecSize << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  // ---- Validate output datatype: must be FP32, FP16, or BF16 ----
  if (output_type != "FP32" && output_type != "FP16" && output_type != "BF16") {
    std::cerr << "Error: output_type must be one of: FP32, FP16, BF16. Got: "
              << output_type << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  // ---- NVFP4 only supports SFVecSize=16 (no ue4m3k32 encoding exists) ----
  if (input_type == "NVFP4" && SFVecSize != 16) {
    std::cerr << "Error: NVFP4 only supports SFVecSize=16 (no ue4m3k32 HW encoding). Got: "
              << SFVecSize << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  // ---- MXFP8 only supports SFVecSize=32 (no ue8m0k16 encoding exists) ----
  if (input_type == "MXFP8" && SFVecSize != 32) {
    std::cerr << "Error: MXFP8 only supports SFVecSize=32 (no ue8m0k16 HW encoding). Got: "
              << SFVecSize << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  // ---- MXFP8 does not support cooperative SF loading ----
  // FP8 max atom K is 256, but cooperative SF with VS=32 on cluster configs requires
  // atom K >= 512. Always use trivial (non-cooperative) SF layout for FP8.
  if (input_type == "MXFP8" && coop_sf) {
    std::cerr << "Error: Cooperative SF loading is not supported for MXFP8 configs "
              << "(max atom K=256 < required 512 for VS=32 cluster)." << std::endl;
    return 1;
  }

  // ---- Print configuration summary ----
  std::cout << "=== Block-Scaled GEMM with Cluster Configuration ===" << std::endl;
  std::cout << "  Input type    : " << input_type << std::endl;
  std::cout << "  Output type   : " << output_type << std::endl;
  std::cout << "  Problem shape : M=" << m << " N=" << n << " K=" << k << std::endl;
  std::cout << "  Transpose     : A=" << transA << " B=" << transB << std::endl;
  std::cout << "  SFVecSize     : " << SFVecSize << std::endl;
  std::cout << "  Cluster shape : <" << cluster_m << ", " << cluster_n << ", 1>" << std::endl;
  std::cout << "  Coop SF load  : " << (coop_sf ? "enabled" : "disabled") << std::endl;
  std::cout << "====================================================" << std::endl;

  using namespace cutlass;

  // ---- Dispatch by input type ----
  // Cluster shape dispatch uses a macro to avoid repetitive template instantiation.
  // For each (cluster_m, cluster_n) combo, instantiate with the correct tile shape.
  //
  // Non-cooperative (default): TileK=256 for both VS=16 and VS=32.
  //   SF SMEM padding (sf_bK = max(bK/VS, 8)) handles the cm_8x32B constraint,
  //   allowing any TileK/VS combo to work safely.
  //
  // Cooperative (--coop_sf=1): TileK=256 for VS=16, TileK=512 for VS=32.
  //   Padding cannot fix cooperative SF (intra-stage CTA collision), so TileK must
  //   be large enough that each CTA's ADMA box satisfies cm_8x32B after cluster
  //   truncation: TileK / SFVecSize / max(ClusterM, ClusterN) >= 8.
#define DISPATCH_CLUSTER(cm, cn, ElementA, ElementB, ElementSF)                       \
  if (cluster_m == cm && cluster_n == cn) {                                           \
    if (SFVecSize == 16) {                                                            \
      if (coop_sf)                                                                    \
        return run_gemm_config<ElementA, ElementB, ElementSF,                         \
          16, BlockScaleTypeMap<ElementSF, 16>::value, 128, 256, 256, cm, cn, true>(  \
            output_type, m, n, k, transA, transB);                                    \
      else                                                                            \
        return run_gemm_config<ElementA, ElementB, ElementSF,                         \
          16, BlockScaleTypeMap<ElementSF, 16>::value, 128, 256, 128, cm, cn, false>(  \
            output_type, m, n, k, transA, transB);                                    \
    } else {                                                                          \
      if (coop_sf)                                                                    \
        return run_gemm_config<ElementA, ElementB, ElementSF,                         \
          32, BlockScaleTypeMap<ElementSF, 32>::value, 128, 256, 512, cm, cn, true>(  \
            output_type, m, n, k, transA, transB);                                    \
      else                                                                            \
        return run_gemm_config<ElementA, ElementB, ElementSF,                         \
          32, BlockScaleTypeMap<ElementSF, 32>::value, 128, 256, 256, cm, cn, false>(  \
            output_type, m, n, k, transA, transB);                                    \
    }                                                                                 \
  }

  if (input_type == "NVFP4") {
    DISPATCH_CLUSTER(2, 1, float_e2m1_t, float_e2m1_t, float_ue4m3_t)
    DISPATCH_CLUSTER(1, 2, float_e2m1_t, float_e2m1_t, float_ue4m3_t)
    DISPATCH_CLUSTER(2, 2, float_e2m1_t, float_e2m1_t, float_ue4m3_t)
  } else if (input_type == "NVFP4+") {
    DISPATCH_CLUSTER(2, 1, float_e2m1_t, float_e2m1_t, float_ue5m3_t)
    DISPATCH_CLUSTER(1, 2, float_e2m1_t, float_e2m1_t, float_ue5m3_t)
    DISPATCH_CLUSTER(2, 2, float_e2m1_t, float_e2m1_t, float_ue5m3_t)
  } else if (input_type == "MXFP4") {
    DISPATCH_CLUSTER(2, 1, float_e2m1_t, float_e2m1_t, float_ue8m0_t)
    DISPATCH_CLUSTER(1, 2, float_e2m1_t, float_e2m1_t, float_ue8m0_t)
    DISPATCH_CLUSTER(2, 2, float_e2m1_t, float_e2m1_t, float_ue8m0_t)
  } else if (input_type == "MXFP8") {
    DISPATCH_CLUSTER(2, 1, float_e4m3_t, float_e4m3_t, float_ue8m0_t)
    DISPATCH_CLUSTER(1, 2, float_e4m3_t, float_e4m3_t, float_ue8m0_t)
    DISPATCH_CLUSTER(2, 2, float_e4m3_t, float_e4m3_t, float_ue8m0_t)
  }

#undef DISPATCH_CLUSTER

  std::cerr << "Error: Unsupported input type: " << input_type << std::endl;
  print_usage(argv[0]);
  return 1;
}