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
// XE4 Block-Scaled GEMM — Generic CLI Test Driver
//
// Supports multiple block-scaled data formats via command-line arguments:
//   NVFP4, NVFP4+, MXFP4, MXFP8
//
// CLI Arguments (all positional, all optional):
//   argv[1]  : M          (default: 512)
//   argv[2]  : N          (default: 1024)
//   argv[3]  : K          (default: 2048)
//   argv[4]  : transA     (default: 'T')
//   argv[5]  : transB     (default: 'N')
//   argv[6]  : input_type (default: "NVFP4", valid: NVFP4/NVFP4+/MXFP4/MXFP8/ALL)
//   argv[7]  : SFVecSize  (default: 16, valid: 16 or 32; ignored when input_type=ALL)
//   argv[8]  : output_type(default: "FP32",  valid: FP32/FP16/BF16; ignored when input_type=ALL)
//   argv[9]  : decouple_sf_load (default: 1, valid: 0 or 1)
//
// Element type mapping:
//   NVFP4  : A/B = float_e2m1_t (4-bit), SF = float_ue4m3_t, BlockScaleType = 5
//   NVFP4+ : A/B = float_e2m1_t (4-bit), SF = float_ue5m3_t, BlockScaleType = 2 (SFVecSize 32) or 3 (SFVecSize 16)
//   MXFP4  : A/B = float_e2m1_t (4-bit), SF = float_ue8m0_t, BlockScaleType = 0 (SFVecize 32) or 1 (SFVecSize 16)
//   MXFP8  : A/B = float_e4m3_t (8-bit), SF = float_ue8m0_t, BlockScaleType = 0 (SFVecSize 32)
//
// All GEMM infrastructure (device kernel, host setup, dispatch, validation)
// lives in amma_adma_xe4_blockscaled_gemm_base.hpp.
//
////////////////////////////////////////////////////////////////////////////////////////////////---

#include "amma_adma_xe4_blockscaled_gemm_base.hpp"
#include <string>
#include <type_traits>
using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Generic Config struct — parameterized on element types, tile shape, etc.
//
// Provides all 16 members required by the base header's GEMM infrastructure.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <class ElementA_, class ElementB_, class ElementC_, class ElementSF_,
          int SFVecSize_, int BlockScaleType_,
          int TileM_, int TileN_, int TileK_>
struct GenericBlockScaledConfig {

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
  using ClusterShape_MNK = cute::Shape<cute::_1, cute::_1, cute::_1>;

  // ---- MMA atom ----
  // Use bs_op_selector to automatically select the correct block-scaled MMA op.
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
      ClusterShape_MNK,                    // Cluster shape (1,1,1 for single-CTA)
      MajorA, MajorB>()                    // A and B layout majors
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
// BlockScaleType, and tile shape, resolve runtime output_type and run GEMM.
// Uses if-constexpr to skip invalid combos (e.g. FP8 + SFVecSize=16).
template <class ElementA, class ElementB, class ElementSF,
          int SVS, int BST, int TM, int TN, int TK>
int run_gemm_config(const std::string& output_type,
                    int m, int n, int k, char transA, char transB, bool decouple_sf_load)
{
  // Guard: FP8 types (float_e4m3_t) only support SFVecSize=32.
  // Prevent instantiation of invalid ADMA copy atoms at compile-time.
  if constexpr (std::is_same_v<ElementA, cutlass::float_e4m3_t> && SVS != 32) {
    std::cerr << "Error: FP8 does not support SFVecSize=" << SVS << std::endl;
    return 1;
  } else {
    if (output_type == "FP32") {
      using Cfg = GenericBlockScaledConfig<ElementA, ElementB, float, ElementSF, SVS, BST, TM, TN, TK>;
      return xe4_blockscaled_gemm::run_blockscaled_gemm<Cfg>(m, n, k, transA, transB, decouple_sf_load);
    }
    if (output_type == "FP16") {
      using Cfg = GenericBlockScaledConfig<ElementA, ElementB, sycl::half, ElementSF, SVS, BST, TM, TN, TK>;
      return xe4_blockscaled_gemm::run_blockscaled_gemm<Cfg>(m, n, k, transA, transB, decouple_sf_load);
    }
    if (output_type == "BF16") {
      using Cfg = GenericBlockScaledConfig<ElementA, ElementB, sycl::ext::oneapi::bfloat16, ElementSF, SVS, BST, TM, TN, TK>;
      return xe4_blockscaled_gemm::run_blockscaled_gemm<Cfg>(m, n, k, transA, transB, decouple_sf_load);
    }
    std::cerr << "Error: Unsupported output type: " << output_type << std::endl;
    return 1;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Run all valid {input_type, SFVecSize, output_type} combos in a single process.
// This keeps the SYCL runtime/simulator alive across all configs.
int run_all_configs(int m, int n, int k, char transA, char transB, bool decouple_sf_load)
{
  using namespace cutlass;

  struct ConfigDesc {
    const char* input_type;
    int sfvec;
    const char* output_type;
  };

  // All valid combos: 4 input types x valid SFVecSizes x 3 output types = 21 configs
  const ConfigDesc configs[] = {
    // NVFP4: SFVecSize=16 only
    {"NVFP4", 16, "FP32"}, {"NVFP4", 16, "FP16"}, {"NVFP4", 16, "BF16"},
    // NVFP4+: SFVecSize=16 and 32
    {"NVFP4+", 16, "FP32"}, {"NVFP4+", 16, "FP16"}, {"NVFP4+", 16, "BF16"},
    {"NVFP4+", 32, "FP32"}, {"NVFP4+", 32, "FP16"}, {"NVFP4+", 32, "BF16"},
    // MXFP4: SFVecSize=16 and 32
    {"MXFP4", 16, "FP32"}, {"MXFP4", 16, "FP16"}, {"MXFP4", 16, "BF16"},
    {"MXFP4", 32, "FP32"}, {"MXFP4", 32, "FP16"}, {"MXFP4", 32, "BF16"},
    // MXFP8: SFVecSize=32 only
    {"MXFP8", 32, "FP32"}, {"MXFP8", 32, "FP16"}, {"MXFP8", 32, "BF16"},
  };

  int total = 0, passed = 0, failed = 0;
  for (const auto& cfg : configs) {
    ++total;
    std::string it(cfg.input_type);
    std::string ot(cfg.output_type);

    std::cout << "\n======================================================================"
              << "\n[" << total << "] " << it << " SFVecSize=" << cfg.sfvec << " output=" << ot
              << "  M=" << m << " N=" << n << " K=" << k
              << " transA=" << transA << " transB=" << transB
              << "\n======================================================================\n";

    int rc = 1;
    try {
      if (it == "NVFP4") {
        rc = run_gemm_config<float_e2m1_t, float_e2m1_t, float_ue4m3_t,
                             16, BlockScaleTypeMap<float_ue4m3_t, 16>::value,
                             128, 256, 128>(ot, m, n, k, transA, transB, decouple_sf_load);
      } else if (it == "NVFP4+" && cfg.sfvec == 16) {
        rc = run_gemm_config<float_e2m1_t, float_e2m1_t, float_ue5m3_t,
                             16, BlockScaleTypeMap<float_ue5m3_t, 16>::value,
                             128, 256, 128>(ot, m, n, k, transA, transB, decouple_sf_load);
      } else if (it == "NVFP4+" && cfg.sfvec == 32) {
        rc = run_gemm_config<float_e2m1_t, float_e2m1_t, float_ue5m3_t,
                             32, BlockScaleTypeMap<float_ue5m3_t, 32>::value,
                             128, 256, 256>(ot, m, n, k, transA, transB, decouple_sf_load);
      } else if (it == "MXFP4" && cfg.sfvec == 16) {
        rc = run_gemm_config<float_e2m1_t, float_e2m1_t, float_ue8m0_t,
                             16, BlockScaleTypeMap<float_ue8m0_t, 16>::value,
                             128, 256, 128>(ot, m, n, k, transA, transB, decouple_sf_load);
      } else if (it == "MXFP4" && cfg.sfvec == 32) {
        rc = run_gemm_config<float_e2m1_t, float_e2m1_t, float_ue8m0_t,
                             32, BlockScaleTypeMap<float_ue8m0_t, 32>::value,
                             128, 256, 256>(ot, m, n, k, transA, transB, decouple_sf_load);
      } else if (it == "MXFP8") {
        rc = run_gemm_config<float_e4m3_t, float_e4m3_t, float_ue8m0_t,
                             32, BlockScaleTypeMap<float_ue8m0_t, 32>::value,
                             128, 256, 256>(ot, m, n, k, transA, transB, decouple_sf_load);
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
            << "\n======================================================================\n";
  return failed;
}

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog
            << " [M] [N] [K] [transA] [transB] [input_type] [SFVecSize]"
            << " [output_type] [decouple_sf_load]\n"
            << "\n"
            << "  M, N, K       : Problem dimensions (default: 512 1024 2048)\n"
            << "  transA/transB : Transpose flags, 'T' or 'N' (default: T N)\n"
            << "  input_type    : NVFP4, NVFP4+, MXFP4, MXFP8, ALL (default: NVFP4)\n"
            << "                  ALL runs all 18 valid combos in a single process\n"
            << "                  (remaining args ignored when ALL is specified)\n"
            << "  SFVecSize     : Scale factor block size, 16 or 32 (default: 16)\n"
            << "  output_type   : FP32, FP16, BF16 (default: FP32)\n"
            << "  decouple_sf_load : Whether to decouple scale factor load from data load (default: 1)\n"
            << "\n"
            << "  BlockScaleType (auto-selected from input_type + SFVecSize):\n"
            << "    NVFP4              + SFVecSize=16 -> ue4m3k16 (type 5)\n"
            << "    NVFP4+             + SFVecSize=16 -> ue5m3k16 (type 3) / ue5m3k32 (type 2)\n"
            << "    MXFP4              + SFVecSize=16 -> ue8m0k16 (type 1) / ue8m0k32 (type 0)\n"
            << "    MXFP4/MXFP8        + SFVecSize=32 -> ue8m0k32 (type 0)\n"
            << "\n"
            << "Examples:\n"
            << "  # Single config: NVFP4+ with SFVecSize=16, FP16 output\n"
            << "  " << prog << " 512 512 512 T N NVFP4+ 16 FP16 1\n"
            << "\n"
            << "  # Single config: MXFP8 with SFVecSize=32, FP32 output\n"
            << "  " << prog << " 512 512 512 T N MXFP8 32 FP32 0\n"
            << "\n"
            << "  # Run ALL 18 valid configs in a single process (keeps simulator alive):\n"
            << "  " << prog << " 512 512 512 T N ALL\n"
            << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  // ---- Set default values ----
  int m = 512;
  int n = 1024;
  int k = 2048;
  char transA = 'T';
  char transB = 'N';
  std::string input_type = "NVFP4";
  int SFVecSize = 16;
  std::string output_type = "FP32";
  bool decouple_sf_load = true;

  // ---- Parse command-line arguments ----
  if (argc >= 2) sscanf(argv[1], "%d", &m);
  if (argc >= 3) sscanf(argv[2], "%d", &n);
  if (argc >= 4) sscanf(argv[3], "%d", &k);
  if (argc >= 5) sscanf(argv[4], "%c", &transA);
  if (argc >= 6) sscanf(argv[5], "%c", &transB);
  if (argc >= 7) input_type = argv[6];
  if (argc >= 8) sscanf(argv[7], "%d", &SFVecSize);
  if (argc >= 9) output_type = argv[8];
  if (argc >= 10) decouple_sf_load = (std::string(argv[9]) != "0");

  // ---- Validate input datatype ----
  if (input_type != "NVFP4" && input_type != "NVFP4+" && input_type != "MXFP4" && input_type != "MXFP8" && input_type != "ALL") {
    std::cerr << "Error: input_type must be one of: NVFP4, NVFP4+, MXFP4, MXFP8, ALL. Got: "
              << input_type << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  // ---- ALL mode: run all 18 valid configs in a single process ----
  // When ALL is specified, remaining args (SFVecSize, output_type) are ignored.
  if (input_type == "ALL") {
    return run_all_configs(m, n, k, transA, transB, decouple_sf_load);
  }

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

  // ---- Print configuration summary ----
  std::cout << "=== Block-Scaled GEMM Configuration ===" << std::endl;
  std::cout << "  Input type    : " << input_type << std::endl;
  std::cout << "  Output type   : " << output_type << std::endl;
  std::cout << "  Problem shape : M=" << m << " N=" << n << " K=" << k << std::endl;
  std::cout << "  Transpose     : A=" << transA << " B=" << transB << std::endl;
  std::cout << "  SFVecSize     : " << SFVecSize << std::endl;
  std::cout << "  Load SF all stages at once: " << (decouple_sf_load ? "Yes" : "No") << std::endl;
  std::cout << "======================================" << std::endl;
  
  using namespace cutlass;

  if (input_type == "NVFP4")
  {
    return run_gemm_config<float_e2m1_t, float_e2m1_t, float_ue4m3_t, 16, BlockScaleTypeMap<float_ue4m3_t, 16>::value, 128, 256, 128>
                          (output_type, m, n, k, transA, transB, decouple_sf_load);
  }
  else if (input_type == "NVFP4+")
  {
    if (SFVecSize == 16)
      return run_gemm_config<float_e2m1_t, float_e2m1_t, float_ue5m3_t, 16, BlockScaleTypeMap<float_ue5m3_t, 16>::value, 128, 256, 128>
                          (output_type, m, n, k, transA, transB, decouple_sf_load);
    else
      return run_gemm_config<float_e2m1_t, float_e2m1_t, float_ue5m3_t, 32, BlockScaleTypeMap<float_ue5m3_t, 32>::value, 128, 256, 256>
                          (output_type, m, n, k, transA, transB, decouple_sf_load);
  }
  else if (input_type == "MXFP4")
  {
    if (SFVecSize == 16)
      return run_gemm_config<float_e2m1_t, float_e2m1_t, float_ue8m0_t, 16, BlockScaleTypeMap<float_ue8m0_t, 16>::value, 128, 256, 128>
                          (output_type, m, n, k, transA, transB, decouple_sf_load);
    else
      return run_gemm_config<float_e2m1_t, float_e2m1_t, float_ue8m0_t, 32, BlockScaleTypeMap<float_ue8m0_t, 32>::value, 128, 256, 256>
                          (output_type, m, n, k, transA, transB, decouple_sf_load);
  }
  else
  {
    return run_gemm_config<float_e4m3_t, float_e4m3_t, float_ue8m0_t, 32, BlockScaleTypeMap<float_ue8m0_t, 32>::value, 128, 256, 256>
                         (output_type, m, n, k, transA, transB, decouple_sf_load);
  }
}
