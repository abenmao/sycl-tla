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

#include "gemm_blockscaled.hpp"

namespace {
// Helper struct for operand specification
struct OperandSpec {
  std::string elem, sf, block;
  
  bool is_plain() const { return elem == "bf16" || elem == "fp16"; }
  bool is_mx() const { return elem == "fp4" || elem == "fp8"; }
  bool empty() const { return elem.empty() && sf.empty() && block.empty(); }
  
  void apply_plain_defaults() {
    if (is_plain()) {
      if (sf.empty()) sf = "ue8m0";
      if (block.empty()) block = "16";
    }
  }
  
  void validate(const char* prefix) const {
    if (elem.empty() || sf.empty() || block.empty()) {
      std::cerr << "Error: " << prefix << ", --sf" << prefix << " and --block" << prefix 
                << " must all be specified together.\n";
      std::exit(1);
    }
    if (elem != "fp4" && elem != "fp8" && elem != "bf16" && elem != "fp16") {
      std::cerr << "Error: " << prefix << "='" << elem << "' invalid. Use fp4, fp8, bf16 or fp16.\n";
      std::exit(1);
    }
    if (sf != "ue4m3" && sf != "ue5m3" && sf != "ue8m0") {
      std::cerr << "Error: --sf" << prefix << "='" << sf << "' invalid. Use ue4m3, ue5m3 or ue8m0.\n";
      std::exit(1);
    }
    if (block != "16" && block != "32") {
      std::cerr << "Error: --block" << prefix << "='" << block << "' invalid. Use 16 or 32.\n";
      std::exit(1);
    }
    
    // Hardware constraints per operand
    if (elem == "fp8" && (sf != "ue8m0" || block != "32")) {
      std::cerr << "Error: fp8 only supports --sf" << prefix << "=ue8m0 --block" << prefix << "=32.\n";
      std::exit(1);
    }
    if (is_plain() && (sf != "ue8m0" || block != "16")) {
      std::cerr << "Error: " << elem << " only supports --sf" << prefix << "=ue8m0 --block" << prefix << "=16.\n";
      std::exit(1);
    }
    if (elem == "fp4" && sf == "ue4m3" && block != "16") {
      std::cerr << "Error: --sf" << prefix << "=ue4m3 requires --block" << prefix << "=16.\n";
      std::exit(1);
    }
    if (sf != "ue8m0" && !is_plain() && elem != "fp4") {
      std::cerr << "Error: --sf" << prefix << "=" << sf << " only valid with " << prefix << "=fp4.\n";
      std::exit(1);
    }
  }
};

void validate_combo(const OperandSpec& a, const OperandSpec& b, const std::string& dtype_d) {
  if (a.is_plain() && b.is_plain()) {
    std::cerr << "Error: both operands cannot be plain dtypes for block-scaled GEMM.\n";
    std::exit(1);
  }
  if (a.is_mx() && b.is_mx()) {
    if (a.sf != b.sf) {
      std::cerr << "Error: --sfA=" << a.sf << " and --sfB=" << b.sf 
                << " must match when both operands are MX.\n";
      std::exit(1);
    }
    if (a.block != b.block) {
      std::cerr << "Error: --blockA=" << a.block << " and --blockB=" << b.block 
                << " must match when both operands are MX.\n";
      std::exit(1);
    }
  }
  if (!dtype_d.empty()) {
    if (dtype_d != "fp32" && dtype_d != "fp16" && dtype_d != "bf16") {
      std::cerr << "Error: --D='" << dtype_d << "' invalid. Use fp32, fp16 or bf16.\n";
      std::exit(1);
    }
    if (a.is_plain() || b.is_plain()) {
      const std::string plain_type = a.is_plain() ? a.elem : b.elem;
      if (dtype_d != "fp32" && dtype_d != plain_type) {
        std::cerr << "Error: --D=" << dtype_d << " invalid for A=" << a.elem 
                  << " B=" << b.elem << ". D must be fp32 or " << plain_type << ".\n";
        std::exit(1);
      }
    }
  }
}

std::vector<std::string> build_config_names(const OperandSpec& a, const OperandSpec& b, const std::string& dtype_d) {
  std::string prefix = (a.elem == b.elem) ? a.elem : a.elem + "a_" + b.elem + "b";
  prefix += "_" + a.sf + "k" + a.block + "_" + b.sf + "k" + b.block;
  
  std::vector<std::string> names;
  if (!dtype_d.empty()) {
    names.push_back(prefix + "_" + dtype_d);
  } else if (a.is_plain() || b.is_plain()) {
    const std::string plain_type = a.is_plain() ? a.elem : b.elem;
    names.push_back(prefix + "_fp32");
    names.push_back(prefix + "_" + plain_type);
  } else {
    names.push_back(prefix + "_fp32");
    names.push_back(prefix + "_fp16");
    names.push_back(prefix + "_bf16");
  }
  return names;
}
}  // namespace

void Options::parse(int argc, char** argv) {
  cutlass::CommandLine cmd(argc, const_cast<char const**>(argv));
  help = cmd.check_cmd_line_flag("help");
  run_all = cmd.check_cmd_line_flag("run-all");
  
  int val;
  if (cmd.check_cmd_line_flag("m")) { cmd.get_cmd_line_argument("m", val); m = val; }
  if (cmd.check_cmd_line_flag("n")) { cmd.get_cmd_line_argument("n", val); n = val; }
  if (cmd.check_cmd_line_flag("k")) { cmd.get_cmd_line_argument("k", val); k = val; }
  if (cmd.check_cmd_line_flag("l")) { cmd.get_cmd_line_argument("l", val); l = val; }
  if (cmd.check_cmd_line_flag("coop_sf")) {
    int csf = 0;
    cmd.get_cmd_line_argument("coop_sf", csf);
    coop_sf = (csf != 0);
  }

  // Parse operand specifications
  OperandSpec a, b;
  std::string dtype_d;
  if (cmd.check_cmd_line_flag("A"))      cmd.get_cmd_line_argument("A",      a.elem);
  if (cmd.check_cmd_line_flag("sfA"))    cmd.get_cmd_line_argument("sfA",    a.sf);
  if (cmd.check_cmd_line_flag("blockA")) cmd.get_cmd_line_argument("blockA", a.block);
  if (cmd.check_cmd_line_flag("B"))      cmd.get_cmd_line_argument("B",      b.elem);
  if (cmd.check_cmd_line_flag("sfB"))    cmd.get_cmd_line_argument("sfB",    b.sf);
  if (cmd.check_cmd_line_flag("blockB")) cmd.get_cmd_line_argument("blockB", b.block);
  if (cmd.check_cmd_line_flag("D"))      cmd.get_cmd_line_argument("D",      dtype_d);

  if (a.empty() && b.empty()) {
    if (!dtype_d.empty())
      std::cerr << "Warning: --D ignored without --A/--B operand spec\n";
    return;
  }
  if (a.empty() && !b.empty()) {
    std::cerr << "Error: --B specified without --A. Specify --A first (B defaults to A).\n";
    std::exit(1);
  }

  // Apply defaults and validate
  a.apply_plain_defaults();
  b.apply_plain_defaults();
  
  if (!a.empty()) a.validate("A");
  
  const bool any_b = cmd.check_cmd_line_flag("B") || cmd.check_cmd_line_flag("sfB") || cmd.check_cmd_line_flag("blockB");
  if (b.empty()) {
    b = a;  // Default B to A
  } else if (any_b) {
    b.validate("B");
  }
  
  validate_combo(a, b, dtype_d);
  configs = build_config_names(a, b, dtype_d);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Compile-time config generation: all valid (SA, SB, D) triples.
///////////////////////////////////////////////////////////////////////////////////////////////////

// Bundles one operand's (ElementData, ElementSF, VecSize).
template <class ElementData_, class ElementSF_, int VS_>
struct Operand {
  using ElementData = ElementData_;
  using ElementSF   = ElementSF_;
  static constexpr int VS = VS_;
};

// Per-operand hardware validity:
// - fp8 only supports ue8m0+VS=32
// - fp4 supports all SF types and VS=16/32
// - bf16/fp16 use identity scaling (ue8m0, VS=16)
template <class EA, class SF, int VS>
constexpr bool is_valid_operand() {
  using e2m1  = cutlass::float_e2m1_t; using e4m3  = cutlass::float_e4m3_t;
  using ue4m3 = cutlass::float_ue4m3_t; using ue5m3 = cutlass::float_ue5m3_t;
  using ue8m0 = cutlass::float_ue8m0_t;
  using bf16_t = sycl::ext::oneapi::bfloat16; using fp16_t = sycl::half;
  if constexpr (std::is_same_v<EA, e4m3>)  return std::is_same_v<SF, ue8m0> && VS == 32;
  if constexpr (std::is_same_v<EA, e2m1>) {
    if constexpr (std::is_same_v<SF, ue4m3>) return VS == 16;
    if constexpr (std::is_same_v<SF, ue5m3>) return VS == 16 || VS == 32;
    if constexpr (std::is_same_v<SF, ue8m0>) return VS == 16 || VS == 32;
  }
  if constexpr (std::is_same_v<EA, bf16_t> || std::is_same_v<EA, fp16_t>)
    return std::is_same_v<SF, ue8m0> && VS == 16;
  return false;
}

// Hardware-valid operand pair rules:
// - MX × MX: SF type and VS must match (e.g., fp4×fp8 both use ue8m0+32)
// - MX × plain: no SF/VS match required
// - plain × plain: invalid (not a block-scaled operation)
template <class SA, class SB>
constexpr bool is_valid_combo() {
  using bf16_t = sycl::ext::oneapi::bfloat16; using fp16_t = sycl::half;
  if constexpr (!is_valid_operand<typename SA::ElementData, typename SA::ElementSF, SA::VS>()) return false;
  if constexpr (!is_valid_operand<typename SB::ElementData, typename SB::ElementSF, SB::VS>()) return false;

  constexpr bool is_plain_a = std::is_same_v<typename SA::ElementData, bf16_t> ||
                               std::is_same_v<typename SA::ElementData, fp16_t>;
  constexpr bool is_plain_b = std::is_same_v<typename SB::ElementData, bf16_t> ||
                               std::is_same_v<typename SB::ElementData, fp16_t>;

  if constexpr (is_plain_a && is_plain_b) return false;  // plain × plain: not block-scaled

  if constexpr (is_plain_a || is_plain_b) {
    return true;  // MX × plain: all VS values supported
  }

  // Both MX: SF type and VS must match.
  if constexpr (!std::is_same_v<typename SA::ElementSF, typename SB::ElementSF>) return false;
  if constexpr (SA::VS != SB::VS) return false;
  return true;
}

// Name helpers.
template <class SA, class SB, class D>
const char* get_config_name() {
  static const std::string s = []{
    using EDA = typename SA::ElementData;
    using EDB = typename SB::ElementData;
    std::string prefix;
    if constexpr (std::is_same_v<EDA, EDB>) {
      prefix = type_name<EDA>();
    } else {
      prefix = std::string(type_name<EDA>()) + "a_" + type_name<EDB>() + "b";
    }
    return prefix
      + "_" + sf_tag<typename SA::ElementSF>() + "k" + std::to_string(SA::VS)
      + "_" + sf_tag<typename SB::ElementSF>() + "k" + std::to_string(SB::VS)
      + "_" + type_name<D>();
  }();
  return s.c_str();
}

// Helper predicates for is_default_config() - improves readability
template <class SA, class SB, class D>
struct ConfigTraits {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  using fp16_t = sycl::half;
  using fp4 = cutlass::float_e2m1_t;
  using fp8 = cutlass::float_e4m3_t;
  using ue4m3 = cutlass::float_ue4m3_t;
  using ue5m3 = cutlass::float_ue5m3_t;
  using ue8m0 = cutlass::float_ue8m0_t;
  
  using EA = typename SA::ElementData;
  using EB = typename SB::ElementData;
  using SFA = typename SA::ElementSF;
  using SFB = typename SB::ElementSF;
  static constexpr int VSA = SA::VS;
  static constexpr int VSB = SB::VS;
  
  // Type checks
  static constexpr bool is_fp4x4   = std::is_same_v<EA,fp4> && std::is_same_v<EB,fp4>;
  static constexpr bool is_fp8x8   = std::is_same_v<EA,fp8> && std::is_same_v<EB,fp8>;
  static constexpr bool is_fp4x8   = std::is_same_v<EA,fp4> && std::is_same_v<EB,fp8>;
  static constexpr bool is_fp8x4   = std::is_same_v<EA,fp8> && std::is_same_v<EB,fp4>;
  static constexpr bool is_fp4xbf16 = std::is_same_v<EA,fp4> && std::is_same_v<EB,bf16_t>;
  static constexpr bool is_bf16xfp4 = std::is_same_v<EA,bf16_t> && std::is_same_v<EB,fp4>;
  static constexpr bool is_fp8xfp16 = std::is_same_v<EA,fp8> && std::is_same_v<EB,fp16_t>;
  static constexpr bool is_fp16xfp8 = std::is_same_v<EA,fp16_t> && std::is_same_v<EB,fp8>;
  
  // SF and VS checks
  static constexpr bool sf_ue4m3 = std::is_same_v<SFA,ue4m3>;
  static constexpr bool sf_ue5m3 = std::is_same_v<SFA,ue5m3>;
  static constexpr bool sf_ue8m0 = std::is_same_v<SFA,ue8m0>;
  static constexpr bool sfB_ue4m3 = std::is_same_v<SFB,ue4m3>;
  static constexpr bool vs16 = (VSA == 16);
  static constexpr bool vs32 = (VSA == 32);
  
  // Output type checks
  static constexpr bool out_any   = true;
  static constexpr bool out_fp32  = std::is_same_v<D,float>;
  static constexpr bool out_fp16  = std::is_same_v<D,fp16_t>;
  static constexpr bool out_bf16  = std::is_same_v<D,bf16_t>;
  static constexpr bool out_fp32_or_fp16 = out_fp32 || out_fp16;
  static constexpr bool out_fp32_or_bf16 = out_fp32 || out_bf16;
};

// Determine if a config should run by default (~18 representative configs)
// Organized into 5 categories, with non-overlapping coverage:
//   1. FP4×FP4 (same MX, all 3 SF types) - 6 configs
//   2. FP8×FP8 (same MX) - 3 configs
//   3. FP4×FP8 (mixed MX, both orderings) - 3 configs
//   4. FP4×plain (both orderings) - 4 configs
//   5. FP8×plain (both orderings) - 2 configs
template <class SA, class SB, class D>
constexpr bool is_default_config() {
  using T = ConfigTraits<SA, SB, D>;
  
  // Category 1: FP4×FP4 (6 configs) - ALL 3 SF types represented
  if constexpr (T::is_fp4x4 && T::sf_ue4m3 && T::vs16) return true;                     // ue4m3×16: fp32, fp16, bf16
  if constexpr (T::is_fp4x4 && T::sf_ue5m3 && T::vs16 && T::out_fp32) return true;      // ue5m3×16: fp32
  if constexpr (T::is_fp4x4 && T::sf_ue5m3 && T::vs32 && T::out_fp32) return true;      // ue5m3×32: fp32 (VS=32 SF padding)
  if constexpr (T::is_fp4x4 && T::sf_ue8m0 && T::vs16 && T::out_fp32) return true;      // ue8m0×16: fp32
  
  // Category 2: FP8×FP8 (3 configs) - all 3 output types (FP8 only supports ue8m0×32)
  if constexpr (T::is_fp8x8) return true;                                                // fp32, fp16, bf16
  
  // Category 3: FP4×FP8 mixed MX (3 configs) - both orderings with output variety
  if constexpr (T::is_fp4x8 && T::sf_ue8m0 && T::vs32 && T::out_fp32) return true;      // FP4×FP8: fp32
  if constexpr (T::is_fp8x4 && T::sf_ue8m0 && T::vs32 && T::out_fp32) return true;      // FP8×FP4: fp32
  if constexpr (T::is_fp8x4 && T::sf_ue8m0 && T::vs32 && T::out_bf16) return true;      // FP8×FP4: bf16
  
  // Category 4: FP4×plain (4 configs) - ue4m3×16 with both orderings and output types
  if constexpr (T::is_fp4xbf16 && T::sf_ue4m3 && T::vs16 && T::out_fp32_or_bf16) return true;  // FP4×BF16: fp32, bf16
  if constexpr (T::is_bf16xfp4 && T::sfB_ue4m3 && T::VSB==16 && T::out_fp32_or_bf16) return true; // BF16×FP4: fp32, bf16
  
  // Category 5: FP8×plain (2 configs) - both orderings
  if constexpr (T::is_fp8xfp16 && T::vs32 && T::out_fp32) return true;                  // FP8×FP16: fp32
  if constexpr (T::is_fp16xfp8 && T::VSB==32 && T::out_fp16) return true;               // FP16×FP8: fp16
  
  return false;
}
// Coverage Summary (18 total configs):
//   Cat1: FP4×FP4 all 3 SF types [ue4m3(3), ue5m3(2), ue8m0(1)] = 6
//   Cat2: FP8×FP8 all outputs = 3
//   Cat3: FP4×FP8 mixed, both orderings = 3  
//   Cat4: FP4×plain, both orderings = 4
//   Cat5: FP8×plain, both orderings = 2

template <class SA, class SB, class D>
struct GeneratedConfig {
  using ElementA          = typename SA::ElementData;
  using ElementB          = typename SB::ElementData;
  using ElementSFA        = typename SA::ElementSF;
  using ElementSFB        = typename SB::ElementSF;
  static constexpr int SFVecSizeA      = SA::VS;
  static constexpr int SFVecSizeB      = SB::VS;
  using ElementD          = D;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  // Hardware: B must be K-major (ColumnMajor) when ElementA != ElementB
  using LayoutB = std::conditional_t<
                    !std::is_same_v<ElementA, ElementB>,
                    cutlass::layout::ColumnMajor,
                    cutlass::layout::RowMajor>;
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 512, 1024, 1};
  using CtaTileShape_MNK  = Shape<_128, _256, _128>;
  using ClusterShape_MNK  = Shape<_1, _1, _1>;
  static constexpr int  PipelineStages  = 4;
  static constexpr bool EnableCooperativeSF = false;
  static constexpr bool run_by_default  = is_default_config<SA, SB, D>();
  static const char*    Name;
};
template <class SA, class SB, class D>
const char* GeneratedConfig<SA,SB,D>::Name = get_config_name<SA,SB,D>();

// Generate all valid config combinations
template <class... Ts> struct TypeList {};
template <bool V, class T>
using Maybe = std::conditional_t<V, std::tuple<T>, std::tuple<>>;

// Hardware output type constraints:
// - MX × plain: D must be fp32 or match the plain type
// - MX × MX: D can be fp32, bf16, or fp16
template <class SA, class SB, class D>
constexpr bool is_valid_d_for_combo() {
  using bf16_t = sycl::ext::oneapi::bfloat16; using fp16_t = sycl::half;
  constexpr bool is_plain_a = std::is_same_v<typename SA::ElementData, bf16_t> ||
                               std::is_same_v<typename SA::ElementData, fp16_t>;
  constexpr bool is_plain_b = std::is_same_v<typename SB::ElementData, bf16_t> ||
                               std::is_same_v<typename SB::ElementData, fp16_t>;
  if constexpr (is_plain_a || is_plain_b) {
    // One operand is plain; D must be fp32 or that plain type.
    using PlainType = std::conditional_t<is_plain_a, typename SA::ElementData,
                                                     typename SB::ElementData>;
    return std::is_same_v<D, float> || std::is_same_v<D, PlainType>;
  }
  return true;  // Both MX: no additional D restriction.
}

template <class SA, class SB>
using Combo = decltype(std::tuple_cat(
  Maybe<is_valid_combo<SA,SB>() && is_valid_d_for_combo<SA,SB,float>(),                       GeneratedConfig<SA,SB,float>>{},
  Maybe<is_valid_combo<SA,SB>() && is_valid_d_for_combo<SA,SB,sycl::half>(),                  GeneratedConfig<SA,SB,sycl::half>>{},
  Maybe<is_valid_combo<SA,SB>() && is_valid_d_for_combo<SA,SB,sycl::ext::oneapi::bfloat16>(), GeneratedConfig<SA,SB,sycl::ext::oneapi::bfloat16>>{}
));
template <class SA, class... SBs>
using ForSA = decltype(std::tuple_cat(Combo<SA, SBs>{}...));
template <class SAList, class SBList> struct BuildConfigs;
template <class... SAs, class... SBs>
struct BuildConfigs<TypeList<SAs...>, TypeList<SBs...>> {
  template <class> struct Unpack;
  template <class... Cs> struct Unpack<std::tuple<Cs...>> { using type = TypeList<Cs...>; };
  using type = typename Unpack<decltype(std::tuple_cat(ForSA<SAs, SBs...>{}...))>::type;
};

// 8 operand types: 5 FP4 variants, 1 FP8, 2 plain types (bf16/fp16).
// is_valid_combo() enforces hardware constraints:
//   - plain × plain: excluded (no block-scaling)
//   - All valid MX/plain and MX/MX combinations allowed
using AllOperands = TypeList<
  Operand<cutlass::float_e2m1_t, cutlass::float_ue4m3_t, 16>,
  Operand<cutlass::float_e2m1_t, cutlass::float_ue5m3_t, 16>,
  Operand<cutlass::float_e2m1_t, cutlass::float_ue5m3_t, 32>,
  Operand<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 16>,
  Operand<cutlass::float_e2m1_t, cutlass::float_ue8m0_t, 32>,
  Operand<cutlass::float_e4m3_t, cutlass::float_ue8m0_t, 32>,
  Operand<sycl::ext::oneapi::bfloat16, cutlass::float_ue8m0_t, 16>,
  Operand<sycl::half,                  cutlass::float_ue8m0_t, 16>
>;

// All hardware-valid (SA, SB) operand pairs × 3 output dtypes.
using AllBlockScaledConfigs = typename BuildConfigs<AllOperands, AllOperands>::type;

// Entry point: compile + run all configs in AllBlockScaledConfigs.
template <class List> struct RunAllConfigs;
template <class... Configs>
struct RunAllConfigs<TypeList<Configs...>> {
  static constexpr size_t count         = sizeof...(Configs);
  static constexpr size_t default_count = (0 + ... + (Configs::run_by_default ? 1 : 0));
  static bool run(const std::vector<std::string>& user_configs, sycl::queue& q) {
    bool pass = (run_if_selected<Configs>(user_configs, q) & ...);
    if (!user_configs.empty()) {
      const std::vector<std::string> known = { Configs::Name... };
      for (const auto& c : user_configs)
        if (std::find(known.begin(), known.end(), c) == known.end())
          std::cerr << "Warning: unknown config '" << c << "' (run --help)\n";
    }
    return pass;
  }
  static void print_names(std::ostream& out) {
    ((out << "  " << Configs::Name << "\n"), ...);
  }
};

// Compiles all valid configs; by default runs ~18 representative tests.
// Use --run-all to test all compiled configs, or --A/--B for specific configs.
using Configs = RunAllConfigs<AllBlockScaledConfigs>;

std::ostream& print_usage(std::ostream& out) {
  out << "Block-Scaled GEMM\n\n"
      << "  --help / --m --n --k --l=<int>\n"
      << "  --run-all              Run all " << Configs::count << " compiled configs (default: " << Configs::default_count << " representative tests)\n"
      << "  --A=fp4|fp8|bf16|fp16  --sfA=ue4m3|ue5m3|ue8m0  --blockA=16|32   (A operand; bf16/fp16 default to ue8m0+16)\n"
      << "  --B=...                --sfB=...                 --blockB=...     (B operand, default=A)\n"
      << "  --D=fp32|fp16|bf16                                      (output dtype, default=all 3)\n\n"
      << "Available configs (use --run-all to test all, or --A/--B for specific configs):\n";
  Configs::print_names(out);
  out << "\nExamples:\n"
      << "  ./xe4_gemm_blockscaled                                      # run " << Configs::default_count << " default representative tests\n"
      << "  ./xe4_gemm_blockscaled --run-all                            # run all " << Configs::count << " compiled configs\n"
      << "  ./xe4_gemm_blockscaled --A=fp4 --sfA=ue4m3 --blockA=16     # symmetric FP4, 3 dtypes\n"
      << "  ./xe4_gemm_blockscaled --A=fp4 --sfA=ue8m0 --blockA=32 --B=fp8 --sfB=ue8m0 --blockB=32 --D=fp32\n"
      << "  ./xe4_gemm_blockscaled --A=fp8 --sfA=ue8m0 --blockA=32 --B=fp4 --sfB=ue8m0 --blockB=32 --D=fp32\n";
  return out;
}

int main(int argc, char** argv) {
  Options::parse(argc, argv);
  if (Options::help) { print_usage(std::cout); return 0; }
  sycl::queue q;
  return Configs::run(Options::configs, q) ? 0 : 1;
}
