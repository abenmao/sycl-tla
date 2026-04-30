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

/**
 * gen_xe4_tmm_layouts.cpp
 *
 * C++ replacement for gen_xe4_tmm_layouts.py.
 *
 * Generates per-MMA-atom TikZ/LaTeX documentation images for the XE4 TMM
 * atom directly from the real CuTe layout types defined in:
 *   include/cute/atom/mma_traits_xe4_tmm.hpp
 *
 * The generator uses cute::print_latex(TiledMMA, color) which internally
 * calls cute::detail::print_latex_mma with the atom's ALayout, BLayout,
 * and CLayout — these are the canonical types from mma_traits_xe4_tmm.hpp,
 * not a re-implementation.
 *
 * Build and run (standalone):
 *   source /opt/intel/oneapi/setvars.sh    # Intel oneAPI required for SYCL types
 *   cmake -S media/docs/cpp/cute -B build/xe4_tmm_doc \
 *         -DCMAKE_CXX_COMPILER=icpx
 *   cmake --build build/xe4_tmm_doc
 *   # .tex files land in build/xe4_tmm_doc/tex/
 *
 * Notes on output differences vs gen_xe4_tmm_layouts.py:
 *  - CuTe's print_latex_mma emits leading '%% LayoutC/A/B: ...' comment lines.
 *  - CuTe uses '\documentclass[convert]{standalone}' (no density= option).
 *  - CuTe omits 'font=\tiny' in the tikzpicture node style.
 *  These cosmetic preamble differences do not affect the rendered layout.
 *
 * The fragment positions (which thread/value owns which matrix element) are
 * always in sync with the hardware spec encoded in mma_traits_xe4_tmm.hpp.
 */

#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_xe4_tmm.hpp>
#include <cute/util/print_latex.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>   // dup, dup2, close, STDOUT_FILENO

namespace {

// ---------------------------------------------------------------------------
// Kelly high-contrast color palette — matches KELLY_COLORS in the Python script
// and gives high-contrast per-thread coloring of TikZ nodes.
// The default CuTe TikzColor_TV uses a different pastel palette; this struct
// restores the Kelly colors used by gen_xe4_tmm_layouts.py.
// ---------------------------------------------------------------------------
struct KellyColor_TV {
  CUTE_HOST_DEVICE char const*
  operator()(int tid, int /*vid*/) const {
    static char const* colors[] = {
      "{rgb,255:red,255;green,179;blue,0}",
      "{rgb,255:red,128;green,62;blue,117}",
      "{rgb,255:red,255;green,104;blue,0}",
      "{rgb,255:red,166;green,189;blue,215}",
      "{rgb,255:red,193;green,0;blue,32}",
      "{rgb,255:red,206;green,162;blue,98}",
      "{rgb,255:red,129;green,112;blue,102}",
      "{rgb,255:red,0;green,125;blue,52}",
      "{rgb,255:red,246;green,118;blue,142}",
      "{rgb,255:red,0;green,83;blue,138}",
      "{rgb,255:red,255;green,122;blue,92}",
      "{rgb,255:red,83;green,55;blue,122}",
      "{rgb,255:red,255;green,142;blue,0}",
      "{rgb,255:red,179;green,40;blue,81}",
      "{rgb,255:red,244;green,200;blue,0}",
      "{rgb,255:red,127;green,24;blue,13}",
      "{rgb,255:red,147;green,170;blue,0}",
      "{rgb,255:red,89;green,51;blue,21}",
      "{rgb,255:red,241;green,58;blue,19}",
      "{rgb,255:red,35;green,44;blue,22}",
      "{rgb,255:red,0;green,161;blue,194}",
      "{rgb,255:red,255;green,200;blue,124}",
      "{rgb,255:red,167;green,79;blue,135}",
      "{rgb,255:red,212;green,176;blue,55}",
      "{rgb,255:red,77;green,166;blue,255}",
      "{rgb,255:red,200;green,128;blue,200}",
      "{rgb,255:red,255;green,80;blue,80}",
      "{rgb,255:red,80;green,200;blue,120}",
      "{rgb,255:red,180;green,140;blue,80}",
      "{rgb,255:red,100;green,180;blue,160}",
      "{rgb,255:red,140;green,90;blue,40}",
      "{rgb,255:red,200;green,200;blue,140}",
    };
    return colors[tid % 32];
  }
};

// ---------------------------------------------------------------------------
// Redirect stdout to a file while calling f(), then restore.
// cute::print_latex_mma writes via printf, so file-descriptor redirection
// is the only way to capture it without modifying the CuTe internals.
// ---------------------------------------------------------------------------
template <class Fn>
void write_stdout_to_file(const std::string& path, Fn&& f) {
  // Save current stdout file descriptor
  int saved_fd = dup(STDOUT_FILENO);
  if (saved_fd == -1) { perror("dup"); std::exit(1); }

  // Open the target file and redirect fd 1 to it
  FILE* out = fopen(path.c_str(), "w");
  if (!out) { perror("fopen"); std::exit(1); }
  dup2(fileno(out), STDOUT_FILENO);
  fclose(out);

  // Generate content — printf/puts inside here land in the file
  f();
  fflush(stdout);

  // Restore fd 1 to the original destination
  dup2(saved_fd, STDOUT_FILENO);
  close(saved_fd);
  // Clear any error/EOF state left on the stdout FILE* by the fd swap
  clearerr(stdout);
}

// ---------------------------------------------------------------------------
// Generate one .tex file for MMA_OP.
// Mirrors gen_xe4_tmm_layouts.py's per-config emit loop.
// ---------------------------------------------------------------------------
template <class MMA_OP>
void gen_tex(const std::string& out_dir, const std::string& label,
             int M, int N, int K, const std::string& dtype) {
  namespace fs = std::filesystem;
  fs::create_directories(out_dir);

  std::string path = out_dir + "/" + label + ".tex";

  write_stdout_to_file(path, [&]() {
    // make_tiled_mma wraps the single atom in a 1-atom TiledMMA.
    // print_latex(TiledMMA, color) calls detail::print_latex_mma with the
    // atom's ALayout, BLayout, CLayout from mma_traits_xe4_tmm.hpp.
    cute::print_latex(
      cute::make_tiled_mma(cute::MMA_Atom<MMA_OP>{}),
      KellyColor_TV{});
  });

  // Print status to stdout (matches Python script's stdout format)
  printf("%Wrote %s  (M=%d, N=%d, K=%d, dtype=%s)\n",
         path.c_str(), M, N, K, dtype.c_str());
}

} // anonymous namespace

int main(int argc, char* argv[]) {
  // Output directory from argv[1], or "." if not supplied (mirrors Python script)
  std::string out_dir = (argc > 1) ? argv[1] : ".";

  // Config: fp16, M=32, N=1, K=16  (label: XE4_TMM.f16f16_M32N1K16)
  gen_tex<cute::XE4_TMM<fp16, fp16, fp16, float, 1>>(
    out_dir, "XE4_TMM.f16f16_M32N1K16", 32, 1, 16, "fp16");

  // Config: fp16, M=32, N=8, K=16  (label: XE4_TMM.f16f16_M32N8K16)
  gen_tex<cute::XE4_TMM<fp16, fp16, fp16, float, 8>>(
    out_dir, "XE4_TMM.f16f16_M32N8K16", 32, 8, 16, "fp16");

  return 0;
}
