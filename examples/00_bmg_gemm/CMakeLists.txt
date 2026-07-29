# Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set(TEST_BATCHES --l=2)
set(TEST_LARGE "--l=513 --m=8 --n=16384 --k=512") # B matrix capacity > uint32_max
set(TEST_SMALL_SHAPE --m=4 --n=8 --k=8 --l=2)


# IGC options that match xpu-perf's build.sh BMG_GEMM_LINK_FLAGS.
# VectorAliasBBThreshold=10000: avoids a miscompile on large alias analysis graphs.
# VISAOptions=-perfmodel: enables performance-model-guided ISA scheduling; this is
#   AOT-only (no runtime env var equivalent) and accounts for the gap between
#   standalone JIT binaries and xpu-perf .so performance.
# cl-intel-256-GRF-per-thread: AOT large-GRF; redundant with the JIT-time
#   IGC_TotalGRFNum=256 env var but required for AOT correctness.
# NOTE: each -Xs flag must be a separate list entry paired with its argument; CMake
#   passes them as individual tokens to the linker driver (icpx).
set(BMG_GEMM_LINK_FLAGS
    -Xs "-options \"-igc_opts 'VectorAliasBBThreshold=10000,VISAOptions=-perfmodel' -cl-intel-256-GRF-per-thread\""
)

cutlass_example_add_executable(
  00_bmg_gemm
  00_bmg_gemm.cpp
  TEST_COMMAND_OPTIONS
  TEST_BATCHES
  TEST_LARGE
  TEST_SMALL_SHAPE
)
target_link_options(00_bmg_gemm PRIVATE ${BMG_GEMM_LINK_FLAGS})

set(TEST_SMALL_SHAPE_PADDABLE --m=1 --n=1 --k=2 --l=2)
cutlass_example_add_executable(
  00_bmg_gemm_padded
  00_bmg_gemm_padded.cpp
  TEST_COMMAND_OPTIONS
  TEST_BATCHES
  TEST_SMALL_SHAPE_PADDABLE
)
target_link_options(00_bmg_gemm_padded PRIVATE ${BMG_GEMM_LINK_FLAGS})

cutlass_example_add_executable(
  00_bmg_gemm_with_sycl_queue
  00_bmg_gemm_with_sycl_queue.cpp
  TEST_COMMAND_OPTIONS
  TEST_BATCHES
)
target_link_options(00_bmg_gemm_with_sycl_queue PRIVATE ${BMG_GEMM_LINK_FLAGS})

# --- int8 (W8A8) reference-default anchor -------------------------------------------
# Plain int8 quantized matmul (s8 x s8 -> s32, per-tensor dequant to bf16), built from
# the same 00_bmg_gemm_tune.cpp source with no TILE_* overrides, i.e. the default tile.
# Always built (not gated behind GEMM_TUNE) so it is available as the cross-validation
# anchor for int8 tuning runs, mirroring how 00_bmg_gemm anchors bf16
# (optimization_skills/11_projection_guided_autotuning.md section 1.2).
#
# CAVEAT, stated honestly: unlike bf16 -- whose 256x256x32 / 8x4 / stages=2 default IS a
# reference-validated configuration -- there is no pre-existing plain-W8A8 GEMM in this
# tree, so int8's "default" tile is simply the bf16 tile reused. It is a baseline to beat,
# NOT a validated optimum; for int8 the measured shortlist is the only source of truth.
# (The only int8 tiles in-tree are benchmarks/gemm/legacy/benchmarks_sycl.hpp's
# 32x128x32 / 1x4, but those belong to the MIXED-PRECISION kernel with u4/s8 weights and
# transposed B loads, so their tile choice does not transfer to plain W8A8.)
cutlass_example_add_executable(
  00_bmg_gemm_int8
  00_bmg_gemm_tune.cpp
)
target_compile_definitions(00_bmg_gemm_int8 PRIVATE IS_INT8)
target_link_options(00_bmg_gemm_int8 PRIVATE ${BMG_GEMM_LINK_FLAGS})

# --- GEMM tile-shape sweep targets (env-guarded, OFF by default) ----------------------
# One executable per (DTYPE x TILE_M x TILE_N x TILE_K x TILE_SG_M x TILE_SG_N x STAGES)
# tile config, feeding the macro override block in 00_bmg_gemm_tune.cpp. These macros map
# 1:1 onto the projection model's GEMM search axes (OpPerf/gemm.py: top_full_configs).
#
# OFF unless configured with -DGEMM_TUNE=ON. run_tuning_candidates.py does that configure
# for you and builds ONLY the specific candidate targets the projection shortlisted --
# never the full cartesian product, which is the entire point of using the projection as
# a filter. Same design as the FMHA_TUNE sweep in
# examples/06_bmg_flash_attention/CMakeLists.txt.
#
# NOTE: the defaults below are intentionally narrow. The cartesian product grows fast and
# every entry is a separate heavy compile, so the runner always passes explicit -D lists
# scoped to the union of values its candidates actually use.
option(GEMM_TUNE "Generate GEMM tile-shape sweep targets for tuning" OFF)
if(GEMM_TUNE)
  if(NOT GEMM_TUNE_DTYPES)
    set(GEMM_TUNE_DTYPES bf16 int8)
  endif()
  if(NOT GEMM_TUNE_TILE_M)
    set(GEMM_TUNE_TILE_M 128 256)
  endif()
  if(NOT GEMM_TUNE_TILE_N)
    set(GEMM_TUNE_TILE_N 128 256)
  endif()
  if(NOT GEMM_TUNE_TILE_K)
    set(GEMM_TUNE_TILE_K 32)
  endif()
  if(NOT GEMM_TUNE_TILE_SG_M)
    set(GEMM_TUNE_TILE_SG_M 4 8)
  endif()
  if(NOT GEMM_TUNE_TILE_SG_N)
    set(GEMM_TUNE_TILE_SG_N 4 8)
  endif()
  if(NOT GEMM_TUNE_STAGES)
    set(GEMM_TUNE_STAGES 2)
  endif()

  foreach(GEMM_DTYPE ${GEMM_TUNE_DTYPES})
    if(GEMM_DTYPE STREQUAL "bf16")
      set(GEMM_DTYPE_MACRO "IS_BFLOAT16")
    elseif(GEMM_DTYPE STREQUAL "int8")
      set(GEMM_DTYPE_MACRO "IS_INT8")
    else()
      message(FATAL_ERROR "GEMM_TUNE_DTYPES: unsupported dtype '${GEMM_DTYPE}' (expected bf16 or int8)")
    endif()
    foreach(T_M ${GEMM_TUNE_TILE_M})
      foreach(T_N ${GEMM_TUNE_TILE_N})
        foreach(T_K ${GEMM_TUNE_TILE_K})
          foreach(T_SGM ${GEMM_TUNE_TILE_SG_M})
            foreach(T_SGN ${GEMM_TUNE_TILE_SG_N})
              foreach(T_ST ${GEMM_TUNE_STAGES})
                # Skip tiles TiledMMAHelper cannot instantiate: every sub-group must own a
                # whole number of DPAS atoms (8 rows x 16 cols). The candidate exporter
                # applies the same rule (OpPerf/export_tuning_candidates.py:
                # gemm_candidate), so this guard should never fire for a projection-driven
                # build -- it is here so a hand-rolled -DGEMM_TUNE_* sweep degrades into
                # "target absent" instead of a confusing template error.
                math(EXPR M_PER_SG "${T_M} % (${T_SGM} * 8)")
                math(EXPR N_PER_SG "${T_N} % (${T_SGN} * 16)")
                if(NOT M_PER_SG EQUAL 0 OR NOT N_PER_SG EQUAL 0)
                  continue()
                endif()
                set(GEMM_TUNE_TARGET
                    "00_bmg_gemm_tune_${GEMM_DTYPE}_m${T_M}_n${T_N}_k${T_K}_sgm${T_SGM}_sgn${T_SGN}_st${T_ST}")
                cutlass_example_add_executable(${GEMM_TUNE_TARGET} 00_bmg_gemm_tune.cpp)
                target_compile_definitions(${GEMM_TUNE_TARGET} PRIVATE
                  ${GEMM_DTYPE_MACRO}
                  TILE_M=${T_M} TILE_N=${T_N} TILE_K=${T_K}
                  TILE_SG_M=${T_SGM} TILE_SG_N=${T_SGN}
                  PIPELINE_STAGES=${T_ST})
                target_link_options(${GEMM_TUNE_TARGET} PRIVATE ${BMG_GEMM_LINK_FLAGS})
              endforeach()
            endforeach()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endif()
