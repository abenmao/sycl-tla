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

set(TEST_GROUPS_2 --groups=2)
set(TEST_GROUPS_4 --groups=4)

cutlass_example_add_executable(
  04_bmg_grouped_gemm
  04_bmg_grouped_gemm.cpp
  TEST_COMMAND_OPTIONS
  TEST_GROUPS_2
  TEST_GROUPS_4
)
if(NOT DPCPP_SYCL_TARGET STREQUAL "spir64")
  # Large-GRF + perfmodel VISA scheduling, matching the GEMM_TUNE sweep below and
  # xpu-perf's BMG_GEMM_LINK_FLAGS. This target is the reference ANCHOR for grouped-GEMM
  # tuning, so it MUST be built with the same flags as the candidates -- otherwise the
  # anchor is ~8x under-measured and every candidate appears to "beat" it.
  # (optimization_skills/11_projection_guided_autotuning.md section 11.1)
  target_link_options( 04_bmg_grouped_gemm PRIVATE -Xs "-options \"-igc_opts 'VectorAliasBBThreshold=10000,VISAOptions=-perfmodel' -cl-intel-256-GRF-per-thread\"" )
endif()

# --- Grouped-GEMM tile-shape sweep targets (env-guarded, OFF by default) --------------
# One executable per (TILE_M x TILE_N x TILE_K x TILE_SG_M x TILE_SG_N x STAGES) config,
# feeding the macro override block in 04_bmg_grouped_gemm_tune.cpp. These map 1:1 onto
# the projection model's grouped-GEMM search axes (OpPerf/group_gemm.py:
# top_group_configs).
#
# OFF unless configured with -DGROUP_GEMM_TUNE=ON. run_tuning_candidates.py does that
# configure for you and builds ONLY the specific candidate targets the projection
# shortlisted -- never the full cartesian product. Same design as FMHA_TUNE and
# GEMM_TUNE.
#
# NOTE the link options below. Unlike FMHA (whose kernel declares grf_size<256> in
# source), GEMM-family kernels get large-GRF ONLY from this compile/link flag, and
# VISAOptions=-perfmodel has NO runtime environment-variable equivalent at all -- it is
# AOT-only. Omitting them costs ~8-9x on this hardware and the result still looks
# plausible, so it is silent. See optimization_skills/11_projection_guided_autotuning.md
# section 11.1. All three must ride on ONE -Xs with ONE argument string: CMake pairs
# -Xs with only the next token, so splitting them into several list entries silently
# drops all but the first.
option(GROUP_GEMM_TUNE "Generate grouped-GEMM tile-shape sweep targets for tuning" OFF)
if(GROUP_GEMM_TUNE)
  set(BMG_GROUP_GEMM_LINK_FLAGS
      -Xs "-options \"-igc_opts 'VectorAliasBBThreshold=10000,VISAOptions=-perfmodel' -cl-intel-256-GRF-per-thread\""
  )

  if(NOT GROUP_GEMM_TUNE_TILE_M)
    set(GROUP_GEMM_TUNE_TILE_M 128 256)
  endif()
  if(NOT GROUP_GEMM_TUNE_TILE_N)
    set(GROUP_GEMM_TUNE_TILE_N 128 256)
  endif()
  if(NOT GROUP_GEMM_TUNE_TILE_K)
    set(GROUP_GEMM_TUNE_TILE_K 32)
  endif()
  if(NOT GROUP_GEMM_TUNE_TILE_SG_M)
    set(GROUP_GEMM_TUNE_TILE_SG_M 4 8)
  endif()
  if(NOT GROUP_GEMM_TUNE_TILE_SG_N)
    set(GROUP_GEMM_TUNE_TILE_SG_N 4 8)
  endif()
  if(NOT GROUP_GEMM_TUNE_STAGES)
    set(GROUP_GEMM_TUNE_STAGES 2)
  endif()

  foreach(T_M ${GROUP_GEMM_TUNE_TILE_M})
    foreach(T_N ${GROUP_GEMM_TUNE_TILE_N})
      foreach(T_K ${GROUP_GEMM_TUNE_TILE_K})
        foreach(T_SGM ${GROUP_GEMM_TUNE_TILE_SG_M})
          foreach(T_SGN ${GROUP_GEMM_TUNE_TILE_SG_N})
            foreach(T_ST ${GROUP_GEMM_TUNE_STAGES})
              # Skip tiles TiledMMAHelper cannot instantiate: each sub-group must own a
              # whole number of 8x16 DPAS atoms. The exporter applies the same rule, so
              # this guard only fires for hand-rolled -DGROUP_GEMM_TUNE_* sweeps.
              math(EXPR M_PER_SG "${T_M} % (${T_SGM} * 8)")
              math(EXPR N_PER_SG "${T_N} % (${T_SGN} * 16)")
              if(NOT M_PER_SG EQUAL 0 OR NOT N_PER_SG EQUAL 0)
                continue()
              endif()
              set(GG_TARGET
                  "04_bmg_grouped_gemm_tune_m${T_M}_n${T_N}_k${T_K}_sgm${T_SGM}_sgn${T_SGN}_st${T_ST}")
              cutlass_example_add_executable(${GG_TARGET} 04_bmg_grouped_gemm_tune.cpp)
              target_compile_definitions(${GG_TARGET} PRIVATE
                TILE_M=${T_M} TILE_N=${T_N} TILE_K=${T_K}
                TILE_SG_M=${T_SGM} TILE_SG_N=${T_SGN}
                PIPELINE_STAGES=${T_ST})
              target_link_options(${GG_TARGET} PRIVATE ${BMG_GROUP_GEMM_LINK_FLAGS})
            endforeach()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endif()
