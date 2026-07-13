# Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
# Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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


set(CUTLASS_APPLICATIONS_DIR ${CUTLASS_DIR}/applications)

# --- Tuning / profiling support (env-guarded, off by default) -----------------------
# Ablation experiment flags: pass -DFMHA_ABLATION_FLAGS="MACRO_NAME" to cmake to inject
# compile definitions that skip individual softmax stages (see the mainloop header,
# applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp). Inert unless set.
if(FMHA_ABLATION_FLAGS)
  add_compile_definitions(${FMHA_ABLATION_FLAGS})
endif()

set(TEST_NO_PAGED "")
set(TEST_PAGED "--use_paged_kv")

foreach(HEAD_DIM 64 72 96 128 192)
  foreach(INPUT_TYPE bfloat16_t float_e5m2_t float_e4m3_t)
    cutlass_example_add_executable(
      06_xe_fmha_fwd_prefill_${INPUT_TYPE}_hdim${HEAD_DIM}
      06_xe_fmha_fwd.cpp
    )

    cutlass_example_add_executable(
      06_xe_fmha_fwd_decode_${INPUT_TYPE}_hdim${HEAD_DIM}
      06_xe_fmha_fwd.cpp
    )

    if (NOT HEAD_DIM STREQUAL 192)
      # specific test for persistent kernel
      cutlass_example_add_executable(
        06_xe_fmha_fwd_decode_persistent_${INPUT_TYPE}_hdim${HEAD_DIM}
        06_xe_fmha_fwd.cpp
      )
    endif()

    if(INPUT_TYPE STREQUAL "bfloat16_t")
      set(INPUT_MACRO "IS_BFLOAT16")
    elseif(INPUT_TYPE STREQUAL "float_e5m2_t")
      set(INPUT_MACRO "IS_FLOAT_E5M2")
    elseif(INPUT_TYPE STREQUAL "float_e4m3_t")
      set(INPUT_MACRO "IS_FLOAT_E4M3")
    endif()

    if (HEAD_DIM STREQUAL "64" OR HEAD_DIM STREQUAL "192")
      target_compile_definitions(06_xe_fmha_fwd_prefill_${INPUT_TYPE}_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} PREFILL ENABLE_SPLIT_BOUNDARY ENABLE_FUSED_SOFTMAX NC_SCHED_FALLBACK SHOW_DIFF=1 INPUT_TYPE=${INPUT_TYPE} ${INPUT_MACRO})
    else()
      target_compile_definitions(06_xe_fmha_fwd_prefill_${INPUT_TYPE}_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} PREFILL SHOW_DIFF=1 INPUT_TYPE=${INPUT_TYPE} ${INPUT_MACRO})
    endif()
    target_compile_definitions(06_xe_fmha_fwd_decode_${INPUT_TYPE}_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} DECODE SHOW_DIFF=1 INPUT_TYPE=${INPUT_TYPE} ${INPUT_MACRO})
    # GQA q_packed decode: pack the query heads (and speculative tokens) sharing a
    # KV head into the QK tile rows. Only hdim 64/128 have a packed tile-shape config
    # (M=16), and the kernel/scheduler packed path is non-persistent only, so restrict
    # the macro accordingly (never add it to PREFILL or the persistent decode target).
    if(HEAD_DIM STREQUAL "64" OR HEAD_DIM STREQUAL "128")
      target_compile_definitions(06_xe_fmha_fwd_decode_${INPUT_TYPE}_hdim${HEAD_DIM} PRIVATE Q_PACKED_DECODE)
    endif()
    if (NOT HEAD_DIM STREQUAL 192)
      target_compile_definitions(06_xe_fmha_fwd_decode_persistent_${INPUT_TYPE}_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} DECODE PERSISTENT SHOW_DIFF=1 INPUT_TYPE=${INPUT_TYPE} ${INPUT_MACRO})
    endif()
  endforeach()
endforeach()

# int8 Flash Attention prefill (int8 x int8 DPAS, static per-tensor quant of P).
foreach(HEAD_DIM 64 128)
  cutlass_example_add_executable(
    06_xe_fmha_fwd_prefill_int8_t_hdim${HEAD_DIM}
    06_xe_fmha_fwd.cpp
  )
  target_compile_definitions(06_xe_fmha_fwd_prefill_int8_t_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} PREFILL SHOW_DIFF=1 INPUT_TYPE=int8_t IS_INT8)
endforeach()

# int8 prefill variant: larger KV tile (128) for hdim64, exploiting int8 register headroom.
# EXPERIMENT RESULT: regressed -18% (fp32 softmax tile grew -> 256-GRF spill). Kept for
# reproducibility; not used by default. See copilot.md "优化实验" section.
cutlass_example_add_executable(
  06_xe_fmha_fwd_prefill_int8_t_hdim64_kv128
  06_xe_fmha_fwd.cpp
)
target_compile_definitions(06_xe_fmha_fwd_prefill_int8_t_hdim64_kv128 PRIVATE HEAD_DIM=64 PREFILL SHOW_DIFF=1 INPUT_TYPE=int8_t IS_INT8 INT8_KV128)

# int8 prefill EXPERIMENT: mimic fp8 dataflow by removing the int8-specific int32->float
# conversions (persistent int32 PV accumulator, no per-block convert/rescale, bit_cast
# scores). NUMERICALLY WRONG -- isolates whether those conversions are the bottleneck.
foreach(HEAD_DIM 64 128)
  cutlass_example_add_executable(
    06_xe_fmha_fwd_prefill_int8_t_hdim${HEAD_DIM}_fp8mimic
    06_xe_fmha_fwd.cpp
  )
  target_compile_definitions(06_xe_fmha_fwd_prefill_int8_t_hdim${HEAD_DIM}_fp8mimic PRIVATE HEAD_DIM=${HEAD_DIM} PREFILL SHOW_DIFF=1 INPUT_TYPE=int8_t IS_INT8 INT8_FP8_MIMIC)
endforeach()

# --- Tile-shape sweep targets (env-guarded, OFF by default) --------------------------
# Prefill sweep: one executable per (HEAD_DIM x TILE_Q x TILE_SEQ_LEN x TILE_HDIM x
# TILE_Q_SG) tile config.  The TILE_* macros feed the prefill override block in
# 06_xe_fmha_fwd.cpp.
#
# Decode sweep: one executable per (HEAD_DIM x TILE_KV x TILE_KV_SG x PERSISTENT).
# TILE_KV  = TileShapeQK[1] = KV_TILE_SIZE (the KV block size; tile_q, tile_hdim are
#             fixed per HEAD_DIM in the decode block).
# TILE_KV_SG = NUM_SG (ReduceK split degree; the number of sub-groups along the KV
#             dimension within one work-group).
# PERSISTENT selects the cross-WG dynamic-split (XeFMHAFwdDynamicSplitKernel) vs the
#             standard per-WG non-split kernel.
#
# Both sweeps are OFF unless the build is configured with -DFMHA_TUNE=ON.
# run_tuning_candidates.py builds only the specific candidate targets the projection
# shortlists, not the whole cartesian product.
option(FMHA_TUNE "Generate FMHA tile-shape sweep targets for tuning" OFF)
if(FMHA_TUNE)
  # ---- Prefill sweep -----------------------------------------------------------------
  if(NOT FMHA_TUNE_HEAD_DIMS)
    set(FMHA_TUNE_HEAD_DIMS 64 128)
  endif()
  if(NOT FMHA_TUNE_TILE_Q)
    set(FMHA_TUNE_TILE_Q 128 256 512)
  endif()
  if(NOT FMHA_TUNE_TILE_SEQ_LEN)
    set(FMHA_TUNE_TILE_SEQ_LEN 32 64 128 256)
  endif()
  if(NOT FMHA_TUNE_TILE_HDIM)
    set(FMHA_TUNE_TILE_HDIM 32 64 128)
  endif()
  if(NOT FMHA_TUNE_TILE_Q_SG)
    set(FMHA_TUNE_TILE_Q_SG 8 16 32)
  endif()
  foreach(HEAD_DIM ${FMHA_TUNE_HEAD_DIMS})
    foreach(TILE_Q ${FMHA_TUNE_TILE_Q})
      foreach(TILE_SEQ_LEN ${FMHA_TUNE_TILE_SEQ_LEN})
        foreach(TILE_HDIM ${FMHA_TUNE_TILE_HDIM})
          foreach(TILE_Q_SG ${FMHA_TUNE_TILE_Q_SG})
            set(TILE_SUFFIX "q${TILE_Q}_seq${TILE_SEQ_LEN}_hdim${TILE_HDIM}_qsg${TILE_Q_SG}")
            # Base prefill target (no codegen variant flags) -- kept for reference
            set(TUNE_TARGET "06_xe_fmha_fwd_prefill_bfloat16_t_hdim${HEAD_DIM}_${TILE_SUFFIX}")
            cutlass_example_add_executable(${TUNE_TARGET} 06_xe_fmha_fwd.cpp)
            target_compile_definitions(${TUNE_TARGET} PRIVATE
              HEAD_DIM=${HEAD_DIM} PREFILL SHOW_DIFF=1 INPUT_TYPE=bfloat16_t IS_BFLOAT16
              TILE_Q=${TILE_Q} TILE_SEQ_LEN=${TILE_SEQ_LEN}
              TILE_HDIM=${TILE_HDIM} TILE_Q_SG=${TILE_Q_SG})
            # Combined ideal-implementation target (_sb_fs): ENABLE_SPLIT_BOUNDARY +
            # ENABLE_FUSED_SOFTMAX. This is the binary that run_tuning_candidates.py
            # builds: the projection assumes ideal implementation (both flags always on),
            # so the benchmarked binary must match.
            set(SB_FS_TARGET "${TUNE_TARGET}_sb_fs")
            cutlass_example_add_executable(${SB_FS_TARGET} 06_xe_fmha_fwd.cpp)
            target_compile_definitions(${SB_FS_TARGET} PRIVATE
              HEAD_DIM=${HEAD_DIM} PREFILL ENABLE_SPLIT_BOUNDARY ENABLE_FUSED_SOFTMAX
              SHOW_DIFF=1 INPUT_TYPE=bfloat16_t IS_BFLOAT16
              TILE_Q=${TILE_Q} TILE_SEQ_LEN=${TILE_SEQ_LEN}
              TILE_HDIM=${TILE_HDIM} TILE_Q_SG=${TILE_Q_SG})
            # Standalone split_boundary variant (_sb): for ablation (split_boundary only)
            if(FMHA_TUNE_SPLIT_BOUNDARY)
              set(SB_TARGET "${TUNE_TARGET}_sb")
              cutlass_example_add_executable(${SB_TARGET} 06_xe_fmha_fwd.cpp)
              target_compile_definitions(${SB_TARGET} PRIVATE
                HEAD_DIM=${HEAD_DIM} PREFILL ENABLE_SPLIT_BOUNDARY SHOW_DIFF=1
                INPUT_TYPE=bfloat16_t IS_BFLOAT16
                TILE_Q=${TILE_Q} TILE_SEQ_LEN=${TILE_SEQ_LEN}
                TILE_HDIM=${TILE_HDIM} TILE_Q_SG=${TILE_Q_SG})
            endif()
            # Standalone fused_softmax variant (_fs): for ablation (fused_softmax only)
            if(FMHA_TUNE_FUSED_SOFTMAX)
              set(FS_TARGET "${TUNE_TARGET}_fs")
              cutlass_example_add_executable(${FS_TARGET} 06_xe_fmha_fwd.cpp)
              target_compile_definitions(${FS_TARGET} PRIVATE
                HEAD_DIM=${HEAD_DIM} PREFILL ENABLE_FUSED_SOFTMAX SHOW_DIFF=1
                INPUT_TYPE=bfloat16_t IS_BFLOAT16
                TILE_Q=${TILE_Q} TILE_SEQ_LEN=${TILE_SEQ_LEN}
                TILE_HDIM=${TILE_HDIM} TILE_Q_SG=${TILE_Q_SG})
            endif()
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()

  # ---- Decode sweep ------------------------------------------------------------------
  if(NOT FMHA_TUNE_DECODE_HEAD_DIMS)
    set(FMHA_TUNE_DECODE_HEAD_DIMS 64 128)
  endif()
  if(NOT FMHA_TUNE_TILE_KV)
    set(FMHA_TUNE_TILE_KV 128 256 512)
  endif()
  if(NOT FMHA_TUNE_TILE_KV_SG)
    set(FMHA_TUNE_TILE_KV_SG 4 8 16)
  endif()
  foreach(HEAD_DIM ${FMHA_TUNE_DECODE_HEAD_DIMS})
    foreach(TILE_KV ${FMHA_TUNE_TILE_KV})
      foreach(TILE_KV_SG ${FMHA_TUNE_TILE_KV_SG})
        # Non-persistent decode variant
        set(DEC_SUFFIX "kv${TILE_KV}_kvsg${TILE_KV_SG}")
        set(DEC_TARGET "06_xe_fmha_fwd_decode_bfloat16_t_hdim${HEAD_DIM}_${DEC_SUFFIX}")
        cutlass_example_add_executable(${DEC_TARGET} 06_xe_fmha_fwd.cpp)
        target_compile_definitions(${DEC_TARGET} PRIVATE
          HEAD_DIM=${HEAD_DIM} DECODE SHOW_DIFF=1 INPUT_TYPE=bfloat16_t IS_BFLOAT16
          TILE_KV=${TILE_KV} TILE_KV_SG=${TILE_KV_SG})
        # Q_PACKED_DECODE is added for hdim64/128 (GQA packing)
        if(HEAD_DIM STREQUAL "64" OR HEAD_DIM STREQUAL "128")
          target_compile_definitions(${DEC_TARGET} PRIVATE Q_PACKED_DECODE)
        endif()

        # Persistent decode variant (cross-WG dynamic-split kernel)
        set(PERST_TARGET "06_xe_fmha_fwd_decode_persistent_bfloat16_t_hdim${HEAD_DIM}_${DEC_SUFFIX}")
        cutlass_example_add_executable(${PERST_TARGET} 06_xe_fmha_fwd.cpp)
        target_compile_definitions(${PERST_TARGET} PRIVATE
          HEAD_DIM=${HEAD_DIM} DECODE PERSISTENT SHOW_DIFF=1 INPUT_TYPE=bfloat16_t IS_BFLOAT16
          TILE_KV=${TILE_KV} TILE_KV_SG=${TILE_KV_SG})
      endforeach()
    endforeach()
  endforeach()
endif()
