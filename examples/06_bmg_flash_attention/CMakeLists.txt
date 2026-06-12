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

set(TEST_NO_PAGED "")
set(TEST_PAGED "--use_paged_kv")

foreach(HEAD_DIM 64 96 128 192)
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

    if (HEAD_DIM STREQUAL 64)
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
