/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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

/*! \file
    \brief Host-side regression tests for uniform Dynamic FMHA partitioning.
*/

#include "cutlass_unit_test.h"

#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"

using cutlass::fmha::kernel::fmha_dynamic_num_partitions;

TEST(XE_FMHA_Fwd_DynamicPartitions, RepresentativeShapes) {
  EXPECT_EQ(fmha_dynamic_num_partitions(56, 4, 64, 128), 13);
  EXPECT_EQ(fmha_dynamic_num_partitions(64, 4, 64, 128), 16);
  EXPECT_EQ(fmha_dynamic_num_partitions(20, 4, 64, 128), 5);

  EXPECT_EQ(fmha_dynamic_num_partitions(128, 1, 1024, 32), 32);
  EXPECT_EQ(fmha_dynamic_num_partitions(128, 1, 7, 128), 7);
  EXPECT_EQ(fmha_dynamic_num_partitions(8, 16, 64, 128), 1);
}

TEST(XE_FMHA_Fwd_DynamicPartitions, EmptyWorkUsesOnePartition) {
  EXPECT_EQ(fmha_dynamic_num_partitions(56, 0, 64, 128), 1);
  EXPECT_EQ(fmha_dynamic_num_partitions(56, 4, 0, 128), 1);
}

TEST(XE_FMHA_Fwd_DynamicPartitions, ProducesUniformNonEmptySlices) {
  for (int saturation_cores : {1, 8, 20, 56, 128}) {
    for (int num_batch_heads : {1, 2, 4, 16}) {
      for (int local_k_blocks : {1, 2, 7, 64, 257}) {
        for (int max_num_partitions : {1, 8, 128}) {
          int target_partitions = cute::max(1, saturation_cores / num_batch_heads);
          target_partitions = cute::min(target_partitions, max_num_partitions);
          target_partitions = cute::min(target_partitions, local_k_blocks);
          int expected_blocks_per_partition = cute::ceil_div(local_k_blocks, target_partitions);
          int expected_partitions = cute::ceil_div(local_k_blocks, expected_blocks_per_partition);

          int num_partitions = fmha_dynamic_num_partitions(
              saturation_cores, num_batch_heads, local_k_blocks, max_num_partitions);
          EXPECT_EQ(num_partitions, expected_partitions);

          int blocks_per_partition = cute::ceil_div(local_k_blocks, num_partitions);
          EXPECT_GE(num_partitions, 1);
          EXPECT_LE(num_partitions, max_num_partitions);
          EXPECT_LE(num_partitions, local_k_blocks);
          EXPECT_GE(num_partitions * blocks_per_partition, local_k_blocks);
          EXPECT_LT((num_partitions - 1) * blocks_per_partition, local_k_blocks);
        }
      }
    }
  }
}