/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation, All rights reserved.
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

#include <string>

/*
 * Structure to hold parameters for GEMM Cluster performance tests
 * Aligned with GEMM_TEST_CONFIG from gemm_cluster.cpp
 */
struct GemmClusterTestParams
{
    std::string test_level; // Test execution level- "L2" for CI tests, "L3" for Nightly tests
    std::string state;      // Test state - "enabled" or "disabled"

    // Problem shape
    int M, N, K, L;

    // Element types (as strings for runtime configuration)
    std::string dtype_a;   // ElementA (fp16, f32, bf8, etc.)
    std::string dtype_b;   // ElementB
    std::string dtype_c;   // ElementC (can be "void")
    std::string dtype_d;   // ElementD
    std::string dtype_acc; // ElementAccumulator

    // Layout types
    std::string layout_a; // LayoutA (RowMajor, ColumnMajor)
    std::string layout_b; // LayoutB
    std::string layout_c; // LayoutC

    // CTA (Cooperative Thread Array) configuration
    int cta_tile_m, cta_tile_n, cta_tile_k; // CtaTileShape_MNK
    int cta_num_m, cta_num_n;               // CtaNum_MN
    int cluster_m, cluster_n, cluster_k;    // ClusterShape_MNK

    // Pipeline stages
    int stages_a; // StagesA

    // Kernel configuration
    bool is_persistent;      // is_persistent mode
    std::string activation;  // activation_type (SiLu, None)
    std::string operation_c; // operationC_type (Mul, Add, BiasAdd, None)
};

// Array of GEMM test parameters - aligned with configs from gemm.cpp
static GemmClusterTestParams gemm_test_params[] = {
    {/* test_level */ "L2",
     /* state */ "enabled",
     /* M, N, K, L */ 2048, 2048, 2048, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 1, 1,
     /* cluster_m, cluster_n, cluster_k */ 2, 2, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"}, // 0

    {/* test_level */ "L2",
     /* state */ "disabled",
     /* M, N, K, L */ 2048, 2048, 2048, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "ColumnMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 1, 1,
     /* cluster_m, cluster_n, cluster_k */ 2, 2, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"}, // 1

    {/* test_level */ "L2",
     /* state */ "enabled",
     /* M, N, K, L */ 2048, 2048, 2048, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "ColumnMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 1, 1,
     /* cluster_m, cluster_n, cluster_k */ 2, 2, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"}, // 2

    {/* test_level */ "L2",
     /* state */ "disabled",
     /* M, N, K, L */ 2048, 2112, 4163, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 1, 1,
     /* cluster_m, cluster_n, cluster_k */ 2, 2, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"}, // 3

    {/* test_level */ "L2",
     /* state */ "disabled",
     /* M, N, K, L */ 2048, 2112, 4163, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "ColumnMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 1, 1,
     /* cluster_m, cluster_n, cluster_k */ 2, 2, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"}, // 4

    {/* test_level */ "L2",
     /* state */ "disabled",
     /* M, N, K, L */ 2048, 2112, 4163, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "ColumnMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 1, 1,
     /* cluster_m, cluster_n, cluster_k */ 2, 2, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"}, // 5

    {/* test_level */ "L2",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 1024, 1024, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"}, // 6

    {/* test_level */ "L2",
     /* state */ "enabled",
     /* M, N, K, L */ 2048, 2048, 2048, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 1, 1,
     /* cluster_m, cluster_n, cluster_k */ 2, 2, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"}, // 7

};
