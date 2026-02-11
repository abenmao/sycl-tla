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
 * Structure to hold parameters for GEMM performance tests
 * Aligned with GEMM_TEST_CONFIG from gemm.cpp
 */
struct GemmTestParams {
    std::string test_level;  // Test execution level- "L2" for CI tests, "L3" for Nightly tests
    std::string state;      // Test state - "enabled" or "disabled"

    // Problem shape
    int M, N, K, L;

    // Element types (as strings for runtime configuration)
    std::string dtype_a;    // ElementA (fp16, f32, bf8, etc.)
    std::string dtype_b;    // ElementB
    std::string dtype_c;    // ElementC (can be "void")
    std::string dtype_d;    // ElementD
    std::string dtype_acc;  // ElementAccumulator

    // Layout types
    std::string layout_a;  // LayoutA (RowMajor, ColumnMajor)
    std::string layout_b;  // LayoutB
    std::string layout_c;  // LayoutC

    // CTA (Cooperative Thread Array) configuration
    int cta_tile_m, cta_tile_n, cta_tile_k;  // CtaTileShape_MNK
    int cta_num_m, cta_num_n;                // CtaNum_MN
    int cluster_m, cluster_n, cluster_k;     // ClusterShape_MNK

    // Pipeline stages
    int stages_a;  // StagesA

    // Kernel configuration
    bool is_persistent;       // is_persistent mode
    std::string activation;   // activation_type (SiLu, None)
    std::string operation_c;  // operationC_type (Mul, Add, BiasAdd, None)
};

// Array of GEMM test parameters - aligned with configs from gemm.cpp
static GemmTestParams gemm_test_params[] = {
    {/* test_level */ "L2",
     /* state */ "enabled",
     /* M, N, K, L */ 2048, 2048, 2048, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"},  // 0

     {/* test_level */ "L2",
     /* state */ "enabled",
     /* M, N, K, L */ 2048, 2048, 2048, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "ColumnMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"},  // 1

     {/* test_level */ "L2",
     /* state */ "disabled",
     /* M, N, K, L */ 2048, 2048, 2048, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "ColumnMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"},  // 2


     {/* test_level */ "L2",
     /* state */ "disabled",
     /* M, N, K, L */ 4096, 3072, 4096, 1, 
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"},  // 3

     {/* test_level */ "L2",
     /* state */ "disabled",
     /* M, N, K, L */ 4096, 3072, 4096, 1, 
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "ColumnMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"},  // 4

     {/* test_level */ "L2",
     /* state */ "disabled",
     /* M, N, K, L */ 4096, 3072, 4096, 1, 
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "ColumnMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"},  // 5

     {/* test_level */ "L2",
     /* state */ "disabled",
     /* M, N, K, L */ 8192, 6144, 2048, 1, 
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "bf16", "bf16", "bf16", "bf16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 512, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ true,
     /* activation, operation_c */ "None", "None"},  // 6


#if 0
    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 0

    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "void", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 1, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "None", "BiasAdd"},  // 1

    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 2048, 2048, 4096, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 2

    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 1024, 1024, 1024, 4,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 3

    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "None", "Add"},  // 4

    /********** Start: All 8 combinations of Layouts **********/
    /* Note: first test is: "RowMajor", "RowMajor", "RowMajor" */

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "ColumnMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 5

    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "ColumnMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 6

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "ColumnMajor", "ColumnMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 7

    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "ColumnMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 8

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "ColumnMajor", "RowMajor", "ColumnMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 9

    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "ColumnMajor", "ColumnMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 10

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "ColumnMajor", "ColumnMajor", "ColumnMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 11

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "ColumnMajor", "ColumnMajor", "ColumnMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 12
    /********** End: All 8 combinations of Layouts **********/

    /********** Start: variation of {cta_num_m, cta_num_n} **********/
    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "void", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "ColumnMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 13

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "void", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "ColumnMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "None", "Add"},  // 14

    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 4,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 15

    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 4,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "None", "Add"},  // 16

    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 4, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 17
    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 4, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "None", "Add"},  // 18

    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 4, 4,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 19

    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 4, 4,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "None", "Add"},  // 20
    /********** End: variation of {cta_num_m, cta_num_n} **********/

    /************* Start: cluster size variation {1,2,4} */
    {/* test_level */ "L3",
     /* state */ "enabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 21

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 2,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 22

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 2, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 23

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 2, 2,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 24

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 2, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 25

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 2, 1, 2,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 26

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 2, 2, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 27

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 2, 2, 2,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 28

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 2, 2, 4,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 29

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 2, 4, 2,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 30

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 2, 4, 4,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 31

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 4, 2, 2,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 32

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 4, 2, 4,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 33

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 4, 4, 2,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 34

    {/* test_level */ "L3",
     /* state */ "disabled",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 4, 4, 4,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},  // 35
    /************* End: cluster size variation {1,2,4} */
#endif
};
