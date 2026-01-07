/*
 * Copyright (c) 2025. All rights reserved.
 */

#pragma once

#include <string>

/*
 * Structure to hold parameters for GEMM performance tests
 * Aligned with GEMM_TEST_CONFIG from gemm.cpp
 */
struct GemmTestParams {
    std::string test_level;  // Test execution level- "L2" for CI tests, "L3" for Nightly tests

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
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},

    {/* test_level */ "L3",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "void", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 1, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "None", "BiasAdd"},

    {/* test_level */ "L3",
     /* M, N, K, L */ 2048, 2048, 4096, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 2,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},


    {/* test_level */ "L3",
     /* M, N, K, L */ 1024, 1024, 1024, 4,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "SiLu", "Mul"},

    {/* test_level */ "L3",
     /* M, N, K, L */ 512, 768, 384, 1,
     /* dtype_a, dtype_b, dtype_c, dtype_d, dtype_acc */
     "fp16", "fp16", "fp16", "fp16", "float",
     /* layout_a, layout_b, layout_c */ "RowMajor", "RowMajor", "RowMajor",
     /* cta_tile_m, cta_tile_n, cta_tile_k */ 256, 256, 128,
     /* cta_num_m, cta_num_n */ 2, 1,
     /* cluster_m, cluster_n, cluster_k */ 1, 1, 1,
     /* stages_a */ 2,
     /* is_persistent */ false,
     /* activation, operation_c */ "None", "Add"},
};
