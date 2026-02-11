
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

#include "gemm_params.hpp"
#include "testFixture.hpp"

using ::testing::TestWithParam;
using namespace cute;

// template-based configuration that can be populated from runtime parameters
template <typename ElementA_T = half_t, typename ElementB_T = half_t, typename ElementC_T = half_t, typename ElementD_T = half_t,
          typename ElementAccumulator_T = float,
          typename LayoutA_T = cutlass::layout::RowMajor, typename LayoutB_T = cutlass::layout::RowMajor, typename LayoutC_T = cutlass::layout::RowMajor,
          ActivationType activation_T = ActivationType::SiLu, OperationCType operationC_T = OperationCType::Mul,
          int CtaTileM = 256, int CtaTileN = 256, int CtaTileK = 128,
          int CtaNumM = 2, int CtaNumN = 1,
          int ClusterM = 1, int ClusterN = 1, int ClusterK = 1,
          bool IsPersistent = false>
struct GEMM_RUNTIME_CONFIG {
    using ElementA = ElementA_T;
    using ElementB = ElementB_T;
    using ElementC = ElementC_T;
    using ElementD = ElementD_T;
    using ElementAccumulator = ElementAccumulator_T;
    using LayoutA = LayoutA_T;
    using LayoutB = LayoutB_T;
    using LayoutC = LayoutC_T;
    using CtaTileShape_MNK = Shape<Int<CtaTileM>, Int<CtaTileN>, Int<CtaTileK>>;
    using CtaNum_MN = Shape<Int<CtaNumM>, Int<CtaNumN>>;
    using ClusterShape_MNK = Shape<Int<ClusterM>, Int<ClusterN>, Int<ClusterK>>;

    static constexpr int StagesA = 2;
    static constexpr bool is_persistent = IsPersistent;
    static constexpr auto activation_type = activation_T;
    static constexpr auto operationC_type = operationC_T;
    static constexpr cute::array<int, 4> ProblemShape_MNKL = {1, 1, 1, 1};
};

#define DISPATCH_GEMM_CONFIG(tile_m_val, tile_n_val, tile_k_val, cta_m_val, cta_n_val, cluster_m_val, cluster_n_val, cluster_k_val, persistent_val) \
    if (cta_tile_m == tile_m_val && cta_tile_n == tile_n_val && cta_tile_k == tile_k_val &&                                          \
        cta_m == cta_m_val && cta_n == cta_n_val &&                                                                                   \
        cluster_m == cluster_m_val && cluster_n == cluster_n_val && cluster_k == cluster_k_val &&                                     \
        is_persistent == persistent_val) {                                                                                             \
        using Config = GEMM_RUNTIME_CONFIG<ElementA, ElementB, ElementC, ElementD, ElementAccumulator,                                \
                                           LayoutA, LayoutB, LayoutC,                                                                 \
                                           activation, operation,                                                                     \
                                           tile_m_val, tile_n_val, tile_k_val,                                                        \
                                           cta_m_val, cta_n_val, cluster_m_val,                                                       \
                                         cluster_n_val, cluster_k_val, persistent_val>;                                               \
        default_run_gemm<Config>();                                                                                                   \
        return;                                                                                                                       \
    }

// Global parameter list for GEMM operations
std::deque<TestParamInfo> GemmOpParamsList;

/*
 * GEMM Operator Test Fixture
 */
class GemmOperator : public testFixture {
   public:

    template <typename ElementA, typename ElementB, typename ElementC, typename ElementD, typename ElementAccumulator,
              typename LayoutA, typename LayoutB, typename LayoutC,
              ActivationType activation, OperationCType operation>
    void executeGemm(int cta_tile_m, int cta_tile_n, int cta_tile_k, int cta_m, int cta_n, int cluster_m, int cluster_n, int cluster_k, bool is_persistent) {
        // Define macros for automatic dispatch generation
        #define DISPATCH_BOTH_PERSISTENT(tm, tn, tk, cm, cn, clm, cln, clk) \
            DISPATCH_GEMM_CONFIG(tm, tn, tk, cm, cn, clm, cln, clk, true) \
            DISPATCH_GEMM_CONFIG(tm, tn, tk, cm, cn, clm, cln, clk, false)

        #define CTA_TILE_CONFIG(tm, tn, tk) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 1, 1, 1, 1, 1) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 1, 1, 2, 2, 2) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 1, 2, 1, 1, 1) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 1, 2, 2, 2, 2) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 2, 1, 1, 1, 1) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 2, 1, 2, 2, 2) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 2, 2, 1, 1, 1) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 2, 2, 2, 1, 1) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 2, 2, 2, 2, 2) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 4, 4, 1, 1, 1) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 4, 4, 2, 2, 2) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 2, 4, 1, 1, 1) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 2, 4, 2, 2, 2) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 4, 2, 1, 1, 1) \
            DISPATCH_BOTH_PERSISTENT(tm, tn, tk, 4, 2, 2, 2, 2)

        // All supported CTA tile sizes
        CTA_TILE_CONFIG(256, 256, 128)
        CTA_TILE_CONFIG(256, 512, 128)

        // If no configuration matched
        std::cout << "Unsupported CTA/Cluster configuration: CTA Tile(" << cta_tile_m << "," << cta_tile_n << "," << cta_tile_k
                  << ") CTA(" << cta_m << "," << cta_n << ") Cluster(" << cluster_m << "," << cluster_n << "," << cluster_k 
                  << ") Persistent=" << (is_persistent ? "true" : "false") << std::endl;
        FAIL() << "Configuration not supported";
        #undef CTA_TILE_CONFIG
        #undef DISPATCH_BOTH_PERSISTENT
        #undef DISPATCH_GEMM_CONFIG
    }

   private:
    void printUnsupportedCombination(const GemmTestParams* params, const std::string& context = "") {
        std::cout << "Unsupported combination" << (context.empty() ? "" : (" for " + context)) << ": \n"
                  << "layoutA=" << params->layout_a << ", layoutB=" << params->layout_b << ", layoutC=" << params->layout_c << "\n"
                  << "dtypeA =" << params->dtype_a << ", dtypeB=" << params->dtype_b << ", dtypeC=" << params->dtype_c << ", dtypeD=" << params->dtype_d << "\n"
                  << "dtype_acc=" << params->dtype_acc << "\n"
                  << "activation=" << params->activation << ", operationC=" << params->operation_c << std::endl;
    }

public:
    void runTest() override {
        auto* params = getTestParams<GemmTestParams>();
        if (params == nullptr) {
            FAIL() << "Invalid test parameters";
            return;
        }

        // check if test is enabled
        if (params->state != "enabled") {
            GTEST_SKIP() << "Test is disabled. Skipping execution";
            return;
        }

        // Timing measurement
        auto start = std::chrono::high_resolution_clock::now();
        // Dispatch based on parameters - organized by layout combinations
        if (params->layout_a == "RowMajor" && params->layout_b == "RowMajor" && params->layout_c == "RowMajor") {
            if (params->dtype_a == "fp16" && params->dtype_b == "fp16" && params->dtype_c == "fp16" && params->dtype_d == "fp16" && params->dtype_acc == "float") {
                if (params->activation == "SiLu" && params->operation_c == "Mul") {
                    executeGemm<fp16, fp16, fp16, fp16, float,
                                cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor,
                                ActivationType::SiLu, OperationCType::Mul>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else if (params->activation == "None" && params->operation_c == "Add") {
                    executeGemm<fp16, fp16, fp16, fp16, float,
                                cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor,
                                ActivationType::None, OperationCType::Add>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else if (params->activation == "None" && params->operation_c == "None") {
                    executeGemm<fp16, fp16, fp16, fp16, float,
                                cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor,
                                ActivationType::None, OperationCType::None>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else {
                    printUnsupportedCombination(params, "RowMajor-RowMajor-RowMajor");
                    FAIL() << "This combination is not supported";
                }
            } else if (params->dtype_a == "fp16" && params->dtype_b == "fp16" && params->dtype_c == "void" && params->dtype_d == "fp16" && params->dtype_acc == "float") {
                if (params->activation == "None" && params->operation_c == "BiasAdd") {
                    executeGemm<fp16, fp16, void, fp16, float,
                                cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor,
                                ActivationType::None, OperationCType::BiasAdd>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else if (params->activation == "None" && params->operation_c == "None") {
                    executeGemm<fp16, fp16, void, fp16, float,
                                cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor,
                                ActivationType::None, OperationCType::None>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else {
                    printUnsupportedCombination(params, "RowMajor-RowMajor-RowMajor");
                    FAIL()
                    << "This combination is not supported";
                }
            } else if(params->dtype_a == "bf16" && params->dtype_b == "bf16" && params->dtype_c == "bf16" && params->dtype_d == "bf16" && params->dtype_acc == "float"){
                if (params->activation == "SiLu" && params->operation_c == "Mul") {
                    executeGemm<bf16, bf16, bf16, bf16, float,
                                cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor,
                                ActivationType::SiLu, OperationCType::Mul>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else if (params->activation == "None" && params->operation_c == "Mul") {
                    executeGemm<bf16, bf16, bf16, bf16, float,
                                cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor,
                                ActivationType::None, OperationCType::Mul>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else if (params->activation == "None" && params->operation_c == "None") {
                    executeGemm<bf16, bf16, bf16, bf16, float,
                                cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor,
                                ActivationType::None, OperationCType::None>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else {
                    printUnsupportedCombination(params, "RowMajor-RowMajor-RowMajor");
                    FAIL() << "This combination is not supported";
                }
            } else {
                printUnsupportedCombination(params, "RowMajor-RowMajor-RowMajor");
                FAIL() << "This combination is not supported";
            }
        } else if (params->layout_a == "RowMajor" && params->layout_b == "ColumnMajor" && params->layout_c == "RowMajor") {
            if (params->dtype_a == "fp16" && params->dtype_b == "fp16" && params->dtype_c == "fp16" && params->dtype_d == "fp16" && params->dtype_acc == "float") {
                if (params->activation == "SiLu" && params->operation_c == "Mul") {
                    executeGemm<fp16, fp16, fp16, fp16, float,
                                cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor,
                                ActivationType::SiLu, OperationCType::Mul>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else if (params->activation == "None" && params->operation_c == "Add") {
                    executeGemm<fp16, fp16, fp16, fp16, float,
                                cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor,
                                ActivationType::None, OperationCType::Add>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else if (params->activation == "None" && params->operation_c == "None") {
                    executeGemm<fp16, fp16, fp16, fp16, float,
                                cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor,
                                ActivationType::None, OperationCType::None>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else {
                    printUnsupportedCombination(params, "RowMajor-ColumnMajor-RowMajor");
                    FAIL() << "This combination is not supported";
                }
            } else if (params->dtype_a == "fp16" && params->dtype_b == "fp16" && params->dtype_c == "void" && params->dtype_d == "fp16" && params->dtype_acc == "float") {
                if (params->activation == "None" && params->operation_c == "BiasAdd") {
                    executeGemm<fp16, fp16, void, fp16, float,
                                cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor,
                                ActivationType::None, OperationCType::BiasAdd>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else if (params->activation == "None" && params->operation_c == "None") {
                    executeGemm<fp16, fp16, void, fp16, float,
                                cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor,
                                ActivationType::None, OperationCType::None>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else {
                    printUnsupportedCombination(params, "RowMajor-ColumnMajor-RowMajor");
                    FAIL()
                        << "This combination is not supported";
                }
            } else if(params->dtype_a == "bf16" && params->dtype_b == "bf16" && params->dtype_c == "bf16" && params->dtype_d == "bf16" && params->dtype_acc == "float"){
                if (params->activation == "SiLu" && params->operation_c == "Mul") {
                    executeGemm<bf16, bf16, bf16, bf16, float,
                                cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor,
                                ActivationType::SiLu, OperationCType::Mul>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else if (params->activation == "None" && params->operation_c == "Mul") {
                    executeGemm<bf16, bf16, bf16, bf16, float,
                                cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor,
                                ActivationType::None, OperationCType::Mul>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else if (params->activation == "None" && params->operation_c == "None") {
                    executeGemm<bf16, bf16, bf16, bf16, float,
                                cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor,
                                ActivationType::None, OperationCType::None>(
                        params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                        params->cta_num_m, params->cta_num_n,
                        params->cluster_m, params->cluster_n, params->cluster_k, params->is_persistent);
                } else {
                    printUnsupportedCombination(params, "RowMajor-ColumnMajor-RowMajor");
                    FAIL() << "This combination is not supported";
                }
            } else {
                printUnsupportedCombination(params, "RowMajor-ColumnMajor-RowMajor");
                FAIL() << "This combination is not supported";
            }
        } else if (params->layout_a == "ColumnMajor" && params->layout_b == "RowMajor" && params->layout_c == "RowMajor") {
            GTEST_SKIP() << "ColumnMajor is giving compilation error. JIRA: JSW-1222";
        } else if (params->layout_a == "ColumnMajor" && params->layout_b == "ColumnMajor" && params->layout_c == "RowMajor") {
            GTEST_SKIP() << "ColumnMajor is giving compilation error. JIRA: JSW-1222";
        } else {
            std::cout << "Unsupported layout combination:\n";
            printUnsupportedCombination(params);
            FAIL() << "This combination is not supported on XE4 hardware";
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
    }

    // Override to provide runtime problem shape from test parameters
    cute::array<int, 4> getRuntimeProblemShape() override {
        auto* params = getTestParams<GemmTestParams>();
        if (params) {
            return {params->M, params->N, params->K, params->L};
        }
        return {0, 0, 0, 0};  // Fallback to default
    }
};

/*
 * Function to create GEMM operation parameters list
 */
bool createGemmOpParamsList(std::deque<TestParamInfo>& GemmOpParamsList,
                            struct GemmTestParams* ptestParam,
                            int n) {
    if (GemmOpParamsList.size() != 0)
        return true;

    for (int i = 0; i < n; i++) {
        std::string testname =
            "gemm" + std::string("_") +
            std::string(ptestParam[i].test_level) + std::string("_") +
            std::string(ptestParam[i].state) + std::string("_") +
            std::string("M_") + std::to_string(ptestParam[i].M) + std::string("_") +
            std::string("N_") + std::to_string(ptestParam[i].N) + std::string("_") +
            std::string("K_") + std::to_string(ptestParam[i].K) + std::string("_") +
            std::string("L_") + std::to_string(ptestParam[i].L) + std::string("_") +
            std::string("dtypeA_") + ptestParam[i].dtype_a + std::string("_") +
            std::string("dtypeB_") + ptestParam[i].dtype_b + std::string("_") +
            std::string("dtypeC_") + ptestParam[i].dtype_c + std::string("_") +
            std::string("dtypeD_") + ptestParam[i].dtype_d + std::string("_") +
            std::string("layoutAcc_") + ptestParam[i].dtype_acc + std::string("_") +
            std::string("layoutA_") + ptestParam[i].layout_a + std::string("_") +
            std::string("layoutB_") + ptestParam[i].layout_b + std::string("_") +
            std::string("layoutC_") + ptestParam[i].layout_c + std::string("_") +
            std::string("ctaTileM_") + std::to_string(ptestParam[i].cta_tile_m) + std::string("_") +
            std::string("ctaTileN_") + std::to_string(ptestParam[i].cta_tile_n) + std::string("_") +
            std::string("ctaTileK_") + std::to_string(ptestParam[i].cta_tile_k) + std::string("_") +
            std::string("ctaNumM_") + std::to_string(ptestParam[i].cta_num_m) + std::string("_") +
            std::string("ctaNumN_") + std::to_string(ptestParam[i].cta_num_n) + std::string("_") +
            std::string("clusterM_") + std::to_string(ptestParam[i].cluster_m) + std::string("_") +
            std::string("clusterN_") + std::to_string(ptestParam[i].cluster_n) + std::string("_") +
            std::string("clusterK_") + std::to_string(ptestParam[i].cluster_k) + std::string("_") +
            std::string("stagesA_") + std::to_string(ptestParam[i].stages_a) + std::string("_") +
            std::string("isPersistent_") + (ptestParam[i].is_persistent ? "true" : "false") + std::string("_") +
            std::string("activation_") + ptestParam[i].activation + std::string("_") +
            std::string("operationC_") + ptestParam[i].operation_c;

        // Convert to upper case before storing
        TestParamInfo::addTestToParamList(GemmOpParamsList, testname, (void*)&ptestParam[i]);
    }
    return true;
}

// Custom name generator that includes the test name as the parameter name
struct GemmTestNameGenerator {
    std::string operator()(const ::testing::TestParamInfo<TestParamInfo>& info) const {
        return info.param.testName;
    }
};

TEST_P(GemmOperator, GemmOpPerfTest) {
    runTest();
}

// Instantiate the test case with custom name generator to show full test names
INSTANTIATE_TEST_SUITE_P(GemmPerfTests, GemmOperator,
                         ::testing::ValuesIn(true == createGemmOpParamsList(GemmOpParamsList,
                                                                            gemm_test_params,
                                                                            sizeof(gemm_test_params) / sizeof(GemmTestParams))
                                                 ? GemmOpParamsList
                                                 : GemmOpParamsList));
