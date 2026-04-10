
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

// ==================== Template Dispatch Helpers ====================
// Map runtime string parameters to compile-time template types using
// generic lambdas. Each dispatcher calls a lambda with type/enum tags
// that carry compile-time information.
//
// Key design:
//   - Layout is dispatched independently (additive: 2-4 branches).
//   - dtype + activation + operation are dispatched as a single flat
//     list of exactly 8 valid tuples (not a Cartesian product), matching
//     the old macro DISPATCH_ALL_TYPE_CONFIGS. This keeps instantiation
//     count identical to the macro approach: 8 type configs × 2 layouts
//     = 16 executeGemm instantiations.

// Carries a type as a value (works for void, unlike direct instantiation)
template <typename T>
struct type_tag { using type = T; };

// Carries an enum value as a type (alias for std::integral_constant)
template <auto V>
using enum_tag = std::integral_constant<decltype(V), V>;

// Dispatch layout combination (A, B, C) from runtime strings to compile-time types.
template <typename Fn>
bool dispatchLayout(const std::string& la, const std::string& lb,
                    const std::string& lc, Fn&& fn) {
    using RM = cutlass::layout::RowMajor;
    using CM = cutlass::layout::ColumnMajor;
    if (la == "RowMajor"    && lb == "RowMajor"    && lc == "RowMajor") return fn(type_tag<RM>{}, type_tag<RM>{}, type_tag<RM>{});
    if (la == "RowMajor"    && lb == "ColumnMajor" && lc == "RowMajor") return fn(type_tag<RM>{}, type_tag<CM>{}, type_tag<RM>{});
    // if (la == "ColumnMajor" && lb == "RowMajor"    && lc == "RowMajor") return fn(type_tag<CM>{}, type_tag<RM>{}, type_tag<RM>{});
    // if (la == "ColumnMajor" && lb == "ColumnMajor" && lc == "RowMajor") return fn(type_tag<CM>{}, type_tag<CM>{}, type_tag<RM>{});
    return false;
}

// Dispatch (dtype_a, dtype_b, dtype_c, dtype_d, acc, activation, operation) as
// a flat list of exactly 8 valid combinations. This avoids the combinatorial
// explosion that separate dispatchers would create.
template <typename Fn>
bool dispatchTypeConfig(const std::string& da, const std::string& db,
                        const std::string& dc, const std::string& dd,
                        const std::string& acc, const std::string& act,
                        const std::string& op, Fn&& fn) {
    // fp16 fp16 fp16 fp16 float + SiLu/Mul
    if (da=="fp16" && db=="fp16" && dc=="fp16" && dd=="fp16" && acc=="float" && act=="SiLu" && op=="Mul")
        return fn(type_tag<fp16>{}, type_tag<fp16>{}, type_tag<fp16>{}, type_tag<fp16>{}, type_tag<float>{},
                  enum_tag<ActivationType::SiLu>{}, enum_tag<OperationCType::Mul>{});
    // fp16 fp16 fp16 fp16 float + None/Add
    if (da=="fp16" && db=="fp16" && dc=="fp16" && dd=="fp16" && acc=="float" && act=="None" && op=="Add")
        return fn(type_tag<fp16>{}, type_tag<fp16>{}, type_tag<fp16>{}, type_tag<fp16>{}, type_tag<float>{},
                  enum_tag<ActivationType::None>{}, enum_tag<OperationCType::Add>{});
    // fp16 fp16 fp16 fp16 float + None/None
    if (da=="fp16" && db=="fp16" && dc=="fp16" && dd=="fp16" && acc=="float" && act=="None" && op=="None")
        return fn(type_tag<fp16>{}, type_tag<fp16>{}, type_tag<fp16>{}, type_tag<fp16>{}, type_tag<float>{},
                  enum_tag<ActivationType::None>{}, enum_tag<OperationCType::None>{});
    // fp16 fp16 void fp16 float + None/BiasAdd
    if (da=="fp16" && db=="fp16" && dc=="void" && dd=="fp16" && acc=="float" && act=="None" && op=="BiasAdd")
        return fn(type_tag<fp16>{}, type_tag<fp16>{}, type_tag<void>{}, type_tag<fp16>{}, type_tag<float>{},
                  enum_tag<ActivationType::None>{}, enum_tag<OperationCType::BiasAdd>{});
    // fp16 fp16 void fp16 float + None/None
    if (da=="fp16" && db=="fp16" && dc=="void" && dd=="fp16" && acc=="float" && act=="None" && op=="None")
        return fn(type_tag<fp16>{}, type_tag<fp16>{}, type_tag<void>{}, type_tag<fp16>{}, type_tag<float>{},
                  enum_tag<ActivationType::None>{}, enum_tag<OperationCType::None>{});
    // bf16 bf16 bf16 bf16 float + SiLu/Mul
    if (da=="bf16" && db=="bf16" && dc=="bf16" && dd=="bf16" && acc=="float" && act=="SiLu" && op=="Mul")
        return fn(type_tag<bf16>{}, type_tag<bf16>{}, type_tag<bf16>{}, type_tag<bf16>{}, type_tag<float>{},
                  enum_tag<ActivationType::SiLu>{}, enum_tag<OperationCType::Mul>{});
    // bf16 bf16 bf16 bf16 float + None/Mul
    if (da=="bf16" && db=="bf16" && dc=="bf16" && dd=="bf16" && acc=="float" && act=="None" && op=="Mul")
        return fn(type_tag<bf16>{}, type_tag<bf16>{}, type_tag<bf16>{}, type_tag<bf16>{}, type_tag<float>{},
                  enum_tag<ActivationType::None>{}, enum_tag<OperationCType::Mul>{});
    // bf16 bf16 bf16 bf16 float + None/None
    if (da=="bf16" && db=="bf16" && dc=="bf16" && dd=="bf16" && acc=="float" && act=="None" && op=="None")
        return fn(type_tag<bf16>{}, type_tag<bf16>{}, type_tag<bf16>{}, type_tag<bf16>{}, type_tag<float>{},
                  enum_tag<ActivationType::None>{}, enum_tag<OperationCType::None>{});
    return false;
}


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

        // Skip ColumnMajor layout_a (JIRA: JSW-1222)
        if (params->layout_a == "ColumnMajor") {
            GTEST_SKIP() << "ColumnMajor is giving compilation error. JIRA: JSW-1222";
            return;
        }

        // Timing measurement
        auto start = std::chrono::high_resolution_clock::now();

        // Template-based dispatch: layout -> type config -> executeGemm
        bool dispatched = dispatchLayout(
            params->layout_a, params->layout_b, params->layout_c,
            [&](auto la_tag, auto lb_tag, auto lc_tag) {
                return dispatchTypeConfig(
                    params->dtype_a, params->dtype_b, params->dtype_c,
                    params->dtype_d, params->dtype_acc,
                    params->activation, params->operation_c,
                    [&](auto ta, auto tb, auto tc, auto td, auto tacc, auto act_tag, auto op_tag) {
                        using TA   = typename decltype(ta)::type;
                        using TB   = typename decltype(tb)::type;
                        using TC   = typename decltype(tc)::type;
                        using TD   = typename decltype(td)::type;
                        using TAcc = typename decltype(tacc)::type;
                        using LA   = typename decltype(la_tag)::type;
                        using LB   = typename decltype(lb_tag)::type;
                        using LC   = typename decltype(lc_tag)::type;
                        constexpr auto act_v = decltype(act_tag)::value;
                        constexpr auto op_v  = decltype(op_tag)::value;

                        executeGemm<TA, TB, TC, TD, TAcc, LA, LB, LC, act_v, op_v>(
                            params->cta_tile_m, params->cta_tile_n, params->cta_tile_k,
                            params->cta_num_m, params->cta_num_n,
                            params->cluster_m, params->cluster_n, params->cluster_k,
                            params->is_persistent);
                        return true;
                    });
            });

        if (!dispatched) {
            printUnsupportedCombination(params);
            FAIL() << "This combination is not supported";
            return;
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
