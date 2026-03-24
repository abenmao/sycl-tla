///////////////////////////////////////////////////////////////////////////////
/// test_xe4_slm_copy.cpp
///
/// Host-side unit test for XE4 G2S/S2G swizzled SLM copy atoms.
///
/// Tests a 32x32 bf16 tile:
///   1. G2S copy: GMEM -> SLM (swizzled)
///   2. Verify even partitions are NOT swapped
///   3. Verify odd  partitions ARE swapped (m XOR 1)
///   4. S2G copy: SLM -> output (unswizzled)
///   5. Verify round-trip matches original
///
///////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_xe4_eu_copy.hpp>
#include <cute/arch/xe4_async_gmma_slm_layout.hpp>

using namespace cute;
using namespace cute::xe4::slm::type1::kmajor;

// ============================================================================
//  Test fixture
// ============================================================================
class XE4SlmCopyTest : public ::testing::Test {
protected:
    using T = int;
    using ElemType = T;

    static constexpr int M  = 192;
    static constexpr int K  = 64;
    static constexpr int SG_SIZE = kSubGroupSize;                 // 32
    static constexpr int CM_K = kBytesPerCmRow /int(sizeof(T)) ; 
    static constexpr int NUM_CM_COLS   = K / CM_K;
    static constexpr int CM_BYTES      = 1024;
    static constexpr int ROWS_PER_TILE = 2;
    static constexpr int ESUB_BANKS    = 4;
    static constexpr int MMA_BANKS     = 4;
    static constexpr int CM_M = ROWS_PER_TILE * ESUB_BANKS * MMA_BANKS;  // 32
    static constexpr int NUM_CM_ROWS   = M / CM_M;
    static constexpr int DST_ELEMS = CM_M * CM_K * NUM_CM_ROWS * NUM_CM_COLS;

    std::vector<T> gmem_;
    std::vector<T> slm_;

    void SetUp() override {
        gmem_.resize(M * K);
        for (int k=0; k< (M*K); k++) { gmem_[k] = (T) 0; }

        slm_.resize(M * K);
        uint16_t x = 1;
        for (int k=0; k< (M*K); k++) { slm_[k] = (T) 0; }

        //print_raw_slm(&slm_[0]);

    }

    void print_raw_slm(const ElemType* buf)
    {

        constexpr int ELEM_PER_ROW = 256/sizeof(ElemType);
        constexpr int num_rows = ((M*K) / ELEM_PER_ROW);
        constexpr int ELEM_SZ = sizeof(ElemType);
        printf("┌─── SLM RAW MEMORY (element slots, %d per row)\n", ELEM_PER_ROW);
        printf("%7s │", "slot");
        for (int e = 0; e < ELEM_PER_ROW; ++e) printf(" %4d", e);
        printf("\n────────┼");
        for (int e = 0; e < ELEM_PER_ROW; ++e) printf("─────");
        printf("\n");

        constexpr int slots_per_row = 256/ ELEM_SZ;
        int i=0;
        for (int r = 0; r < num_rows; ++r) {
            int base = r * slots_per_row;
            printf(" %5d  │", base);
            for (int e = 0; e < ELEM_PER_ROW; ++e) {
                T val = buf[i++];
                if (val >= 0) printf(" %4d", (int) val);
                else          printf("    .");
            }
            int cm_bytes_in_elems = CM_BYTES / ELEM_SZ;
            int cm_idx = base / cm_bytes_in_elems;
            int row_in_cm = (base % cm_bytes_in_elems) / slots_per_row;
            if (row_in_cm == 0)
                printf("  -> CM(%d,%d)", cm_idx / NUM_CM_COLS, cm_idx % NUM_CM_COLS);
            printf("\n");
        }
        printf("\n");
    }

    // Fill GMEM with pattern: gmem[m][k] = value_fn(m, k)
    template <typename Fn>
    void fill_gmem(Fn value_fn) {
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                gmem_[m * K + k] = static_cast<T>(value_fn(m, k));
            }
        }
    }

    template <typename Fn>
    void fill_smem(Fn value_fn) {
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                slm_[m * K + k] = static_cast<T>(value_fn(m, k));
            }
        }
    }

    // Run all 32 threads for G2S copy
    void run_g2s() {
        for (int tid = 0; tid < SG_SIZE;  tid++) {
            simulate_g2s_thread(tid);
        }
        //print_raw_slm(&slm_[0]);
    }

    // Run all 32 threads for S2R copy //
    void run_s2r(std::vector<T>& out) {
        for (int tid = 0; tid < SG_SIZE;  tid++) {
            std::vector<T> thread_slice;
            simulate_s2r_thread(tid, thread_slice);

            int sr = 2*tid;
            size_t slice_idx  = 0;
            while (sr < M) {
              for (size_t i=0; i<ROWS_PER_TILE*K; i++, slice_idx++) {
                out[sr*K + i] = thread_slice[slice_idx];
              }
              sr += (ROWS_PER_TILE*SG_SIZE);
            }
        }
        //print_raw_slm(&slm_[0]);
    }

    void run_g2r(std::vector<T>& out) {
        for (int tid = 0; tid < SG_SIZE;  tid++) {
            std::vector<T> thread_slice;
            simulate_g2r_thread(tid, thread_slice);

            int sr = 2*tid;
            size_t slice_idx  = 0;
            while (sr < M) {
              for (size_t i=0; i<ROWS_PER_TILE*K; i++, slice_idx++) {
                out[sr*K + i] = thread_slice[slice_idx];
              }
              sr += (ROWS_PER_TILE*SG_SIZE);
            }
        }
        //print_raw_slm(&slm_[0]);
    }
    

    // Run all 32 threads for S2G copy into output buffer
    std::vector<T> run_s2g() {
        std::vector<T> out(M * K, static_cast<T>(0.0f));
        for (int tid = 0; tid < SG_SIZE; tid++) {
            simulate_s2g_thread(tid, out.data());
        }
        return out;
    }

    // Read element from SLM using raw tiled layout
    T read_slm(int m, int k) const {
        auto slm_layout = make_slm_layout_elem<T, M, K>();
        return slm_[slm_layout(m, k)];
    }

protected:

    void simulate_g2s_thread(int thread_id) {
        auto gmem_layout = make_layout(make_shape(Int<M>{}, Int<K>{}),
                                       make_stride(Int<K>{}, _1{}));
        auto gmem_tensor = make_tensor(gmem_.data(), gmem_layout);

        auto slm_layout = make_slm_layout_elem<T, M, K>();
        auto slm_tensor = make_tensor( slm_.data(), slm_layout);

        auto tiled_copy =
            make_xe4_g2s_tiled_copy<T, M, K>(gmem_tensor, slm_tensor);
        auto thr_copy   = tiled_copy.get_thread_slice(thread_id);

        auto tSrc = thr_copy.partition_S(gmem_tensor);
        auto tDst = thr_copy.partition_D(slm_tensor);

        copy(tiled_copy, tSrc, tDst);
    }


    void simulate_s2g_thread(int thread_id, T* out_flat) {
        auto slm_layout = make_slm_layout_elem<T, M, K>();
        auto slm_tensor = make_tensor(slm_.data(), slm_layout);

        auto out_layout = make_layout(make_shape(Int<M>{}, Int<K>{}),
                                      make_stride(Int<K>{}, _1{}));
        auto out_tensor = make_tensor(out_flat, out_layout);

        auto tiled_copy =
            make_xe4_s2g_tiled_copy<T, M, K>(slm_tensor, out_tensor);
        auto thr_copy   = tiled_copy.get_thread_slice(thread_id);

        auto tSrc = thr_copy.partition_S(slm_tensor);
        auto tDst = thr_copy.partition_D(out_tensor);

        copy(tiled_copy, tSrc, tDst);
    }

    void simulate_r2g_thread(int thread_id, const std::vector<T>& reg_data) {
      auto gmem_layout = make_layout(make_shape(Int<M>{}, Int<K>{}),
                                     make_stride(Int<K>{}, _1{}));
      auto gmem_tensor = make_tensor(gmem_.data(), gmem_layout);

      auto reg_layout = make_xe4_reg_layout_for_tiled_copy<T, M, K>();
      auto rSrc = make_tensor(reg_data.data(), reg_layout);

      auto tiled_copy = make_xe4_r2g_tiled_copy<T, M, K>();
      auto thr_copy = tiled_copy.get_thread_slice(thread_id);

      auto tDst = thr_copy.partition_D(gmem_tensor);

      copy(tiled_copy, rSrc, tDst);
    }

    void simulate_g2r_thread(int thread_id, std::vector<T>& reg_data) {
      reg_data.clear();

      auto gmem_layout = make_layout(make_shape(Int<M>{}, Int<K>{}),
                                     make_stride(Int<K>{}, _1{}));
      auto gmem_tensor = make_tensor(gmem_.data(), gmem_layout);

      auto reg_layout = make_xe4_reg_layout_for_tiled_copy<T, M, K>();
      auto rDst = make_tensor<T>(reg_layout);

      auto tiled_copy = make_xe4_g2r_tiled_copy<T, M, K>();
      auto thr_copy = tiled_copy.get_thread_slice(thread_id);

      auto tSrc = thr_copy.partition_S(gmem_tensor);

      copy(tiled_copy, tSrc, rDst);

      reg_data.resize( size(reg_layout) );
      auto reg_tensor = make_tensor( reg_data.data(), reg_layout );
      copy(rDst, reg_tensor);
    }

    void simulate_s2r_thread(int thread_id, std::vector<T>& reg_data) {
      reg_data.clear();

      auto slm_layout = make_slm_layout_elem<T, M, K>();
      auto slm_tensor = make_tensor(slm_.data(), slm_layout);

      auto reg_layout = make_xe4_reg_layout_for_tiled_copy<T, M, K>();
      auto rDst = make_tensor<T>(reg_layout);

      auto tiled_copy =
          make_xe4_s2r_tiled_copy<T, M, K>(slm_tensor, rDst);
      auto thr_copy = tiled_copy.get_thread_slice(thread_id);

      auto tSrc = thr_copy.partition_S(slm_tensor);


      copy(tiled_copy, tSrc, rDst);


      reg_data.resize( size(reg_layout) );
      auto reg_tensor = make_tensor( reg_data.data(), reg_layout );
      copy(rDst, reg_tensor);
    }
};


// ============================================================================
//  TEST: Even partition rows are NOT swizzled
// ============================================================================
TEST_F(XE4SlmCopyTest, EvenPartitionNotSwizzled) {
    fill_gmem([](int m, int k) -> float { return static_cast<float>(m*K + k); });

    run_g2s();

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < CM_K; k++) {
            float got      = static_cast<float>(read_slm(m, k));
            float expected = static_cast<float>(gmem_[m * K + k]);
            EXPECT_EQ(got, expected)
                << "SLM[" << m << "][" << k << "] = " << got
                << ", expected " << expected << " (even partition, no swizzle)";
        }
    }
}

// ============================================================================
//  TEST: Odd partition rows ARE swizzled (m XOR 1)
// ============================================================================
TEST_F(XE4SlmCopyTest, OddPartitionSwizzled) {
    fill_gmem([](int m, int k) -> float { return static_cast<float>(m * 32 + k); });
    run_g2s();

    for (int m = 0; m < M; m++) {
        for (int k = CM_K; k < 2 * CM_K; k++) {
            float got      = static_cast<float>(read_slm(m, k));
            int swizzled_m = m ^ 1;
            float expected = static_cast<float>(gmem_[swizzled_m * K + k]);
            EXPECT_EQ(got, expected)
                << "SLM[" << m << "][" << k << "] = " << got
                << ", expected " << expected
                << " (odd partition, gmem[" << swizzled_m << "][" << k << "])";
        }
    }
}

// ============================================================================
//  TEST: Swizzle is non-trivial -- odd partition rows differ from identity
// ============================================================================
TEST_F(XE4SlmCopyTest, SwizzleIsNontrivial) {
    // gmem[m][k] = m for all k, so even/odd rows are distinguishable
    fill_gmem([](int m, int /*k*/) -> float { return static_cast<float>(m); });
    run_g2s();

    for (int m = 0; m < M; m++) {
        int k = CM_K;  // first element of odd partition
        float got                = static_cast<float>(read_slm(m, k));
        float expected_swizzled  = static_cast<float>(m ^ 1);
        EXPECT_EQ(got, expected_swizzled)
            << "SLM[" << m << "][" << k << "] = " << got
            << ", expected " << expected_swizzled
            << " (swizzle should swap row " << m << " with row " << (m ^ 1) << ")";
    }
}

// ============================================================================
//  TEST: Even partition identity preserved
// ============================================================================
TEST_F(XE4SlmCopyTest, EvenPartitionIdentity) {
    fill_gmem([](int m, int /*k*/) -> float { return static_cast<float>(m); });
    run_g2s();

    for (int m = 0; m < M; m++) {
        int k = 0;  // first element of even partition
        float got      = static_cast<float>(read_slm(m, k));
        float expected = static_cast<float>(m);
        EXPECT_EQ(got, expected)
            << "SLM[" << m << "][" << k << "] = " << got
            << ", expected " << expected << " (even partition should not swizzle)";
    }
}

// ============================================================================
//  TEST: Round-trip G2S -> S2G recovers original data
// ============================================================================
TEST_F(XE4SlmCopyTest, RoundTrip) {
    fill_gmem([](int m, int k) -> float { return static_cast<float>(m * 32 + k); });
    run_g2s();
    auto out = run_s2g();

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            float got = static_cast<float>(out[m * K + k]);
            float exp = static_cast<float>(gmem_[m * K + k]);
            EXPECT_EQ(got, exp)
                << "out[" << m << "][" << k << "] = " << got
                << ", expected " << exp << " (round-trip mismatch)";
        }
    }
}

// ============================================================================
//  TEST: Round-trip with sequential pattern
// ============================================================================
TEST_F(XE4SlmCopyTest, RoundTripSequential) {
    fill_gmem([](int m, int k) -> float { return static_cast<float>(m * 100 + k); });
    run_g2s();
    auto out = run_s2g();

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            float got = static_cast<float>(out[m * K + k]);
            float exp = static_cast<float>(gmem_[m * K + k]);
            EXPECT_EQ(got, exp)
                << "out[" << m << "][" << k << "] = " << got
                << ", expected " << exp;
        }
    }
}

// ============================================================================
//  TEST: Round-trip with constant rows (stress swizzle symmetry)
// ============================================================================
TEST_F(XE4SlmCopyTest, RoundTripConstantRows) {
    fill_gmem([](int m, int /*k*/) -> float { return static_cast<float>(m); });
    run_g2s();
    auto out = run_s2g();

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            float got = static_cast<float>(out[m * K + k]);
            float exp = static_cast<float>(gmem_[m * K + k]);
            EXPECT_EQ(got, exp)
                << "out[" << m << "][" << k << "] = " << got
                << ", expected " << exp;
        }
    }
}

TEST_F(XE4SlmCopyTest, CheckRowsOfaSingleThread) {
    fill_gmem([](int m, int /*k*/) -> float { return static_cast<float>(m); });
    run_g2s();

    std::vector<T> reg_data_across_all_threads;

    reg_data_across_all_threads.resize(M*K);


    run_s2r(reg_data_across_all_threads);

    EXPECT_EQ(reg_data_across_all_threads, gmem_);
}

TEST_F(XE4SlmCopyTest, CheckRowsOfaSingleThread_G2R) {
    fill_gmem([](int m, int /*k*/) -> float { return static_cast<float>(m); });
    run_g2s();

    std::vector<T> reg_data_across_all_threads;

    reg_data_across_all_threads.resize(M*K);

    run_g2r(reg_data_across_all_threads);
    EXPECT_EQ(reg_data_across_all_threads, gmem_);
}


TEST_F(XE4SlmCopyTest, CheckRowsOfaSingleThread_R2G) {
    fill_gmem([](int m, int /*k*/) -> float { return static_cast<float>(m); });

    std::vector<T> expected_result = gmem_;

    // gmem is now all zeros //
    fill_gmem([](int m, int /*k*/) -> float { return static_cast<float>(0); });

    for (int tid=0; tid<SG_SIZE; tid++) {
      auto reg_layout = make_xe4_reg_layout_for_tiled_copy<T, M, K>();
      std::vector<T> reg_data(size(reg_layout), 0);
      int tid_m = ROWS_PER_TILE*tid;

      size_t idx = 0;
      while (tid_m < M) {
        for (int k=0; k<K; k++) {
          reg_data[idx++] = tid_m;
        }
        for (int k=0; k<K; k++) {
          reg_data[idx++] = tid_m+1;
        }
        tid_m += (ROWS_PER_TILE*SG_SIZE);
      }
      simulate_r2g_thread(tid, reg_data);
    }

#if 0
    for (size_t i=0; i<gmem_.size(); i++) {
      if (i%K == 0) {
        std::cout << std::endl;
      }
      std::cout << gmem_[i] << " ";
    }
#endif
    EXPECT_EQ(expected_result, gmem_);
}
