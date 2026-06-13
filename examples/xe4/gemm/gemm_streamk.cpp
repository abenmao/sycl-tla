#include "gemm_streamk.hpp"
#include <gtest/gtest.h>
#include <iostream>

#include "cutlass/gemm/kernel/xe4_tile_scheduler_stream_k.hpp"

using DecompositionMode = cutlass::gemm::kernel::detail::PersistentTileSchedulerXe4StreamKParams::DecompositionMode;
using ReductionMode = cutlass::gemm::kernel::detail::PersistentTileSchedulerXe4StreamKParams::ReductionMode;

// Base configuration for StreamK GEMM tests.
// Uses identity epilogue (D = A*B) so StreamK fixup is straightforward.
template <DecompositionMode Mode, int Splits = 1>
struct STREAMK_CONFIG {
  using ElementA = fp16;
  using ElementB = fp16;
  using ElementC = fp16;
  using ElementD = fp16;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using CtaTileShape_MNK = Shape<_256, _256, _128>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  static constexpr int StagesA = 2;
  static constexpr DecompositionMode decomposition_mode = Mode;
  static constexpr int splits = Splits;
  static constexpr auto activation_type = ActivationType::SiLu;
  static constexpr auto operationC_type = OperationCType::Mul;

  template <typename TileSchedulerArguments>
  static TileSchedulerArguments get_scheduler_args() {
    TileSchedulerArguments sched_args;
    sched_args.splits = splits;
    sched_args.decomposition_mode = decomposition_mode;
    sched_args.reduction_mode = ReductionMode::Deterministic;
    return sched_args;
  }
};

// Data-parallel baselines for comparison against StreamK shapes.
// Uses DecompositionMode::DataParallel to force pure DP and measure the wave-quantization loss.

// 5-tile DP baseline: 4 XE cores, 5 tiles → 1 full wave + 1 tail (25% waste).
struct STREAMK_DATA_PARALLEL_5T : public STREAMK_CONFIG<DecompositionMode::DataParallel> {
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {256, 1280, 8192, 1};
  static constexpr const char* decomposition_name = "DataParallel (5 tiles, 25% tail)";
};

// 9-tile DP baseline: 4 XE cores, 9 tiles → 2 full waves + 1 tail (12.5% waste).
struct STREAMK_DATA_PARALLEL_9T : public STREAMK_CONFIG<DecompositionMode::DataParallel> {
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {768, 768, 8192, 1};
  static constexpr const char* decomposition_name = "DataParallel (9 tiles, 12.5% tail)";
};

// Heuristic mode: scheduler decides between DP/SK based on wave quantization.
// 9-tile problem (tail=1, tail*2=2 < ctas_per_wave=4) → heuristic naturally selects SK.
struct STREAMK_HEURISTIC : public STREAMK_CONFIG<DecompositionMode::Heuristic> {
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {768, 768, 8192, 1};
  static constexpr const char* decomposition_name = "Heuristic";
};

// Split-K with 2 splits: each output tile is split across 2 workgroups along K.
// Requires workspace for partial accumulator reduction.
struct STREAMK_SPLIT_K_2 : public STREAMK_CONFIG<DecompositionMode::SplitK, 2> {
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 512, 1024, 1};
  static constexpr const char* decomposition_name = "SplitK (splits=2)";
};

// Split-K with 4 splits for a problem with large K.
struct STREAMK_SPLIT_K_4 : public STREAMK_CONFIG<DecompositionMode::SplitK, 4> {
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 512, 2048, 1};
  static constexpr const char* decomposition_name = "SplitK (splits=4)";
};

// Stream-K: 5 output tiles (1×5), 4 XE cores → 1 full wave + 1 tail (25% waste in DP).
// tail*2=2 < ctas_per_wave=4: heuristic would naturally select SK here.
// K=8192 (64 K-tiles): savings >> fixup overhead → strong SK utilization.
// Expected gain over DP: ~15-20%.
struct STREAMK_STREAM_K_5T : public STREAMK_CONFIG<DecompositionMode::StreamK> {
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {256, 1280, 8192, 1};
  static constexpr const char* decomposition_name = "StreamK (5 tiles, 25% tail)";
};

// Stream-K: 9 output tiles (3×3), 4 XE cores → 2 full waves + 1 tail (12.5% waste in DP).
// tail*2=2 < ctas_per_wave=4: heuristic naturally selects SK.
// K=8192 (64 K-tiles): savings > fixup overhead → moderate SK utilization.
// Expected gain over DP: ~10-15%.
struct STREAMK_STREAM_K_9T : public STREAMK_CONFIG<DecompositionMode::StreamK> {
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {768, 768, 8192, 1};
  static constexpr const char* decomposition_name = "StreamK (9 tiles, 12.5% tail)";
};

// Original 6-tile case (50% tail): heuristic picks DP, only forced SK gives ~5% gain.
// Kept for reference — shows cost model correctly limits SK to profitable geometries.
struct STREAMK_STREAM_K : public STREAMK_CONFIG<DecompositionMode::StreamK> {
  static constexpr cute::array<int, 4> ProblemShape_MNKL = {512, 768, 8192, 1};
  static constexpr const char* decomposition_name = "StreamK (6 tiles, 50% tail)";
};

template <typename T>
class GemmStreamKTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(GemmStreamKTest);

TYPED_TEST_P(GemmStreamKTest, simple_run) {
  run_gemm_streamk<TypeParam>();
}

REGISTER_TYPED_TEST_SUITE_P(GemmStreamKTest, simple_run);

using GemmStreamKTests = ::testing::Types<
    STREAMK_DATA_PARALLEL_5T,
    STREAMK_DATA_PARALLEL_9T,
    STREAMK_SPLIT_K_2,
    STREAMK_SPLIT_K_4,
    STREAMK_HEURISTIC,
    STREAMK_STREAM_K_5T, 
    STREAMK_STREAM_K_9T,
    STREAMK_STREAM_K
    >;

INSTANTIATE_TYPED_TEST_SUITE_P(GemmStreamK, GemmStreamKTest, GemmStreamKTests);

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int run_tests = RUN_ALL_TESTS();
  if (!::testing::UnitTest::GetInstance()->test_to_run_count()) {
    std::cout << "No tests were run.\n";
    return 1;
  }
  return run_tests;
}
