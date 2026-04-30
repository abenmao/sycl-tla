#pragma once
#include <cute/tensor.hpp>

namespace cute {

// ============================================================================
// default_warp_layout_for_tiled_mma
//
// Compile-time metafunction that computes the subgroup layout
// Shape<SgM, SgN, SgK> to pass as the AtomLayoutMNK argument of
// make_tiled_mma for an XE4_TMM atom.
//
// Template parameters:
//   MMA_Op          — the XE4_TMM atom type (provides M, N_val, K)
//   bM, bN, bK      — workgroup tile dimensions
//   WorkgroupSize   — total number of threads in the workgroup
//   SubgroupSize    — threads per subgroup (default: 32)
//   NumProducerSGs  — producer subgroups not participating in compute (default: 0)
//
// Algorithm:
//   1. num_consumer_sgs = WorkgroupSize / SubgroupSize - NumProducerSGs
//   2. Sort (bM, bN, bK) by tile size descending (largest first).
//   3. For the largest dimension, assign sg_dim = min(tile/atom, remaining_budget).
//      remaining_budget /= sg_dim.
//   4. Repeat for the second and third dimensions.
//   5. static_assert if the budget is not fully consumed (subgroups don't evenly tile).
//
// Usage:
//   using SgLayout = cute::default_warp_layout_for_tiled_mma<
//       TMM_Op, bM, bN, bK, kWorkGroupSize, kSubGroupSize, kNumProducerSGs>;
//   auto tiled_mma = make_tiled_mma(MMA_Atom<TMM_Op>{}, typename SgLayout::type{},
//                                   Tile<Int<bM>, Int<bN>, Int<bK>>{});
// ============================================================================
template <class MMA_Op, int bM, int bN, int bK,
          int WorkgroupSize, int SubgroupSize = 32, int NumProducerSGs = 0>
struct default_warp_layout_for_tiled_mma {
  static_assert(WorkgroupSize > 0, "WorkgroupSize must be positive");
  static_assert(SubgroupSize > 0, "SubgroupSize must be positive");
  static_assert(WorkgroupSize % SubgroupSize == 0,
      "WorkgroupSize must be evenly divisible by SubgroupSize");
  static_assert(bM > 0 && bN > 0 && bK > 0,
      "Tile dimensions bM, bN, bK must all be positive");
  // Validate that MMA_Op exposes the required static members (M, N_val, K)
  static_assert(MMA_Op::M > 0 && MMA_Op::N_val > 0 && MMA_Op::K > 0,
      "MMA_Op must provide positive static constexpr members M, N_val, and K");

  static constexpr int num_consumer_sgs = WorkgroupSize / SubgroupSize - NumProducerSGs;

  static constexpr int AtomM = MMA_Op::M;
  static constexpr int AtomN = MMA_Op::N_val;
  static constexpr int AtomK = MMA_Op::K;

  static_assert(num_consumer_sgs > 0,
      "num_consumer_sgs must be positive — check WorkgroupSize, SubgroupSize, NumProducerSGs");
  static_assert(bM % AtomM == 0, "bM must be divisible by atom M dimension");
  static_assert(bN % AtomN == 0, "bN must be divisible by atom N dimension");
  static_assert(bK % AtomK == 0, "bK must be divisible by atom K dimension");

  // Number of atoms per tile dimension
  static constexpr int t_M = bM / AtomM;
  static constexpr int t_N = bN / AtomN;
  static constexpr int t_K = bK / AtomK;

  // Check of any uneven subgroup atom distribution //
  static constexpr int total_Atoms = (t_M * t_N * t_K);

  static_assert((total_Atoms % num_consumer_sgs) == 0,
      "Total atoms of the tile must be divisible by number of consumer SGs");

  // ---- Sort (bM, bN, bK) descending to find processing order ----
  // p0 = index of the largest tile dimension (0=M, 1=N, 2=K).
  // Tie-breaking priority for p0: M > N > K (lower index wins on equal tile values).
  static constexpr int p0 =
      (bM >= bN && bM >= bK) ? 0 :
      (bN >= bM && bN >= bK) ? 1 : 2;

  // Candidates for p2 (smallest): the two indices other than p0
  static constexpr int p2_ca = (p0 == 0) ? 1 : 0;
  static constexpr int p2_cb = (p0 == 2) ? 1 : 2;

  // Tile values for the two p2 candidates
  static constexpr int b_p2_ca = (p2_ca == 0) ? bM : (p2_ca == 1) ? bN : bK;
  static constexpr int b_p2_cb = (p2_cb == 0) ? bM : (p2_cb == 1) ? bN : bK;

  // Tie-breaking priority for p2: K > N > M (higher dimension index wins on equal
  // tile values — i.e. K is preferred over N over M for the smallest slot).
  static constexpr int p2 = (b_p2_ca < b_p2_cb) ? p2_ca
                           : (b_p2_ca > b_p2_cb) ? p2_cb
                           : (p2_ca > p2_cb ? p2_ca : p2_cb);
  static constexpr int p1 = 3 - p0 - p2;  // the remaining index

  // Atom counts per dimension for each sorted position
  static constexpr int t_p0 = (p0 == 0) ? t_M : (p0 == 1) ? t_N : t_K;
  static constexpr int t_p1 = (p1 == 0) ? t_M : (p1 == 1) ? t_N : t_K;
  static constexpr int t_p2 = (p2 == 0) ? t_M : (p2 == 1) ? t_N : t_K;

  // ---- Greedy assignment: assign to largest dimension first ----
  static constexpr int sg_p0 =
      (num_consumer_sgs < t_p0) ? num_consumer_sgs : t_p0;
  static_assert(num_consumer_sgs % sg_p0 == 0,
      "Residue in SG assignment: num_consumer_sgs is not evenly divisible by "
      "the SG count for the largest tile dimension. "
      "Check bM, bN, bK, atom dims, and WorkgroupSize.");

  static constexpr int rem1  = num_consumer_sgs / sg_p0;
  static constexpr int sg_p1 = (rem1 < t_p1) ? rem1 : t_p1;
  static_assert(rem1 % sg_p1 == 0,
      "Residue in SG assignment: remaining SGs are not evenly divisible by "
      "the SG count for the second tile dimension. "
      "Check bM, bN, bK, atom dims, and WorkgroupSize.");

  static constexpr int rem2  = rem1 / sg_p1;
  static constexpr int sg_p2 = rem2;
  static_assert(sg_p2 <= t_p2,
      "Remaining SGs exceed the atom count of the smallest tile dimension. "
      "Check bM, bN, bK, atom dims, and WorkgroupSize.");

  // ---- Map from sorted indices back to M, N, K ----
  static constexpr int sg_M =
      (p0 == 0) ? sg_p0 : (p1 == 0) ? sg_p1 : sg_p2;
  static constexpr int sg_N =
      (p0 == 1) ? sg_p0 : (p1 == 1) ? sg_p1 : sg_p2;
  static constexpr int sg_K =
      (p0 == 2) ? sg_p0 : (p1 == 2) ? sg_p1 : sg_p2;

  static_assert(sg_M * sg_N * sg_K == num_consumer_sgs,
      "Internal error: sg_M * sg_N * sg_K must equal num_consumer_sgs");

  // The layout type to pass as the AtomLayoutMNK argument of make_tiled_mma
  using type = Layout<Shape<Int<sg_M>, Int<sg_N>, Int<sg_K>>>;
};


} // namespace mma_utils
