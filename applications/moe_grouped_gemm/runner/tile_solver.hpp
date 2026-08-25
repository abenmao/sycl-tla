/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief MoE grouped-GEMM tile solver — the pure, device-code-free decision of
           which compiled tile to run for a given (dtype, N, K, M, num_experts).
*/

#ifndef CUTLASS_MOE_TILE_SOLVER_HPP
#define CUTLASS_MOE_TILE_SOLVER_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace cutlass::moe {

// TILE SELECTION (host, device-code-free), no efficiency ranking: pick_best_solution()
// returns GREEDY for every shape, except a small hardcoded set of (dtype, M, N, K) --
// held in a best-solution hash table -- that run DOUBLE BUFFER with a named tile.

// One compiled tile's geometry. `name` is for diagnostics only.
struct TileGeom {
  int blk_m;
  int blk_n;
  int blk_k;
  const char *name;
  bool is_db;         // double-buffer tile (vs greedy)
  bool is_bigk = false;  // mxfp4 greedy variant with doubled tiny-expert K (large-K)
};

// A shape that runs DOUBLE BUFFER instead of greedy, with the EXACT DB tile to
// use (blk_m, blk_n) -- the tile the old efficiency ranker would have picked.
// Match key is (dtype, M, N, K).
struct DbShape { const char *dtype; int m; int n; int k; int db_blk_m; int db_blk_n; };

inline constexpr DbShape kHardcodedDbShapes[] = {
    {"mxfp4_moe",    171, 3584, 1280, 192, 256},
    {"mxfp4_moe",    427, 3584, 1280, 224, 256},
    {"mxfp4_moe",    171, 2560, 3584, 192, 256},
    {"mxfp8_e4m3_moe", 427, 2560, 3584, 224, 256},
    {"mxfp4_moe",    427, 2560, 3584, 224, 256},
};

// (dtype,M,N,K) -> target DB tile. Backed by a function-local static hash table
// built once (on first call) from kHardcodedDbShapes and reused for every launch
// in the process -- the config runner registers all lines and runs them in one
// process, so the static persists across tests. Key packs the four fields into a
// string so no custom tuple hash is needed.
inline const DbShape *lookup_db_shape(const std::string &dtype, int M, int N, int K) {
  static const std::unordered_map<std::string, const DbShape *> table = [] {
    std::unordered_map<std::string, const DbShape *> t;
    for (auto const &s : kHardcodedDbShapes)
      t.emplace(std::string(s.dtype) + '|' + std::to_string(s.m) + '|' +
                    std::to_string(s.n) + '|' + std::to_string(s.k),
                &s);
    return t;
  }();
  auto it = table.find(dtype + '|' + std::to_string(M) + '|' + std::to_string(N) +
                       '|' + std::to_string(K));
  return it == table.end() ? nullptr : it->second;
}

// Index of the first compiled kernel satisfying `match`, or -1. Callers need the
// Index of the first kernel whose geometry satisfies `match`, or -1. `kernels` is
// the registry's entry list (read by reference, no copy) and `geom_of` projects an
// entry to its TileGeom -- the entry type lives in a downstream header, so it's a
// template. Callers need the INDEX; the scan is a handful of kernels, once/launch.
template <class Entry, class GeomOf, class Match>
inline int find_kernel(std::vector<Entry> const &kernels, GeomOf geom_of, Match match) {
  for (size_t i = 0; i < kernels.size(); ++i)
    if (match(geom_of(kernels[i]))) return int(i);
  return -1;
}

// Pick the compiled kernel to run for this problem, as an index into `kernels`
// (the registry's entry list, passed by const ref -- read in place, not copied).
// `geom_of(entry) -> TileGeom const&` projects each entry to its geometry.
//
// Two cases:
//   DOUBLE BUFFER -- only if M is uniform, the caller didn't force greedy, AND the
//                    best-solution hash table has an entry for this exact
//                    (dtype,M,N,K). We then want the DB kernel with that entry's
//                    tile geometry.
//   GREEDY        -- every other shape. The ONE greedy kernel handles both
//                    uniform and dynamic M (chosen at runtime), so M-mode does
//                    not pick a variant; only the mxfp4 BigK tile choice does.
// In each case we look for the ideal kernel; if it wasn't compiled into this
// binary we fall back to one of the same family so we still run something.
// `dynamic_m` (M varies across experts) is decided once upstream, where the M
// list is built, and passed in -- not re-derived here. It still gates DB (DB is
// uniform-M only). `M` is the uniform per-expert M, only meaningful (and only
// used) when !dynamic_m.
template <class Entry, class GeomOf>
inline int pick_best_solution(std::vector<Entry> const &kernels, GeomOf geom_of,
                              const std::string &dtype, int N, int K, int M,
                              bool dynamic_m, bool force_greedy = false) {
  if (kernels.empty()) {
    // No kernels registered for this dtype -- returning 0 indexes an empty list,
    // so the caller must not dereference it. Should never happen (select_tile
    // already rejects empty registries), but flag it if it does.
    std::cerr << "[MoE tile-select] ERROR: no compiled kernels for dtype='" << dtype
              << "' (N=" << N << " K=" << K << ") -- cannot select a tile.\n";
    return 0;
  }

  // Does the best-solution table say this shape should use double buffer?
  // (DB is uniform-M only, and a forced-greedy override skips the table.)
  const DbShape *db =
      (force_greedy || dynamic_m) ? nullptr : lookup_db_shape(dtype, M, N, K);

  int chosen;
  if (db) {
    chosen = find_kernel(kernels, geom_of, [&](TileGeom const &k) {  // named DB tile
      return k.is_db && k.blk_m == db->db_blk_m && k.blk_n == db->db_blk_n;
    });
    if (chosen < 0)                                         // fallback: any DB tile
      chosen = find_kernel(kernels, geom_of, [](TileGeom const &k) { return k.is_db; });
  } else {
    // The ONE greedy kernel handles both uniform and dynamic M at runtime, so
    // M-mode no longer selects a variant. mxfp4 at large K (K>1536) uses the BigK
    // greedy variant (doubled tiny-expert K); every other greedy shape uses the
    // base (non-BigK) variant.
    const bool want_bigk = (dtype == "mxfp4_moe") && (K > 1536);
    chosen = find_kernel(kernels, geom_of, [&](TileGeom const &k) {  // greedy + BigK
      return !k.is_db && k.is_bigk == want_bigk;
    });
    if (chosen < 0)  // fallback: any greedy tile, ignore BigK (variant may not be compiled)
      chosen = find_kernel(kernels, geom_of,
                           [](TileGeom const &k) { return !k.is_db; });
  }
  // Unreachable unless the registry for this dtype is misconfigured (e.g. only DB
  // kernels compiled in but a greedy one is needed, or vice versa) -- both the
  // ideal match and the same-family fallback missed. Warn and run kernels[0]
  // rather than crash, but this shape's result will likely be wrong.
  if (chosen < 0) {
    std::cerr << "[MoE tile-select] WARNING: no kernel matched dtype='" << dtype
              << "' M=" << M << " N=" << N << " K=" << K
              << " dynamic_m=" << (dynamic_m ? 1 : 0) << " (db=" << (db ? 1 : 0)
              << ", " << kernels.size() << " registered) -- falling back to "
              << "kernels[0]; result may be wrong.\n";
    chosen = 0;
  }

  #ifdef MOE_DEBUG_PRINT
  TileGeom const &g = geom_of(kernels[chosen]);
  std::cerr << "[MoE tile-select] dtype='" << dtype << "' M=" << M
            << " N=" << N << " K=" << K << " dynamic_m=" << (dynamic_m ? 1 : 0)
            << " -> ";
  if (g.is_db) {
    std::cerr << "DOUBLE-BUFFER tile " << g.blk_m << "x" << g.blk_n << "x" << g.blk_k;
  } else {
    // Greedy picks its M-tile per expert on-device -- no host-side tile to report.
    // The ONE greedy kernel handles both M-modes at runtime.
    std::cerr << "GREEDY (" << (dynamic_m ? "dynamic-M" : "uniform-M")
              << ", per-expert split chosen in kernel"
              << (g.is_bigk ? ", BigK" : "") << ")";
  }
  std::cerr << " (force_greedy=" << (force_greedy ? "1" : "0") << ")\n";
  #endif
  return chosen;
}

}  // namespace cutlass::moe

#endif  // CUTLASS_MOE_TILE_SOLVER_HPP
