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
#include <vector>

namespace cutlass::moe {

// TILE AUTO-SELECTION (host, device-code-free). pick_tile() ranks the compiled
// tile GEOMETRIES for a dtype and returns the winning index. Policy (DB-first):
//   1. Best double-buffer tile = highest total efficiency (wave*tile) among DB
//      tiles whose count (M/M_tile)*(N/N_tile) is EVEN (DB kernel requirement).
//   2. Best regular tile = highest total efficiency among single-buffer tiles.
//   3. Pick DB unless the regular best is >= 6% more efficient (total_eff).

// One compiled tile's geometry. `name` is for diagnostics only.
struct TileGeom {
  int blk_m;
  int blk_n;
  int blk_k;
  const char *name;
  bool is_db; // double-buffer tile (vs regular single-buffer)
};

enum class TileDtypeModel { kBf16, kFp8, kMxFp8, kMxFp4, kUnknown };

// Map a DTYPE tag to its metrics dtype.
inline TileDtypeModel tile_dtype_model_from_tag(const std::string &dtype) {
  if (dtype.rfind("fp8_tensor", 0) == 0) return TileDtypeModel::kFp8;
  if (dtype.rfind("mxfp8", 0) == 0)      return TileDtypeModel::kMxFp8;
  if (dtype.rfind("mxfp4", 0) == 0)      return TileDtypeModel::kMxFp4;
  if (dtype.rfind("bf16", 0) == 0)       return TileDtypeModel::kBf16;
  return TileDtypeModel::kUnknown;
}

struct TileSkuParams {
  int k_atom;
  double elem_bytes;
  bool has_scale;
  bool is_fp8;
  double a_regs_per_atom;
  double b_regs_per_atom;
  double b_scale_regs_total;
};

inline TileSkuParams tile_sku_params(TileDtypeModel d) {
  switch (d) {
  case TileDtypeModel::kFp8:
    return {64, 1.0, true, true, 8.0 + 8.0 / 64.0, 16.0, 1.0 / 64.0};
  case TileDtypeModel::kMxFp8:
    return {64, 1.0, true, false, 8.0 + 2.0 / 64.0, 16.0 + 2.0 / 64.0, 0.0};
  case TileDtypeModel::kMxFp4:
    return {128, 0.5, true, false, 8.0 + 4.0 / 64.0, 16.0 + 4.0 / 64.0, 0.0};
  case TileDtypeModel::kBf16:
  default:
    return {32, 2.0, false, false, 8.0, 16.0, 0.0};
  }
}

struct TileMetrics { double l1_bw; double l2_bw; double regs; bool valid; };

// L1/L2/regs for a specific (x, y) under layout (0=4x8, 1=8x4).
inline TileMetrics tile_metrics_for(TileSkuParams const &p, int x, int y,
                                    int layout) {
  const double CYCLES = 8.0 / (14.0 / 32.0); // ~18.286 cycles per DPAS
  const double A_DATA = (8.0 * p.k_atom * p.elem_bytes) / CYCLES;
  const double B_DATA = (16.0 * p.k_atom * p.elem_bytes) / CYCLES;
  double A_SCALE = 0.0, B_SCALE_PER_ATOM = 0.0, B_SCALE_TOTAL = 0.0;
  if (p.has_scale) {
    if (p.is_fp8) {
      A_SCALE = 8.0 / CYCLES;       // row-wise A scale (8 rows)
      B_SCALE_TOTAL = 1.0 / CYCLES; // tensor-wise B scale (1 total)
    } else {
      A_SCALE = (8.0 * p.k_atom / 32.0) / CYCLES;
      B_SCALE_PER_ATOM = (16.0 * p.k_atom / 32.0) / CYCLES;
    }
  }

  double a_total, b_total;
  if (p.has_scale && p.is_fp8) {
    a_total = (A_DATA + A_SCALE) * x;
    b_total = B_DATA * y + B_SCALE_TOTAL;
  } else if (p.has_scale) {
    a_total = (A_DATA + A_SCALE) * x;
    b_total = (B_DATA + B_SCALE_PER_ATOM) * y;
  } else {
    a_total = A_DATA * x;
    b_total = B_DATA * y;
  }

  const double xy = double(x) * double(y);
  const double l1 = (a_total + b_total) / xy;
  const double regs =
      p.is_fp8 ? (p.a_regs_per_atom * x + p.b_regs_per_atom * y +
                  p.b_scale_regs_total + 16.0 * xy)
               : (p.a_regs_per_atom * x + p.b_regs_per_atom * y + 16.0 * xy);

  double l2;
  if (layout == 0) { // 4x8 : L2 = A_COST/y + 2*B_COST/x
    if (p.has_scale && p.is_fp8)
      l2 = (A_DATA + A_SCALE) / y + 2.0 * (B_DATA / x + B_SCALE_TOTAL / xy);
    else if (p.has_scale)
      l2 = (A_DATA + A_SCALE) / y + 2.0 * (B_DATA + B_SCALE_PER_ATOM) / x;
    else
      l2 = A_DATA / y + 2.0 * B_DATA / x;
  } else { // 8x4 : L2 = 2*A_COST/x + B_COST/y
    if (p.has_scale && p.is_fp8)
      l2 = 2.0 * (A_DATA + A_SCALE) / x + (B_DATA / y + B_SCALE_TOTAL / xy);
    else if (p.has_scale)
      l2 = 2.0 * (A_DATA + A_SCALE) / x + (B_DATA + B_SCALE_PER_ATOM) / y;
    else
      l2 = 2.0 * A_DATA / x + B_DATA / y;
  }

  const double MAX_REGS = 400.0, MAX_L1 = 64.0, MAX_L2 = 64.0;
  const bool valid = (l1 <= MAX_L1) && (regs <= MAX_REGS) && (l2 <= MAX_L2);
  return {l1, l2, regs, valid};
}

// Invert (blk_m, blk_n) to (x, y) for whichever of the two layouts produce
// integer reuse factors; keep the constraint-valid variant with lower (regs,L2,L1).
inline TileMetrics compute_tile_metrics(TileDtypeModel d, int blk_m, int blk_n) {
  const TileSkuParams p = tile_sku_params(d);
  bool have = false, best_valid = false;
  TileMetrics best{0.0, 0.0, 0.0, false};
  auto consider = [&](int layout, int x, int y) {
    if (x < 1 || y < 1)
      return;
    const TileMetrics m = tile_metrics_for(p, x, y, layout);
    const bool better =
        !have || (m.valid != best_valid
                      ? m.valid
                      : std::make_tuple(m.regs, m.l2_bw, m.l1_bw) <
                            std::make_tuple(best.regs, best.l2_bw, best.l1_bw));
    if (better) { best = m; best_valid = m.valid; have = true; }
  };
  if (blk_m % 32 == 0 && blk_n % 128 == 0) consider(0, blk_m / 32, blk_n / 128);
  if (blk_m % 64 == 0 && blk_n % 64 == 0)  consider(1, blk_m / 64, blk_n / 64);
  return best;
}

// Per-candidate metrics for one problem shape.
struct TileRow {
  double total_eff;
  int64_t mn_tiles;
  int64_t pad;
  double l1;
  double l2;
  bool valid;
  bool even;
};

struct BasketPick {
  int index = -1;
  double total_eff = 0.0;
};

// Best tile in `pool`: highest total_eff, ties broken by the weighted model
// (65*eff + 10*bw + 15*tile + 10*pad), then lowest index.
inline BasketPick rank_basket(std::vector<TileRow> const &rows,
                              std::vector<size_t> const &pool) {
  if (pool.empty())
    return {};
  double max_l2 = 0.0, max_l1 = 0.0;
  int64_t max_mn = 0, max_pad = 0;
  for (size_t i : pool) {
    max_l2 = std::max(max_l2, rows[i].l2);
    max_l1 = std::max(max_l1, rows[i].l1);
    max_mn = std::max(max_mn, rows[i].mn_tiles);
    max_pad = std::max(max_pad, rows[i].pad);
  }
  constexpr double W_EFF = 65.0, W_BW = 10.0, W_TILE = 15.0, W_PAD = 10.0;
  constexpr double BW_L2 = 0.7, BW_L1 = 0.3;
  const double max_combined_bw = BW_L2 * max_l2 + BW_L1 * max_l1;
  constexpr double kEffEps = 1e-9;

  BasketPick pick{-1, -1.0};
  double best_eff = -1.0, best_score = -1.0;
  for (size_t i : pool) {
    const auto &r = rows[i];
    const double combined_bw = BW_L2 * r.l2 + BW_L1 * r.l1;
    const double bw_score =
        max_combined_bw > 0.0 ? 1.0 - (combined_bw / max_combined_bw) : 1.0;
    const double tile_score =
        max_mn > 0 ? 1.0 - (double(r.mn_tiles) / max_mn) : 1.0;
    const double pad_score =
        max_pad > 0 ? 1.0 - (double(r.pad) / max_pad) : 1.0;
    const double score = W_EFF * r.total_eff + W_BW * bw_score +
                         W_TILE * tile_score + W_PAD * pad_score;
    const bool better =
        pick.index < 0 || (r.total_eff > best_eff + kEffEps) ||
        (r.total_eff > best_eff - kEffEps && score > best_score);
    if (better) {
      best_eff = r.total_eff;
      best_score = score;
      pick = {static_cast<int>(i), r.total_eff};
    }
  }
  return pick;
}

// Choose the best compiled tile among `cands` for this problem. Returns an index
// into `cands` (never negative when `cands` is non-empty).
inline int pick_tile(std::vector<TileGeom> const &cands, const std::string &dtype,
                     int N, int /*K*/, std::vector<int> const &M_per_expert,
                     int num_experts) {
  if (cands.size() <= 1)
    return 0;

  int64_t m_total = 0;
  for (int m : M_per_expert)
    m_total += m;
  const int m_rep =
      num_experts > 0 ? static_cast<int>(m_total / num_experts) : 0;

  // Small-M exception: pick the smallest compiled tile (min area; tie: smaller
  // blk_m, then blk_n).
  if (m_rep <= 96) {
    int best = 0;
    int64_t best_area = std::numeric_limits<int64_t>::max();
    for (size_t i = 0; i < cands.size(); ++i) {
      const int64_t area = int64_t(cands[i].blk_m) * cands[i].blk_n;
      const bool tie_smaller =
          area == best_area &&
          std::make_pair(cands[i].blk_m, cands[i].blk_n) <
              std::make_pair(cands[best].blk_m, cands[best].blk_n);
      if (area < best_area || tie_smaller) {
        best_area = area;
        best = static_cast<int>(i);
      }
    }
    return best;
  }

  const TileDtypeModel d = tile_dtype_model_from_tag(dtype);
  std::vector<TileRow> rows(cands.size());
  for (size_t i = 0; i < cands.size(); ++i) {
    const auto &c = cands[i];
    const int tiles_m = (m_rep + c.blk_m - 1) / c.blk_m;
    const int tiles_n = (N + c.blk_n - 1) / c.blk_n;
    const int64_t covered_m = int64_t(tiles_m) * c.blk_m;
    const int64_t covered_n = int64_t(tiles_n) * c.blk_n;
    const double tile_eff =
        double(int64_t(m_rep) * N) / double(covered_m * covered_n);
    const int64_t mn_tiles = int64_t(tiles_m) * tiles_n;
    const int64_t total_tiles = mn_tiles * num_experts;
    double wave_eff;
    if (total_tiles % 32 == 0) {
      wave_eff = 1.0;
    } else {
      const double waves = double(total_tiles) / 32.0;
      wave_eff = waves / std::ceil(waves);
    }
    const TileMetrics tm = compute_tile_metrics(d, c.blk_m, c.blk_n);
    rows[i] = {tile_eff * wave_eff, mn_tiles,
               (covered_m - m_rep) + (covered_n - N), tm.l1_bw, tm.l2_bw,
               tm.valid, (mn_tiles % 2) == 0};
  }

  std::vector<size_t> db_pool, reg_pool;
  for (size_t i = 0; i < rows.size(); ++i) {
    if (!rows[i].valid)
      continue;
    if (cands[i].is_db) {
      if (rows[i].even)
        db_pool.push_back(i);
    } else {
      reg_pool.push_back(i);
    }
  }
  if (db_pool.empty())
    for (size_t i = 0; i < rows.size(); ++i)
      if (cands[i].is_db && rows[i].even)
        db_pool.push_back(i);
  if (reg_pool.empty())
    for (size_t i = 0; i < rows.size(); ++i)
      if (!cands[i].is_db)
        reg_pool.push_back(i);

  const BasketPick db = rank_basket(rows, db_pool);
  const BasketPick reg = rank_basket(rows, reg_pool);

  constexpr double kSingleBufferMargin = 1.06; // single-buffer must beat DB by >= 6%
  int chosen;
  const char *reason;
  if (db.index < 0) {
    chosen = reg.index; reason = "no eligible DB tile -> regular";
  } else if (reg.index < 0) {
    chosen = db.index;  reason = "no regular tile -> double-buffer";
  } else if (reg.total_eff >= kSingleBufferMargin * db.total_eff) {
    chosen = reg.index; reason = "regular >= 6% better -> regular";
  } else {
    chosen = db.index;  reason = "double-buffer (default; regular < 6% better)";
  }
  chosen = chosen >= 0 ? chosen : 0;

  // Opt-in diagnostics (-DFULL_RUN_TIMING_AND_VERIFY=ON).

  #ifdef FULL_RUN_TIMING_AND_VERIFY
    std::cerr << "\n[MoE tile-select DEBUG] dtype='" << dtype << "' N=" << N
              << " M/expert=" << m_rep << " experts=" << num_experts << "\n"
              << "  " << cands.size() << " candidates (" << db_pool.size()
              << " DB-eligible, " << reg_pool.size() << " regular):\n"
              << "    kind  " << std::left << std::setw(14) << "tile"
              << std::setw(11) << "total_eff" << std::setw(9) << "mn_tiles"
              << "even valid\n";
    std::cerr << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < cands.size(); ++i) {
      const auto &c = cands[i];
      const auto &r = rows[i];
      const bool is_dbw  = (static_cast<int>(i) == db.index);
      const bool is_regw = (static_cast<int>(i) == reg.index);
      std::ostringstream geo;
      geo << c.blk_m << "x" << c.blk_n << "x" << c.blk_k;
      std::cerr << "    " << std::left << std::setw(6) << (c.is_db ? "DB" : "REG")
                << std::setw(14) << geo.str()
                << std::setw(11) << r.total_eff
                << std::setw(9) << r.mn_tiles
                << std::setw(5) << (r.even ? "y" : "n")
                << std::setw(6) << (r.valid ? "y" : "n")
                << (static_cast<int>(i) == chosen ? " <== CHOSEN" : "")
                << (is_dbw && static_cast<int>(i) != chosen ? " (best DB)" : "")
                << (is_regw && static_cast<int>(i) != chosen ? " (best REG)" : "")
                << "\n";
    }
    std::cerr << "  best DB total_eff=" << (db.index >= 0 ? db.total_eff : 0.0)
              << "  best REG total_eff=" << (reg.index >= 0 ? reg.total_eff : 0.0)
              << "  (margin 1.06)\n"
              << "  decision: " << reason << "\n";
    std::cerr.unsetf(std::ios::floatfield);
  #endif
  return chosen;
}
}  // namespace cutlass::moe

#endif  // CUTLASS_MOE_TILE_SOLVER_HPP
