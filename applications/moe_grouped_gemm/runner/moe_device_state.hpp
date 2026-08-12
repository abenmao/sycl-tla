/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Tile registry (dtype -> compiled tiles, + pick_best_solution lookup) and
           the entry points that run a client-built VendorTensorMapping
           (moe_run /
           moe_run_double_buffer). No device memory is owned or touched here: the
           client allocates, fills, uploads every operand — A/B/D, the padded
           scale surfaces (scale_surface_geom / pack_moe_scales in
           runner/moe_types.hpp), the device per-expert counts — and validates
           the mapping (moe_validate_mapping) before launch. For the benchmark
           that happens in MoEBenchmarkRunner::run's build_inputs.
*/

#pragma once

#include "moe_grouped_gemm/runner/moe_gemm_runner.hpp"
#include "moe_grouped_gemm/runner/tile_solver.hpp"

#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace cutlass::moe {

// Tile registry. moe_api.cpp registers every compiled tile (geometry + kind +
// run fn), keyed by dtype tag; launch_moe() does the tile-select + dispatch. The
// run fn is type-erased to double, so no cute/Config type crosses into this
// header.

// Which impl the tile's run function calls.
enum class KernelKind {
  Greedy,       // moe_run_impl_greedy<Config>        — greedy split, uniform + dynamic M
  DoubleBuffer, // moe_run_impl_double_buffer<Config> — uniform-M
};

// Per-tile run fn: runs the tile's kernel, returns elapsed ms. The exact Config
// (SGLayout, TileShape, element types) is baked in at registration — it can't be
// recovered from the dtype string alone.
using TileRunFn =
    double (*)(const void *vendor_tm, int verify, std::string *error);

struct TileEntry {
  TileGeom   geom;
  KernelKind kind;
  TileRunFn  run;
};

inline std::map<std::string, std::vector<TileEntry>> &moe_tile_registry() {
  static std::map<std::string, std::vector<TileEntry>> r;
  return r;
}

// Called at static-init time by moe_api.cpp (one per compiled tile).
inline void moe_register_tile(const char *dtype, TileGeom geom,
                              KernelKind kind, TileRunFn run) {
  moe_tile_registry()[dtype].push_back({geom, kind, run});
}

// Registry lookup + pick_best_solution() for `dtype` and this problem shape.
// `M`/`dynamic_m` are decided once by the caller (where the M list is built) and
// passed in. Returns the chosen TileEntry, or nullptr if the dtype has no
// registered tiles (sets *error).
inline const TileEntry *select_tile(const char *dtype, int N, int K, int M,
                                     bool dynamic_m, std::string *error,
                                     bool force_greedy = false) {
  auto &reg = moe_tile_registry();
  auto it = reg.find(dtype);
  if (it == reg.end() || it->second.empty()) {
    if (error)
      *error = std::string("no registered tile dtype named '") + dtype + "'";
    return nullptr;
  }
  auto const &entries = it->second;

  // Read the entries in place; project each to its geom (no copy of the list).
  const int idx = pick_best_solution(
      entries, [](TileEntry const &e) -> TileGeom const & { return e.geom; },
      dtype, N, K, M, dynamic_m, force_greedy);
  return &entries[idx];
}

// Run through the double-buffer kernel, which requires uniform M across experts
// (throws otherwise) and takes it as a scalar. The client's scale surfaces are
// already in the layout this kernel reads, so no repack.
template <class Config, class ElementA, class ElementD, class ElementScaleIn>
double moe_run_double_buffer(
    const VendorTensorMapping<ElementA, ElementScaleIn, ElementD> &tm) {
  const int uniform_m = tm.num_experts > 0 ? tm.experts_token_count[0] : 0;

  if constexpr (Config::scale_kind == ScaleKind::Plain)
    return moe_launch_timed_double_buffer<Config>(tm, uniform_m);
  else
    return moe_launch_timed_double_buffer_scaled<Config>(tm, uniform_m);
}

} // namespace cutlass::moe
