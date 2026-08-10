/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Tile registry (dtype -> compiled tiles, + pick_tile lookup) and the entry
           points that run a client-built VendorTensorMapping (moe_run /
           moe_run_double_buffer). No device memory is owned or touched here: the
           client allocates, fills and uploads every operand — A/B/D, the padded
           scale surfaces (see scale_surface_geom / pack_moe_scales in
           runner/moe_types.hpp), and the device copy of the per-expert counts, and
           validates the finished mapping (moe_validate_mapping) before the launch.
           For the benchmark that all happens in MoEBenchmarkRunner::run's
           build_inputs.
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
// run function), keyed by dtype tag; launch_moe() does pick_tile() + dispatch.
// The run function is type-erased to double, so no cute/Config type crosses into
// this header.

// Which impl the tile's run function calls.
enum class KernelKind {
  Regular,      // moe_run_impl<Config>              — variable-M, any dtype
  DoubleBuffer, // moe_run_impl_double_buffer<Config> — uniform-M
};

// Per-tile run function: runs the tile's kernel, returns elapsed ms. The exact
// Config (SGLayout, TileShape, element types) is baked in at registration time —
// it cannot be recovered from the dtype string alone.
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

// Registry lookup + pick_tile() for `dtype` and this problem shape. Returns the
// chosen TileEntry, or nullptr if the dtype has no registered tiles (*error set).
inline const TileEntry *select_tile(const char *dtype, int N, int K,
                                     std::vector<int> const &M_per_expert,
                                     int num_experts, std::string *error) {
  auto &reg = moe_tile_registry();
  auto it = reg.find(dtype);
  if (it == reg.end() || it->second.empty()) {
    if (error)
      *error = std::string("no registered tile dtype named '") + dtype + "'";
    return nullptr;
  }
  auto const &entries = it->second;

  std::vector<TileGeom> geoms;
  geoms.reserve(entries.size());
  for (auto const &e : entries)
    geoms.push_back(e.geom);

  const int idx = pick_tile(geoms, dtype, N, K, M_per_expert, num_experts);
  return &entries[idx];
}

// One timed launch of the variable-M kernel from a client-built mapping (elapsed
// ms). Pure launch: every operand — A/B/D, the padded scale surfaces, the
// per-expert count array — is already on the device, allocated and filled by the
// client (for the benchmark, MoEBenchmarkRunner::run's build_inputs). Nothing is
// allocated, padded, copied or re-derived here.
template <class Config, class ElementA, class ElementD, class ElementScaleIn>
double moe_run(const VendorTensorMapping<ElementA, ElementScaleIn, ElementD> &tm) {
  return moe_launch_timed<Config>(tm);
}

// Same through the double-buffer kernel, which requires uniform M across experts
// (throws otherwise) and takes it as a scalar. The client's scale surfaces are
// already in the layout this kernel reads, so no repack is needed.
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
