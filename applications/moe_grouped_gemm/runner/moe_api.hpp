/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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

/*! \file
    \brief Thin, device-code-free interface between the Google-Benchmark harness
           and the MoE grouped-GEMM kernel launch.
*/

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace moe_bench {

// Verification toggle passed across the TU boundary (device / host check).
enum VerifyKind { kVerifyNone = 0, kVerifyDevice = 1, kVerifyHost = 2 };

// Single entry point (defined in moe_api.cpp). The client passes a host-side
// VendorTensorMapping<ElementA, float, ElementD> type-erased, with every device
// buffer (A/B/D, packed scale surfaces, device copy of the per-expert counts)
// already allocated and filled. Selects the best tile, optionally verifies, runs
// one timed kernel, returns elapsed ms (-1.0 on failure, *error set). The launch
// path allocates no device memory itself; --verify=device transiently allocates a
// reference buffer (and FP32 A/B copies for scaled dtypes).
//   dtype        : the .in line's first token (selects the tile candidate list)
//   force_greedy : from the .in override_use_greedy_always flag; when true always
//                  picks greedy (ignores the hardcoded DB set).
double launch_moe(const void *vendor_tm, const char *dtype, int verify,
                  std::string *error, bool force_greedy = false);

} // namespace moe_bench
