/***************************************************************************************************
 * Copyright (c) 2025 Intel Corporation, All rights reserved.
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

#include "xe4_fmha_fwd_runner.hpp"

struct ProblemConfig {
    using ElementInputQ = bf16;       // dtype of Q
    using ElementInputKV = bf16;      // dtype of K and V
    using ElementS = bf16;           // dtype of S
    using ElementP = ElementInputKV;  // dtype of P
    using ElementAccumulator = float; // dtype of accum
    using ElementOutput = bf16;      // dtype of output
};

struct ProblemConfig_F32_S {
    using ElementInputQ = bf16;       // dtype of Q
    using ElementInputKV = bf16;      // dtype of K and V
    using ElementS = float;           // dtype of S
    using ElementP = ElementInputKV;  // dtype of P
    using ElementAccumulator = float; // dtype of accum
    using ElementOutput = bf16;      // dtype of output
};

int main(int argc, const char **argv) {
  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  // FMHAConfig<Shape<_128 /*Q blk*/, _128 /*v head dim*/, _512 /*KV blk*/, _128 /*qk head dim*/>, 16 /*NumSoftmaxWarps*/, 16 /*NumThreadPerRow*/, 2, 2, false/*IsPersistent*/>::run<ProblemConfig>(options);
  FMHAConfig<Shape<_128 /*Q blk*/, _128 /*v head dim*/, _512 /*KV blk*/, _128 /*qk head dim*/>, 16 /*NumSoftmaxWarps*/, 16 /*NumThreadPerRow*/, 1, 1, true/*IsPersistent*/>::run<ProblemConfig>(options);

  // FMHAConfig<Shape<_128 /*Q blk*/, _128 /*v head dim*/, _256 /*KV blk*/, _128 /*qk head dim*/>, 16 /*NumSoftmaxWarps*/, 8 /*NumThreadPerRow*/>::run<ProblemConfig>(options);
  // FMHAConfig<Shape<_128 /*Q blk*/, _128 /*v head dim*/, _128 /*KV blk*/, _128 /*qk head dim*/>, 16 /*NumSoftmaxWarps*/, 8 /*NumThreadPerRow*/>::run<ProblemConfig>(options);

  // FMHAConfig<Shape<_128 /*Q blk*/, _128 /*v head dim*/, _256 /*KV blk*/, _128 /*qk head dim*/>, 16 /*NumSoftmaxWarps*/, 8 /*NumThreadPerRow*/, 2, 2, true/*IsPersistent*/>::run<ProblemConfig>(options); 
  // FMHAConfig<Shape<_128 /*Q blk*/, _128 /*v head dim*/, _256 /*KV blk*/, _128 /*qk head dim*/>, 16 /*NumSoftmaxWarps*/, 16 /*NumThreadPerRow*/, 2, 2, true/*IsPersistent*/>::run<ProblemConfig_F32_S>(options);


  return 0;
}
