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
#pragma once

#include "cutlass/cutlass.h"

namespace cutlass::flash_attention::kernel {

using namespace cute;

struct XeFlashIndividualTileScheduler {

  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeFlashIndividualTileScheduler(Params const& params) : params(params) {}

  template<class ProblemSize, class TileShape>
  static Params to_underlying_arguments(ProblemSize const& problem_size, TileShape const& tile_shape) {
    //problem_size = [batch, num_heads, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo]
    dim3 grid(size(ceil_div(shape<5>(problem_size), shape<1>(tile_shape))),
              size(ceil_div(shape<2>(problem_size), shape<0>(tile_shape))),
              size(shape<0>(problem_size) * shape<1>(problem_size)));
    
    return Params{grid, {shape<1>(problem_size)}};
  }

  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  CUTLASS_DEVICE
  auto get_block_coord() {
    int block_decode = BlockIdxZ();
    int bidh;
    params.divmod_num_heads(block_decode, bidh, block_decode);
    return make_coord(BlockIdxX(), BlockIdxY(), block_decode, bidh);
  }

  CUTLASS_DEVICE
  XeFlashIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }

};
} // namespace cutlass::flash_attention::kernel
