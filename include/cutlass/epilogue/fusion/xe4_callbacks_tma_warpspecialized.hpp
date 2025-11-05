/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Fusion callbacks specializations for the sm90 TMA warp-specialized (ws) epilogue
*/

#pragma once

#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#if defined(SYCL_INTEL_XE4_TARGET)
#include "cutlass/epilogue/fusion/xe4_visitor_load_tma_warpspecialized.hpp"
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

// D = activation(acc)
template<
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Xe4EltAct =
  Sm90EVT<
    Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>, // activation(acc)
    Sm90AccFetch // acc
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::EltAct<ActivationFn, ElementOutput, ElementCompute, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Xe4EltAct<ActivationFn, ElementOutput, ElementCompute, RoundStyle> {

  using Impl = Xe4EltAct<ActivationFn, ElementOutput, ElementCompute, RoundStyle>;
  using Operation = fusion::EltAct<ActivationFn, ElementOutput, ElementCompute, RoundStyle>;

  struct Arguments {
    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {                     // unary op : activation(acc)
          {},                     // leaf args : acc
          activation              // unary args: activation
        };                   // end unary op
    }
  };
  // Ctor inheritance
  using Impl::Impl;
};

// D = activation(acc) * C
template<
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Xe4EltActMul =
  Sm90EVT<
    Sm90Compute<multiplies, ElementOutput, ElementCompute, RoundStyle>, // activation(acc) * C
      Sm90EVT<
        Sm90Compute<ActivationFn, ElementCompute, ElementCompute, RoundStyle>, // activation(acc)
        Sm90AccFetch // acc
      >,
      Sm90SrcFetch<ElementSource> // C
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementSource,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::EltActMul<ActivationFn, ElementOutput, ElementCompute, ElementSource, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Xe4EltActMul<ActivationFn, ElementOutput, ElementCompute, ElementSource, RoundStyle> {

  using Impl = Xe4EltActMul<ActivationFn, ElementOutput, ElementCompute, ElementSource, RoundStyle>;
  using Operation = fusion::EltActMul<ActivationFn, ElementOutput, ElementCompute, ElementSource, RoundStyle>;

  struct Arguments {
    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {   // binary op: activation(acc) * C
          {                     // unary op : activation(acc)
            {},                     // leaf args : acc
            activation              // unary args: activation
          },                    // end unary op
          {},                   // leaf args : C
          {}                    // binary args : multiplies
        };  // end binary op
    }
  };
  // Ctor inheritance
  using Impl::Impl;
};

// D = activation(acc) + C
template<
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementSource = ElementOutput,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Xe4EltActAdd =
  Sm90EVT<
    Sm90Compute<plus, ElementOutput, ElementCompute, RoundStyle>, // activation(acc) + C
      Sm90EVT<
        Sm90Compute<ActivationFn, ElementCompute, ElementCompute, RoundStyle>, // activation(acc)
        Sm90AccFetch // acc
      >,
      Sm90SrcFetch<ElementSource> // C
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  template <class> class ActivationFn,
  class ElementOutput,
  class ElementCompute,
  class ElementSource,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::EltActAdd<ActivationFn, ElementOutput, ElementCompute, ElementSource, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Xe4EltActAdd<ActivationFn, ElementOutput, ElementCompute, ElementSource, RoundStyle> {

  using Impl = Xe4EltActAdd<ActivationFn, ElementOutput, ElementCompute, ElementSource, RoundStyle>;
  using Operation = fusion::EltActAdd<ActivationFn, ElementOutput, ElementCompute, ElementSource, RoundStyle>;

  struct Arguments {
    using ActivationArguments = typename Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>::Arguments;
    ActivationArguments activation = ActivationArguments();

    operator typename Impl::Arguments() const {
      return
        {   // binary op: activation(acc) + C
          {                     // unary op : activation(acc)
            {},                     // leaf args : acc
            activation              // unary args: activation
          },                    // end unary op
          {},                   // leaf args : C
          {}                    // binary args : plus
        };  // end binary op
    }
  };
  // Ctor inheritance
  using Impl::Impl;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

// D = acc + per-row bias
template<
  class CtaTileShapeMNK,
  class ElementOutput,
  class ElementCompute,
  class ElementBias = ElementOutput,
  int AlignmentBias = 128 / sizeof_bits_v<ElementBias>,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
using Xe4PerColBias =
  Sm90EVT<Sm90Compute<plus, ElementOutput, ElementCompute, RoundStyle>, // acc + bias
    Sm90AccFetch, // acc
    Xe4RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0,_1,int64_t>, AlignmentBias> // bias
  >;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  class ElementBias,
  int AlignmentBias,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    fusion::PerColBias<ElementOutput, ElementCompute, ElementBias, AlignmentBias, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile
> : Xe4PerColBias<
      CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias, AlignmentBias, RoundStyle> {
  using Impl = Xe4PerColBias<
    CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias, AlignmentBias, RoundStyle>;
  using Operation = fusion::PerColBias<
    ElementOutput, ElementCompute, ElementBias, AlignmentBias, RoundStyle>;

  struct Arguments {
    using StrideBias = Stride<_0,_1,int64_t>;
    ElementBias const* bias_ptr = nullptr;
    StrideBias dBias = {};

    operator typename Impl::Arguments() const {
      return
        {     // binary op : acc + bias
          {},                     // leaf args : acc
          {bias_ptr, ElementBias(0), dBias}, // leaf args : bias
          {} // binary args : plus
        };   // end binary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
