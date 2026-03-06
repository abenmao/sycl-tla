/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2025 INTEL CORPORATION & AFFILIATES. All rights reserved.
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

#if (SYCL_INTEL_TARGET == 40)

#include <cute/atom/copy_traits.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/prefetch.hpp>
#include <cute/numeric/integral_ratio.hpp>
#include <cute/arch/copy_xe4_desc.hpp>
#include <cute/atom/copy_traits_xe4_dma_legacy.hpp>
#include <cute/arch/copy_xe4_adma.hpp>

namespace cute
{
namespace detail {
template <class GmemTmaBasisStrides_, class TmaGmemBasis_, class TmaSwizzle_>
struct AuxTmaParams {
  using GmemStrides  = GmemTmaBasisStrides_;    // Strides for Gmem mode -> Tma coord mode, may be dynamic
  GmemStrides g_stride_;
  using TmaGmemBasis = TmaGmemBasis_;           // Layout for Tma box shape -> Gmem mode(s), always static
  static_assert(is_static<TmaGmemBasis>::value);
  using TmaSwizzle   = TmaSwizzle_;             // Tma swizzle, always Swizzle<B,M,S>
  static_assert(is_static<TmaSwizzle>::value);
};
//
// MAKE_TMA_COPY and related
//


// Custom version of coalesce that greedily combines modes only up to size-256
// Look at each element and the back of the stack (in order of priority)
// back(NewLayout)  get<I>(OldLayout)
//      s0:d0           _1:d1     =>  continue
//      _1:d0           s1:d1     =>  replace_back     s1:d1
//      s0:d0           s1:s0*d0  =>  replace_back  s0*s1:d0   if s0*s1 <= 256
//      s0:d0           s1:d1     =>  append           s1:d1
//
// @pre OldShape and OldStride are flat
template <int I, class OldShape, class OldStride, class NewShape, class NewStride>
CUTE_HOST_DEVICE constexpr
auto
coalesce_256_impl(OldShape const& old_shape, OldStride const& old_stride,
                  NewShape const& new_shape, NewStride const& new_stride)
{
  if constexpr (I == rank_v<OldShape>) {
    // Base case, we're done
    if constexpr (is_constant<1, NewShape>::value) {
      return Layout<_1,_0>{};
    } else {
      return Layout<NewShape,NewStride>{new_shape,new_stride};
    }
  } else if constexpr (is_constant<1, decltype(get<I>(old_shape))>::value) {
    // shape<I>(layout) == _1, skip it and continue
    return coalesce_256_impl<I+1>(old_shape, old_stride, new_shape, new_stride);
  } else if constexpr (is_constant<1, NewShape>::value) {
    // Replace our shape-1 with anything (Can only happen on input new_shape/new_stride)
    return coalesce_256_impl<I+1>(old_shape, old_stride, get<I>(old_shape), get<I>(old_stride));
  } else if constexpr (is_constant<true, decltype(back(new_shape) * back(new_stride) == get<I>(old_stride) &&
                                                  get<I>(old_shape) * back(new_shape) <= Int<256>{})>::value) {
    // Merge modes because the shapes and strides match and the merge is 256 or less
    return coalesce_256_impl<I+1>(old_shape, old_stride,
                                  replace_back(new_shape, get<I>(old_shape) * back(new_shape)),
                                  new_stride);
  } else {
    // Can't replace or merge, so append a new mode
    return coalesce_256_impl<I+1>(old_shape, old_stride,
                                  append(new_shape,  get<I>(old_shape)),
                                  append(new_stride, get<I>(old_stride)));
  }

  CUTE_GCC_UNREACHABLE;
}

// Combine all the modes that are possible to combine
// Does not respect the profile of the layout, but does preserve total size
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
coalesce_256(Layout<Shape,Stride> const& layout)
{
  auto flat_shape  = flatten(layout.shape());
  auto flat_stride = flatten(layout.stride());
  return coalesce_256_impl<1>(flat_shape, flat_stride, get<0>(flat_shape), get<0>(flat_stride));
}

template <class TmaInternalType,
          class GEngine, class GLayout,
          class SShape, class SStride,
          class VShape, class VStride>
CUTE_HOST_DEVICE constexpr
auto
construct_tma_gbasis(Tensor<GEngine,GLayout> const& gtensor,       // The original GMEM Tensor
                     Layout<SShape,SStride>  const& slayout,       // The layout of SMEM
                     Layout<VShape,VStride>  const& cta_v_map)     // smem_idx to hier gmode
{
  //
  // TMA parameter checking
  //

  // CUTE_STATIC_ASSERT_V(product_each(shape(slayout)) == product_each(shape(cta_v_map)),
  //                      "TMA requires CTA_Tile and SLayout top-level shape equivalence.");
  CUTE_STATIC_ASSERT_V(size(slayout) == size(cta_v_map),
                       "TMA requires CTA_Tile and SLayout top-level size equivalence.");

#if 0
  print("gtensor         : "); print(gtensor); print("\n");
  print("slayout         : "); print(slayout); print("\n");
  print("cta_v_map       : "); print(cta_v_map); print("\n");
#endif

  //
  // TMA slayout manipulation
  //

  // Invert the smem to get the largest contiguous vector in the smem layout
  // smem idx -> smem coord
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));

  // Compose with the V-Map to convert smem coord (CTA val idx) to gmem mode
  // smem idx -> gmem mode
  auto sidx2gmode_full = coalesce(composition(cta_v_map, inv_smem_layout));

#if 0
  print("inv_smem_layout : "); print(inv_smem_layout); print("\n");
  print("sidx2gmode_full : "); print(sidx2gmode_full); print("\n");
#endif

  //
  // TMA gtensor truncation
  //

  // Truncate any incompatibilities -- no starting in the middle of gmodes
  auto smem_rank = find_if(stride(sidx2gmode_full), [](auto e) {
    [[maybe_unused]] auto v = basis_value(e);
    return not is_constant<1,decltype(v)>{};
  });
  static_assert(smem_rank > 0, "Could not find a common tile-gmem vectorization. Does the Tile select out major GMEM modes?");

  // Keep only the static-1 basis modes into gmem
  auto sidx2gmode = take<0,smem_rank>(sidx2gmode_full);

#if 0
  print("smem_rank  : "); print(smem_rank); print("\n");
  print("sidx2gmode : "); print(sidx2gmode); print("\n");
#endif

  //
  // TMA gtensor manipulation
  //

  // The smem vector is the same units as gtensor, so compose first and then recast
  // tma_val_idx:gmem_strides
  auto tile_gstride = recast<TmaInternalType>(gtensor.compose(sidx2gmode)).layout();
  // Coalesce modes up to size-256 (the maximum TMA box extent in units of TmaInternalType)
  // tma_box_shape:gmem_strides
  auto tma_gstride  = coalesce_256(tile_gstride);

  // Perform the tiling, recast, and coalesce to the gmem vector again, but with indirections to the gtensor modes
  auto gbasis = make_identity_layout(shape(gtensor));
  auto tile_gbasis_tmp = gbasis.compose(sidx2gmode);

  // Instead of the recast (gbasis doesn't have type info), replace the shape with the already-recasted shape
  // tma_box_shape:gmem_mode
  auto tile_gbasis = make_layout(shape(tile_gstride), stride(tile_gbasis_tmp));

  // "Coalesce" the tile basis into a compatible shape with the tma_gstride
  auto tma_gbasis_tile = tile_gbasis.compose(make_layout(wrap(shape(tma_gstride))));

  // Recast the original tensor for shape/stride inspections
  Tensor gtensor_T = recast<TmaInternalType>(gtensor);

  // Find missing bases that don't appear in tile_gbasis
  auto tile_gbasis_remaining_stride = filter_tuple(flatten(shape (gtensor_T)), flatten(stride(gtensor_T)),
                                                   flatten(stride(gbasis)),
                                                   [&](auto s, auto d, auto e)
  {
    if constexpr (is_constant<1, decltype(s)>::value || is_constant<0, decltype(d)>::value) {
      return cute::tuple<>{};          // If size-1 or stride-0, then don't append
    } else {
      using E = decltype(e);
      auto has_e = any_of(flatten(stride(tma_gbasis_tile)), [] (auto tb) { return tb == E{}; });
      if constexpr (decltype(has_e)::value) {
        return cute::tuple<>{};        // If d was found, then don't append
      } else {
        return cute::tuple<E>(e);      // Else, this is missing so append
      }
    }
  });

  // Append the remaining basis modes that contribute to the TMA with size-1
  auto tile_gbasis_remaining_shape = repeat<rank(tile_gbasis_remaining_stride)>(Int<1>{});
  auto tma_gbasis_full = make_layout(tuple_cat(wrap( shape(tma_gbasis_tile)), wrap(tile_gbasis_remaining_shape )),
                                     tuple_cat(wrap(stride(tma_gbasis_tile)), wrap(tile_gbasis_remaining_stride)));

  // Group the trailing modes to make this max rank-5 -- TMA rank limitation
  // tma_box_shape:gmem_mode
  auto tma_gbasis = group<cute::min(rank(tma_gbasis_full),4),-1>(tma_gbasis_full);

#if 0
  print("tile_gstride : "); print(tile_gstride); print("\n");
  print("tma_gstride  : "); print(tma_gstride); print("\n");
  print("gbasis       : "); print(gbasis); print("\n");
  print("tile_gbasis  : "); print(tma_gbasis_tile); print("\n");
  print("tma_gbasis   : "); print(tma_gbasis); print("\n");
#endif

  return tma_gbasis;
}

template <class GEngine, class GLayout,
          class TmaGmemBasisStride,
          class ShapeT, size_t TmaRank>
CUTE_HOST_DEVICE constexpr
void
fill_tma_gmem_shape_stride(Tensor<GEngine,GLayout>   const& gtensor,           // Gmem Shapes and Strides, in units of TmaInternalType
                           TmaGmemBasisStride        const& tma_gbasis_stride, // Map Tma mode idx -> Gmem mode(s)
                           cute::array<ShapeT,   TmaRank> & gmem_prob_shape,   // Tma Shapes, uint32_t or uin64_t
                           cute::array<uint64_t, TmaRank> & gmem_prob_stride)  // Tma Strides
{
  static_assert(is_tuple<TmaGmemBasisStride>::value);
  static_assert(is_same<uint32_t, ShapeT>::value || is_same<uint64_t, ShapeT>::value);

  using TmaInternalType = typename GEngine::value_type;
  constexpr int tma_rank = decltype(rank(tma_gbasis_stride))::value;
  static_assert(TmaRank >= tma_rank);

  auto gmem_shape  =  shape(gtensor);
  auto gmem_stride = stride(gtensor);
  // Use the indirections in tma_gbasis_stride into gtensor to construct the tma gmem shapes/strides
  for_each(make_seq<tma_rank>{}, [&](auto i) {
    constexpr int tma_i_rank = decltype(rank<i>(tma_gbasis_stride))::value;
    if constexpr (tma_i_rank == 1) {
      // Trivial contribution of this gmem mode to this tma mode
      auto ej = unwrap(get<i>(tma_gbasis_stride));
      gmem_prob_shape[i]  = basis_get(ej, gmem_shape);
      gmem_prob_stride[i] = basis_get(ej, gmem_stride);
    } else {
      // Apply a recurrence to each gmem mode that contributes to this tma mode
      for_each(get<i>(tma_gbasis_stride), [&](auto ej) {
        // Problem shape
        uint64_t shape_j  = basis_get(ej, gmem_shape);
        // Problem stride (in bytes)
        uint64_t stride_j = basis_get(ej, gmem_stride);
        uint64_t old_stride = gmem_prob_stride[i];
        gmem_prob_stride[i] = gcd(gmem_prob_stride[i], stride_j);

        if (gmem_prob_stride[i] != 0) {
          // Recurrence: g_shape = (s_i - 1) * (d_i / gcd_j d_j) + 1
          gmem_prob_shape[i] = (gmem_prob_shape[i]-1) * (old_stride / gmem_prob_stride[i])
                             +            (shape_j-1) * (stride_j   / gmem_prob_stride[i])
                             + 1;
        } else {
          gmem_prob_shape[i] = shape_j;
        }
      });
    }
  });
}

// Overload for an existing Copy_Traits
template <class GEngine, class GLayout,
          class Op, class Bits, class Aux,
          class ShapeT, size_t TmaRank>
CUTE_HOST_DEVICE constexpr
void
fill_tma_gmem_shape_stride(Copy_Traits<Op,Bits,Aux>  const& tma_traits,
                           Tensor<GEngine,GLayout>   const& gtensor,           // Gmem Shapes and Strides, value_type = TmaInternalType
                           cute::array<ShapeT,   TmaRank> & gmem_prob_shape,   // Tma Shapes, uint32_t or uin64_t
                           cute::array<uint64_t, TmaRank> & gmem_prob_stride)  // Tma Strides
{
  return fill_tma_gmem_shape_stride(gtensor, stride(typename Aux::TmaGmemBasis{}),
                                    gmem_prob_shape, gmem_prob_stride);
}

// Use a sidx2gmode to read through the GMEM tensor
//   and construct a TMA Descriptor for the resulting instruction
// At the same time, construct the Tma Tensor's Stride to generate
//   the TMA coordinates that the instruction consumes.
//
template <class TmaInternalType,
          class GEngine, class GLayout,
          class TShape, class TStride,
          int B, int M, int S>
CUTE_HOST_RTC
auto
make_tma_copy_desc(Tensor<GEngine,GLayout> const& gtensor,         // The original GMEM Tensor
                   Layout<TShape,TStride>  const& tma_gbasis,      // TMA mode -> GMEM mode mapping
                   Swizzle<B,M,S>          const& swizzle,         // Swizzle fn on smem_idx
                   uint32_t                       num_multicast)   // The number of CTAs in multicasting
{
  //
  // TMA desc creation
  //

  constexpr int tma_dim = decltype(rank(tma_gbasis))::value;

  //
  // TMA gmem desc info
  //

  // Recast the original tensor for shape/stride inspections
  Tensor gtensor_T = recast<TmaInternalType>(gtensor);

  void* gmem_address = (void*) raw_pointer_cast(gtensor_T.data());
  auto  gmem_layout  = gtensor_T.layout();

  cute::array<uint64_t, 5> gmem_prob_shape  = {1,1,1,1,1};
  cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};

  fill_tma_gmem_shape_stride(gtensor_T, stride(tma_gbasis), gmem_prob_shape, gmem_prob_stride);

  // convert strides to byte strides
  for(uint64_t& stride : gmem_prob_stride) {
    stride = (stride * sizeof_bits_v<TmaInternalType>) / 8;
  }

  //
  // TMA smem desc info
  //

  sycl::marray<uint32_t, tma_dim> smem_box_shape(uint32_t(1));
  sycl::marray<uint32_t, tma_dim> smem_box_stride(uint32_t(1));

  // The smem box is simply given by the sizes of the modes in tma_gbasis
  for_each(make_seq<tma_dim>{}, [&](auto i) {
    smem_box_shape[i] *= size<i>(tma_gbasis);
  });
  // Finally, truncate the tma box by the num_multicast
  for (uint32_t i = tma_dim-1, multicast = num_multicast; multicast > 1; --i) {
    uint32_t new_mult = ceil_div(multicast, smem_box_shape[i]);
    smem_box_shape[i] = ceil_div(smem_box_shape[i], multicast);
    multicast = new_mult;
  }

    //
    // Construct the descriptor
    //

    TmaDescriptor tma_desc{};

    //
    // TMA general info
    //

  #if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)

    CUtensorMapDataType     tma_format      = TMA::to_CUtensorMapDataType<TmaInternalType>();
    CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    // TMA smem swizzle type
    TMA::SmemSwizzleBits swizzle_bits = get_tma_swizzle_bits(swizzle);
    TMA::SmemSwizzleBase swizzle_base = get_tma_swizzle_base(swizzle);
    CUtensorMapSwizzle smem_swizzle = TMA::to_CUtensorMapSwizzle(swizzle_bits, swizzle_base);
    CUresult result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &tma_desc,
        tma_format,
        tma_dim,
        gmem_address,
        gmem_prob_shape.data(),
        gmem_prob_stride.data() + 1,  // gmem_prob_stride[0] implicitly 1
        smem_box_shape.data(),
        smem_box_stride.data(),
        tma_interleave,
        smem_swizzle,
        tma_l2Promotion,
        tma_oobFill);

    if (result != CUDA_SUCCESS) {
      std::cerr << "TMA Desc Addr:   " << &tma_desc
                << "\nformat         " << tma_format
                << "\ndim            " << tma_dim
                << "\ngmem_address   " << gmem_address
                << "\nglobalDim      " << gmem_prob_shape
                << "\nglobalStrides  " << gmem_prob_stride
                << "\nboxDim         " << smem_box_shape
                << "\nelementStrides " << smem_box_stride
                << "\ninterleave     " << tma_interleave
                << "\nswizzle        " << smem_swizzle
                << "\nl2Promotion    " << tma_l2Promotion
                << "\noobFill        " << tma_oobFill << std::endl;
      std::cerr << "Error: Failed to initialize the TMA descriptor " << result << std::endl;
      assert(false);
    }

  #endif // (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)

  auto recast_ratio = cute::trait_ratio(sizeof_bits<typename GEngine::value_type>{},
                                        sizeof_bits<             TmaInternalType>{});

  auto gbasis = make_basis_like(shape(gtensor));

  // Finally, get the inverse permutation of the E<i> bases for the mocked gmem stride
  auto gmem_tma_basis_stride = transform_leaf(gbasis, [&](auto ei) {
    auto si = basis_get(ei,  shape(gmem_layout));
    auto di = basis_get(ei, stride(gmem_layout));
    if constexpr (is_constant<1, decltype(si)>::value || is_constant<0, decltype(di)>::value) {
      return Int<0>{};                  // If size-1 or stride-0, return arithmetic identity -- no contribution to the TMA
    } else {
      auto tma_gmem_basis_stride = stride(tma_gbasis);
      // Find j such that E<i> is in stride<j>(tma_gbasis)
      using EI = decltype(ei);
      [[maybe_unused]] auto j = find_if(tma_gmem_basis_stride, [&](auto tma_stride_j) { return any_of(tma_stride_j, [&](auto dj) { return dj == EI{}; }); });
      if constexpr (decltype(j == rank(tma_gmem_basis_stride))::value) {
        return Int<0>{};               // If not-found, return arithmetic identity -- no contribution to the TMA
      } else
      if constexpr (decltype(j == Int<0>{})::value) {
        auto scale = recast_ratio * basis_get(ei, stride(gtensor));
        return E<j>{} * scale;         // Return TMA Coord basis -- with a recast scale factor
      } else
      if constexpr (decltype(rank<j>(tma_gmem_basis_stride) == Int<1>{})::value) {
        return E<j>{};                 // Return TMA Coord basis -- known scale of Int<1>{}
      } else {
        int32_t scale = ceil_div(int32_t(di * sizeof_bits_v<TmaInternalType> / cute::max(gmem_prob_stride[j], uint64_t{16})), 8);
        return E<j>{} * scale;         // Return TMA Coord basis -- with a dynamic scale factor
      }
    }
  });

#if 0
    print("gmem_tma_basis_stride : "); print(gmem_tma_basis_stride); print("\n");
#endif

  using AuxParams = AuxTmaParams<decltype(gmem_tma_basis_stride),
                                 decltype(tma_gbasis),
                                 decltype(swizzle)>;

  sycl::marray<uint32_t, tma_dim> gmem_shape;
  for_each(make_seq<tma_dim>{}, [&](auto i) {gmem_shape[i] = gmem_prob_shape[i];});
  sycl::marray<uint64_t, tma_dim-1> gmem_stride;
  for_each(make_seq<tma_dim-1>{}, [&](auto i) {gmem_stride[i] = gmem_prob_stride[i+1];});
  auto tma_desc_details = make_tuple(gmem_shape, gmem_stride, smem_box_shape, smem_box_stride);
  return cute::make_tuple(tma_desc_details, AuxParams{gmem_tma_basis_stride});
}

template <class TmaInternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class VShape, class VStride>
CUTE_HOST_RTC
auto
make_tma_copy_atom(CopyOp,
                   Tensor<GEngine,GLayout> const& gtensor,       // Full GMEM Tensor
                   SLayout                 const& slayout,       // CTA Tile of SMEM, potentially swizzled
                   uint32_t                const& num_multicast, // The number of CTAs involved in multicasting
                   Layout<VShape,VStride>  const& cta_v_map)     // V: CTA val idx -> gmem mode
{
  //
  // TMA truncated layout
  //

  auto smem_swizzle = get_swizzle_portion(slayout);
  auto smem_layout  = get_nonswizzle_portion(slayout);

  auto tma_gbasis = detail::construct_tma_gbasis<TmaInternalType>(gtensor, smem_layout, cta_v_map);

  //
  // Construct the TMA Desc and the strides of the TMA Tensor
  //

  auto [tma_desc, aux_params] = detail::make_tma_copy_desc<TmaInternalType>(gtensor,
                                                                            tma_gbasis,
                                                                            smem_swizzle,
                                                                            num_multicast);

  //
  // Construct the Copy_Traits
  //
  auto gmem_ptr = cute::raw_pointer_cast(recast<TmaInternalType>(gtensor).data());
  constexpr int num_bits_per_tma = size(tma_gbasis) * sizeof_bits_v<TmaInternalType>;
  using DmaCache = Xe4DmaCache<decltype(tma_desc), decltype(aux_params), decltype(gmem_ptr)>;
  using Traits = Copy_Traits<Xe4CopyOp<CopyOp>, cute::C<num_bits_per_tma>, DmaCache>;
  using Atom   = Copy_Atom<Traits, typename GEngine::value_type>;

  Traits tma_traits{{tma_desc, aux_params, gmem_ptr}};

#if 0
  print("num_bits_per_tma :  "); print(num_bits_per_tma); print("\n");
  print("g_stride_bases   :  "); print(tma_traits.aux_params_.g_stride_); print("\n");
#endif

  // Return the Copy_Atom
  return Atom{tma_traits};
}

// The "logical TMA tid" is a map from the CTA rank to its logical id
// within the instruction.  It works like a mask or ordering on the
// CTAs.  For non-multicast TMA, all CTAs should map to 0.  For
// multicast TMA of size 4, CTAs will be mapped to {0,1,2,3}.
template <class TmaInternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class TShape, class TStride,
          class VShape, class VStride>
CUTE_HOST_RTC
auto
make_tma_copy_tiled(CopyOp                  const& copy_op,
                    Tensor<GEngine,GLayout> const& gtensor,     // Full GMEM Tensor
                    SLayout                 const& slayout,     // CTA Tile of SMEM
                    Layout<TShape,TStride>  const& cta_t_map,   // T: CTA thr idx -> logical TMA tid
                    Layout<VShape,VStride>  const& cta_v_map)   // V: CTA val idx -> gmem mode
{
  Copy_Atom atom = make_tma_copy_atom<TmaInternalType>(copy_op, gtensor, slayout,
                                                       cosize(cta_t_map), cta_v_map);

  //
  // Construct the TiledCopy
  //

  [[maybe_unused]] auto cta_tiler = product_each(shape(cta_v_map));

  auto num_elems_per_tma = size<1>(typename decltype(atom)::RefLayout{}) / static_value<sizeof_bits<typename GEngine::value_type>>();

  // smem idx -> smem coord
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
  // CTA V -> smem_coord
  auto layout_v = composition(inv_smem_layout, num_elems_per_tma);
  // Scale that up to cover all of the smem_coords
  auto layout_V = tile_to_shape(make_layout(layout_v), size(cta_v_map));
  // CTA T -> smem idx
  auto layout_t = make_layout(cosize(cta_t_map), safe_div(num_elems_per_tma, cosize(cta_t_map)));
  // CTA TID -> smem coord
  auto layout_T = composition(inv_smem_layout, composition(layout_t, cta_t_map));
  // Combine with the T mapping
  [[maybe_unused]] auto layout_TV = make_layout(layout_T, layout_V);

#if 0
  print("cta_tiler : "); print(cta_tiler); print("\n");
  print("layout_v : "); print(layout_v); print("\n");
  print("layout_V : "); print(layout_V); print("\n");
  print("layout_t : "); print(layout_t); print("\n");
  print("layout_T : "); print(layout_T); print("\n");
  print("layout_TV : "); print(layout_TV); print("\n");
#endif

  return TiledCopy<decltype(atom), decltype(layout_TV), decltype(cta_tiler)>{atom};
}

} // end namespace detail

/** Make a CuTe CTA-collective TiledCopy for a TMA operation.
 *
 * @param CopyOp The target copy operation: XE4_ADMA_LOAD, XE4_ADMA_LOAD_MULTICAST, XE4_ADMA_STORE
 * @param gtensor The GMEM Tensor to be involved in the TMA.
 * @param slayout The SMEM Layout to be involved in the TMA.
 * @param cta_tile The CTA-local tile that each CTA will be tiling GMEM with.
 *                 This is often the blk_shape that is used to tile the GMEM for CTAs:
 *                   local_tile(gtensor, blk_shape, blk_coord) -> CTA-local tile of gtensor
 * @param cluster_size When using XE4_ADMA_LOAD_MULTICAST, this can be a (static) power-of-2 <= 16
 *                   defining the multicast size (used to further partition the SMEM)
 *                 Else, static-1
 *
 * This code attempts to maximize the TMA box size. It does this by tracing
 * the SMEM "vector" -- the inverse of the smem layout -- to find the largest
 * contiguous array of smem that can be written to/from global memory given
 * the constraints that the TMA instruction imposes.
 *
 * This is accomplished by assigning "basis" strides to the GMEM to track which
 * modes of SMEM map to which modes of GMEM, then reorder the modes of GMEM according
 * to the SMEM vector, and then using those GMEM/SMEM modes to fill in the desc.
 *
 * Examples:
     using T = float;
     T* gptr = nullptr;

    {
    // Simple 2D
    Tensor gtensor = make_tensor(gptr, make_shape(1024, 256), GenRowMajor{}); // K-Major GMEM
    auto slayout   = make_layout(make_shape(_64{}, _32{}), GenRowMajor{});    // K-Major SMEM
    auto tma = make_tma_copy(XE4_ADMA_LOAD{}, gtensor, slayout);
    }

    {
    // GMMA 2D
    Tensor gtensor = make_tensor(gptr, make_shape(1024, 256));                                 // MN-Major GMEM
    auto slayout   = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(_128{},_64{})); // MN-Major Swizzled+Tiled 128x64 SMEM
    auto tma = make_tma_copy(XE4_ADMA_LOAD{}, gtensor, slayout);
    }

    {
    // 3D
    Tensor gtensor = make_tensor(gptr, make_shape(1024, 32, 512), make_stride(64, Int<1>{}, 65536)); // GMEM
    auto slayout   = make_layout(make_shape(_16{}, _8{}, _2{}), make_stride(_16{}, _1{}, _8{}));     // SMEM w/ same major-mode
    auto tma = make_tma_copy(XE4_ADMA_LOAD{}, gtensor, slayout);
    }

    {
    // cuTENSOR 4D
    auto layout = make_shape(make_shape(32,40),make_shape(make_shape(8,8),656)); // GMEM
    auto cta_tile    = make_shape(_128{},make_shape(_32{},_2{}));                // GMEM Tiling:
                                                                                 //   Take 128-elem from m: m0 must divide 128,
                                                                                 //                         m-last may be predicated
                                                                                 //   Take 32-elem from k0, 2-elem from k1
    auto slayout = make_layout(cta_tile);                                        // Col-Major SMEM
    auto tma = make_tma_copy(XE4_ADMA_LOAD{}, gtensor, slayout, cta_tile, Int<1>{});
    }
 *
 * Check the TMA box size and desc:
    print("TMA Box size:  "); print(typename decltype(tma)::Tiler_MN{}); print("\n");
    print("TMA desc     : "); print(tma.tma_desc_); print("\n");
 *
 * Usage:
     Tensor mA = tma_a.get_tma_tensor(make_shape(M,N));        // (M,N) TMA coord tensor
     Tensor gA = local_tile(mA, cta_tile, cta_coord);          // (BLK_M,BLK_N) TMA coord tensor for this CTA
     Tensor sA = make_tensor(make_smem_ptr<T>(sptr), slayout); // (BLK_M,BLK_N) SMEM tensor

     auto cta_tma = tma.get_slice(cta_idx_in_cluster);         // Slice for multicast partitioning
     Tensor tAgA = cta_tma.partition_S(gA);                    // Partition for src
     Tensor tAsA = cta_tma.partition_D(sA);                    // Partition for dst

     copy(tma.with(barrier, mcast_mask), tAgA, tAsA);          // copy with supporting TMA params
 */
template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size>
CUTE_HOST_RTC
auto
make_tma_copy(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout,
              CTA_Tiler               const& cta_tiler,
              Cluster_Size            const& cluster_size)
{
  if constexpr (is_base_of_v<xe4::ASYNC_ROW_IM2COL, CopyOp>) {
	  //ESYCL JM
    return make_im2col_tma_copy(copy_op,
                                gtensor,
                                slayout,
                                cta_tiler,
                                cluster_size);
  } else {
    auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
    auto cta_t_tile = make_layout(cluster_size);
    // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
    using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
    return detail::make_tma_copy_tiled<TmaType>(copy_op,
                                                gtensor, slayout,
                                                cta_t_tile, cta_v_tile);
  }
}

// Explicit defaulting
template <class CopyOp,
          class GEngine, class GLayout,
          class SLayout>
CUTE_HOST_RTC
auto
make_tma_copy(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout)
{
  return make_tma_copy(copy_op, gtensor, slayout, product_each(shape(slayout)), Int<1>{});
}

// Explicit defaulting
template <class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class Cluster_Size>
CUTE_HOST_RTC
auto
make_tma_copy(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout,
              Cluster_Size            const& cluster_size)
{
  return make_tma_copy(copy_op, gtensor, slayout, product_each(shape(slayout)), cluster_size);
}

////////////////////////////////////
// Experimental Make TMA Atom and Partitioner
///////////////////////////////////

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size = Int<1>>
CUTE_HOST_RTC
auto
make_tma_atom(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout,
              CTA_Tiler               const& cta_tiler,
              Cluster_Size            const& cluster_size = {})
{
  auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_atom<TmaType>(copy_op,
                                             gtensor, slayout,
                                             size(cluster_size), cta_v_tile);
}

// The "VectorCopy Partitioner" for TMA
template <class... Args,
          class CtaCoord,
          class TShape, class TStride,
          class SEngine, class SLayout,
          class GEngine, class GLayout>
CUTE_DEVICE
auto
tma_partition(Copy_Atom<Args...>      const& copy_atom,
              CtaCoord                const& cta_coord,
              Layout<TShape,TStride>  const& cta_layout,  // T: CTA coord -> logical multicast id
              Tensor<SEngine,SLayout> const& stensor,     // SMEM Tensor (TMATile, Rest...)
              Tensor<GEngine,GLayout> const& gtensor)     // GMEM Tensor (TMATile, Rest...)
{
  CUTE_STATIC_ASSERT_V(size<0>(stensor) == size<0>(gtensor));

  // Invert the smem to get the largest contiguous vector in the smem layout
  Layout inv_smem_layout = right_inverse(get_nonswizzle_portion(layout<0>(stensor)));
  // Scale that up to cover all of the smem_coords
  Layout layout_v = tile_to_shape(make_layout(inv_smem_layout), size<0>(stensor));

  // Factor out the single-instrucion portion
  Layout tma_layout_v = make_layout(Int<Copy_Atom<Args...>::NumValSrc>{});
  auto layout_V = make_tile(logical_divide(layout_v, tma_layout_v));

  // Append with _ until we cover all Rest... modes
  auto glayout_V = append<GLayout::rank>(layout_V, _);
  auto slayout_V = append<SLayout::rank>(layout_V, _);
  // Transform tile mode and coalesce
  Tensor gtensor_v = coalesce(gtensor.compose(glayout_V), Shape<Shape<_1,_1>>{});    // ((TMA,TMA_Iter), Rest...)
  Tensor stensor_v = coalesce(stensor.compose(slayout_V), Shape<Shape<_1,_1>>{});    // ((TMA,TMA_Iter), Rest...)
  // Offset inside the TMA-mode for the multicast
  auto multicast_offset = cta_layout(cta_coord) * (size(tma_layout_v) / cosize(cta_layout));
  auto multicast_coord  = make_coord(make_coord(multicast_offset, Int<0>{}));
  auto gcoord = append<GLayout::rank>(multicast_coord, Int<0>{});
  auto scoord = append<SLayout::rank>(multicast_coord, Int<0>{});

  Tensor gresult = domain_offset(gcoord, gtensor_v);
  Tensor sresult = domain_offset(scoord, stensor_v);

  return cute::make_tuple(gresult, sresult);
}

// Explicit defaults for cta_coord and cta_layout
template <class... Args,
          class SEngine, class SLayout,
          class GEngine, class GLayout>
CUTE_DEVICE
auto
tma_partition(Copy_Atom<Args...>      const& copy_atom,
              Tensor<SEngine,SLayout> const& stensor,     // SMEM Tensor (TMATile, Rest...)
              Tensor<GEngine,GLayout> const& gtensor)     // GMEM Tensor (TMATile, Rest...)
{
  return tma_partition(copy_atom, Int<0>{}, Layout<_1,_0>{}, stensor, gtensor);
}

// TMA Multicast Masks Calculation
template <class CtaLayout, class CtaCoord>
CUTE_HOST_DEVICE constexpr
uint16_t
create_tma_multicast_mask(CtaLayout const& cta_layout_vmnk,
                          CtaCoord  const& cta_coord_vmnk)
{
  auto [cta_layout, elected_cta] = slice_and_offset(cta_coord_vmnk, cta_layout_vmnk);

  uint16_t mcast_mask = 0;
  if constexpr (rank_v<decltype(cta_layout)> == 0) {
    // Trivial case with no additional ctas
    mcast_mask = uint16_t(1);
  } else
  if constexpr (rank_v<decltype(cta_layout)> == 1 and depth_v<decltype(cta_layout)> <= 1 and
                not is_static<decltype(cta_layout)>::value) {
    // Get the instruction code -- optimized for dynamic flat-rank-1 cta_layout
    mcast_mask = uint16_t(1);
    // Smear by stride<0> (may want to predicate on stride<0> mag?)
    mcast_mask |= mcast_mask << (1*stride<0>(cta_layout));
    mcast_mask |= mcast_mask << (2*stride<0>(cta_layout));
    mcast_mask |= mcast_mask << (4*stride<0>(cta_layout));
    mcast_mask |= mcast_mask << (8*stride<0>(cta_layout));
    // Select shape<0>
    mcast_mask &= (uint16_t(-1) >> (16 - shape<0>(cta_layout) * stride<0>(cta_layout)));
  } else {
    // Get the instruction code -- generic path
    for (int i = 0; i < size(cta_layout); ++i) {
      mcast_mask |= uint16_t(1) << cta_layout(i);
    }
  }
  // Shift by the instruction's elected block rank (dynamic)
  mcast_mask <<= elected_cta;
  return mcast_mask;
}

// Projections multicast mask
template <int Mode, int... Modes, class CtaLayout, class CtaCoord>
CUTE_HOST_DEVICE constexpr
uint16_t
create_tma_multicast_mask(CtaLayout const& cta_layout_vmnk,
                          CtaCoord  const& cta_coord_vmnk)
{
  return create_tma_multicast_mask<Modes...>(cta_layout_vmnk, replace<Mode>(cta_coord_vmnk, _));
}

////////////////////////////////////
// Make TMA copy A/B/C
///////////////////////////////////

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size>
CUTE_HOST_RTC
auto
make_tma_copy_A(CopyOp                  const& copy_op,
                Tensor<GEngine,GLayout> const& gtensor,
                SLayout                 const& slayout,
                CTA_Tiler               const& cta_tiler,
                Cluster_Size            const& cluster_size)
{
  // Keep only MK modes from MNK
  auto cta_tiler_mk = remove<1>(cta_tiler);

  // mcast along N mode for this M load, if any
  auto cluster_size_n = size<1>(cluster_size);

  // ESYCL JM
  if constexpr (is_base_of_v<xe4::ASYNC_ROW_IM2COL, CopyOp>) {
	  //ESYCL JM
    return make_im2col_tma_copy(copy_op,
                                gtensor,
                                slayout,
                                cta_tiler_mk,
                                cluster_size_n);
  } else {
    auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler_mk);
    auto cta_t_tile = make_layout(cluster_size_n);

    // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
    using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
    auto tma_copy = detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout, cta_t_tile, cta_v_tile);
    return tma_copy;
  }
}

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size>
CUTE_HOST_RTC
auto
make_tma_copy_B(CopyOp                  const& copy_op,
                Tensor<GEngine,GLayout> const& gtensor,
                SLayout                 const& slayout,
                CTA_Tiler               const& cta_tiler,
                Cluster_Size            const& cluster_size)
{
  // Keep only NK modes from MNK
  auto cta_tiler_nk = remove<0>(cta_tiler);

  // mcast along M mode for this N load, if any
  auto cluster_size_m = size<0>(cluster_size);

  // ESYCL JM
  if constexpr (is_base_of_v<xe4::ASYNC_ROW_IM2COL, CopyOp>) {
	  // ESYCL JM
    return make_im2col_tma_copy(copy_op,
                                gtensor,
                                slayout,
                                cta_tiler_nk,
                                cluster_size_m);
  } else {
    auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler_nk);
    auto cta_t_tile = make_layout(cluster_size_m);

    // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
    using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
    auto tma_copy = detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout, cta_t_tile, cta_v_tile);
    return tma_copy;
  }
}

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler>
CUTE_HOST_RTC
auto
make_tma_copy_C(CopyOp                  const& copy_op,
                Tensor<GEngine,GLayout> const& gtensor,
                SLayout                 const& slayout,
                CTA_Tiler               const& cta_tiler)
{
  // Keep only MN modes from MNK
  auto cta_tiler_mn = remove<2>(cta_tiler);

  // ESYCL JM
  if constexpr (is_base_of_v<xe4::ASYNC_ROW_IM2COL, CopyOp>) {
	  // ESYCL JM
    return make_im2col_tma_copy(copy_op,
                                gtensor,
                                slayout,
                                cta_tiler_mn,
                                _1{});
  } else {
    auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler_mn);

    // No multicast, so only 1 CTA involved
    auto cta_t_map = Layout<_1,_0>{};

    // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
    using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
    auto tma_copy = detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout, cta_t_map, cta_v_tile);
    return tma_copy;
  }
}

////////////////////////////////////
// Make TMA
///////////////////////////////////

#if !defined(__CUDACC_RTC__)
/** Make a CuTe CTA-collective TiledCopy for a TMA operation.
 *
 * @param CopyOp The target copy operation: SM100_TMA_2SM_LOAD
 * @param gtensor The GMEM Tensor to be involved in the TMA.
 * @param slayout The SMEM Layout to be involved in the TMA.
 * @param cluster_tile The Cluster-local tile that each Cluster will be tiling GMEM with.
 *                     This is often the cluster_tile_shape that is used to tile the GMEM:
 *                       local_tile(gtensor, cluster_tile_shape, cluster_coord)
 *                         -> Cluster-local tile of GMEM
 * @param mma The TiledMMA that defines the Cluster-Tile to Block-Tile partitioning.
 *
 * This code attempts to maximize the TMA box size. It does this by tracing
 * the SMEM "vector" -- the inverse of the smem layout -- to find the largest
 * contiguous array of smem that can be written to/from global memory given
 * the constraints that the TMA instruction imposes.
 *
 * This is accomplished by assigning "basis" strides to the GMEM to track which
 * modes of SMEM map to which modes of GMEM, then reordering the modes of GMEM according
 * to the SMEM vector, and then using those GMEM/SMEM modes to fill in the desc.
 *
 * Examples:
 */
template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class Cluster_Tiler,
          class... Args>
CUTE_HOST
auto
make_tma_copy_A(CopyOp                  const& copy_op,
                Tensor<GEngine,GLayout> const& gtensor,        // (M, K, ...)
                SLayout                 const& slayout,        // (MMA, MMA_M, MMA_K, ...)
                Cluster_Tiler           const& cluster_tiler,  // (TILER_M, TILER_N, TILER_K, ...)
                TiledMMA<Args...>       const& mma)
{
  // Keep only MK modes from MNK
  auto cluster_tiler_mk = remove<1>(cluster_tiler);
  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(cluster_tiler_mk);     // (TILE_M, TILE_K, ...)
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_A(g_tile))(_, repeat<rank(g_tile)>(_));    // (MMA, MMA_M, MMA_K, ...)

  // ESYCL JM
  auto cta_t_vmnk_strides = [](){
    if constexpr (is_same_v<CopyOp, XE4_ADMA_LOAD_MULTICAST> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_MULTICAST>) {
      return Stride<_0,_0,_1,_0>{};                    // VMNK: Use only the N-CTAs in the Multicast
    } else
    if constexpr (is_same_v<CopyOp, XE4_ADMA_LOAD>  ||
                  is_same_v<CopyOp, XE4_ADMA_STORE> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD>) {
      return Stride<_0,_0,_0,_0>{};                    // VMNK: Use no CTAs in Non-Multicast
    } else {
      static_assert(dependent_false<CopyOp>, "Unsupported TMA");
    }
  }();

  auto cta_t_shape = shape(mma.get_thr_layout_vmnk());
  // cta rank -> logical cta idx
  auto cta_t_map  = coalesce(make_layout(cta_t_shape, compact_col_major(cta_t_shape, cta_t_vmnk_strides)));

  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout, cta_t_map, cta_v_tile);
}

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class Cluster_Tiler,
          class... Args>
CUTE_HOST
auto
make_tma_copy_B(CopyOp                  const& copy_op,
                Tensor<GEngine,GLayout> const& gtensor,        // (N, K, ...)
                SLayout                 const& slayout,        // (MMA, MMA_N, MMA_K, ...)
                Cluster_Tiler           const& cluster_tiler,  // (TILE_M, TILE_N, TILE_K, ...)
                TiledMMA<Args...>       const& mma)
{
  // Keep only NK modes from MNK
  auto cluster_tiler_nk = remove<0>(cluster_tiler);
  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(cluster_tiler_nk);     // (TILE_N, TILE_K, ...)
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_B(g_tile))(_, repeat<rank(g_tile)>(_));    // (MMA, MMA_N, MMA_K, ...)

  // ESYCL JM
  auto cta_t_vmnk_strides = [](){
    if constexpr (is_same_v<CopyOp, XE4_ADMA_LOAD_MULTICAST> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD_MULTICAST>) {
      return Stride<_0,_1,_0,_0>{};                    // VMNK: Use only the M-CTAs in the Multicast
    } else
    if constexpr (is_same_v<CopyOp, XE4_ADMA_LOAD>  ||
                  is_same_v<CopyOp, XE4_ADMA_STORE> ||
                  is_same_v<CopyOp, SM100_TMA_2SM_LOAD>) {
      return Stride<_0,_0,_0,_0>{};                    // VMNK: Use no CTAs in Non-Multicast
    } else {
      static_assert(dependent_false<CopyOp>, "Unsupported TMA");
    }
  }();

  auto cta_t_shape = shape(mma.get_thr_layout_vmnk());
  // cta rank -> logical cta idx
  auto cta_t_map  = coalesce(make_layout(cta_t_shape, compact_col_major(cta_t_shape, cta_t_vmnk_strides)));

  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout, cta_t_map, cta_v_tile);
}

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class Cluster_Tiler,
          class... Args>
CUTE_HOST
auto
make_tma_copy_C(CopyOp                  const& copy_op,
                Tensor<GEngine,GLayout> const& gtensor,        // (M, N, ...)
                SLayout                 const& slayout,        // (MMA, MMA_M, MMA_N, ...)
                Cluster_Tiler           const& cluster_tiler,  // (TILE_M, TILE_N, TILE_K, ...)
                TiledMMA<Args...>       const& mma)
{
  // Keep only MN modes from MNK
  auto cluster_tiler_mn = remove<2>(cluster_tiler);
  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(cluster_tiler_mn);     // (TILE_M, TILE_N, ...)
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_C(g_tile))(_, repeat<rank(g_tile)>(_));    // (MMA, MMA_M, MMA_N, ...)

  // ESYCL JM
  static_assert(is_same_v<CopyOp, XE4_ADMA_LOAD>  ||
                is_same_v<CopyOp, XE4_ADMA_STORE> ||
                is_same_v<CopyOp, SM100_TMA_2SM_LOAD>,
                "Unsupported TMA Op, expected a non-multicast TMA");

  // No multicast, so only 1 CTA involved
  auto cta_t_map = Layout<_1,_0>{};

  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_tiled<TmaType>(copy_op, gtensor, slayout, cta_t_map, cta_v_tile);
}

////////////////////////////////////
// Experimental Make TMA Atom
///////////////////////////////////

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class MMA_Tiler,
          class... Args,
          class ClusterShapeVMNK>
CUTE_HOST
auto
make_tma_atom_A(CopyOp                  const& copy_op,
                Tensor<GEngine,GLayout> const& gtensor,        // (M, K, ...)
                SLayout                 const& slayout,        // (MMA, MMA_M, MMA_K, ...)
                MMA_Tiler               const& mma_tiler,      // (TILE_M, TILE_N, TILE_K, ...)
                TiledMMA<Args...>       const& mma,
                ClusterShapeVMNK        const& cluster_shape)  // (CTA_V, CTA_M, CTA_N, CTA_K)
{
  // Keep only MK modes from MNK
  auto mma_tiler_mk = remove<1>(mma_tiler);

  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(mma_tiler_mk);         // (TILE_M, TILE_K, ...)

  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_A(g_tile))(_, repeat<rank(g_tile)>(_));    // (MMA, MMA_M, MMA_K, ...)

#if 0
  print("(tma_a) slayout:      "); print(slayout);      print("\n");
  print("(tma_a) mma_tiler_nk: "); print(mma_tiler_nk); print("\n");
  print("(tma_a) g_tile:       "); print(g_tile);       print("\n");
  print("(tma_a) mma_tiler:    "); print(mma_tiler);    print("\n");
  print("(tma_a) cta_v_tile:   "); print(cta_v_tile);   print("\n");
#endif

  // The size of the multicasting
  auto num_multicast = [&](){
    if constexpr (is_base_of_v<xe4::DMA_MULTICAST, CopyOp>) {
      return size<2>(cluster_shape);                   // VMNK: Use only the N-CTAs in the Multicast
    } else {
      return Int<1>{};                                 // VMNK: Use no CTAs in Non-Multicast
    }
  }();

  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_atom<TmaType>(copy_op, gtensor, slayout, num_multicast, cta_v_tile);
}

template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class MMA_Tiler,
          class... Args,
          class ClusterShapeVMNK>
CUTE_HOST
auto
make_tma_atom_B(CopyOp                  const& copy_op,
                Tensor<GEngine,GLayout> const& gtensor,        // (N, K, ...)
                SLayout                 const& slayout,        // (MMA, MMA_N, MMA_K, ...)
                MMA_Tiler               const& mma_tiler,      // (TILE_M, TILE_N, TILE_K, ...)
                TiledMMA<Args...>       const& mma,
                ClusterShapeVMNK        const& cluster_shape)  // (CTA_V, CTA_M, CTA_N, CTA_K)
{
  // Keep only NK modes from MNK
  auto mma_tiler_nk = remove<0>(mma_tiler);
  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(mma_tiler_nk);         // (TILE_N, TILE_K, ...)
  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_B(g_tile))(_, repeat<rank(g_tile)>(_));    // (MMA, MMA_N, MMA_K, ...)

#if 0
  print("(tma_b) slayout:      "); print(slayout);      print("\n");
  print("(tma_b) mma_tiler_nk: "); print(mma_tiler_nk); print("\n");
  print("(tma_b) g_tile:       "); print(g_tile);       print("\n");
  print("(tma_b) mma_tiler:    "); print(mma_tiler);    print("\n");
  print("(tma_b) cta_v_tile:   "); print(cta_v_tile);   print("\n");
#endif

  // The size of the multicasting
  auto num_multicast = [&](){
    if constexpr (is_base_of_v<xe4::DMA_MULTICAST, CopyOp>) {
      return size<1>(cluster_shape);                   // VMNK: Use only the M-CTAs in the Multicast
    } else {
      return Int<1>{};                                 // VMNK: Use no CTAs in Non-Multicast
    }
  }();

  // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
  using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
  return detail::make_tma_copy_atom<TmaType>(copy_op, gtensor, slayout, num_multicast, cta_v_tile);
}

#endif // !defined(__CUDACC_RTC__)

} // end namespace cute
#endif
