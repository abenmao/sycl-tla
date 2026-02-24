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
#pragma once

/*! \file
  \brief im2col make_tma_copy
*/
#if (SYCL_INTEL_TARGET == 40)

#include "cute/arch/copy_xe4_desc.hpp"
#include "cute/tensor.hpp"

#include "cute/algorithm/prefetch.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/cuda_host_adapter.hpp"

#include <cute/arch/copy_xe4_dma.hpp>

namespace cute
{
template <int NumBytesPerCopy, class GTensor>
CUTE_HOST_DEVICE auto
make_async_row_copy_desc(GTensor const& gtensor)
{
  using T = typename GTensor::value_type;

  Im2ColTmaDescriptor<T, NumBytesPerCopy> tdesc_ptr;
  tdesc_ptr.bytes[0] = reinterpret_cast<uint64_t>(&gtensor.data()[0]);
  tdesc_ptr.bytes[1] = shape<0>(gtensor);
  tdesc_ptr.bytes[2] = shape<1>(gtensor);
  tdesc_ptr.bytes[3] = shape<2>(gtensor);
  tdesc_ptr.bytes[4] = shape<3>(gtensor);

  tdesc_ptr.bytes[6] = stride<1>(gtensor);
  tdesc_ptr.bytes[7] = stride<2>(gtensor);
  tdesc_ptr.bytes[8] = stride<3>(gtensor);

  return tdesc_ptr;
}

template <class TensorDesc, class CoordTensor>
struct Xe4Im2ColCache {
  template <typename CopyOp>
  using OpUnpack = XE4_COPY_Unpack<CopyOp>;

  CUTE_HOST_DEVICE constexpr
  auto get_tensor_desc() const {
    return &tensor_desc_;
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor([[maybe_unused]] GShape const& g_shape) const {
    return coord_tensor_;
  }

  template <typename... Args>
  CUTE_HOST_DEVICE constexpr
  auto make_args_tuple(Args&&... args) const {
    return make_tuple(&tensor_desc_, static_cast<Args&&>(args)...);
  }

  template <typename GmemPtr>
  CUTE_HOST_DEVICE constexpr
  void set_gmem_ptr(const GmemPtr &gmem_ptr) {
    tensor_desc_.bytes[0] = reinterpret_cast<uint64_t>(gmem_ptr);
  }

  TensorDesc tensor_desc_;
  CoordTensor coord_tensor_;
};

namespace detail {

/// @brief Creates a TMA descriptor for im2col TMA load.
///
/// @param tensor_cwhdn Global activation tensor (A matrix of Fprop).
///   This is the original (not im2col-transformed) tensor in global
///   memory.
///
/// @param slayout Rank 2 (M,K) shared memory layout of the activation
///   tensor.  Here, K is "GEMM K," not the filter tensor's mode of
///   the same name.
//////
/// @param traversal_stride Traversal strides convolution parameter
//////
/// Each of padding_shape, traversal_stride, and dilation_shape is a
/// tuple whose size is the number of spatial modes (e.g., 3 for a 5-D
/// convolution).
///
/// @return TMA descriptor for im2col TMA load
template <class EngineA, class LayoutA,
          class SmemSwizzle, class TMALayout,
          class LowerCornerStride,
          class UpperCornerStride,
          class LowerPaddingStride,
          class UpperPaddingStride,
          class TraversalStride,
          class LowerSRTStride,
          class DilationStride>
CUTE_HOST
auto
make_im2col_tma_copy_desc(
    Tensor<EngineA, LayoutA>    const& tensor_cwhdn,       // (C,W,H,D,N)
    uint32_t                           range_c,            // TILE_C
    uint32_t                           range_whdn,         // TILE_WHDN
    SmemSwizzle                 const& smem_swizzle,       // Swizzle
    TMALayout                   const& tma_layout_vt,      // TMA layout
    LowerCornerStride           const& lower_corner_whd,   // WHD offset of the "base pointer"
    UpperCornerStride           const& upper_corner_whd,   // WHD upper corner
    LowerPaddingStride          const& lower_padding_whd,  // WHD lower padding
    UpperPaddingStride          const& upper_padding_whd,  // WHD upper padding
    TraversalStride             const& stride_whd,         // WHD traversal stride
    LowerSRTStride              const& lower_srt,          // SRT offset of the "base pointer"
    DilationStride              const& stride_srt,          // SRT stride - dilation
    TMA::DescriptorAuxParams    const& aux_params = {})
{
  using value_type = typename EngineA::value_type;

  constexpr uint32_t num_total_modes   = LayoutA::rank;
  constexpr int      num_spatial_modes = num_total_modes - 2;

  // Gmem starting address
  void* gmem_address = (void*) raw_pointer_cast(tensor_cwhdn.data());

  // Gmem extents are just the tensor shape
  cute::array<uint64_t, 5> gmem_prob_shape = {1,1,1,1,1};
  for_each(make_seq<num_total_modes>{}, [&](auto i) {
    gmem_prob_shape[i] = static_cast<uint64_t>(shape<i>(tensor_cwhdn));
  });

  // Gmem strides are byte strides of the activation tensor in CWHDN order
  cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};
  for_each(make_seq<num_total_modes>{}, [&](auto i) {
    gmem_prob_stride[i] = sizeof(value_type) * stride<i>(tensor_cwhdn);
  });

  // Traversal strides are a function of the dilation shape
  // corresponding to spatial (WHD) modes.
  cute::array<uint32_t, 5> tma_traversal_strides = {1,1,1,1,1};
  for_each(make_seq<num_spatial_modes>{}, [&](auto i) {
    tma_traversal_strides[i+1] = static_cast<uint32_t>(get<i>(stride_whd));
  });

  cute::array<int32_t, num_spatial_modes> tma_lower_corner{};
  for_each(make_seq<num_spatial_modes>{}, [&](auto i) {
    tma_lower_corner[i] = static_cast<int32_t>(get<i>(lower_corner_whd));
  });

  cute::array<int32_t, num_spatial_modes> tma_upper_corner{};
  for_each(make_seq<num_spatial_modes>{}, [&](auto i) {
    tma_upper_corner[i] = static_cast<int32_t>(get<i>(upper_corner_whd));
  });

  using T = typename EngineA::value_type;
  constexpr int num_bytes_per_tma =
    []{
        if constexpr (decltype(stride<0>(smem_swizzle))::value == 1) {
            return decltype(shape<0>(smem_swizzle))::value * sizeof(T);
        } else {
            return decltype(shape<1>(smem_swizzle))::value * sizeof(T);
        }
    }();

  using Im2ColDesc = Im2ColTmaDescriptor<T, num_bytes_per_tma>;
  Im2ColDesc tma_desc = make_async_row_copy_desc<num_bytes_per_tma>(tensor_cwhdn);

  //
  // Calculate gemm shapes and linearized shapes based on tma layout tiling.
  //

  // Compute [w, h, d, n]
  // q/p/z = (w/h/d + (upper_corner_whd - lower_corner_whd - 1)) / stride_whd + 1
  auto gemm_mn_ = cute::transform(cute::make_seq<num_spatial_modes>{}, [&](auto i) {
    return (shape<i+1>(tensor_cwhdn) + get<i>(upper_corner_whd) - get<i>(lower_corner_whd) - Int<1>{}) / get<i>(stride_whd) + Int<1>{};
  });
  auto gemm_mn = append(gemm_mn_, shape<num_spatial_modes+1>(tensor_cwhdn));

  // Compute [c, s, r, t]
  // fprop/wgrad, s/r/t = 1 + (upper_padding_whd - upper_corner_whd) / stride_srt
  // wgrad,       s/r/t = 1 + (lower_padding_whd - lower_corner_whd) / stride_srt
  auto gemm_k_ = cute::transform(cute::make_seq<num_spatial_modes>{}, [&](auto i) {
    auto padding_size = conditional_return(get<i>(stride_srt) > Int<0>{},
                                           get<i>(upper_padding_whd) - get<i>(upper_corner_whd),
                                           get<i>(lower_corner_whd)  - get<i>(lower_padding_whd));
    return Int<1>{} + padding_size / get<i>(stride_srt);
  });
  auto gemm_k = prepend(gemm_k_, shape<0>(tensor_cwhdn));

  // For fprop/dgrad kernel, gemm_shapes is ((q, p, z, n), (c, s, r, t))
  // For wgrad kernel, gemm_shapes is ((c, s, r, t), (q, p, z, n))
  auto gemm_shapes_common = make_shape(gemm_mn, gemm_k);

  auto gemm_shapes = make_shape(
      basis_get(stride<0,1>(tma_layout_vt), gemm_shapes_common),
      basis_get(stride<0,0>(tma_layout_vt), gemm_shapes_common));

  // For fprop/dgrad kernel, linearized shapes is (whdn, (c, s, r, t))
  // For wgrad kernel linearized shapes is ((c, s, r, t), whdn)
  auto linear_shapes_common = make_shape(size(gemm_mn), gemm_k);
  auto linear_shapes = make_shape(
      basis_get(stride<0,1>(tma_layout_vt), linear_shapes_common),
      basis_get(stride<0,0>(tma_layout_vt), linear_shapes_common));

  //
  // Calculate gmem basis stride based on tma layout tiling.
  //

  auto tma_basis_scale = make_shape(Int<1>{}, stride_whd, Int<1>{}, stride_srt);
  auto tma_basis = elem_scale(tma_basis_scale, make_basis_like(tma_basis_scale));

  auto gbasis_strides_common = make_stride(
      append(get<1>(tma_basis), get<2>(tma_basis)),
      prepend(get<3>(tma_basis), get<0>(tma_basis)));    // ((w,h,d,n),(c,s,r,t))
  auto gbasis_strides = make_stride(
      basis_get(stride<0,1>(tma_layout_vt), gbasis_strides_common),
      basis_get(stride<0,0>(tma_layout_vt), gbasis_strides_common));

  //
  // Create tma tensor
  //

  auto lower_corner = make_arithmetic_tuple(Int<0>{}, lower_corner_whd, Int<0>{}, lower_srt);

  auto tensor_multimode = make_tensor(ArithmeticTupleIterator(lower_corner), gemm_shapes, gbasis_strides);
  auto tensor_linear = make_identity_tensor(linear_shapes);
  auto tma_tensor = make_tensor(tensor_multimode.data(), composition(
      tensor_multimode.layout(),
      tensor_linear(Int<0>{}),
      tensor_linear.layout()));

  return cute::make_tuple(tma_desc, tma_tensor);
}

template <class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class VShape, class VStride,
          class LowerCornerStride,
          class UpperCornerStride,
          class LowerPaddingStride,
          class UpperPaddingStride,
          class TraversalStride,
          class LowerSRTStride,
          class DilationStride>
CUTE_HOST_RTC
auto
make_tma_atom_im2col(CopyOp,
                     Tensor<GEngine,GLayout>      const& gtensor,           // Full GMEM Tensor: ((w, h, d, n), c)
                     SLayout                      const& slayout,           // CTA Tile of SMEM, potentially swizzled
                     int32_t                      const& num_multicast,     // The number of CTAs involved in multicasting
                     Layout<VShape,VStride>       const& cta_v_map,         // V: CTA val idx -> gmem mode
                     LowerCornerStride            const& lower_corner_whd,
                     UpperCornerStride            const& upper_corner_whd,
                     LowerPaddingStride           const& lower_padding_whd,
                     UpperPaddingStride           const& upper_padding_whd,
                     TraversalStride              const& stride_whd,        // traversal stride
                     LowerSRTStride               const& lower_srt,
                     DilationStride               const& stride_srt,        // dilation
                     TMA::DescriptorAuxParams     const& aux_params = {})
{
  //
  // TMA parameter checking
  //

  CUTE_STATIC_ASSERT_V(product_each(shape(slayout)) == product_each(shape(cta_v_map)),
    "TMA requires CTA_Tile and SLayout top-level shape equivalence.");

  //
  // TMA slayout manipulation
  //

  // Invert the smem to get the largest contiguous vector in the smem layout
  auto inv_smem_layout = right_inverse(get_nonswizzle_portion(slayout));
  // trunc_smem_idx -> trunc_smem_coord

  // Map from smem idx to a gmem mode
  auto sidx_to_gmode = coalesce(composition(cta_v_map, inv_smem_layout));

#if 0
  print("g_layout         : "); print(gtensor.layout()); print("\n");
  print("s_layout         : "); print(slayout); print("\n");
  print("cta_t_map        : "); print(cta_t_map); print("\n");
  print("cta_v_map        : "); print(cta_v_map); print("\n");
  print("inv_smem         : "); print(inv_smem_layout); print("\n");
  print("sidx_to_gmode    : "); print(sidx_to_gmode); print("\n");
#endif

  //
  // TMA gtensor manipulation
  //

  // Generate a TupleBasis for the gtensor
  auto glayout_basis = make_identity_layout(product_each(shape(gtensor)));

  // Tile the modes of gtensor with the truncated cta_v_map o inv_smem_layout_trunc
  auto tma_layout_full = flatten(composition(glayout_basis, sidx_to_gmode));

  // Truncate any incompatibilities -- no starting in the middle of gmodes
  auto smem_rank = find_if(stride(tma_layout_full), [](auto e) {
    [[maybe_unused]] auto v = basis_value(e);
    return not is_constant<1,decltype(v)>{};
  });
  static_assert(smem_rank >= 2, "IM2COL expects at least 2 modes of the smem to vectorize with gmem.");
  // IM2COL uses a maximum of 2 modes
  constexpr int smem_tma_rank = cute::min(int(smem_rank), 2);

  // Keep only the static-1 basis modes into gmem
  auto tma_layout_trunc = take<0,smem_tma_rank>(tma_layout_full);

  // Split according to the portion each multicast CTA will be responsible for
  auto tma_layout_vt = logical_divide(tma_layout_trunc, safe_div(size(tma_layout_trunc), num_multicast));

#if 0
  print("glayout_basis   : "); print(glayout_basis); print("\n");
  print("tma_layout_full : "); print(tma_layout_full); print("\n");

  print("tma_layout_trunc: "); print(tma_layout_trunc); print("\n");
  print("tma_layout_vt   : "); print(tma_layout_vt); print("\n");
#endif

  auto range_c    = size<0,0>(tma_layout_vt);
  auto range_whdn = size<0,1>(tma_layout_vt);
  Tensor gtensor_cwhdn = make_tensor(gtensor.data(),
                                     flatten(make_layout(make_layout(basis_get(stride<0,0>(tma_layout_vt), gtensor.shape()),
                                                                     basis_get(stride<0,0>(tma_layout_vt), gtensor.stride())),
                                                         make_layout(basis_get(stride<0,1>(tma_layout_vt), gtensor.shape()),
                                                                     basis_get(stride<0,1>(tma_layout_vt), gtensor.stride())))));
  auto [tma_desc, tma_tensor] = make_im2col_tma_copy_desc(
      gtensor_cwhdn,
      range_c,
      range_whdn,
      slayout,
      tma_layout_vt,
      lower_corner_whd,
      upper_corner_whd,
      lower_padding_whd,
      upper_padding_whd,
      stride_whd,
      lower_srt,
      stride_srt,
      aux_params);

  //
  // Construct the Copy_Traits
  //

  using T = typename GEngine::value_type;
  constexpr int num_bits_per_tma = decltype(size<0, 0>(tma_layout_trunc))::value * sizeof(T) * 8;
  constexpr int num_bytes_per_tma = decltype(size<0, 0>(tma_layout_trunc))::value * sizeof(T);

  using Im2ColDesc = Im2ColTmaDescriptor<T, num_bytes_per_tma>;
  using Im2ColCache = Xe4Im2ColCache<Im2ColDesc, decltype(tma_tensor)>;
  using Traits = Copy_Traits<Xe4CopyOp<CopyOp>, cute::C<num_bits_per_tma>, Im2ColCache>;
  using Atom = Copy_Atom<Traits, typename GEngine::value_type>;

#if 0
  print("num_bits      :  "); print(num_bits_per_tma); print("\n");
#endif

  Traits tma_traits{tma_desc, tma_tensor};
  // Return the Copy_Atom
  return Atom{tma_traits};
}

/// Make a TiledCopy for im2col TMA load.
///
/// @param copy_op The copy implementation: either
///   *LOAD_IM2COL or *LOAD_IM2COL_MULTICAST for XE4
///
/// @param tensor_cwhdn The global tensor to use for im2col TMA loads.
///   For Fprop convolutions, this is the activation tensor.  This is
///   the "original tensor that points to global memory, not the
///   coordinate (im2col-transformed) tensor.
///
/// @param slayout Layout of shared memory tile.
///
/// @param stride_whd The traversal strides convolution
///   parameter.
///
/// @return TiledCopy specialization for im2col TMA loads.
template <class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class TShape, class TStride,
          class VShape, class VStride,
          class LowerCornerStride,
          class UpperCornerStride,
          class LowerPaddingStride,
          class UpperPaddingStride,
          class TraversalStride,
          class LowerSRTStride,
          class DilationStride>
CUTE_HOST_RTC
auto
make_tma_copy_im2col(CopyOp                       const& copy_op,
                     Tensor<GEngine,GLayout>      const& gtensor,
                     SLayout                      const& slayout,
                     Layout<TShape,TStride>       const& cta_t_map,          // CTA tid -> logical TMA tid
                     Layout<VShape,VStride>       const& cta_v_map,          // CTA vid -> gmem coord
                     LowerCornerStride            const& lower_corner_whd,
                     UpperCornerStride            const& upper_corner_whd,
                     LowerPaddingStride           const& lower_padding_whd,
                     UpperPaddingStride           const& upper_padding_whd,
                     TraversalStride              const& stride_whd,         // traversal stride
                     LowerSRTStride               const& lower_srt,
                     DilationStride               const& stride_srt,         // dilation
                     TMA::DescriptorAuxParams     const& aux_params = {})
{
  //
  // TMA parameter checking
  //

  Copy_Atom atom = make_tma_atom_im2col(copy_op, gtensor, slayout, cosize(cta_t_map), cta_v_map,
                                        lower_corner_whd, upper_corner_whd, lower_padding_whd,
                                        upper_padding_whd, stride_whd, lower_srt, stride_srt, aux_params);
  //
  // Construct the TiledCopy
  //
  if constexpr (decltype(stride<0>(slayout))::value == 1) {
    auto layout_t = make_layout(make_shape(Int<1>{}, Int<cutlass::NumThreadsPerWarp>{}));
    auto layout_v = make_layout(make_shape(shape<0>(cta_v_map), Int<1>{}));

    return make_tiled_copy(atom, layout_t, layout_v);
  } else {
    auto layout_t = make_layout(make_shape(Int<cutlass::NumThreadsPerWarp>{}, Int<1>{}));
    auto layout_v = make_layout(make_shape(Int<1>{}, shape<1>(cta_v_map)));

    return make_tiled_copy(atom, layout_t, layout_v);
  }
}

/// Make a TiledCopy for im2col TMA with no offsets.
/// E.g. im2col TMA load for C and im2col TMA store for D.
template <class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class TShape, class TStride,
          class VShape, class VStride>
CUTE_HOST_RTC
auto
make_tma_copy_im2col(CopyOp                  const& copy_op,
                     Tensor<GEngine,GLayout> const& gtensor,
                     SLayout                 const& slayout,
                     Layout<TShape,TStride>  const& cta_t_map,          // CTA tid -> logical TMA tid
                     Layout<VShape,VStride>  const& cta_v_map)          // CTA vid -> gmem coord
{
  constexpr int num_spatial_modes = rank<0>(GLayout{}) - 1;
  return make_tma_copy_im2col(copy_op, gtensor, slayout, cta_t_map, cta_v_map,
                              append<num_spatial_modes>(Stride<_0>{}, Int<0>{}),  // lower_corner_whd
                              append<num_spatial_modes>(Stride<_0>{}, Int<0>{}),  // upper_corner_whd
                              append<num_spatial_modes>(Stride<_0>{}, Int<0>{}),  // lower_padding_whd
                              append<num_spatial_modes>(Stride<_0>{}, Int<0>{}),  // upper_padding_whd
                              append<num_spatial_modes>(Stride<_1>{}, Int<1>{}),  // stride_whd
                              append<num_spatial_modes>(Stride<_0>{}, Int<0>{}),  // lower_srt
                              append<num_spatial_modes>(Stride<_1>{}, Int<1>{})); // stride_srt
}

} // namespace detail



template <class CopyOp,
          class Engine0, class Layout0,
          class SLayout,
          class CTATiler,
          class MulticastSize,
          class LowerCornerStride,
          class UpperCornerStride,
          class LowerPaddingStride,
          class UpperPaddingStride,
          class TraversalStride,
          class LowerSRTStride,
          class DilationStride>
CUTE_HOST_RTC
auto
make_im2col_tma_copy(CopyOp                   const& copy_op,
                     Tensor<Engine0, Layout0> const& tensor_cwhdn,
                     SLayout                  const& slayout,
                     CTATiler                 const& cta_tiler,
                     MulticastSize            const& multicast_size,
                     LowerCornerStride        const& lower_corner_whd,
                     UpperCornerStride        const& upper_corner_whd,
                     LowerPaddingStride       const& lower_padding_whd,
                     UpperPaddingStride       const& upper_padding_whd,
                     TraversalStride          const& stride_whd,
                     LowerSRTStride           const& lower_srt,
                     DilationStride           const& stride_srt)
{
  auto cta_v_tile = make_identity_layout(product_each(shape(tensor_cwhdn))).compose(cta_tiler);
  auto cta_t_tile = make_layout(multicast_size);

  return detail::make_tma_copy_im2col(copy_op, tensor_cwhdn,
                                      slayout, cta_t_tile, cta_v_tile,
                                      lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt);
}

// Explicit default for multicast_size
template <class CopyOp,
          class Engine0, class Layout0,
          class SLayout,
          class CTATiler,
          class LowerCornerStride,
          class UpperCornerStride,
          class LowerPaddingStride,
          class UpperPaddingStride,
          class TraversalStride,
          class LowerSRTStride,
          class DilationStride>
CUTE_HOST_RTC
auto
make_im2col_tma_copy(CopyOp                   const& copy_op,
                     Tensor<Engine0, Layout0> const& tensor_cwhdn,
                     SLayout                  const& slayout,
                     CTATiler                 const& cta_tiler,
                     LowerCornerStride        const& lower_corner_whd,
                     UpperCornerStride        const& upper_corner_whd,
                     LowerPaddingStride       const& lower_padding_whd,
                     UpperPaddingStride       const& upper_padding_whd,
                     TraversalStride          const& stride_whd,
                     LowerSRTStride           const& lower_srt,
                     DilationStride           const& stride_srt)
{
  return make_im2col_tma_copy(copy_op, tensor_cwhdn, slayout, cta_tiler, Int<1>{},
                              lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt);
}

// Explicit default for cta_tiler and multicast_size
template <class CopyOp,
          class Engine0, class Layout0,
          class SLayout,
          class LowerCornerStride,
          class UpperCornerStride,
          class LowerPaddingStride,
          class UpperPaddingStride,
          class TraversalStride,
          class LowerSRTStride,
          class DilationStride>
CUTE_HOST_RTC
auto
make_im2col_tma_copy(CopyOp                   const& copy_op,
                     Tensor<Engine0, Layout0> const& tensor_cwhdn,
                     SLayout                  const& slayout,
                     LowerCornerStride        const& lower_corner_whd,
                     UpperCornerStride        const& upper_corner_whd,
                     LowerPaddingStride       const& lower_padding_whd,
                     UpperPaddingStride       const& upper_padding_whd,
                     TraversalStride          const& stride_whd,
                     LowerSRTStride           const& lower_srt,
                     DilationStride           const& stride_srt)
{
  return make_im2col_tma_copy(copy_op, tensor_cwhdn, slayout, product_each(shape(slayout)), Int<1>{},
                              lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt);
}

// No offsets copy.
template <class CopyOp,
          class Engine0, class Layout0,
          class SLayout,
          class CTATiler,
          class MulticastSize>
CUTE_HOST_RTC
auto
make_im2col_tma_copy(CopyOp                   const& copy_op,
                     Tensor<Engine0, Layout0> const& tensor_cwhdn,
                     SLayout                  const& slayout,
                     CTATiler                 const& cta_tiler,
                     MulticastSize            const& multicast_size)
{
  auto cta_v_tile = make_identity_layout(product_each(shape(tensor_cwhdn))).compose(cta_tiler);
  auto cta_t_tile = make_layout(multicast_size);

  return detail::make_tma_copy_im2col(copy_op, tensor_cwhdn, slayout, cta_t_tile, cta_v_tile);
}

// Explicit default for multicast_size
template <class CopyOp,
          class Engine0, class Layout0,
          class SLayout,
          class CTATiler>
CUTE_HOST_RTC
auto
make_im2col_tma_copy(CopyOp                   const& copy_op,
                     Tensor<Engine0, Layout0> const& tensor_cwhdn,
                     SLayout                  const& slayout,
                     CTATiler                 const& cta_tiler)
{
  return make_im2col_tma_copy(copy_op, tensor_cwhdn, slayout, cta_tiler, Int<1>{});
}

// Explicit default for cta_tiler and multicast_size
template <class CopyOp,
          class Engine0, class Layout0,
          class SLayout>
CUTE_HOST_RTC
auto
make_im2col_tma_copy(CopyOp                   const& copy_op,
                     Tensor<Engine0, Layout0> const& tensor_cwhdn,
                     SLayout                  const& slayout)
{
  return make_im2col_tma_copy(copy_op, tensor_cwhdn, slayout, product_each(shape(slayout)), Int<1>{});
}

} // namespace cute
#endif
