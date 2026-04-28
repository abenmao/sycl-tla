#pragma once

#include <sycl/sycl.hpp>

#include <cute/arch/mma_xe4_desc.hpp>
#include <cute/arch/copy_xe4_adma.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/atom/copy_atom.hpp>

#include <cute/atom/copy_traits_sm90_tma.hpp>

#include <cute/layout.hpp>

namespace cute {

template <class CopyOp>
struct ADMA_LOAD_Unpack {
  template <class... Args,
           class TS, class SLayout,
           class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(
      Copy_Traits<CopyOp, Args...> const& traits,
      Tensor<TS, SLayout>          const& src,
      Tensor<TD, DLayout>               & dst) {
    static_assert(is_smem<TD>::value, "ASYNCL_ADMA_LOAD requires the destination be shared memory.");

    auto src_coord = src.data().coord_;
    auto* dst_ptr = cute::raw_pointer_cast(dst.data());
#if 0
    auto [c0,c1,c2,c3,c4] = append<5>(src_coord, 0);
    // TODO: change it to sycl concepts
    printf("THR (%d,%d,%d) BLK (%d,%d,%d) TMACRD (%d,%d,%d,%d,%d) SMEMADDR (%p)\n",
          threadIdx.x, threadIdx.y, threadIdx.z,
          blockIdx.x, blockIdx.y, blockIdx.z,
          int32_t(c0), int32_t(c1), int32_t(c2), int32_t(c3), int32_t(c4), dst_ptr);
#endif
    return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                make_tuple(dst_ptr, src_coord), seq<0, 1>{});
  }
};

template <class CopyOp>
struct Obsolete_XE4_COPY_Unpack
{
  template <class... Args,
            class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    constexpr auto isLoadOperation = !cute::is_base_of<XE4_ASYNC_TENSOR_STORE, CopyOp>::value;
    constexpr auto isIm2ColOperation = false;

    auto as_xe4_coord = [](auto const& t) {
      return to_vec<int32_t>(flatten_to_tuple(t));
    };

    if constexpr (isLoadOperation) {
      auto dst_ptr = cute::raw_pointer_cast(dst.data());
      if constexpr(isIm2ColOperation) {
        auto src_coord = as_xe4_coord(src(Int<0>{}));
        return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                    traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                    make_tuple(dst_ptr, src_coord), seq<0, 1>{});
      } else {
        auto src_coord = as_xe4_coord(src.data().coord_);
        return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                    traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                    make_tuple(dst_ptr, src_coord), seq<0, 1>{});
      }
    } else {
      auto src_ptr = cute::raw_pointer_cast(src.data());
      if constexpr(isIm2ColOperation) {
        auto dst_coord = as_xe4_coord(take<0,3>(dst(Int<0>{})));
        return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                    traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                    make_tuple(src_ptr, dst_coord), seq<0, 1>{});
      } else {
        auto dst_coord = as_xe4_coord(dst.data().coord_);
        return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                    traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                    make_tuple(src_ptr, dst_coord), seq<0, 1>{});
      }
    }
  }
};

template <typename CopyOperation>
struct Obsolete_Xe4CopyOp {};

template <typename CopyOperation>
struct Obsolete_Xe4CopyOpWrapper : CopyOperation {};

template <class GmemDetails, class AuxParams, class GmemPtr>
struct Obsolete_Xe4DmaCache {
  template <typename CopyOp>
  using OpUnpack = Obsolete_XE4_COPY_Unpack<CopyOp>;

  Obsolete_Xe4DmaCache() = default;

  Obsolete_Xe4DmaCache(
      GmemDetails const& gmem_details, AuxParams const& aux_params, GmemPtr gmem_ptr)
    : gmem_details_(gmem_details), aux_params_(aux_params), gmem_ptr_(gmem_ptr) {}

  CUTE_DEVICE void
  set_gmem_ptr(GmemPtr gmem_ptr) {
    gmem_ptr_ = gmem_ptr;
  }

  CUTE_DEVICE void
  set_tensor_desc(uint64_t* tensor_desc) const {
    constexpr int tma_dim = rank_v<typename AuxParams::TmaGmemBasis>;
    auto [gmem_shape, gmem_stride, roi_shape, element_stride, matrix_desc] = gmem_details_;

    tensordesc_fill_dim_size<tma_dim>(tensor_desc, gmem_shape);
    tensordesc_fill_dim_stride<tma_dim>(tensor_desc, gmem_stride);
    tensordesc_fill_roitensor_dim_size<tma_dim>(tensor_desc, roi_shape);
    tensordesc_fill_element_stride<tma_dim>(tensor_desc, element_stride);

    tdesc_ptr_ = tensor_desc;
    slm_desc = matrix_desc;
  }

  CUTE_HOST_DEVICE constexpr
  auto get_tensor_desc() const {
    return tdesc_ptr_;
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  template <typename... Args>
  CUTE_HOST_DEVICE constexpr
  auto make_args_tuple(Args&&... args) const {
    return make_tuple(tdesc_ptr_, gmem_ptr_, slm_desc, static_cast<Args&&>(args)...);
  }

  GmemDetails gmem_details_;
  AuxParams aux_params_;
  GmemPtr gmem_ptr_ {nullptr};
  mutable uint64_t* tdesc_ptr_ { nullptr };
  mutable uint32_t slm_desc;
};

template <class CopyOperation, class NumBitsPerTMA, class DmaCache>
struct Copy_Traits<Obsolete_Xe4CopyOp<CopyOperation>, NumBitsPerTMA, DmaCache>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  DmaCache cache_;

  template<class ABarrier>
  CUTE_HOST_DEVICE constexpr
  auto with(ABarrier * abar_ptr, [[maybe_unused]] uint32_t const& multicast_mask = 0) const {
    using Wrapper = Obsolete_Xe4CopyOpWrapper<CopyOperation>;
    using OpUnpack = typename DmaCache::template OpUnpack<Wrapper>;

    auto opargs = cache_.make_args_tuple(abar_ptr, multicast_mask);
    return Copy_Traits<Wrapper, NumBitsPerTMA, decltype(opargs), OpUnpack>{opargs};
  }

  template<class ABarrier, class DimIndex>
  CUTE_HOST_DEVICE constexpr
  auto with(DimIndex const& dim_index, uint32_t const& dim_size, ABarrier * abar_ptr, [[maybe_unused]] uint32_t const& multicast_mask = 0) const {
    using Wrapper = Obsolete_Xe4CopyOpWrapper<CopyOperation>;
    using OpUnpack = typename DmaCache::template OpUnpack<Wrapper>;

    auto opargs = cache_.make_args_tuple(dim_index, dim_size, abar_ptr);
    return Copy_Traits<Wrapper, NumBitsPerTMA, decltype(opargs), OpUnpack>{opargs};
  }

  CUTE_HOST_DEVICE constexpr
  auto get_tensor_desc() const {
    return cache_.get_tensor_desc();
  }

  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto get_tma_tensor(GShape const& g_shape) const {
    return cache_.get_tma_tensor(g_shape);
  }

  // Don't try to execute a copy with XE4_TMA_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

template <class CopyOperation, class NumBitsPerTMA, class OpArgsTuple, template<class> class OpUnpack>
struct Copy_Traits<Obsolete_Xe4CopyOpWrapper<CopyOperation>, NumBitsPerTMA, OpArgsTuple, OpUnpack<Obsolete_Xe4CopyOpWrapper<CopyOperation>>> : OpUnpack<Obsolete_Xe4CopyOpWrapper<CopyOperation>>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
  using DstLayout = SrcLayout;
  using RefLayout = SrcLayout;

  OpArgsTuple const opargs_;

  Copy_Traits(OpArgsTuple const& opargs) : opargs_(opargs) {}
};

namespace detail {

// Use a sidx2gmode to read through the GMEM tensor
//   and construct a TMA Descriptor for the resulting instruction
// At the same time, construct the Tma Tensor's Stride to generate
//   the TMA coordinates that the instruction consumes.
//
template <class InternalType,
          class GEngine, class GLayout,
          class TShape, class TStride>
CUTE_HOST_RTC
auto
make_adma_copy_desc(
    Tensor<GEngine,GLayout> const& gtensor,         // The original GMEM Tensor
    Layout<TShape,TStride>  const& adma_gbasis,     // ADMA mode -> GMEM mode mapping
    uint32_t                       matrix_desc,
    uint32_t                       num_multicast)   // The number of CTAs in multicasting
{
  //
  // Tensor desc creation
  //

  constexpr int t_dim = decltype(rank(adma_gbasis))::value;

  //
  // Tensor Descriptor info
  //

  // Recast the original tensor for shape/stride inspections
  Tensor gtensor_T = recast<InternalType>(gtensor);

  void* gmem_address = (void*) raw_pointer_cast(gtensor_T.data());
  auto  gmem_layout  = gtensor_T.layout();

  cute::array<uint64_t, 5> gmem_prob_shape  = {1,1,1,1,1};
  cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};

  fill_tma_gmem_shape_stride(gtensor_T, stride(adma_gbasis), gmem_prob_shape, gmem_prob_stride);

#if 0
  assert((reinterpret_cast<uint64_t>(gmem_address) & 0b1111) == 0);  // Address must be 16B-aligned

  assert(gmem_prob_shape[0] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[0] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[1] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[1] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[2] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[2] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[3] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[3] <= (uint64_t(1) << 32));         // Size must be max 2^32
  assert(gmem_prob_shape[4] >= (uint64_t(1)));               // Size must be min 1
  assert(gmem_prob_shape[4] <= (uint64_t(1) << 32));         // Size must be max 2^32

  // TMA descriptor does not store the zeroth stride and assumes it is 1 (TmaInternalType element).
  assert(gmem_prob_stride[0] == 1 && "Majorness of smem doesn't match majorness of gmem");
#endif

  // convert strides to byte strides
  for(uint64_t& stride : gmem_prob_stride) {
    stride = (stride * sizeof_bits_v<InternalType>) / 8;
  }

#if 0
  // Assert the byte strides. Tma Descriptor uses byte strides
  assert((gmem_prob_stride[1]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[1] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[2]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[2] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[3]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[3] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[4]) < (uint64_t(1) << 40));       // Stride must be max 2^40
  assert((gmem_prob_stride[4] & 0b1111) == 0);               // Stride must be multiple of 16B (128b)
#endif

  //
  // TMA smem desc info
  //
  sycl::marray<uint32_t, t_dim> smem_box_shape(uint32_t(1));
  sycl::marray<uint32_t, t_dim> smem_box_stride(uint32_t(1));

  // The smem box is simply given by the sizes of the modes in adma_gbasis
  for_each(make_seq<t_dim>{}, [&](auto i) {
    smem_box_shape[i] *= size<i>(adma_gbasis);
  });
  // Finally, truncate the tma box by the num_multicast
  for (uint32_t i = t_dim-1, multicast = num_multicast; multicast > 1; --i) {
#if 0
    assert(smem_box_shape[i] % multicast == 0 || multicast % smem_box_shape[i] == 0);
#endif
    uint32_t new_mult = ceil_div(multicast, smem_box_shape[i]);
    smem_box_shape[i] = ceil_div(smem_box_shape[i], multicast);
    multicast = new_mult;
  }

#if 0
  assert(smem_box_shape[0] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[0] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[1] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[1] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[2] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[2] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[3] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[3] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256
  assert(smem_box_shape[4] >= (uint32_t(1)));                // Size must be min 1
  assert(smem_box_shape[4] <= (uint32_t(1) << 8));           // Size must be max 2^8 = 256

  assert(smem_box_stride[0] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[0] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[1] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[1] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[2] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[2] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[3] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[3] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
  assert(smem_box_stride[4] >= (uint32_t(1)));               // Stride must be min 1
  assert(smem_box_stride[4] <= (uint32_t(8)));               // Stride must be max 2^3 = 8
#endif

    //
    // Construct the descriptor
    //

    uint64_t* tma_desc{};

    //
    // TMA general info
    //

#if 0

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
        t_dim,
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
                << "\ndim            " << t_dim
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
                                        sizeof_bits<InternalType>{});

  auto gbasis = make_basis_like(shape(gtensor));

  // Finally, get the inverse permutation of the E<i> bases for the mocked gmem stride
  auto gmem_tma_basis_stride = transform_leaf(gbasis, [&](auto ei) {
    auto si = basis_get(ei,  shape(gmem_layout));
    auto di = basis_get(ei, stride(gmem_layout));
    if constexpr (is_constant<1, decltype(si)>::value || is_constant<0, decltype(di)>::value) {
      return Int<0>{};                  // If size-1 or stride-0, return arithmetic identity -- no contribution to the TMA
    } else {
      auto tma_gmem_basis_stride = stride(adma_gbasis);
      // Find j such that E<i> is in stride<j>(adma_gbasis)
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
        int32_t scale = ceil_div(int32_t(di * sizeof_bits_v<InternalType> / cute::max(gmem_prob_stride[j], uint64_t{16})), 8);
        return E<j>{} * scale;         // Return TMA Coord basis -- with a dynamic scale factor
      }
    }
  });

#if 0
    print("gmem_tma_basis_stride : "); print(gmem_tma_basis_stride); print("\n");
#endif

  using AuxParams = AuxTmaParams<decltype(gmem_tma_basis_stride),
                                 decltype(adma_gbasis),
                                 Swizzle<0, 4, 3>>; // XXX: do we need this?

  // Carry tensor descriptor message out of here.
  sycl::marray<uint32_t, t_dim> gmem_shape;
  for_each(make_seq<t_dim>{}, [&](auto i) {gmem_shape[i] = gmem_prob_shape[i];});
  sycl::marray<uint64_t, t_dim-1> gmem_stride;
  for_each(make_seq<t_dim-1>{}, [&](auto i) {gmem_stride[i] = gmem_prob_stride[i+1];});
  auto tma_desc_details = make_tuple(
      gmem_shape, gmem_stride, smem_box_shape, smem_box_stride, matrix_desc);
  return cute::make_tuple(tma_desc_details, AuxParams{gmem_tma_basis_stride});
  // return cute::make_tuple(tma_desc, AuxParams{gmem_tma_basis_stride});
}

template <class InternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class VShape, class VStride>
CUTE_HOST_RTC
auto
make_adma_copy_atom(
    CopyOp,
    Tensor<GEngine,GLayout> const& gtensor,       // Full GMEM Tensor
    SLayout                 const& slayout,       // Group Tile of SMEM, potentially swizzled
    uint32_t                const& num_multicast, // The number of Groups involved in multicasting
    uint32_t                const& matrix_desc,
    Layout<VShape,VStride>  const& cta_v_map)     // V: CTA val idx -> gmem mode
{
  //
  // AMMA truncated layout
  //

  // auto smem_swizzle = get_swizzle_portion(slayout);
  auto smem_layout  = get_nonswizzle_portion(slayout);

  auto adma_gbasis = detail::construct_tma_gbasis<InternalType>(gtensor, smem_layout, cta_v_map);

  //
  // Construct the AMMA Desc and the strides of the AMMA Tensor
  //

  auto [tma_desc, aux_params] = detail::make_adma_copy_desc<InternalType>(
      gtensor, adma_gbasis, matrix_desc, num_multicast);

  //
  // Construct the Copy_Traits
  //

  constexpr int num_bits_per_tma = size(adma_gbasis) * sizeof_bits_v<InternalType>;
#if defined(SYCL_INTEL_XE4_TARGET)
  auto gmem_ptr = cute::raw_pointer_cast(recast<InternalType>(gtensor).data());
  using DmaCache = Obsolete_Xe4DmaCache<decltype(tma_desc), decltype(aux_params), decltype(gmem_ptr)>;
  using Traits = Copy_Traits<Obsolete_Xe4CopyOp<CopyOp>, cute::C<num_bits_per_tma>, DmaCache>;
  using Atom   = Copy_Atom<Traits, typename GEngine::value_type>;

  Traits tma_traits{{tma_desc, aux_params, gmem_ptr}};

#else
  using Traits = Copy_Traits<CopyOp, cute::C<num_bits_per_tma>, decltype(aux_params)>;
  using Atom   = Copy_Atom<Traits, typename GEngine::value_type>;

  Traits tma_traits{tma_desc, aux_params};

#endif

#if 0
  print("num_bits_per_tma :  "); print(num_bits_per_tma); print("\n");
  print("g_stride_bases   :  "); print(tma_traits.aux_params_.g_stride_); print("\n");
#endif

  // Return the Copy_Atom
  return Atom{tma_traits};
}

template <class SLayout>
CUTE_HOST
auto
make_matrix_descriptor(
    SLayout const& slayout, bool is_A_matrix = false
) {
  auto strides = slayout.stride();

  MatrixDescriptor matrix_desc{};
  if (is_A_matrix) {
    matrix_desc.Type = get<1>(strides) == 1 /* runtime check */?
      MatrixDescriptor::Type1 : MatrixDescriptor::Type2;
  } else {
    matrix_desc.Type = MatrixDescriptor::Type1;
  }

  matrix_desc.Pitch = get<1>(strides) == 1 ?
    get<0>(strides) >> 2 : get<1>(strides) >> 2;

  return matrix_desc.raw_;
}

// The "logical TMA tid" is a map from the CTA rank to its logical id
// within the instruction.  It works like a mask or ordering on the
// CTAs.  For non-multicast TMA, all CTAs should map to 0.  For
// multicast TMA of size 4, CTAs will be mapped to {0,1,2,3}.
template <class InternalType,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class TShape, class TStride,
          class VShape, class VStride>
CUTE_HOST_RTC
auto
make_adma_copy_tiled(
    CopyOp                  const& copy_op,
    Tensor<GEngine,GLayout> const& gtensor,     // Full GMEM Tensor
    SLayout                 const& slayout,     // CTA Tile of SMEM
    Layout<TShape,TStride>  const& cta_t_map,   // T: CTA thr idx -> logical TMA tid
    Layout<VShape,VStride>  const& cta_v_map)   // V: CTA val idx -> gmem mode
{
  auto matrix_desc = make_matrix_descriptor(coalesce(slayout));
  Copy_Atom atom = make_adma_copy_atom<InternalType>(
      copy_op, gtensor, slayout, cosize(cta_t_map), matrix_desc, cta_v_map);

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

}

template <class InternalType = void,
         class CopyOp,
         class GEngine, class GLayout,
         class SLayout,
         class MMA_Tiler,
         class... Args,
         class ClusterShapeVMNK>
CUTE_HOST
auto
make_adma_atom_A_xe4(
    CopyOp                   const& copy_op,
    Tensor<GEngine, GLayout> const& gtensor,        // (M, K, ...)
    SLayout                  const& slayout,        // (MMA, MMA_M, MMA_K, ...)
    MMA_Tiler                const& mma_tiler,      // (TILE_M, TILE_N, TILE_K, ...)
    TiledMMA<Args...>        const& mma,
    ClusterShapeVMNK         const& cluster_shape   // (CTA_V, CTA_M, CTA_N, CTA_K)
) {
  // Keep only MK modes from MNK
  auto mma_tiler_mk = remove<1>(mma_tiler);

  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(mma_tiler_mk);

  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_A(g_tile))(_, repeat<rank(g_tile)>(_));

  auto matrix_desc = detail::make_matrix_descriptor(layout<0>(slayout), true);

#if 0
  print("(tma_a) slayout:      "); print(slayout);      print("\n");
  print("(tma_a) mma_tiler_nk: "); print(mma_tiler_nk); print("\n");
  print("(tma_a) g_tile:       "); print(g_tile);       print("\n");
  print("(tma_a) mma_tiler:    "); print(mma_tiler);    print("\n");
  print("(tma_a) cta_v_tile:   "); print(cta_v_tile);   print("\n");
#endif

  auto num_multicast = [&]() {
    if constexpr (is_same_v<CopyOp, XE4_ASYNC_TENSOR_LOAD_MULTICAST>)
      return size<2>(cluster_shape);
    else if constexpr (is_same_v<CopyOp, XE4_ASYNC_TENSOR_LOAD>)
      return Int<1>{};
    else
      static_assert(dependent_false<CopyOp>, "Unsupported CopyOp");
  }();

  using AmmaType = conditional_t<is_same<void, InternalType>::value, typename GEngine::value_type, InternalType>;
  return detail::make_adma_copy_atom<AmmaType>(
      copy_op, gtensor, slayout, num_multicast, matrix_desc, cta_v_tile);
}

template <class InternalType = void,
         class CopyOp,
         class GEngine, class GLayout,
         class SLayout,
         class MMA_Tiler,
         class... Args,
         class ClusterShapeVMNK>
CUTE_HOST
auto
make_adma_atom_B_xe4(
    CopyOp                   const& copy_op,
    Tensor<GEngine, GLayout> const& gtensor,        // (M, K, ...)
    SLayout                  const& slayout,        // (MMA, MMA_M, MMA_K, ...)
    MMA_Tiler                const& mma_tiler,      // (TILE_M, TILE_N, TILE_K, ...)
    TiledMMA<Args...>        const& mma,
    ClusterShapeVMNK         const& cluster_shape   // (CTA_V, CTA_M, CTA_N, CTA_K)
) {
  // Keep only NK modes from MNK
  auto mma_tiler_nk = remove<0>(mma_tiler);

  // cluster tile coord -> gtensor coord
  auto g_tile = make_identity_layout(shape(gtensor)).compose(mma_tiler_nk);

  // cta val idx -> gmem mode
  auto cta_v_tile = layout<1>(mma.thrfrg_B(g_tile))(_, repeat<rank(g_tile)>(_));

  auto matrix_desc = detail::make_matrix_descriptor(layout<0>(slayout), false);

#if 0
  print("(tma_b) slayout:      "); print(slayout);      print("\n");
  print("(tma_b) mma_tiler_nk: "); print(mma_tiler_nk); print("\n");
  print("(tma_b) g_tile:       "); print(g_tile);       print("\n");
  print("(tma_b) mma_tiler:    "); print(mma_tiler);    print("\n");
  print("(tma_b) cta_v_tile:   "); print(cta_v_tile);   print("\n");
#endif

  auto num_multicast = [&]() {
    if constexpr (is_same_v<CopyOp, XE4_ASYNC_TENSOR_LOAD_MULTICAST>)
      return size<1>(cluster_shape);
    else if constexpr (is_same_v<CopyOp, XE4_ASYNC_TENSOR_LOAD>)
      return Int<1>{};
  }();

  using AmmaType = conditional_t<is_same<void, InternalType>::value, typename GEngine::value_type, InternalType>;
  return detail::make_adma_copy_atom<AmmaType>(
      copy_op, gtensor, slayout, num_multicast, matrix_desc, cta_v_tile);
}

template <class InternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size>
CUTE_HOST_RTC
auto
make_adma_copy(CopyOp                 const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout,
              CTA_Tiler               const& cta_tiler,
              Cluster_Size            const& cluster_size)
{
  if constexpr (false) { // TODO: Add im2col support
    return make_im2col_tma_copy(copy_op,
                                gtensor,
                                slayout,
                                cta_tiler,
                                cluster_size);
  } else {
    auto cta_v_tile = make_identity_layout(shape(gtensor)).compose(cta_tiler);
    auto cta_t_tile = make_layout(cluster_size);
    // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
    using AdmaType = conditional_t<is_same<void, InternalType>::value, typename GEngine::value_type, InternalType>;
    return detail::make_adma_copy_tiled<AdmaType>(copy_op,
                                                gtensor, slayout,
                                                cta_t_tile, cta_v_tile);
  }
}

}
