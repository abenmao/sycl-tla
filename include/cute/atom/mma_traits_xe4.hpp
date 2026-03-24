#pragma once

#include <cute/config.hpp>
#include <cute/pointer_sparse.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/arch/mma_xe4_desc.hpp>
#include <cute/arch/mma_xe4.hpp>

namespace cute {
namespace AMMA {

// Standard Rank-2 tensor descriptor //
template<typename T, int M, int K, int FullK, typename tdesc_ptr_t=uint64_t*>
CUTE_HOST_DEVICE
void make_local_tile_tensor_descriptor(tdesc_ptr_t tdesc) {
  using namespace cute;

  uint32_t dim_sizes[2] = {M, K};
  tensordesc_fill_dim_size<2>(tdesc, dim_sizes);

  uint64_t dim_strides[1] = { FullK*sizeof(T) }; 
  tensordesc_fill_dim_stride<2>(tdesc, dim_strides); 

  // element_stride: packed stride within each dimension (1 means contiguous)
  uint32_t elem_strides[2] = { 1, 1 };
  tensordesc_fill_element_stride<2>(tdesc, elem_strides);

  // roitensor_dim_size: how much of the tensor to actually copy (= full tile)
  uint32_t roi_sizes[2] = { M, K };
  tensordesc_fill_roitensor_dim_size<2>(tdesc, roi_sizes);
}

template <Major major, bool is_A, class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
MatrixDescriptor
make_matrix_desc(Tensor<TEngine, TLayout> const& tensor) {
  // static_assert(is_smem<TEngine>::value, "AMMA Descriptors can only be constructed on smem.");
  static_assert(TLayout::rank == 2, "AMMA Descriptors can only be constructed on rank-2 tensors.");
  using value_type = typename TEngine::value_type;

  MatrixDescriptor desc;
  uint32_t start_address = cast_smem_ptr_to_uint(raw_pointer_cast(tensor.data()));
  desc.StartAddress = start_address>>9;

  constexpr auto minorDim =
    cute::is_constant<1, decltype(front(get<0>(tensor.stride())))>::value ? 1 : 0;
  static_assert(minorDim == (int)major, "Layout conflicts with Major");

  desc.Pitch = size<minorDim>(tensor.stride()) >> 2;
  desc.Type = is_A && major == Major::MN ?
    MatrixDescriptor::Type2 : MatrixDescriptor::Type1;

  return desc;
}

template <Major major, class T>
CUTE_HOST_DEVICE constexpr
MatrixDescriptor
make_matrix_desc(T const& tensor) {
  using Engine = typename T::engine_type;
  using Layout = typename T::layout_type;
  return make_matrix_desc<major, false, Engine, Layout>(tensor);
}

// Create a Type3 (scale factor) MatrixDescriptor from a rank-2 SMEM tensor.
// SF layout is always MN-major: dim-0 (MN) has unit stride, dim-1 (K/VS) stride
// gives the pitch.  The hardware uses Type3 to distinguish SF tiles from data tiles.
template <class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
MatrixDescriptor
make_sf_matrix_desc(Tensor<TEngine, TLayout> const& tensor) {
  static_assert(TLayout::rank == 2, "SF descriptors require rank-2 tensors.");

  // Assert MN-major: first-dimension stride must be 1
  static_assert(
    cute::is_constant<1, decltype(front(get<0>(tensor.stride())))>::value,
    "Scale factor layout must be MN-major (first dimension stride == 1)");

  MatrixDescriptor desc;
  uint32_t start_address = cast_smem_ptr_to_uint(raw_pointer_cast(tensor.data()));
  desc.StartAddress = start_address >> 9;

  // Pitch = number of MN elements per column >> 2.
  // For MN-major unit-stride layouts, the leading dimension equals size<0>.
  desc.Pitch = size<0>(tensor) >> 2;
  desc.Type  = MatrixDescriptor::Type3;

  return desc;
}

struct DescriptorIterator
{
  using reference    = MatrixDescriptor;
  using element_type = MatrixDescriptor;
  using value_type   = MatrixDescriptor;

  MatrixDescriptor desc_;

  // Dereference returns the MatrixDescriptor
  CUTE_HOST_DEVICE constexpr
  reference operator*() const { return desc_; }

  // Advance and return a new MatrixDescriptor
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  reference operator[](Index const& i) const { return *(*this + i); }

  // Return an advanced iterator
  template <class Index>
  CUTE_HOST_DEVICE constexpr
  DescriptorIterator operator+(Index const& offset) const
  {
    MatrixDescriptor ret;
    ret.raw_ = desc_.raw_ + uint32_t(offset >> 9);
    return { ret };
  }
};

CUTE_HOST_DEVICE constexpr
MatrixDescriptor
raw_pointer_cast(DescriptorIterator const& ptr) {
  return ptr.desc_;
}

CUTE_HOST_DEVICE void
print(DescriptorIterator const&) {
  printf("AMMA::DescriptorIterator");
}

template <AMMA::Major, bool = false>
struct smem_desc : DescriptorIterator {};

// Scale-factor fragment descriptor (type-3 MatrixDescriptor, always MN-major pitch)
struct smem_sf_desc : DescriptorIterator {};

} // namespace AMMA

// Customization point for creating a AMMA::smem_desc Tensor
template <AMMA::Major major, bool is_A>
struct MakeTensor<AMMA::smem_desc<major, is_A>> {
  template <int Dim0, int Stride> constexpr auto coreMatrixStride() {
    constexpr int Cols = 32;
    constexpr int TotalBytes = 1024;
    constexpr int RowBytes = TotalBytes / Cols;
    constexpr int Ret = (Stride >= Dim0) ? Stride : RowBytes * Stride;
    return C<Ret>{};
  }

  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine, TLayout> const &smem_tensor) {
    constexpr auto layout =
        decltype(recast<uint8_t const>(smem_tensor).layout()){};
    static_assert(decltype(layout)::rank >= 3,"MakeTensor<AMMA::smem_desc> requires a (recasted) layout of rank >= 3.");

    constexpr auto stride = layout.stride();
    constexpr int S1 = get<1>(layout.stride());
    constexpr int S2 = get<2>(layout.stride());
    using Stride1 = decltype(coreMatrixStride<size<0>(layout), S1>());
    using Stride2 = decltype(coreMatrixStride<size<0>(layout), S2>());

    // TODO: Enable this test and use 'make_smem_ptr'
    // static_assert(is_smem<TEngine>::value, "Expected SMEM Tensor to construct a AMMA Desc Tensor");
    return make_tensor(
        AMMA::DescriptorIterator{
          AMMA::make_matrix_desc<major, is_A>(tensor<0>(smem_tensor))
        },
        make_layout(
          tuple_cat(make_tuple(_1{}), take<1, -1>(shape(layout))),
          tuple_cat(make_tuple(_0{}, Stride1{}, Stride2{}), take<3, -1>(stride))
        )
    );
  }
};

// Customization point for creating an AMMA::smem_sf_desc Tensor (type-3 descriptor).
// Scale-factor descriptors are always MN-major; Pitch is derived from the stride of
// the second dimension (K/VS) of the SF smem tensor.  Unlike data descriptors, no
// Type1-vs-Type2 branching is needed — Type3 is unconditional.
//
// The input smem_tensor is expected to be rank-3+ (partitioned SF tile):
//   mode-0: the 2D SF tile  (MN, K/VS)
//   mode-1+: iteration / pipeline dimensions
// tensor<0>(smem_tensor) is passed to make_sf_matrix_desc to build the first
// descriptor; higher-mode byte strides drive DescriptorIterator advancement.
template <>
struct MakeTensor<AMMA::smem_sf_desc> {
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine, TLayout> const &smem_tensor) {
    static_assert(TLayout::rank >= 3,
      "smem_sf_desc requires a rank-3+ SF tensor: (SF_tile, iter, pipe, ...)");
    // Recast to byte layout for stride computation.
    // SF SMEM layouts may have hierarchical (nested-tuple) strides at
    // iteration modes (e.g. (blk_MN, blk_K)), so we forward them directly
    // via take<1,-1> rather than extracting as scalar ints.
    constexpr auto layout =
        decltype(recast<uint8_t const>(smem_tensor).layout()){};
    static_assert(decltype(layout)::rank >= 3,
      "MakeTensor<AMMA::smem_sf_desc> requires a (recasted) layout of rank >= 3.");

    // SF byte strides are used directly — no core-matrix adjustment is needed
    // because SF tiles use simple MN-major layout without sub-tile blocking.
    return make_tensor(
        AMMA::DescriptorIterator{
          AMMA::make_sf_matrix_desc(tensor<0>(smem_tensor))
        },
        make_layout(
          tuple_cat(make_tuple(_1{}), take<1, -1>(shape(layout))),
          tuple_cat(make_tuple(_0{}), take<1, -1>(stride(layout)))
        )
    );
  }
};

} // namespace cute
