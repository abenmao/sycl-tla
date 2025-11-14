#pragma once

#include <cute/config.hpp>
#include <cute/pointer_sparse.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/arch/mma_xe4_desc.hpp>
#include <cute/arch/mma_xe4.hpp>

namespace cute {
namespace AMMA {

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

}

// Customization point for creating a AMMA::smem_desc Tensor
template <AMMA::Major major, bool is_A>
struct MakeTensor<AMMA::smem_desc<major, is_A>>
{
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    constexpr auto layout = decltype(recast<uint8_t const>(smem_tensor).layout()) {};
    constexpr auto coreMatrixStride = [Dim0 = size<0>(layout)](auto stride) {
      constexpr int Cols = 32;
      constexpr int TotalBytes = 1024;
      constexpr int RowBytes = TotalBytes / Cols;
      constexpr int Ret = (stride >= Dim0) ? stride : RowBytes * stride;
      return C<Ret>{};
    };

    constexpr auto stride = layout.stride();
    constexpr auto stride1 = coreMatrixStride(get<1>(stride));
    constexpr auto stride2 = coreMatrixStride(get<2>(stride));

    // TODO: Enable this test and use 'make_smem_ptr'
    // static_assert(is_smem<TEngine>::value, "Expected SMEM Tensor to construct a AMMA Desc Tensor");
    return make_tensor(
        AMMA::DescriptorIterator{
          AMMA::make_matrix_desc<major, is_A>(tensor<0>(smem_tensor))
        }, 
        make_layout(
          tuple_cat(make_tuple(_1{}), take<1, -1>(shape(layout))),
          tuple_cat(make_tuple(_0{}, stride1, stride2), take<3, -1>(stride))
        )
    );
  }
};

}
