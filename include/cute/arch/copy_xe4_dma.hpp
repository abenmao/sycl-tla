#pragma once

#include "inline_pisa.hpp"
#include "util.hpp"

namespace cute {

template <typename CopyOperation>
struct Xe4CopyOp {};

namespace xe4 {

////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_TENSOR_LOAD: Initiates a async tensor copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct DMA_LOAD {};
struct DMA_STORE {};
struct DMA_MULTICAST {};

template <slm_matrix_type cm_type>
struct ASYNC_TENSOR_LOAD : public DMA_LOAD
{
  template<class TS, class TD, class Coord>
  CUTE_HOST_DEVICE static void
  copy(uint64_t const* tdesc_ptr, TS* gmem_ptr, uint64_t const* abar_ptr, TD* slm_ptr, Coord const& coord)
  {
    constexpr int dim = 2;
    async_tensor_load<dim, cm_type>(slm_space_cast(slm_ptr), gmem_ptr, tdesc_ptr, abar_ptr, coord.data());
  }

  template<class DimIdx, class TS, class TD, class Coord>
  CUTE_HOST_DEVICE static void
  copy(uint64_t const* tdesc_ptr, TS* gmem_ptr, DimIdx const& dim_index, uint32_t dim_size, uint64_t const* abar_ptr, TD* slm_ptr, Coord const& coord)
  {
    tensordesc_set_dim_size<DimIdx::value>(tdesc_ptr, dim_size);
    copy(tdesc_ptr, gmem_ptr, abar_ptr, slm_ptr, coord);
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_TENSOR_STORE : Initiates a async tensor copy from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

template <slm_matrix_type cm_type>
struct ASYNC_TENSOR_STORE : public DMA_STORE
{
  template<class TS, class TD, class Coord>
  CUTE_HOST_DEVICE static void
  copy(uint64_t const* tdesc_ptr, TS* gmem_ptr, uint64_t const* abar_ptr, TD* slm_ptr, Coord const& coord)
  {
    constexpr int dim = 2;
    async_tensor_store<dim, cm_type>(gmem_ptr, slm_space_cast(slm_ptr), tdesc_ptr, abar_ptr, coord.data());
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_TENSOR_LOAD_MULTICAST: Initiates a async tensor copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

template <slm_matrix_type cm_type>
struct ASYNC_TENSOR_LOAD_MULTICAST : public DMA_LOAD, public DMA_MULTICAST
{
  template<class TS, class TD, class Coord>
  CUTE_HOST_DEVICE static void
  copy(uint64_t const* tdesc_ptr, TS* gmem_ptr, uint64_t const* abar_ptr, uint32_t multicast_mask, TD* slm_ptr, Coord const& coord)
  {
    constexpr int dim = 2;
    async_tensor_load<dim, cm_type>(slm_space_cast(slm_ptr), gmem_ptr, tdesc_ptr, abar_ptr, multicast_mask, coord.data());
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// SLM_VLOAD: Initiates a slm load from shared memory to register
////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t VS>
struct SLM_VLOAD
{
  template<typename SlmType, typename RegType>
  CUTE_HOST_DEVICE static void
  copy(SlmType* slm_ptr, RegType* reg_ptr)
  {
    slm_vload<VS>(reg_ptr, slm_space_cast(slm_ptr));
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// SLM_VSTORE: Initiates a slm store from register to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t VS>
struct SLM_VSTORE
{
  template<typename SlmType, typename RegType>
  CUTE_HOST_DEVICE static void
  copy(RegType* reg_ptr, SlmType* slm_ptr)
  {
    slm_vstore<VS>(slm_space_cast(slm_ptr), reg_ptr);
  }
};

/// ASYNC_ROW_LOAD_IM2COL: Initiates an im2col async row copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct ASYNC_ROW_IM2COL {};

template<typename T, int NumBytesPerCopy>
struct alignas(64) Im2ColDescriptor {
  uint64_t bytes[10];   // support from 3D tensor to 5D tensor
};

template <int NumBytesPerCopy, class GTensor>
CUTE_HOST_DEVICE auto
make_async_row_copy_desc(GTensor const& gtensor)
{
  using T = typename GTensor::value_type;

  Im2ColDescriptor<T, NumBytesPerCopy> tdesc_ptr;
  tdesc_ptr.bytes[0] = reinterpret_cast<uint64_t>(gtensor.data());
  tdesc_ptr.bytes[1] = shape<0>(layout<1>(gtensor));
  tdesc_ptr.bytes[2] = shape<0>(layout<0>(gtensor));
  tdesc_ptr.bytes[3] = shape<1>(layout<0>(gtensor));
  tdesc_ptr.bytes[4] = shape<2>(layout<0>(gtensor));

  tdesc_ptr.bytes[6] = stride<0>(layout<0>(gtensor)) * sizeof(T);
  tdesc_ptr.bytes[7] = stride<1>(layout<0>(gtensor)) * sizeof(T);
  tdesc_ptr.bytes[8] = stride<2>(layout<0>(gtensor)) * sizeof(T);

  return tdesc_ptr;
}

template <typename T>
inline uint32_t get_copy_size(const int32_t coord, const uint32_t shape, uint32_t width_2d) {
    uint32_t left_size = (shape - coord) * sizeof(T);
    uint32_t copy_size = left_size < width_2d ? left_size : width_2d;
    return copy_size;
}

template <slm_matrix_type cm_type>
struct ASYNC_ROW_LOAD_IM2COL_4D : public DMA_LOAD, public ASYNC_ROW_IM2COL
{
  template<class TS, class TG, int NumBytesPerCopy>
  CUTE_HOST_DEVICE static void
  copy(Im2ColDescriptor<TG, NumBytesPerCopy> const* tma_desc, uint64_t const* abar_ptr,
                              TS const* slm_ptr,
                              int32_t crd_c, int32_t crd_w, int32_t crd_h, int32_t crd_n, int32_t crd_s, int32_t crd_r)
  {
    constexpr uint32_t slm_stride = NumBytesPerCopy / sizeof(TG);
    TG* gmem_address = reinterpret_cast<TG*>(tma_desc->bytes[0]);
    TS const* slm_inst_ptr = slm_ptr - get_lane_id() * slm_stride;
    int32_t crd1 = crd_w + crd_s;
    int32_t crd2 = crd_h + crd_r;
    bool is_coord_valid = (crd1 >= 0) & (crd1 < tma_desc->bytes[2]);
    is_coord_valid = is_coord_valid & (crd2 >= 0) & (crd2 < tma_desc->bytes[3]);
    is_coord_valid = is_coord_valid & (crd_n >= 0) & (crd_n < tma_desc->bytes[4]);

    uint32_t offset = crd_c * sizeof(TG) + crd1 * tma_desc->bytes[6] + crd2 * tma_desc->bytes[7] + crd_n * tma_desc->bytes[8];
    offset = is_coord_valid ? offset : 0;
    uint32_t copy_size = is_coord_valid ? get_copy_size<TG>(crd_c, tma_desc->bytes[1], NumBytesPerCopy) : 0;
    uint64_t offset_a64 = reinterpret_cast<uint64_t>(gmem_address + offset / sizeof(TG));
    row_copy_tiled_a64_load<cm_type, NumBytesPerCopy, TG>(slm_inst_ptr, offset_a64, copy_size, abar_ptr);
  }
};

template <slm_matrix_type cm_type>
struct ASYNC_ROW_LOAD_IM2COL : public DMA_LOAD, public ASYNC_ROW_IM2COL
{
  template<class TS, class TG, int NumBytesPerCopy>
  CUTE_HOST_DEVICE static void
  copy(Im2ColDescriptor<TG, NumBytesPerCopy> const* tma_desc, uint64_t const* abar_ptr,
                              TS const* slm_ptr,
                              int32_t crd_c, int32_t crd_w, int32_t crd_h, int32_t crd_n,
                              int32_t crd_s, int32_t crd_r)
  {
    using Impl = ASYNC_ROW_LOAD_IM2COL_4D<cm_type>;
    return Impl::template copy<TS, TG, NumBytesPerCopy>(
      tma_desc, abar_ptr, slm_ptr, crd_c, crd_w, crd_h, crd_n, crd_s, crd_r);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// ASYNC_ROW_STORE_IM2COL: Initiates an im2col async row copy from shared memory to global memory
////////////////////////////////////////////////////////////////////////////////////////////////////

template <slm_matrix_type cm_type>
struct XE4_ASYNC_ROW_STORE_IM2COL_4D : public DMA_STORE, public ASYNC_ROW_IM2COL
{
  template<class TS, class TG, int NumBytesPerCopy>
  CUTE_HOST_DEVICE static void
  copy(Im2ColDescriptor<TG, NumBytesPerCopy> const* tma_desc, uint64_t const* abar_ptr, TS const* slm_ptr,
                              int32_t crd_c, int32_t crd_w, int32_t crd_h, int32_t crd_n)
  {
    constexpr uint32_t slm_stride = NumBytesPerCopy / sizeof(TG);
    TG* gmem_address = reinterpret_cast<TG*>(tma_desc->bytes[0]);
    TS const* slm_inst_ptr = slm_ptr - get_lane_id() * slm_stride;
    bool is_coord_valid = (crd_w >= 0) & (crd_w < tma_desc->bytes[2]);
    is_coord_valid = is_coord_valid & (crd_h >= 0) & (crd_h < tma_desc->bytes[3]);
    is_coord_valid = is_coord_valid & (crd_n >= 0) & (crd_n < tma_desc->bytes[4]);

    uint32_t offset = crd_c * sizeof(TG) + crd_w * tma_desc->bytes[6] + crd_h * tma_desc->bytes[7] + crd_n * tma_desc->bytes[8];
    offset = is_coord_valid ? offset : 0;
    uint32_t copy_size = is_coord_valid ? get_copy_size<TG>(crd_c, tma_desc->bytes[1], NumBytesPerCopy) : 0;
    uint64_t offset_a64 = reinterpret_cast<uint64_t>(gmem_address + offset / sizeof(TG));
    row_copy_tiled_a64_store<cm_type, NumBytesPerCopy, TG>(slm_inst_ptr, offset_a64, copy_size, abar_ptr);
  }
};

template <slm_matrix_type cm_type>
struct ASYNC_ROW_STORE_IM2COL : public DMA_STORE, public ASYNC_ROW_IM2COL
{
  template<class TS, class TG, int NumBytesPerCopy>
  CUTE_HOST_DEVICE static void
  copy(Im2ColDescriptor<TG, NumBytesPerCopy> const* tma_desc, uint64_t const* abar_ptr,
                              TS const* slm_ptr,
                              int32_t crd_c, int32_t crd_w, int32_t crd_h, int32_t crd_n)
  {
    using Impl = XE4_ASYNC_ROW_STORE_IM2COL_4D<cm_type>;
    return Impl::template copy<TS, TG, NumBytesPerCopy>(tma_desc, abar_ptr, slm_ptr, crd_c, crd_w, crd_h, crd_n);
  }
};

} // namespace xe4


template <class CopyOp>
struct XE4_COPY_Unpack
{
  template <class... Args,
            class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout>           const& src,
              Tensor<TD,DLayout>                & dst)
  {
    constexpr auto isLoadOperation = !cute::is_base_of<xe4::DMA_STORE, CopyOp>::value;

    if constexpr (isLoadOperation) {
      auto dst_ptr = cute::raw_pointer_cast(dst.data());
      auto src_coord = cute::to_array<int32_t>(src.data().coord_);
      return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                  traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                  make_tuple(dst_ptr, src_coord), seq<0, 1>{});
    } else {
      auto src_ptr = cute::raw_pointer_cast(src.data());
      auto dst_coord = cute::to_array<int32_t>(dst.data().coord_);
      return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                                  traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
                                  make_tuple(src_ptr, dst_coord), seq<0, 1>{});
    }
  }
};

} // namespace cute