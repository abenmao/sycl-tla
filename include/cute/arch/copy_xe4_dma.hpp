#pragma once

#include "inline_pisa.hpp"
#include "util.hpp"

namespace cute {

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
    async_tensor_load<cm_type>(tdesc_ptr, slm_space_cast(slm_ptr), gmem_ptr, coord, abar_ptr);
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
  copy(uint64_t const* tdesc_ptr, TS gmem_ptr, uint64_t const* abar_ptr, TD* slm_ptr, Coord const& coord)
  {
    async_tensor_store<cm_type>(tdesc_ptr, slm_space_cast(slm_ptr), gmem_ptr, coord, abar_ptr);
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
    async_tensor_load<cm_type>(tdesc_ptr, slm_space_cast(slm_ptr), gmem_ptr, coord, abar_ptr, multicast_mask);
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// XE4_LDSM: Initiates a slm load from shared memory to register
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int VS, class TS, class TD = TS>
struct XE4_LDSM
{
  CUTE_HOST_DEVICE static void
  copy(const TS* slm_ptr, TD* reg_ptr)
  {
    slm_vload<VS>(reg_ptr, slm_space_cast(slm_ptr));
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// XE4_STSM: Initiates a slm store from register to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int VS, class TS, class TD = TS>
struct XE4_STSM
{
  CUTE_HOST_DEVICE static void
  copy(const TS* reg_ptr, TD* slm_ptr)
  {
    slm_vstore<VS>(slm_space_cast(slm_ptr), reg_ptr);
  }
};

/// ASYNC_ROW_LOAD_IM2COL: Initiates an im2col async row copy from global memory to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////

struct ASYNC_ROW_IM2COL {};

template <typename T>
inline uint32_t get_copy_size(const uint32_t coord, const uint32_t shape, uint32_t width_2d) {
    uint32_t left_size = (shape - coord) * sizeof(T);
    uint32_t copy_size = left_size < width_2d ? left_size : width_2d;
    return copy_size;
}

template <slm_matrix_type cm_type>
struct ASYNC_ROW_LOAD_IM2COL_4D : public DMA_LOAD, public ASYNC_ROW_IM2COL
{
  template<class TS, class TG, int NumBytesPerCopy>
  CUTE_HOST_DEVICE static void
  copy(Im2ColTmaDescriptor<TG, NumBytesPerCopy> const* tma_desc, uint64_t const* abar_ptr,
                              TS const* slm_ptr,
                              int32_t crd_c, int32_t crd_w, int32_t crd_h, int32_t crd_n, int32_t crd_s, int32_t crd_r)
  {
    TG* gmem_address = reinterpret_cast<TG*>(tma_desc->bytes[0]);
    int32_t crd1 = crd_w + crd_s;
    int32_t crd2 = crd_h + crd_r;

    bool is_batch_coord_valid = (crd_n < tma_desc->bytes[4]);
    bool is_coord_valid1 = (crd1 >= 0) && (crd1 < tma_desc->bytes[2]);
    bool is_coord_valid2 = (crd2 >= 0) && (crd2 < tma_desc->bytes[3]);
    bool is_coord_valid = is_batch_coord_valid && is_coord_valid1 && is_coord_valid2;
    uint64_t offset = crd_c + crd1 * tma_desc->bytes[6] + crd2 * tma_desc->bytes[7] + crd_n * tma_desc->bytes[8];
    uint32_t copy_size = is_coord_valid ? get_copy_size<TG>(crd_c, tma_desc->bytes[1], NumBytesPerCopy) : 0;
    uint64_t offset_a64 = uint64_t(gmem_address) + offset * sizeof(TG);
    row_copy_tiled_a64_load<cm_type, NumBytesPerCopy, TG>(slm_space_cast(slm_ptr), offset_a64, copy_size, abar_ptr);
  }
};

template <slm_matrix_type cm_type>
struct ASYNC_ROW_LOAD_IM2COL : public DMA_LOAD, public ASYNC_ROW_IM2COL
{
  template<class TS, class TG, int NumBytesPerCopy, class Coord>
  CUTE_HOST_DEVICE static void
  copy(Im2ColTmaDescriptor<TG, NumBytesPerCopy> const* tma_desc, uint64_t const* abar_ptr,
                              TS const* slm_ptr, Coord coord)
  {
    using Impl = ASYNC_ROW_LOAD_IM2COL_4D<cm_type>;
    return Impl::template copy<TS, TG, NumBytesPerCopy>(
      tma_desc, abar_ptr, slm_ptr, coord[0], coord[1], coord[2], coord[3], coord[4], coord[5]);
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
  copy(Im2ColTmaDescriptor<TG, NumBytesPerCopy> const* tma_desc, uint64_t const* abar_ptr, TS const* slm_ptr,
                              int32_t crd_c, int32_t crd_w, int32_t crd_h, int32_t crd_n)
  {
    TG* gmem_address = reinterpret_cast<TG*>(tma_desc->bytes[0]);
    bool is_batch_coord_valid = (crd_n < tma_desc->bytes[4]);
    bool is_coord_valid1 = (crd_w >= 0) && (crd_w < tma_desc->bytes[2]);
    bool is_coord_valid2 = (crd_h >= 0) && (crd_h < tma_desc->bytes[3]);
    bool is_coord_valid = is_batch_coord_valid && is_coord_valid1 && is_coord_valid2;

    uint64_t offset = crd_c + crd_w * tma_desc->bytes[6] + crd_h * tma_desc->bytes[7] + crd_n * tma_desc->bytes[8];
    uint32_t copy_size = is_coord_valid ? get_copy_size<TG>(crd_c, tma_desc->bytes[1], NumBytesPerCopy) : 0;
    uint64_t offset_a64 = uint64_t(gmem_address) + offset * sizeof(TG);
    row_copy_tiled_a64_store<cm_type, NumBytesPerCopy, TG>(slm_space_cast(slm_ptr), offset_a64, copy_size, abar_ptr);
  }
};

template <slm_matrix_type cm_type>
struct ASYNC_ROW_STORE_IM2COL : public DMA_STORE, public ASYNC_ROW_IM2COL
{
  template<class TS, class TG, int NumBytesPerCopy, class Coord>
  CUTE_HOST_DEVICE static void
  copy(Im2ColTmaDescriptor<TG, NumBytesPerCopy> const* tma_desc, uint64_t const* abar_ptr,
                              TS const* slm_ptr, Coord coord)
  {

    using Impl = XE4_ASYNC_ROW_STORE_IM2COL_4D<cm_type>;
    return Impl::template copy<TS, TG, NumBytesPerCopy>(tma_desc, abar_ptr, slm_ptr, coord[0], coord[1], coord[2], coord[3]);
  }
};

} // namespace xe4
} // namespace cute
