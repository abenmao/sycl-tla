#pragma once

#include <cute/arch/asm_helper.hpp>
#include <cute/arch/mma_xe4_desc.hpp>

namespace sycl {
#ifdef __SYCL_DEVICE_ONLY__
  template <class T, int N> using vec_t = T __attribute__((ext_vector_type(N)));
#else
  template <class T, int N> using vec_t = sycl::marray<T, N>;
#endif
}

namespace cute {
namespace detail {

template <typename T, class... Ts, size_t... I>
sycl::vec_t<T, sizeof...(Ts)>
to_vec_impl(cute::tuple<Ts...> const& t, std::index_sequence<I...>) {
    return sycl::vec_t<T, sizeof...(Ts)>{ static_cast<T>(get<I>(t))... };
}

}

template <typename T, class... Ts>
auto to_vec(cute::tuple<Ts...> const& t) {
    return detail::to_vec_impl<T>(t, std::index_sequence_for<Ts...>{});
}

namespace detail {

/*
 * When the destination or source operand is in global memory, the address must be data-element aligned.
 * The address operand form [var + reg] or [reg + reg] is not supported by this instruction.
 * toff is a 32-bit signed integer vector. The vector size must match tensor dimension specified in the
 * instruction qualifier.
 * When ds is set to 4b or 6b, only tiled layout and type1 matrix is supported. For copy from global memory
 * to SLM, fm must set to zero.
 * Only type 1 and type 3 matrices are supported for copy from SLM to global memory.
 * This instruction can only be issued by one elected work-item in a subgroup.
 */

enum CacheCtrl{
  L2uc_L3uc = 0,
  L2uc_L3c, L2uc_L3wb,
  L2c_L3uc, L2wb_L3uc,
  L2c_L3c, L2wb_L3wb
};

enum FillMethod {
  Zero = 0, Nan
};
}
}

template <cute::detail::CacheCtrl> struct cachectrl;
template <> struct cachectrl<cute::detail::CacheCtrl::L2c_L3uc> {
  static constexpr fixstr::fixed_string value {".l2c.L3uc"};};
template <> struct cachectrl<cute::detail::CacheCtrl::L2wb_L3uc> {
  static constexpr fixstr::fixed_string value {".l2wb.L3uc"};};

template <cute::detail::FillMethod> struct padfill;
template <> struct padfill<cute::detail::FillMethod::Zero> {
  static constexpr fixstr::fixed_string value {".zero"};};
template <> struct padfill<cute::detail::FillMethod::Nan> {
  static constexpr fixstr::fixed_string value {".nan"};};

template <cute::detail::CacheCtrl CC> constexpr auto _cc = cachectrl<CC>::value;
template <cute::detail::FillMethod FM> constexpr auto _fl = padfill<FM>::value;

namespace cute {
namespace detail {
/*
 * Tensor Descriptor will contain global memory data-type
 */

template <typename DataType, CacheCtrl CacheType, FillMethod FM>
struct AsyncTensorGlobal2SLM {
  template <size_t N> static inline void Copy(
      MatrixDescriptor Mat, DataType const* GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
      sycl::vec_t<int32_t, N> const& coord
  ) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
      ("async_tensor_copy.shared_workgroup.global."+_s<N>+"d"+_at<DataType>+_fl<FM>+_cc<CacheType>+".abarrier %0, [%1], [%2], [%3], %4;\n")
      ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }

  template <size_t N> static inline void Copy(
    MatrixDescriptor Mat, DataType const* GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
      sycl::vec_t<int32_t, N> const& coord, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
      ("async_tensor_copy.shared_cluster.global."+_s<N>+"d"+_at<DataType>+_fl<FM>+_cc<CacheType>+".abarrier %0, [%1], [%2], [%3], %4, %5;\n")
      ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord), "r"(wg_mask));
#endif
  }
};

template <int BitWidth, CacheCtrl CacheType> struct AsyncTensorSLM2Global {
  template <size_t N> static inline void
  Copy(void const * GmemPtr, MatrixDescriptor Mat, uint64_t* pAbar, TensorPayload* pTDesc,
      sycl::vec_t<int32_t, N> const& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
      ("async_tensor_copy.global.shared_workgroup."+_s<N>+"d."+_s<BitWidth>+"b"+_cc<CacheType>+".abarrier [%0], %1, [%2], [%3], %4;\n")
      ::"r"(GmemPtr), "r"(Mat), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }
};

}
}
