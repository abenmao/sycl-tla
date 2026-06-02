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
to_vec_impl(tuple<Ts...> const& t, std::index_sequence<I...>) {
    return sycl::vec_t<T, sizeof...(Ts)>{ static_cast<T>(get<I>(t))... };
}

}

template <typename T, class... Ts>
auto to_vec(tuple<Ts...> const& t) {
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

// Cache control hints for async_tensor_copy / prefetch / reduce instructions.
// These map to PISA asm suffixes like ".L2c.L3uc", ".L2wb.L3wb", etc.
// The naming convention is: L2<policy>_L3<policy> where:
//   uc = uncached, c = cached, wb = write-back
enum CacheCtrl{
  None=0,
  L2uc_L3uc,
  L2uc_L3c, L2uc_L3wb,
  L2c_L3uc, L2wb_L3uc,
  L2c_L3c, L2wb_L3wb
};

enum FillMethod {
  Zero = 0, Nan
};

// Completion mode modifier (`<.cm>`)
enum CompletionMode {
  CM_Unspecified = 0, CM_Write, CM_Read
};

template <CacheCtrl CC> struct CacheHint {
  constexpr static CacheCtrl value = CC;
};

template <FillMethod FM> struct FillMode {
  constexpr static FillMethod value = FM;
};

template <CompletionMode CM> struct CompletionModeHint {
  constexpr static CompletionMode value = CM;
};
}
}

template <cute::detail::CacheCtrl> struct cachectrl;
template <> struct cachectrl<cute::detail::CacheCtrl::L2c_L3uc> {
  static constexpr fixstr::fixed_string value {".l2c.L3uc"};};
template <> struct cachectrl<cute::detail::CacheCtrl::L2wb_L3uc> {
  static constexpr fixstr::fixed_string value {".l2wb.L3uc"};};
template <> struct cachectrl<cute::detail::CacheCtrl::L2uc_L3uc> {
  static constexpr fixstr::fixed_string value {".L2uc.L3uc"};};
template <> struct cachectrl<cute::detail::CacheCtrl::L2uc_L3c> {
  static constexpr fixstr::fixed_string value {".L2uc.L3c"};};
template <> struct cachectrl<cute::detail::CacheCtrl::L2uc_L3wb> {
  static constexpr fixstr::fixed_string value {".L2uc.L3wb"};};
template <> struct cachectrl<cute::detail::CacheCtrl::L2c_L3c> {
  static constexpr fixstr::fixed_string value {".L2c.L3c"};};
template <> struct cachectrl<cute::detail::CacheCtrl::L2wb_L3wb> {
  static constexpr fixstr::fixed_string value {".L2wb.L3wb"};};

template <cute::detail::FillMethod> struct padfill;
template <> struct padfill<cute::detail::FillMethod::Zero> {
  static constexpr fixstr::fixed_string value {".zero"};};
template <> struct padfill<cute::detail::FillMethod::Nan> {
  static constexpr fixstr::fixed_string value {".nan"};};

template <cute::detail::CompletionMode> struct cmplmode;
template <> struct cmplmode<cute::detail::CompletionMode::CM_Unspecified> {
  static constexpr fixstr::fixed_string value {""};};
template <> struct cmplmode<cute::detail::CompletionMode::CM_Write> {
  static constexpr fixstr::fixed_string value {".write"};};
template <> struct cmplmode<cute::detail::CompletionMode::CM_Read> {
  static constexpr fixstr::fixed_string value {".read"};};

template <cute::detail::CacheCtrl CC> constexpr auto _cc = cachectrl<CC>::value;
template <cute::detail::FillMethod FM> constexpr auto _fl = padfill<FM>::value;
template <cute::detail::CompletionMode CM> constexpr auto _cm = cmplmode<CM>::value;

namespace cute {
namespace detail {
/*
 * Tensor Descriptor will contain global memory data-type
 */

template <typename DataType>
struct AsyncTensorGlobal2SLM {
  template <size_t N, CacheCtrl CacheType = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
  static inline void Copy(
      MatrixDescriptor Mat, DataType const* GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
      sycl::vec_t<int32_t, N> const& coord, CacheHint<CacheType> = {}, FillMode<FM> = {}
  ) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
      ("async_tensor_copy.shared_workgroup.global."+_s<N>+"d"+_at<DataType>+_fl<FM>+_cc<CacheType>+".abarrier %0, [%1], [%2], [%3], %4;\n")
      ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }

  template <size_t N, CacheCtrl CacheType = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
  static inline void Copy(
      MatrixDescriptor Mat, DataType const* GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
      sycl::vec_t<int32_t, N> const& coord, uint32_t wg_mask, CacheHint<CacheType> = {}, FillMode<FM> = {}) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
      ("async_tensor_copy.shared_cluster.global."+_s<N>+"d"+_at<DataType>+_fl<FM>+_cc<CacheType>+".abarrier %0, [%1], [%2], [%3], %4, %5;\n")
      ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord), "r"(wg_mask));
#endif
  }
};

template <int BitWidth>
struct AsyncTensorSLM2Global {
  template <size_t N, CacheCtrl CacheType = CacheCtrl::L2wb_L3uc>
  static inline void
  Copy(void * GmemPtr, MatrixDescriptor Mat, uint64_t* pAbar, TensorPayload* pTDesc,
      sycl::vec_t<int32_t, N> const& coord, CacheHint<CacheType> = {}) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
      ("async_tensor_copy.global.shared_workgroup."+_s<N>+"d."+_s<BitWidth>+"b"+_cc<CacheType>+".abarrier [%0], %1, [%2], [%3], %4;\n")
      ::"r"(GmemPtr), "r"(Mat), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// AsyncTensorGlobalPrefetch — Issues async_tensor_prefetch.Nd for L2 cache warming.
///
/// Fire-and-forget: no barrier, no SLM destination. The prefetch instruction hints the
/// memory subsystem to bring the described tensor tile into the specified cache level(s).
/// Used by XE4_ADMA_PREFETCH::copy() → dispatched from Copy_Traits<XE4_ADMA_PREFETCH>.
///
/// Template params:
///   DataType  — element type (determines bit-width suffix, e.g. ".16b" for half)
///   CacheType — cache control hint (typically L2c_L3uc for prefetch-into-L2)
///
/// Instruction format:
///   async_tensor_prefetch.<N>d.<BitWidth>b.<CacheCtrl>.global [gmem_ptr], [tdesc], coord;
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename DataType, CacheCtrl CacheType>
struct AsyncTensorGlobalPrefetch {
  template <size_t N> static inline void Copy(
      DataType const* GmemPtr, TensorPayload* pTDesc,
      sycl::vec_t<int32_t, N> const& coord
  ) {
#if defined (__SYCL_DEVICE_ONLY__)
    constexpr unsigned int BitWidth =sizeof_bits_v<DataType>;
    static_assert(cmp_values<N, 1,2,3,4,5>());
    static_assert(cmp_values<sizeof(DataType) * 8, 8, 16, 32, 64>());
    asm volatile (
      ("async_tensor_prefetch." + _s<N>+"d." + _s<BitWidth> + "b"
       +_cc<CacheType> + ".global [%0], [%1], %2;\n")
      ::"r"(GmemPtr), "r"(pTDesc), "r"(coord));
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// AsyncTensorReduce — Issues async_tensor_fred/ired for SLM → gmem atomic reduction.
///
/// Dispatched by XE4_ADMA_STORE_REDUCE::copy() (via Copy_Traits → ADMA_STORE_Unpack).
/// Atomically reduces SLM data into global memory at the tensor-descriptor coordinates.
///
/// Specialized on RedType:
///   FloatType → async_tensor_fred (floating-point reduction)
///   IntType   → async_tensor_ired (integer reduction)
///
/// Template params on Copy:
///   CacheType — Must be L2wb_L3wb or L2uc_L3wb (HW constraint for reduce instructions)
///   Rop       — Reduction operation (Add, Min, Max, Smin, Smax, etc.)
///   BarType   — Completion mechanism:
///               Abarrier:  includes abar_ptr operand (arrival barrier)
///               Groupsync: omits abar_ptr, uses workgroup sync instead
///   T         — Element data type (determines asm data-type suffix via _mdtype)
///   N         — Tensor dimensionality (1d–5d)
///
/// Instruction format (FloatType, Abarrier):
///   async_tensor_fred.global.shared_workgroup.2d.add.hf.L2wb.L3wb.abarrier [gmem], mat, [abar], [tdesc], coord;
/// Instruction format (IntType, Groupsync):
///   async_tensor_ired.global.shared_workgroup.2d.add.32b.L2wb.L3wb.groupsync [gmem], mat, [tdesc], coord;
////////////////////////////////////////////////////////////////////////////////////////////////////
template <RedType RDType>
struct AsyncTensorReduce;

// FloatType specialization: generates async_tensor_fred instructions
template <>
struct AsyncTensorReduce<RedType::FloatType> {
  template <CacheCtrl CacheType, RedOp Rop, BarrierType BarType, typename T, size_t N>
  static inline void Copy(
      MatrixDescriptor Mat, T const* GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
      sycl::vec_t<int32_t, N> const& coord
  ) {
#if defined (__SYCL_DEVICE_ONLY__)
    static_assert(cmp_values<N, 1,2,3,4,5>());
    static_assert(CacheType == L2wb_L3wb || CacheType == L2uc_L3wb);

    if constexpr (BarType == BarrierType::Abarrier) {
      asm volatile (
        ("async_tensor_fred.global.shared_workgroup." + _s<N> + "d"
         + _red_algo<Rop> + _mdtype<T> + _cc<CacheType>
         + ".abarrier [%0], %1, [%2], [%3], %4;\n")
        ::"r"(GmemPtr), "r"(Mat), "r"(pAbar), "r"(pTDesc), "r"(coord));
    } else {
      asm volatile (
        ("async_tensor_fred.global.shared_workgroup." + _s<N> + "d"
         + _red_algo<Rop> + _mdtype<T> + _cc<CacheType>
         + ".groupsync [%0], %1, [%2], %3;\n")
        ::"r"(GmemPtr), "r"(Mat), "r"(pTDesc), "r"(coord));
    }
#endif
  }
};

// IntType specialization: generates async_tensor_ired instructions
template <>
struct AsyncTensorReduce<RedType::IntType> {
  template <CacheCtrl CacheType, RedOp Rop, BarrierType BarType, typename T, size_t N>
  static inline void Copy(
      MatrixDescriptor Mat, T const* GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
      sycl::vec_t<int32_t, N> const& coord
  ) {
#if defined (__SYCL_DEVICE_ONLY__)
    static_assert(cmp_values<N, 1,2,3,4,5>());
    static_assert(CacheType == L2wb_L3wb || CacheType == L2uc_L3wb);

    if constexpr (BarType == BarrierType::Abarrier) {
      asm volatile (
        ("async_tensor_ired.global.shared_workgroup." + _s<N> + "d"
         + _red_algo<Rop> + _mdtype<T> + _cc<CacheType>
         + ".abarrier [%0], %1, [%2], [%3], %4;\n")
        ::"r"(GmemPtr), "r"(Mat), "r"(pAbar), "r"(pTDesc), "r"(coord));
    } else {
      asm volatile (
        ("async_tensor_ired.global.shared_workgroup." + _s<N> + "d"
         + _red_algo<Rop> + _mdtype<T> + _cc<CacheType>
         + ".groupsync [%0], %1, [%2], %3;\n")
        ::"r"(GmemPtr), "r"(Mat), "r"(pTDesc), "r"(coord));
    }
#endif
  }
};

}
}
