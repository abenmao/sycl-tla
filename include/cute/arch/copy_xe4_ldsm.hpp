#pragma once

#if (SYCL_INTEL_TARGET == 40)

#include <cute/numeric/int.hpp>
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

//First Version will support only these modes
//ArrayofVector support will be added later
enum LDSMMode {
  Scalar,
  Vector,
  CoopVector,
  ArrayOfVectors,
  UnorderedVector
};

// Recursive case: compare against first, then the rest
template<int value, int... vals>
constexpr bool cmp_values() {
  return ((value == vals) || ...);
}

enum MatrixType {
  Type1=0,
  Type2=1,
  Type3=3
};

}

using namespace cute::detail;

template <class SLayout>
constexpr auto get_vector_dir(SLayout Layout)
{
  static constexpr auto strides = Layout.stride();
  static constexpr auto vdir =  get<1>(strides) == 1
	                             ? cute::Vecdir::Vrow 
				     : cute::Vecdir::Vcol;
  return vdir;
}

// LOAD MATRIX Layer
// Initial verssion with xoff and yoff ==0
template <typename T, class SLayout, LDSMMode Mode, uint32_t vlen, cute::Vecdir vdir>
struct XE4_LDSTMatrixBase {

  using ValType = T;
  static constexpr cute::Vecdir getVdir() {
    if constexpr (Mode == CoopVector || Mode == UnorderedVector) {
      return ((vdir == cute::Vecdir::Vrow) ? cute::Vecdir::Cooprow
	              : (vdir == cute::Vecdir::Vcol) ? cute::Vecdir::Coopcol
		      : vdir);
    } else {
      return vdir;
    }
  }
  static constexpr LDSMMode CopyMode = Mode;
  static constexpr int Vlen = vlen;
  static constexpr bool Coop = (Mode == CoopVector);
  static constexpr cute::Vecdir Vdir = getVdir();
  static constexpr bool Unordered = (Mode == UnorderedVector);

  static constexpr int BitWidth =sizeof_bits_v<T>;
  static constexpr int CopyBitsPerThread = BitWidth*Vlen;
  static_assert(CopyBitsPerThread <= 256);

  static constexpr auto strides = SLayout{}.stride();
  static constexpr MatrixType Type = get<1>(strides) == 1 ? MatrixType::Type1 : MatrixType::Type2;
  static constexpr cute::morder Order = (Mode == UnorderedVector) ? cute::morder::unordered : cute::morder::ordered;

  static constexpr inline void check_row_vector_constraints() {
    if constexpr (Type==MatrixType::Type1) {
      static_assert((BitWidth == 64 && cmp_values<vlen, 1,2,4>()) ||
                    (BitWidth == 32 && cmp_values<vlen, 1,2,4,8>()) ||
                    (BitWidth == 16 && cmp_values<vlen, 1,2,4,8,16>()) ||
                    (BitWidth == 8 && cmp_values<vlen, 1,2,4,8,16,32>()) ||
                    (BitWidth == 6 && cmp_values<vlen, 1,2,4,8,16,32>()) ||
                    (BitWidth == 4 && cmp_values<vlen, 1,2,4,8,16,32>()));
    } else if constexpr (Type == MatrixType::Type2) {
      static_assert((BitWidth == 64 && cmp_values<vlen, 1,2,4>()) ||
                    (BitWidth == 32 && cmp_values<vlen, 1,2,4,8>()) ||
                    (BitWidth == 16 && cmp_values<vlen, 1,2,4,8>()) ||
                    (BitWidth == 8 && cmp_values<vlen, 1,2,4,8>()));
    } else if constexpr (Type == MatrixType::Type3) {
      static_assert(dependent_false<T>, "Matrix type supported now for row access");
      //static_assert(BitWidth == 8 && cmp_values(vlen, 1,2,4,8));
    } else {
      static_assert(dependent_false<T>, "Matrix type supported");
    }
  }
  static constexpr inline void check_col_vector_constraints() {
    if constexpr (Type==MatrixType::Type1) {
      static_assert((cmp_values<BitWidth, 64,32,16,8,6>() && cmp_values<vlen, 2,4>()));
    } else if constexpr (Type == MatrixType::Type2) {
      static_assert((BitWidth == 64 && vlen == 2) ||
                    (BitWidth == 32 && cmp_values<vlen, 2,4>()) ||
                    (BitWidth == 16 && cmp_values<vlen, 2,4,8>()) ||
                    (BitWidth == 8 && cmp_values<vlen, 2,4,8,16>()));
    } else if constexpr (Type == MatrixType::Type3) {
      static_assert(dependent_false<T>, "Matrix type supported for column access");
    } else {
      static_assert(dependent_false<T>, "Matrix type supported");
    }
  }
  static constexpr inline void check_coop_row_vector_constraints() {
    if constexpr (Type==MatrixType::Type1) {
      static_assert((BitWidth == 64 && vlen == 4) ||
                    (BitWidth == 32 && vlen == 8) ||
                    (BitWidth == 16 && vlen == 16) ||
                    (BitWidth == 8 && vlen == 32) ||
                    (BitWidth == 6 && vlen == 32) ||
                    (BitWidth == 4 && vlen == 32));
    } else {
      static_assert(dependent_false<T>, "Matrix type not supported for cooperative row access");
    }
  }
  static constexpr inline void check_coop_col_vector_constraints() {
    if constexpr (Type==MatrixType::Type1) {
      static_assert(BitWidth == 4 && cmp_values<vlen, 2,4,8>());
    } else {
      static_assert(dependent_false<T>, "Matrix type not supported for cooperative col access");
    }
  }
  static constexpr inline bool check_constraints() {
    if constexpr (Mode == Vector) {
       if constexpr (Vdir == cute::Vecdir::Vrow) check_row_vector_constraints();
       if constexpr (Vdir == cute::Vecdir::Vcol) check_col_vector_constraints();
    } else if constexpr(Mode == CoopVector) {
       if constexpr (Vdir == cute::Vecdir::Cooprow) check_coop_row_vector_constraints();
       if constexpr (Vdir == cute::Vecdir::Coopcol) check_coop_col_vector_constraints();
    } else if constexpr (Mode == UnorderedVector) {
       if constexpr (Vdir == cute::Vecdir::Cooprow) check_coop_row_vector_constraints();
       else {
          static_assert(dependent_false<T>, "Cooperative column access not supported for Unordered access");
       }
    } else if constexpr (Mode == ArrayOfVectors) {
       static_assert(dependent_false<T>, "Array of Vector access not supported now.");
    }
    return true;
  }
  static_assert(check_constraints());
  XE4_LDSTMatrixBase() {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// XE4_MATRIX_LOAD Load initiates a matrix copy from shared memory to registers
////////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute::detail;
template<typename T, class SLayout, LDSMMode Mode=UnorderedVector, uint32_t VLEN=8, cute::Vecdir VDIR=cute::Vecdir::Vrow>
struct XE4_LOAD_MATRIX : XE4_LDSTMatrixBase<T, SLayout, Mode, VLEN, VDIR> 
{
  //static constexpr BitWidth = sizeof(T) * BITS_PER_BYTE;

  using Super = XE4_LDSTMatrixBase<T, SLayout, Mode, VLEN, VDIR>;
  template <size_t Dim>
  CUTE_HOST_DEVICE static void
  copy(T *reg_ptr, const uint32_t mat_desc_,
       const sycl::marray<uint16_t, Dim>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    static_assert(Dim==2);
    load_matrix<T, Super::Vlen, uint32_t, 0,0,Super::Vdir,0,0,arrdir::none,Super::Order>(reg_ptr, mat_desc_, coord);
#endif
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// XE4_STORE_MTARIX : Initiates a matrix copy from registers to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute::detail;
template<typename T, class SLayout, LDSMMode Mode=UnorderedVector, uint32_t VLEN=8, cute::Vecdir VDIR=cute::Vecdir::Vrow>
struct XE4_STORE_MATRIX : XE4_LDSTMatrixBase<T, SLayout, Mode, VLEN, VDIR> 
{
  //static constexpr BitWidth = sizeof(T) * BITS_PER_BYTE;

  using Super = XE4_LDSTMatrixBase<T, SLayout, Mode, VLEN, VDIR>;
  template <size_t Dim>
  CUTE_HOST_DEVICE static void
  copy(T const* reg_ptr, const uint32_t mat_desc_,
       const sycl::marray<uint16_t, Dim>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    static_assert(Dim==2);
    store_matrix<T, Super::Vlen, uint32_t, 0,0,Super::Vdir,0,0,arrdir::none,Super::Order>(mat_desc_, reg_ptr, coord);
#endif
  }
};
//
// Load Matrix Scalar Load
template<typename T, class SLayout>
using XE4_LDSM_Scaler = XE4_LOAD_MATRIX<T, SLayout, LDSMMode::Scalar, 0,cute::Vecdir::none>;

// Simple Vector
template <typename T, class SLayout, uint32_t vlen, cute::Vecdir vdir=cute::Vecdir::Vrow>
using XE4_LDSM_Vector = XE4_LOAD_MATRIX<T, SLayout, LDSMMode::Vector, vlen, vdir>;

// Load Matrix Cooperative Vector
template <typename T, class SLayout, uint32_t vlen, cute::Vecdir vdir=cute::Vecdir::Cooprow>
using XE4_LDSM_CoopVector = XE4_LOAD_MATRIX<T, SLayout, LDSMMode::CoopVector, vlen, vdir>;

// Load Matrix Unordered Vector
// Only Cooprow supported in unordered
template <typename T, class SLayout, uint32_t vlen, cute::Vecdir>
using XE4_LDSM_UVector = XE4_LOAD_MATRIX<T, SLayout, LDSMMode::UnorderedVector, vlen,
                                         cute::Vecdir::Cooprow>;
// Store Matrix Scalar Load
template<typename T, class SLayout>
using XE4_STSM_Scaler = XE4_STORE_MATRIX<T, SLayout, LDSMMode::Scalar, 0,cute::Vecdir::none>;

// Store Simple Vector
template <typename T, class SLayout, uint32_t vlen, cute::Vecdir vdir=cute::Vecdir::Vrow>
using XE4_STSM_Vector = XE4_STORE_MATRIX<T, SLayout, LDSMMode::Vector, vlen, vdir>;

// Store Matrix Cooperative Vector
template <typename T, class SLayout, uint32_t vlen, cute::Vecdir vdir=cute::Vecdir::Cooprow>
using XE4_STSM_CoopVector = XE4_STORE_MATRIX<T, SLayout, LDSMMode::CoopVector, vlen, vdir>;

// Store Matrix Unordered Vector
// Only Cooprow supported in unordered
template <typename T, class SLayout, uint32_t vlen, cute::Vecdir>
using XE4_STSM_UVector = XE4_STORE_MATRIX<T, SLayout, LDSMMode::UnorderedVector, vlen,
                                         cute::Vecdir::Cooprow>;

} // namespace cute
#endif
