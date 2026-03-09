#pragma once

#if (SYCL_INTEL_TARGET == 40)

#include <type_traits>
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
  ArrayOfVectors, // Array Of Vectors
  UnorderedVector,
  UnorderedArrOfVectors
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
  constexpr auto strides = Layout.stride();
  constexpr auto vdir =  get<1>(strides) == 1
	                             ? cute::Vecdir::Vrow
				     : cute::Vecdir::Vcol;
  return vdir;
}

// LOAD MATRIX Layer
// Initial verssion with xoff and yoff ==0
template <typename T, class SLayout, LDSMMode Mode, uint32_t vlen, cute::Vecdir vdir,
	  uint32_t alen=0, cute::Arrdir adir=cute::Arrdir::none>
struct XE4_LDSTMatrixBase {

  using ValType = T;
  static constexpr cute::Vecdir getVdir() {
    if constexpr (Mode == CoopVector || Mode == UnorderedVector
		                     || Mode == UnorderedArrOfVectors) {
      return ((vdir == cute::Vecdir::Vrow) ? cute::Vecdir::Cooprow
	              : (vdir == cute::Vecdir::Vcol) ? cute::Vecdir::Coopcol
		      : vdir);
    } else {
      return vdir;
    }
  }
  static constexpr LDSMMode CopyMode = Mode;
  static constexpr int Vlen = vlen;
  static constexpr int Alen = alen;
  static constexpr bool Coop = (Mode == CoopVector);
  static constexpr cute::Vecdir Vdir = getVdir();
  static constexpr cute::Arrdir Adir = adir;
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
    } else if constexpr (Mode == UnorderedVector || Mode == UnorderedArrOfVectors) {
       if constexpr (Vdir == cute::Vecdir::Cooprow) check_coop_row_vector_constraints();
       else {
          static_assert(dependent_false<T>, "Cooperative column access not supported for Unordered access");
       }
       if constexpr(Mode == UnorderedArrOfVectors) {
         static_assert(cmp_values<alen, 2,4,8>());
	 static_assert(Adir == cute::Arrdir::Arow);
       }
    } else if constexpr (Mode == ArrayOfVectors) {
       //static_assert(dependent_false<T>, "Array of Vector access not supported now.");
       static_assert(cmp_values<alen, 1,2,4,8>());
       // Acol need to be supported.
    }
    return true;
  }
  static_assert(check_constraints());
  XE4_LDSTMatrixBase() {}
};

enum RedType {
  None,
  IntType,
  FloatType
};

template <typename T, cute::MredOp Rop, class SLayout, LDSMMode Mode, uint32_t Vlen, cute::Vecdir Vdir,
	  uint32_t Alen=0, cute::Arrdir Adir=cute::Arrdir::none>
struct XE4_MatrixRedBase : XE4_LDSTMatrixBase<T, SLayout, Mode, Vlen, Vdir, Alen, Adir> {

  static constexpr RedType rtype = (std::is_same_v<T, int> ||
                                    std::is_same_v<T, uint32_t> ||
                                    std::is_same_v<T, uint16_t>)
	                         ? RedType::IntType
				 : ((std::is_same_v<T, cutlass::tfloat32_t> ||
                                      std::is_same_v<T, float> ||
                                      std::is_same_v<T, sycl::ext::oneapi::bfloat16> ||
                                      std::is_same_v<T, cutlass::half_t> ||
                                      std::is_same_v<T, sycl::half>)
				 ? RedType::FloatType
				 : RedType::None);

  using Super = XE4_LDSTMatrixBase<T, SLayout, Mode, Vlen, Vdir, Alen, Adir>;
  static constexpr cute::MredOp RedOp = Rop;
  static constexpr inline void check_red_mode_contraints() {
    if constexpr (Mode == Scalar) {
      static_assert(Vlen == 0 && Alen == 0);
    } else if constexpr(Mode == Vector || Mode == ArrayOfVectors) {
      static_assert(Vlen >= 1 && Vlen <= 32 && (Vlen & Vlen-1) == 0);
      static_assert(Vdir == cute::Vecdir::Vrow);
      Super::check_row_vector_constraints();
    }
    if constexpr(Mode == ArrayOfVectors) {
      static_assert(cmp_values<Alen, 1,2,4,8>());
      static_assert(Adir == cute::Arrdir::Arow || Adir == cute::Arrdir::Acol);
    }
  }
  static constexpr inline bool check_constraints() {
    // Only three modes supported for reduction ops
    static_assert(cmp_values<Mode, Scalar, Vector, ArrayOfVectors>());
    static_assert(rtype != RedType::None, "DataTypes should be Int or float types of 16 or 32 bits");
    static_assert(Vlen >= 0 && Vlen <= 32);
    static_assert(Super::BitWidth == 16 || Super::BitWidth == 32);
    if constexpr (Rop == MredOp::Incwrap || Rop == MredOp::Decwrap) {
      static_assert(Super::BitWidth==32 && rtype == RedType::IntType);
    }
    if constexpr (rtype == RedType::FloatType) {
      static_assert((Rop == MredOp::Add || Rop == MredOp::Min || Rop == MredOp::Max));
    }
    if constexpr (rtype == RedType::IntType) {
      static_assert(!(Rop == MredOp::Min || Rop == MredOp::Max));
    }
    check_red_mode_contraints();
    return true;
    // NEED TO ADD check for aligment
  }
  static_assert(check_constraints());
  XE4_MatrixRedBase() : XE4_LDSTMatrixBase<T, SLayout, Mode, Vlen, Vdir, Alen, Adir>() {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// XE4_MATRIX_LOAD Load initiates a matrix copy from shared memory to registers
////////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute::detail;
template<typename T, class SLayout, LDSMMode Mode=UnorderedVector,
         uint32_t VLEN=8, cute::Vecdir VDIR=cute::Vecdir::Vrow,
	 uint32_t ALEN=0, cute::Arrdir ADIR=cute::Arrdir::none>
struct XE4_LOAD_MATRIX : XE4_LDSTMatrixBase<T, SLayout, Mode, VLEN, VDIR, ALEN, ADIR>
{
  //static constexpr BitWidth = sizeof(T) * BITS_PER_BYTE;

  static constexpr int AStride = (ALEN==0) ? 0 : 1;
  using Super = XE4_LDSTMatrixBase<T, SLayout, Mode, VLEN, VDIR, ALEN, ADIR>;
  template <size_t Dim>
  CUTE_HOST_DEVICE static void
  copy(T *reg_ptr, const uint32_t mat_desc_,
       const sycl::marray<uint16_t, Dim>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    static_assert(Dim==2);
    load_matrix<T, Super::Vlen, uint32_t, 0,0,Super::Vdir,
	        ALEN,AStride,ADIR,
		Super::Order>(reg_ptr, mat_desc_, coord);
#endif
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////
/// XE4_STORE_MTARIX : Initiates a matrix copy from registers to shared memory
////////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute::detail;
template<typename T, class SLayout, LDSMMode Mode=UnorderedVector,
         uint32_t VLEN=8, cute::Vecdir VDIR=cute::Vecdir::Vrow,
	 uint32_t ALEN=0, cute::Arrdir ADIR=cute::Arrdir::none>
struct XE4_STORE_MATRIX : XE4_LDSTMatrixBase<T, SLayout, Mode, VLEN, VDIR, ALEN, ADIR>
{
  //static constexpr BitWidth = sizeof(T) * BITS_PER_BYTE;

  static constexpr int AStride = (ALEN==0) ? 0 : 1;
  using Super = XE4_LDSTMatrixBase<T, SLayout, Mode, VLEN, VDIR, ALEN, ADIR>;
  template <size_t Dim>
  CUTE_HOST_DEVICE static void
  copy(T const* reg_ptr, const uint32_t mat_desc_,
       const sycl::marray<uint16_t, Dim>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    static_assert(Dim==2);
    store_matrix<T, Super::Vlen, uint32_t, 0,0,Super::Vdir,
	         ALEN,AStride,ADIR,
		 Super::Order>(mat_desc_, reg_ptr, coord);
#endif
  }
};

/////////////////////////////////////////////////////////////////////////
//    LOAD MATRIX
/////////////////////////////////////////////////////////////////////////
// Load Matrix Scalar Load
template<typename T, class SLayout>
using XE4_LDSM_Scaler = XE4_LOAD_MATRIX<T, SLayout, LDSMMode::Scalar, 0,cute::Vecdir::none>;

// Simple Vector
template <typename T, class SLayout, uint32_t vlen, cute::Vecdir vdir=cute::Vecdir::Vrow>
using XE4_LDSM_Vector = XE4_LOAD_MATRIX<T, SLayout, LDSMMode::Vector, vlen, vdir>;

// Simple Vector
template <typename T, class SLayout, uint32_t vlen, cute::Vecdir vdir=cute::Vecdir::Vrow,
          uint32_t alen=2, cute::Arrdir adir=cute::Arrdir::Arow>
using XE4_LDSM_AOfVector = XE4_LOAD_MATRIX<T, SLayout, LDSMMode::ArrayOfVectors, vlen, vdir>;

// Load Matrix Cooperative Vector
template <typename T, class SLayout, uint32_t vlen, cute::Vecdir vdir=cute::Vecdir::Cooprow>
using XE4_LDSM_CoopVector = XE4_LOAD_MATRIX<T, SLayout, LDSMMode::CoopVector, vlen, vdir>;

// Load Matrix Unordered Vector
// Cooprow and AofV (row dir) supported in unordered
template <typename T, class SLayout, uint32_t vlen>
using XE4_LDSM_UVector = XE4_LOAD_MATRIX<T, SLayout, LDSMMode::UnorderedVector, vlen,
                                         cute::Vecdir::Cooprow>;
//
// Load Matrix Unordered Array of Vector
// Cooprow and AofV (row dir) supported in unordered
template <typename T, class SLayout, uint32_t vlen, uint32_t alen>
using XE4_LDSM_UAOfVector = XE4_LOAD_MATRIX<T, SLayout, LDSMMode::UnorderedArrOfVectors, vlen,
                                            cute::Vecdir::Cooprow, alen, cute::Arrdir::Arow>;
//
/////////////////////////////////////////////////////////////////////////
//    STORE MATRIX
/////////////////////////////////////////////////////////////////////////
// Store Matrix Scalar Store
template<typename T, class SLayout>
using XE4_STSM_Scaler = XE4_STORE_MATRIX<T, SLayout, LDSMMode::Scalar, 0,cute::Vecdir::none>;

// Store Simple Vector
template <typename T, class SLayout, uint32_t vlen, cute::Vecdir vdir=cute::Vecdir::Vrow>
using XE4_STSM_Vector = XE4_STORE_MATRIX<T, SLayout, LDSMMode::Vector, vlen, vdir>;

// Store Matrix Cooperative Vector
template <typename T, class SLayout, uint32_t vlen, cute::Vecdir vdir=cute::Vecdir::Cooprow>
using XE4_STSM_CoopVector = XE4_STORE_MATRIX<T, SLayout, LDSMMode::CoopVector, vlen, vdir>;

// Store Matrix Unordered Vector
template <typename T, class SLayout, uint32_t vlen>
using XE4_STSM_UVector = XE4_STORE_MATRIX<T, SLayout, LDSMMode::UnorderedVector, vlen,
                                         cute::Vecdir::Cooprow>;

// Store Matrix Coop Unordered Array of Vector
template <typename T, class SLayout, uint32_t vlen, uint32_t alen>
using XE4_STSM_UAOfVector = XE4_STORE_MATRIX<T, SLayout, LDSMMode::UnorderedVector, vlen,
                                         cute::Vecdir::Cooprow, alen, cute::Arrdir::Arow>;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// XE4_REDUCE_MATRIX Initiates a matrix reducetion from shared memory to registers
////////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute::detail;
template<typename T, cute::MredOp Rop,
	 class SLayout, LDSMMode Mode=UnorderedVector,
         uint32_t VLEN=8, cute::Vecdir VDIR=cute::Vecdir::Vrow,
         uint32_t ALEN=0, cute::Arrdir ADIR=cute::Arrdir::none>
struct XE4_REDUCE_MATRIX : XE4_MatrixRedBase<T, Rop, SLayout, Mode, VLEN, VDIR, ALEN, ADIR>
{
  //static constexpr BitWidth = sizeof(T) * BITS_PER_BYTE;

  static constexpr int AStride = (ALEN==0) ? 0 : 1;
  using Super = XE4_MatrixRedBase<T, Rop, SLayout, Mode, VLEN, VDIR, ALEN, ADIR>;
  template <size_t Dim>
  CUTE_HOST_DEVICE static void
  copy(T const *reg_ptr, const uint32_t mat_desc_,
       const sycl::marray<uint16_t, Dim>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    static_assert(Dim==2);
    red_matrix<T, Rop, Super::Vlen, uint32_t, 0,0,Super::Vdir,
               ALEN, AStride, ADIR
              >(reg_ptr, mat_desc_, coord);
#endif
  }
};

} // namespace cute
#endif
