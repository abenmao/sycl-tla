#pragma once

#include <random>
#include "base_datatype.hpp"
#include <sycl/multi_ptr.hpp> // for address_space_cast
#include <sycl/sycl.hpp>

static constexpr uint32_t LANESIZE = 32;
static constexpr uint32_t BITS_PER_BYTE = 8;

#ifdef __SYCL_DEVICE_ONLY__
#define INLINE_PISA(...) asm volatile(__VA_ARGS__)
#else
#define INLINE_PISA(...)
#endif

#ifdef __SYCL_DEVICE_ONLY__
template <class T, int N>
using vector_t = typename std::conditional_t<N == 1, T, T __attribute__((ext_vector_type(N)))>;
#else
template <class T, int N>
using vector_t = sycl::marray<T, N>;
#endif

template <class T, class F>
T vec_as(const F &x) {
  return sycl::bit_cast<T>(x);
}

template <class T, size_t N>
inline auto as_vector_t(const sycl::marray<T, N> &arr) {
  using result_type = vector_t<T, N>;
  constexpr size_t newN = sizeof(result_type) / sizeof(T);

  if constexpr (N == newN) { return sycl::bit_cast<result_type>(arr); }

  sycl::marray<T, newN> ext_arr;
#pragma unroll
  for (size_t i = 0; i < N; ++i) {
    ext_arr[i] = arr[i];
  }

  return sycl::bit_cast<result_type>(ext_arr);
}

#define XETLA_MARKER(message) [[deprecated(message)]]

template <auto val>
XETLA_MARKER("Help function to print value")
inline constexpr void XETLA_PRINT() {}

template <typename type>
XETLA_MARKER("Help function to print type")
inline constexpr void XETLA_PRINT() {}

enum class data_type : uint8_t {
  u8 = 0,
  i8 = 1,
  u16 = 2,
  i16 = 3,
  u32 = 4,
  i32 = 5,
  u64 = 6,
  i64 = 7,
  bf8 = 16,
  hf8 = 17,
  bf16 = 18,
  fp16 = 19,
  tf32 = 20,
  fp32 = 21,
  fp64 = 22,
  e1 = 23,
  e2m1 = 24,
  e3m0 = 25,
  u4 = 26,
  i4 = 27,
  u1 = 28,
  i1 = 29,
  u2 = 30,
  i2 = 31,
  invld = 0xFF // invalid
};

enum class slm_matrix_type : uint8_t {
  /// 32-byte in leading dimension and 32 elements in the other dimension.
  /// Mainly for matA k-major, matB n-major, matB k-major, matC/D n-major.
  type1 = 0,

  /// 32 elements in leading dimension and 32/element_size_in_bytes in the
  /// other dimension. Mainly for matA m-major.
  type2 = 1,

  /// 32 elements in leading dimension and 8/element_size_in_bytes in the
  /// other dimension. Mainly for matA m-major.
  type3 = 2
};

// Note: this is not aligned with WP, but aligned with cmodel. Let's keep what
// it is for now.
enum class cm_layout_t : uint8_t {
  /// no sub blocking, each block stores to slm with width=512B, stride=512B
  /// (linear). Mainly for matB.
  linear = 0,
  /// split core matrix dim0 into num_bank sub blocks, each sub block stores
  /// to slm with width=64B(or 2x64B), stride=512B. Mainly for matA m-major.
  horizontal_split = 1,
  /// split core matrix dim1 into num_bank sub blocks, each sub block stores
  /// to slm with width=64B(or 2x64B), stride=512B. Mainly for matA k-major,
  /// matC n-major.
  vertical_split = 2
};

enum class cm_size_t : uint8_t {
  /// matrix A k-major for all data types, matrix A m-major for 8-bit data
  /// type, matrix C/D n-major for all data types
  cm_32x32B = 0,
  /// matrix A m-major for 16-bit data types
  cm_16x64B = 1,
  /// matrix A m-major for 32-bit data types
  cm_8x128B = 2,
  /// matrix A m-major for 64-bit data types
  cm_4x256B = 3,
  /// matrix B n-major for 8-bit data types
  cm_32x16B = 4,
  /// matrix B k-major for all data types, matrix B n-major for 16-bit data
  /// types
  cm_16x32B = 5,
  /// matrix B n-major for 32-bit data types
  cm_8x64B = 6,
  /// matrix B n-major for 64-bit data types
  cm_4x128B = 7,
  /// meta data
  cm_8x32B = 8
};

enum class slm_layout_t : uint8_t { linear = 0, tiled = 1 };

template <cm_size_t cmSize>
constexpr uint32_t get_height() {
  if constexpr (cmSize == cm_size_t::cm_32x32B || cmSize == cm_size_t::cm_32x16B) {
    return 32;
  } else if constexpr (cmSize == cm_size_t::cm_16x64B || cmSize == cm_size_t::cm_16x32B) {
    return 16;
  } else if constexpr (cmSize == cm_size_t::cm_8x128B || cmSize == cm_size_t::cm_8x64B ||
                       cmSize == cm_size_t::cm_8x32B) {
    return 8;
  } else if constexpr (cmSize == cm_size_t::cm_4x256B || cmSize == cm_size_t::cm_4x128B) {
    return 4;
  } else {
    static_assert(false, "Unsupported cm size");
  }
}

template <cm_size_t cmSize>
constexpr uint32_t get_width_in_bytes() {
  if constexpr (cmSize == cm_size_t::cm_32x32B || cmSize == cm_size_t::cm_16x32B || cmSize == cm_size_t::cm_8x32B) {
    return 32;
  } else if constexpr (cmSize == cm_size_t::cm_16x64B || cmSize == cm_size_t::cm_8x64B) {
    return 64;
  } else if constexpr (cmSize == cm_size_t::cm_8x128B || cmSize == cm_size_t::cm_4x128B) {
    return 128;
  } else if constexpr (cmSize == cm_size_t::cm_4x256B) {
    return 256;
  } else if constexpr (cmSize == cm_size_t::cm_32x16B) {
    return 16;
  } else {
    static_assert(false, "Unsupported cm size");
  }
}

template <data_type dtype>
constexpr uint32_t size_of() {
  if constexpr (dtype == data_type::u8 || dtype == data_type::i8 || dtype == data_type::bf8 ||
                dtype == data_type::hf8) {
    return 1;
  } else if constexpr (dtype == data_type::u16 || dtype == data_type::i16 || dtype == data_type::bf16 ||
                       dtype == data_type::fp16) {
    return 2;
  } else if constexpr (dtype == data_type::u32 || dtype == data_type::i32 || dtype == data_type::tf32 ||
                       dtype == data_type::fp32) {
    return 4;
  } else if constexpr (dtype == data_type::u64 || dtype == data_type::i64 || dtype == data_type::fp64) {
    return 8;
  } else {
    static_assert(false, "Unsupported data type");
  }
}

template <typename dtype>
constexpr uint32_t sizeof_bits() {
  if constexpr (std::is_same_v<dtype, fp4_e3m0>) {
    return 4;
  } else if constexpr (std::is_same_v<dtype, e8m0> || std::is_same_v<dtype, hf8> || std::is_same_v<dtype, bf8> ||
                       std::is_same_v<dtype, int8_t>) {
    return 8;
  } else if constexpr (std::is_same_v<dtype, fp16> || std::is_same_v<dtype, bf16>) {
    return 16;
  } else if constexpr (std::is_same_v<dtype, float> || std::is_same_v<dtype, int32_t>) {
    return 32;
  } else {
    static_assert(false, "Unsupported data type");
  }
}

template <slm_matrix_type cm_type, data_type dtype>
constexpr cm_size_t get_core_matrix_size() {
  if constexpr (cm_type == slm_matrix_type::type1) {
    return cm_size_t::cm_32x32B;
  } else if constexpr (cm_type == slm_matrix_type::type2) {
    if constexpr (size_of<dtype>() == 1) {
      return cm_size_t::cm_32x32B;
    } else if constexpr (size_of<dtype>() == 2) {
      return cm_size_t::cm_16x64B;
    } else if constexpr (size_of<dtype>() == 4) {
      return cm_size_t::cm_8x128B;
    } else if constexpr (size_of<dtype>() == 8) {
      return cm_size_t::cm_4x256B;
    } else {
      static_assert(false, "Supports up to 64-bit.");
    }
  } else if constexpr (cm_type == slm_matrix_type::type3) {
    if constexpr (size_of<dtype>() == 1) {
      return cm_size_t::cm_8x32B;
    } else {
      static_assert(false, "Only support 8-bit.");
    }
  } else {
    static_assert(false, "Unsupported core matrix type.");
  }
}

template <typename T>
constexpr data_type get_dtype() {
  if constexpr (std::is_same<std::remove_cv_t<T>, uint8_t>::value) {
    return data_type::u8;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, int8_t>::value) {
    return data_type::i8;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, uint16_t>::value) {
    return data_type::u16;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, int16_t>::value) {
    return data_type::i16;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, uint32_t>::value) {
    return data_type::u32;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, int32_t>::value) {
    return data_type::i32;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, uint64_t>::value) {
    return data_type::u64;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, int64_t>::value) {
    return data_type::i64;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, bf8>::value) {
    return data_type::bf8;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, hf8>::value) {
    return data_type::hf8;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, bf16>::value) {
    return data_type::bf16;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, fp16>::value) {
    return data_type::fp16;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, tf32>::value) {
    return data_type::tf32;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, float>::value) {
    return data_type::fp32;
  } else if constexpr (std::is_same<std::remove_cv_t<T>, double>::value) {
    return data_type::fp64;
  } else {
    return data_type::invld;
  }
}

enum class cache_hint_t : uint8_t { L2uc_L3uc = 0, L2uc_L3c = 1, L2c_L3uc = 2, L2c_L3c = 3 };

template <typename T>
struct is_floating_t {
  static constexpr bool value = false;
};

template <>
struct is_floating_t<fp4_e3m0> {
  static constexpr bool value = true;
};

template <>
struct is_floating_t<bf8> {
  static constexpr bool value = true;
};

template <>
struct is_floating_t<hf8> {
  static constexpr bool value = true;
};

template <>
struct is_floating_t<bf16> {
  static constexpr bool value = true;
};

template <>
struct is_floating_t<fp16> {
  static constexpr bool value = true;
};

template <>
struct is_floating_t<float> {
  static constexpr bool value = true;
};

template <>
struct is_floating_t<double> {
  static constexpr bool value = true;
};

template <>
struct is_floating_t<e8m0> {
  static constexpr bool value = true;
};

template <typename T>
struct uint_type {
  static constexpr bool is_uint8 = sizeof(T) == 1;
  static constexpr bool is_uint16 = sizeof(T) == 2;
  static constexpr bool is_uint32 = sizeof(T) == 4;
  static constexpr bool is_uint64 = sizeof(T) == 8;
  using type = typename std::conditional<
      is_uint8, uint8_t,
      typename std::conditional<
          is_uint16, uint16_t,
          typename std::conditional<is_uint32, uint32_t,
                                    typename std::conditional<is_uint64, uint64_t, void>::type>::type>::type>::type;
};

template <int Size>
struct get_uint_type {
  static constexpr bool is_uint8 = Size == 1;
  static constexpr bool is_uint16 = Size == 2;
  static constexpr bool is_uint32 = Size == 4;
  static constexpr bool is_uint64 = Size == 8;
  using type = typename std::conditional<
      is_uint8, uint8_t,
      typename std::conditional<
          is_uint16, uint16_t,
          typename std::conditional<is_uint32, uint32_t,
                                    typename std::conditional<is_uint64, uint64_t, void>::type>::type>::type>::type;
};

template <typename T>
using uint_type_t = typename uint_type<T>::type;

template <int Size>
using get_uint_type_t = typename get_uint_type<Size>::type;

enum class mem_layout : uint8_t { row_major = 0, col_major = 1 };
enum class sparsity_repr_t : uint8_t { A4xB2 = 0 };

template <class T>
inline auto slm_space_cast(T *slm_ptr) {
  return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(slm_ptr)
      .get();
}

template <typename T, int N, typename SyclGroup>
inline auto alloc_slm_buffer(const SyclGroup &group) {
#ifndef DRY_RUN
  auto multi_ptr_ = sycl::ext::oneapi::group_local_memory_for_overwrite<T[N]>(group);
  return slm_space_cast(*multi_ptr_);
#else
  return static_cast<T *>(0);
#endif
}

template <uint32_t x_ = 1, uint32_t y_ = 1, uint32_t z_ = 1>
struct ternary_vec {
  static constexpr uint32_t x = x_;
  static constexpr uint32_t y = y_;
  static constexpr uint32_t z = z_;
};

// combine sizeof...(I) sycl::vec to one big sycl::vec
template <typename T, int N, size_t... I>
inline auto array_to_vec(sycl::vec<T, N> src[sizeof...(I)], std::index_sequence<I...>) {
  return sycl::vec<T, N * sizeof...(I)>(src[I]...);
}

template <typename T, size_t N, size_t... I>
inline auto array_to_vec(sycl::marray<T, N> src[sizeof...(I)], std::index_sequence<I...>) {
  return sycl::marray<T, N * sizeof...(I)>(src[I]...);
}

template <typename dtype>
void inner_block_trans(std::vector<dtype> &dst, std::vector<dtype> &src, int size_x, int size_y, int blk_size) {
  for (int i = 0; i < size_y; i += blk_size) {
    for (int j = 0; j < size_x; j += blk_size) {
      for (int ii = 0; ii < blk_size; ii++) {
        for (int jj = 0; jj < blk_size; jj++) {
          dst[(i + jj) * size_x + j + ii] = src[(i + ii) * size_x + j + jj];
        }
      }
    }
  }
}

template <typename T>
sycl::vec<uint32_t, 2> get_cm_size(slm_matrix_type cm_type) {
  uint32_t data_size = sizeof(T);
  uint32_t leading_size = 32, secondary_size = 32;
  if (cm_type == slm_matrix_type::type1) {
    leading_size /= data_size;
  } else {
    secondary_size /= data_size;
  }
  return sycl::vec<uint32_t, 2> {leading_size, secondary_size};
}

template <typename T, uint32_t Dim, typename DstT = uint64_t>
inline sycl::vec<DstT, Dim - 1> get_stride_from_shape(const sycl::vec<uint32_t, Dim> &shape) {
  sycl::vec<DstT, Dim - 1> res;
  DstT accumulted = sizeof(T);
#pragma unroll
  for (uint32_t i = 0; i < Dim - 1; i++) {
    accumulted *= shape[i];
    res[i] = accumulted;
  }
  return res;
}

template <typename T, uint32_t Dim, typename DstT = uint64_t>
inline DstT get_offset(const sycl::marray<int32_t, Dim> &coord, const sycl::vec<DstT, Dim - 1> &stride) {
  DstT offset = coord[0] * sizeof(T);
#pragma unroll
  for (uint32_t i = 1; i < Dim; i++) {
    offset += coord[i] * stride[i - 1];
  }
  return offset;
}

template <uint32_t Dim>
inline bool is_within_boundary(const sycl::marray<int32_t, Dim> &coord, const sycl::vec<uint32_t, Dim> &gmem_shape) {
  bool res = true;
#pragma unroll
  for (uint32_t i = 0; i < Dim; i++) {
    res = res && (coord[i] >= 0) && (coord[i] < gmem_shape[i]);
  }
  return res;
};

namespace row_copy {
inline uint32_t generate_predicate_mask(uint32_t num) {
  assert(num <= 32 && "predicate_mask length is 32 bits");
  if (num == 32) { return UINT32_MAX; }
  return (1u << num) - 1;
}

template <typename dtype>
inline uint32_t get_offset(const sycl::marray<int32_t, 2> &gmem_coord, const sycl::vec<uint32_t, 2> &gmem_size,
                           const sycl::vec<uint64_t, 1> &gmem_stride) {
  bool is_valid_coord = is_within_boundary<2>(gmem_coord, gmem_size);
  uint32_t offset = is_valid_coord ? gmem_coord[1] * gmem_stride[0] + gmem_coord[0] * sizeof(dtype) : 0;
  return offset;
}

template <typename dtype>
inline uint64_t get_offset_a64(dtype *ptr, const sycl::marray<int32_t, 2> &gmem_coord,
                               const sycl::vec<uint32_t, 2> &gmem_size, const sycl::vec<uint64_t, 1> &gmem_stride) {
  bool is_valid_coord = is_within_boundary<2>(gmem_coord, gmem_size);
  uint32_t offset = is_valid_coord ? gmem_coord[1] * gmem_stride[0] + gmem_coord[0] * sizeof(dtype) : 0;
  uint32_t dtype_offset = offset / sizeof(dtype);
  return reinterpret_cast<uint64_t>(ptr + dtype_offset);
}

template <typename dtype>
inline uint32_t get_copy_size(const sycl::marray<int32_t, 2> &gmem_coord, const sycl::vec<uint32_t, 2> &gmem_size,
                              uint32_t width_2d) {
  bool is_valid_coord = is_within_boundary<2>(gmem_coord, gmem_size);
  uint32_t copy_size = is_valid_coord ? (gmem_size[0] - gmem_coord[0]) * sizeof(dtype) : 0;
  copy_size = copy_size < width_2d ? copy_size : width_2d;
  return copy_size;
}
} // namespace row_copy

inline e8m0 random_scale() {
  static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  static std::default_random_engine engine(seed);
  static std::uniform_int_distribution<uint8_t> distribution(121,
                                                             128); // 2^-6 - 2^1

  uint8_t scale = distribution(engine);
  while (scale == 0xff) {
    scale = distribution(engine);
  }

  return e8m0(scale);
}

inline uint8_t random_sparsity() {
  static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  static std::default_random_engine engine(seed);
  static std::uniform_int_distribution<uint8_t> distribution(0, 3);

  uint8_t ret = 0;
  uint8_t data[8] = {0, 0, 1, 1, 0, 0, 1, 1};
  for (uint32_t i = 0; i < 4; i++) {
    std::swap(data[i], data[distribution(engine)]);
  }
  for (uint32_t i = 0; i < 4; i++) {
    std::swap(data[i + 4], data[distribution(engine) + 4]);
  }
  for (uint32_t i = 0; i < 8; i++) {
    ret |= (data[i] << i);
  }
  return ret;
}

inline float random_float() {
  double mean = 0.0;
  double stddev = 1.0;

  // Create a random number generator
  std::random_device rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

  // Create a normal distribution with the specified mean and standard deviation
  std::normal_distribution<> d(mean, stddev);
  return d(gen);
}

template <typename T>
inline T random_int(T lower, T upper) {
  // Ensure lower <= upper
  if (lower > upper) {
    throw std::invalid_argument("random_int: lower bound must be less than or equal to upper bound.");
  }

  // Create a random number generator
  std::random_device rd;
  std::mt19937 gen(rd());

  if constexpr (std::is_integral_v<T>) {
    // Use uniform_int_distribution for integral types
    std::uniform_int_distribution<T> dist(lower, upper);
    return (dist(gen));
  } else {
    static_assert(sizeof(T) == 0, "random_int only supports integral types.");
  }
}

template <typename T>
struct data_pack_helper {
  static std::vector<uint8_t> pack(const std::vector<T> &data) {
    std::vector<uint8_t> packed_data(data.size() * sizeof(T));
    std::memcpy(packed_data.data(), data.data(), data.size() * sizeof(T));
    return packed_data;
  }

  static std::vector<T> unpack(const std::vector<uint8_t> &buffer) {
    std::vector<T> unpacked_data(buffer.size() / sizeof(T));
    std::memcpy(unpacked_data.data(), buffer.data(), buffer.size());
    return unpacked_data;
  }
};

template <typename T>
std::vector<uint8_t> pack_data(std::vector<T> &data) {
  return data_pack_helper<T>::pack(data);
}

template <typename T>
std::vector<T> unpack_data(std::vector<uint8_t> &buffer) {
  return data_pack_helper<T>::unpack(buffer);
}

namespace conv2d {
struct src0_payload_t {
  uint32_t offset;
  uint32_t copy_size;
};

struct problem_shape_t {
  uint32_t in_batch;
  uint32_t in_channel;
  uint32_t in_height;
  uint32_t in_width;
  uint32_t kernel_num;
  uint32_t kernel_channel;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t padding_top;
  uint32_t padding_bottom;
  uint32_t padding_left;
  uint32_t padding_right;
  uint32_t out_batch;
  uint32_t out_channel;
  uint32_t out_height;
  uint32_t out_width;
  // added to support stride and dillation in conv2d
  uint32_t stride_h;
  uint32_t stride_w;
  uint32_t dilation_h;
  uint32_t dilation_w;

  problem_shape_t() = default;
  problem_shape_t(const problem_shape_t &problem_shape) = default;

  problem_shape_t(const sycl::vec<uint32_t, 4> &input, const sycl::vec<uint32_t, 4> &kernel,
                  const sycl::vec<uint32_t, 2> &padding_lower, const sycl::vec<uint32_t, 2> &padding_upper,
                  const sycl::vec<uint32_t, 2> &stride = {1, 1}, const sycl::vec<uint32_t, 2> &dilation = {1, 1}) {
    // Input: NHWC
    in_batch = input[3];
    in_channel = input[0];
    in_height = input[2];
    in_width = input[1];

    // Kernel: KRSC
    kernel_num = kernel[3];
    kernel_channel = kernel[0];
    kernel_height = kernel[2];
    kernel_width = kernel[1];

    padding_top = padding_upper[1];
    padding_bottom = padding_lower[1];
    padding_left = padding_lower[0];
    padding_right = padding_upper[0];

    stride_h = stride[0];
    stride_w = stride[1];

    dilation_h = dilation[0];
    dilation_w = dilation[1];

    out_batch = in_batch;
    out_channel = kernel_num;

    // modified to compute the output size considering stride and dilation
    int32_t out_height_tmp =
        (((int32_t)in_height + padding_top + padding_bottom - (kernel_height - 1) * dilation_h - 1) / stride_h) + 1;
    int32_t out_width_tmp =
        (((int32_t)in_width + padding_left + padding_right - (kernel_width - 1) * dilation_w - 1) / stride_w) + 1;
    assert(out_height_tmp > 0 && out_width_tmp > 0);
    out_height = out_height_tmp;
    out_width = out_width_tmp;
  }

  problem_shape_t &operator=(const problem_shape_t &problem_shape) = default;

  uint32_t get_in_batch() const { return in_batch; }

  uint32_t get_in_channel() const { return in_channel; }

  uint32_t get_in_height() const { return in_height; }

  uint32_t get_in_width() const { return in_width; }

  uint32_t get_kernel_num() const { return kernel_num; }

  uint32_t get_kernel_channel() const { return kernel_channel; }

  uint32_t get_kernel_height() const { return kernel_height; }

  uint32_t get_kernel_width() const { return kernel_width; }

  uint32_t get_padding_top() const { return padding_top; }

  uint32_t get_padding_bottom() const { return padding_bottom; }

  uint32_t get_padding_left() const { return padding_left; }

  uint32_t get_padding_right() const { return padding_right; }

  uint32_t get_out_batch() const { return out_batch; }

  uint32_t get_out_channel() const { return out_channel; }

  uint32_t get_out_height() const { return out_height; }

  uint32_t get_out_width() const { return out_width; }

  uint32_t get_stride_h() const { return stride_h; }

  uint32_t get_stride_w() const { return stride_w; }

  uint32_t get_dilation_h() const { return dilation_h; }

  uint32_t get_dilation_w() const { return dilation_w; }

  void print() const {
    std::cout << "Conv 2D problem shape:" << std::endl;
    std::cout << "\tinput\t{" << in_channel << ", " << in_width << ", " << in_height << ", " << in_batch << "}"
              << std::endl;
    std::cout << "\tkernel\t{" << kernel_channel << ", " << kernel_width << ", " << kernel_height << ", " << kernel_num
              << "}" << std::endl;
    std::cout << "\toutput\t{" << out_channel << ", " << out_width << ", " << out_height << ", " << out_batch << "}"
              << std::endl;
    std::cout << "\tpadding_lower\t{" << padding_left << ", " << padding_bottom << "}" << std::endl;
    std::cout << "\tpadding_upper\t{" << padding_right << ", " << padding_top << "}" << std::endl;
    std::cout << "\tstride\t{" << stride_h << ", " << stride_w << "}" << std::endl;
    std::cout << "\tdilation\t{" << dilation_h << ", " << dilation_w << "}" << std::endl;
  }
};

} // namespace conv2d
