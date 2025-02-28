#pragma once

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// for tf32, bf8 and hf8, we define them in the namespace base_type
// copied from xetla
// for mx_fp4, we also define it in the namespace base_type, it's self defined

namespace base_type {

using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

/// @brief xetla tf32 data type.
/// The difference between tf32 and fp32 is:
///
/// fp32: 0_00000000_00000000000000000000000
///
/// tf32: 0_00000000_0000000000
/// @note
/// The member function in tf32 class is only used in host side.
/// For device side, we will automatically convert it to its native type.
/// @see native_type_t
///
struct tf32 {
  uint32_t data;

  //#ifdef __SYCL_DEVICE_ONLY__
  //    ///
  //#else
  operator float() const {
    uint32_t temp = data;
    return *reinterpret_cast<float *>(&temp);
  }

  tf32(float val) { data = (*reinterpret_cast<uint32_t *>(&val)) & 0xFFFFE000; }

  tf32 &operator=(float val) {
    this->data = (*reinterpret_cast<uint32_t *>(&val)) & 0xFFFFE000;
    return *this;
  }

  //#endif
};

/// @brief xetla bf8 data type.
/// The difference between bf8 and fp16 is:
///
/// fp16: 0_00000_0000000000
///
/// bf8:  0_00000_00
/// FP32 => BF8 conversion is handled in two steps
/// First convert to FP16, then to BF8/FP32
/// @note
/// The member function in bf8 class is only used in host side.
/// For device side, we will automatically convert it to its native type.
/// @see native_type_t
///
struct bf8 {
#ifdef _WIN32
  typedef unsigned short ushort;
#endif
  uint8_t data = 0;

  operator float() const {
    uint16_t temp = data;
    temp = temp << 0x8;
    fp16 temp_fp16 = *reinterpret_cast<fp16 *>(&temp);
    float temp_fp32 = temp_fp16;
    return temp_fp32;
  }

  bf8() {}

  bf8(float val) {
    fp16 val_fp16 = val;
    ushort *p = (ushort *)&val_fp16;
    ushort sign = p[0] >> 15;
    ushort exp = (p[0] >> 10) & 0b11111;
    ushort mant = p[0] & 0x3FF;
    ushort q_mant = (mant >> 8) & 0b11;
    uint8_t ret_tmp;
    // RNE rounding to convert FP16 mantissa to BF8 mantissa
    q_mant += (mant & 0x80) && ((mant & 0x7F) || (mant & 0x100));
    ret_tmp = sign << 7 | exp << 2;
    ret_tmp += q_mant;
    data = ret_tmp;
  }

  bf8 &operator=(float val) {

    fp16 val_fp16 = val;
    ushort *p = (ushort *)&val_fp16;
    ushort sign = p[0] >> 15;
    ushort exp = (p[0] >> 10) & 0b11111;
    ushort mant = p[0] & 0x3FF;
    ushort q_mant = (mant >> 8) & 0b11;
    uint8_t ret_tmp;

    // RNE rounding to convert FP16 mantissa to BF8 mantissa
    q_mant += (mant & 0x80) && ((mant & 0x7F) || (mant & 0x100));
    ret_tmp = sign << 7 | exp << 2;
    ret_tmp += q_mant;
    this->data = ret_tmp;
    return *this;
  }
};

/// @brief xetla hf8 data type.
/// The difference between hf8 and fp16 is:
///
/// fp16: 0_00000_0000000000
///
/// hf8:  0_0000_000
/// FP32 => HF8 conversion is handled in two steps
/// First convert to FP16, then to HF8/FP32
/// @note
/// The member function in hf8 class is only used in host side.
/// For device side, we will automatically convert it to its native type.
/// @see native_type_t
///
struct hf8 {
  uint8_t data;
  static constexpr int16_t max_exp_unbiased = 8;
  static constexpr int16_t min_exp_unbiased = -6;
  static constexpr int16_t exp_bias = 7;
  static constexpr uint16_t exp_size = 4;
  static constexpr uint16_t mant_size = 3;
  static constexpr uint8_t nan = 0x7f;
  static constexpr uint8_t max_val = 0x7e;
  static constexpr bool is_saturation = false;

  hf8() {}

  operator float() const {
    // Extract individual fields from hf8
    uint16_t sign = data >> 7;
    uint16_t exp = (data >> 3) & 0b1111;
    uint16_t mant = data & 0x07;
    uint16_t dst_val;
    if ((exp == 0xf) && (mant == 0x7)) {
      dst_val = 0x7fff;
    } else if ((exp == 0) && (mant == 0)) {
      dst_val = 0;
    } else if ((exp == 0) && (mant != 0)) {
      uint16_t lz_count = (mant > 3) ? 0 : ((mant > 1) ? 1 : 2);
      uint16_t dst_exp = exp - exp_bias + 15 - lz_count;
      uint16_t dst_mant = (mant << (lz_count + 1)) & 0x7;
      dst_val = (dst_exp << 10) | (dst_mant << 7);
    } else {
      uint16_t dst_exp = exp - exp_bias + 15;
      dst_val = (dst_exp << 10) | (mant << 7);
    }

    uint16_t temp = (sign << 15) | dst_val;
    fp16 temp_fp16 = *reinterpret_cast<fp16 *>(&temp);
    float temp_fp32 = temp_fp16;
    return temp_fp32;
  }

  hf8(float val) {
    // Convert to fp16
    fp16 val_fp16 = val;
    uint16_t *p = (uint16_t *)&val_fp16;
    uint16_t src = p[0];
    // Convert to hf8
    static constexpr uint16_t src_exp_size = 5;
    static constexpr uint16_t src_mant_size = 10;
    static constexpr uint16_t src_exp_bias = (1 << (src_exp_size - 1)) - 1;
    static constexpr uint16_t src_exp_mask = (1 << src_exp_size) - 1;
    static constexpr uint16_t src_mant_mask = (1 << src_mant_size) - 1;

    uint16_t src_sign = src >> (src_exp_size + src_mant_size);
    uint16_t src_exp = (src >> src_mant_size) & src_exp_mask;
    int16_t src_exp_unbiased = src_exp - src_exp_bias;
    uint16_t src_mant = src & src_mant_mask;

    bool is_src_inf_nan = src_exp == 0x1f;
    bool is_overflow = (src_exp_unbiased > max_exp_unbiased)
        // max normal mantissa is 0b110, RNE round
        || ((src_exp_unbiased == max_exp_unbiased) && (src_mant > 0x0340));
    bool is_zero = (src_exp_unbiased < (min_exp_unbiased - mant_size));
    bool is_denorm = (src_exp_unbiased < min_exp_unbiased) && (!is_zero);

    uint8_t dst_val;
    if (is_src_inf_nan) {
      dst_val = nan;
    } else if (is_overflow) {
      dst_val = is_saturation ? max_val : nan;
    } else if (is_zero) {
      dst_val = 0;
    } else if (is_denorm) {
      // src_denormal case already in is_zero branch
      uint16_t src_m = src_mant | 0x0400;
      int16_t shift_out_bit = min_exp_unbiased - src_exp_unbiased;
      bool sticky_flag = (src_m & ((1 << shift_out_bit) - 1)) != 0;
      src_m = src_m >> shift_out_bit;
      // RNE rounding
      uint16_t tail_size = src_mant_size - mant_size;
      // exclude the rounding bit
      sticky_flag = sticky_flag || ((src_m & ((1 << (tail_size - 1)) - 1)) != 0);
      bool lsb_bit = src_m & (1 << tail_size);
      bool rnd_bit = src_m & (1 << (tail_size - 1));
      bool carry = (lsb_bit && rnd_bit) || (rnd_bit && sticky_flag);

      dst_val = (src_m >> tail_size) + carry;
    } else {
      uint16_t tail_size = src_mant_size - mant_size;
      // exclude the rounding bit
      bool sticky_flag = (src_mant & ((1 << (tail_size - 1)) - 1)) != 0;
      bool lsb_bit = src_mant & (1 << tail_size);
      bool rnd_bit = src_mant & (1 << (tail_size - 1));
      bool carry = (lsb_bit && rnd_bit) || (rnd_bit && sticky_flag);
      uint16_t src_m = (src_mant >> tail_size) + carry;
      uint16_t src_e = src_exp_unbiased + exp_bias;
      // overflow will be handled in is_overflow
      dst_val = (src_e << mant_size) + src_m;
    }
    data = (src_sign << (exp_size + mant_size)) | dst_val;
  }

  hf8 &operator=(float val) {
    // Convert to fp16
    fp16 val_fp16 = val;
    uint16_t *p = (uint16_t *)&val_fp16;
    uint16_t src = p[0];
    // Convert to hf8
    static constexpr uint16_t src_exp_size = 5;
    static constexpr uint16_t src_mant_size = 10;
    static constexpr uint16_t src_exp_bias = (1 << (src_exp_size - 1)) - 1;
    static constexpr uint16_t src_exp_mask = (1 << src_exp_size) - 1;
    static constexpr uint16_t src_mant_mask = (1 << src_mant_size) - 1;

    uint16_t src_sign = src >> (src_exp_size + src_mant_size);
    uint16_t src_exp = (src >> src_mant_size) & src_exp_mask;
    int16_t src_exp_unbiased = src_exp - src_exp_bias;
    uint16_t src_mant = src & src_mant_mask;

    bool is_src_inf_nan = src_exp == 0x1f;
    bool is_overflow = (src_exp_unbiased > max_exp_unbiased)
        // max normal mantissa is 0b110, RNE round
        || ((src_exp_unbiased == max_exp_unbiased) && (src_mant > 0x0340));
    bool is_zero = (src_exp_unbiased < (min_exp_unbiased - mant_size));
    bool is_denorm = (src_exp_unbiased < min_exp_unbiased) && (!is_zero);

    uint8_t dst_val;
    if (is_src_inf_nan) {
      dst_val = nan;
    } else if (is_overflow) {
      dst_val = is_saturation ? max_val : nan;
    } else if (is_zero) {
      dst_val = 0;
    } else if (is_denorm) {
      // src_denormal case already in is_zero branch
      uint16_t src_m = src_mant | 0x0400;
      int16_t shift_out_bit = min_exp_unbiased - src_exp_unbiased;
      bool sticky_flag = (src_m & ((1 << shift_out_bit) - 1)) != 0;
      src_m = src_m >> shift_out_bit;
      // RNE rounding
      uint16_t tail_size = src_mant_size - mant_size;
      // exclude the rounding bit
      sticky_flag = sticky_flag | (src_m & ((1 << (tail_size - 1)) - 1));
      bool lsb_bit = src_m & (1 << tail_size);
      bool rnd_bit = src_m & (1 << (tail_size - 1));
      bool carry = (lsb_bit && rnd_bit) || (rnd_bit && sticky_flag);

      dst_val = (src_m >> tail_size) + carry;
    } else {
      uint16_t tail_size = src_mant_size - mant_size;
      // exclude the rounding bit
      bool sticky_flag = src_mant & ((1 << (tail_size - 1)) - 1);
      bool lsb_bit = src_mant & (1 << tail_size);
      bool rnd_bit = src_mant & (1 << (tail_size - 1));
      bool carry = (lsb_bit && rnd_bit) || (rnd_bit && sticky_flag);
      uint16_t src_m = (src_mant >> tail_size) + carry;
      uint16_t src_e = src_exp_unbiased + exp_bias;
      // overflow will be handled in is_overflow
      dst_val = (src_e << mant_size) + src_m;
    }
    this->data = (src_sign << (exp_size + mant_size)) | dst_val;
    return *this;
  }
};

namespace impl {
// inside of namespace impl, not expose to user
struct fp4_e3m0 {
  uint8_t data;
  fp4_e3m0() = default;

  explicit fp4_e3m0(uint8_t val) { data = val; }

  // down cvt from float
  explicit fp4_e3m0(float val) {
    // initial implementation, now round to nearest, no other rounding mode
    // supported
    if (std::isnan(val) || std::isinf(val)) { data = 0x8; }

    static std::vector<float> LUT = {-16, -8, -4, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4, 8, 16};
    static std::vector<uint8_t> D_LUT = {0xf, 0xe, 0xd, 0xc, 0xb, 0xa, 0x9, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7};

    // find the closet value in LUT
    auto it = std::lower_bound(LUT.begin(), LUT.end(), val);

    if (it == LUT.begin()) {
      data = D_LUT.front();
    } else if (it == LUT.end()) {
      data = D_LUT.back();
    } else {
      int idx = it - LUT.begin();
      if (std::abs(LUT[idx] - val) < std::abs(LUT[idx - 1] - val)) {
        data = D_LUT[idx];
      } else {
        data = D_LUT[idx - 1];
      }
    }
  }

  // only process least 4bits
  explicit operator bf8() const {
    std::vector<uint8_t> LUT = {0x00, 0x34, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c,
                                0x7f, 0xb4, 0xb8, 0xbc, 0xc0, 0xc4, 0xc8, 0xcc};
    uint32_t idx = data & 0xf;
    uint8_t looked_val = LUT[idx];
    bf8 looked_val_bf8 = (*reinterpret_cast<bf8 *>(&looked_val));
    return looked_val_bf8;
  }

  // only process least 4bits
  explicit operator fp16() const {
    std::vector<uint16_t> LUT = {0x0000, 0x3400, 0x3800, 0x3c00, 0x4000, 0x4400, 0x4800, 0x4c00,
                                 0x7fff, 0xb400, 0xb800, 0xbc00, 0xc000, 0xc400, 0xc800, 0xcc00};
    uint32_t idx = data & 0xf;
    uint16_t looked_val = LUT[idx];
    fp16 looked_val_fp16 = (*reinterpret_cast<fp16 *>(&looked_val));
    return looked_val_fp16;
  }

  explicit operator float() const {
    constexpr float FNAN = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> LUT = {0, 0.25, 0.5, 1, 2, 4, 8, 16, FNAN, -0.25, -0.5, -1, -2, -4, -8, -16};
    uint32_t idx = data & 0xf;
    float looked_val = LUT[idx];
    return looked_val;
  }
};

} // namespace impl

struct fp4_e3m0x2 {
  fp4_e3m0x2() = default;

  fp4_e3m0x2(uint8_t val) {
    high = val >> 4;
    low = val & 0xf;
  }

  fp4_e3m0x2(uint8_t hi, uint8_t li) {
    high = hi;
    low = li;
  }

  template <typename type_to>
  type_to cvt(int index) const {
    if (index == 0) {
      return static_cast<type_to>(impl::fp4_e3m0(low));
    } else {
      return static_cast<type_to>(impl::fp4_e3m0(high));
    }
  }

  void set(int index, float val) {
    if (index == 0) {
      low = impl::fp4_e3m0(val).data;
    } else {
      high = impl::fp4_e3m0(val).data;
    }
  }

  uint8_t low : 4;
  uint8_t high : 4;

  static constexpr uint32_t largest_pow_of_2 = 16;
};

static_assert(sizeof(fp4_e3m0x2) == 1);

struct e8m0 {
  static constexpr int bias = 127;
  uint8_t data;

  e8m0() = default;

  explicit e8m0(uint8_t val) { this->data = val; }

  explicit e8m0(float val) {
    if (std::isnan(val)) {
      data = 0xff;
      return;
    }
    // get the exp bits of the float
    // Mask to extract the exponent bits (bits 23 to 30)
    uint32_t expMask = 0x7F800000;
    uint32_t *valBits = reinterpret_cast<uint32_t *>(&val);
    data = ((*valBits & expMask) >> 23);
  }

  explicit operator float() const {
    if (data == 0xff) { return std::numeric_limits<float>::quiet_NaN(); }
    return std::pow(2, int(data) - bias);
  }

  explicit operator double() const { return static_cast<double>(static_cast<float>(*this)); }

  // the function for << operator
  friend std::ostream &operator<<(std::ostream &os, const e8m0 &e) {
    os << static_cast<float>(e);
    return os;
  }
};

static_assert(sizeof(e8m0) == 1);

} // namespace base_type

using bf8 = base_type::bf8;
using hf8 = base_type::hf8;
using bf16 = base_type::bf16;
using fp16 = base_type::fp16;
using tf32 = base_type::tf32;
using e8m0 = base_type::e8m0;
using fp4_e3m0 = base_type::impl::fp4_e3m0;

template <typename T>
struct type_repr;

template <typename T>
constexpr auto type_repr_v = type_repr<T>::value;

#define DEFINE_TYPE_REPR(Type, Name) \
  template <> \
  struct type_repr<Type> { \
    static constexpr std::string_view value = Name; \
  }

DEFINE_TYPE_REPR(float, "FP32");
DEFINE_TYPE_REPR(double, "FP64");
DEFINE_TYPE_REPR(uint32_t, "U32");
DEFINE_TYPE_REPR(uint16_t, "U16");
DEFINE_TYPE_REPR(bf8, "BF8");
DEFINE_TYPE_REPR(hf8, "HF8");
DEFINE_TYPE_REPR(bf16, "BF16");
DEFINE_TYPE_REPR(fp16, "FP16");
DEFINE_TYPE_REPR(tf32, "TF32");
DEFINE_TYPE_REPR(e8m0, "E8M0");
DEFINE_TYPE_REPR(fp4_e3m0, "FP4E3M0");
