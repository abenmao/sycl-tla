#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <mkl.h>
#include <sstream>
#include <cute/arch/xe4_util.hpp>

template <typename T>
struct tolerance_traits;

template <>
struct tolerance_traits<uint8_t> {
  static constexpr uint8_t abs_tolerance = 0;
  static constexpr double rel_tolerance = 0;
  static constexpr uint32_t ulp_tolerance = 0;
};

template <>
struct tolerance_traits<int8_t> {
  static constexpr uint8_t abs_tolerance = 0;
  static constexpr double rel_tolerance = 0;
  static constexpr uint32_t ulp_tolerance = 0;
};

template <>
struct tolerance_traits<uint16_t> {
  static constexpr uint16_t abs_tolerance = 0;
  static constexpr double rel_tolerance = 0;
  static constexpr uint32_t ulp_tolerance = 0;
};

template <>
struct tolerance_traits<uint32_t> {
  static constexpr uint32_t abs_tolerance = 0;
  static constexpr double rel_tolerance = 0;
  static constexpr uint32_t ulp_tolerance = 0;
};

template <>
struct tolerance_traits<int32_t> {
  static constexpr int32_t abs_tolerance = 0;
  static constexpr double rel_tolerance = 0;
  static constexpr uint32_t ulp_tolerance = 0;
};

template <>
struct tolerance_traits<uint64_t> {
  static constexpr uint32_t abs_tolerance = 0;
  static constexpr double rel_tolerance = 0;
  static constexpr uint32_t ulp_tolerance = 0;
};

template <>
struct tolerance_traits<float> {
  static constexpr double abs_tolerance = 0.25;
  static constexpr double rel_tolerance = 0.001;
  static constexpr uint32_t ulp_tolerance = 8;
};

template <>
struct tolerance_traits<double> {
  static constexpr double abs_tolerance = 0.25;
  static constexpr double rel_tolerance = 0.001;
  static constexpr uint32_t ulp_tolerance = 8;
};

template <>
struct tolerance_traits<bf8> {
  static constexpr double abs_tolerance = 0.25;
  static constexpr double rel_tolerance = 0.01;
  static constexpr uint32_t ulp_tolerance = 8;
};

template <>
struct tolerance_traits<bf16> {
  static constexpr double abs_tolerance = 0.25;
  static constexpr double rel_tolerance = 0.001;
  static constexpr uint32_t ulp_tolerance = 8;
};

template <>
struct tolerance_traits<fp16> {
  static constexpr double abs_tolerance = 0.25;
  static constexpr double rel_tolerance = 0.001;
  static constexpr uint32_t ulp_tolerance = 8;
};

// template <>
// struct tolerance_traits<tf32> {
//     static constexpr double abs_tolerance = 0.25;
//     static constexpr double rel_tolerance = 0.001;
//     static constexpr uint32_t ulp_tolerance = 65536;
// };

template <typename T>
struct tolerance {
  double abs_tolerance;
  double rel_tolerance;
  uint32_t ulp_tolerance;

  tolerance() :
      abs_tolerance(tolerance_traits<T>::abs_tolerance),
      rel_tolerance(tolerance_traits<T>::rel_tolerance),
      ulp_tolerance(tolerance_traits<T>::ulp_tolerance) {}

  tolerance(double abs_tol, double rel_tol, uint32_t ulp_tol) :
      abs_tolerance(abs_tol), rel_tolerance(rel_tol), ulp_tolerance(ulp_tol) {}
};

struct log_base {
  int inf_cnt = 0;
  int nan_cnt = 0;
  int normal_cnt = 0;
  int sum_cnt = 0;

  log_base() = default;

  void increase_inf_cnt() {
    inf_cnt++;
    sum_cnt++;
  }

  void increase_nan_cnt() {
    nan_cnt++;
    sum_cnt++;
  }

  void increase_normal_cnt() {
    normal_cnt++;
    sum_cnt++;
  }
};

class error_log : public log_base {
  public:
  void print() {
    std::cout << "A total of " << sum_cnt << " errors were generated." << std::endl;
    if (sum_cnt) {
      std::cout << "\terrors caused by Inf: " << inf_cnt << std::endl
                << "\terrors caused by NaN: " << nan_cnt << std::endl
                << "\terrors caused by normal cases: " << normal_cnt << std::endl;
    }
  }
};

class warning_log : public log_base {
  public:
  void print() {
    std::cout << "A total of " << sum_cnt << " warnings were generated." << std::endl;
    if (sum_cnt) {
      std::cout << "\twarnings caused by Inf: " << inf_cnt << std::endl
                << "\twarnings caused by NaN: " << nan_cnt << std::endl
                << std::endl;
    }
  }
};

template <typename T, typename enable = void>
class logger_t {};

template <typename T>
class logger_t<T, std::enable_if_t<is_floating_t<T>::value>> {
  error_log err_log = {};
  warning_log warn_log = {};

  tolerance<T> tol;

  int print_cnt;

  double max_abs_diff = -1;
  double max_abs_gold = 0;
  double max_abs_res = 0;

  double max_rel_diff = -1;
  double max_rel_gold = 0;
  double max_rel_res = 0;

  int32_t max_ulp_diff = -1;
  uint32_t max_ulp_gold = 0;
  uint32_t max_ulp_res = 0;

  void update_log_for_normal(T gold, T res, int32_t idx) {
    double abs_diff = std::fabs(static_cast<double>(gold) - static_cast<double>(res));
    double max_val = std::max(std::fabs(static_cast<double>(gold)), std::fabs(static_cast<double>(res)));
    double rel_diff = (static_cast<double>(max_val) == 0.f) ? 0.f : static_cast<double>(abs_diff / max_val);
    uint32_t ulp_diff = get_ulp_diff(gold, res);

    update_max_diff(gold, res, abs_diff, rel_diff, ulp_diff);

    bool are_equal = abs_diff <= tol.abs_tolerance || rel_diff <= tol.rel_tolerance || ulp_diff <= tol.ulp_tolerance;

    if (!are_equal) {
      if (idx != -1) {
        if (err_log.sum_cnt < print_cnt) {
          std::cout << "\tidx: " << idx << " data_val: " << res << " gold_val: " << gold << std::endl;
        }
      }
      err_log.increase_normal_cnt();
    }
  }

  public:
  logger_t(tolerance<T> tol_ = {}, int print_cnt_ = 1000) : tol(tol_), print_cnt(print_cnt_) {}

  void log(T gold, T res, int32_t idx = -1) {
    if (is_infinity(gold) || is_infinity(res)) {
      if (is_infinity(gold) && is_infinity(res) && have_same_sign(gold, res)) {
        warn_log.increase_inf_cnt();
      } else {
        err_log.increase_inf_cnt();
      }
    } else if (is_nan(gold) || is_nan(res)) {
      if (is_nan(gold) && is_nan(res)) {
        warn_log.increase_nan_cnt();
      } else {
        err_log.increase_nan_cnt();
      }
    } else {
      update_log_for_normal(gold, res, idx);
    }
  }

  void print_summary() {
    warn_log.print();
    err_log.print();

    std::cout << "max_abs_diff = " << max_abs_diff << ", ";
    std::cout << "gold = " << max_abs_gold << ", res = " << max_abs_res << std::endl;
    std::cout << "max_rel_diff = " << max_rel_diff << ", ";
    std::cout << "gold = " << max_rel_gold << ", res = " << max_rel_res << std::endl;
    std::cout << "max_ulp_diff = " << max_ulp_diff << ", ";
    std::cout << "gold = " << std::hex << max_ulp_gold << ", res = " << max_ulp_res << std::dec << std::endl
              << std::endl;
  }

  int get_err_cnt() { return err_log.sum_cnt; }

  private:
  bool have_same_sign(T a, T b) { return std::signbit(static_cast<double>(a)) == std::signbit(static_cast<double>(b)); }

  bool is_infinity(T a) { return std::isinf(static_cast<double>(a)); }

  bool is_nan(T num) { return std::isnan(static_cast<double>(num)); }

  int32_t get_ulp_diff(T a, T b) {
    uint32_t ulp_a = *reinterpret_cast<uint_type_t<T> *>(&a);
    uint32_t ulp_b = *reinterpret_cast<uint_type_t<T> *>(&b);
    return ulp_a > ulp_b ? ulp_a - ulp_b : ulp_b - ulp_a;
  }

  void update_max_diff(T gold, T res, double abs_diff, double rel_diff, int32_t ulp_diff) {
    if (abs_diff > max_abs_diff) {
      max_abs_diff = abs_diff;
      max_abs_gold = static_cast<double>(gold);
      max_abs_res = static_cast<double>(res);
    }
    if (rel_diff > max_rel_diff) {
      max_rel_diff = rel_diff;
      max_rel_gold = static_cast<double>(gold);
      max_rel_res = static_cast<double>(res);
    }
    if (ulp_diff > max_ulp_diff) {
      max_ulp_diff = ulp_diff;
      max_ulp_gold = *reinterpret_cast<uint_type_t<T> *>(&gold);
      max_ulp_res = *reinterpret_cast<uint_type_t<T> *>(&res);
    }
  }
};

template <typename T>
class logger_t<T, std::enable_if_t<std::is_integral_v<T>>> {
  error_log err_log = {};

  tolerance<T> tol;

  int print_cnt;

  uint64_t max_abs_diff = -1;
  T max_abs_gold = 0;
  T max_abs_res = 0;

  double max_rel_diff = -1;
  T max_rel_gold = 0;
  T max_rel_res = 0;

  void update_log_for_normal(T gold, T res, int32_t idx) {
    auto abs_diff = std::fabs(gold - res);
    auto max_val = std::max(std::fabs(gold), std::fabs(res));
    double rel_diff = (max_val == static_cast<double>(0.0))
        ? static_cast<T>(0.0)
        : static_cast<uint64_t>(abs_diff) / static_cast<uint64_t>(max_val);

    update_max_diff(gold, res, abs_diff, rel_diff);

    bool are_equal = abs_diff <= tol.abs_tolerance || rel_diff <= tol.rel_tolerance;

    if (!are_equal) {
      if (idx != -1) {
        if (err_log.sum_cnt < print_cnt) {
          std::cout << "\tidx: " << idx << " data_val: " << res << " gold_val: " << gold << std::endl;
        }
      }
      err_log.increase_normal_cnt();
    }
  }

  public:
  logger_t(tolerance<T> tol_ = {}, int print_cnt_ = 10) : tol(tol_), print_cnt(print_cnt_) {}

  void log(T gold, T res, int32_t idx = -1) { update_log_for_normal(gold, res, idx); }

  void print_summary() {
    err_log.print();
    std::cout << "max_abs_diff = " << uint64_t(max_abs_diff) << ", ";
    std::cout << "gold = " << int32_t(max_abs_gold) << ", res = " << int32_t(max_abs_res) << std::endl;
    std::cout << "max_rel_diff = " << max_rel_diff << ", ";
    std::cout << "gold = " << int32_t(max_rel_gold) << ", res = " << int32_t(max_rel_res) << std::endl << std::endl;
  }

  int get_err_cnt() { return err_log.sum_cnt; }

  private:
  void update_max_diff(T gold, T res, T abs_diff, double rel_diff) {
    if ((max_abs_diff == std::numeric_limits<uint64_t>::max()) || (static_cast<uint64_t>(abs_diff) > max_abs_diff)) {
      max_abs_diff = abs_diff;
      max_abs_gold = gold;
      max_abs_res = res;
    }
    if ((max_rel_diff == -1) || (rel_diff > max_rel_diff)) {
      max_rel_diff = rel_diff;
      max_rel_gold = gold;
      max_rel_res = res;
    }
  }
};

template <uint32_t Dim>
uint32_t get_size(const sycl::vec<uint32_t, Dim> &shape) {
  uint32_t res = 1;
  for (uint32_t i = 0; i < Dim; i++) {
    res *= shape[i];
  }
  return res;
}

template <class dtype>
std::vector<dtype> get_box_data(dtype *base_ptr, uint32_t pitch, const sycl::vec<int32_t, 2> &gmem_coord,
                                const sycl::vec<uint32_t, 2> &box_shape, const uint32_t padding_left = 0,
                                const uint32_t padding_right = 0) {
  std::vector<dtype> ret(get_size<2>(box_shape));
  for (uint32_t i = padding_left; i < box_shape[1] - padding_right; i++) {
    int32_t row_idx = i + gmem_coord[1];
    for (uint32_t j = 0; j < box_shape[0]; j++) {
      int32_t col_idx = j + gmem_coord[0];
      uint32_t dst_offset = i * box_shape[0] + j;
      uint32_t src_offset = row_idx * pitch / sizeof(dtype) + col_idx;
      ret[dst_offset] = base_ptr[src_offset];
    }
  }
  return ret;
}

template <class dtype_a, class dtype_b, class blas_type = float>
void get_gemm_gold(const size_t m, const size_t k, const size_t n, const mem_layout layout_a, const mem_layout layout_b,
                   dtype_a *A, dtype_b *B, blas_type *C, const blas_type alpha = 1.0, const blas_type beta = 0.0,
                   const CBLAS_LAYOUT layout = CblasRowMajor) {
  std::vector<blas_type> tmp_A(A, A + m * k);
  std::vector<blas_type> tmp_B(B, B + k * n);

  CBLAS_TRANSPOSE transa, transb;
  transa = transb = CblasNoTrans;
  size_t lda, ldb, ldc;

  if (layout == CblasRowMajor) {
    ldc = n > 1 ? n : 1;
    if (layout_a == mem_layout::col_major) transa = CblasTrans;
    if (layout_b == mem_layout::col_major) transb = CblasTrans;
    if (transa == CblasNoTrans)
      lda = k > 1 ? k : 1;
    else
      lda = m > 1 ? m : 1;
    if (transb == CblasNoTrans)
      ldb = n > 1 ? n : 1;
    else
      ldb = k > 1 ? k : 1;
  } else {
    ldc = m > 1 ? m : 1;
    if (layout_a == mem_layout::row_major) transa = CblasTrans;
    if (layout_b == mem_layout::row_major) transb = CblasTrans;
    if (transa == CblasNoTrans)
      lda = m > 1 ? m : 1;
    else
      lda = k > 1 ? k : 1;
    if (transb == CblasNoTrans)
      ldb = k > 1 ? k : 1;
    else
      ldb = n > 1 ? n : 1;
  }

  if constexpr (std::is_same<std::remove_cv_t<blas_type>, float>::value)
    cblas_sgemm(layout, transa, transb, m, n, k, alpha, tmp_A.data(), lda, tmp_B.data(), ldb, beta, C, ldc);
  else if constexpr (std::is_same<std::remove_cv_t<blas_type>, double>::value)
    cblas_dgemm(layout, transa, transb, m, n, k, alpha, tmp_A.data(), lda, tmp_B.data(), ldb, beta, C, ldc);
}

struct NoOp {};

template <typename dtype_c, typename dtype_acc>
int check_and_log(dtype_c *C, dtype_acc *golden, uint32_t matrix_m, uint32_t matrix_n, tolerance<dtype_c> tol) {
  std::vector<dtype_c> gold_c(golden, golden + matrix_m * matrix_n);
  logger_t<dtype_c> logger(tol);
  for (uint32_t i = 0; i < matrix_m * matrix_n; i++) {
    dtype_c cpu = gold_c[i];
    dtype_c gpu = C[i];
    logger.log(cpu, gpu, i);
  }

  auto err_cnt = logger.get_err_cnt();
  logger.print_summary();
  return err_cnt;
}

template <typename TA, typename TB, typename TC, typename TAcc = float, typename Prelogue = NoOp, typename Epilogue = NoOp>
int validate_gemm_result(TA *A, TB *B, TC *C, uint32_t matrix_m, uint32_t matrix_n, uint32_t matrix_k,
                         mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major,
						             Prelogue prelogue = Prelogue{}, Epilogue epilogue = Epilogue{}, bool negative_axb = false, tolerance<TC> tol = {}) {
  TAcc alpha = negative_axb ? -1.0 : 1.0;
  std::vector<TAcc> gold_acc(matrix_m * matrix_n, 0);

  if constexpr (std::is_same_v<Prelogue, NoOp>) {
    uint32_t matrix_k_sparse = matrix_k;
    get_gemm_gold(matrix_m, matrix_k_sparse, matrix_n, layout_a, layout_b, A, B, gold_acc.data(), alpha);
  } else {
    auto [A_, B_] = prelogue(A, B, matrix_m, matrix_n, layout_a, layout_b);
    get_gemm_gold(matrix_m, B_.size() / matrix_n, matrix_n, layout_a, layout_b, A_.data(), B_.data(), gold_acc.data(), alpha);
  }

  if constexpr (std::is_same_v<Epilogue, NoOp>) {
    return check_and_log(C, gold_acc.data(), matrix_m, matrix_n, tol);
  } else {
    auto gold_c = epilogue(gold_acc);
    return check_and_log(C, gold_c.data(), matrix_m, matrix_n, tol);
  }
}

template <typename dtype_a, typename dtype_acc, typename dtype_mxfp_meta>
void upcast_mxfp_mat_a(dtype_a *A, dtype_mxfp_meta *a_meta, uint32_t m, uint32_t k, mem_layout layout_a, bool a_scaling,
                       dtype_acc *upcast_a, uint32_t scale_ele_num) {
  bool trans_a = layout_a == mem_layout::col_major;

  // pre-processing for a, no need to consider packed type
  if (trans_a) {
    for (uint32_t i = 0; i < m; i++) { // m dimension
      for (uint32_t j = 0; j < k; j++) { // k dimension
        dtype_acc elem = static_cast<float>(A[j * m + i]);

        if (a_scaling) {
          float scale = float(a_meta[(j / scale_ele_num) * m + i]); // meta is m-major
          upcast_a[j * m + i] = elem * scale;
        } else {
          upcast_a[j * m + i] = elem;
        }
      }
    }
  } else {
    for (uint32_t i = 0; i < m; i++) { // m dimension
      for (uint32_t j = 0; j < k; j++) { // k dimension
        dtype_acc elem = static_cast<float>(A[i * k + j]);

        if (a_scaling) {
          float scale = float(a_meta[(j / scale_ele_num) * m + i]); // meta is m-major
          upcast_a[i * k + j] = elem * scale;
        } else {
          upcast_a[i * k + j] = elem;
        }
      }
    }
  }
}

template <typename dtype_b, typename dtype_acc, typename dtype_mxfp_meta>
void upcast_mxfp_mat_b(dtype_b *B, dtype_mxfp_meta *b_meta, uint32_t k, uint32_t n, mem_layout layout_b, bool b_scaling,
                       dtype_acc *upcast_b, uint32_t scale_ele_num) {
  bool trans_b = layout_b == mem_layout::col_major;

  // pre-processing for b, need to consider packed type
  if (trans_b) {
    for (uint32_t i = 0; i < n; i++) { // n dimension
      for (uint32_t j = 0; j < k; j++) { // k dimension
        dtype_acc elem = static_cast<float>(B[i * k + j]);

        if (b_scaling) {
          float scale = float(b_meta[(j / scale_ele_num) * n + i]); // meta is n-major
          upcast_b[i * k + j] = elem * scale;
        } else {
          upcast_b[i * k + j] = elem;
        }
      }
    }
  } else {
    for (uint32_t i = 0; i < n; i++) { // n dimension
      for (uint32_t j = 0; j < k; j++) { // k dimension
        dtype_acc elem = static_cast<float>(B[j * n + i]);

        if (b_scaling) {
          float scale = float(b_meta[(j / scale_ele_num) * n + i]); // meta is n-major
          upcast_b[j * n + i] = elem * scale;
        } else {
          upcast_b[j * n + i] = elem;
        }
      }
    }
  }
}

template <typename dtype_a, typename dtype_b, typename dtype_c, typename dtype_meta, typename dtype_acc = float>
std::vector<dtype_c> calculate_golden_result(dtype_a *A, dtype_b *B, dtype_meta *a_meta, dtype_meta *b_meta,
                                             uint32_t matrix_m, uint32_t matrix_n, uint32_t matrix_k,
                                             mem_layout layout_a = mem_layout::row_major,
                                             mem_layout layout_b = mem_layout::row_major, bool a_scaling = false,
                                             bool b_scaling = false, bool negative_axb = false) {
  const size_t m = matrix_m;
  const size_t k = matrix_k;
  const size_t n = matrix_n;

  std::vector<dtype_acc> gold_acc(m * n, 0);
  dtype_acc alpha = negative_axb ? -1.0 : 1.0;

  static constexpr uint32_t scale_ele_num = 32;
  std::vector<dtype_acc> upcast_a(m * k);
  std::vector<dtype_acc> upcast_b(n * k);

  upcast_mxfp_mat_a(A, a_meta, m, k, layout_a, a_scaling, upcast_a.data(), scale_ele_num);
  upcast_mxfp_mat_b(B, b_meta, k, n, layout_b, b_scaling, upcast_b.data(), scale_ele_num);

  get_gemm_gold<dtype_acc, dtype_acc, dtype_acc>(m, k, n, layout_a, layout_b, upcast_a.data(), upcast_b.data(),
                                                 gold_acc.data(), alpha);

  return std::vector<dtype_c>(gold_acc.data(), gold_acc.data() + matrix_m * matrix_n);
}

template <typename dtype_a, typename dtype_b, typename dtype_c, typename dtype_meta, typename dtype_acc = float>
int validate_mxfp_gemm_result(dtype_a *A, dtype_b *B, dtype_c *C, uint32_t matrix_m, uint32_t matrix_n,
                              uint32_t matrix_k, bool a_scaling, bool b_scaling, dtype_meta *a_meta, dtype_meta *b_meta,
                              mem_layout layout_a = mem_layout::row_major, mem_layout layout_b = mem_layout::row_major,
                              bool negative_axb = false, tolerance<dtype_c> tol = {}) {

  std::vector<dtype_c> gold_c = calculate_golden_result<dtype_a, dtype_b, dtype_c, dtype_meta, dtype_acc>(
      A, B, a_meta, b_meta, matrix_m, matrix_n, matrix_k, layout_a, layout_b, a_scaling, b_scaling, negative_axb);
  return check_and_log(C, gold_c.data(), matrix_m, matrix_n, tol);
}

constexpr uint32_t get_sparsity_ratio(sparsity_repr_t sparsity_repr) {
  switch (sparsity_repr) {
    case sparsity_repr_t::A4xB2: return 2;
    default: throw std::runtime_error("Unsupported sparsity representation.");
  }
}

uint32_t get_compressed_dst_size(sparsity_repr_t sparsity_repr) {
  switch (sparsity_repr) {
    case sparsity_repr_t::A4xB2: return 4;
    default: throw std::runtime_error("Unsupported sparsity representation.");
  }
}

uint32_t get_compressed_src_size(sparsity_repr_t sparsity_repr) {
  switch (sparsity_repr) {
    case sparsity_repr_t::A4xB2: return 2;
    default: throw std::runtime_error("Unsupported sparsity representation.");
  }
}

bool check_sparsity_repr_byte(sparsity_repr_t sparsity_repr, uint8_t byte) {
  uint32_t width_a = get_compressed_dst_size(sparsity_repr);
  uint32_t width_b = get_compressed_src_size(sparsity_repr);
  uint32_t rep_num = 8u / width_a;

  for (uint32_t i = 0; i < rep_num; ++i) {
    uint32_t valid_bits = 0;
    for (uint32_t j = 0; j < width_a; ++j) {
      valid_bits += byte & 1u;
      byte >>= 1u;
    }
    if (valid_bits != width_b) { return false; }
  }
  return true;
}

template <class dtype_a, class dtype_b, class dtype_meta, class blas_type = float>
void get_sparsity_gemm_gold(const size_t m, const size_t k, const size_t n, const mem_layout layout_a,
                            const mem_layout layout_b, dtype_a *A, dtype_b *B, blas_type *C, dtype_meta *meta,
                            sparsity_repr_t sparsity_repr, const blas_type alpha = 1.0,
                            const CBLAS_LAYOUT layout = CblasRowMajor) {
  uint32_t sparsity_ratio = get_sparsity_ratio(sparsity_repr);
  std::vector<dtype_b> dense_B(k * n * sparsity_ratio);

  // we only support A4xB2 sparsity
  assert(sparsity_repr == sparsity_repr_t::A4xB2);

  uint32_t k_elem_per_metabyte = 8u / sparsity_ratio;
  assert(k % k_elem_per_metabyte == 0);
  uint32_t k_slice_num = k / k_elem_per_metabyte;

  for (uint32_t i = 0; i < k_slice_num; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      uint32_t spars_B_k = i * k_elem_per_metabyte;
      uint32_t dense_B_k = i * 8u;
      uint32_t meta_k = i;
      uint8_t meta_byte = meta[meta_k * n + j]; // meta is n-major
      assert(check_sparsity_repr_byte(sparsity_repr, meta_byte));
      for (uint32_t k0 = 0, k1 = 0; k0 < 8; ++k0) {
        uint32_t dense_B_i = dense_B_k + k0;
        uint32_t spars_B_i = spars_B_k + k1;
        if ((meta_byte >> k0) & 1u) {
          if (layout_b == mem_layout::row_major) {
            dense_B[dense_B_i * n + j] = B[spars_B_i * n + j];
          } else if (layout_b == mem_layout::col_major) {
            dense_B[dense_B_i + j * k * sparsity_ratio] = B[spars_B_i + j * k];
          }
          k1++;
        } else {
          if (layout_b == mem_layout::row_major) {
            dense_B[dense_B_i * n + j] = 0;
          } else if (layout_b == mem_layout::col_major) {
            dense_B[dense_B_i + j * k * sparsity_ratio] = 0;
          }
        }
      }
    }
  }

  get_gemm_gold<dtype_a, dtype_b, blas_type>(m, k * sparsity_ratio, n, layout_a, layout_b, A, dense_B.data(), C, alpha,
                                             layout);
}

template <typename dtype_a, typename dtype_b, typename dtype_c, typename dtype_acc = float,
          typename dtype_meta = uint8_t>
int validate_sparsity_gemm_result(dtype_a *A, dtype_b *B, dtype_c *C, dtype_meta *meta, uint32_t matrix_m,
                                  uint32_t matrix_n, uint32_t matrix_k, sparsity_repr_t sparsity_repr,
                                  mem_layout layout_a = mem_layout::row_major,
                                  mem_layout layout_b = mem_layout::row_major, bool negative_axb = false,
                                  tolerance<dtype_c> tol = {}) {
  std::vector<dtype_acc> gold_acc(matrix_m * matrix_n, 0);
  dtype_acc alpha = negative_axb ? -1.0 : 1.0;
  get_sparsity_gemm_gold<dtype_a, dtype_b, dtype_meta, dtype_acc>(matrix_m, matrix_k, matrix_n, layout_a, layout_b, A,
                                                                  B, gold_acc.data(), meta, sparsity_repr, alpha);

  return check_and_log(C, gold_acc.data(), matrix_m, matrix_n, tol);
}

template <typename dtype_a, typename dtype_b, typename dtype_c, typename dtype_mxfp_meta,
          typename dtype_spars_meta = uint8_t, typename dtype_acc = float>
int validate_mxfp_sparsity_gemm_result(dtype_a *A, dtype_b *B, dtype_c *C, uint32_t matrix_m, uint32_t matrix_n,
                                       uint32_t matrix_k, sparsity_repr_t sparsity_repr, dtype_spars_meta *spars_meta,
                                       bool a_scaling, bool b_scaling, dtype_mxfp_meta *a_meta, dtype_mxfp_meta *b_meta,
                                       mem_layout layout_a = mem_layout::row_major,
                                       mem_layout layout_b = mem_layout::row_major, bool negative_axb = false,
                                       tolerance<dtype_c> tol = {}) {
  static constexpr uint32_t scale_ele_num = 32;
  uint32_t sparsity_ratio = get_sparsity_ratio(sparsity_repr);

  std::vector<dtype_acc> upcast_a(matrix_m * matrix_k * sparsity_ratio);
  std::vector<dtype_acc> upcast_b(matrix_n * matrix_k);

  upcast_mxfp_mat_a(A, a_meta, matrix_m, matrix_k * sparsity_ratio, layout_a, a_scaling, upcast_a.data(),
                    scale_ele_num);
  upcast_mxfp_mat_b(B, b_meta, matrix_k, matrix_n, layout_b, b_scaling, upcast_b.data(), scale_ele_num);

  std::vector<dtype_acc> gold_acc(matrix_m * matrix_n, 0);
  dtype_acc alpha = negative_axb ? -1.0 : 1.0;

  get_sparsity_gemm_gold<dtype_acc, dtype_acc, dtype_spars_meta, dtype_acc>(
      matrix_m, matrix_k, matrix_n, layout_a, layout_b, upcast_a.data(), upcast_b.data(), gold_acc.data(), spars_meta,
      sparsity_repr, alpha);

  return check_and_log(C, gold_acc.data(), matrix_m, matrix_n, tol);
}

template <uint32_t is_sparsity, uint32_t is_mxfp_a, uint32_t is_mxfp_b, typename dtype_a, typename dtype_b,
          typename dtype_c, typename dtype_mxfp_meta, typename dtype_spars_meta = uint8_t, typename dtype_acc = float>
int validate_mxfp_sparsity_gemm(dtype_a *A, dtype_b *B, dtype_c *C, uint32_t matrix_m, uint32_t matrix_n,
                                uint32_t matrix_k, dtype_mxfp_meta *a_meta, dtype_mxfp_meta *b_meta,
                                sparsity_repr_t spars_repr, dtype_spars_meta *spars_meta,
                                mem_layout layout_a = mem_layout::row_major,
                                mem_layout layout_b = mem_layout::row_major, bool negative_axb = false,
                                tolerance<dtype_c> tol = {}) {

  constexpr uint32_t is_mxfp = is_mxfp_a | is_mxfp_b;
  if constexpr (is_sparsity && !is_mxfp) {
    return validate_sparsity_gemm_result(A, B, C, spars_meta, matrix_m, matrix_n, matrix_k, spars_repr, layout_a,
                                         layout_b, negative_axb, tol);
  } else if constexpr (!is_sparsity && is_mxfp) {
    return validate_mxfp_gemm_result(A, B, C, matrix_m, matrix_n, matrix_k, is_mxfp_a, is_mxfp_b, a_meta, b_meta,
                                     layout_a, layout_b, negative_axb, tol);
  } else if constexpr (is_sparsity && is_mxfp) {
    return validate_mxfp_sparsity_gemm_result(A, B, C, matrix_m, matrix_n, matrix_k, spars_repr, spars_meta, is_mxfp_a,
                                              is_mxfp_b, a_meta, b_meta, layout_a, layout_b, negative_axb, tol);
  } else {
    return validate_gemm_result(A, B, C, matrix_m, matrix_n, matrix_k, layout_a, layout_b);
  }
}

template <typename dtype_src, typename dtype_dst, typename dtype_acc = float>
void softmax_golden(dtype_src *src, dtype_dst *dst, int m, int n) {
  std::vector<dtype_acc> softmax_acc(m * n, 0);
  std::vector<dtype_acc> softmax_max(m, 0);
  std::vector<dtype_acc> softmax_sum(m, 0);

  for (int i = 0; i < m; i++) {
    softmax_max[i] = src[i * n];
    for (int j = 1; j < n; j++) {
      softmax_max[i] = (src[i * n + j] > softmax_max[i]) ? dtype_acc(src[i * n + j]) : softmax_max[i];
    }
    //        std::cout << "row: " << i << " softmax_max value is: " <<
    //        softmax_max[i] << std::endl;
  }
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      softmax_acc[i * n + j] = sycl::exp(dtype_acc(src[i * n + j]) - softmax_max[i]);
    }
  }
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      softmax_sum[i] += softmax_acc[i * n + j];
    }
    //        std::cout << "row: " << i << " softmax_sum value is: " <<
    //        softmax_sum[i] << std::endl;
  }
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      softmax_acc[i * n + j] /= softmax_sum[i];
    }
  }
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      dst[i * n + j] = softmax_acc[i * n + j];
    }
  }
}

template <typename dtype_qkt, typename dtype_q, typename dtype_k, typename dtype_v, typename dtype_o,
          typename dtype_acc = float>
int validate_mha_fwd_result(dtype_q *Q_h, dtype_k *K_h, dtype_v *V_h, dtype_o *O_h, size_t seq_q, size_t seq_kv,
                            size_t head_num, size_t batch_num, size_t head_size, tolerance<dtype_o> tol = {}) {
  using matrix_o_t = std::vector<dtype_o>;
  using multi_head_matrix_o_t = std::vector<matrix_o_t>;
  using multi_batch_head_matrix_o_t = std::vector<multi_head_matrix_o_t>;
  multi_batch_head_matrix_o_t gold_o(batch_num, multi_head_matrix_o_t(head_num, matrix_o_t(seq_q * head_size)));

  mem_layout layout_q = mem_layout::row_major;
  mem_layout layout_k = mem_layout::col_major;
  mem_layout layout_qkt = mem_layout::row_major;
  mem_layout layout_v = mem_layout::row_major;

  uint32_t hidden_size = head_size * head_num;
  for (size_t b = 0; b < batch_num; b++) {
    for (size_t h = 0; h < head_num; h++) {
      std::vector<dtype_q> temp_q(seq_q * head_size);
      std::vector<dtype_k> temp_k(seq_kv * head_size);
      std::vector<dtype_v> temp_v(seq_kv * head_size);
      std::vector<dtype_acc> temp_qkt_acc(seq_q * seq_kv, 0);
      std::vector<dtype_acc> temp_o_acc(seq_q * head_size, 0);

      for (size_t i = 0; i < seq_q; i++) {
        auto itr_q_temp_start = temp_q.begin() + i * head_size;
        auto itr_q_start = Q_h + b * seq_q * hidden_size + i * hidden_size + h * head_size;
        std::copy(itr_q_start, itr_q_start + head_size, itr_q_temp_start);
      }
      for (size_t i = 0; i < seq_kv; i++) {
        auto itr_k_temp_start = temp_k.begin() + i * head_size;
        auto itr_k_start = K_h + b * seq_kv * hidden_size + i * hidden_size + h * head_size;
        std::copy(itr_k_start, itr_k_start + head_size, itr_k_temp_start);

        auto itr_v_temp_start = temp_v.begin() + i * head_size;
        auto itr_v_start = V_h + b * seq_kv * hidden_size + i * hidden_size + h * head_size;
        std::copy(itr_v_start, itr_v_start + head_size, itr_v_temp_start);
      }

      get_gemm_gold<dtype_q, dtype_k, dtype_acc>(seq_q, head_size, seq_kv, layout_q, layout_k, temp_q.data(),
                                                 temp_k.data(), temp_qkt_acc.data());
      std::vector<dtype_qkt> temp_qkt(temp_qkt_acc.data(), temp_qkt_acc.data() + seq_q * seq_kv);
      softmax_golden(temp_qkt.data(), temp_qkt.data(), seq_q, seq_kv);
      get_gemm_gold<dtype_qkt, dtype_v, dtype_acc>(seq_q, seq_kv, head_size, layout_qkt, layout_v, temp_qkt.data(),
                                                   temp_v.data(), temp_o_acc.data());

      std::vector<dtype_o> temp_o(temp_o_acc.data(), temp_o_acc.data() + seq_q * head_size);

      std::copy(temp_o.begin(), temp_o.end(), gold_o[b][h].begin());
    }
  }

  logger_t<dtype_o> logger(tol);
  for (size_t i = 0; i < batch_num * seq_q * hidden_size; i++) {
    size_t batch_idx = i / (seq_q * hidden_size);
    size_t head_idx = (i % hidden_size) / head_size;
    size_t idx_x = i % head_size;
    size_t idx_y = (i / hidden_size) % seq_q;
    dtype_o cpu = gold_o[batch_idx][head_idx][idx_y * head_size + idx_x];
    dtype_o gpu = O_h[i];
    logger.log(cpu, gpu, i);
  }
  logger.print_summary();
  return logger.get_err_cnt();
}

/*
template <typename T>
int compare_buffers(T *src_ptr, T *dst_ptr, const shape_t &src_shape, const
stride_t &src_stride, const shape_t &dst_shape, const stride_t &dst_stride,
const coord_t &src_coord, const coord_t &dst_coord, const shape_t &box_shape,
tolerance<T> tol = {}) { logger_t<T> logger(tol);

    for (uint32_t coord4_in_box = 0; coord4_in_box < box_shape.get(4);
coord4_in_box++) { uint32_t coord4_in_gmem = coord4_in_box + src_coord.get(4);
        for (uint32_t coord3_in_box = 0; coord3_in_box < box_shape.get(3);
coord3_in_box++) { uint32_t coord3_in_gmem = coord3_in_box + src_coord.get(3);
            for (uint32_t coord2_in_box = 0; coord2_in_box < box_shape.get(2);
coord2_in_box++) { uint32_t coord2_in_gmem = coord2_in_box + src_coord.get(2);
                for (uint32_t coord1_in_box = 0; coord1_in_box <
box_shape.get(1); coord1_in_box++) { for (uint32_t coord0_in_box = 0;
coord0_in_box < box_shape.get(0); coord0_in_box++) { uint32_t coord1_in_src =
coord1_in_box + src_coord.get(1); uint32_t coord0_in_src = coord0_in_box +
src_coord.get(0); uint32_t coord1_in_dst = coord1_in_box + dst_coord.get(1);
                        uint32_t coord0_in_dst = coord0_in_box +
dst_coord.get(0); coord_t src_coord_updated {coord0_in_src, coord1_in_src,
coord2_in_gmem, coord3_in_gmem, coord4_in_gmem}; coord_t dst_coord_updated
{coord0_in_dst, coord1_in_dst, coord2_in_gmem, coord3_in_gmem, coord4_in_gmem};
                        if (is_within_boundary(src_coord_updated, src_shape)
                                && is_within_boundary(dst_coord_updated,
dst_shape)) { auto src_offset = get_global_idx<T>(src_coord_updated,
src_stride); auto dst_offset = get_global_idx<T>(dst_coord_updated, dst_stride);

                            T src = src_ptr[src_offset];
                            T dst = dst_ptr[dst_offset];

                            logger.log(src, dst, coord1_in_box *
box_shape.get(0) + coord0_in_box);
                        }
                    }
                }
            }
        }
    }

    logger.print_summary();

    return logger.get_err_cnt();
}

template <typename T>
int compare_buffers(T *src_ptr, T *dst_ptr, const uint32_t src_elem, const
uint32_t dst_elem, tolerance<T> tol = {}) { logger_t<T> logger(tol);

    for (auto i = 0u; i < src_elem; i++) {
        logger.log(src_ptr[i], dst_ptr[i], i);
    }
    for (auto i = src_elem; i < dst_elem; i++) {
        logger.log(0, dst_ptr[i], i);
    }

    logger.print_summary();

    return logger.get_err_cnt();
}

template <typename T>
int compare_buffers(const std::vector<T> &reg, const T *dst_ori_ptr, const T
*dst_ptr, const uint32_t src_elem, const uint32_t dst_elem, const reduce_op op,
        tolerance<T> tol = {}) {
    logger_t<T> logger(tol);

    for (auto i = 0u; i < src_elem; i++) {
        auto gold = dst_ori_ptr[i];
        if (op == reduce_op::INC || op == reduce_op::DEC) {
            gold = (op == reduce_op::INC) ? gold + 1 : gold - 1;
            gold = (gold < 0) ? 0 : gold;
            gold = (gold > reg[i]) ? reg[i] : gold;
        } else if (op == reduce_op::IADD) {
            gold += reg[i];
        }
        logger.log(gold, dst_ptr[i], i);
    }
    for (auto i = src_elem; i < dst_elem; i++) {
        auto gold = dst_ori_ptr[i];
        if (op == reduce_op::INC || op == reduce_op::DEC) {
            gold = 0;
        } else if (op == reduce_op::IADD) {
        }
        logger.log(gold, dst_ptr[i], i);
    }

    logger.print_summary();

    return logger.get_err_cnt();
}
*/

namespace row_copy_validation {
template <slm_layout_t slm_layout, typename T>
uint32_t get_err_cnt_global2slm(const std::vector<T> &gmem, const std::vector<T> &slm,
                                const sycl::vec<uint32_t, 2> &gmem_size, const sycl::vec<uint64_t, 1> &gmem_stride,
                                const uint32_t width_2d, const T init_value, const uint32_t predicate_mask,
                                const bool is_8b_aligned = false) {
  assert(slm_layout == slm_layout_t::linear);

  using dtype = T;
  uint32_t data_size = sizeof(dtype);

  std::bitset<32> predicate_mask_bits(predicate_mask);
  uint32_t width_2d_elem = width_2d / data_size;

  uint32_t err_cnt = 0;
  for (uint32_t lane = 0, slm_idx = 0; lane < LANESIZE; lane++) {
    int32_t offset0 = is_8b_aligned ? 0 : lane;
    sycl::vec<int32_t, 2> gmem_coord(offset0, lane);
    uint32_t offset = row_copy::get_offset<dtype>(gmem_coord, gmem_size, gmem_stride);
    uint32_t copy_size = row_copy::get_copy_size<dtype>(gmem_coord, gmem_size, width_2d);
    uint32_t copy_elem = copy_size / data_size;
    for (uint32_t elem = 0; elem < width_2d_elem; elem++, slm_idx++) {
      uint32_t gmem_idx = offset / data_size + elem;
      dtype golden = predicate_mask_bits[lane] ? ((elem < copy_elem) ? gmem[gmem_idx] : dtype(0)) : init_value;
      if (slm[slm_idx] != golden) {
        err_cnt++;
        std::cout << " (global2slm) slm: " << slm[slm_idx] << " mismatch with golden: " << golden
                  << " at idx = " << slm_idx << std::endl;
      }
    }
  }

  return err_cnt;
}

template <slm_layout_t slm_layout, typename T>
uint32_t get_err_cnt_slm2global(const std::vector<T> &gmem_src, const std::vector<T> &gmem_dst,
                                const sycl::vec<uint32_t, 2> &gmem_size, const sycl::vec<uint64_t, 1> &gmem_stride,
                                const uint32_t width_2d, const T init_value, const uint32_t predicate_mask = UINT32_MAX,
                                const bool is_8b_aligned = false) {
  if (slm_layout == slm_layout_t::tiled) { assert(predicate_mask == UINT32_MAX); }

  using dtype = T;
  uint32_t data_size = sizeof(dtype);

  std::bitset<32> predicate_mask_bits(predicate_mask);

  uint32_t err_cnt = 0;
  for (uint32_t lane = 0; lane < LANESIZE; lane++) {
    int32_t offset0 = is_8b_aligned ? 0 : lane;
    sycl::vec<int32_t, 2> gmem_coord(offset0, lane);
    uint32_t offset = row_copy::get_offset<dtype>(gmem_coord, gmem_size, gmem_stride);
    uint32_t copy_size = row_copy::get_copy_size<dtype>(gmem_coord, gmem_size, width_2d);
    uint32_t copy_elem = copy_size / data_size;
    uint32_t idx_min = offset / data_size;
    uint32_t idx_max = idx_min + copy_elem;

    for (uint32_t elem = 0; elem < gmem_size[0]; elem++) {
      uint32_t gmem_idx = lane * gmem_size[0] + elem;
      bool is_valid_coord = (gmem_idx >= idx_min) && (gmem_idx < idx_max);
      dtype golden = predicate_mask_bits[lane] ? (is_valid_coord ? gmem_src[gmem_idx] : init_value) : init_value;

      if (gmem_dst[gmem_idx] != golden) {
        err_cnt++;
        std::cout << " (slm2global) gmem_dst: " << gmem_dst[gmem_idx] << " mismatch with golden: " << golden
                  << " at idx = " << gmem_idx << std::endl;
      }
    }
  }

  return err_cnt;
}

template <slm_layout_t slm_layout, typename T>
uint32_t get_err_cnt_global2slm(const std::vector<T> &gmem, const std::vector<T> &slm,
                                const sycl::vec<uint32_t, 2> &gmem_size, const sycl::vec<uint64_t, 1> &gmem_stride,
                                const uint32_t width_2d, const slm_matrix_type cm_type,
                                const bool is_8b_aligned = false) {
  assert(slm_layout == slm_layout_t::tiled);

  using dtype = T;
  uint32_t data_size = sizeof(dtype);

  uint32_t width_2d_elem = width_2d / data_size;
  uint32_t height_2d = LANESIZE;

  const sycl::vec<uint32_t, 2> cm_size = get_cm_size<dtype>(cm_type);
  assert(width_2d_elem % cm_size[0] == 0 && height_2d % cm_size[1] == 0);
  const uint32_t cm_bytes = cm_size[0] * cm_size[1] * data_size;
  const uint32_t cm_switch_stride = width_2d_elem * cm_size[1] * data_size;

  const uint32_t slm_bank_num = 4;
  const uint32_t slm_bank_size = 64;
  const uint32_t slm_width = slm_bank_num * slm_bank_size;
  const uint32_t mma_bank_num = 4;
  const uint32_t mma_bank_size = slm_width / mma_bank_num;
  const uint32_t slm_bank_swizzling_size = slm_bank_size / 2;
  const uint32_t cm_split_x = (cm_type == slm_matrix_type::type2) ? mma_bank_num : 1;
  const uint32_t cm_split_y = (cm_type == slm_matrix_type::type1) ? mma_bank_num : 1;
  sycl::vec<uint32_t, 2> sub_cm_shape = {cm_size[0] / cm_split_x, cm_size[1] / cm_split_y};

  uint32_t err_cnt = 0;
  for (uint32_t lane = 0; lane < LANESIZE; lane++) {
    int32_t offset0 = is_8b_aligned ? 0 : lane;
    sycl::vec<int32_t, 2> gmem_coord(offset0, lane);
    uint32_t offset = row_copy::get_offset<dtype>(gmem_coord, gmem_size, gmem_stride);
    uint32_t copy_size = row_copy::get_copy_size<dtype>(gmem_coord, gmem_size, width_2d);
    uint32_t offset_elem = offset / data_size;
    uint32_t copy_elem = copy_size / data_size;

    for (uint32_t elem = 0; elem < width_2d_elem; elem++) {
      uint32_t cm_grid_y = lane / cm_size[1];
      uint32_t cm_idx_y = lane % cm_size[1];

      uint32_t cm_grid_x = elem / cm_size[0];
      uint32_t cm_idx_x = elem % cm_size[0];

      const bool need_swizzling = cm_grid_x & 1u;
      uint32_t cm_smem_offset = cm_grid_y * cm_switch_stride + cm_grid_x * cm_bytes;
      uint32_t cm_idx_offset = 0;
      if (cm_type == slm_matrix_type::type2) {
        uint32_t bank_offset = cm_idx_x / sub_cm_shape[0] * mma_bank_size;
        uint32_t sub_cm_idx_x = cm_idx_x % sub_cm_shape[0];
        uint32_t sub_cm_idx_linear_offset = (cm_idx_y * sub_cm_shape[0] + sub_cm_idx_x) * data_size;
        uint32_t sub_cm_idx_bank_offset = sub_cm_idx_linear_offset % mma_bank_size;
        uint32_t sub_cm_idx_bank_start_addr = sub_cm_idx_linear_offset / mma_bank_size * slm_width;
        cm_idx_offset = sub_cm_idx_bank_start_addr + sub_cm_idx_bank_offset + bank_offset;
      } else if (cm_type == slm_matrix_type::type1) {
        uint32_t bank_offset = cm_idx_y / sub_cm_shape[1] * mma_bank_size;
        uint32_t sub_cm_idx_y = cm_idx_y % sub_cm_shape[1];
        uint32_t sub_cm_idx_linear_offset = (sub_cm_idx_y * cm_size[0] + cm_idx_x) * data_size;
        uint32_t sub_cm_idx_bank_offset = sub_cm_idx_linear_offset % mma_bank_size;
        uint32_t sub_cm_idx_bank_start_addr = sub_cm_idx_linear_offset / mma_bank_size * slm_width;
        cm_idx_offset = sub_cm_idx_bank_start_addr + sub_cm_idx_bank_offset + bank_offset;
      }
      // half slm_bank_size swizzling
      uint32_t cm_idx_bank_offset = cm_idx_offset % slm_bank_size;
      uint32_t cm_idx_bank_start_addr = cm_idx_offset / slm_bank_size * slm_bank_size;
      uint32_t cm_idx_smem_offset = need_swizzling
          ? (cm_idx_bank_start_addr + (cm_idx_bank_offset + slm_bank_swizzling_size) % slm_bank_size)
          : cm_idx_offset;
      auto slm_idx = (cm_smem_offset + cm_idx_smem_offset) / data_size;

      auto idx = offset_elem + elem;
      dtype golden = elem < copy_elem ? gmem[idx] : dtype(0);
      if (slm[slm_idx] != golden) {
        err_cnt++;
        std::cout << " (global2slm_tiled) slm: " << slm[slm_idx] << " mismatch with golden: " << golden
                  << " at idx = " << idx << std::endl;
      }
    }
  }

  return err_cnt;
}
} // namespace row_copy_validation

template <class dtype>
std::vector<dtype> im2col(dtype *base_ptr, const sycl::vec<uint32_t, 4> &gmem_shape,
                          const sycl::vec<int32_t, 4> &gmem_coord, const sycl::vec<uint32_t, 4> &box_shape) {
  uint32_t box_size = box_shape[0] * box_shape[1] * box_shape[2] * box_shape[3];
  std::vector<dtype> ret(box_size);

  auto coord_c = gmem_coord[0];
  auto coord_w = gmem_coord[1];
  auto coord_h = gmem_coord[2];
  auto coord_n = gmem_coord[3];
  assert(box_shape[3] == 1); // only support batch == 1 for now

  auto gmem_stride = get_stride_from_shape<dtype, 4>(gmem_shape);

  for (uint32_t h = 0; h < box_shape[2]; h++) {
    if (coord_h + h < 0 || coord_h + h >= gmem_shape[2]) { continue; }
    for (uint32_t w = 0; w < box_shape[1]; w++) {
      if (coord_w + w < 0 || coord_w + w >= gmem_shape[1]) { continue; }
      sycl::vec<int32_t, 4> coord {coord_c, coord_w + w, coord_h + h, coord_n};
      auto src_offset = get_offset<dtype, 4>(coord, gmem_stride);
      auto src_idx = src_offset / sizeof(dtype);

      auto dst_offset = (h * box_shape[1] + w) * gmem_shape[0];
      std::copy(base_ptr + src_idx, base_ptr + src_idx + gmem_shape[0], ret.begin() + dst_offset);
    }
  }

  return ret;
}
