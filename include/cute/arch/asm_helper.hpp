#pragma once
#include <cute/arch/fixed_string.hpp>

namespace cute {

template <int N> struct fixed_s;

// Static number enumeration
template <> struct fixed_s<1>   { static constexpr fixstr::fixed_string value {"1"};  };
template <> struct fixed_s<2>   { static constexpr fixstr::fixed_string value {"2"};  };
template <> struct fixed_s<3>   { static constexpr fixstr::fixed_string value {"3"};  };
template <> struct fixed_s<4>   { static constexpr fixstr::fixed_string value {"4"};  };
template <> struct fixed_s<5>   { static constexpr fixstr::fixed_string value {"5"};  };
template <> struct fixed_s<8>   { static constexpr fixstr::fixed_string value {"8"};  };
template <> struct fixed_s<16>  { static constexpr fixstr::fixed_string value {"16"}; };
template <> struct fixed_s<32>  { static constexpr fixstr::fixed_string value {"32"}; };
template <> struct fixed_s<64>  { static constexpr fixstr::fixed_string value {"64"}; };
template <> struct fixed_s<128> { static constexpr fixstr::fixed_string value {"128"};};
template <> struct fixed_s<256> { static constexpr fixstr::fixed_string value {"256"};};
template <> struct fixed_s<512> { static constexpr fixstr::fixed_string value {"512"};};

// Type name enumeration
template <typename> struct fixed_type;
template <> struct fixed_type<float> { static constexpr fixstr::fixed_string value {"F32"};};
template <> struct fixed_type<sycl::half> { static constexpr fixstr::fixed_string value {"F16"};};
template <> struct fixed_type<sycl::ext::oneapi::bfloat16> { static constexpr fixstr::fixed_string value {"BF16"};};

template <typename> struct pisa_type;
template <> struct pisa_type<float> { static constexpr fixstr::fixed_string value {".f32"};};
template <> struct pisa_type<fp16> { static constexpr fixstr::fixed_string value {".f16"};};
template <> struct pisa_type<cutlass::bfloat16_t> { static constexpr fixstr::fixed_string value {".bf16"};};

// For TensorPipe
enum class tred_red_dim {
  none,
  rednd,
  redmd
};

template <tred_red_dim> struct tensor_red_type;
template <> struct tensor_red_type<tred_red_dim::none> { static constexpr fixstr::fixed_string value {""};};
template <> struct tensor_red_type<tred_red_dim::rednd> { static constexpr fixstr::fixed_string value {".rednd"};};
template <> struct tensor_red_type<tred_red_dim::redmd> { static constexpr fixstr::fixed_string value {".redmd"};};


enum class tred_algo {
  amax,
  amin,
  max,
  min,
  f32add
};

template <tred_algo> struct tensor_red_algo;
template <> struct tensor_red_algo<tred_algo::amax> { static constexpr fixstr::fixed_string value {".amax"};};
template <> struct tensor_red_algo<tred_algo::amin> { static constexpr fixstr::fixed_string value {".amin"};};
template <> struct tensor_red_algo<tred_algo::max> { static constexpr fixstr::fixed_string value {".max"};};
template <> struct tensor_red_algo<tred_algo::min> { static constexpr fixstr::fixed_string value {".min"};};
template <> struct tensor_red_algo<tred_algo::f32add> { static constexpr fixstr::fixed_string value {".f32add"};};


enum class tred_round_mode {
  none,
  mode_re,
  mode_ru,
  mode_rd,
  mode_rz,
  mode_rna
};

template <tred_round_mode> struct tensor_red_round_type;
template <> struct tensor_red_round_type<tred_round_mode::none> { static constexpr fixstr::fixed_string value {""};};
template <> struct tensor_red_round_type<tred_round_mode::mode_re> { static constexpr fixstr::fixed_string value {".re"};};
template <> struct tensor_red_round_type<tred_round_mode::mode_ru> { static constexpr fixstr::fixed_string value {".ru"};};
template <> struct tensor_red_round_type<tred_round_mode::mode_rd> { static constexpr fixstr::fixed_string value {".rd"};};
template <> struct tensor_red_round_type<tred_round_mode::mode_rz> { static constexpr fixstr::fixed_string value {".rz"};};
template <> struct tensor_red_round_type<tred_round_mode::mode_rna> { static constexpr fixstr::fixed_string value {".rna"};};

template <bool saturation> struct tensor_red_dsat;
template <> struct tensor_red_dsat <true> { static constexpr fixstr::fixed_string value {".dsat"};};
template <> struct tensor_red_dsat <false> { static constexpr fixstr::fixed_string value {""};};

template <bool acc> struct tensor_red_acc;
template <> struct tensor_red_acc<true> { static constexpr fixstr::fixed_string value {".acc"};};
template <> struct tensor_red_acc<false> { static constexpr fixstr::fixed_string value {""};};

template <bool mxnd> struct tensor_exp_mxnd;
template <> struct tensor_exp_mxnd<true> { static constexpr fixstr::fixed_string value {".mxnd"};};
template <> struct tensor_exp_mxnd<false> { static constexpr fixstr::fixed_string value {""};};

template <bool mxnd> struct tensor_exp_xch;
template <> struct tensor_exp_xch<true> { static constexpr fixstr::fixed_string value {".xch"};};
template <> struct tensor_exp_xch<false> { static constexpr fixstr::fixed_string value {""};};


template <typename> struct rd_type;
template <> struct rd_type<sycl::half> {static constexpr fixstr::fixed_string value {".16b.fp"};};
template <> struct rd_type<cutlass::half_t> {static constexpr fixstr::fixed_string value {".16b.fp"};};
template <> struct rd_type<sycl::ext::oneapi::bfloat16> {
  static constexpr fixstr::fixed_string value {".16b.fp"};
};

template <int N> constexpr auto _s = fixed_s<N>::value;
template <typename T> constexpr auto _t = fixed_type<T>::value;
template <typename T> constexpr auto _p = pisa_type<T>::value;
template <typename T> constexpr auto _at = rd_type<T>::value;
template <tred_red_dim red_dim> constexpr auto _rdim = tensor_red_type<red_dim>::value;
template <tred_algo algo> constexpr auto _ral = tensor_red_algo<algo>::value;
template <tred_round_mode mode> constexpr auto _rmo = tensor_red_round_type<mode>::value;
template <bool dsat> constexpr auto _sat = tensor_red_dsat<dsat>::value;
template <bool acc> constexpr auto _acc = tensor_red_acc<acc>::value;
template <bool mxnd> constexpr auto _mxnd = tensor_exp_mxnd<mxnd>::value;
template <bool xch> constexpr auto _xch = tensor_exp_xch<xch>::value;

}
