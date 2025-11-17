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
template <> struct fixed_s<16>  { static constexpr fixstr::fixed_string value {"16"}; };
template <> struct fixed_s<64>  { static constexpr fixstr::fixed_string value {"64"}; };
template <> struct fixed_s<128> { static constexpr fixstr::fixed_string value {"128"};};
template <> struct fixed_s<256> { static constexpr fixstr::fixed_string value {"256"};};

// Type name enumeration
template <typename> struct fixed_type;
template <> struct fixed_type<float> { static constexpr fixstr::fixed_string value {"F32"};};
template <> struct fixed_type<sycl::half> { static constexpr fixstr::fixed_string value {"F16"};};
template <> struct fixed_type<sycl::ext::oneapi::bfloat16> { static constexpr fixstr::fixed_string value {"BF16"};};

// Major enumeration
template <AMMA::Major> struct ammajor;

template <> struct ammajor<AMMA::Major::MN> {
  static constexpr fixstr::fixed_string value {".am"};
};

template <> struct ammajor<AMMA::Major::K> {
  static constexpr fixstr::fixed_string value {""};
};

template <cute::AMMA::Major> struct bkmajor;
template <> struct bkmajor<AMMA::Major::MN> {
  static constexpr fixstr::fixed_string value {""};
};

template <> struct bkmajor<AMMA::Major::K> {
  static constexpr fixstr::fixed_string value {".bk"};
};

template <typename> struct rd_type;
template <> struct rd_type<sycl::half> {static constexpr fixstr::fixed_string value {".16b.fp"};};
template <> struct rd_type<sycl::ext::oneapi::bfloat16> {
  static constexpr fixstr::fixed_string value {".16b.fp"};
};

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

template <int N> constexpr auto _s = fixed_s<N>::value;
template <typename T> constexpr auto _t = fixed_type<T>::value;
template <typename T> constexpr auto _at = rd_type<T>::value;
template <cute::AMMA::Major major> constexpr auto _am = ammajor<major>::value;
template <cute::AMMA::Major major> constexpr auto _bk = bkmajor<major>::value;
template <cute::detail::CacheCtrl CC> constexpr auto _cc = cachectrl<CC>::value;
template <cute::detail::FillMethod FM> constexpr auto _fl = padfill<FM>::value;

}
