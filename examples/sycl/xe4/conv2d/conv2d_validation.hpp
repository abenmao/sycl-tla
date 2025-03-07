#pragma once

#include <cassert>
#include <stdexcept>
#include <vector>
#include <oneapi/dnnl/dnnl.hpp>

#include "validation.hpp"

inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();

  if (!handle) throw std::runtime_error("handle is nullptr.");

  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
    if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
    for (size_t i = 0; i < size; ++i)
      dst[i] = ((uint8_t *)handle)[i];
    return;
  }

  assert(!"not expected");
}

inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();

  if (!handle) throw std::runtime_error("handle is nullptr.");

  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
    if (!src) throw std::runtime_error("get_data_handle returned nullptr.");
    for (size_t i = 0; i < size; ++i)
      ((uint8_t *)handle)[i] = src[i];
    return;
  }

  assert(!"not expected");
}

template <class dtype_activate, class dtype_filter, class dtype_acc = float>
void get_conv2d_gold_by_onednn(dtype_activate *tensor_x, dtype_filter *tensor_w, dtype_acc *tensor_y,
                               const conv2d::problem_shape_t &problem_shape) {
  // first get the conv2d parameters from problem_shape
  auto N = problem_shape.get_in_batch();
  auto H = problem_shape.get_in_height();
  auto W = problem_shape.get_in_width();
  auto C = problem_shape.get_in_channel();
  auto K = problem_shape.get_kernel_num();
  auto R = problem_shape.get_kernel_height();
  auto S = problem_shape.get_kernel_width();
  auto P = problem_shape.get_out_height();
  auto Q = problem_shape.get_out_width();
  auto padding_top = problem_shape.get_padding_top();
  auto padding_bottom = problem_shape.get_padding_bottom();
  auto padding_left = problem_shape.get_padding_left();
  auto padding_right = problem_shape.get_padding_right();
  auto stride_h = problem_shape.get_stride_h();
  auto stride_w = problem_shape.get_stride_w();
  auto dilation_h = problem_shape.get_dilation_h();
  auto dilation_w = problem_shape.get_dilation_w();

  // for non-dilation conv, oneDNN define this value to be zero instead of 1
  dilation_h = dilation_h - 1;
  dilation_w = dilation_w - 1;

  // Initialize oneDNN engine and stream
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream s(eng);

  // define tensor dimensions
  dnnl::memory::dims src_dims = {N, C, H, W};
  dnnl::memory::dims weights_dims = {K, C, R, S};
  dnnl::memory::dims dst_dims = {N, K, P, Q};
  // dnnl::memory::dims strides = {problem_size.stride_h,
  // problem_size.stride_w};
  dnnl::memory::dims strides = {stride_h, stride_w};
  dnnl::memory::dims dilation = {dilation_h, dilation_w};
  dnnl::memory::dims padding_l = {padding_top, padding_left};
  dnnl::memory::dims padding_r = {padding_bottom, padding_right};

  // Allocate buffers and initialize
  std::vector<dtype_acc> src_onednn(tensor_x, tensor_x + N * H * W * C);
  std::vector<dtype_acc> weight_onednn(tensor_w, tensor_w + K * R * S * C);

  // create memory objects
  dnnl::memory::desc src_md, weight_md, dst_md;
  if constexpr (std::is_same_v<dtype_acc, float>) {
    src_md = dnnl::memory::desc({src_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nhwc);
    weight_md = dnnl::memory::desc({weights_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ohwi);
    dst_md = dnnl::memory::desc({dst_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nhwc);
  } else if constexpr (std::is_same_v<dtype_acc, int32_t>) {
    src_md = dnnl::memory::desc({src_dims}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::nhwc);
    weight_md = dnnl::memory::desc({weights_dims}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::ohwi);
    dst_md = dnnl::memory::desc({dst_dims}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::nhwc);
  } else {
    throw std::runtime_error("OneDNN only supports f32 or s32 for accumulator.");
  }

  auto src_mem = dnnl::memory(src_md, eng);
  write_to_dnnl_memory(src_onednn.data(), src_mem);
  auto weight_mem = dnnl::memory(weight_md, eng);
  write_to_dnnl_memory(weight_onednn.data(), weight_mem);
  auto dst_mem = dnnl::memory(dst_md, eng);

  // create convolution primitive descriptor
  auto conv_pd = dnnl::convolution_forward::primitive_desc(eng, dnnl::prop_kind::forward_inference,
                                                           dnnl::algorithm::convolution_direct, src_md, weight_md,
                                                           dst_md, strides, dilation, padding_l, padding_r);

  std::unordered_map<int, dnnl::memory> net_args = {
      {DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, weight_mem}, {DNNL_ARG_DST, dst_mem}};
  dnnl::convolution_forward(conv_pd).execute(s, net_args);

  s.wait();

  read_from_dnnl_memory(tensor_y, dst_mem);
}

// use onednn to compute the groundtruth of conv2d
template <typename dtype_activate, typename dtype_filter, typename dtype_output, typename dtype_acc = float>
int validate_conv2d_result_by_onednn(dtype_activate *tensor_x, dtype_filter *tensor_w, dtype_output *tensor_y,
                                     const conv2d::problem_shape_t &problem_shape, tolerance<dtype_output> tol = {}) {
  // get conv2d parameters from problem_shape
  auto N = problem_shape.get_in_batch();
  auto K = problem_shape.get_kernel_num();
  auto P = problem_shape.get_out_height(); // cutlass definition P, Q means
      // the output height and width
  auto Q = problem_shape.get_out_width();

  std::vector<dtype_acc> gold_acc(N * P * Q * K, 0.0);
  get_conv2d_gold_by_onednn<dtype_activate, dtype_filter, dtype_acc>(tensor_x, tensor_w, gold_acc.data(),
                                                                     problem_shape);

  std::vector<dtype_output> gold_c(gold_acc.data(), gold_acc.data() + N * P * Q * K);
  logger_t<dtype_output> logger(tol);
  for (uint32_t i = 0; i < N * P * Q * K; i++) {
    dtype_output cpu = gold_c[i];
    dtype_output gpu = tensor_y[i];
    logger.log(cpu, gpu, i);
  }
  logger.print_summary();
  return logger.get_err_cnt();
}

template <class dtype_activate, class dtype_filter, class dtype_acc = float>
void get_conv2d_backprop_data_gold_by_onednn(dtype_activate *tensor_x, dtype_filter *tensor_w,
                                             dtype_acc *tensor_diff_data,
                                             const conv2d::problem_shape_t &problem_shape) {
  // first get the conv2d parameters from problem_shape
  auto N = problem_shape.get_in_batch();
  auto H = problem_shape.get_in_height();
  auto W = problem_shape.get_in_width();
  auto C = problem_shape.get_in_channel();
  auto K = problem_shape.get_kernel_num();
  auto R = problem_shape.get_kernel_height();
  auto S = problem_shape.get_kernel_width();
  auto P = problem_shape.get_out_height();
  auto Q = problem_shape.get_out_width();
  auto padding_top = problem_shape.get_padding_top();
  auto padding_bottom = problem_shape.get_padding_bottom();
  auto padding_left = problem_shape.get_padding_left();
  auto padding_right = problem_shape.get_padding_right();
  auto stride_h = problem_shape.get_stride_h();
  auto stride_w = problem_shape.get_stride_w();
  auto dilation_h = problem_shape.get_dilation_h();
  auto dilation_w = problem_shape.get_dilation_w();

  // for non-dilation conv, oneDNN define this value to be zero instead of 1
  dilation_h = dilation_h - 1;
  dilation_w = dilation_w - 1;

  // Initialize oneDNN engine and stream
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream s(eng);

  // define tensor dimensions
  dnnl::memory::dims src_dims = {N, C, H, W};
  dnnl::memory::dims weights_dims = {K, C, R, S};
  dnnl::memory::dims dst_dims = {N, K, P, Q};
  dnnl::memory::dims strides = {stride_h, stride_w};
  dnnl::memory::dims dilation = {dilation_h, dilation_w};
  dnnl::memory::dims padding_l = {padding_top, padding_left};
  dnnl::memory::dims padding_r = {padding_bottom, padding_right};

  // Allocate buffers and initialize
  std::vector<dtype_acc> src_onednn(tensor_x, tensor_x + N * P * Q * K);
  std::vector<dtype_acc> weight_onednn(tensor_w, tensor_w + K * R * S * C);

  // create memory objects
  dnnl::memory::desc src_md, weight_md, dst_md;
  if constexpr (std::is_same_v<dtype_acc, float>) {
    src_md = dnnl::memory::desc({src_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nhwc);
    weight_md = dnnl::memory::desc({weights_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ohwi);
    dst_md = dnnl::memory::desc({dst_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nhwc);
  } else if constexpr (std::is_same_v<dtype_acc, int32_t>) {
    src_md = dnnl::memory::desc({src_dims}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::nhwc);
    weight_md = dnnl::memory::desc({weights_dims}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::ohwi);
    dst_md = dnnl::memory::desc({dst_dims}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::nhwc);
  } else {
    throw std::runtime_error("OneDNN only supports f32 or s32 for accumulator.");
  }

  // create convolution primitive descriptor
  auto conv_pd = dnnl::convolution_forward::primitive_desc(eng, dnnl::prop_kind::forward_inference,
                                                           dnnl::algorithm::convolution_direct, src_md, weight_md,
                                                           dst_md, strides, dilation, padding_l, padding_r);

  dnnl::memory::desc diff_dst_md, diff_src_md;
  if constexpr (std::is_same_v<dtype_acc, float>) {
    diff_src_md = dnnl::memory::desc({src_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nhwc);
    diff_dst_md = dnnl::memory::desc({dst_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nhwc);
  } else if constexpr (std::is_same_v<dtype_acc, int32_t>) {
    diff_src_md = dnnl::memory::desc({src_dims}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::nhwc);
    diff_dst_md = dnnl::memory::desc({dst_dims}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::nhwc);
  } else {
    throw std::runtime_error("OneDNN only supports f32 or s32 for accumulator.");
  }

  auto diff_dst_mem = dnnl::memory(diff_dst_md, eng);
  write_to_dnnl_memory(src_onednn.data(), diff_dst_mem);
  auto weight_mem = dnnl::memory(weight_md, eng);
  write_to_dnnl_memory(weight_onednn.data(), weight_mem);
  auto diff_src_mem = dnnl::memory(diff_src_md, eng);

  // create the backward weights primitive descriptor
  auto conv_bwd_data_pd =
      dnnl::convolution_backward_data::primitive_desc(eng, dnnl::algorithm::convolution_direct, src_md, weight_md,
                                                      diff_dst_md, strides, dilation, padding_l, padding_r, conv_pd);
  std::unordered_map<int, dnnl::memory> conv_bwd_data_net_args = {
      {DNNL_ARG_WEIGHTS, weight_mem}, {DNNL_ARG_DIFF_DST, diff_dst_mem}, {DNNL_ARG_DIFF_SRC, diff_src_mem}};
  dnnl::convolution_backward_data(conv_bwd_data_pd).execute(s, conv_bwd_data_net_args);

  s.wait();
  read_from_dnnl_memory(tensor_diff_data, diff_src_mem);
}

template <typename dtype_activate, typename dtype_filter, typename dtype_output, typename dtype_acc = float>
int validate_conv2d_result_by_dgrad_onednn(dtype_activate *tensor_x, dtype_filter *tensor_w, dtype_output *tensor_y,
                                           const conv2d::problem_shape_t &problem_shape,
                                           tolerance<dtype_output> tol = {}) {
  // get conv2d parameters from problem_shape
  auto N = problem_shape.get_in_batch();
  auto C = problem_shape.get_in_channel();
  auto H = problem_shape.get_in_height();
  auto W = problem_shape.get_in_width();

  std::vector<dtype_acc> gold_acc(N * H * W * C, 0.0);
  get_conv2d_backprop_data_gold_by_onednn<dtype_activate, dtype_filter, dtype_acc>(tensor_x, tensor_w, gold_acc.data(),
                                                                                   problem_shape);

  std::vector<dtype_output> gold_c(gold_acc.data(), gold_acc.data() + N * H * W * C);
  logger_t<dtype_output> logger(tol);
  for (uint32_t i = 0; i < N * H * W * C; i++) {
    dtype_output cpu = gold_c[i];
    dtype_output gpu = tensor_y[i];
    logger.log(cpu, gpu, i);
  }
  logger.print_summary();
  return logger.get_err_cnt();
}

template <class dtype_activate, class dtype_filter, class dtype_acc = float>
void get_conv2d_backprop_weight_gold_by_onednn(dtype_activate *tensor_x, dtype_filter *tensor_w,
                                               dtype_acc *tensor_diff_weight,
                                               const conv2d::problem_shape_t &problem_shape) {
  // first get the conv2d parameters from problem_shape
  auto N = problem_shape.get_in_batch();
  auto H = problem_shape.get_in_height();
  auto W = problem_shape.get_in_width();
  auto C = problem_shape.get_in_channel();
  auto K = problem_shape.get_kernel_num();
  auto R = problem_shape.get_kernel_height();
  auto S = problem_shape.get_kernel_width();
  auto P = problem_shape.get_out_height();
  auto Q = problem_shape.get_out_width();
  auto padding_top = problem_shape.get_padding_top();
  auto padding_bottom = problem_shape.get_padding_bottom();
  auto padding_left = problem_shape.get_padding_left();
  auto padding_right = problem_shape.get_padding_right();
  auto stride_h = problem_shape.get_stride_h();
  auto stride_w = problem_shape.get_stride_w();
  auto dilation_h = problem_shape.get_dilation_h();
  auto dilation_w = problem_shape.get_dilation_w();

  // for non-dilation conv, oneDNN define this value to be zero instead of 1
  dilation_h = dilation_h - 1;
  dilation_w = dilation_w - 1;

  // Initialize oneDNN engine and stream
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
  dnnl::stream s(eng);

  // define tensor dimensions
  dnnl::memory::dims src_dims = {N, C, H, W};
  dnnl::memory::dims weights_dims = {K, C, R, S};
  dnnl::memory::dims dst_dims = {N, K, P, Q};
  dnnl::memory::dims strides = {stride_h, stride_w};
  dnnl::memory::dims dilation = {dilation_h, dilation_w};
  dnnl::memory::dims padding_l = {padding_top, padding_left};
  dnnl::memory::dims padding_r = {padding_bottom, padding_right};

  // Allocate buffers and initialize
  std::vector<dtype_acc> src_onednn(tensor_w, tensor_w + N * H * W * C);
  std::vector<dtype_acc> weight_onednn(tensor_x, tensor_x + N * P * Q * K);

  // create memory objects
  dnnl::memory::desc src_md, weight_md, dst_md;
  if constexpr (std::is_same_v<dtype_acc, float>) {
    src_md = dnnl::memory::desc({src_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nhwc);
    weight_md = dnnl::memory::desc({weights_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ohwi);
    dst_md = dnnl::memory::desc({dst_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nhwc);
  } else if constexpr (std::is_same_v<dtype_acc, int32_t>) {
    src_md = dnnl::memory::desc({src_dims}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::nhwc);
    weight_md = dnnl::memory::desc({weights_dims}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::ohwi);
    dst_md = dnnl::memory::desc({dst_dims}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::nhwc);
  } else {
    throw std::runtime_error("OneDNN only supports f32 or s32 for accumulator.");
  }

  // create convolution primitive descriptor
  auto conv_pd = dnnl::convolution_forward::primitive_desc(eng, dnnl::prop_kind::forward_inference,
                                                           dnnl::algorithm::convolution_direct, src_md, weight_md,
                                                           dst_md, strides, dilation, padding_l, padding_r);

  dnnl::memory::desc diff_dst_md, diff_weights_md;
  if constexpr (std::is_same_v<dtype_acc, float>) {
    diff_dst_md = dnnl::memory::desc({dst_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nhwc);
    diff_weights_md = dnnl::memory::desc({weights_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ohwi);
  } else if constexpr (std::is_same_v<dtype_acc, int32_t>) {
    diff_dst_md = dnnl::memory::desc({dst_dims}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::nhwc);
    diff_weights_md = dnnl::memory::desc({weights_dims}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::ohwi);
  } else {
    throw std::runtime_error("OneDNN only supports f32 or s32 for accumulator.");
  }

  auto src_mem = dnnl::memory(src_md, eng);
  write_to_dnnl_memory(src_onednn.data(), src_mem);
  auto diff_dst_mem = dnnl::memory(diff_dst_md, eng);
  write_to_dnnl_memory(weight_onednn.data(), diff_dst_mem);
  auto diff_weights_mem = dnnl::memory(diff_weights_md, eng);

  // create the backward weights primitive descriptor
  auto conv_bwd_weights_pd =
      dnnl::convolution_backward_weights::primitive_desc(eng, dnnl::algorithm::convolution_direct, src_md, weight_md,
                                                         diff_dst_md, strides, dilation, padding_l, padding_r, conv_pd);

  std::unordered_map<int, dnnl::memory> conv_bwd_weight_net_args = {
      {DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DIFF_DST, diff_dst_mem}, {DNNL_ARG_DIFF_WEIGHTS, diff_weights_mem}};
  dnnl::convolution_backward_weights(conv_bwd_weights_pd).execute(s, conv_bwd_weight_net_args);

  s.wait();
  read_from_dnnl_memory(tensor_diff_weight, diff_weights_mem);
}

template <typename dtype_activate, typename dtype_filter, typename dtype_output, typename dtype_acc = float>
int validate_conv2d_result_by_wgrad_onednn(dtype_activate *tensor_x, dtype_filter *tensor_w, dtype_output *tensor_y,
                                           const conv2d::problem_shape_t &problem_shape,
                                           tolerance<dtype_output> tol = {}) {
  // get conv2d parameters from problem_shape
  auto K = problem_shape.get_kernel_num();
  auto R = problem_shape.get_kernel_height();
  auto S = problem_shape.get_kernel_width();
  auto C = problem_shape.get_in_channel();

  std::vector<dtype_acc> gold_acc(K * R * S * C, 0.0);
  get_conv2d_backprop_weight_gold_by_onednn<dtype_activate, dtype_filter, dtype_acc>(tensor_x, tensor_w,
                                                                                     gold_acc.data(), problem_shape);

  std::vector<dtype_output> gold_c(gold_acc.data(), gold_acc.data() + K * R * S * C);
  logger_t<dtype_output> logger(tol);
  for (uint32_t i = 0; i < K * R * S * C; i++) {
    dtype_output cpu = gold_c[i];
    dtype_output gpu = tensor_y[i];
    logger.log(cpu, gpu, i);
  }
  logger.print_summary();
  return logger.get_err_cnt();
}
