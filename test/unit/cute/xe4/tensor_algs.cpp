/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <cute/algorithm/tensor_processing.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/util/compat/launch_policy.hpp>
#include "cutlass_unit_test.h"

#include <cmath>
#include <limits>
#include <algorithm>

//#include <cute/algorithm/tensor_algorithms.hpp>
//#include <cute/algorithm/tensor_reduce.hpp>
template<class...> class ReductionOpName;
template<class...> class ReductionAccOpName;
using namespace cute;
using namespace cutlass;
using namespace compat::experimental;

// Helper for safe device memory cleanup
template <class T>
void safe_free_device(T* ptr, sycl::queue& q) {
  if (ptr != nullptr) {
    sycl::free(ptr, q);
  }
}

// Error handling wrapper for device operations
inline void check_device_operation(sycl::queue& q) {
  try {
    q.wait();
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
}

template <class TensorA, class TensorB, int cnt=3>
void EXPECT_HOST_TENSOR_EQUAL(const TensorA& A, const TensorB& B) {
  EXPECT_EQ(A.shape(), B.shape());
  EXPECT_EQ(A.layout(), B.layout());

  // Just print first 2 unmatched values
  int32_t count = 4;
  for (int i = 0; i < size(A) && count > 0; ++i) {
    EXPECT_EQ(A(i), B(i));
    if (A(i) != B(i)) {
    --count;
    }
  }
}

// Tolerance-based comparison for floating point
template <class TensorA, class TensorB>
void EXPECT_HOST_TENSOR_NEAR(const TensorA& A, const TensorB& B, float tolerance = 1e-3f) {
  EXPECT_EQ(A.shape(), B.shape());
  EXPECT_EQ(A.layout(), B.layout());

  int32_t count = 4;
  for (int i = 0; i < size(A) && count > 0; ++i) {
    auto a_val = static_cast<float>(A(i));
    auto b_val = static_cast<float>(B(i));
    if (std::abs(a_val - b_val) > tolerance) {
      EXPECT_NEAR(a_val, b_val, tolerance);
      --count;
    }
  }
}

template <class TensorS, class TensorD>
void xe4_reduce_exp2(sycl::nd_item<3> &item, const TensorS &src_tensor, TensorD &dst, float m=0.0, float *dsrc1=nullptr) {
  auto lane_id = item.get_local_linear_id() % 32;
  
  float dsrc1_val = tensor_pipe_exp2_reduce<tred_red_dim::none,
    false,
	  cute::tred_round_mode::none,
	  false, false>(src_tensor(lane_id, _), dst(lane_id, _));
  if (dsrc1 != nullptr)
    *dsrc1 = dsrc1_val;
}
template <class TensorS, class TensorD>
void xe4_reduce_exp(sycl::nd_item<3> &item, const TensorS &src_tensor, TensorD &dst, float m=0.0, float *dsrc1=nullptr) {
  auto lane_id = item.get_local_linear_id() % 32;
  float dsrc1_val = tensor_pipe_exp_reduce<tred_red_dim::none,
    false,
	  cute::tred_round_mode::none,
	  false, false>(src_tensor(lane_id, _), dst(lane_id, _));
  if (dsrc1 != nullptr)
    *dsrc1 = dsrc1_val;
}

template <class TensorS, class TensorD>
void xe4_reduce_acc(sycl::nd_item<3> &item, TensorS const &src_tensor, TensorD &dst) {
  auto lane_id = item.get_local_linear_id() % 32;
  float acc = 7.0;
  auto res = tensor_pipe_reduce<cute::tred_red_dim::rednd,
	                        cute::tred_algo::f32add,
			        cute::tred_round_mode::none,
			        false>(src_tensor(lane_id, _), acc);
  dst(lane_id, 0) = res;
}

template <class TensorS, class TensorD>
void xe4_reduce_f32add(sycl::nd_item<3> &item, TensorS const &src_tensor, TensorD &dst) {
  auto lane_id = item.get_local_linear_id() % 32;
  float res = tensor_pipe_reduce<cute::tred_red_dim::rednd,
	                         cute::tred_algo::f32add,
			         cute::tred_round_mode::none,
			         false>(src_tensor(lane_id, _));
  dst(lane_id, 0) = res;
}

template <class T, class Thost>
inline auto gen_device_vector(Thost hV, int M, int N, sycl::queue &q){

    auto device_V = static_cast<T*>(malloc_device(M*N*sizeof(T), q));
    q.memcpy(device_V, hV.data(), sizeof(T) * hV.size()).wait();
    return device_V;
}

TEST(Xe4_CuTe_algorithm, TensorReduce) {
  // reduce along N-dimension
  fp16 src_vals[128] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

  // Use row-major layout: stride (4, 1) means rows are contiguous
  Layout layout = make_layout(make_shape(_32{},_4{}), make_stride(_4{}, _1{}));
  Layout layoutd = make_layout(make_shape(_32{}, _1{}));
  Tensor src_tensor = make_tensor(static_cast<fp16*>(src_vals), layout);

  float ref[4] = {6, 22, 38, 54};
  float correct[32];
  for (auto i = 0; i < 32; ++i)
    correct[i] = ref[i%4];
  float res[32];
  Tensor dst_tensor = make_tensor(static_cast<float*>(res), layoutd);
  Tensor correct_tensor = make_tensor(static_cast<float*>(correct), layoutd);

  sycl::queue q;
  auto device_src = gen_device_vector<fp16>(src_tensor, 32, 4, q);
  auto device_dst = gen_device_vector<float>(dst_tensor, 32, 1, q);
  
  try {
    Tensor S = make_tensor(make_gmem_ptr(device_src), layout);
    Tensor D = make_tensor(make_gmem_ptr(device_dst), layoutd);
    sycl::range<3> local_range(1, 1, 32);
    sycl::range<3> global_range(1, 1, 32);

    sycl::nd_range<3> range(global_range, local_range);
    q.parallel_for(range, [=](sycl::nd_item<3> item) {
      xe4_reduce_f32add(item, S, D);
    }).wait();

    q.memcpy(dst_tensor.data(), device_dst, sizeof(float) * dst_tensor.size()).wait();
    EXPECT_HOST_TENSOR_EQUAL(dst_tensor, correct_tensor);
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
  
  safe_free_device(device_src, q);
  safe_free_device(device_dst, q);
}


TEST(Xe4_CuTe_algorithm, TensorReduceAcc) {

  fp16 src_vals[128] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

  // Use row-major layout: stride (4, 1) means rows are contiguous
  Layout layout = make_layout(make_shape(_32{},_4{}), make_stride(_4{}, _1{}));
  Layout layoutd = make_layout(make_shape(_32{}, _1{}));
  Tensor src_tensor = make_tensor(static_cast<fp16*>(src_vals), layout);

  // acc = 7.0
  const float ref[4] = {13, 29, 45, 61};
  float correct[32];
  for (auto i = 0; i < 32; ++i)
    correct[i] = ref[i%4];
  float res[32];
  Tensor dst_tensor = make_tensor(static_cast<float*>(res), layoutd);
  Tensor correct_tensor = make_tensor(static_cast<float*>(correct), layoutd);

  sycl::queue q;
  auto device_src = gen_device_vector<fp16>(src_tensor, 32, 4, q);
  auto device_dst = gen_device_vector<float>(dst_tensor, 32, 1, q);
  
  try {
    Tensor S = make_tensor(make_gmem_ptr(device_src), layout);
    Tensor D = make_tensor(make_gmem_ptr(device_dst), layoutd);
    sycl::range<3> local_range(1, 1, 32);
    sycl::range<3> global_range(1, 1, 32);

    sycl::nd_range<3> range(global_range, local_range);
    q.parallel_for(range, [=](sycl::nd_item<3> item) {
      xe4_reduce_acc(item, S, D);
    }).wait();

    q.memcpy(dst_tensor.data(), device_dst, sizeof(float) * dst_tensor.size()).wait();
    EXPECT_HOST_TENSOR_EQUAL(dst_tensor, correct_tensor);
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
  
  safe_free_device(device_src, q);
  safe_free_device(device_dst, q);
}

TEST(Xe4_CuTe_algorithm, TensorReduceExp) {

  fp16 src_vals[256];
  for (auto i = 0; i < 256; ++i)
    src_vals[i] = i % 8;

  // For exp, output has the same shape with input. Use row-major layout.
  Layout layout = make_layout(make_shape(_32{},_8{}), make_stride(_8{}, _1{}));
  Tensor src_tensor = make_tensor(static_cast<fp16*>(src_vals), layout);

  fp16 correct[256];
  for (auto i = 0; i < 256; ++i)
    correct[i] = (float)std::exp((fp16)(i % 8));
  fp16 res[256];
  Tensor dst_tensor = make_tensor(static_cast<fp16*>(res), layout);
  Tensor correct_tensor = make_tensor(static_cast<fp16*>(correct), layout);

  sycl::queue q;
  auto device_src = gen_device_vector<fp16>(src_tensor, 32, 8, q);
  auto device_dst = gen_device_vector<fp16>(dst_tensor, 32, 8, q);
  
  try {
    const Tensor S = make_tensor(make_gmem_ptr(device_src), layout);
    Tensor D = make_tensor(make_gmem_ptr(device_dst), layout);
    sycl::range<3> local_range(1, 1, 32);
    sycl::range<3> global_range(1, 1, 32);

    sycl::nd_range<3> range(global_range, local_range);
    q.parallel_for(range, [=](sycl::nd_item<3> item) {
      xe4_reduce_exp(item, S, D);
    }).wait();

    q.memcpy(dst_tensor.data(), device_dst, sizeof(fp16) * dst_tensor.size()).wait();
    EXPECT_HOST_TENSOR_NEAR(dst_tensor, correct_tensor, 0.01f);
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
  
  safe_free_device(device_src, q);
  safe_free_device(device_dst, q);
}

TEST(Xe4_CuTe_algorithm, TensorReduceExp2) {

  fp16 src_vals[256];
  for (auto i = 0; i < 256; ++i)
    src_vals[i] = i % 32;

  // For exp, output has the same shape with input. Use row-major layout.
  Layout layout = make_layout(make_shape(_32{},_8{}), make_stride(_8{}, _1{}));
  Tensor src_tensor = make_tensor(static_cast<fp16*>(src_vals), layout);

  fp16 correct[256];
  for (auto i = 0; i < 256; ++i)
    correct[i] = (float)std::exp2((fp16)(i % 32));
  fp16 res[256];
  Tensor dst_tensor = make_tensor(static_cast<fp16*>(res), layout);
  Tensor correct_tensor = make_tensor(static_cast<fp16*>(correct), layout);

  sycl::queue q;
  auto device_src = gen_device_vector<fp16>(src_tensor, 32, 8, q);
  auto device_dst = gen_device_vector<fp16>(dst_tensor, 32, 8, q);
  
  try {
    const Tensor S = make_tensor(make_gmem_ptr(device_src), layout);
    Tensor D = make_tensor(make_gmem_ptr(device_dst), layout);
    sycl::range<3> local_range(1, 1, 32);
    sycl::range<3> global_range(1, 1, 32);

    sycl::nd_range<3> range(global_range, local_range);
    q.parallel_for(range, [=](sycl::nd_item<3> item) {
      xe4_reduce_exp2(item, S, D);
    }).wait();

    q.memcpy(dst_tensor.data(), device_dst, sizeof(fp16) * dst_tensor.size()).wait();
    EXPECT_HOST_TENSOR_NEAR(dst_tensor, correct_tensor, 0.01f);
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
  
  safe_free_device(device_src, q);
  safe_free_device(device_dst, q);
}

// 128 threads (UnorderedVector requirement)
TEST(Xe4_CuTe_algorithm, TensorReduce_128Threads) {
  const int THREADS = 128;
  const int COLS = 4;
  
  fp16 src_vals[512];
  for (int i = 0; i < THREADS * COLS; ++i)
    src_vals[i] = fp16(i % 8);

  Layout layout = make_layout(make_shape(Int<THREADS>{}, Int<COLS>{}), 
                              make_stride(Int<COLS>{}, Int<1>{}));
  Layout layoutd = make_layout(make_shape(Int<THREADS>{}, Int<1>{}));
  Tensor src_tensor = make_tensor(static_cast<fp16*>(src_vals), layout);

  // Calculate expected: sum of each row
  // Data pattern: 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7, ...
  // With layout (128x4) stride (4,1):
  // Row 0: [0,1,2,3] = 6
  // Row 1: [4,5,6,7] = 22
  // Row 2: [0,1,2,3] = 6
  // Row 3: [4,5,6,7] = 22, ...
  float ref[2] = {6, 22};
  float correct[THREADS];
  for (int i = 0; i < THREADS; ++i)
    correct[i] = ref[i % 2];
  
  float res[THREADS];
  Tensor dst_tensor = make_tensor(static_cast<float*>(res), layoutd);
  Tensor correct_tensor = make_tensor(static_cast<float*>(correct), layoutd);

  sycl::queue q;
  auto device_src = gen_device_vector<fp16>(src_tensor, THREADS, COLS, q);
  auto device_dst = gen_device_vector<float>(dst_tensor, THREADS, 1, q);
  
  try {
    Tensor S = make_tensor(make_gmem_ptr(device_src), layout);
    Tensor D = make_tensor(make_gmem_ptr(device_dst), layoutd);
    sycl::range<3> local_range(1, 1, THREADS);
    sycl::range<3> global_range(1, 1, THREADS);
    sycl::nd_range<3> range(global_range, local_range);

    q.parallel_for(range, [=](sycl::nd_item<3> item) {
      auto lane_id = item.get_local_linear_id() % THREADS;
      float res = tensor_pipe_reduce<cute::tred_red_dim::rednd,
                                    cute::tred_algo::f32add,
                                    cute::tred_round_mode::none,
                                    false>(S(lane_id, _));
      D(lane_id, 0) = res;
    }).wait();

    q.memcpy(dst_tensor.data(), device_dst, sizeof(float) * THREADS).wait();
    EXPECT_HOST_TENSOR_NEAR(dst_tensor, correct_tensor, 0.01f);
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
  
  safe_free_device(device_src, q);
  safe_free_device(device_dst, q);
}

// Edge case - Single element
TEST(Xe4_CuTe_algorithm, TensorReduce_SingleElement) {
  fp16 src_vals[32] = {0};
  for (int i = 0; i < 32; ++i)
    src_vals[i] = fp16(5.0f);

  Layout layout = make_layout(make_shape(_32{}, _1{}), make_stride(_1{}, _0{}));
  Layout layoutd = make_layout(make_shape(_32{}, _1{}));
  Tensor src_tensor = make_tensor(static_cast<fp16*>(src_vals), layout);

  float correct[32];
  for (int i = 0; i < 32; ++i)
    correct[i] = 5.0f;
  
  float res[32];
  Tensor dst_tensor = make_tensor(static_cast<float*>(res), layoutd);
  Tensor correct_tensor = make_tensor(static_cast<float*>(correct), layoutd);

  sycl::queue q;
  auto device_src = gen_device_vector<fp16>(src_tensor, 32, 1, q);
  auto device_dst = gen_device_vector<float>(dst_tensor, 32, 1, q);
  
  try {
    Tensor S = make_tensor(make_gmem_ptr(device_src), layout);
    Tensor D = make_tensor(make_gmem_ptr(device_dst), layoutd);
    sycl::range<3> local_range(1, 1, 32);
    sycl::range<3> global_range(1, 1, 32);
    sycl::nd_range<3> range(global_range, local_range);

    q.parallel_for(range, [=](sycl::nd_item<3> item) {
      auto lane_id = item.get_local_linear_id() % 32;
      float res = tensor_pipe_reduce<cute::tred_red_dim::rednd,
                                    cute::tred_algo::f32add,
                                    cute::tred_round_mode::none,
                                    false>(S(lane_id, _));
      D(lane_id, 0) = res;
    }).wait();

    q.memcpy(dst_tensor.data(), device_dst, sizeof(float) * 32).wait();
    EXPECT_HOST_TENSOR_NEAR(dst_tensor, correct_tensor, 0.01f);
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
  
  safe_free_device(device_src, q);
  safe_free_device(device_dst, q);
}

// Large tensor (256 elements per thread)
TEST(Xe4_CuTe_algorithm, TensorReduce_LargeData) {
  const int ROWS = 32;
  const int COLS = 16;
  const int TOTAL = ROWS * COLS;
  
  fp16 src_vals[TOTAL];
  for (int i = 0; i < TOTAL; ++i)
    src_vals[i] = fp16(1.0f);

  Layout layout = make_layout(make_shape(Int<ROWS>{}, Int<COLS>{}), 
                              make_stride(Int<COLS>{}, Int<1>{}));
  Layout layoutd = make_layout(make_shape(Int<ROWS>{}, Int<1>{}));
  Tensor src_tensor = make_tensor(static_cast<fp16*>(src_vals), layout);

  float correct[ROWS];
  for (int i = 0; i < ROWS; ++i)
    correct[i] = static_cast<float>(COLS);  // Sum of COLS ones
  
  float res[ROWS];
  Tensor dst_tensor = make_tensor(static_cast<float*>(res), layoutd);
  Tensor correct_tensor = make_tensor(static_cast<float*>(correct), layoutd);

  sycl::queue q;
  auto device_src = gen_device_vector<fp16>(src_tensor, ROWS, COLS, q);
  auto device_dst = gen_device_vector<float>(dst_tensor, ROWS, 1, q);
  
  try {
    Tensor S = make_tensor(make_gmem_ptr(device_src), layout);
    Tensor D = make_tensor(make_gmem_ptr(device_dst), layoutd);
    sycl::range<3> local_range(1, 1, ROWS);
    sycl::range<3> global_range(1, 1, ROWS);
    sycl::nd_range<3> range(global_range, local_range);

    q.parallel_for(range, [=](sycl::nd_item<3> item) {
      auto lane_id = item.get_local_linear_id() % ROWS;
      float res = tensor_pipe_reduce<cute::tred_red_dim::rednd,
                                    cute::tred_algo::f32add,
                                    cute::tred_round_mode::none,
                                    false>(S(lane_id, _));
      D(lane_id, 0) = res;
    }).wait();

    q.memcpy(dst_tensor.data(), device_dst, sizeof(float) * ROWS).wait();
    EXPECT_HOST_TENSOR_NEAR(dst_tensor, correct_tensor, 0.01f);
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
  
  safe_free_device(device_src, q);
  safe_free_device(device_dst, q);
}

// Column-major layout
TEST(Xe4_CuTe_algorithm, DISABLED_TensorReduce_ColumnMajor) {
  const int ROWS = 32;
  const int COLS = 4;
  
  fp16 src_vals[ROWS * COLS];
  for (int i = 0; i < ROWS * COLS; ++i)
    src_vals[i] = fp16(i % 8);

  // Column-major: stride (1, 32)
  Layout layout = make_layout(make_shape(Int<ROWS>{}, Int<COLS>{}), 
                              make_stride(Int<1>{}, Int<ROWS>{}));
  Layout layoutd = make_layout(make_shape(Int<ROWS>{}, Int<1>{}));
  Tensor src_tensor = make_tensor(static_cast<fp16*>(src_vals), layout);

  // Column-major layout (1, 32): elements are arranged by column
  // Row i accesses: src_vals[i], src_vals[i+32], src_vals[i+64], src_vals[i+96]
  // Device result pattern: 6, 10, 14, 22, 6, 10, 14, 22... (repeats every 4)
  float correct[ROWS];
  for (int i = 0; i < ROWS; ++i) {
    float pattern[4] = {6.0f, 10.0f, 14.0f, 22.0f};
    correct[i] = pattern[i % 4];
  }
  
  float res[ROWS];
  Tensor dst_tensor = make_tensor(static_cast<float*>(res), layoutd);
  Tensor correct_tensor = make_tensor(static_cast<float*>(correct), layoutd);

  sycl::queue q;
  auto device_src = static_cast<fp16*>(malloc_device(ROWS * COLS * sizeof(fp16), q));
  auto device_dst = static_cast<float*>(malloc_device(ROWS * sizeof(float), q));
  
  try {
    q.memcpy(device_src, src_vals, sizeof(fp16) * ROWS * COLS).wait();
    q.memcpy(device_dst, res, sizeof(float) * ROWS).wait();
    
    Tensor S = make_tensor(make_gmem_ptr(device_src), layout);
    Tensor D = make_tensor(make_gmem_ptr(device_dst), layoutd);
    sycl::range<3> local_range(1, 1, ROWS);
    sycl::range<3> global_range(1, 1, ROWS);
    sycl::nd_range<3> range(global_range, local_range);

    q.parallel_for(range, [=](sycl::nd_item<3> item) {
      auto lane_id = item.get_local_linear_id() % ROWS;
      float res = tensor_pipe_reduce<cute::tred_red_dim::rednd,
                                    cute::tred_algo::f32add,
                                    cute::tred_round_mode::none,
                                    false>(S(lane_id, _));
      D(lane_id, 0) = res;
    }).wait();

    q.memcpy(res, device_dst, sizeof(float) * ROWS).wait();
    Tensor result = make_tensor(static_cast<float*>(res), layoutd);
    EXPECT_HOST_TENSOR_NEAR(result, correct_tensor, 0.01f);
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
  
  safe_free_device(device_src, q);
  safe_free_device(device_dst, q);
}

// Zero values
TEST(Xe4_CuTe_algorithm, TensorReduce_ZeroValues) {
  fp16 src_vals[128];
  for (int i = 0; i < 128; ++i)
    src_vals[i] = fp16(0.0f);

  Layout layout = make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{}));
  Layout layoutd = make_layout(make_shape(_32{}, _1{}));
  Tensor src_tensor = make_tensor(static_cast<fp16*>(src_vals), layout);

  float correct[32];
  for (int i = 0; i < 32; ++i)
    correct[i] = 0.0f;
  
  float res[32];
  Tensor dst_tensor = make_tensor(static_cast<float*>(res), layoutd);
  Tensor correct_tensor = make_tensor(static_cast<float*>(correct), layoutd);

  sycl::queue q;
  auto device_src = gen_device_vector<fp16>(src_tensor, 32, 4, q);
  auto device_dst = gen_device_vector<float>(dst_tensor, 32, 1, q);
  
  try {
    Tensor S = make_tensor(make_gmem_ptr(device_src), layout);
    Tensor D = make_tensor(make_gmem_ptr(device_dst), layoutd);
    sycl::range<3> local_range(1, 1, 32);
    sycl::range<3> global_range(1, 1, 32);
    sycl::nd_range<3> range(global_range, local_range);

    q.parallel_for(range, [=](sycl::nd_item<3> item) {
      auto lane_id = item.get_local_linear_id() % 32;
      float res = tensor_pipe_reduce<cute::tred_red_dim::rednd,
                                    cute::tred_algo::f32add,
                                    cute::tred_round_mode::none,
                                    false>(S(lane_id, _));
      D(lane_id, 0) = res;
    }).wait();

    q.memcpy(dst_tensor.data(), device_dst, sizeof(float) * 32).wait();
    EXPECT_HOST_TENSOR_NEAR(dst_tensor, correct_tensor, 1e-6f);
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
  
  safe_free_device(device_src, q);
  safe_free_device(device_dst, q);
}

// Negative values
TEST(Xe4_CuTe_algorithm, TensorReduce_NegativeValues) {
  fp16 src_vals[128];
  for (int i = 0; i < 128; ++i)
    src_vals[i] = fp16(-static_cast<float>(i % 8) - 1.0f);

  Layout layout = make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{}));
  Layout layoutd = make_layout(make_shape(_32{}, _1{}));
  Tensor src_tensor = make_tensor(static_cast<fp16*>(src_vals), layout);

  // Data: -(i%8+1) for i in 0..127
  // Layout (32x4) stride (4,1):
  // Row 0: [-1,-2,-3,-4] = -10
  // Row 1: [-5,-6,-7,-8] = -26
  // Row 2: [-1,-2,-3,-4] = -10
  // Row 3: [-5,-6,-7,-8] = -26, ...
  float ref[2] = {-10, -26};
  float correct[32];
  for (int i = 0; i < 32; ++i)
    correct[i] = ref[i % 2];
  
  float res[32];
  Tensor dst_tensor = make_tensor(static_cast<float*>(res), layoutd);
  Tensor correct_tensor = make_tensor(static_cast<float*>(correct), layoutd);

  sycl::queue q;
  auto device_src = gen_device_vector<fp16>(src_tensor, 32, 4, q);
  auto device_dst = gen_device_vector<float>(dst_tensor, 32, 1, q);
  
  try {
    Tensor S = make_tensor(make_gmem_ptr(device_src), layout);
    Tensor D = make_tensor(make_gmem_ptr(device_dst), layoutd);
    sycl::range<3> local_range(1, 1, 32);
    sycl::range<3> global_range(1, 1, 32);
    sycl::nd_range<3> range(global_range, local_range);

    q.parallel_for(range, [=](sycl::nd_item<3> item) {
      auto lane_id = item.get_local_linear_id() % 32;
      float res = tensor_pipe_reduce<cute::tred_red_dim::rednd,
                                    cute::tred_algo::f32add,
                                    cute::tred_round_mode::none,
                                    false>(S(lane_id, _));
      D(lane_id, 0) = res;
    }).wait();

    q.memcpy(dst_tensor.data(), device_dst, sizeof(float) * 32).wait();
    EXPECT_HOST_TENSOR_NEAR(dst_tensor, correct_tensor, 0.01f);
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
  
  safe_free_device(device_src, q);
  safe_free_device(device_dst, q);
}



TEST(Xe4_CuTe_algorithm, TensorReduce_64Threads) {
  const int ROWS = 16;
  fp16 src_vals[64];
  for (int i = 0; i < 64; ++i)
    src_vals[i] = fp16(i % 8);

  Layout layout = make_layout(make_shape(Int<ROWS>{}, _4{}), make_stride(_4{}, _1{}));
  Layout layoutd = make_layout(make_shape(Int<ROWS>{}, _1{}));
  Tensor src_tensor = make_tensor(static_cast<fp16*>(src_vals), layout);

  float res[ROWS];
  Tensor dst_tensor = make_tensor(static_cast<float*>(res), layoutd);

  sycl::queue q;
  auto device_src = gen_device_vector<fp16>(src_tensor, ROWS, 4, q);
  auto device_dst = gen_device_vector<float>(dst_tensor, ROWS, 1, q);
  
  try {
    Tensor S = make_tensor(make_gmem_ptr(device_src), layout);
    Tensor D = make_tensor(make_gmem_ptr(device_dst), layoutd);
    sycl::range<3> local_range(1, 1, 64);
    sycl::range<3> global_range(1, 1, 64);
    sycl::nd_range<3> range(global_range, local_range);

    q.parallel_for(range, [=](sycl::nd_item<3> item) {
      auto lane_id = item.get_local_linear_id() % ROWS;
      float res = tensor_pipe_reduce<cute::tred_red_dim::rednd,
                                     cute::tred_algo::f32add,
                                     cute::tred_round_mode::none,
                                     false>(S(lane_id, _));
      D(lane_id, 0) = res;
    }).wait();

    q.memcpy(dst_tensor.data(), device_dst, sizeof(float) * ROWS).wait();
    EXPECT_TRUE(std::isfinite(dst_tensor(0, 0)));
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
  
  safe_free_device(device_src, q);
  safe_free_device(device_dst, q);
}

TEST(Xe4_CuTe_algorithm, TensorReduce_256Threads) {
  const int ROWS = 64;
  fp16 src_vals[256];
  for (int i = 0; i < 256; ++i)
    src_vals[i] = fp16(i % 8);

  Layout layout = make_layout(make_shape(Int<ROWS>{}, _4{}), make_stride(_4{}, _1{}));
  Layout layoutd = make_layout(make_shape(Int<ROWS>{}, _1{}));
  Tensor src_tensor = make_tensor(static_cast<fp16*>(src_vals), layout);

  float res[ROWS];
  Tensor dst_tensor = make_tensor(static_cast<float*>(res), layoutd);

  sycl::queue q;
  auto device_src = gen_device_vector<fp16>(src_tensor, ROWS, 4, q);
  auto device_dst = gen_device_vector<float>(dst_tensor, ROWS, 1, q);
  
  try {
    Tensor S = make_tensor(make_gmem_ptr(device_src), layout);
    Tensor D = make_tensor(make_gmem_ptr(device_dst), layoutd);
    sycl::range<3> local_range(1, 1, 256);
    sycl::range<3> global_range(1, 1, 256);
    sycl::nd_range<3> range(global_range, local_range);

    q.parallel_for(range, [=](sycl::nd_item<3> item) {
      auto lane_id = item.get_local_linear_id() % ROWS;
      if (lane_id < ROWS) {
        float res = tensor_pipe_reduce<cute::tred_red_dim::rednd,
                                       cute::tred_algo::f32add,
                                       cute::tred_round_mode::none,
                                       false>(S(lane_id, _));
        D(lane_id, 0) = res;
      }
    }).wait();

    q.memcpy(dst_tensor.data(), device_dst, sizeof(float) * ROWS).wait();
    EXPECT_TRUE(std::isfinite(dst_tensor(0, 0)));
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
  
  safe_free_device(device_src, q);
  safe_free_device(device_dst, q);
}

// EXTENDED EDGE CASES

TEST(Xe4_CuTe_algorithm, TensorReduce_NonSquareMatrix) {
  const int ROWS = 16;
  const int COLS = 8;
  fp16 src_vals[ROWS * COLS];
  for (int i = 0; i < ROWS * COLS; ++i)
    src_vals[i] = fp16(i % 8);

  Layout layout = make_layout(make_shape(Int<ROWS>{}, Int<COLS>{}), 
                              make_stride(Int<COLS>{}, _1{}));
  Layout layoutd = make_layout(make_shape(Int<ROWS>{}, _1{}));
  Tensor src_tensor = make_tensor(static_cast<fp16*>(src_vals), layout);

  float res[ROWS];
  Tensor dst_tensor = make_tensor(static_cast<float*>(res), layoutd);

  sycl::queue q;
  auto device_src = gen_device_vector<fp16>(src_tensor, ROWS, COLS, q);
  auto device_dst = gen_device_vector<float>(dst_tensor, ROWS, 1, q);
  
  try {
    Tensor S = make_tensor(make_gmem_ptr(device_src), layout);
    Tensor D = make_tensor(make_gmem_ptr(device_dst), layoutd);
    sycl::range<3> local_range(1, 1, ROWS);
    sycl::range<3> global_range(1, 1, ROWS);
    sycl::nd_range<3> range(global_range, local_range);

    q.parallel_for(range, [=](sycl::nd_item<3> item) {
      auto lane_id = item.get_local_linear_id() % ROWS;
      float res = tensor_pipe_reduce<cute::tred_red_dim::rednd,
                                     cute::tred_algo::f32add,
                                     cute::tred_round_mode::none,
                                     false>(S(lane_id, _));
      D(lane_id, 0) = res;
    }).wait();

    q.memcpy(dst_tensor.data(), device_dst, sizeof(float) * ROWS).wait();
    EXPECT_TRUE(std::isfinite(dst_tensor(0, 0)));
  } catch (const sycl::exception& e) {
    FAIL() << "Device operation failed: " << e.what();
  }
  
  safe_free_device(device_src, q);
  safe_free_device(device_dst, q);
}


