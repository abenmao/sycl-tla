/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation. All rights reserved.
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

/*******************************************************************************
 * CuTe MMA_Traits for XE4 TMM (synchronous, register-based Tensor Matrix Multiply).
 *
 * TMM fragment layout specification (from XE4 ISA 3.1.3):
 *
 *   Matrix A[32,K]:  M-major. Lane i owns row i. K elements packed into
 *                    ceil(K * bits(A_type) / 32) consecutive 32-bit registers.
 *
 *   Matrix B[K,N]:   K-major. Elements packed along K into 32-bit units,
 *                    distributed column-by-column across 32 lanes.
 *                    Lane = (col * K/F + packed_k) % 32
 *                    Reg  = (col * K/F + packed_k) / 32
 *
 *   Matrix C/D[32,N]: Same as Matrix A -- lane i owns row i.
 *                    N elements packed into ceil(N * bits(C_type) / 32) registers.
 *
 * The layouts below encode these mappings as CuTe (tid, vid) -> flat_coord layouts,
 * so TiledMMA / gemm() can partition tensors for TMM automatically.
 **************************************************************************************************/
#pragma once

#include <cute/config.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/arch/mma_xe4_tmm.hpp>
#include <cute/atom/mma_traits.hpp>

namespace cute {

// ============================================================================
// Reusable TMM fragment layout builders
// ============================================================================

namespace tmm {

// ---------- packingFactor: how many elements fit in one 32-bit register ----------
//   F = 32 / numOfBits(T)
//   e.g. float->1, fp16->2, bf16->2, e5m2->4, e4m3->4, e2m1->8
template <class T>
inline constexpr int packF = 32 / cute::sizeof_bits_v<T>;

// ---------- Fragment register counts (uint32 units per lane) ----------
//   A[32,K]:  nRegsA = ceil(K * bits(T) / 32) = ceil(K / packF)
//   B[K,N]:   nRegsB = ceil(N * K / packF / 32)  -- total 32-bit units / 32 lanes
//   C[32,N]:  nRegsC = ceil(N * bits(T) / 32) = ceil(N / packF)
template <class T, int K>
inline constexpr int nRegsA = ceil_div(K, packF<T>); //(K + packF<T> - 1) / packF<T>;

template <class T, int K, int N>
inline constexpr int nRegsB = ceil_div(N*K*cute::sizeof_bits_v<T>, 1024); //(N * K * cute::sizeof_bits_v<T> + 1023) / 1024;

template <class T, int N>
inline constexpr int nRegsC = ceil_div(N, packF<T>); //(N + packF<T> - 1) / packF<T>;

// ============================================================================
// Matrix A Layout:  (tid, vid) -> crd2idx(M, K)
//
// A[M=32, K]:  lane i (tid=i) owns row i.
// Each lane has nRegsA registers, each holding packF consecutive K-elements.
//
// Mapping:
//   tid  = i  -> M-coordinate = i
//   vid  = r  -> K-coordinate = r * packF ... (r+1)*packF - 1
//
// In flat (M,K) coordinates with K-inner (row-major A):
//   flat_idx(tid, vid) = tid * K + vid   (vid already in packed-element units)
//
// As CuTe layout over the ELEMENT space (M*K elements total):
//   Shape:  (32, nRegsA * packF)  = (32, K)
//   Stride: (K,  1)                         -- M-major = row-major
//
// But MMA_Traits layouts map (tid, vid) to the logical (M,K) coordinate pair,
// not a flat index. CuTe flattens (M,K) into a single linear space where the
// layout's codomain IS that flat space. For a row-major A[32,K]:
//   crd2idx(m, k) = m * K + k
//   tid -> m, vid -> k (in element space, packF elements per register)
// ============================================================================
template <class T, int M, int K>
using ALayout = Layout<
    Shape <Int<M>,               Int<K>>,
    Stride<_1,               Int<M>> // switch this for col-major A //
>;


// ============================================================================
// Max lanes (registers) used in fragment layout of B
// 
// B(0) : denote the first column of matrix B[KxN] (i.e. B[0:K-1, 0]
//
// max_lanes = min{ (elem_bits*K)/32, 32}
// ============================================================================
template <typename T, int K, int N>
struct tmmLayoutBMaxLanes{
  static constexpr int elem_bits = cute::sizeof_bits_v<T>;
  static constexpr int total_bits = K*N*elem_bits;
  static constexpr int value = 
      cute::min(Int< total_bits / 32>{}, Int<32>{});
};

// ============================================================================
// Matrix B Layout:  (tid, vid) -> crd2idx(K, N)
//
// B[K, N]:  elements are packed along K into 32-bit units, then distributed
// across 32 lanes column by column (N-dimension).
//
// Let F = packF<T> (elements per uint32), P = K/F (packed-K units per column).
//
// STEP-0: Nested Layout of Packed B:
// --------------------------
// B[K, N] --- 32-bit-packing_transform ---> B_packed[K/F, N] 
//
//  If B is K-major: 
//                      (note stride within the pack is N) 
//  shape : stride 
//  (K, N) : (N, 1) -> ((K/F, N), (F)) : ((F*N, 1), N) -> ((P , N), F) : ((F*N, 1), N)
//
//
// STEP-1: If P*N % 32 == 0 (evenly distributed across 32 lanes) (also P*N >=32)
// -------------------------------------------------------------
// Column n contributes P packed entries. These are laid out sequentially across
// lanes. The global index of packed entry (p, n) = p * N + n.
//   lane = global_idx % 32
//   reg  = global_idx / 32
//
// In element space, packed entry p of column n covers elements:
// { B[p*F, n], B[p*F+1, n], ... B[p*F+(F-1), n] } 
//
// The CuTe layout needs to map (tid, vid) -> crd2idx(k, n) where crd2idx(k,n) = k*N + n.
//
// Each lane holds nRegsB registers. The vid iterates over those registers, and
// within each register over F elements.
//
// For the simplest encoding (N*K evenly distributes across 32 lanes), the layout
// factors into:
//
//   Shape:  (32,       nRegsB * F)
//             |              |
//           tid=lane     vid in elements
//
//
//  NESTED THREAD LAYOUT:
//  ---------------------
//  Recall that when K/F < 32 we pack multiple columns into the packed encoding
//  of B. Notice that that if we just use 32 as shape with a positive stride
//  then the result will no longer be a TV layout as some elements can go out
//  of bounds i.e. layout(t,v) > (N*K). We solve this problem by breaking the
//  32 lanes into a shape (K/F x 32/(K/F) ) in a column major form. 
//
//     col-i             col-i+1
//   ( pack_0)          ( pack_0) 
//   ( pack_1)          ( pack_1)
//      .
//      .
//   (pack_(K/F -1))
//
//  Resulting nested thread layout:
//
//   (K/F , (32*F)/K) : (N*F, 1)
// 
// The stride is tricky because consecutive elements within a lane come from
// different (k,n) positions. For CuTe's flat (K*N) space:
//
//   Within one register (F elements of one column): 
//   stride = N (since B[K, N] is a row major layout) 
// 
//   Across registers (next column batch): stride = (32/(K/F)) 
//   The lane offset: stride = N*F (next packed-K unit)
//
// For even distribution (N*P % 32 == 0, N*P >=32):
//
//  Note: 32/(K/F) denotes the number of columns in original matrix packed into
//  single column of the encoded matrix
//
// For the common case (K*N / (F*32) = nRegsB, evenly distributed):
//   lane l gets entries at positions: l, l+32, l+64, ...  in the global packed array.
//   Each packed entry p at global position g covers elements at flat offset g*F in NK-space.
//
// Layout:  Shape  (( (K/F, (32*F)/K),         (packF, nRegsB)),
//          Stride (( (N*F, 1),        (N,     (32*F)/K)))
//
// This maps:  crd2idx(tid, vid) = tid*N*F + vid_sub*N + vid_reg*((32*F)/K)
//   where vid_sub in [0,F), vid_reg in [0, nRegsB)
//
// ============================================================================
template <class T, int K, int N>
using BLayout = Layout<
    Shape < Shape< Int<K/packF<T>>, 
                   Int<(tmmLayoutBMaxLanes<T, K, N>::value * packF<T>)/K> >, 
					  Shape <Int<packF<T>>,   Int<nRegsB<T,K,N>>>>,
    Stride< Stride<Int<N*packF<T>>, _1> , 
            Stride<Int<N>, Int<(tmmLayoutBMaxLanes<T, K, N>::value * packF<T>)/K>>>
>;

// ============================================================================
// Matrix C/D Layout:  (tid, vid) -> crd2idx(M, N)
//
// C[32, N] and D[32, N] have the SAME layout as Matrix A.
// Lane i owns row i. N elements packed into nRegsC registers.
//
//   tid  = i -> M-coordinate = i
//   vid  = element index within the N-columns of row i
//
// In flat (M,N) coordinates with N-inner (row-major C):
//   Shape:  (32, N)
//   Stride: (N,  1)
// ============================================================================
template <class T, int M, int N>
using CLayout = ALayout<T, M, N>;


} // namespace tmm

// ============================================================================
// MMA_Traits specialization for XE4_TMM
// ============================================================================
template <class d_type, class a_type, class b_type, class c_type, int N>
struct MMA_Traits<XE4_TMM<d_type, a_type, b_type, c_type, N>>
{
  using MMA_Op = XE4_TMM<d_type, a_type, b_type, c_type, N>;

  using ValTypeD = d_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  // Fragment element types = the logical element types (NOT uint32).
  // The generic mma_unpack will recast to register types via MMA_Op::{A,B,C,D}Registers.
  using FrgTypeA = a_type;
  using FrgTypeB = b_type;
  using FrgTypeC = c_type;

  static constexpr int M_atom = MMA_Op::M;
  static constexpr int K_atom = MMA_Op::K;

  using Shape_MNK = Shape<Int<M_atom>, Int<N>, Int<K_atom>>;

  // 32 lanes cooperate on each TMM instruction
  using ThrID = Layout<_32>;

  // (tid, vid) -> (m, k)   -- Matrix A fragment layout
  using ALayout = tmm::ALayout<a_type, M_atom, K_atom>;

  // (tid, vid) -> (n, k)   -- Matrix B fragment layout
  using BLayout = tmm::BLayout<b_type, K_atom, N>;

  // (tid, vid) -> (m, n)   -- Matrix C/D fragment layout
  using CLayout = tmm::CLayout<c_type, M_atom, N>;
};

template <class d_type, class a_type, class b_type, class c_type, int N,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr
void
mma_unpack(MMA_Traits<XE4_TMM<d_type, a_type, b_type, c_type, N>> const&,
           Tensor<TD, DLayout>      & D,
           Tensor<TA, ALayout> const& A,
           Tensor<TB, BLayout> const& B,
           Tensor<TC, CLayout> const& C)
{
  static_assert(is_rmem<TD>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TA>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TB>::value, "Expected registers in MMA_Atom::call");
  static_assert(is_rmem<TC>::value, "Expected registers in MMA_Atom::call");

  using MMA_Op   = XE4_TMM<d_type, a_type, b_type, c_type, N>;
  using RegTypeD = typename MMA_Op::DRegisters;
  using RegTypeA = typename MMA_Op::ARegisters;
  using RegTypeB = typename MMA_Op::BRegisters;
  using RegTypeC = typename MMA_Op::CRegisters;

  Tensor rA = recast<RegTypeA>(A);
  Tensor rB = recast<RegTypeB>(B);
  Tensor rD = recast<RegTypeD>(D);
  Tensor rC = recast<RegTypeC>(C);

  CUTE_STATIC_ASSERT_V(size(rA) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size(rB) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size(rD) == Int<1>{});
  CUTE_STATIC_ASSERT_V(size(rC) == Int<1>{});

  MMA_Op::fma(rD[0], rA[0], rB[0], rC[0]);
}

template <class d_type, class a_type, class b_type, class c_type, int N,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE constexpr
void
mma_unpack(MMA_Traits<XE4_TMM<d_type, a_type, b_type, c_type, N>> const& traits,
           Tensor<TD, DLayout>     && D,
           Tensor<TA, ALayout> const& A,
           Tensor<TB, BLayout> const& B,
           Tensor<TC, CLayout> const& C)
{
  mma_unpack(traits, D, A, B, C);
}


} // namespace cute
