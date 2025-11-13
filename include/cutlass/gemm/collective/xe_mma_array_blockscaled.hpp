/***************************************************************************************************
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/fp8_to_fp16.h"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cutlass/gemm/collective/xe_blockscaled_mma.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////
namespace cutlass::gemm::collective {

template <
  int Stages,
  class Schedule,
  class TileShape_,
  class ElementPairA_,
  class StridePairA_,
  class ElementPairB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyPairA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyPairB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopIntelXeXMX16BlockScaledGroup<Stages, Schedule>,
    TileShape_,
    ElementPairA_,
    StridePairA_,
    ElementPairB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyPairA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyPairB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopIntelXeXMX16BlockScaledGroup<Stages, Schedule>;
  using WorkgroupTileShape = TileShape_;

  using GmemTiledCopyPairA = GmemTiledCopyPairA_;
  using GmemTiledCopyPairB = GmemTiledCopyPairB_;

  using TiledMma = TiledMma_;
  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
  using ElementAMma = typename TiledMma::ValTypeA;
  using ElementBMma = typename TiledMma::ValTypeB;
  using StridePairA = StridePairA_;
  using StridePairB = StridePairB_;

  using ElementMMA = typename TiledMma_::ValTypeA;

  // A and B matrices
  using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
  using StrideA  = remove_cvref_t<decltype(get<0>(StridePairA{}))>;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;

  using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
  using StrideB  = remove_cvref_t<decltype(get<0>(StridePairB{}))>;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;

  static_assert(is_same_v<ElementA, ElementAMma>, "Elment type and MMA type should be same");
  static_assert(is_same_v<ElementB, ElementBMma>, "Elment type and MMA type should be same");
  // SFA and SFB

  static_assert(is_same_v<decltype(get<1>(ElementPairA{})), decltype(get<1>(ElementPairB{}))>, "Scale Element type should be symmetic");
  using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
  using StrideScaleA = remove_cvref_t<decltype(get<1>(StridePairA{}))>;
  using StrideScaleB = remove_cvref_t<decltype(get<1>(StridePairB{}))>;


  // TODO(Codeplay): Create a ScaledTensor class to encapsulate scale logic
  using ElementScaleA = ElementSF;
  using InternalStrideScaleA = cute::remove_pointer_t<StrideScaleA>;

  using ElementScaleB = ElementSF;
  using InternalStrideScaleB = cute::remove_pointer_t<StrideScaleB>;

  using ElementAccumulator = typename TiledMma::ValTypeC;


  // using GmemTiledCopyA = void;
  // using GmemTiledCopyB = void;
  using GmemTiledCopyA = typename std::tuple_element<0, GmemTiledCopyPairA>::type;
  using GmemTiledCopyB = typename std::tuple_element<0, GmemTiledCopyPairB>::type;
  using GmemTiledCopyScaleA = typename std::tuple_element<1, GmemTiledCopyPairA>::type;
  using GmemTiledCopyScaleB = typename std::tuple_element<1, GmemTiledCopyPairB>::type;

  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  using MmaType = typename TiledMma::ValTypeA; // ValTypeA and ValTypeB are always same and reflects MMA type on intel Xe


  static constexpr bool kSupportedElementA =
      cute::is_same_v<ElementA, cutlass::float_e5m2_t> ||
      cute::is_same_v<ElementA, cutlass::float_e4m3_t> ||
      cute::is_same_v<ElementA, cutlass::float_e2m1_t>;

  static constexpr bool kSupportedElementB =
      cute::is_same_v<ElementA, cutlass::float_e5m2_t> ||
      cute::is_same_v<ElementA, cutlass::float_e4m3_t> ||
      cute::is_same_v<ElementA, cutlass::float_e2m1_t>;

   static constexpr bool kScaleALeftmostUnitStride = [] {
    if constexpr (cute::is_same_v<InternalStrideScaleA, void>) {
      return false;
    } else {
      using LeftmostStrideA = remove_cvref_t<decltype(get<0>(InternalStrideScaleA{}))>;
      return cute::is_same_v<LeftmostStrideA, _1>;
    }
  }();

  static constexpr bool kScaleBLeftmostUnitStride = [] {
    if constexpr (cute::is_same_v<InternalStrideScaleB, void>) {
      return false;
    } else {
      using LeftmostStrideB = remove_cvref_t<decltype(get<0>(InternalStrideScaleB{}))>;
      return cute::is_same_v<LeftmostStrideB, _1>;
    }
  }();
  
  static_assert(std::is_same_v<TransformA, cute::identity>, "Transformation for A is not currently supported on Intel PVC");
  static_assert(std::is_same_v<TransformB, cute::identity>, "Transformation for B is not currently supported on Intel PVC");
  static_assert(kSupportedElementA && kSupportedElementB,
                "Intel Xe blockscaled MMA only supports bf8 and hf8 operand types.");
  static_assert(kScaleALeftmostUnitStride,
                "Intel Xe blockscaled MMA requires scale A leftmost stride to be _1.");
  static_assert(kScaleBLeftmostUnitStride,
                "Intel Xe blockscaled MMA requires scale B leftmost stride to be _1.");

public:
  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;

  static constexpr int BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr int BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr int BLK_K = get<2>(WorkgroupTileShape{});

  static constexpr int ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr int ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());

  static constexpr int SG_M = ceil_div(BLK_M, ATOM_M);
  static constexpr int SG_N = ceil_div(BLK_N, ATOM_N);
  static constexpr int SG_K = ceil_div(BLK_K, ATOM_K);
  using SubgroupTileShape = Shape<C<SG_M>, C<SG_N>, C<SG_K>>;

  static constexpr auto GROUP_K = 32;

  static_assert(SG_K >= 32, "Intel Xe blockscaled MMA requires SG_K to be at least 32.");

  static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;
  using CopyThreadShapeRev = decltype(cute::reverse(CopyThreadShape{}));

  // Helper to get tensor types
  template<class Element, class Stride>
  using TensorType = decltype(make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                                        make_layout(make_shape(int{}, int{}, int{}), Stride{})));

  using DefScaleType = cutlass::float_ue8m0_t;
  using NonVoidElementScaleA = cute::conditional_t<cute::is_void_v<ElementScaleA>, DefScaleType, ElementScaleA>;
  using NonVoidStrideScaleA = cute::conditional_t<cute::is_same_v<InternalStrideScaleA, void>, cute::Stride<_1, int64_t, int64_t>, InternalStrideScaleA>;                                          
  
  using NonVoidElementScaleB = cute::conditional_t<cute::is_void_v<ElementScaleB>, DefScaleType, ElementScaleB>;
  using NonVoidStrideScaleB = cute::conditional_t<cute::is_same_v<InternalStrideScaleB, void>, cute::Stride<_1, int64_t, int64_t>, InternalStrideScaleB>;

  static_assert(sizeof_bits_v<NonVoidElementScaleA> == 8 && sizeof_bits_v<NonVoidElementScaleB> == 8);

  using GmemTiledCopyNonVoidScaleA = typename scale_copy_traits<NonVoidElementScaleA, SG_K / GROUP_K, SG_M>::type;
  using GmemTiledCopyNonVoidScaleB = typename scale_copy_traits<NonVoidElementScaleB, SG_K / GROUP_K, SG_N>::type;

// Conditionally select the TiledCopy type for ScaleA
  using SelectedGmemTiledCopyScaleA = cute::conditional_t<
      cute::is_void_v<GmemTiledCopyScaleA>,
      GmemTiledCopyNonVoidScaleA,
      GmemTiledCopyScaleA
    >;

  using SelectedGmemTiledCopyScaleB = cute::conditional_t<
      cute::is_void_v<GmemTiledCopyScaleB>,
      GmemTiledCopyNonVoidScaleB,
      GmemTiledCopyScaleB
    >;
  
  template<
  class SelectedGmemTiledCopyScale,
  class StrideScale,
  class ElementScale
  >
  struct TiledCopyScaleTraits {
    using traits_load_scale = Copy_Traits<SelectedGmemTiledCopyScale, StrideScale>;
    using atom_load_scale = Copy_Atom<traits_load_scale, ElementScale>;
    using val_layout_load_scale = decltype(make_layout(shape_div(typename traits_load_scale::BlockShape{}, CopyThreadShapeRev{})));
    using Copy_Scale = decltype(make_tiled_copy(atom_load_scale{}, Layout<CopyThreadShapeRev>{}, val_layout_load_scale{}));
  };

  using Copy_ScaleA = typename TiledCopyScaleTraits<SelectedGmemTiledCopyScaleA, InternalStrideScaleA, ElementScaleA>::Copy_Scale;
  using Copy_ScaleB = typename TiledCopyScaleTraits<SelectedGmemTiledCopyScaleB, InternalStrideScaleB, ElementScaleB>::Copy_Scale;

  using TensorMKL = decltype(make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), make_shape(0,0,0), InternalStrideA{}));   //(m, k)
  using TensorNKL = decltype(make_tensor(make_gmem_ptr(static_cast<ElementB const*>(nullptr)), make_shape(0,0,0), InternalStrideB{}));   //(n, k)
  using TensorScaleA = decltype(make_tensor(make_gmem_ptr(static_cast<ElementScaleA const*>(nullptr)), make_shape(0,0,0), InternalStrideScaleA{}));   //(m, scale_k)
  using TensorScaleB = decltype(make_tensor(make_gmem_ptr(static_cast<ElementScaleB const*>(nullptr)), make_shape(0,0,0), InternalStrideScaleB{}));   //(n, scale_k)
  using MainloopTensors = cute::tuple<TensorMKL, TensorNKL, TensorScaleA, TensorScaleB>;
  // Host side kernel arguments
  struct Arguments {
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
    ElementScaleA const** ptr_SA = nullptr;
    StrideScaleA dSA{};
    ElementScaleB const** ptr_SB = nullptr;
    StrideScaleB dSB{};
    // TODO: Current implementation only support group size = 32 ans Set as GROUP_K = 32
    int group_size = GROUP_K;
  };

  using Params = Arguments;

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const &problem_shape,
                          Arguments const &args, void *workspace) {
    (void)workspace;

    auto problem_shape_MNK = repeat_like(typename ProblemShape::UnderlyingProblemShape{}, int32_t(1));;
    auto init_M = get<0>(problem_shape_MNK);
    auto init_N = get<1>(problem_shape_MNK);
    auto init_K = get<2>(problem_shape_MNK);

    return Params{
      args
    };
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape problem_shapes,
      Arguments const& args) {
    constexpr int copy_alignment_bits = 128;
    constexpr int batch_alignment_bits = 512;
    auto problem_shape_MNKL = append<4>(problem_shapes, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;

    constexpr int min_aligned_elements_A = copy_alignment_bits / sizeof_bits<ElementA>::value;
    constexpr int min_aligned_elements_B = copy_alignment_bits / sizeof_bits<ElementB>::value;
    constexpr int min_batch_aligned_elements_A = batch_alignment_bits / sizeof_bits<ElementA>::value;
    constexpr int min_batch_aligned_elements_B = batch_alignment_bits / sizeof_bits<ElementB>::value;
    for (int i = 0; i < problem_shapes.groups(); i++) {
      auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
      auto [M,N,K,L] = problem_shape_MNKL;

      implementable &= cutlass::detail::check_alignment<min_aligned_elements_A>(cute::make_shape(M,K,L), InternalStrideA{});
      implementable &= cutlass::detail::check_alignment<min_aligned_elements_B>(cute::make_shape(N,K,L), InternalStrideB{});

      if (L > 1) {
        implementable &= get<2>(InternalStrideA{}) % min_batch_aligned_elements_A == 0;
        implementable &= get<2>(InternalStrideB{}) % min_batch_aligned_elements_B == 0;
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for XE 2D copy.\n");
    }

    return implementable;
  }

  template <
  int scale_traits_size,
  int scale_traits_num,
  class SelectedGmemTiledCopyScale
  >
  CUTLASS_DEVICE static auto
  make_scale_copy_iterator(int coord, int l_coord, int k_tile_count) {
      return make_tensor(make_inttuple_iter(make_coord(coord, 0, l_coord)),
                         make_layout(make_shape(Int<scale_traits_size>{}, Int<scale_traits_num>{}, _1{}, k_tile_count),
                                     make_stride(E<0>{} * _16{}, E<0>{} * size<1>(typename SelectedGmemTiledCopyScale::BlockShape{}), _0{}, E<1>{} * (SG_K / GROUP_K))));
  }

  /// Perform a subgroup-scoped matrix multiply-accumulate
  template <class FrgTensorD,
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class BlkCoord,
    class LoadTensors
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      BlkCoord const &blk_coord,
      int const &K_start,
      int thread_idx,
      Params const& mainloop,
      LoadTensors const& load_tensors) 
  {
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    // Partition the copying of A and B tiles across the threads
    (void)blk_coord;
    auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(TiledMma{}, get<0>(load_tensors)(_, _, 0));
    auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(TiledMma{}, get<1>(load_tensors)(_, _, 0));

    Copy_ScaleA tiled_copy_scaleA{Copy_ScaleA{}.with(get<2>(load_tensors))};
    Copy_ScaleB tiled_copy_scaleB{Copy_ScaleB{}.with(get<3>(load_tensors))};

    auto thr_copy_a = copy_a.get_slice(thread_idx);
    auto thr_copy_b = copy_b.get_slice(thread_idx);

    // Instantiate the MMA object and get thread slice
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    /* Register fragments for MMA */
    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));

    /* Register fragments for copies */
    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

    /* Partition global tensor (proxies) for copies */
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    
    /* Create prefetch TiledCopy instances */
    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);
      
    auto thr_prefetch_A = prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(thread_idx);

    /* Partition global tensor (proxies) for prefetch */
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    // If IsATransformed, we need modes M_atom, and M_iter from fragment_A layout else we need mode N_iter from fragment_B layout.
    static constexpr auto scaleA_traits_size = decltype(size(typename SelectedGmemTiledCopyScaleA::BlockShape{}))::value / SubgroupSize;
    static constexpr auto scaleA_traits_num = SG_M / size<1>(typename SelectedGmemTiledCopyScaleA::BlockShape{});
    using FragScaleALayout = Layout<Shape<Int<scaleA_traits_size>, Int<scaleA_traits_num>, _1>>;
    Tensor fragment_scaleA = make_tensor<ElementScaleA>(FragScaleALayout{});

    static constexpr int scaleB_traits_size = decltype(size(typename SelectedGmemTiledCopyScaleB::BlockShape{}))::value / SubgroupSize;
    static constexpr int scaleB_traits_num = SG_N / size<1>(typename SelectedGmemTiledCopyScaleB::BlockShape{});
    using FragScaleBLayout = Layout<Shape<Int<scaleB_traits_size>, Int<scaleB_traits_num>, _1>>;
    Tensor fragment_scaleB = make_tensor<ElementScaleB>(FragScaleBLayout{});
    auto [m_idx, n_idx, k_idx, l_idx] = blk_coord;
    const int m_coord = m_idx * BLK_M + (get_sub_group_id() / ATOM_N) * SG_M;
    const int n_coord = n_idx * BLK_N + (get_sub_group_id() % ATOM_N) * SG_N;
    const int l_coord = l_idx;

    auto copy_iter_sA = make_scale_copy_iterator<scaleA_traits_size, scaleA_traits_num, SelectedGmemTiledCopyScaleA>(m_coord, l_coord, k_tile_count);
    auto copy_iter_sB = make_scale_copy_iterator<scaleB_traits_size, scaleB_traits_num, SelectedGmemTiledCopyScaleB>(n_coord, l_coord, k_tile_count);

#define PRINT(x) print(#x ": "); print(x); print("\n");

#if CUTLASS_ENABLE_DEBUG_PRINTS
#define PRINT(x) print(#x ": "); print(x); print("\n");
    if (cute::thread(LOG_THREAD, LOG_GROUP)) {
      print("======================= A: \n");
      PRINT(tAgA);

      PRINT(tCrA);
      PRINT(tArA);
      PRINT(copy_a);
      PRINT(fragment_scaleA);

      print("======================= B: \n");
      PRINT(tBgB);

      PRINT(tCrB);
      PRINT(tBrB);
      PRINT(copy_b);
      PRINT(fragment_scaleB);
      }
#undef PRINT
  #endif

    const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));
    constexpr int barrier_scope = 2;
    int prefetch_k = k_start_idx;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prefetch_k++) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    constexpr int k_reload_factor = cute::max(GROUP_K / BLK_K, 1);

    //
    // Mainloop
    //
    for (int k_tile = k_start_idx; k_tile < k_tile_count + k_start_idx; k_tile++, prefetch_k++) {
      barrier_arrive(barrier_scope);

      // Copy gmem to rmem for the first k_tile
      copy(copy_a, tAgA(_,_,_,k_tile), tArA);
      copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

      copy(tiled_copy_scaleA, copy_iter_sA(_, _, _, k_tile / k_reload_factor), fragment_scaleA);
      copy(tiled_copy_scaleB, copy_iter_sB(_, _, _, k_tile / k_reload_factor), fragment_scaleB);

      if(prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      }

      // reorder
      reorder(tArA, tCrA);
      reorder(tBrB, tCrB);

      using mma_M = Int<decltype(size<1>(tCrA.shape()))::value>;
      using mma_N = Int<decltype(size<1>(tCrB.shape()))::value>;
      using mma_K = Int<decltype(size<2>(tCrA.shape()))::value>;

      using scaleASize = decltype(size(fragment_scaleA));
      using scaleBSize = decltype(size(fragment_scaleB));

      Tensor scaleA = recast<intel::vector_t<ElementScaleA, scaleASize::value * 2>>(make_tensor(fragment_scaleA.data(), Shape<scaleASize>{}));
      Tensor scaleB = recast<intel::vector_t<ElementScaleB, scaleBSize::value * 2>>(make_tensor(fragment_scaleB.data(), Shape<scaleBSize>{}));

      // this gemm_m_iteraions indicate which iteration for m in gemm which is used for select scale date with this offset
      auto gemm_m_iteraions = make_tensor<uint8_t>(Shape<_1>{});
      auto gemm_n_iteraions = make_tensor<uint8_t>(Shape<_1>{});

      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < mma_K{}; k++) {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < mma_N{}; n++) {
          gemm_n_iteraions[n] = n;
          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < mma_M{}; m++) {
            gemm_m_iteraions[m] = m;
            cute::gemm(tiled_mma, make_zip_tensor(tCrA(_, m, k), scaleA, gemm_m_iteraions) , make_zip_tensor(tCrB(_, n, k), scaleB, gemm_n_iteraions), accum(_, m, n));
          }
        }
      }

      barrier_wait(barrier_scope);
    }
  }

  template<typename ProblemShape_MNKL>
  CUTLASS_DEVICE auto update_tensor_shape_stride(
    Params const& mainloop_params,
    int32_t const& next_group,
    ProblemShape_MNKL const& problem_shape_mnkl) {
      const int32_t M = get<0>(problem_shape_mnkl);
      const int32_t N = get<1>(problem_shape_mnkl);
      const int32_t K = get<2>(problem_shape_mnkl);

      auto scale_k = cute::ceil_div(K, GROUP_K);

      ElementA const* ptr_A_curr_batch = static_cast<ElementA const*>(mainloop_params.ptr_A[next_group]);
      ElementB const* ptr_B_curr_batch = static_cast<ElementB const*>(mainloop_params.ptr_B[next_group]);
      ElementSF const* ptr_SFA_curr_batch = static_cast<ElementSF const*>(mainloop_params.ptr_SA[next_group]);
      ElementSF const* ptr_SFB_curr_batch = static_cast<ElementSF const*>(mainloop_params.ptr_SB[next_group]);

      TensorMKL mA = make_tensor(make_gmem_ptr(ptr_A_curr_batch), make_shape(M, K,(int32_t)1), mainloop_params.dA[next_group]);
      TensorNKL mB = make_tensor(make_gmem_ptr(ptr_B_curr_batch), make_shape(N, K,(int32_t)1), mainloop_params.dB[next_group]);
      TensorScaleA mScaleA = make_tensor(make_gmem_ptr(ptr_SFA_curr_batch), make_shape(M, scale_k, (int32_t)1), mainloop_params.dSA[next_group]);
      TensorScaleB mScaleB = make_tensor(make_gmem_ptr(ptr_SFB_curr_batch), make_shape(N, scale_k, (int32_t)1), mainloop_params.dSB[next_group]);
      return cute::make_tuple(mA, mB, mScaleA, mScaleB);
  }  
};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
