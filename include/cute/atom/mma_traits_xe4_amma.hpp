#pragma once

#include <cute/arch/mma_xe4_amma.hpp>
#include <cute/atom/mma_traits_xe4.hpp>

namespace cute {

template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct MMA_Traits<
  XE4_AMMA<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>>
{
  using ValTypeD = d_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  using FrgTypeA = AMMA::smem_desc<a_major, true>;
  using FrgTypeB = AMMA::smem_desc<b_major>;
  using FrgTypeC = AMMA::smem_desc<a_major>;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // No control
  MMAControl ctrl_ {};

  template <bool NoAcc = false,
            class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    auto desc_d = D[0];
    auto desc_a = A[0];
    auto desc_b = B[0];
    auto desc_c = C[0];

    XE4_AMMA<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>::fma(
      traits.ctrl_, desc_d, desc_a, desc_b, desc_c
    );
  }

  template <AMMA::Tracking Method, typename ... Barriers>
  CUTE_HOST_DEVICE constexpr static
  auto
  with(AMMA::TrackMethod<Method> method, MMAControl ctrl, Barriers... barriers) {
    return with(d_type {}, method, ctrl, barriers...);
  }

  template <typename T, AMMA::Tracking Method, typename ... Barriers>
  CUTE_HOST_DEVICE constexpr static
  auto
  with(T, AMMA::TrackMethod<Method>, MMAControl ctrl, Barriers... barriers) {
    if constexpr (Method == AMMA::Tracking::None) {
      return  MMA_Traits<
        XE4_AMMA<T, a_type, b_type, c_type, M, N, K, a_major, b_major>>
        {ctrl};
    } else if constexpr (Method == AMMA::Tracking::GroupSync) {
      return  MMA_Traits<
        XE4_AMMA_GROUPSYNC<T, a_type, b_type, c_type, M, N, K, a_major, b_major>>
        {ctrl};
    } else if constexpr (Method == AMMA::Tracking::D) {
      return  MMA_Traits<
        XE4_AMMA_D<T, a_type, b_type, c_type, M, N, K, a_major, b_major>>
        {ctrl, barriers...};
    } else if constexpr (Method == AMMA::Tracking::AB) {
      return  MMA_Traits<
        XE4_AMMA_AB<T, a_type, b_type, c_type, M, N, K, a_major, b_major>>
        {ctrl, barriers...};
    } else if constexpr (Method == AMMA::Tracking::DB) {
      return  MMA_Traits<
        XE4_AMMA_DB<T, a_type, b_type, c_type, M, N, K, a_major, b_major>>
        {ctrl, barriers...};
    } else if constexpr (Method == AMMA::Tracking::DAB) {
      return  MMA_Traits<
        XE4_AMMA_DAB<T, a_type, b_type, c_type, M, N, K, a_major, b_major>>
        {ctrl, barriers...};
    } else {
      static_assert(dependent_false<AMMA::TrackMethod<Method>>,
          "Unknown abarrier tracking pattern");
    }

    CUTE_GCC_UNREACHABLE;
  }
};

template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct MMA_Traits<
  XE4_AMMA_GROUPSYNC<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>>
{
  using ValTypeD = d_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  using FrgTypeA = AMMA::smem_desc<a_major, true>;
  using FrgTypeB = AMMA::smem_desc<b_major>;
  using FrgTypeC = AMMA::smem_desc<a_major>;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // No control
  MMAControl ctrl_ {};

  template <bool NoAcc = false,
            class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    auto desc_d = D[0];
    auto desc_a = A[0];
    auto desc_b = B[0];
    auto desc_c = C[0];

    XE4_AMMA_GROUPSYNC<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>::fma(
      traits.ctrl_, desc_d, desc_a, desc_b, desc_c
    );
  }
};

template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct MMA_Traits<
  XE4_AMMA_D<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>>
{
  using ValTypeD = d_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  using FrgTypeA = AMMA::smem_desc<a_major, true>;
  using FrgTypeB = AMMA::smem_desc<b_major>;
  using FrgTypeC = AMMA::smem_desc<a_major>;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // No control
  MMAControl ctrl_ {};

  uint64_t* d_barrier_;

  template <bool NoAcc = false,
            class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    auto desc_d = D[0];
    auto desc_a = A[0];
    auto desc_b = B[0];
    auto desc_c = C[0];

    XE4_AMMA_D<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>::fma(
      traits.ctrl_, desc_d, desc_a, desc_b, desc_c, traits.d_barrier_
    );
  }
};

template <class d_type, class a_type, class b_type, class c_type,
  int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct MMA_Traits<
  XE4_AMMA_DB<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>>
{
  using ValTypeD = d_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  using FrgTypeA = AMMA::smem_desc<a_major, true>;
  using FrgTypeB = AMMA::smem_desc<b_major>;
  using FrgTypeC = AMMA::smem_desc<a_major>;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape <_1, Shape <Int<M>, Int<K>>>,
    Stride<_0, Stride<    _1, Int<M>>>>;
  using BLayout = Layout<Shape <_1, Shape <Int<N>, Int<K>>>,
    Stride<_0, Stride<    _1, Int<N>>>>;
  using CLayout = Layout<Shape <_1, Shape <Int<M>, Int<N>>>,
    Stride<_0, Stride<    _1, Int<M>>>>;

  // No control
  MMAControl ctrl_{};

  uint64_t* d_barrier_;
  uint64_t* b_barrier_;

  // Dummy mask
  uint32_t mask_[2];

  template <bool NoAcc = false,
    class TD, class DLayout,
    class TA, class ALayout,
    class TB, class BLayout,
    class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
    void
    mma_unpack(MMA_Traits          const& traits,
      Tensor<TD, DLayout>& D,
      Tensor<TA, ALayout> const& A,
      Tensor<TB, BLayout> const& B,
      Tensor<TC, CLayout> const& C)
  {
    auto desc_d = D[0];
    auto desc_a = A[0];
    auto desc_b = B[0];
    auto desc_c = C[0];

    XE4_AMMA_DB<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>::fma(
      traits.ctrl_, desc_d, desc_a, desc_b, desc_c,
      traits.d_barrier_, traits.b_barrier_
    );
  }
};

template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct MMA_Traits<
  XE4_AMMA_AB<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>>
{
  using ValTypeD = d_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  using FrgTypeA = AMMA::smem_desc<a_major, true>;
  using FrgTypeB = AMMA::smem_desc<b_major>;
  using FrgTypeC = AMMA::smem_desc<a_major>;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // No control
  MMAControl ctrl_ {};

  uint64_t* a_barrier_;
  uint64_t* b_barrier_;

  // Dummy mask
  uint32_t mask0_, mask1_;

  template <bool NoAcc = false,
            class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    auto desc_d = D[0];
    auto desc_a = A[0];
    auto desc_b = B[0];
    auto desc_c = C[0];

    XE4_AMMA_AB<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>::fma(
      traits.ctrl_, desc_d, desc_a, desc_b, desc_c, traits.a_barrier_, traits.b_barrier_
    );
  }
};

template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct MMA_Traits<
  XE4_AMMA_DAB<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>>
{
  using ValTypeD = d_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  using FrgTypeA = AMMA::smem_desc<a_major, true>;
  using FrgTypeB = AMMA::smem_desc<b_major>;
  using FrgTypeC = AMMA::smem_desc<a_major>;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // No control
  MMAControl ctrl_ {};

  uint64_t* d_barrier_;
  uint64_t* a_barrier_;
  uint64_t* b_barrier_;

  // Dummy mask
  uint32_t mask0_, mask1_;

  template <bool NoAcc = false,
            class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    auto desc_d = D[0];
    auto desc_a = A[0];
    auto desc_b = B[0];
    auto desc_c = C[0];

    XE4_AMMA_DAB<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>::fma(
      traits.ctrl_, desc_d, desc_a, desc_b, desc_c, traits.d_barrier_,
      traits.a_barrier_, traits.b_barrier_
    );
  }
};

template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct MMA_Traits<
  XE4_AMMA_AB_CLUSTER<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>>
{
  using ValTypeD = d_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  using FrgTypeA = AMMA::smem_desc<a_major, true>;
  using FrgTypeB = AMMA::smem_desc<b_major>;
  using FrgTypeC = AMMA::smem_desc<a_major>;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // No control
  MMAControl ctrl_ {};

  uint64_t* a_barrier_;
  uint64_t* b_barrier_;

  // Dummy mask
  uint32_t a_mask_;
  uint32_t b_mask_;

  template <bool NoAcc = false,
            class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    auto desc_d = D[0];
    auto desc_a = A[0];
    auto desc_b = B[0];
    auto desc_c = C[0];

    XE4_AMMA_AB_CLUSTER<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>::fma(
      traits.ctrl_, desc_d, desc_a, desc_b, desc_c, traits.a_barrier_, traits.b_barrier_,
      traits.a_mask_, traits.b_mask_
    );
  }

  template <AMMA::Tracking Method, typename ... Barriers>
  CUTE_HOST_DEVICE constexpr static
  auto
  with(AMMA::TrackMethod<Method> method, MMAControl ctrl, Barriers... barriers) {
    return with(d_type {}, method, ctrl, barriers...);
  }

  template <typename T, AMMA::Tracking Method, typename ... Barriers>
  CUTE_HOST_DEVICE constexpr static
  auto
  with(T, AMMA::TrackMethod<Method>, MMAControl ctrl, Barriers... barriers) {
    if constexpr (Method == AMMA::Tracking::AB) {
      return  MMA_Traits<
        XE4_AMMA_AB_CLUSTER<T, a_type, b_type, c_type, M, N, K, a_major, b_major>>
        {ctrl, barriers...};
    } else if constexpr (Method == AMMA::Tracking::DAB) {
      return  MMA_Traits<
        XE4_AMMA_DAB_CLUSTER<T, a_type, b_type, c_type, M, N, K, a_major, b_major>>
        {ctrl, barriers...};
    } else if constexpr (Method == AMMA::Tracking::DB) {
      static_assert(sizeof...(barriers) == 4, "DB cluster tracking expects {d_barrier, b_barrier, unused_a_mask, b_mask} (a_mask is a placeholder). ");
      return  MMA_Traits<
        XE4_AMMA_DB_CLUSTER<T, a_type, b_type, c_type, M, N, K, a_major, b_major>>
        {ctrl, barriers...};
    } else {
      static_assert(dependent_false<AMMA::TrackMethod<Method>>,
          "Unknown abarrier tracking pattern");
    }

    CUTE_GCC_UNREACHABLE;
  }
};

template <class d_type, class a_type, class b_type, class c_type,
         int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct MMA_Traits<
  XE4_AMMA_DAB_CLUSTER<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>>
{
  using ValTypeD = d_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  using FrgTypeA = AMMA::smem_desc<a_major, true>;
  using FrgTypeB = AMMA::smem_desc<b_major>;
  using FrgTypeC = AMMA::smem_desc<a_major>;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,
                         Stride<_0,Stride<    _1,Int<N>>>>;
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,
                         Stride<_0,Stride<    _1,Int<M>>>>;

  // No control
  MMAControl ctrl_ {};

  uint64_t* d_barrier_;
  uint64_t* a_barrier_;
  uint64_t* b_barrier_;

  uint32_t a_mask_;
  uint32_t b_mask_;

  template <bool NoAcc = false,
            class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    auto desc_d = D[0];
    auto desc_a = A[0];
    auto desc_b = B[0];
    auto desc_c = C[0];

    XE4_AMMA_DAB_CLUSTER<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>::fma(
      traits.ctrl_, desc_d, desc_a, desc_b, desc_c,
      traits.d_barrier_, traits.a_barrier_, traits.b_barrier_,
      traits.a_mask_, traits.b_mask_
    );
  }
};

// ===============================================================================
// MMA_Traits for Block-Scaled AMMA FP4 Operations
//
// These traits add SF (scale factor) descriptor support:
//   - FrgTypeSFA / FrgTypeSFB = AMMA::smem_sf_desc (Type-3 descriptor iterators)
//   - ValTypeSF for the scale factor element type
//   - sf_a_ / sf_b_ member descriptors passed to fma() via mma_unpack()
//   - with() method that accepts SF descriptors
// ===============================================================================

// Common type members and layout definitions shared by all block-scaled traits.
// Captured via a macro to avoid repetition across the four tracking variants.
#define CUTE_AMMA_BS_TRAIT_COMMON_TYPES(d, a, b, c, sfa, sfb)   \
  using ValTypeD = d;                                            \
  using ValTypeA = a;                                            \
  using ValTypeB = b;                                            \
  using ValTypeC = c;                                            \
  using ValTypeSFA = sfa;                                        \
  using ValTypeSFB = sfb;                                        \
  using FrgTypeA   = AMMA::smem_desc<a_major, true>;             \
  using FrgTypeB   = AMMA::smem_desc<b_major>;                   \
  using FrgTypeC   = AMMA::smem_desc<a_major>;                   \
  using FrgTypeSFA = AMMA::smem_sf_desc;                         \
  using FrgTypeSFB = AMMA::smem_sf_desc;                         \
  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;               \
  using ThrID = Layout<_1>;                                      \
  using ALayout = Layout<Shape <_1,Shape <Int<M>,Int<K>>>,       \
                         Stride<_0,Stride<    _1,Int<M>>>>;      \
  using BLayout = Layout<Shape <_1,Shape <Int<N>,Int<K>>>,       \
                         Stride<_0,Stride<    _1,Int<N>>>>;      \
  using CLayout = Layout<Shape <_1,Shape <Int<M>,Int<N>>>,       \
                         Stride<_0,Stride<    _1,Int<M>>>>;

// ----- Block-scaled AMMA: No barrier tracking -----
template <class d_type, class a_type, class b_type, class c_type,
         class sf_a_type, class sf_b_type,
         int M, int N, int K, int VSA, int VSB, AMMA::Major a_major, AMMA::Major b_major,
         AMMA::BlockScaleMode Mode>
struct MMA_Traits<
  XE4_AMMA_BlockScaled<d_type, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
{
  CUTE_AMMA_BS_TRAIT_COMMON_TYPES(d_type, a_type, b_type, c_type, sf_a_type, sf_b_type)

  MMAControl ctrl_ {};
  uint32_t sf_a_ {};
  uint32_t sf_b_ {};

  template <bool NoAcc = false,
            class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    XE4_AMMA_BlockScaled<d_type, a_type, b_type, c_type, sf_a_type, sf_b_type,
                M, N, K, VSA, VSB, a_major, b_major, Mode>::fma(
      traits.ctrl_, D[0], A[0], B[0], C[0], traits.sf_a_, traits.sf_b_
    );
  }

  template <AMMA::Tracking Method, typename ... Args>
  CUTE_HOST_DEVICE constexpr static
  auto
  with(AMMA::TrackMethod<Method> method, MMAControl ctrl,
       uint32_t sf_a, uint32_t sf_b, Args... args) {
    return with(d_type{}, method, ctrl, sf_a, sf_b, args...);
  }

  template <typename T, AMMA::Tracking Method, typename ... Args>
  CUTE_HOST_DEVICE constexpr static
  auto
  with(T, AMMA::TrackMethod<Method>, MMAControl ctrl,
       uint32_t sf_a, uint32_t sf_b, Args... args) {
    if constexpr (Method == AMMA::Tracking::None) {
      return MMA_Traits<
        XE4_AMMA_BlockScaled<T, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
        {ctrl, sf_a, sf_b};
    } else if constexpr (Method == AMMA::Tracking::D) {
      return MMA_Traits<
        XE4_AMMA_BlockScaled_D<T, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
        {ctrl, sf_a, sf_b, args...};
    } else if constexpr (Method == AMMA::Tracking::AB) {
      if constexpr (sizeof...(args) > 2) {
        return MMA_Traits<
          XE4_AMMA_BlockScaled_AB_CLUSTER<T, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
          {ctrl, sf_a, sf_b, args...};
      } else {
        return MMA_Traits<
          XE4_AMMA_BlockScaled_AB<T, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
          {ctrl, sf_a, sf_b, args...};
      }
    } else if constexpr (Method == AMMA::Tracking::DAB) {
      if constexpr (sizeof...(args) > 3) {
        return MMA_Traits<
          XE4_AMMA_BlockScaled_DAB_CLUSTER<T, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
          {ctrl, sf_a, sf_b, args...};
      } else {
        return MMA_Traits<
          XE4_AMMA_BlockScaled_DAB<T, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
          {ctrl, sf_a, sf_b, args...};
      }
    } else {
      static_assert(dependent_false<AMMA::TrackMethod<Method>>,
          "Unknown barrier tracking pattern");
    }

    CUTE_GCC_UNREACHABLE;
  }
};

// ----- Block-scaled AMMA: D-barrier tracking -----
template <class d_type, class a_type, class b_type, class c_type,
         class sf_a_type, class sf_b_type,
         int M, int N, int K, int VSA, int VSB, AMMA::Major a_major, AMMA::Major b_major,
         AMMA::BlockScaleMode Mode>
struct MMA_Traits<
  XE4_AMMA_BlockScaled_D<d_type, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
{
  CUTE_AMMA_BS_TRAIT_COMMON_TYPES(d_type, a_type, b_type, c_type, sf_a_type, sf_b_type)

  MMAControl ctrl_ {};
  uint32_t sf_a_ {};
  uint32_t sf_b_ {};
  uint64_t* d_barrier_;

  template <bool NoAcc = false,
            class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    XE4_AMMA_BlockScaled_D<d_type, a_type, b_type, c_type, sf_a_type, sf_b_type,
                  M, N, K, VSA, VSB, a_major, b_major, Mode>::fma(
      traits.ctrl_, D[0], A[0], B[0], C[0], traits.sf_a_, traits.sf_b_,
      traits.d_barrier_
    );
  }
};

// ----- Block-scaled AMMA: A+B-barrier tracking -----
template <class d_type, class a_type, class b_type, class c_type,
         class sf_a_type, class sf_b_type,
         int M, int N, int K, int VSA, int VSB, AMMA::Major a_major, AMMA::Major b_major,
         AMMA::BlockScaleMode Mode>
struct MMA_Traits<
  XE4_AMMA_BlockScaled_AB<d_type, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
{
  CUTE_AMMA_BS_TRAIT_COMMON_TYPES(d_type, a_type, b_type, c_type, sf_a_type, sf_b_type)

  MMAControl ctrl_ {};
  uint32_t sf_a_ {};
  uint32_t sf_b_ {};
  uint64_t* a_barrier_;
  uint64_t* b_barrier_;

  template <bool NoAcc = false,
            class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    XE4_AMMA_BlockScaled_AB<d_type, a_type, b_type, c_type, sf_a_type, sf_b_type,
                   M, N, K, VSA, VSB, a_major, b_major, Mode>::fma(
      traits.ctrl_, D[0], A[0], B[0], C[0], traits.sf_a_, traits.sf_b_,
      traits.a_barrier_, traits.b_barrier_
    );
  }
};

// ----- Block-scaled AMMA: D+A+B-barrier tracking -----
template <class d_type, class a_type, class b_type, class c_type,
         class sf_a_type, class sf_b_type,
         int M, int N, int K, int VSA, int VSB, AMMA::Major a_major, AMMA::Major b_major,
         AMMA::BlockScaleMode Mode>
struct MMA_Traits<
  XE4_AMMA_BlockScaled_DAB<d_type, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
{
  CUTE_AMMA_BS_TRAIT_COMMON_TYPES(d_type, a_type, b_type, c_type, sf_a_type, sf_b_type)

  MMAControl ctrl_ {};
  uint32_t sf_a_ {};
  uint32_t sf_b_ {};
  uint64_t* d_barrier_;
  uint64_t* a_barrier_;
  uint64_t* b_barrier_;

  template <bool NoAcc = false,
            class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    XE4_AMMA_BlockScaled_DAB<d_type, a_type, b_type, c_type, sf_a_type, sf_b_type,
                    M, N, K, VSA, VSB, a_major, b_major, Mode>::fma(
      traits.ctrl_, D[0], A[0], B[0], C[0], traits.sf_a_, traits.sf_b_,
      traits.d_barrier_, traits.a_barrier_, traits.b_barrier_
    );
  }
};

// ----- Block-scaled AMMA: A+B-barrier tracking, cluster (multicast masks) -----
template <class d_type, class a_type, class b_type, class c_type,
         class sf_a_type, class sf_b_type,
         int M, int N, int K, int VSA, int VSB, AMMA::Major a_major, AMMA::Major b_major,
         AMMA::BlockScaleMode Mode>
struct MMA_Traits<
  XE4_AMMA_BlockScaled_AB_CLUSTER<d_type, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
{
  CUTE_AMMA_BS_TRAIT_COMMON_TYPES(d_type, a_type, b_type, c_type, sf_a_type, sf_b_type)

  MMAControl ctrl_ {};
  uint32_t sf_a_ {};
  uint32_t sf_b_ {};
  uint64_t* a_barrier_;
  uint64_t* b_barrier_;
  uint32_t a_mask_;
  uint32_t b_mask_;

  template <bool NoAcc = false,
            class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    XE4_AMMA_BlockScaled_AB_CLUSTER<d_type, a_type, b_type, c_type, sf_a_type, sf_b_type,
                   M, N, K, VSA, VSB, a_major, b_major, Mode>::fma(
      traits.ctrl_, D[0], A[0], B[0], C[0], traits.sf_a_, traits.sf_b_,
      traits.a_barrier_, traits.b_barrier_,
      traits.a_mask_, traits.b_mask_
    );
  }

  template <AMMA::Tracking Method, typename ... Args>
  CUTE_HOST_DEVICE constexpr static
  auto
  with(AMMA::TrackMethod<Method> method, MMAControl ctrl,
       uint32_t sf_a, uint32_t sf_b, Args... args) {
    return with(d_type{}, method, ctrl, sf_a, sf_b, args...);
  }

  template <typename T, AMMA::Tracking Method, typename ... Args>
  CUTE_HOST_DEVICE constexpr static
  auto
  with(T, AMMA::TrackMethod<Method>, MMAControl ctrl,
       uint32_t sf_a, uint32_t sf_b, Args... args) {
    if constexpr (Method == AMMA::Tracking::AB) {
      return MMA_Traits<
        XE4_AMMA_BlockScaled_AB_CLUSTER<T, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
        {ctrl, sf_a, sf_b, args...};
    } else if constexpr (Method == AMMA::Tracking::DAB) {
      return MMA_Traits<
        XE4_AMMA_BlockScaled_DAB_CLUSTER<T, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
        {ctrl, sf_a, sf_b, args...};
    } else {
      static_assert(dependent_false<AMMA::TrackMethod<Method>>,
          "Unknown abarrier tracking pattern");
    }

    CUTE_GCC_UNREACHABLE;
  }
};

template <class d_type, class a_type, class b_type, class c_type,
  int M, int N, int K, AMMA::Major a_major, AMMA::Major b_major>
struct MMA_Traits<
  XE4_AMMA_DB_CLUSTER<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>>
{
  using ValTypeD = d_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;

  using FrgTypeA = AMMA::smem_desc<a_major, true>;
  using FrgTypeB = AMMA::smem_desc<b_major>;
  using FrgTypeC = AMMA::smem_desc<a_major>;

  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape <_1, Shape <Int<M>, Int<K>>>,
    Stride<_0, Stride<    _1, Int<M>>>>;
  using BLayout = Layout<Shape <_1, Shape <Int<N>, Int<K>>>,
    Stride<_0, Stride<    _1, Int<N>>>>;
  using CLayout = Layout<Shape <_1, Shape <Int<M>, Int<N>>>,
    Stride<_0, Stride<    _1, Int<M>>>>;

  // No control
  MMAControl ctrl_{};

  uint64_t* d_barrier_;
  uint64_t* b_barrier_;

  // Dummy mask
  uint32_t a_mask_;
  uint32_t b_mask_;

  template <bool NoAcc = false,
    class TD, class DLayout,
    class TA, class ALayout,
    class TB, class BLayout,
    class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
    void
    mma_unpack(MMA_Traits          const& traits,
      Tensor<TD, DLayout>& D,
      Tensor<TA, ALayout> const& A,
      Tensor<TB, BLayout> const& B,
      Tensor<TC, CLayout> const& C)
  {
    auto desc_d = D[0];
    auto desc_a = A[0];
    auto desc_b = B[0];
    auto desc_c = C[0];

    XE4_AMMA_DB_CLUSTER<d_type, a_type, b_type, c_type, M, N, K, a_major, b_major>::fma(
      traits.ctrl_, desc_d, desc_a, desc_b, desc_c,
      traits.d_barrier_, traits.b_barrier_,
      traits.b_mask_
    );
  }
};

// ----- Block-scaled AMMA: D+A+B-barrier tracking, cluster (multicast masks) -----
template <class d_type, class a_type, class b_type, class c_type,
         class sf_a_type, class sf_b_type,
         int M, int N, int K, int VSA, int VSB, AMMA::Major a_major, AMMA::Major b_major,
         AMMA::BlockScaleMode Mode>
struct MMA_Traits<
  XE4_AMMA_BlockScaled_DAB_CLUSTER<d_type, a_type, b_type, c_type, sf_a_type, sf_b_type, M, N, K, VSA, VSB, a_major, b_major, Mode>>
{
  CUTE_AMMA_BS_TRAIT_COMMON_TYPES(d_type, a_type, b_type, c_type, sf_a_type, sf_b_type)

  MMAControl ctrl_ {};
  uint32_t sf_a_ {};
  uint32_t sf_b_ {};
  uint64_t* d_barrier_;
  uint64_t* a_barrier_;
  uint64_t* b_barrier_;
  uint32_t a_mask_;
  uint32_t b_mask_;

  template <bool NoAcc = false,
            class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr friend
  void
  mma_unpack(MMA_Traits          const& traits,
             Tensor<TD, DLayout>      & D,
             Tensor<TA, ALayout> const& A,
             Tensor<TB, BLayout> const& B,
             Tensor<TC, CLayout> const& C)
  {
    XE4_AMMA_BlockScaled_DAB_CLUSTER<d_type, a_type, b_type, c_type, sf_a_type, sf_b_type,
                    M, N, K, VSA, VSB, a_major, b_major, Mode>::fma(
      traits.ctrl_, D[0], A[0], B[0], C[0], traits.sf_a_, traits.sf_b_,
      traits.d_barrier_, traits.a_barrier_, traits.b_barrier_,
      traits.a_mask_, traits.b_mask_
    );
  }
};

#undef CUTE_AMMA_BS_TRAIT_COMMON_TYPES

} // namespace cute
