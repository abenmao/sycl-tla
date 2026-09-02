/*!
  \file xe35_chunk_gated_delta_rule_kernels.hpp
  \brief Xe35 SYCL device kernels for the four-stage chunkwise Gated DeltaNet
         (GDN) attention forward pass, plus the host-side kernel_launcher.

  Algorithm overview (one chunk of C=64 tokens at a time, per head; C=64 is the
  baseline Xe2 (BMG) value inherited from the upstream port and not retuned for
  Xe3 -- see cutlass::gdn::kChunkSize):
    Stage 1 -- chunk_compute_A_o2 (fuses the former chunk_prepare):
       Per (chunk, k_head): per v_head compute the per-token cumulative gate
       a[t] = cumsum_t( softplus(a+dt_bias)*(-exp(A_log)) ) (fused from chunk_prepare),
       then L2-normalize Q (with 1/sqrt(D) scale) and K in-place.
       Build the lower-triangular transition matrix
      L[m,n] = (K_m·K_n)*exp(a[m]-a[n])*b[m] (A), with L[m,m]=1, L[m,n]=0 for m<n,
      and the decay-gated O2[m,n] = (Q_m·K_n)*exp(a[m]-a[n]) for m>=n. L encodes
      within-chunk token mixing. The cumsum gate is also written back to gmem for
      the wu/fwd_o stages below.
    Stage 2 -- chunk_inverse:
      Invert L in-place (lower-triangular, so a block forward-substitution suffices).
      Done by the DPAS-accelerated `chunk_inverse_opt_kernel`.
    Stage 3 -- chunk_compute_wu:
      U = L^-1 * V * diag(b)  (V projection)
      W = L^-1 * K * diag(exp(a)*b)  (K-weighted update, only when a prior state exists)
    Stage 4 -- chunk_fwd_o:
      O = Q*S^T*exp(g) + O2*U   (inter-chunk + intra-chunk output)
      S_{out} = exp(g_last)*S_prev + U^T * K_scaled  (SSM state update)

  Placed in namespace cutlass::gdn::detail; called by kernel_launcher() which
  is exposed through xe35_chunk_gated_delta_rule_launch.hpp.
*/

#pragma once

#include <sycl/sycl.hpp>

#include "xe35_chunk_gated_delta_rule_gemm.hpp"
// Public API header — lightweight (cutlass/cutlass.h + sycl/sycl.hpp only,
// no device code), so it is safe to pull into this device-only path. Provides
// cutlass::gdn::kChunkSize as the single source of truth for the chunk size.
#include "gdn_attention/xe35_chunk_gated_delta_rule.hpp"
// EventManager singleton: when CUTLASS_SYCL_PROFILING_ENABLED is set, each
// kernel's sycl::event is registered here so GPU_Clock/SYCLTimer can sum the
// device-side command_end - command_start spans. addEvent() is a guarded
// no-op when profiling is off, so the calls below are unconditional.
#include "cutlass/util/sycl_event_manager.hpp"

namespace cutlass::gdn::detail {

using namespace cute;

static constexpr int MaxThreadsPerXeCore = 512;
/* Contiguous elements per lane in compute_A_o2's norm loop. Sets the gmem
 * access width; raising it reassociates the fp32 L2 sum. */
static constexpr int elem_per_item = 2;
static constexpr int sub_group_size = 16;
static_assert(
    64 % elem_per_item == 0,
    "elem_per_item must divide the 64-element head_k_dim granularity");
static constexpr float eps = 0.000001f;
/* How much of a q/k row chunk_compute_A_o2 keeps in registers. Constant, not
 * head_k_dim, because a runtime-indexed array spills; excess is re-read. */
static constexpr int kNormCacheHeadKDim = 128;
/* Single source of truth: the device-side chunk size is the public
 * cutlass::gdn::kChunkSize (defined in xe35_chunk_gated_delta_rule.hpp, included
 * above). Aliased here so the device kernels can keep using the short name.
 * Its value (64) is the baseline Xe2 (BMG) choice, not retuned
 * for Xe3 (CRI).
 */
static constexpr int chunk_size = cutlass::gdn::kChunkSize;

/* GEMM tile policies: WGTile = (M,N,K) per work-group; SGLayout = sub-group
 * arrangement within the work-group.  Each kernel stage picks one policy to
 * control register pressure vs. ILP trade-off. */
struct chunk_gemm_policy_64x64x32_2x1 {
  using WGTile = Shape<_64, _64, _32>;
  using SGLayout = Layout<Shape<_2, _1, _1>, Stride<_1, _1, _0>>;
};

struct chunk_gemm_policy_64x64x32_2x2 {
  using WGTile = Shape<_64, _64, _32>;
  using SGLayout = Layout<Shape<_2, _2, _1>, Stride<_2, _1, _0>>;
};

struct chunk_gemm_policy_64x64x32_4x2 {
  using WGTile = Shape<_64, _64, _32>;
  using SGLayout = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
};

struct chunk_gemm_policy_64x64x64_4x2 {
  using WGTile = Shape<_64, _64, _64>;
  using SGLayout = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
};

struct chunk_gemm_policy_16x16x16 {
  using WGTile = Shape<_16, _16, _16>;
  using SGLayout = Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>;
};

using chunk_gemm_policy_compute_A_O2 = chunk_gemm_policy_64x64x32_4x2;
using chunk_gemm_policy_inverse = chunk_gemm_policy_16x16x16;
using chunk_gemm_policy_compute_wu = chunk_gemm_policy_64x64x32_4x2;
using chunk_gemm_policy_fwd_o = chunk_gemm_policy_64x64x64_4x2;

/* One lane's contiguous q/k elements. gmem is read and written as word_t, so
 * the access width is one message wide by construction. */
template <class T, int N>
struct norm_pack {
  using word_t = cute::uint_byte_t<sizeof(T) * N>;
  T e[N];
};

CUTE_DEVICE float
act_softplus(float& x, float beta = 1.0f, float threshold = 20.0f) {
  if (beta * x < threshold) {
    return sycl::log(1.0f + sycl::native::exp(beta * x)) / beta;
  } else
    return x;
}

/* Per (chunk, k_head) pair, it L2-normalizes that head's Q and K rows in-place
 * and then issues two 64×64 GEMMs with the same K operand in K_tensor,
 * writing to two distinct destinations:
 *
 *   A[v_head, chunk, :]  <- masked  K · K^T          (lower-tri L, fed to inverse)
 *   o2[v_head, chunk, :] <- masked  Q · K^T          (decay-gated O2, fed to fwd_o)
 *
 * The GEMMs run once per kv_head_id and the result is reused across the kv_ratio
 * v-heads in that group; the inner loop only applies the per-v_head masks and
 * stores to the two destinations. K-side tile loads from the second GEMM hit
 * the L1/L2 cache populated by the first. Pipeline change: this kernel runs
 * BEFORE chunk_inverse, so o2 must live in its own buffer (compute_wu still
 * overwrites A with U/W intermediates by reading the inverted L).
 *
 * Mask differences mirror the originals:
 *   compute_A : m>n -> *= exp(g[m]-g[n]) * b[m]; m==n -> 1; m<n -> 0
 *   compute_o2: m>=n -> *= exp(g[m]-g[n]);                  m<n -> 0
 */
template <typename T, class TiledMMA>
CUTE_DEVICE void chunk_compute_A_o2_kernel(
    const sycl::local_accessor<float, 1>& slm_mem_const,
    T* A,
    T* o2,
    T* q,
    T* k,
    const float* b,
    float* a,
    const float* A_log,
    const T* dt_bias,
    const int* query_start_loc,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const int wg_id = item.get_group(1);
  const int wgs_total = item.get_group_range(1);

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);

  static constexpr auto ATOM_M =
      get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N =
      get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

  static constexpr auto SG_M = tile_m / ATOM_M;  // BLK_M / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;  // BLK_N / ATOM_N;

  /* This WG hosts groups_per_wg co-resident groups; a *group* is the
   * ATOM_M*ATOM_N-subgroup cooperative unit that owns one (chunk, k_head). On
   * Xe3p (MaxThreadsPerXeCore=512, lanes_per_group=size(mma)=128) that is 4
   * groups/WG. All group-local ids below derive from the raw WG-linear id. */
  static constexpr int sgs_per_group = ATOM_M * ATOM_N;
  static constexpr int lanes_per_group = sgs_per_group * sub_group_size;
  /* WG size is MaxThreadsPerXeCore; exact divisibility guarantees no remainder
   * lanes, whose group_id would exceed groups_per_wg and index past the SLM. */
  static_assert(
      MaxThreadsPerXeCore % lanes_per_group == 0,
      "MaxThreadsPerXeCore must be an exact multiple of lanes_per_group");
  const int local_id_raw = item.get_local_linear_id();
  const int local_range_raw = item.get_local_range().size();
  const int groups_per_wg = local_range_raw / lanes_per_group;
  const int group_id = local_id_raw / lanes_per_group;
  const int local_id = local_id_raw % lanes_per_group;

  auto sg = item.get_sub_group();
  const int sg_id = sg.get_group_linear_id() % sgs_per_group;
  const int sg_range = sgs_per_group;
  const int sg_local_id = sg.get_local_linear_id();

  float* slm_mem = static_cast<float*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());

  auto thr_mma = mma.get_slice(local_id);

  const int sg_local_m_coord = sg_id / ATOM_N;
  const int sg_local_n_coord = sg_id % ATOM_N;
  int m_sg_start = sg_local_m_coord * SG_M;
  int n_sg_start = sg_local_n_coord * SG_N;

  const int kv_ratio = num_v_heads / num_k_heads;
  const float q_scale = 1.0f / sycl::sqrt(static_cast<float>(head_k_dim));

  /* Per-group SLM slot: this group's kv_ratio cumsum gates (v_head-major,
   * chunk_size floats each). The prolog fills all kv_ratio v_heads; Phase 3
   * reads slot g_idx directly — no gmem readback. */
  float* group_slm_ptr = slm_mem + group_id * kv_ratio * chunk_size;

  /* Total chunks across all batches (each batch is padded to a whole number of
   * chunks in the virtual seqlen), one work item per (chunk, k_head) pair.
   * kv_head_id = w % num_k_heads keeps consecutive groups on the same chunk so
   * co-resident groups reuse that chunk's K/Q rows in L1. */
  int total_chunks_all = 0;
  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const int seq_len = query_start_loc[batch_id + 1] - query_start_loc[batch_id];
    total_chunks_all += (seq_len + chunk_size - 1) / chunk_size;
  }
  const int total_work = total_chunks_all * num_k_heads;
  const int total_groups = wgs_total * groups_per_wg;
  const int global_group_id = wg_id * groups_per_wg + group_id;

  /* Persistent loop. All groups in a WG run the same num_iters (they share
   * WG-scope barriers); trailing groups with no work iterate with
   * has_work=false — skipping gmem / DPAS but still arriving at every
   * unconditional barrier so busy siblings never stall. */
  const int num_iters = (total_work + total_groups - 1) / total_groups;
  for (int it = 0; it < num_iters; ++it) {
    const int w = global_group_id + it * total_groups;
    const bool has_work = (w < total_work);

    /* Safe placeholders keep idle groups' address math in-bounds. */
    int kv_head_id = 0;
    int chunk_start_offset = 0;
    int current_chunk_size = 0;
    if (has_work) {
      kv_head_id = w % num_k_heads;
      const int chunk_global = w / num_k_heads;

      /* Decode the global chunk index into its (batch, local chunk) so the
       * partial-chunk tail can be sized. */
      int seq_len = 0;
      int local_chunk_idx = chunk_global;
      for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        seq_len = query_start_loc[batch_id + 1] - query_start_loc[batch_id];
        const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
        if (local_chunk_idx < current_chunks) break;
        local_chunk_idx -= current_chunks;
      }
      chunk_start_offset = chunk_global * chunk_size;
      current_chunk_size = chunk_size;
      if ((local_chunk_idx + 1) * chunk_size > seq_len) {
        current_chunk_size = seq_len - local_chunk_idx * chunk_size;
      }
    }

    /* Fused Stage-1 cumsum gate, hoisted so its fence folds into the barrier
     * below. v_heads striped one-per-sub-group keep each whole-chunk scan
     * within one sub-group:
     *   a[t] = cumsum_{i<=t} softplus(a[i]+dt_bias)*(-exp(A_log))
     * Writes each token's running total to BOTH gmem a[] (wu/fwd_o read it)
     * and this v_head's SLM gate slot (the mask below reads it there — no
     * gmem read-back). SLM tail past current_chunk_size is zero-padded so
     * exp(0)=1 on those rows/cols. */
    if (has_work) {
      const int vh_lo = kv_head_id * kv_ratio;
      const int vh_hi = vh_lo + kv_ratio;
      for (int vh = vh_lo + sg_id; vh < vh_hi; vh += sg_range) {
        const float A_log_exp_h = -sycl::native::exp(A_log[vh]);
        const float dt_bias_h = static_cast<float>(dt_bias[vh]);
        float* vh_slm_ptr = group_slm_ptr + (vh - vh_lo) * chunk_size;
        constexpr int local_num = chunk_size / sub_group_size;
        float g_local[local_num] = {};
        float g_local_sum = 0.0f;
        CUTE_UNROLL
        for (int c = 0; c < local_num; ++c) {
          g_local[c] =
              a[(chunk_start_offset + sg_local_id * local_num + c) +
                vh * total_virtual_seqlen];
        }
        CUTE_UNROLL
        for (int c = 0; c < local_num; ++c) {
          float a_h = g_local[c] + dt_bias_h;
          a_h = act_softplus(a_h) * A_log_exp_h;
          g_local[c] = a_h;
          g_local_sum += a_h;
        }
        g_local_sum = sycl::inclusive_scan_over_group(
            sg, g_local_sum, sycl::plus<float>());
        CUTE_UNROLL
        for (int c = local_num - 1; c >= 0; --c) {
          const int tok = sg_local_id * local_num + c;
          a[(chunk_start_offset + tok) + vh * total_virtual_seqlen] =
              g_local_sum;
          vh_slm_ptr[tok] = (tok < current_chunk_size) ? g_local_sum : 0.0f;
          g_local_sum -= g_local[c];
        }
      }
    }

    /* Fused K/Q L2-normalization, one SG per row reducing and rescaling both
     * K and Q. The K·Kᵀ / Q·Kᵀ GEMM pair below is hoisted out of the v_head
     * loop: the kv_ratio v_heads sharing a kv_head_id get identical raw
     * accumulators, so compute them once per k_head and mask per v_head.
     *
     * CRITICAL ASSUMPTION: num_v_heads is an exact integer multiple of
     * num_k_heads (num_v_heads >= num_k_heads, GQA-style head grouping).
     *
     * No has_work guard: idle groups have current_chunk_size == 0, so the
     * row loop runs zero iterations on its own.
     *
     * The reduce pass caches each lane's q/k slice so the rescale pass reads
     * registers, not gmem. One lane owns each element, so this is bit-exact. */
    static constexpr int k_elems_per_step = sub_group_size * elem_per_item;
    static constexpr int kNormCacheSteps =
        kNormCacheHeadKDim / k_elems_per_step;
    /* head_k_dim is a multiple of chunk_size, so the step guards below are
     * uniform scalar branches, not per-lane masks. */
    const int k_steps = head_k_dim / k_elems_per_step;
    const int k_steps_cached =
        (k_steps < kNormCacheSteps) ? k_steps : kNormCacheSteps;
    const int k_dim_uncached = k_steps_cached * k_elems_per_step;

    for (int row = sg_id; row < current_chunk_size; row += sg_range) {
      int64_t handle_idx =
          (static_cast<int64_t>(chunk_start_offset) + row) * num_k_heads +
          kv_head_id;
      auto q_norm_ptr = q + handle_idx * head_k_dim;
      auto k_norm_ptr = k + handle_idx * head_k_dim;
      float q_sum = 0.0f;
      float k_sum = 0.0f;
      /* Live only within this row iteration. */
      float q_reg[kNormCacheSteps * elem_per_item];
      float k_reg[kNormCacheSteps * elem_per_item];
      using norm_pack_t = norm_pack<T, elem_per_item>;
      using norm_word_t = typename norm_pack_t::word_t;
      /* Padding would widen the access past the elements it carries. */
      static_assert(
          sizeof(norm_pack_t) == sizeof(norm_word_t),
          "norm_pack must be exactly its elements");
      CUTE_UNROLL
      for (int s = 0; s < kNormCacheSteps; ++s) {
        if (s >= k_steps_cached) break;
        const int k_dim_idx =
            s * k_elems_per_step + sg_local_id * elem_per_item;
        norm_word_t q_w;
        norm_word_t k_w;
        __builtin_memcpy(&q_w, q_norm_ptr + k_dim_idx, sizeof(q_w));
        __builtin_memcpy(&k_w, k_norm_ptr + k_dim_idx, sizeof(k_w));
        const norm_pack_t q_pk = platform::bit_cast<norm_pack_t>(q_w);
        const norm_pack_t k_pk = platform::bit_cast<norm_pack_t>(k_w);
        CUTE_UNROLL
        for (int e = 0; e < elem_per_item; ++e) {
          float q_value = static_cast<float>(q_pk.e[e]);
          float k_value = static_cast<float>(k_pk.e[e]);
          q_reg[s * elem_per_item + e] = q_value;
          k_reg[s * elem_per_item + e] = k_value;
          q_sum += q_value * q_value;
          k_sum += k_value * k_value;
        }
      }
      /* Uncached remainder: reduce from gmem, re-read in the tail below. */
      for (int k_dim_idx = k_dim_uncached + sg_local_id * elem_per_item;
            k_dim_idx < head_k_dim;
            k_dim_idx += k_elems_per_step) {
        CUTE_UNROLL
        for (int e = 0; e < elem_per_item; ++e) {
          float q_value = q_norm_ptr[k_dim_idx + e];
          float k_value = k_norm_ptr[k_dim_idx + e];
          q_sum += q_value * q_value;
          k_sum += k_value * k_value;
        }
      }
      q_sum = sycl::reduce_over_group(sg, q_sum, sycl::plus<>());
      k_sum = sycl::reduce_over_group(sg, k_sum, sycl::plus<>());
      q_sum = sycl::sqrt(q_sum + eps);
      k_sum = sycl::sqrt(k_sum + eps);
      CUTE_UNROLL
      for (int s = 0; s < kNormCacheSteps; ++s) {
        if (s >= k_steps_cached) break;
        const int k_dim_idx =
            s * k_elems_per_step + sg_local_id * elem_per_item;
        norm_pack_t q_pk;
        norm_pack_t k_pk;
        CUTE_UNROLL
        for (int e = 0; e < elem_per_item; ++e) {
          q_pk.e[e] = static_cast<T>(
              q_reg[s * elem_per_item + e] / q_sum * q_scale);
          k_pk.e[e] = static_cast<T>(
              k_reg[s * elem_per_item + e] / k_sum);
        }
        const norm_word_t q_w = platform::bit_cast<norm_word_t>(q_pk);
        const norm_word_t k_w = platform::bit_cast<norm_word_t>(k_pk);
        __builtin_memcpy(q_norm_ptr + k_dim_idx, &q_w, sizeof(q_w));
        __builtin_memcpy(k_norm_ptr + k_dim_idx, &k_w, sizeof(k_w));
      }
      for (int k_dim_idx = k_dim_uncached + sg_local_id * elem_per_item;
            k_dim_idx < head_k_dim;
            k_dim_idx += k_elems_per_step) {
        CUTE_UNROLL
        for (int e = 0; e < elem_per_item; ++e) {
          q_norm_ptr[k_dim_idx + e] = static_cast<T>(
              static_cast<float>(q_norm_ptr[k_dim_idx + e]) / q_sum *
              q_scale);
          k_norm_ptr[k_dim_idx + e] = static_cast<T>(
              static_cast<float>(k_norm_ptr[k_dim_idx + e]) / k_sum);
        }
      }
    }
    item.barrier(sycl::access::fence_space::global_and_local);

    auto k_ptr = k +
                  static_cast<int64_t>(chunk_start_offset) * num_k_heads *
                      head_k_dim +
                  kv_head_id * head_k_dim;
    auto K_tensor_shape = make_shape(chunk_size, head_k_dim);
    auto K_tensor = make_tensor(
        make_gmem_ptr(k_ptr),
        make_layout(
            K_tensor_shape, make_stride(head_k_dim * num_k_heads, _1{})));

    auto q_ptr = q +
                  static_cast<int64_t>(chunk_start_offset) * num_k_heads *
                      head_k_dim +
                  kv_head_id * head_k_dim;
    auto Q_tensor_shape = make_shape(chunk_size, head_k_dim);
    auto Q_tensor = make_tensor(
        make_gmem_ptr(q_ptr),
        make_layout(
            Q_tensor_shape, make_stride(head_k_dim * num_k_heads, _1{})));

    /* Accumulator setup, the shared GEMM, and the per-v_head mask/store are all
     * v_head-independent-or-register work that idle groups never read, so guard
     * the whole span with one has_work block; the base fragments and layout-only
     * identity tensor live inside it too (unused when has_work=false). Idle
     * groups fall straight through to the unconditional WAR barrier below. */
    if (has_work) {
      /* Identity tensor and base accumulators are layout-only — independent
       * of v_head_id since A and o2 share the same per-chunk shape/stride. */
      auto C_tensor_shape = make_shape(chunk_size, chunk_size);
      Tensor cC = make_identity_tensor(C_tensor_shape);
      Tensor gC =
          local_tile(cC, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});

      auto tSrA_base = thr_mma.partition_sg_fragment_C(gC);
      auto tSrO2_base = thr_mma.partition_sg_fragment_C(gC);

      clear(tSrA_base);
      clear(tSrO2_base);
      gemm_TTS_shareB_multi_tile(
          local_id, K_tensor, Q_tensor, K_tensor, tSrA_base,
          tSrO2_base, 0, 0, mma);

      for (int g_idx = 0; g_idx < kv_ratio; ++g_idx) {
        const int v_head_id = kv_head_id * kv_ratio + g_idx;

        /* This v_head's cumsum gate was written straight into SLM by the
         * hoisted gate loop above (with the tail already zero-padded); the
         * global_and_local barrier after normalization fenced it, so no
         * per-v_head read-back is needed — just point at the slot. */
        float* g_slm_ptr = group_slm_ptr + g_idx * chunk_size;

        auto A_ptr = A +
                      static_cast<int64_t>(v_head_id) * total_virtual_seqlen *
                          chunk_size +
                      chunk_start_offset * chunk_size;
        auto A_tensor = make_tensor(
            make_gmem_ptr(A_ptr),
            make_layout(C_tensor_shape, make_stride(chunk_size, _1{})));

        auto copy_A_c = get_block_2d_copy_D<void>(mma, A_tensor);
        auto thr_copy_A_c = copy_A_c.get_slice(local_id);
        auto tCrA_c = thr_copy_A_c.partition_sg_fragment_S(gC);
        auto tCgA_c = thr_copy_A_c.partition_D(gC);

        auto O2_ptr = o2 +
                      static_cast<int64_t>(v_head_id) * total_virtual_seqlen *
                          chunk_size +
                      chunk_start_offset * chunk_size;
        auto O2_tensor = make_tensor(
            make_gmem_ptr(O2_ptr),
            make_layout(C_tensor_shape, make_stride(chunk_size, _1{})));

        auto copy_O2_c = get_block_2d_copy_D<void>(mma, O2_tensor);
        auto thr_copy_O2_c = copy_O2_c.get_slice(local_id);
        auto tCrO2_c = thr_copy_O2_c.partition_sg_fragment_S(gC);
        auto tCgO2_c = thr_copy_O2_c.partition_D(gC);

        /* Per-v_head working copies of the shared base accumulators; the mask
         * mutates these in place while the base stays clean for the next
         * iteration. */
        auto tSrA_c = thr_mma.partition_sg_fragment_C(gC);
        auto tSrO2_c = thr_mma.partition_sg_fragment_C(gC);
        cute::copy(tSrA_base, tSrA_c);
        cute::copy(tSrO2_base, tSrO2_c);

        /* Fused mask:
         *   L  (A):  m>n  => *=exp(g[m]-g[n])*b[m]; m==n => 1; m<n => 0.
         *   O2:      m>=n => *=exp(g[m]-g[n]);                m<n => 0.
         * b[m] depends only on sm, so hoist it into a per-lane register. */
        float beta_reg[SG_M];
        CUTE_UNROLL
        for (int sm = 0; sm < SG_M; ++sm) {
          beta_reg[sm] =
              b[(chunk_start_offset + m_sg_start + sm) +
                v_head_id * total_virtual_seqlen];
        }
        CUTE_UNROLL
        for (int sn = 0; sn < SG_N / sub_group_size; ++sn) {
          int n_idx =
              n_sg_start + sn * sub_group_size + sg_local_id;
          CUTE_UNROLL
          for (int sm = 0; sm < SG_M; ++sm) {
            int m_idx = m_sg_start + sm;
            int idx = sn * SG_M + sm;
            float beta_value = beta_reg[sm];
            float e =
                sycl::native::exp(g_slm_ptr[m_idx] - g_slm_ptr[n_idx]);

            tSrA_c(idx) *= e * beta_value;
            tSrO2_c(idx) *= e;
            if (m_idx == n_idx) {
              tSrA_c(idx) = 1.0f;
            }
            if (m_idx < n_idx) {
              tSrA_c(idx) = 0.0f;
              tSrO2_c(idx) = 0.0f;
            }
            /* Tail rows (m past current_chunk_size) hold un-normalized K/Q,
             * so L's exp(g[m]-g[n])*(K_m·K_n)*b[m] entries there overflow and
             * the lower-triangular inverse blows up to Inf on those rows.
             *   L  (A):  m>=cur => identity row (m==n => 1; else 0).
             *   O2:      m>=cur => 0.
             * Keeps L block-triangular so L^-1/U/W tail rows stay finite; the
             * zero tail weights downstream then give 0, not 0*Inf = NaN. */
            if (m_idx >= current_chunk_size) {
              tSrA_c(idx) = (m_idx == n_idx) ? 1.0f : 0.0f;
              tSrO2_c(idx) = 0.0f;
            }
          }
        }

        reorder(tSrA_c, tCrA_c);
        copy(copy_A_c, tCrA_c, tCgA_c);
        reorder(tSrO2_c, tCrO2_c);
        copy(copy_O2_c, tCrO2_c, tCgO2_c);
      }
    }

    /* One WAR fence per chunk (not per v_head): each g_idx read its own SLM
     * slot, so no barrier is needed between them — only before the next
     * work-item's gate loop overwrites these slots. */
    item.barrier(sycl::access::fence_space::local_space);
  }
}

/* Block extent of the inversion, pinned to the sub-group size: step 1 gives
 * each of the 16 lanes one row of a diagonal block. */
static constexpr int inv_block = sub_group_size;
static constexpr int inv_blocks = chunk_size / inv_block;
static_assert(chunk_size % inv_block == 0,
              "inverse kernel needs a whole number of blocks per chunk");

/* Quadrant = 2x2 blocks: the widest 2D load message (32 rows x 64 bytes, see
 * max_h/load_width in copy_traits_xe_2d.hpp) and the off-diagonal GEMM tile. */
static constexpr int inv_quad = 2 * inv_block;
static_assert(inv_blocks % 2 == 0,
              "inverse kernel needs a whole number of 2x2 block quadrants");
static constexpr int inv_quads = inv_blocks / 2;
/* Step 3 is the closed form for a 2x2 quadrant grid; more would need a
 * recurrence. */
static_assert(inv_quads == 2,
              "inverse kernel is written for a 2x2 grid of quadrants");

/* L(i,i) as a 16x16 view of whichever 32x32 diagonal quadrant carries it:
 * blocks 0,1 sit in the top-left quadrant, blocks 2,3 in the bottom-right. */
template <class FragTL, class FragBR, class BlockFrag, class I>
CUTE_DEVICE auto inv_diag_sub(FragTL& q_tl, FragBR& q_br,
                              BlockFrag const& a_frag, I i) {
  if constexpr (i < _2{}) {
    return sub_frag(q_tl, a_frag, i, i);
  } else {
    return sub_frag(q_br, a_frag, i - _2{}, i - _2{});
  }
}

/* Invert one diagonal 16x16 block into `tCrC`, in registers, no DPAS. An A
 * fragment hands lane l column l indexed by row, so A_col[r] is element (r,l). */
template <class SubGroup, class DiagFrag, class CFrag>
CUTE_DEVICE void invert_diag_block(SubGroup const& sg, int sg_local_id,
                                   DiagFrag const& L_ii, CFrag& tCrC) {
  float A_col[inv_block];
  for_each(make_seq<inv_block>{},
           [&](auto r) { A_col[r] = static_cast<float>(L_ii(r)); });

  /* Forward substitution: row r is minus the dot product of rows 0..r-1
   * against row r of L, broadcast from lane e before lane e writes it. */
  CUTE_UNROLL
  for (int r = 1; r < inv_block; ++r) {
    float L_row_r[inv_block];
    CUTE_UNROLL
    for (int e = 0; e < r; ++e) {
      L_row_r[e] = sycl::group_broadcast(sg, A_col[r], e);
    }

    float dot = 0.0f;
    CUTE_UNROLL
    for (int e = 0; e < r; ++e) {
      dot -= A_col[e] * L_row_r[e];
    }

    if (r != sg_local_id) {
      A_col[r] = dot;
    }
  }

  /* The inverse is unit lower triangular too, so A_col holds the whole column
   * and the block stores in 2 messages, not inv_block - 1. */
  for_each(make_seq<inv_block>{}, [&](auto r) { tCrC(r) = A_col[r]; });
}

template <typename T, class TiledMMA>
CUTE_DEVICE void chunk_inverse_opt_kernel(
    T* A,
    const int* query_start_loc,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_v_heads) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

  auto sg = item.get_sub_group();
  int sg_id = sg.get_group_linear_id();
  int sg_range = sg.get_group_linear_range();
  int sg_local_id = sg.get_local_linear_id();

  int group_id = item.get_group(1);
  int group_range = item.get_group_range(1);

  /* One sub-group inverts one (chunk, v_head) matrix: fixed v_head, striding
   * across chunks by chunk_range. */
  int total_sg_range = group_range * sg_range;
  int total_sg_id = group_id * sg_range + sg_id;

  /* launch_stage_inverse() floors the grid at total_sg_range >= num_v_heads, so
   * the max() only guards a caller that sizes its own grid. */
  const int chunk_range = cute::max(1, total_sg_range / num_v_heads);
  int chunk_id = total_sg_id % chunk_range;
  const int v_head_id = total_sg_id / chunk_range;

  /* Drop the tail (total_sg_range need not be a multiple of num_v_heads) before
   * any A access. Uniform across the lanes, so the broadcasts stay safe. */
  if (v_head_id >= num_v_heads) {
    return;
  }

  int pre_chunks = 0;

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();
  auto thr_mma = mma.get_slice(sg_local_id);

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const int seq_start_offset = query_start_loc[batch_id];
    const int seq_end_offset = query_start_loc[batch_id + 1];
    const int seq_len = seq_end_offset - seq_start_offset;

    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int cumsum_chunks = pre_chunks + current_chunks;

    if (chunk_id >= cumsum_chunks) {
      pre_chunks = cumsum_chunks;
      continue;
    }

    while (chunk_id < cumsum_chunks) {
      const int chunk_start_offset = chunk_id * chunk_size;

      auto A_ptr =
          A +
          static_cast<int64_t>(v_head_id) * total_virtual_seqlen * chunk_size +
          chunk_start_offset * chunk_size;

      /* Block forward substitution over a 4x4 grid of 16x16 blocks, in a 2x2
       * grid of 32x32 quadrants. See xe35_gdn_attention.md, "Stage 2". */

      /* Two views of the same in-place matrix: L feeds the DPAS A operand
       * (row-major), L_T the B operand. L_T block (n,k) is L block (k,n). */
      auto L_shape = make_shape(Int<chunk_size>{}, Int<chunk_size>{});
      auto L = make_tensor(make_gmem_ptr(A_ptr),
                           make_layout(L_shape, make_stride(chunk_size, _1{})));
      auto L_T =
          make_tensor(make_gmem_ptr(A_ptr),
                      make_layout(L_shape, make_stride(_1{}, chunk_size)));

      /* gemm_STS_sg runs one block per call, so the MMA tile is one block. */
      CUTE_STATIC_ASSERT_V(get<0>(wg_tile) == Int<inv_block>{});
      CUTE_STATIC_ASSERT_V(get<1>(wg_tile) == Int<inv_block>{});
      CUTE_STATIC_ASSERT_V(get<2>(wg_tile) == Int<inv_block>{});

      /* Every block GEMM has the same MxNxK extent, so the fragments and store
       * atom are built once. */
      Tensor cL = make_identity_tensor(L_shape);
      Tensor c_blk = local_tile(cL, wg_tile, make_coord(_0{}, _0{}, _0{}),
                                Step<_1, _1, X>{});
      auto tCrC = thr_mma.partition_sg_fragment_C(c_blk);

      auto copy_D = get_block_2d_copy_D<void>(mma, L);
      auto thr_copy_D = copy_D.get_slice(sg_local_id);
      auto tCrD = thr_copy_D.partition_sg_fragment_S(c_blk);

      /* Block coordinates are cute::Int<> so each offset folds into the 2D-copy
       * descriptor as an immediate, not per-GEMM address math. */
      auto blk_tile = select<0, 1>(wg_tile);

      /* One TiledMMA shapes both the 32x32 quadrant loads and step 3's two
       * 32x32x32 GEMMs, so the loaded fragment is the operand they consume. */
      using MMAQuad = typename TiledMMAHelper<
          MMA_Atom<XE_DPAS_TT<8, float, cutlass::platform::remove_cv_t<T>>>,
          Layout<Shape<Int<inv_quad>, Int<inv_quad>, Int<inv_quad>>>,
          Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>>::TiledMMA;
      MMAQuad mma_quad{};
      auto quad_mnk = mma_quad.tile_mnk();
      CUTE_STATIC_ASSERT_V(get<0>(quad_mnk) == Int<inv_quad>{});
      CUTE_STATIC_ASSERT_V(get<1>(quad_mnk) == Int<inv_quad>{});
      CUTE_STATIC_ASSERT_V(get<2>(quad_mnk) == Int<inv_quad>{});
      auto quad_tile = select<0, 2>(quad_mnk);
      auto thr_mma_quad = mma_quad.get_slice(sg_local_id);
      auto copy_A_quad = get_block_2d_copy_A<void>(mma_quad, L);
      auto thr_copy_A_quad = copy_A_quad.get_slice(sg_local_id);

      /* Step 3 accumulates a whole quadrant: quadrant-sized C fragment and
       * store atom, alongside the 16x16 pair above. */
      Tensor c_quad_mn = local_tile(cL, select<0, 1>(quad_mnk),
                                    make_coord(_0{}, _0{}));
      auto tQrC = thr_mma_quad.partition_sg_fragment_C(c_quad_mn);
      auto copy_Dq = get_block_2d_copy_D<void>(mma_quad, L);
      auto thr_copy_Dq = copy_Dq.get_slice(sg_local_id);
      auto tQrD = thr_copy_Dq.partition_sg_fragment_S(c_quad_mn);
      auto quad_mn = select<0, 1>(quad_mnk);

      /* tQrC += tQrA * L_T quadrant (blk_n, blk_k). B cannot alias the MMA
       * fragment at 32-wide N, so gemm_STS_sg will not do: load and reorder. */
      Tensor cB_quad = local_tile(make_identity_tensor(L_T.shape()),
                                  select<1, 2>(quad_mnk),
                                  make_coord(_0{}, _0{}));
      auto copy_Bq = get_block_2d_copy_B<void>(mma_quad, L_T);
      auto thr_copy_Bq = copy_Bq.get_slice(sg_local_id);
      auto tQrB = thr_mma_quad.partition_sg_fragment_B(cB_quad);
      auto cprBq = thr_copy_Bq.partition_sg_fragment_D(cB_quad);
      auto gemm_quad = [&](auto const& tQrA, auto blk_n, auto blk_k) {
        Tensor gB = local_tile(make_identity_tensor(L_T.shape()),
                               select<1, 2>(quad_mnk),
                               make_coord(blk_n, blk_k));
        copy(copy_Bq, thr_copy_Bq.partition_S(gB), cprBq);
        reorder(cprBq, tQrB);
        cute::gemm(mma_quad, tQrA, tQrB, tQrC);
      };

      Tensor c_quad = local_tile(cL, quad_tile, make_coord(_0{}, _0{}));
      auto q_tl = thr_mma_quad.partition_sg_fragment_A(c_quad);  // L(0:2, 0:2)
      auto q_br = thr_mma_quad.partition_sg_fragment_A(c_quad);  // L(2:4, 2:4)
      /* Step 3's two operands in the same 32x32 A form: Q^-1, assembled block
       * by block as step 2 produces it, and the intermediate -Q^-1 C. */
      auto rQinv = thr_mma_quad.partition_sg_fragment_A(c_quad);
      auto rU = thr_mma_quad.partition_sg_fragment_A(c_quad);
      load_sg_tile_A(copy_A_quad, thr_copy_A_quad, cL, quad_tile, q_tl, _0{},
                     _0{});
      load_sg_tile_A(copy_A_quad, thr_copy_A_quad, cL, quad_tile, q_br, _1{},
                     _1{});

      /* Shape template for sub_frag(): a 16x16 block of a 32x32 A fragment. */
      auto a_frag = thr_mma.partition_sg_fragment_A(c_blk);
      /* Step 1 indexes tCrC by row, which needs it compact in row order. */
      static_assert(
          is_same_v<decltype(coalesce(tCrC.layout())),
                    Layout<Int<inv_block>, _1>>,
          "block C fragment is not one compact run of inv_block elements");

      /* Step 1's two 16x16 register operands: the inverted diagonal block and
       * the intermediate M = -Inv(b1,b1) L(b1,b0). */
      auto rInv_bb = thr_mma.partition_sg_fragment_A(c_blk);
      auto rM = thr_mma.partition_sg_fragment_A(c_blk);

      // Step 1 on block (i,i), reading it from whichever quadrant holds it.
      auto invert_diag = [&](auto i) {
        invert_diag_block(sg, sg_local_id, inv_diag_sub(q_tl, q_br, a_frag, i),
                          tCrC);
      };
      // tCrC -> block (i,j); tQrC -> quadrant (qi,qj).
      auto store_block = [&](auto i, auto j) {
        store_sg_tile(copy_D, thr_copy_D, cL, blk_tile, tCrC, tCrD, i, j);
      };
      auto store_quad = [&](auto qi, auto qj) {
        store_sg_tile(copy_Dq, thr_copy_Dq, cL, quad_mn, tQrC, tQrD, qi, qj);
      };

      /* Step 2: invert the two diagonal quadrants. Quadrant (1,1) is Q^-1,
       * which step 3 needs as a 32x32 A operand, so it is banked into rQinv. */
      for_each(make_seq<inv_quads>{}, [&](auto q) {
        constexpr bool keep = decltype(q)::value == inv_quads - 1;
        auto b0 = q * _2{};
        auto b1 = b0 + _1{};

        invert_diag(b0);
        if constexpr (keep) {
          reorder(tCrC, sub_frag(rQinv, a_frag, _0{}, _0{}));
        }
        store_block(b0, b0);

        invert_diag(b1);
        reorder(tCrC, rInv_bb);
        if constexpr (keep) {
          reorder(tCrC, sub_frag(rQinv, a_frag, _1{}, _1{}));
        }
        store_block(b1, b1);

        /* B operands index the transposed view: L(p,r) is at (n,k) = (r,p). */
        clear(tCrC);
        gemm_STS_sg(rInv_bb, L_T, tCrC, b0, b1, mma);
        negate_frag(tCrC);
        reorder(tCrC, rM);

        clear(tCrC);
        gemm_STS_sg(rM, L_T, tCrC, b0, b0, mma);
        store_block(b1, b0);
        if constexpr (keep) {
          reorder(tCrC, sub_frag(rQinv, a_frag, _1{}, _0{}));
          /* Q^-1's strictly-upper block is never written above, so zero it
           * before step 3's GEMM reads it. */
          auto z = sub_frag(rQinv, a_frag, _0{}, _1{});
          clear(z);
        }
      });

      /* Step 3: the bottom-left quadrant, -Q^-1 C P^-1. U = -Q^-1 * C, then
       * R = U * P^-1; both B operands are already in gmem. */
      clear(tQrC);
      gemm_quad(rQinv, _0{}, _1{});
      negate_frag(tQrC);
      reorder(tQrC, rU);

      clear(tQrC);
      gemm_quad(rU, _0{}, _0{});
      store_quad(_1{}, _0{});

      chunk_id += chunk_range;
    }
    pre_chunks = cumsum_chunks;
  }
}

template <typename T, class TiledMMA>
CUTE_DEVICE void chunk_compute_wu_kernel(
    const sycl::local_accessor<float, 1>& slm_mem_const,
    T* A,
    T* w,
    T* u,
    const T* q,
    const T* k,
    const T* v,
    const float* b,
    const float* a,
    const float* A_log,
    const T* dt_bias,
    const int* query_start_loc,
    const bool* has_initial_state,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();
  int chunk_id = item.get_group(1);
  const int global_chunk_range = item.get_group_range(1);

  auto sg = item.get_sub_group();
  int sg_local_id = sg.get_local_linear_id();

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();

  /* The work-group is a full Xe core, holding num_tiles independent MMA
   * tiles (size(mma) work-items each), so each tile drives its own v_head
   * and all sub-groups issue DPAS. Never add a work-group-scope barrier in
   * the per-tile region below -- tiles run different numbers of v_heads and
   * would deadlock. */
  const int tile_size = size(TiledMMA{});
  const int num_tiles = item.get_local_range(2) / tile_size;
  const int tile_id = local_id / tile_size;
  const int tile_local_id = local_id % tile_size;

  auto thr_mma = mma.get_slice(tile_local_id);

  /* One beta/g slice per sub-group, not per tile: every sub-group of a tile
   * reads the full chunk_size diagonal span (via sg_local_id), so a shared
   * per-tile slice would create a cross-sub-group dependency. Per-sub-group
   * slices keep the fill lane-private and barrier-free. */
  const int sg_slot = static_cast<int>(sg.get_group_linear_id());
  float* slm_mem = static_cast<float*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());
  float* g_slm_ptr = slm_mem + sg_slot * chunk_size * 2;
  float* beta_slm_ptr = g_slm_ptr + chunk_size;

  int pre_chunks = 0;

  const int kv_ratio = num_v_heads / num_k_heads;

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    // nullptr has_initial_state means "every batch has carry-over state" --
    // Caller pre-zeroes
    // ssm_state for batches that should not carry over; see the docstring
    // on GDNArguments::has_initial_state.
    const bool initial_state =
        (has_initial_state == nullptr) || has_initial_state[batch_id];
    const int seq_start_offset = query_start_loc[batch_id];
    const int seq_end_offset = query_start_loc[batch_id + 1];
    const int seq_len = seq_end_offset - seq_start_offset;

    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int cumsum_chunks = pre_chunks + current_chunks;

    if (chunk_id >= cumsum_chunks) {
      pre_chunks = cumsum_chunks;
      continue;
    }

    while (chunk_id < cumsum_chunks) {
      const int chunk_start_offset = chunk_id * chunk_size;

      /* Strided v_head split across the work-group's tiles: tile T starts at
       * v_head T and strides by num_tiles. A tile past num_v_heads just runs
       * zero iterations and falls through -- safe since the region is
       * barrier-free. */
      for (int v_head_id = tile_id; v_head_id < num_v_heads;
           v_head_id += num_tiles) {
        /* Precompute per-token scaling factors into SLM:
         *   beta[t]   = b[t]              (delta-rule write scale)
         *   g[t] = exp(a[t]) * b[t]       (decay * beta, used for W)
         * a[t] here is the CUMSUM gate from stage 1; exp(a[m]-a[n]) gives
         * the product of per-token decays from n to m, but using the
         * cumsum directly is cheaper (one exp per token vs. one per pair).
         * Filled with stride sub_group_size from sg_local_id: lane L writes
         * exactly the indices it later reads via the GEMM's diagonal lookup,
         * so no barrier is needed (relies on mma_K == sub_group_size). */
        CUTE_UNROLL
        for (int e = sg_local_id; e < chunk_size; e += sub_group_size) {
          float beta_value =
              b[(chunk_start_offset + e) + v_head_id * total_virtual_seqlen];
          float a_value =
              a[(chunk_start_offset + e) + v_head_id * total_virtual_seqlen];
          beta_slm_ptr[e] = beta_value;
          g_slm_ptr[e] = sycl::exp(a_value) * beta_value;
        }

        auto A_ptr = A +
                     static_cast<int64_t>(v_head_id) * total_virtual_seqlen *
                         chunk_size +
                     chunk_start_offset * chunk_size;
        auto A_tensor_shape = make_shape(chunk_size, chunk_size);
        auto A_tensor = make_tensor(
            make_gmem_ptr(A_ptr),
            make_layout(A_tensor_shape, make_stride(chunk_size, _1{})));

        auto v_ptr = v +
                     static_cast<int64_t>(chunk_start_offset) * num_v_heads *
                         head_v_dim +
                     v_head_id * head_v_dim;
        auto V_tensor_T_shape = make_shape(head_v_dim, chunk_size);
        auto V_tensor_T = make_tensor(
            make_gmem_ptr(v_ptr),
            make_layout(
                V_tensor_T_shape, make_stride(_1{}, head_v_dim * num_v_heads)));
        auto U_ptr = u +
                     static_cast<int64_t>(v_head_id) * total_virtual_seqlen *
                         head_v_dim +
                     chunk_start_offset * head_v_dim;
        auto U_tensor_shape = make_shape(chunk_size, head_v_dim);
        auto U_tensor = make_tensor(
            make_gmem_ptr(U_ptr),
            make_layout(U_tensor_shape, make_stride(head_v_dim, _1{})));

        Tensor cU = make_identity_tensor(U_tensor.shape());
        auto copy_U_c = get_block_2d_copy_D<void>(mma, U_tensor);
        auto thr_copy_U_c = copy_U_c.get_slice(tile_local_id);

        for (int dv = 0; dv < head_v_dim / chunk_size; ++dv) {
          Tensor gU_C =
              local_tile(cU, wg_tile, make_coord(0, dv, 0), Step<_1, _1, X>{});
          auto tCrU_c = thr_copy_U_c.partition_sg_fragment_S(gU_C);
          auto tCgU_c = thr_copy_U_c.partition_D(gU_C);
          auto tSrU_c = thr_mma.partition_sg_fragment_C(gU_C);
          /* U[m,:] = sum_n  L^-1[m,n] * V[n,:] * b[n]  -- scaled V projection. */
          clear(tSrU_c);
          gemm_TTS_k_multi_tile(
              A_tensor,
              V_tensor_T,
              tSrU_c,
              0,
              dv,
              mma,
              beta_slm_ptr,
              tile_local_id);
          reorder(tSrU_c, tCrU_c);
          copy(copy_U_c, tCrU_c, tCgU_c);
        }

        /* W is only needed when a previous state exists (chunk > 0 or has_initial_state).
         * W[m,:] = sum_n  L^-1[m,n] * K[n,:] * exp(a[n]) * b[n]
         * fwd_o uses W to correct U by subtracting W @ S_prev^T. */
        if (((chunk_id - pre_chunks) != 0) || initial_state) {
          auto k_ptr = k +
                       static_cast<int64_t>(chunk_start_offset) * num_k_heads *
                           head_k_dim +
                       (v_head_id / kv_ratio) * head_k_dim;
          auto K_tensor_T_shape = make_shape(head_k_dim, chunk_size);
          auto K_tensor_T = make_tensor(
              make_gmem_ptr(k_ptr),
              make_layout(
                  K_tensor_T_shape,
                  make_stride(_1{}, head_k_dim * num_k_heads)));
          auto W_ptr = w +
                       static_cast<int64_t>(v_head_id) * total_virtual_seqlen *
                           head_k_dim +
                       chunk_start_offset * head_k_dim;
          auto W_tensor_shape = make_shape(chunk_size, head_k_dim);
          auto W_tensor = make_tensor(
              make_gmem_ptr(W_ptr),
              make_layout(W_tensor_shape, make_stride(head_k_dim, _1{})));

          Tensor cW = make_identity_tensor(W_tensor.shape());
          auto copy_W_c = get_block_2d_copy_D<void>(mma, W_tensor);
          auto thr_copy_W_c = copy_W_c.get_slice(tile_local_id);

          for (int dk = 0; dk < head_k_dim / chunk_size; ++dk) {
            Tensor gW_C = local_tile(
                cW, wg_tile, make_coord(0, dk, 0), Step<_1, _1, X>{});
            auto tCrW_c = thr_copy_W_c.partition_sg_fragment_S(gW_C);
            auto tCgW_c = thr_copy_W_c.partition_D(gW_C);
            auto tSrW_c = thr_mma.partition_sg_fragment_C(gW_C);
            /* W[m,:] = sum_n L^-1[m,n] * K[n,:] * g[n]  (g = exp(a)*b). */
            clear(tSrW_c);
            gemm_TTS_k_multi_tile(
                A_tensor,
                K_tensor_T,
                tSrW_c,
                0,
                dk,
                mma,
                g_slm_ptr,
                tile_local_id);
            reorder(tSrW_c, tCrW_c);
            copy(copy_W_c, tCrW_c, tCgW_c);
          }
        }
      }
      chunk_id += global_chunk_range;
    }
    pre_chunks = cumsum_chunks;
  }
}

template <typename T, typename StateT, class TiledMMA>
CUTE_DEVICE void chunk_fwd_o_kernel(
    const sycl::local_accessor<float, 1>& slm_mem_const,  // [2 * chunk_size]
    T* core_attn_out,  // [total_seqlen, num_v_heads, head_v_dim]
    T* o2,  // [num_v_heads, total_virtual_seqlen, chunk_size], O2 = masked Q·K^T
    T* w,  // [num_v_heads, total_virtual_seqlen, head_k_dim]
    T* u,  // [num_v_heads, total_virtual_seqlen, head_v_dim]
    const T* q,  // [total_virtual_seqlen, num_k_heads, head_k_dim]
    const T* k,  // [total_virtual_seqlen, num_k_heads, head_k_dim]
    const float* a,
    StateT*
        ssm_state,  // [cache_batch_size, num_v_heads, head_v_dim, head_k_dim]
    const int ssm_state_stride_0,
    const int* query_start_loc,
    const int* cache_indices,
    const bool* has_initial_state,
    const int batch_size,
    const int total_virtual_seqlen,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();
  int current_batch_id = item.get_group(0);
  int v_head_id = item.get_group(1);
    /* This WG owns one head_v_dim tile;*/
  const int dv = item.get_group(2);
  int local_range = item.get_local_range(2);

  auto sg = item.get_sub_group();
  int sg_local_id = sg.get_local_linear_id();

  float* slm_mem = static_cast<float*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());
  /* Two vectors are consumed here: g_multi feeds the state-update GEMM's
   * per-token diagonal, g_exp scales the Q.S term. The raw cumulative-decay
   * vector that compute_wu stages has no reader in this kernel. */
  float* g_multi_slm_ptr = slm_mem;
  float* g_exp_slm_ptr = g_multi_slm_ptr + chunk_size;

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);

  static constexpr auto ATOM_M =
      get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N =
      get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

  static constexpr auto SG_M = tile_m / ATOM_M;  // BLK_M / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;  // BLK_N / ATOM_N;

  auto sg_local_m_coord = cutlass::get_sub_group_id() / ATOM_N;
  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  int m_tile_start = 0;
  int n_tile_start = 0;
  int m_sg_start = sg_local_m_coord * SG_M;
  int n_sg_start = sg_local_n_coord * SG_N;

  const int kv_ratio = num_v_heads / num_k_heads;
  const int kv_head_id = v_head_id / kv_ratio;

  int pre_chunks = 0;

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    // See chunk_compute_wu_kernel for the rationale; same nullable contract.
    const bool initial_state =
        (has_initial_state == nullptr) || has_initial_state[batch_id];
    const int seq_start_offset = query_start_loc[batch_id];
    const int seq_end_offset = query_start_loc[batch_id + 1];
    const int seq_len = seq_end_offset - seq_start_offset;

    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;

    if (current_batch_id != batch_id) {
      pre_chunks += current_chunks;
      continue;
    }

    StateT* ssm_state_ptr =
        ssm_state +
        static_cast<int64_t>(cache_indices[batch_id]) * ssm_state_stride_0 +
        v_head_id * head_v_dim * head_k_dim;

    /* S is loop-invariant per (v_head, dv): this WG owns the slice for every
     * chunk below, so its tensor and copy atoms are built once here rather
     * than per chunk. */
    StateT* S_ptr = ssm_state_ptr;
    auto S_tensor_shape = make_shape(head_v_dim, head_k_dim);
    auto S_tensor = make_tensor(
        make_gmem_ptr(S_ptr),
        make_layout(S_tensor_shape, make_stride(head_k_dim, _1{})));
    Tensor cS = make_identity_tensor(S_tensor.shape());
    auto copy_S_c = get_block_2d_copy_C<void>(mma, S_tensor);
    auto copy_S_d = get_block_2d_copy_D<void>(mma, S_tensor);
    auto thr_copy_S_c = copy_S_c.get_slice(local_id);
    auto thr_copy_S_d = copy_S_d.get_slice(local_id);
    auto thr_mma = mma.get_slice(local_id);

    for (int chunk_id = 0; chunk_id < current_chunks; ++chunk_id) {
      const bool has_prev_state = (chunk_id != 0) || initial_state;
      const int out_chunk_offset = seq_start_offset + chunk_id * chunk_size;
      const int chunk_offset = (pre_chunks + chunk_id) * chunk_size;

      int current_chunk_size = chunk_size;
      if ((chunk_id + 1) * chunk_size > seq_len) {
        current_chunk_size = seq_len - chunk_id * chunk_size;
      }

      float g_last_value =
          a[(chunk_offset + current_chunk_size - 1) +
            v_head_id * total_virtual_seqlen];
      float g_last_value_exp = sycl::exp(g_last_value);

      /* [1] Publishes the previous chunk's S; above the staging so it also
       * WAR-fences the previous chunk's g_multi / g_exp readers. */
      item.barrier(sycl::access::fence_space::global_and_local);

      CUTE_UNROLL
      for (int e = local_id; e < current_chunk_size; e += local_range) {
        float g_cumsum_value =
            a[(chunk_offset + e) + v_head_id * total_virtual_seqlen];
        g_multi_slm_ptr[e] = sycl::exp(g_last_value - g_cumsum_value);
        g_exp_slm_ptr[e] = sycl::exp(g_cumsum_value);
      }

      CUTE_UNROLL
      for (int e = current_chunk_size + local_id; e < chunk_size;
           e += local_range) {
        g_multi_slm_ptr[e] = 0.0f;
        g_exp_slm_ptr[e] = 0.0f;
      }

      auto W_ptr = w + v_head_id * total_virtual_seqlen * head_k_dim +
                   chunk_offset * head_k_dim;
      auto W_tensor_shape = make_shape(chunk_size, head_k_dim);
      auto W_tensor = make_tensor(
          make_gmem_ptr(W_ptr),
          make_layout(W_tensor_shape, make_stride(head_k_dim, _1{})));

      auto U_ptr = u + v_head_id * total_virtual_seqlen * head_v_dim +
                   chunk_offset * head_v_dim;
      auto U_tensor_shape = make_shape(chunk_size, head_v_dim);
      auto U_tensor = make_tensor(
          make_gmem_ptr(U_ptr),
          make_layout(U_tensor_shape, make_stride(head_v_dim, _1{})));

      Tensor cU = make_identity_tensor(U_tensor.shape());

      auto copy_U_c = get_block_2d_copy_C<void>(mma, U_tensor);
      auto copy_U_d = get_block_2d_copy_D<void>(mma, U_tensor);

      auto thr_copy_U_c = copy_U_c.get_slice(local_id);
      auto thr_copy_U_d = copy_U_d.get_slice(local_id);

      auto O2_ptr =
          o2 +
          static_cast<int64_t>(v_head_id) * total_virtual_seqlen * chunk_size +
          chunk_offset * chunk_size;
      auto O2_tensor_shape = make_shape(current_chunk_size, chunk_size);
      auto O2_tensor = make_tensor(
          make_gmem_ptr(O2_ptr),
          make_layout(O2_tensor_shape, make_stride(chunk_size, _1{})));

      auto U_tensor_T_shape = make_shape(head_v_dim, chunk_size);
      auto U_tensor_T = make_tensor(
          make_gmem_ptr(U_ptr),
          make_layout(U_tensor_T_shape, make_stride(_1{}, head_v_dim)));
      auto O_ptr = core_attn_out + out_chunk_offset * num_v_heads * head_v_dim +
                   v_head_id * head_v_dim;
      auto O_tensor_shape = make_shape(current_chunk_size, head_v_dim);
      auto O_tensor = make_tensor(
          make_gmem_ptr(O_ptr),
          make_layout(
              O_tensor_shape, make_stride(num_v_heads * head_v_dim, _1{})));

      Tensor cO = make_identity_tensor(O_tensor.shape());
      auto copy_O_c = get_block_2d_copy_D<void>(mma, O_tensor);
      auto thr_copy_O_c = copy_O_c.get_slice(local_id);

      Tensor gO_C =
          local_tile(cO, wg_tile, make_coord(0, dv, 0), Step<_1, _1, X>{});
      auto tSrO_c = thr_mma.partition_sg_fragment_C(gO_C);
      /* O = diag(exp(a)) * Q.S + O2 * U. The Q.S term exists only when a
       * previous state does; O2.U always does. Clear unconditionally so the
       * tail below adds O2.U and stores O on both paths. */
      clear(tSrO_c);

      if (has_prev_state) {
        // O2 is now produced upstream in chunk_compute_A_o2 and read from the
        // workspace above, so Q is no longer resident in outer scope here; load
        // it locally for the Q·Sᵀ inter-chunk term. dv comes from the grid
        // (item.get_group(2)), so there is no in-kernel dv loop.
        auto q_ptr =
            q + chunk_offset * num_k_heads * head_k_dim + kv_head_id * head_k_dim;
        auto Q_tensor_shape = make_shape(current_chunk_size, head_k_dim);
        auto Q_tensor = make_tensor(
            make_gmem_ptr(q_ptr),
            make_layout(
                Q_tensor_shape, make_stride(head_k_dim * num_k_heads, _1{})));

        Tensor gU_C =
            local_tile(cU, wg_tile, make_coord(0, dv, 0), Step<_1, _1, X>{});
        auto tSrU_d = thr_mma.partition_sg_fragment_C(gU_C);
        clear(tSrU_d);

        /* W.S (the U correction) and Q.S (O's inter-chunk term) share the same
         * B operand, n tile and k extent, so they run as one k-loop over a
         * single S load instead of two gemm_TTS calls. Neither term depends on
         * the other, so Q.S may precede the U write-back. This helper also
         * carries no per-k-tile work-group barrier, unlike gemm_TTS; the SLM
         * and gmem hand-off barriers are the explicit [1] / [2] pair. */
        gemm_TTS_shareB_multi_tile(
            local_id, W_tensor, Q_tensor, S_tensor, tSrU_d, tSrO_c, 0, dv, mma);

        /* U -= W.S in place through gmem: the corrected U is re-read below and
         * by the state update under different tilings. */
        auto tCrU_d = thr_copy_U_d.partition_sg_fragment_S(gU_C);
        auto tCgU_d = thr_copy_U_d.partition_D(gU_C);
        auto tCrU_c_save = thr_copy_U_c.partition_sg_fragment_D(gU_C);
        reorder(tSrU_d, tCrU_c_save);

        auto tCgU_c = thr_copy_U_c.partition_S(gU_C);
        auto tCrU_c = thr_copy_U_c.partition_sg_fragment_D(gU_C);
        copy(copy_U_c, tCgU_c, tCrU_c);

        CUTE_UNROLL
        for (int i = 0; i < tCrU_c_save.size(); ++i) {
          tCrU_c(i) -= tCrU_c_save(i);
        }

        reorder(tCrU_c, tCrU_d);
        copy(copy_U_d, tCrU_d, tCgU_d);
      }

      /* [2] RAW for the staged gates and the corrected U; WAR for S. Runs on
       * both paths -- gemm_TTS_k_multi reads g_multi unconditionally. */
      item.barrier(sycl::access::fence_space::global_and_local);

      if (has_prev_state) {
        /* Fold the per-token decay into Q.S only; O2.U must not be scaled. */
        CUTE_UNROLL
        for (int sn = 0; sn < SG_N / sub_group_size; ++sn) {
          int n_idx =
              n_tile_start + n_sg_start + sn * sub_group_size + sg_local_id;
          CUTE_UNROLL
          for (int sm = 0; sm < SG_M; ++sm) {
            int m_idx = m_tile_start + m_sg_start + sm;
            tSrO_c(sn * SG_M + sm) *= g_exp_slm_ptr[(m_idx)];
          }
        }
      }

      gemm_TTS(O2_tensor, U_tensor_T, tSrO_c, 0, dv, mma);
      {
        auto tCrO_c = thr_copy_O_c.partition_sg_fragment_S(gO_C);
        auto tCgO_c = thr_copy_O_c.partition_D(gO_C);
        reorder(tSrO_c, tCrO_c);
        copy(copy_O_c, tCrO_c, tCgO_c);
      }

      auto k_ptr =
          k + chunk_offset * num_k_heads * head_k_dim + kv_head_id * head_k_dim;
      auto K_tensor_T_shape = make_shape(head_k_dim, chunk_size);
      auto K_tensor_T = make_tensor(
          make_gmem_ptr(k_ptr),
          make_layout(
              K_tensor_T_shape, make_stride(_1{}, head_k_dim * num_k_heads)));

      for (int dk = 0; dk < head_k_dim / chunk_size; ++dk) {
        Tensor gS_C =
            local_tile(cS, wg_tile, make_coord(dv, dk, 0), Step<_1, _1, X>{});
        auto tCrS_d = thr_copy_S_d.partition_sg_fragment_S(gS_C);
        auto tCgS_d = thr_copy_S_d.partition_D(gS_C);
        auto tSrS_d = thr_mma.partition_sg_fragment_C(gS_C);

        /* Seed accumulator with exp(g_last) * S_prev when previous state
         * exists; otherwise start from zeros. */
        if (has_prev_state) {
          auto tCgS_c = thr_copy_S_c.partition_S(gS_C);
          auto tCrS_c = thr_copy_S_c.partition_sg_fragment_D(gS_C);
          copy(copy_S_c, tCgS_c, tCrS_c);

          reorder(tCrS_c, tSrS_d);
          CUTE_UNROLL
          for (int i = 0; i < tCrS_c.size(); ++i) {
            tSrS_d(i) *= g_last_value_exp;
          }
        } else {
          clear(tSrS_d);
        }

        gemm_TTS_k_multi(
            U_tensor_T, K_tensor_T, tSrS_d, dv, dk, mma, g_multi_slm_ptr);
        reorder(tSrS_d, tCrS_d);
        copy(copy_S_d, tCrS_d, tCgS_d);
      }
    }
    pre_chunks += current_chunks;
  }
}

template <typename T, typename StateTag>
class ChunkComputeAO2Kernel;

template <typename T, typename StateTag>
class ChunkInverseOptKernel;

template <typename T, typename StateTag>
class ChunkComputeWUKernel;

template <typename T, typename StateT>
class ChunkFwdOKernel;

/* ---------------------------------------------------------------------------
 * Per-stage launch entries (one SYCL submit each).
 *
 * Each function submits EXACTLY ONE of the five GDN kernels and returns its
 * sycl::event. They are the single source of truth for each stage's grid / SLM
 * / MMA-policy setup: kernel_launcher() below calls all five in sequence (the
 * normal fused path), and the per-kernel harness (examples/14) calls exactly
 * one to time+verify a single stage in isolation. Splitting them here -- rather
 * than duplicating the launch math in the harness -- guarantees the harness
 * dispatches the same kernel the production launcher does (one launch per UT).
 *
 * `props` is the shared sub_group_size + grf_size property set; `xe_core_count`
 * is the device multiprocessor count. Both are computed once by the caller.
 * ------------------------------------------------------------------------- */

template <typename T, typename StateT, typename Props>
sycl::event launch_stage_compute_A_o2(
    sycl::queue& queue, Props const& props, int xe_core_count,
    T* A, T* o2, T* q, T* k, const float* b, float* a,
    const float* A_log, const T* dt_bias,
    const int* query_start_loc, const int total_virtual_seqlen,
    const int batch_size, const int num_k_heads, const int head_k_dim,
    const int num_v_heads) {
  using Element_non_CV = cutlass::platform::remove_cv_t<T>;
  auto op = XE_DPAS_TT<8, float, Element_non_CV>{};
  using WGTileComputeA_o2 = chunk_gemm_policy_compute_A_O2::WGTile;
  using SGLayoutComputeA_o2 = chunk_gemm_policy_compute_A_O2::SGLayout;
  using MMAComputeA_o2 = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>, Layout<WGTileComputeA_o2>,
      SGLayoutComputeA_o2>::TiledMMA;
  auto mmaComputeA_o2 = MMAComputeA_o2{};
  /* Co-resident groups: one WG fills a whole XeCore (MaxThreadsPerXeCore
   * lanes) and hosts GroupsPerWg cooperative groups of size(mma) lanes each;
   * every group owns one (chunk, k_head) per persistent-loop iteration. Grid
   * is one WG per XeCore. */
  int GroupsPerWgComputeA_o2 = MaxThreadsPerXeCore / size(mmaComputeA_o2);

  sycl::range<3> local_compute_A_o2(1, 1, MaxThreadsPerXeCore);
  sycl::range<3> global_compute_A_o2(1, xe_core_count, 1);
  /* One gate slot per v_head sharing a k_head (kv_ratio slots), per co-resident
   * group: the hoisted cumsum writes each v_head's gate straight into its
   * group's SLM slot and the mask reads it there — no per-v_head gmem
   * read-back. */
  int kv_ratio_A_o2 = num_v_heads / num_k_heads;
  int slm_size_compute_A_o2 =
      GroupsPerWgComputeA_o2 * kv_ratio_A_o2 * chunk_size;
  auto ev = queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> local_mem(
        sycl::range<1>(slm_size_compute_A_o2), cgh);
    cgh.parallel_for<ChunkComputeAO2Kernel<T, StateT>>(
        sycl::nd_range<3>{global_compute_A_o2 * local_compute_A_o2, local_compute_A_o2},
        props,
        [=](auto) {
          chunk_compute_A_o2_kernel<T, MMAComputeA_o2>(
              local_mem, A, o2, q, k, b, a, A_log, dt_bias, query_start_loc,
              total_virtual_seqlen, batch_size, num_k_heads, head_k_dim,
              num_v_heads);
        });
  });
  EventManager::getInstance().addEvent(ev);
  return ev;
}

/* One work-group per Xe-core, floored at one sub-group per v_head. Each
 * sub-group inverts one (chunk, v_head) 64x64 matrix, grid-striding chunks. */
template <typename T, typename StateT, typename Props>
sycl::event launch_stage_inverse(
    sycl::queue& queue, Props const& props, int xe_core_count,
    T* A, const int* query_start_loc, const int total_virtual_seqlen,
    const int batch_size, const int num_v_heads) {
  using Element_non_CV = cutlass::platform::remove_cv_t<T>;
  auto op = XE_DPAS_TT<8, float, Element_non_CV>{};
  using WGTileInverse   = chunk_gemm_policy_inverse::WGTile;
  using SGLayoutInverse = chunk_gemm_policy_inverse::SGLayout;
  using MMAInverse      = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>, Layout<WGTileInverse>, SGLayoutInverse>::TiledMMA;
  sycl::range<3> local_inverse(1, 1, MaxThreadsPerXeCore);
  /* A fixed v_head per sub-group means the grid must supply one sub-group per
   * v_head or the tail heads go silently uninverted. Never binds in practice. */
  constexpr int sgs_per_wg = MaxThreadsPerXeCore / sub_group_size;
  const int wg_count = cute::max(
      xe_core_count, (num_v_heads + sgs_per_wg - 1) / sgs_per_wg);
  sycl::range<3> global_inverse(1, wg_count, 1);
  auto ev = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<ChunkInverseOptKernel<T, StateT>>(
        sycl::nd_range<3>{global_inverse * local_inverse, local_inverse},
        props,
        [=](auto) {
          chunk_inverse_opt_kernel<T, MMAInverse>(
              A, query_start_loc, total_virtual_seqlen, batch_size, num_v_heads);
        });
  });
  EventManager::getInstance().addEvent(ev);
  return ev;
}

template <typename T, typename StateT, typename Props>
sycl::event launch_stage_compute_wu(
    sycl::queue& queue, Props const& props, int xe_core_count,
    T* A, T* w, T* u, T* q, T* k, const T* v, const float* b, float* a,
    const float* A_log, const T* dt_bias, const int* query_start_loc,
    const bool* has_initial_state, const int total_virtual_seqlen,
    const int batch_size, const int num_k_heads, const int head_k_dim,
    const int num_v_heads, const int head_v_dim) {
  using Element_non_CV = cutlass::platform::remove_cv_t<T>;
  auto op = XE_DPAS_TT<8, float, Element_non_CV>{};
  using WGTileComputeWU = chunk_gemm_policy_compute_wu::WGTile;
  using SGLayoutComputeWU = chunk_gemm_policy_compute_wu::SGLayout;
  using MMAComputeWU = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>, Layout<WGTileComputeWU>, SGLayoutComputeWU>::TiledMMA;
  auto mmaComputeWU = MMAComputeWU{};
  /* Local range is a full Xe core; the kernel fans it out across
   num_tiles_wu = MaxThreadsPerXeCore / size(mma) independent MMA tiles so
   all 32 sub-groups issue DPAS. Grid mapping (one work-group per chunk_id)
   is unchanged.
   */
  int MaxThreadsPerWorkgroupComputeWU = MaxThreadsPerXeCore;
  sycl::range<3> local_compute_wu(1, 1, MaxThreadsPerWorkgroupComputeWU);
  sycl::range<3> global_compute_wu(
      1, xe_core_count * MaxThreadsPerXeCore / MaxThreadsPerWorkgroupComputeWU, 1);
  // One beta/g pair per sub-group, not per tile (see kernel's SLM comment).
  int slm_size_compute_wu =
      (MaxThreadsPerWorkgroupComputeWU / sub_group_size) * chunk_size * 2;
  auto ev = queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> local_mem(
        sycl::range<1>(slm_size_compute_wu), cgh);
    cgh.parallel_for<ChunkComputeWUKernel<T, StateT>>(
        sycl::nd_range<3>{global_compute_wu * local_compute_wu, local_compute_wu},
        props,
        [=](auto) {
          chunk_compute_wu_kernel<T, MMAComputeWU>(
              local_mem, A, w, u, q, k, v, b, a, A_log, dt_bias,
              query_start_loc, has_initial_state, total_virtual_seqlen,
              batch_size, num_k_heads, head_k_dim, num_v_heads, head_v_dim);
        });
  });
  EventManager::getInstance().addEvent(ev);
  return ev;
}

template <typename T, typename StateT, typename Props>
sycl::event launch_stage_fwd_o(
    sycl::queue& queue, Props const& props,
    T* core_attn_out, T* o2, T* w, T* u, T* q, T* k, float* a,
    StateT* ssm_state, const int ssm_state_stride_0,
    const int* query_start_loc, const int* cache_indices,
    const bool* has_initial_state, const int batch_size,
    const int total_virtual_seqlen, const int num_k_heads, const int head_k_dim,
    const int num_v_heads, const int head_v_dim) {
  using Element_non_CV = cutlass::platform::remove_cv_t<T>;
  auto op = XE_DPAS_TT<8, float, Element_non_CV>{};
  using WGTileFwdO = chunk_gemm_policy_fwd_o::WGTile;
  using SGLayoutFwdO = chunk_gemm_policy_fwd_o::SGLayout;
  using MMAFwdO = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>, Layout<WGTileFwdO>, SGLayoutFwdO>::TiledMMA;
  auto mmaFwdO = MMAFwdO{};
  int MaxThreadsPerWorkgroupFwdO = size(mmaFwdO);
  sycl::range<3> local_fwd_o(1, 1, MaxThreadsPerWorkgroupFwdO);
  /* Parallelize the head_v_dim tiling (dv) over the grid: each WG owns one
   * (batch, v_head, dv) tile and runs the sequential chunk loop over just its
   * own dv-slice of U/S/O. Tiles are independent — the SSM-state recurrence
   * touches only S[dv,:] — so no cross-dv sync is needed. */
  int dv_tiles_fwd_o = head_v_dim / chunk_size;
  sycl::range<3> global_fwd_o(batch_size, num_v_heads, dv_tiles_fwd_o);
  int slm_size_fwd_o = chunk_size + chunk_size;
  auto ev = queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> local_mem(
        sycl::range<1>(slm_size_fwd_o), cgh);
    cgh.parallel_for<ChunkFwdOKernel<T, StateT>>(
        sycl::nd_range<3>{global_fwd_o * local_fwd_o, local_fwd_o},
        props,
        [=](auto) {
          chunk_fwd_o_kernel<T, StateT, MMAFwdO>(
              local_mem, core_attn_out, o2, w, u, q, k, a, ssm_state,
              ssm_state_stride_0, query_start_loc, cache_indices,
              has_initial_state, batch_size, total_virtual_seqlen,
              num_k_heads, head_k_dim, num_v_heads, head_v_dim);
        });
  });
  EventManager::getInstance().addEvent(ev);
  return ev;
}

template <typename T, typename StateT>
void kernel_launcher(
    sycl::queue& queue,
    T* core_attn_out,
    T* q,
    T* k,
    const T* v,
    T* A,
    T* o2,
    T* w,
    T* u,
    const float* b,
    float* a,
    const float* A_log,
    const T* dt_bias,
    StateT* ssm_state,
    const int ssm_state_stride_0,
    const int* query_start_loc,
    const int* cache_indices,
    const bool* has_initial_state,
    const int batch_size,
    const int total_virtual_seqlen,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  /* Machine-only grid: persistent kernels (prepare, compute_A, inverse,
   * compute_wu) size their grid to the Xe-core array and let their internal
   * grid-stride loops (`chunk_id += global_chunk_range`) sweep all chunks over
   * as many passes as needed. */
  int xe_core_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{
      syclex::sub_group_size<cute::detail::subgroup_size>,
#if defined(SYCL_TARGET_INTEL_GPU_CRI)
      intelex::grf_size<512>
#else
      intelex::grf_size<256>
#endif
  };

  launch_stage_compute_A_o2<T, StateT>(
      queue, kernel_props, xe_core_count, A, o2, q, k, b, a, A_log, dt_bias,
      query_start_loc, total_virtual_seqlen, batch_size, num_k_heads,
      head_k_dim, num_v_heads);

  launch_stage_inverse<T, StateT>(
      queue, kernel_props, xe_core_count, A, query_start_loc,
      total_virtual_seqlen, batch_size, num_v_heads);

  launch_stage_compute_wu<T, StateT>(
      queue, kernel_props, xe_core_count, A, w, u, q, k, v, b, a, A_log,
      dt_bias, query_start_loc, has_initial_state, total_virtual_seqlen,
      batch_size, num_k_heads, head_k_dim, num_v_heads, head_v_dim);

  launch_stage_fwd_o<T, StateT>(
      queue, kernel_props, core_attn_out, o2, w, u, q, k, a, ssm_state,
      ssm_state_stride_0, query_start_loc, cache_indices, has_initial_state,
      batch_size, total_virtual_seqlen, num_k_heads, head_k_dim, num_v_heads,
      head_v_dim);
}


}  // namespace cutlass::gdn::detail
