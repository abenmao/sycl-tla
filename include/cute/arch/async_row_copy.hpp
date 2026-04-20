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

#pragma once

#include <cute/arch/asm_helper.hpp>
#include <cute/arch/mma_xe4_desc.hpp>

namespace cute {
namespace detail {

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Asynchronous Row Copy: Global Memory → Shared Local Memory (Load)
///
/// Addressing Mode: .a64 (64-bit absolute address)
///
/// The .a64 mode uses a 64-bit absolute address to specify the source row in global memory.
/// This provides full 64-bit address space access without requiring a separate base pointer.
///
/// Instruction format:
///   async_row_copy.shared_workgroup.global.linear.<RowSize>.a64.<BitWidth>.uint.<FillMode>.<CacheCtrl>.abarrier
///     [slm_ptr], [gmem_addr], [abar_ptr], size
///
/// Parameters:
///   RowSize  - Row size in bytes (16, 32, 64, 128, 256, 512, 1024, 2048)
///   BitWidth - Element bit-width (4, 6, 8, 16, 32, 64)
///   FillMode - Fill mode for out-of-bounds (.zero or .nan)
///   CacheCtrl - Cache policy (.L2c.L3uc, .L2c.L3c, .L2uc.L3uc, .L2uc.L3c)
///   slm_ptr  - Destination address in shared local memory
///   gmem_addr - Source 64-bit absolute address in global memory
///   abar_ptr - Arrival barrier pointer for completion tracking
///   size     - Number of bytes to copy
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType, uint32_t RowSize, int BitWidth, CacheCtrl CC = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
struct AsyncRowCopyGlobal2SLM_A64_Impl {
  CUTE_HOST_DEVICE static void copy(void* slm_ptr, uint64_t gmem_addr, uint64_t const* abar_ptr, uint32_t size) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile(
      ("async_row_copy.shared_workgroup.global.linear."+_s<RowSize>+".a64."+_bw<BitWidth>+_dt<DataType>+_fl<FM>+_cc<CC>+".abarrier [%0], [%1], [%2], %3;")
      ::"r"(slm_ptr), "r"(gmem_addr), "r"(abar_ptr), "r"(size));
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Asynchronous Row Copy with Cluster Multicast: Global Memory → Shared Local Memory
///
/// Addressing Mode: .a64 with cluster multicast (.shared_cluster)
///
/// The multicast variant broadcasts the loaded data to multiple workgroups within a cluster.
/// The wg_mask parameter specifies which workgroups in the cluster receive the data.
///
/// Instruction format:
///   async_row_copy.shared_cluster.global.linear.<RowSize>.a64.<BitWidth>.uint.<FillMode>.<CacheCtrl>.abarrier
///     [slm_ptr], [gmem_addr], [abar_ptr], size, wg_mask
///
/// Additional parameter:
///   wg_mask - Workgroup mask bitmap (e.g., 0xF = broadcast to all 4 WGs in a 2x2 cluster)
///
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType, uint32_t RowSize, int BitWidth, CacheCtrl CC = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
struct AsyncRowCopyGlobal2SLM_A64_Multicast_Impl {
  CUTE_HOST_DEVICE static void copy(void* slm_ptr, uint64_t gmem_addr, uint64_t const* abar_ptr, uint32_t size, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile(
      ("async_row_copy.shared_cluster.global.linear."+_s<RowSize>+".a64."+_bw<BitWidth>+_dt<DataType>+_fl<FM>+_cc<CC>+".abarrier [%0], [%1], [%2], %3, %4;")
      ::"r"(slm_ptr), "r"(gmem_addr), "r"(abar_ptr), "r"(size), "r"(wg_mask));
#endif
  }
};

template <uint32_t RowSize>
struct AsyncRowCopyGlobal2SLM_A64
{
  // Standard workgroup-local copy
  template <typename DataType, CacheCtrl CC = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
  CUTE_HOST_DEVICE static void
  Copy(DataType* slm_ptr, uint64_t gmem_addr, uint32_t size, uint64_t const* abar_ptr,
       CacheHint<CC> = {}, FillMode<FM> = {})
  {
    AsyncRowCopyGlobal2SLM_A64_Impl<DataType, RowSize, sizeof_bits_v<DataType>, CC, FM>::copy(slm_ptr, gmem_addr, abar_ptr, size);
  }

  // Cluster multicast variant
  template <typename DataType, CacheCtrl CC = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
  CUTE_HOST_DEVICE static void
  Copy(DataType* slm_ptr, uint64_t gmem_addr, uint32_t size, uint64_t const* abar_ptr, uint32_t wg_mask,
       CacheHint<CC> = {}, FillMode<FM> = {})
  {
    AsyncRowCopyGlobal2SLM_A64_Multicast_Impl<DataType, RowSize, sizeof_bits_v<DataType>, CC, FM>::copy(slm_ptr, gmem_addr, abar_ptr, size, wg_mask);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Asynchronous Row Copy: Shared Local Memory → Global Memory (Store)
///
/// Addressing Mode: .a64 (64-bit absolute address)
///
/// Stores data from shared local memory back to global memory at a 64-bit absolute address.
///
/// Instruction format:
///   async_row_copy.global.shared_workgroup.linear.<RowSize>.a64.<BitWidth>.L2wb.L3uc.abarrier
///     [gmem_addr], [slm_ptr], [abar_ptr], size
///
/// Cache policy: L2wb (write-back in L2), L3uc (uncached in L3)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <uint32_t RowSize, int BitWidth, CacheCtrl CC = CacheCtrl::L2wb_L3uc>
struct AsyncRowCopySLM2Global_A64_Impl {
  CUTE_HOST_DEVICE static void copy(uint64_t gmem_addr, void* slm_ptr, uint64_t const* abar_ptr, uint32_t size) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile(
      ("async_row_copy.global.shared_workgroup.linear."+_s<RowSize>+".a64."+_bw<BitWidth>+_cc<CC>+".abarrier [%0], [%1], [%2], %3;")
      ::"r"(gmem_addr), "r"(slm_ptr), "r"(abar_ptr), "r"(size));
#endif
  }
};

template <uint32_t RowSize>
struct AsyncRowCopySLM2Global_A64
{
  template <typename DataType, CacheCtrl CC = CacheCtrl::L2wb_L3uc>
  CUTE_HOST_DEVICE static void
  Copy(uint64_t gmem_addr, DataType* slm_ptr, uint32_t size, uint64_t const* abar_ptr,
       CacheHint<CC> = {})
  {
    AsyncRowCopySLM2Global_A64_Impl<RowSize, sizeof_bits_v<DataType>, CC>::copy(gmem_addr, slm_ptr, abar_ptr, size);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Asynchronous Row Copy: Global Memory → Shared Local Memory (Load)
///
/// Addressing Mode: .a32s (32-bit base pointer + signed 32-bit offset)
///
/// The .a32s mode uses a 32-bit base pointer plus a signed 32-bit byte offset to compute the
/// source address: effective_address = gmem_ptr + offset.
///
/// Instruction format:
///   async_row_copy.shared_workgroup.global.linear.<RowSize>.a32s.<BitWidth>.uint.zero.L2c.L3uc.abarrier
///     [slm_ptr], [gmem_ptr], [abar_ptr], offset, size
///
/// Parameters:
///   gmem_ptr - 32-bit base pointer in global memory
///   offset   - Signed 32-bit byte offset relative to gmem_ptr
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType, uint32_t RowSize, int BitWidth, CacheCtrl CC = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
struct AsyncRowCopyGlobal2SLM_A32S_Impl {
  CUTE_HOST_DEVICE static void copy(void* slm_ptr, void* gmem_ptr, uint64_t const* abar_ptr, int32_t offset, uint32_t size) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile(
      ("async_row_copy.shared_workgroup.global.linear."+_s<RowSize>+".a32s."+_bw<BitWidth>+_dt<DataType>+_fl<FM>+_cc<CC>+".abarrier [%0], [%1], [%2], %3, %4;")
      ::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Asynchronous Row Copy with Cluster Multicast: Global Memory → Shared Local Memory
///
/// Addressing Mode: .a32s with cluster multicast (.shared_cluster)
///
/// The multicast variant broadcasts the loaded data to multiple workgroups within a cluster.
/// Operand order matches the reference: slm_ptr, [gmem_ptr], [abar_ptr], offset, size, wg_mask
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType, uint32_t RowSize, int BitWidth, CacheCtrl CC = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
struct AsyncRowCopyGlobal2SLM_A32S_Multicast_Impl {
  CUTE_HOST_DEVICE static void copy(void* slm_ptr, void* gmem_ptr, uint64_t const* abar_ptr, int32_t offset, uint32_t size, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile(
      ("async_row_copy.shared_cluster.global.linear."+_s<RowSize>+".a32s."+_bw<BitWidth>+_dt<DataType>+_fl<FM>+_cc<CC>+".abarrier [%0], [%1], [%2], %3, %4, %5;")
      ::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset), "r"(size), "r"(wg_mask));
#endif
  }
};

template <uint32_t RowSize>
struct AsyncRowCopyGlobal2SLM_A32S
{
  // Standard workgroup-local copy
  template <typename DataType, CacheCtrl CC = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
  CUTE_HOST_DEVICE static void
  Copy(DataType* slm_ptr, DataType* gmem_ptr, int32_t offset, uint32_t size, uint64_t const* abar_ptr,
       CacheHint<CC> = {}, FillMode<FM> = {})
  {
    AsyncRowCopyGlobal2SLM_A32S_Impl<DataType, RowSize, sizeof_bits_v<DataType>, CC, FM>::copy(slm_ptr, gmem_ptr, abar_ptr, offset, size);
  }

  // Cluster multicast variant
  template <typename DataType, CacheCtrl CC = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
  CUTE_HOST_DEVICE static void
  Copy(DataType* slm_ptr, DataType* gmem_ptr, int32_t offset, uint32_t size, uint64_t const* abar_ptr, uint32_t wg_mask,
       CacheHint<CC> = {}, FillMode<FM> = {})
  {
    AsyncRowCopyGlobal2SLM_A32S_Multicast_Impl<DataType, RowSize, sizeof_bits_v<DataType>, CC, FM>::copy(slm_ptr, gmem_ptr, abar_ptr, offset, size, wg_mask);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Asynchronous Row Copy: Shared Local Memory → Global Memory (Store)
///
/// Addressing Mode: .a32s (32-bit base pointer + signed 32-bit offset)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <uint32_t RowSize, int BitWidth, CacheCtrl CC = CacheCtrl::L2wb_L3uc>
struct AsyncRowCopySLM2Global_A32S_Impl {
  CUTE_HOST_DEVICE static void copy(void* gmem_ptr, void* slm_ptr, uint64_t const* abar_ptr, int32_t offset, uint32_t size) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile(
      ("async_row_copy.global.shared_workgroup.linear."+_s<RowSize>+".a32s."+_bw<BitWidth>+_cc<CC>+".abarrier [%0], [%1], [%2], %3, %4;")
      ::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
#endif
  }
};

template <uint32_t RowSize>
struct AsyncRowCopySLM2Global_A32S
{
  template <typename DataType, CacheCtrl CC = CacheCtrl::L2wb_L3uc>
  CUTE_HOST_DEVICE static void
  Copy(DataType* gmem_ptr, DataType* slm_ptr, int32_t offset, uint32_t size, uint64_t const* abar_ptr,
       CacheHint<CC> = {})
  {
    AsyncRowCopySLM2Global_A32S_Impl<RowSize, sizeof_bits_v<DataType>, CC>::copy(gmem_ptr, slm_ptr, abar_ptr, offset, size);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Asynchronous Row Copy: Global Memory → Shared Local Memory (Load)
///
/// Addressing Mode: .a32u (32-bit base pointer + unsigned 32-bit offset)
///
/// The .a32u mode uses a 32-bit base pointer plus an unsigned 32-bit byte offset to compute the
/// source address: effective_address = gmem_ptr + offset.
///
/// Instruction format:
///   async_row_copy.shared_workgroup.global.linear.<RowSize>.a32u.<BitWidth>.uint.zero.L2c.L3uc.abarrier
///     [slm_ptr], [gmem_ptr], [abar_ptr], offset, size
///
/// Parameters:
///   gmem_ptr - 32-bit base pointer in global memory
///   offset   - Unsigned 32-bit byte offset relative to gmem_ptr
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType, uint32_t RowSize, int BitWidth, CacheCtrl CC = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
struct AsyncRowCopyGlobal2SLM_A32U_Impl {
  CUTE_HOST_DEVICE static void copy(void* slm_ptr, void* gmem_ptr, uint64_t const* abar_ptr, uint32_t offset, uint32_t size) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile(
      ("async_row_copy.shared_workgroup.global.linear."+_s<RowSize>+".a32u."+_bw<BitWidth>+_dt<DataType>+_fl<FM>+_cc<CC>+".abarrier [%0], [%1], [%2], %3, %4;")
      ::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Asynchronous Row Copy with Cluster Multicast: Global Memory → Shared Local Memory
///
/// Addressing Mode: .a32u with cluster multicast (.shared_cluster)
///
/// The multicast variant broadcasts the loaded data to multiple workgroups within a cluster.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType, uint32_t RowSize, int BitWidth, CacheCtrl CC = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
struct AsyncRowCopyGlobal2SLM_A32U_Multicast_Impl {
  CUTE_HOST_DEVICE static void copy(void* slm_ptr, void* gmem_ptr, uint64_t const* abar_ptr, uint32_t offset, uint32_t size, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile(
      ("async_row_copy.shared_cluster.global.linear."+_s<RowSize>+".a32u."+_bw<BitWidth>+_dt<DataType>+_fl<FM>+_cc<CC>+".abarrier [%0], [%1], [%2], %3, %4, %5;")
      ::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset), "r"(size), "r"(wg_mask));
#endif
  }
};

template <uint32_t RowSize>
struct AsyncRowCopyGlobal2SLM_A32U
{
  // Standard workgroup-local copy
  template <typename DataType, CacheCtrl CC = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
  CUTE_HOST_DEVICE static void
  Copy(DataType* slm_ptr, DataType* gmem_ptr, uint32_t offset, uint32_t size, uint64_t const* abar_ptr,
       CacheHint<CC> = {}, FillMode<FM> = {})
  {
    AsyncRowCopyGlobal2SLM_A32U_Impl<DataType, RowSize, sizeof_bits_v<DataType>, CC, FM>::copy(slm_ptr, gmem_ptr, abar_ptr, offset, size);
  }

  // Cluster multicast variant
  template <typename DataType, CacheCtrl CC = CacheCtrl::L2c_L3uc, FillMethod FM = FillMethod::Zero>
  CUTE_HOST_DEVICE static void
  Copy(DataType* slm_ptr, DataType* gmem_ptr, uint32_t offset, uint32_t size, uint64_t const* abar_ptr, uint32_t wg_mask,
       CacheHint<CC> = {}, FillMode<FM> = {})
  {
    AsyncRowCopyGlobal2SLM_A32U_Multicast_Impl<DataType, RowSize, sizeof_bits_v<DataType>, CC, FM>::copy(slm_ptr, gmem_ptr, abar_ptr, offset, size, wg_mask);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Asynchronous Row Copy: Shared Local Memory → Global Memory (Store)
///
/// Addressing Mode: .a32u (32-bit base pointer + unsigned 32-bit offset)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <uint32_t RowSize, int BitWidth, CacheCtrl CC = CacheCtrl::L2wb_L3uc>
struct AsyncRowCopySLM2Global_A32U_Impl {
  CUTE_HOST_DEVICE static void copy(void* gmem_ptr, void* slm_ptr, uint64_t const* abar_ptr, uint32_t offset, uint32_t size) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile(
      ("async_row_copy.global.shared_workgroup.linear."+_s<RowSize>+".a32u."+_bw<BitWidth>+_cc<CC>+".abarrier [%0], [%1], [%2], %3, %4;")
      ::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
#endif
  }
};

template <uint32_t RowSize>
struct AsyncRowCopySLM2Global_A32U
{
  template <typename DataType, CacheCtrl CC = CacheCtrl::L2wb_L3uc>
  CUTE_HOST_DEVICE static void
  Copy(DataType* gmem_ptr, DataType* slm_ptr, uint32_t offset, uint32_t size, uint64_t const* abar_ptr,
       CacheHint<CC> = {})
  {
    AsyncRowCopySLM2Global_A32U_Impl<RowSize, sizeof_bits_v<DataType>, CC>::copy(gmem_ptr, slm_ptr, abar_ptr, offset, size);
  }
};

}  // namespace detail
}  // namespace cute
