#pragma once

#include "inline_pisa.hpp"

namespace cute {
namespace xe4 {

// Elect one thread in the warp. The elected thread gets its predicate set to true, all others obtain false.
CUTE_HOST_DEVICE uint32_t elect_one_sync()
{
#if defined(SYCL_INTEL_XE4_TARGET)
  uint32_t laneid = get_lane_id();
  return laneid == 0;
#else
  return true;
#endif
}

} // end namespace xe4
} // end namespace cute