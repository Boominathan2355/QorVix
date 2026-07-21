#pragma once

#include <cstddef>
#include <cstring>

#include "qorvix/memory/tier.hpp"

namespace qorvix::memory {

// Moves bytes between two buffers that may live in different tiers. This is the seam that keeps
// the memory manager tier-agnostic: the default engine assumes host-addressable memory (memcpy),
// and the CUDA backend supplies an engine that routes GPU tiers through cudaMemcpy (Phase 4).
class ITransferEngine {
 public:
  virtual ~ITransferEngine() = default;
  virtual void copy(void* dst, Tier dstTier, const void* src, Tier srcTier,
                    std::size_t bytes) = 0;
};

// All tiers host-addressable (HostRam + DiskNvme). Correct until a GpuVram tier exists.
class HostTransferEngine final : public ITransferEngine {
 public:
  void copy(void* dst, Tier /*dstTier*/, const void* src, Tier /*srcTier*/,
            std::size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
};

}  // namespace qorvix::memory
