#pragma once

#include <cstddef>

#include "qorvix/memory/tier.hpp"

namespace qorvix::memory {

// Backend that provides raw slabs (whole memory pages) for one tier. Sub-allocation never
// happens here — PagePool carves slabs into tensor-sized pieces. Implementations are not
// required to be thread-safe; PagePool/MemoryManager serialize access.
//
// Tier backends:
//   HostRam  — HostSlabAllocator (below)
//   DiskNvme — DiskSlabAllocator (disk_allocator.hpp)
//   GpuVram  — arrives with the CUDA backend in Phase 4, behind this same interface
//              (cudaMalloc/cuMemCreate slabs). Everything above the allocator is
//              tier-agnostic, so the GPU tier drops in without touching pool/manager code.
class ISlabAllocator {
 public:
  virtual ~ISlabAllocator() = default;

  // Returns a slab of at least `bytes` bytes, or nullptr on failure.
  virtual void* allocate(std::size_t bytes) = 0;
  virtual void deallocate(void* ptr, std::size_t bytes) noexcept = 0;
  virtual Tier tier() const noexcept = 0;
};

// Tier 2: OS-page-aligned host RAM.
class HostSlabAllocator final : public ISlabAllocator {
 public:
  // 4 KiB slab alignment: OS page boundary, and (being a multiple of kSubAllocAlign) makes
  // sub-allocation offset alignment equal pointer alignment.
  static constexpr std::size_t kSlabAlign = 4096;

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr, std::size_t bytes) noexcept override;
  Tier tier() const noexcept override { return Tier::HostRam; }
};

}  // namespace qorvix::memory
