#pragma once

#include <cstddef>
#include <cstdint>

namespace qorvix::memory {

inline constexpr std::size_t kKiB = 1024;
inline constexpr std::size_t kMiB = 1024 * kKiB;

// Storage tiers, fastest first. Offload/eviction flows toward higher enum values
// (GpuVram -> HostRam -> DiskNvme); there is nothing below DiskNvme.
enum class Tier : std::uint8_t {
  GpuVram = 0,
  HostRam = 1,
  DiskNvme = 2,
};

inline const char* tierName(Tier t) noexcept {
  switch (t) {
    case Tier::GpuVram: return "gpu-vram";
    case Tier::HostRam: return "host-ram";
    case Tier::DiskNvme: return "disk-nvme";
  }
  return "unknown";
}

// Standard page sizes (SPEC: 4/8/16/32/64 MB). Requests larger than the largest standard size
// get a dedicated "huge" page rounded up to PoolConfig::hugeGranularity and released as soon as
// it empties (huge pages are bespoke; standard pages are pooled).
inline constexpr std::size_t kStandardPageSizes[] = {
    4 * kMiB, 8 * kMiB, 16 * kMiB, 32 * kMiB, 64 * kMiB,
};

// Alignment of every sub-allocation handed out of a page. 256 bytes satisfies CUDA device
// pointer alignment and keeps tensor rows tensor-core friendly, so host-staged buffers can be
// copied to the GPU tier without re-layout in Phase 4.
inline constexpr std::size_t kSubAllocAlign = 256;

constexpr std::size_t alignUp(std::size_t value, std::size_t alignment) noexcept {
  return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

}  // namespace qorvix::memory
