#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>

#include "qorvix/memory/slab_allocator.hpp"

namespace qorvix::memory {

// Tier 3: NVMe/disk-backed slabs. Each slab is a spool file mapped read-write, so tier-3 memory
// is directly addressable (plain memcpy migrates buffers between HostRam and DiskNvme) while
// the OS pages cold data out of RAM. Files are deleted on deallocate; anything left over is
// cleaned up on destruction. This is the storage engine behind the spec's DiskCacheManager.
class DiskSlabAllocator final : public ISlabAllocator {
 public:
  explicit DiskSlabAllocator(std::filesystem::path spoolDir);
  ~DiskSlabAllocator() override;

  DiskSlabAllocator(const DiskSlabAllocator&) = delete;
  DiskSlabAllocator& operator=(const DiskSlabAllocator&) = delete;

  void* allocate(std::size_t bytes) override;
  void deallocate(void* ptr, std::size_t bytes) noexcept override;
  Tier tier() const noexcept override { return Tier::DiskNvme; }

  const std::filesystem::path& spoolDir() const noexcept { return dir_; }
  std::size_t liveSlabs() const noexcept { return slabs_.size(); }
  const std::string& lastError() const noexcept { return error_; }

 private:
  struct Slab {
    std::filesystem::path path;
    std::size_t size = 0;
#if defined(_WIN32)
    void* fileHandle = nullptr;     // HANDLE
    void* mappingHandle = nullptr;  // HANDLE
#else
    int fd = -1;
#endif
  };

  void release(void* ptr, Slab& slab) noexcept;

  std::filesystem::path dir_;
  std::uint64_t nextId_ = 0;
  std::unordered_map<void*, Slab> slabs_;
  std::string error_;
};

}  // namespace qorvix::memory
