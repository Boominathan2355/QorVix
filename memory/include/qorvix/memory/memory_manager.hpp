#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "qorvix/memory/page_pool.hpp"
#include "qorvix/memory/tier.hpp"
#include "qorvix/memory/transfer.hpp"

namespace qorvix::memory {

class MemoryManager;

namespace detail {

// Registry record for one buffer. Multiple names may map to the same entry (shared buffers via
// alias); refs counts outstanding TensorRefs across all names.
struct BufferEntry {
  Allocation alloc;
  Tier tier = Tier::HostRam;
  std::size_t bytes = 0;
  int refs = 0;
  std::uint64_t lastUsed = 0;  // LRU clock for eviction ordering
  std::string primaryName;
  std::vector<std::string> names;
};

}  // namespace detail

// RAII handle to a registered buffer. Holding one pins the buffer (it cannot be evicted or
// migrated). Move-only; releases its reference on destruction. Must not outlive the
// MemoryManager that produced it.
class TensorRef {
 public:
  TensorRef() = default;
  TensorRef(TensorRef&& other) noexcept;
  TensorRef& operator=(TensorRef&& other) noexcept;
  ~TensorRef();

  TensorRef(const TensorRef&) = delete;
  TensorRef& operator=(const TensorRef&) = delete;

  bool valid() const noexcept { return entry_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

  std::byte* data() const noexcept;
  std::size_t size() const noexcept;
  Tier tier() const noexcept;
  const std::string& name() const noexcept;

  void reset();  // release early

 private:
  friend class MemoryManager;
  TensorRef(MemoryManager* mgr, std::shared_ptr<detail::BufferEntry> entry)
      : mgr_(mgr), entry_(std::move(entry)) {}

  MemoryManager* mgr_ = nullptr;
  std::shared_ptr<detail::BufferEntry> entry_;
};

struct TierStats {
  PoolStats pool;
  std::size_t buffers = 0;         // distinct buffers resident in this tier
  std::size_t zeroRefBuffers = 0;  // resident but unpinned (eviction candidates)
};

// Global allocation authority (SPEC: Unified Memory Manager + TensorRegistry). Owns one
// PagePool per configured tier and a name -> buffer registry with reference counting.
//
// Behaviors:
//  - create() places a buffer in the preferred tier, falling back down the tier chain
//    (GpuVram -> HostRam -> DiskNvme) if the preferred tier isn't configured or is full even
//    after eviction (memory-aware placement).
//  - Released (zero-ref) buffers stay resident as cache. Under budget pressure the LRU zero-ref
//    buffer in the contended tier is offloaded to the next tier down — or freed from the last
//    tier — until the allocation fits (smart eviction + offloading).
//  - alias() maps a second name to the same buffer (shared buffers). migrate() explicitly moves
//    an unpinned buffer between tiers, preserving contents.
//
// Migration copies with memcpy, which requires host-addressable tiers; that holds for
// HostRam/DiskNvme today. The Phase 4 GPU tier adds explicit transfer ops on its allocator for
// the same flow. Defragmentation is reported (largestFreeBlock) but not yet compacted;
// prefetch/predictive loading arrive with real workloads in later phases.
//
// All public methods are thread-safe (single coarse mutex — revisit under scheduler load).
class MemoryManager {
 public:
  struct TierSpec {
    std::unique_ptr<ISlabAllocator> allocator;
    PoolConfig config;
  };

  // `transfer` moves bytes during migration/eviction; defaults to HostTransferEngine. A CUDA
  // build passes a CudaTransferEngine so a GpuVram tier can migrate to/from host tiers.
  explicit MemoryManager(std::map<Tier, TierSpec> tiers,
                         std::unique_ptr<ITransferEngine> transfer = nullptr);
  ~MemoryManager() = default;

  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;

  // Convenience: HostRam + DiskNvme tiers with default page sizes (the pre-CUDA configuration).
  static std::unique_ptr<MemoryManager> makeHostAndDisk(std::size_t ramBudgetBytes,
                                                        std::size_t diskBudgetBytes,
                                                        const std::filesystem::path& spoolDir);

  // Registers and allocates a new named buffer (refcount 1). Invalid TensorRef on failure
  // (duplicate name, zero size, or all tiers exhausted) — see lastError().
  TensorRef create(const std::string& name, std::size_t bytes, Tier preferred = Tier::HostRam);

  // Pins an existing buffer (refcount +1). Invalid if the name is unknown.
  TensorRef acquire(const std::string& name);

  // Maps `aliasName` to the buffer behind `existing` and returns a pinned ref to it.
  TensorRef alias(const std::string& existing, const std::string& aliasName);

  // Moves an unpinned buffer to `target`, preserving contents. False if the buffer is pinned,
  // unknown, or the target tier is unconfigured/full.
  bool migrate(const std::string& name, Tier target);

  // Removes a name. The underlying buffer is freed once no names and no refs remain.
  bool drop(const std::string& name);

  bool contains(const std::string& name) const;
  std::optional<Tier> tierOf(const std::string& name) const;
  TierStats stats(Tier tier) const;
  const std::string& lastError() const noexcept { return lastError_; }

 private:
  friend class TensorRef;
  using EntryPtr = std::shared_ptr<detail::BufferEntry>;

  void releaseRef(const EntryPtr& entry);  // called by ~TensorRef

  Allocation allocateWithEvictionLocked(Tier tier, std::size_t bytes);
  bool migrateEntryLocked(const EntryPtr& entry, Tier target);
  void freeEntryLocked(const EntryPtr& entry);
  EntryPtr lruZeroRefLocked(Tier tier) const;
  std::optional<Tier> nextLowerConfiguredLocked(Tier tier) const;

  mutable std::mutex mutex_;
  std::map<Tier, std::unique_ptr<PagePool>> pools_;
  std::unique_ptr<ITransferEngine> transfer_;
  std::unordered_map<std::string, EntryPtr> entries_;
  std::uint64_t clock_ = 0;
  std::string lastError_;
};

}  // namespace qorvix::memory
