#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "qorvix/memory/page.hpp"
#include "qorvix/memory/slab_allocator.hpp"
#include "qorvix/memory/tier.hpp"

namespace qorvix::memory {

struct PoolConfig {
  // Ascending standard page sizes. Production uses kStandardPageSizes; tests substitute
  // KiB-scale sizes to exercise the same logic cheaply.
  std::vector<std::size_t> pageSizes{std::begin(kStandardPageSizes), std::end(kStandardPageSizes)};

  // Cap on total slab bytes reserved from the backend (0 = unlimited). Exceeding it fails the
  // allocation; MemoryManager reacts by evicting and retrying.
  std::size_t budgetBytes = 0;

  // Granularity for "huge" pages (requests larger than the largest standard size).
  std::size_t hugeGranularity = 4 * kMiB;
};

// A sub-allocation handed out by a PagePool. Only meaningful to the pool that produced it.
struct Allocation {
  void* ptr = nullptr;
  std::size_t bytes = 0;
  std::uint32_t pageId = 0;

  explicit operator bool() const noexcept { return ptr != nullptr; }
};

struct PoolStats {
  std::size_t reservedBytes = 0;   // slab bytes held from the backend
  std::size_t usedBytes = 0;       // live sub-allocation payload
  std::size_t pageCount = 0;
  std::size_t budgetBytes = 0;
  std::size_t largestFreeBlock = 0;  // fragmentation signal
};

// All memory of one tier, organized as pages (SPEC: no direct tensor allocations — everything
// goes through the page manager). First-fit across existing pages; grows by the smallest
// standard page that fits; oversized requests get bespoke huge pages that are released the
// moment they empty. Standard pages are pooled (kept when empty) until trim() is called.
// Not thread-safe; MemoryManager serializes access.
class PagePool {
 public:
  PagePool(std::unique_ptr<ISlabAllocator> slabs, PoolConfig config);
  ~PagePool();

  PagePool(const PagePool&) = delete;
  PagePool& operator=(const PagePool&) = delete;

  Allocation allocate(std::size_t bytes);
  void free(const Allocation& allocation);

  // Releases empty pages back to the backend; returns the number of bytes released.
  std::size_t trim();

  PoolStats stats() const;
  Tier tier() const noexcept { return slabs_->tier(); }
  const std::string& lastError() const noexcept { return lastError_; }

 private:
  struct PageRec {
    void* base = nullptr;
    std::size_t size = 0;
    std::unique_ptr<MemoryPage> page;
    bool huge = false;
  };

  void releasePage(std::map<std::uint32_t, PageRec>::iterator it);

  std::unique_ptr<ISlabAllocator> slabs_;
  PoolConfig config_;
  std::map<std::uint32_t, PageRec> pages_;  // ordered by id == insertion order (oldest first)
  std::uint32_t nextId_ = 1;
  std::size_t reserved_ = 0;
  std::string lastError_;
};

}  // namespace qorvix::memory
