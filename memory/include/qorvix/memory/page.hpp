#pragma once

#include <cstddef>
#include <map>
#include <unordered_map>

#include "qorvix/memory/tier.hpp"

namespace qorvix::memory {

// One memory page: a raw slab carved into aligned sub-allocations by a first-fit free list
// with coalescing on free. Deliberately simple and provably correct; the interface is small
// enough that a faster in-page allocator (buddy/slab bins) can replace it later without
// touching PagePool or MemoryManager.
//
// Does not own the slab — PagePool pairs each MemoryPage with its backing ISlabAllocator slab.
class MemoryPage {
 public:
  MemoryPage(void* base, std::size_t size);

  MemoryPage(const MemoryPage&) = delete;
  MemoryPage& operator=(const MemoryPage&) = delete;

  // Returns a pointer aligned to `alignment` (offset-aligned relative to base; the base itself
  // is at least as aligned by every slab backend), or nullptr if no free block fits.
  void* allocate(std::size_t bytes, std::size_t alignment = kSubAllocAlign);

  // Returns false if `ptr` is not a live allocation from this page.
  bool free(void* ptr) noexcept;

  std::size_t size() const noexcept { return size_; }
  std::size_t bytesUsed() const noexcept { return used_; }
  std::size_t largestFreeBlock() const noexcept;
  bool empty() const noexcept { return live_.empty(); }

 private:
  std::byte* base_;
  std::size_t size_;
  std::size_t used_ = 0;
  std::map<std::size_t, std::size_t> free_;            // offset -> block size, ordered
  std::unordered_map<std::size_t, std::size_t> live_;  // offset -> allocation size
};

}  // namespace qorvix::memory
