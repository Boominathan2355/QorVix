#include "qorvix/memory/page.hpp"

namespace qorvix::memory {

MemoryPage::MemoryPage(void* base, std::size_t size)
    : base_(static_cast<std::byte*>(base)), size_(size) {
  free_.emplace(0, size);
}

void* MemoryPage::allocate(std::size_t bytes, std::size_t alignment) {
  if (bytes == 0) return nullptr;
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    const std::size_t off = it->first;
    const std::size_t blockSize = it->second;
    const std::size_t alignedOff = alignUp(off, alignment);
    const std::size_t pad = alignedOff - off;
    if (blockSize < pad || blockSize - pad < bytes) continue;

    const std::size_t tail = blockSize - pad - bytes;
    free_.erase(it);
    if (pad > 0) free_.emplace(off, pad);                    // leading alignment gap stays free
    if (tail > 0) free_.emplace(alignedOff + bytes, tail);   // remainder stays free
    live_.emplace(alignedOff, bytes);
    used_ += bytes;
    return base_ + alignedOff;
  }
  return nullptr;
}

bool MemoryPage::free(void* ptr) noexcept {
  const auto* p = static_cast<std::byte*>(ptr);
  if (p < base_ || p >= base_ + size_) return false;
  std::size_t off = static_cast<std::size_t>(p - base_);

  auto it = live_.find(off);
  if (it == live_.end()) return false;
  std::size_t blockSize = it->second;
  live_.erase(it);
  used_ -= blockSize;

  // Coalesce with the following free block, if adjacent.
  auto next = free_.lower_bound(off);
  if (next != free_.end() && next->first == off + blockSize) {
    blockSize += next->second;
    free_.erase(next);
  }
  // Coalesce with the preceding free block, if adjacent.
  auto prev = free_.lower_bound(off);
  if (prev != free_.begin()) {
    --prev;
    if (prev->first + prev->second == off) {
      off = prev->first;
      blockSize += prev->second;
      free_.erase(prev);
    }
  }
  free_.emplace(off, blockSize);
  return true;
}

std::size_t MemoryPage::largestFreeBlock() const noexcept {
  std::size_t largest = 0;
  for (const auto& [off, size] : free_) {
    if (size > largest) largest = size;
  }
  return largest;
}

}  // namespace qorvix::memory
