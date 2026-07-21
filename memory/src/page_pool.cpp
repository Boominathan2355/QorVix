#include "qorvix/memory/page_pool.hpp"

#include <utility>

namespace qorvix::memory {

PagePool::PagePool(std::unique_ptr<ISlabAllocator> slabs, PoolConfig config)
    : slabs_(std::move(slabs)), config_(std::move(config)) {}

PagePool::~PagePool() {
  for (auto& [id, rec] : pages_) {
    slabs_->deallocate(rec.base, rec.size);
  }
}

Allocation PagePool::allocate(std::size_t bytes) {
  if (bytes == 0) {
    lastError_ = "zero-byte allocation";
    return {};
  }

  // First fit across existing pages, oldest first.
  for (auto& [id, rec] : pages_) {
    if (void* p = rec.page->allocate(bytes)) {
      return {p, bytes, id};
    }
  }

  // No page fits — grow. Smallest standard page that fits, or a bespoke huge page.
  std::size_t pageSize = 0;
  bool huge = false;
  for (std::size_t s : config_.pageSizes) {
    if (s >= bytes) {
      pageSize = s;
      break;
    }
  }
  if (pageSize == 0) {
    pageSize = alignUp(bytes, config_.hugeGranularity);
    huge = true;
  }

  if (config_.budgetBytes != 0 && reserved_ + pageSize > config_.budgetBytes) {
    lastError_ = "budget exceeded on " + std::string(tierName(tier())) + " (reserved " +
                 std::to_string(reserved_) + " + page " + std::to_string(pageSize) + " > " +
                 std::to_string(config_.budgetBytes) + ")";
    return {};
  }

  void* base = slabs_->allocate(pageSize);
  if (!base) {
    lastError_ = "backend allocation of " + std::to_string(pageSize) + " bytes failed on " +
                 tierName(tier());
    return {};
  }

  PageRec rec;
  rec.base = base;
  rec.size = pageSize;
  rec.page = std::make_unique<MemoryPage>(base, pageSize);
  rec.huge = huge;

  void* p = rec.page->allocate(bytes);  // cannot fail: fresh page, pageSize >= bytes
  const std::uint32_t id = nextId_++;
  reserved_ += pageSize;
  pages_.emplace(id, std::move(rec));
  return {p, bytes, id};
}

void PagePool::free(const Allocation& allocation) {
  if (!allocation) return;
  auto it = pages_.find(allocation.pageId);
  if (it == pages_.end()) return;
  it->second.page->free(allocation.ptr);

  // Huge pages are bespoke — return them to the backend the moment they empty. Standard pages
  // stay pooled for reuse until trim().
  if (it->second.huge && it->second.page->empty()) {
    releasePage(it);
  }
}

std::size_t PagePool::trim() {
  std::size_t released = 0;
  for (auto it = pages_.begin(); it != pages_.end();) {
    if (it->second.page->empty()) {
      released += it->second.size;
      auto victim = it++;
      releasePage(victim);
    } else {
      ++it;
    }
  }
  return released;
}

void PagePool::releasePage(std::map<std::uint32_t, PageRec>::iterator it) {
  reserved_ -= it->second.size;
  slabs_->deallocate(it->second.base, it->second.size);
  pages_.erase(it);
}

PoolStats PagePool::stats() const {
  PoolStats s;
  s.reservedBytes = reserved_;
  s.budgetBytes = config_.budgetBytes;
  s.pageCount = pages_.size();
  for (const auto& [id, rec] : pages_) {
    s.usedBytes += rec.page->bytesUsed();
    const std::size_t largest = rec.page->largestFreeBlock();
    if (largest > s.largestFreeBlock) s.largestFreeBlock = largest;
  }
  return s;
}

}  // namespace qorvix::memory
