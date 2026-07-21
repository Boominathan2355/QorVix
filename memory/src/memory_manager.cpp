#include "qorvix/memory/memory_manager.hpp"

#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <utility>

#include "qorvix/memory/disk_allocator.hpp"

namespace qorvix::memory {

// ---- TensorRef -----------------------------------------------------------------------------

TensorRef::TensorRef(TensorRef&& other) noexcept
    : mgr_(std::exchange(other.mgr_, nullptr)), entry_(std::move(other.entry_)) {}

TensorRef& TensorRef::operator=(TensorRef&& other) noexcept {
  if (this != &other) {
    reset();
    mgr_ = std::exchange(other.mgr_, nullptr);
    entry_ = std::move(other.entry_);
  }
  return *this;
}

TensorRef::~TensorRef() { reset(); }

void TensorRef::reset() {
  if (mgr_ && entry_) mgr_->releaseRef(entry_);
  mgr_ = nullptr;
  entry_.reset();
}

std::byte* TensorRef::data() const noexcept {
  return entry_ ? static_cast<std::byte*>(entry_->alloc.ptr) : nullptr;
}

std::size_t TensorRef::size() const noexcept { return entry_ ? entry_->bytes : 0; }

Tier TensorRef::tier() const noexcept { return entry_ ? entry_->tier : Tier::HostRam; }

const std::string& TensorRef::name() const noexcept {
  static const std::string kEmpty;
  return entry_ ? entry_->primaryName : kEmpty;
}

// ---- MemoryManager -------------------------------------------------------------------------

MemoryManager::MemoryManager(std::map<Tier, TierSpec> tiers) {
  for (auto& [tier, spec] : tiers) {
    pools_.emplace(tier,
                   std::make_unique<PagePool>(std::move(spec.allocator), std::move(spec.config)));
  }
}

std::unique_ptr<MemoryManager> MemoryManager::makeHostAndDisk(
    std::size_t ramBudgetBytes, std::size_t diskBudgetBytes,
    const std::filesystem::path& spoolDir) {
  std::map<Tier, TierSpec> tiers;
  tiers[Tier::HostRam] = {std::make_unique<HostSlabAllocator>(), PoolConfig{.budgetBytes = ramBudgetBytes}};
  tiers[Tier::DiskNvme] = {std::make_unique<DiskSlabAllocator>(spoolDir),
                           PoolConfig{.budgetBytes = diskBudgetBytes}};
  return std::make_unique<MemoryManager>(std::move(tiers));
}

TensorRef MemoryManager::create(const std::string& name, std::size_t bytes, Tier preferred) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (bytes == 0) {
    lastError_ = "zero-byte tensor '" + name + "'";
    return {};
  }
  if (entries_.count(name)) {
    lastError_ = "tensor '" + name + "' already exists";
    return {};
  }

  // Try the preferred tier, then fall down the chain (memory-aware placement).
  for (auto t = std::optional<Tier>(preferred); t; t = nextLowerConfiguredLocked(*t)) {
    if (!pools_.count(*t)) continue;  // preferred tier may be unconfigured (e.g. no GPU yet)
    Allocation alloc = allocateWithEvictionLocked(*t, bytes);
    if (!alloc) continue;

    auto entry = std::make_shared<detail::BufferEntry>();
    entry->alloc = alloc;
    entry->tier = *t;
    entry->bytes = bytes;
    entry->refs = 1;
    entry->lastUsed = ++clock_;
    entry->primaryName = name;
    entry->names.push_back(name);
    entries_.emplace(name, entry);
    return TensorRef(this, std::move(entry));
  }

  lastError_ = "no tier could hold " + std::to_string(bytes) + " bytes for '" + name + "'";
  return {};
}

TensorRef MemoryManager::acquire(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(name);
  if (it == entries_.end()) {
    lastError_ = "unknown tensor '" + name + "'";
    return {};
  }
  it->second->refs += 1;
  it->second->lastUsed = ++clock_;
  return TensorRef(this, it->second);
}

TensorRef MemoryManager::alias(const std::string& existing, const std::string& aliasName) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(existing);
  if (it == entries_.end()) {
    lastError_ = "unknown tensor '" + existing + "'";
    return {};
  }
  if (entries_.count(aliasName)) {
    lastError_ = "name '" + aliasName + "' already exists";
    return {};
  }
  EntryPtr entry = it->second;
  entry->names.push_back(aliasName);
  entry->refs += 1;
  entry->lastUsed = ++clock_;
  entries_.emplace(aliasName, entry);
  return TensorRef(this, std::move(entry));
}

bool MemoryManager::migrate(const std::string& name, Tier target) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(name);
  if (it == entries_.end()) {
    lastError_ = "unknown tensor '" + name + "'";
    return false;
  }
  EntryPtr entry = it->second;
  if (entry->refs != 0) {
    lastError_ = "tensor '" + name + "' is pinned (refs=" + std::to_string(entry->refs) + ")";
    return false;
  }
  if (entry->tier == target) return true;
  if (!pools_.count(target)) {
    lastError_ = std::string("tier ") + tierName(target) + " is not configured";
    return false;
  }
  return migrateEntryLocked(entry, target);
}

bool MemoryManager::drop(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(name);
  if (it == entries_.end()) return false;
  EntryPtr entry = it->second;
  entries_.erase(it);
  std::erase(entry->names, name);
  if (entry->names.empty() && entry->refs == 0) {
    pools_.at(entry->tier)->free(entry->alloc);
    entry->alloc = {};
  }
  return true;
}

bool MemoryManager::contains(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return entries_.count(name) != 0;
}

std::optional<Tier> MemoryManager::tierOf(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(name);
  if (it == entries_.end()) return std::nullopt;
  return it->second->tier;
}

TierStats MemoryManager::stats(Tier tier) const {
  std::lock_guard<std::mutex> lock(mutex_);
  TierStats s;
  auto it = pools_.find(tier);
  if (it != pools_.end()) s.pool = it->second->stats();

  std::unordered_set<const detail::BufferEntry*> seen;
  for (const auto& [name, entry] : entries_) {
    if (entry->tier != tier || !seen.insert(entry.get()).second) continue;
    s.buffers += 1;
    if (entry->refs == 0) s.zeroRefBuffers += 1;
  }
  return s;
}

void MemoryManager::releaseRef(const EntryPtr& entry) {
  std::lock_guard<std::mutex> lock(mutex_);
  entry->refs -= 1;
  entry->lastUsed = ++clock_;
  // If every name was dropped while refs were still held, free once the last ref goes away.
  if (entry->refs == 0 && entry->names.empty() && entry->alloc) {
    pools_.at(entry->tier)->free(entry->alloc);
    entry->alloc = {};
  }
}

Allocation MemoryManager::allocateWithEvictionLocked(Tier tier, std::size_t bytes) {
  PagePool& pool = *pools_.at(tier);
  for (;;) {
    if (Allocation a = pool.allocate(bytes)) return a;

    // Full: offload the least-recently-used unpinned buffer and retry. Each iteration removes
    // one candidate from this tier, so the loop terminates.
    EntryPtr victim = lruZeroRefLocked(tier);
    if (!victim) {
      lastError_ = pool.lastError();
      return {};
    }
    if (auto lower = nextLowerConfiguredLocked(tier); lower && migrateEntryLocked(victim, *lower)) {
      // offloaded one tier down
    } else {
      freeEntryLocked(victim);  // last tier (or lower tiers full): buffer is dropped
    }
    pool.trim();
  }
}

bool MemoryManager::migrateEntryLocked(const EntryPtr& entry, Tier target) {
  Allocation dst = allocateWithEvictionLocked(target, entry->bytes);
  if (!dst) return false;

  // Both current tiers are host-addressable (see class comment); the GPU tier will route this
  // through its allocator's transfer ops in Phase 4.
  std::memcpy(dst.ptr, entry->alloc.ptr, entry->bytes);

  PagePool& src = *pools_.at(entry->tier);
  src.free(entry->alloc);
  src.trim();
  entry->alloc = dst;
  entry->tier = target;
  return true;
}

void MemoryManager::freeEntryLocked(const EntryPtr& entry) {
  for (const std::string& n : entry->names) entries_.erase(n);
  entry->names.clear();
  if (entry->alloc) {
    pools_.at(entry->tier)->free(entry->alloc);
    entry->alloc = {};
  }
}

MemoryManager::EntryPtr MemoryManager::lruZeroRefLocked(Tier tier) const {
  EntryPtr best;
  for (const auto& [name, entry] : entries_) {
    if (entry->tier != tier || entry->refs != 0 || !entry->alloc) continue;
    if (!best || entry->lastUsed < best->lastUsed) best = entry;
  }
  return best;
}

std::optional<Tier> MemoryManager::nextLowerConfiguredLocked(Tier tier) const {
  auto next = [](Tier t) -> std::optional<Tier> {
    switch (t) {
      case Tier::GpuVram: return Tier::HostRam;
      case Tier::HostRam: return Tier::DiskNvme;
      case Tier::DiskNvme: return std::nullopt;
    }
    return std::nullopt;
  };
  for (auto t = next(tier); t; t = next(*t)) {
    if (pools_.count(*t)) return t;
  }
  return std::nullopt;
}

}  // namespace qorvix::memory
