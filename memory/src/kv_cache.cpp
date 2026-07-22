#include "qorvix/memory/kv_cache.hpp"

#include <utility>

namespace qorvix::memory {

GlobalKvCache::GlobalKvCache(KvCacheConfig config, std::size_t poolBytes) : config_(config) {
  if (config_.layers <= 0 || config_.kvDim <= 0 || config_.tokensPerPage <= 0) return;

  const std::size_t perPage = pageFloats();
  totalPages_ = poolBytes / (perPage * sizeof(float));
  arena_.assign(totalPages_ * perPage, 0.0f);
  freePages_.reserve(totalPages_);
  // Hand out low page ids first (pop from the back).
  for (std::size_t i = totalPages_; i-- > 0;) freePages_.push_back(static_cast<std::uint32_t>(i));
}

SessionId GlobalKvCache::open() {
  if (totalPages_ == 0) return kInvalidSession;
  const SessionId id = nextSession_++;
  Session s;
  s.pages.resize(config_.layers);
  sessions_.emplace(id, std::move(s));
  return id;
}

void GlobalKvCache::freeSessionPages(Session& s) {
  for (auto& layerPages : s.pages) {
    for (std::uint32_t p : layerPages) freePages_.push_back(p);
    layerPages.clear();
  }
  s.length = 0;
}

void GlobalKvCache::close(SessionId session) {
  auto it = sessions_.find(session);
  if (it == sessions_.end()) return;
  freeSessionPages(it->second);
  sessions_.erase(it);
}

void GlobalKvCache::reset(SessionId session) {
  auto it = sessions_.find(session);
  if (it == sessions_.end()) return;
  freeSessionPages(it->second);
}

int GlobalKvCache::length(SessionId session) const {
  auto it = sessions_.find(session);
  return it == sessions_.end() ? -1 : it->second.length;
}

bool GlobalKvCache::appendToken(SessionId session) {
  auto it = sessions_.find(session);
  if (it == sessions_.end()) return false;
  Session& s = it->second;

  // A new token at position `length` needs a fresh page in every layer when it starts a page.
  if (s.length % config_.tokensPerPage == 0) {
    if (freePages_.size() < static_cast<std::size_t>(config_.layers)) return false;  // exhausted
    for (int layer = 0; layer < config_.layers; ++layer) {
      const std::uint32_t page = freePages_.back();
      freePages_.pop_back();
      s.pages[layer].push_back(page);
    }
  }
  s.length += 1;
  return true;
}

float* GlobalKvCache::slot(SessionId session, int layer, int pos, bool wantV) {
  auto it = sessions_.find(session);
  if (it == sessions_.end()) return nullptr;
  const Session& s = it->second;
  if (layer < 0 || layer >= config_.layers || pos < 0 || pos >= s.length) return nullptr;

  const std::size_t pageIndex = static_cast<std::size_t>(pos) / config_.tokensPerPage;
  const std::size_t within = static_cast<std::size_t>(pos) % config_.tokensPerPage;
  const std::uint32_t page = s.pages[layer][pageIndex];

  float* base = arena_.data() + static_cast<std::size_t>(page) * pageFloats();
  const std::size_t vHalf = static_cast<std::size_t>(config_.tokensPerPage) * config_.kvDim;
  return base + (wantV ? vHalf : 0) + within * config_.kvDim;
}

float* GlobalKvCache::kSlot(SessionId session, int layer, int pos) {
  return slot(session, layer, pos, /*wantV=*/false);
}

float* GlobalKvCache::vSlot(SessionId session, int layer, int pos) {
  return slot(session, layer, pos, /*wantV=*/true);
}

}  // namespace qorvix::memory
