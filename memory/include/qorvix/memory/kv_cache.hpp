#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace qorvix::memory {

// Configuration for a GlobalKvCache. A "page" holds `tokensPerPage` tokens of K and V for a
// single layer, so a sequence of length L needs ceil(L/tokensPerPage) pages per layer.
struct KvCacheConfig {
  int layers = 0;
  int kvDim = 0;           // n_kv_heads * head_dim (elements of K or V per token per layer)
  int tokensPerPage = 128;
};

using SessionId = std::uint32_t;
inline constexpr SessionId kInvalidSession = 0;

// Paged, multi-session KV cache (SPEC "Unified KV Cache": shared pool, paged KV). Pages are
// drawn from a single pre-sized arena and mapped to (session, layer, token) via per-session page
// tables — so distinct sessions are isolated, a sequence's KV need not be physically contiguous,
// and closing a session returns its pages to the shared pool. This replaces per-model contiguous
// KV caches; the scheduler (Phase 7c) drives many sessions against one instance.
//
// Not thread-safe; the scheduler serializes access.
class GlobalKvCache {
 public:
  // Pre-allocates `poolBytes` of page storage (rounded down to a whole number of pages).
  GlobalKvCache(KvCacheConfig config, std::size_t poolBytes);

  const KvCacheConfig& config() const noexcept { return config_; }

  // Opens a new empty session; returns kInvalidSession only if the config is degenerate.
  SessionId open();
  // Frees a session's pages back to the pool. No-op for an unknown id.
  void close(SessionId session);
  // Clears a session's tokens (frees its pages) but keeps the session open.
  void reset(SessionId session);

  bool contains(SessionId session) const { return sessions_.count(session) != 0; }
  int length(SessionId session) const;  // tokens currently cached; -1 if unknown

  // Reserves capacity for one more token (position == current length), allocating pages if the
  // token crosses a page boundary. Returns false if the pool is exhausted (nothing is changed).
  bool appendToken(SessionId session);

  // Pointers to the kvDim-element K/V slot for (layer, pos). pos must be < length(session).
  // Returns nullptr on a bad session/layer/pos. Pointers are valid until the page is freed.
  float* kSlot(SessionId session, int layer, int pos);
  float* vSlot(SessionId session, int layer, int pos);

  std::size_t totalPages() const noexcept { return totalPages_; }
  std::size_t pagesInUse() const noexcept { return totalPages_ - freePages_.size(); }
  std::size_t pagesFree() const noexcept { return freePages_.size(); }
  std::size_t sessionCount() const noexcept { return sessions_.size(); }

 private:
  struct Session {
    int length = 0;
    // pages[layer] is the ordered list of page ids backing that layer's tokens.
    std::vector<std::vector<std::uint32_t>> pages;
  };

  std::size_t pageFloats() const noexcept {
    return static_cast<std::size_t>(config_.tokensPerPage) * config_.kvDim * 2;  // K and V
  }
  float* slot(SessionId session, int layer, int pos, bool wantV);
  void freeSessionPages(Session& s);

  KvCacheConfig config_;
  std::vector<float> arena_;
  std::size_t totalPages_ = 0;
  std::vector<std::uint32_t> freePages_;  // stack of available page ids
  std::unordered_map<SessionId, Session> sessions_;
  SessionId nextSession_ = 1;
};

}  // namespace qorvix::memory
