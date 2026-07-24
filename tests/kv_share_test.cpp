#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <vector>

#include "qorvix/memory/kv_cache.hpp"

using namespace qorvix::memory;

// Coverage for the KV primitives that shipped without any: store/fetch, prefix page sharing, and
// the page refcounting that sharing introduces. Refcount bugs here are use-after-free or silent
// cross-session corruption, so they are worth pinning before anything is built on top.

namespace {

constexpr int kLayers = 2, kKvDim = 4, kTokensPerPage = 4;

KvCacheConfig cfg(KvCacheQuantType q = KvCacheQuantType::F32) {
  KvCacheConfig c;
  c.layers = kLayers;
  c.kvDim = kKvDim;
  c.tokensPerPage = kTokensPerPage;
  c.quantType = q;
  return c;
}

// Bytes for `pages` pages across all layers.
std::size_t poolFor(int pages) {
  return static_cast<std::size_t>(pages) * kTokensPerPage * kKvDim * 2 * sizeof(float);
}

// Appends `n` tokens and writes a recognisable pattern into every K/V slot.
void fill(GlobalKvCache& kv, SessionId s, int n, float base) {
  for (int t = 0; t < n; ++t) {
    REQUIRE(kv.appendToken(s));
    for (int l = 0; l < kLayers; ++l) {
      std::vector<float> k(kKvDim), v(kKvDim);
      for (int i = 0; i < kKvDim; ++i) {
        k[i] = base + t * 10.0f + l + i * 0.1f;
        v[i] = -(base + t * 10.0f + l + i * 0.1f);
      }
      REQUIRE(kv.storeK(s, l, t, k.data()));
      REQUIRE(kv.storeV(s, l, t, v.data()));
    }
  }
}

}  // namespace

TEST_CASE("store/fetch round-trips every K and V slot", "[kvshare]") {
  GlobalKvCache kv(cfg(), poolFor(64));
  const SessionId s = kv.open();
  REQUIRE(s != kInvalidSession);
  fill(kv, s, 6, 100.0f);

  for (int t = 0; t < 6; ++t)
    for (int l = 0; l < kLayers; ++l) {
      std::vector<float> k(kKvDim, 0.0f), v(kKvDim, 0.0f);
      REQUIRE(kv.fetchK(s, l, t, k.data()));
      REQUIRE(kv.fetchV(s, l, t, v.data()));
      for (int i = 0; i < kKvDim; ++i) {
        const float want = 100.0f + t * 10.0f + l + i * 0.1f;
        REQUIRE(k[i] == want);
        REQUIRE(v[i] == -want);
      }
    }
}

TEST_CASE("store/fetch reject out-of-range positions", "[kvshare]") {
  GlobalKvCache kv(cfg(), poolFor(16));
  const SessionId s = kv.open();
  fill(kv, s, 2, 1.0f);
  std::vector<float> buf(kKvDim, 0.0f);

  REQUIRE_FALSE(kv.fetchK(s, 0, 2, buf.data()));         // pos == length
  REQUIRE_FALSE(kv.fetchK(s, kLayers, 0, buf.data()));   // bad layer
  REQUIRE_FALSE(kv.storeK(s + 999, 0, 0, buf.data()));   // unknown session
}

TEST_CASE("sharePrefix gives the target the source's whole pages", "[kvshare]") {
  GlobalKvCache kv(cfg(), poolFor(64));
  const SessionId src = kv.open();
  fill(kv, src, 10, 100.0f);  // 10 tokens = 2 full pages of 4, plus a partial

  const SessionId tgt = kv.open();
  REQUIRE(kv.sharePrefix(tgt, src, 8));  // 8 tokens == exactly 2 pages
  REQUIRE(kv.length(tgt) == 8);

  // The target must see the source's cached K/V without having written anything itself.
  for (int t = 0; t < 8; ++t)
    for (int l = 0; l < kLayers; ++l) {
      std::vector<float> k(kKvDim, 0.0f);
      REQUIRE(kv.fetchK(tgt, l, t, k.data()));
      REQUIRE(k[0] == 100.0f + t * 10.0f + l);
    }
}

TEST_CASE("sharePrefix rounds down to whole pages", "[kvshare]") {
  GlobalKvCache kv(cfg(), poolFor(64));
  const SessionId src = kv.open();
  fill(kv, src, 10, 100.0f);

  const SessionId tgt = kv.open();
  // 6 tokens is 1.5 pages; only the first whole page can be shared, since a partially-filled
  // shared page would be written by both sessions.
  REQUIRE(kv.sharePrefix(tgt, src, 6));
  REQUIRE(kv.length(tgt) == kTokensPerPage);
}

TEST_CASE("sharePrefix refuses invalid requests", "[kvshare]") {
  GlobalKvCache kv(cfg(), poolFor(64));
  const SessionId src = kv.open();
  fill(kv, src, 8, 1.0f);
  const SessionId tgt = kv.open();

  REQUIRE_FALSE(kv.sharePrefix(tgt, src + 999, 4));  // unknown source
  REQUIRE_FALSE(kv.sharePrefix(tgt + 999, src, 4));  // unknown target
  REQUIRE_FALSE(kv.sharePrefix(tgt, src, 0));        // nothing to share
  REQUIRE_FALSE(kv.sharePrefix(tgt, src, 99));       // longer than the source

  // A target that already holds tokens must not be re-based onto someone else's pages.
  fill(kv, tgt, 1, 5.0f);
  REQUIRE_FALSE(kv.sharePrefix(tgt, src, 4));
}

TEST_CASE("shared pages survive the donor closing and are freed only once", "[kvshare]") {
  // The refcount contract: closing one sharer must not hand a still-referenced page back to the
  // pool, or the next open() would scribble over live KV.
  GlobalKvCache kv(cfg(), poolFor(8));  // deliberately small so double-free shows up as reuse
  const SessionId src = kv.open();
  fill(kv, src, 8, 100.0f);

  const SessionId tgt = kv.open();
  REQUIRE(kv.sharePrefix(tgt, src, 8));

  kv.close(src);  // donor goes away; target still references the pages

  for (int t = 0; t < 8; ++t) {
    std::vector<float> k(kKvDim, 0.0f);
    REQUIRE(kv.fetchK(tgt, 0, t, k.data()));
    REQUIRE(k[0] == 100.0f + t * 10.0f);  // still intact
  }

  kv.close(tgt);  // last reference: now the pages may return to the pool

  // The pool must be fully reusable — if close() had double-freed, this would over-count.
  const SessionId again = kv.open();
  REQUIRE(again != kInvalidSession);
  fill(kv, again, 8, 7.0f);
  std::vector<float> k(kKvDim, 0.0f);
  REQUIRE(kv.fetchK(again, 0, 0, k.data()));
  REQUIRE(k[0] == 7.0f);
}

TEST_CASE("two targets can share one donor's prefix", "[kvshare]") {
  GlobalKvCache kv(cfg(), poolFor(64));
  const SessionId src = kv.open();
  fill(kv, src, 8, 100.0f);

  const SessionId a = kv.open(), b = kv.open();
  REQUIRE(kv.sharePrefix(a, src, 8));
  REQUIRE(kv.sharePrefix(b, src, 8));

  kv.close(a);  // dropping one sharer must not disturb the other
  std::vector<float> k(kKvDim, 0.0f);
  REQUIRE(kv.fetchK(b, 0, 3, k.data()));
  REQUIRE(k[0] == 100.0f + 3 * 10.0f);
}

TEST_CASE("unimplemented quantized KV is refused, not silently stored as F32", "[kvshare]") {
  // The enum exists but no quantized layout is implemented. Constructing with one must not hand
  // back a working F32 cache — that would promise a memory saving it does not deliver.
  for (auto q : {KvCacheQuantType::Q8_0, KvCacheQuantType::Q4_0}) {
    GlobalKvCache kv(cfg(q), poolFor(64));
    REQUIRE(kv.open() == kInvalidSession);
  }
  GlobalKvCache ok(cfg(KvCacheQuantType::F32), poolFor(64));
  REQUIRE(ok.open() != kInvalidSession);
}
