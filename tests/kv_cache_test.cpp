#include <catch2/catch_test_macros.hpp>

#include "qorvix/memory/kv_cache.hpp"

using namespace qorvix::memory;

namespace {
// layers=2, kvDim=4, 8 tokens/page -> pageFloats = 8*4*2 = 64 floats = 256 bytes.
KvCacheConfig tinyCfg() { return {2, 4, 8}; }
}  // namespace

TEST_CASE("append allocates pages per layer on page boundaries", "[kv]") {
  GlobalKvCache kv(tinyCfg(), /*poolBytes=*/64 * 256);  // 64 pages
  const std::size_t total = kv.totalPages();
  REQUIRE(total == 64);

  SessionId s = kv.open();
  REQUIRE(s != kInvalidSession);
  REQUIRE(kv.length(s) == 0);

  // First token opens one page per layer (2 pages).
  REQUIRE(kv.appendToken(s));
  REQUIRE(kv.length(s) == 1);
  REQUIRE(kv.pagesInUse() == 2);

  // Tokens 2..8 fit in the same pages.
  for (int i = 0; i < 7; ++i) REQUIRE(kv.appendToken(s));
  REQUIRE(kv.length(s) == 8);
  REQUIRE(kv.pagesInUse() == 2);

  // Token 9 crosses the page boundary -> 2 more pages.
  REQUIRE(kv.appendToken(s));
  REQUIRE(kv.length(s) == 9);
  REQUIRE(kv.pagesInUse() == 4);
}

TEST_CASE("k/v slots are distinct, writable, and paged correctly", "[kv]") {
  GlobalKvCache kv(tinyCfg(), 64 * 256);
  SessionId s = kv.open();
  for (int i = 0; i < 10; ++i) REQUIRE(kv.appendToken(s));  // spans 2 pages per layer

  // Write a unique value into every (layer, pos) K and V slot.
  for (int layer = 0; layer < 2; ++layer) {
    for (int pos = 0; pos < 10; ++pos) {
      float* k = kv.kSlot(s, layer, pos);
      float* v = kv.vSlot(s, layer, pos);
      REQUIRE(k != nullptr);
      REQUIRE(v != nullptr);
      REQUIRE(k != v);
      for (int e = 0; e < 4; ++e) {
        k[e] = 100.0f * layer + pos + 0.1f * e;
        v[e] = -(100.0f * layer + pos + 0.1f * e);
      }
    }
  }
  // Read them back — including across the page boundary at pos 8.
  for (int layer = 0; layer < 2; ++layer) {
    for (int pos = 0; pos < 10; ++pos) {
      const float* k = kv.kSlot(s, layer, pos);
      const float* v = kv.vSlot(s, layer, pos);
      for (int e = 0; e < 4; ++e) {
        const float expect = 100.0f * layer + pos + 0.1f * e;
        REQUIRE(k[e] == expect);
        REQUIRE(v[e] == -expect);
      }
    }
  }
}

TEST_CASE("sessions are isolated and freed on close", "[kv]") {
  GlobalKvCache kv(tinyCfg(), 64 * 256);
  SessionId a = kv.open();
  SessionId b = kv.open();
  REQUIRE(a != b);

  REQUIRE(kv.appendToken(a));
  REQUIRE(kv.appendToken(b));
  kv.kSlot(a, 0, 0)[0] = 11.0f;
  kv.kSlot(b, 0, 0)[0] = 22.0f;
  REQUIRE(kv.kSlot(a, 0, 0)[0] == 11.0f);  // b's write didn't touch a
  REQUIRE(kv.kSlot(b, 0, 0)[0] == 22.0f);

  const std::size_t used = kv.pagesInUse();
  kv.close(a);
  REQUIRE_FALSE(kv.contains(a));
  REQUIRE(kv.pagesInUse() == used - 2);  // a's 2 pages returned to the pool
  REQUIRE(kv.length(b) == 1);            // b unaffected
}

TEST_CASE("reset frees pages but keeps the session", "[kv]") {
  GlobalKvCache kv(tinyCfg(), 64 * 256);
  SessionId s = kv.open();
  for (int i = 0; i < 5; ++i) kv.appendToken(s);
  REQUIRE(kv.pagesInUse() == 2);

  kv.reset(s);
  REQUIRE(kv.contains(s));
  REQUIRE(kv.length(s) == 0);
  REQUIRE(kv.pagesInUse() == 0);

  // Reusable after reset.
  REQUIRE(kv.appendToken(s));
  REQUIRE(kv.length(s) == 1);
}

TEST_CASE("appendToken fails cleanly when the pool is exhausted", "[kv]") {
  // 2 pages total, 2 layers -> room for exactly one page-block (8 tokens).
  GlobalKvCache kv(tinyCfg(), 2 * 256);
  REQUIRE(kv.totalPages() == 2);
  SessionId s = kv.open();
  for (int i = 0; i < 8; ++i) REQUIRE(kv.appendToken(s));  // fills both pages
  REQUIRE(kv.pagesFree() == 0);

  // Ninth token needs new pages but the pool is empty — fails without changing state.
  REQUIRE_FALSE(kv.appendToken(s));
  REQUIRE(kv.length(s) == 8);
}
