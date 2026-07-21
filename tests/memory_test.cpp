#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <filesystem>
#include <vector>

#include "qorvix/memory/disk_allocator.hpp"
#include "qorvix/memory/memory_manager.hpp"
#include "qorvix/memory/page.hpp"
#include "qorvix/memory/page_pool.hpp"
#include "qorvix/memory/slab_allocator.hpp"

namespace fs = std::filesystem;
using namespace qorvix::memory;

namespace {

// KiB-scale page sizes so the tests exercise pool logic without MB allocations.
PoolConfig tinyConfig(std::size_t budget = 0) {
  PoolConfig cfg;
  cfg.pageSizes = {4 * kKiB, 8 * kKiB, 16 * kKiB};
  cfg.hugeGranularity = 4 * kKiB;
  cfg.budgetBytes = budget;
  return cfg;
}

fs::path scratchDir(const std::string& name) {
  fs::path dir = fs::temp_directory_path() / ("qorvix_mem_" + name);
  fs::remove_all(dir);
  return dir;
}

void fillPattern(std::byte* p, std::size_t n, std::uint8_t seed) {
  for (std::size_t i = 0; i < n; ++i) p[i] = static_cast<std::byte>((seed + i) & 0xFF);
}

bool checkPattern(const std::byte* p, std::size_t n, std::uint8_t seed) {
  for (std::size_t i = 0; i < n; ++i) {
    if (p[i] != static_cast<std::byte>((seed + i) & 0xFF)) return false;
  }
  return true;
}

std::map<Tier, MemoryManager::TierSpec> ramDiskSpecs(std::size_t ramBudget, std::size_t diskBudget,
                                                     const fs::path& spool) {
  std::map<Tier, MemoryManager::TierSpec> tiers;
  tiers[Tier::HostRam] = {std::make_unique<HostSlabAllocator>(), tinyConfig(ramBudget)};
  tiers[Tier::DiskNvme] = {std::make_unique<DiskSlabAllocator>(spool), tinyConfig(diskBudget)};
  return tiers;
}

}  // namespace

TEST_CASE("MemoryPage aligns, frees, and coalesces", "[memory][page]") {
  std::vector<std::byte> slab(16 * kKiB);
  MemoryPage page(slab.data(), slab.size());

  void* a = page.allocate(300);
  void* b = page.allocate(100);
  void* c = page.allocate(1000);
  REQUIRE(a != nullptr);
  REQUIRE(b != nullptr);
  REQUIRE(c != nullptr);
  auto off = [&](void* p) {
    return static_cast<std::size_t>(static_cast<std::byte*>(p) - slab.data());
  };
  REQUIRE(off(a) % kSubAllocAlign == 0);
  REQUIRE(off(b) % kSubAllocAlign == 0);
  REQUIRE(off(c) % kSubAllocAlign == 0);
  REQUIRE(page.bytesUsed() == 1400);

  // Free the middle allocation and take the same bytes again — the hole is reused.
  REQUIRE(page.free(b));
  void* b2 = page.allocate(100);
  REQUIRE(b2 == b);

  // Unknown pointers are rejected.
  int stackVar = 0;
  REQUIRE_FALSE(page.free(&stackVar));

  REQUIRE(page.free(a));
  REQUIRE(page.free(b2));
  REQUIRE(page.free(c));
  REQUIRE(page.empty());
  // Full coalescing: one free block spanning the whole page.
  REQUIRE(page.largestFreeBlock() == slab.size());
}

TEST_CASE("PagePool shares pages, grows, enforces budget, trims", "[memory][pool]") {
  PagePool pool(std::make_unique<HostSlabAllocator>(), tinyConfig(/*budget=*/32 * kKiB));

  // Four 1 KiB allocations share one 4 KiB page.
  std::vector<Allocation> small;
  for (int i = 0; i < 4; ++i) {
    Allocation a = pool.allocate(1 * kKiB);
    REQUIRE(a);
    small.push_back(a);
  }
  REQUIRE(pool.stats().pageCount == 1);

  // A 10 KiB request opens the smallest standard page that fits (16 KiB).
  Allocation big = pool.allocate(10 * kKiB);
  REQUIRE(big);
  REQUIRE(pool.stats().pageCount == 2);
  REQUIRE(pool.stats().reservedBytes == 20 * kKiB);

  // Budget: 20 KiB reserved + 16 KiB new page would exceed 32 KiB.
  Allocation denied = pool.allocate(10 * kKiB);
  REQUIRE_FALSE(denied);
  REQUIRE(pool.lastError().find("budget") != std::string::npos);

  // Free everything; pages stay pooled until trim() releases them.
  for (const auto& a : small) pool.free(a);
  pool.free(big);
  REQUIRE(pool.stats().pageCount == 2);
  REQUIRE(pool.trim() == 20 * kKiB);
  REQUIRE(pool.stats().reservedBytes == 0);
}

TEST_CASE("oversized requests get bespoke huge pages, released on free", "[memory][pool]") {
  PagePool pool(std::make_unique<HostSlabAllocator>(), tinyConfig());

  Allocation huge = pool.allocate(20 * kKiB);  // > largest standard page (16 KiB)
  REQUIRE(huge);
  REQUIRE(pool.stats().reservedBytes == 20 * kKiB);  // rounded to 4 KiB granularity

  pool.free(huge);  // huge pages are released immediately, no trim needed
  REQUIRE(pool.stats().pageCount == 0);
  REQUIRE(pool.stats().reservedBytes == 0);
}

TEST_CASE("DiskSlabAllocator provides file-backed writable slabs", "[memory][disk]") {
  const fs::path dir = scratchDir("disk");
  {
    DiskSlabAllocator alloc(dir);
    void* p = alloc.allocate(8 * kKiB);
    REQUIRE(p != nullptr);
    REQUIRE(alloc.liveSlabs() == 1);
    REQUIRE(fs::exists(dir));
    REQUIRE(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}) == 1);

    // The mapping is writable and readable.
    fillPattern(static_cast<std::byte*>(p), 8 * kKiB, 7);
    REQUIRE(checkPattern(static_cast<std::byte*>(p), 8 * kKiB, 7));

    alloc.deallocate(p, 8 * kKiB);
    REQUIRE(alloc.liveSlabs() == 0);
    REQUIRE(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}) == 0);
  }
  // Destructor removes the (empty) spool directory.
  REQUIRE_FALSE(fs::exists(dir));
}

TEST_CASE("MemoryManager refcounts, caches zero-ref buffers, drops", "[memory][manager]") {
  MemoryManager mgr(ramDiskSpecs(0, 0, scratchDir("mgr_refs")));

  {
    TensorRef w = mgr.create("w", 1 * kKiB);
    REQUIRE(w.valid());
    REQUIRE(w.tier() == Tier::HostRam);
    fillPattern(w.data(), w.size(), 3);
  }  // released: refcount 0, but stays resident as cache

  REQUIRE(mgr.contains("w"));
  {
    TensorRef w = mgr.acquire("w");
    REQUIRE(w.valid());
    REQUIRE(checkPattern(w.data(), w.size(), 3));
  }

  REQUIRE(mgr.create("w", 1 * kKiB).valid() == false);  // duplicate name
  REQUIRE(mgr.drop("w"));
  REQUIRE_FALSE(mgr.contains("w"));
  REQUIRE_FALSE(mgr.acquire("w").valid());
}

TEST_CASE("alias shares one buffer across names", "[memory][manager]") {
  MemoryManager mgr(ramDiskSpecs(0, 0, scratchDir("mgr_alias")));

  TensorRef t = mgr.create("t", 512);
  TensorRef u = mgr.alias("t", "u");
  REQUIRE(u.valid());
  REQUIRE(t.data() == u.data());  // same underlying buffer

  fillPattern(t.data(), 512, 9);
  REQUIRE(checkPattern(u.data(), 512, 9));

  // Dropping one name keeps the buffer reachable through the other.
  t.reset();
  REQUIRE(mgr.drop("t"));
  REQUIRE(mgr.contains("u"));
  REQUIRE(checkPattern(u.data(), 512, 9));

  auto s = mgr.stats(Tier::HostRam);
  REQUIRE(s.buffers == 1);
}

TEST_CASE("migrate moves buffers between tiers preserving contents", "[memory][manager]") {
  MemoryManager mgr(ramDiskSpecs(0, 0, scratchDir("mgr_migrate")));

  {
    TensorRef t = mgr.create("t", 2 * kKiB, Tier::HostRam);
    fillPattern(t.data(), t.size(), 42);
    // Pinned buffers cannot migrate.
    REQUIRE_FALSE(mgr.migrate("t", Tier::DiskNvme));
  }

  REQUIRE(mgr.migrate("t", Tier::DiskNvme));
  REQUIRE(mgr.tierOf("t") == Tier::DiskNvme);
  {
    TensorRef t = mgr.acquire("t");
    REQUIRE(t.tier() == Tier::DiskNvme);
    REQUIRE(checkPattern(t.data(), t.size(), 42));
  }
  // And back up.
  REQUIRE(mgr.migrate("t", Tier::HostRam));
  {
    TensorRef t = mgr.acquire("t");
    REQUIRE(checkPattern(t.data(), t.size(), 42));
  }
}

TEST_CASE("budget pressure offloads the LRU zero-ref buffer to the next tier", "[memory][manager]") {
  // RAM budget of exactly two 4 KiB pages; disk unlimited.
  MemoryManager mgr(ramDiskSpecs(8 * kKiB, 0, scratchDir("mgr_evict")));

  {
    TensorRef a = mgr.create("a", 3 * kKiB);
    fillPattern(a.data(), a.size(), 1);
  }
  {
    TensorRef b = mgr.create("b", 3 * kKiB);
    fillPattern(b.data(), b.size(), 2);
  }
  REQUIRE(mgr.stats(Tier::HostRam).pool.reservedBytes == 8 * kKiB);

  // No RAM left: creating c must offload the LRU zero-ref buffer (a) to disk.
  TensorRef c = mgr.create("c", 3 * kKiB);
  REQUIRE(c.valid());
  REQUIRE(c.tier() == Tier::HostRam);
  REQUIRE(mgr.tierOf("a") == Tier::DiskNvme);
  REQUIRE(mgr.tierOf("b") == Tier::HostRam);

  // The offloaded buffer survives with its contents.
  TensorRef a = mgr.acquire("a");
  REQUIRE(checkPattern(a.data(), a.size(), 1));
}

TEST_CASE("create falls back to a lower tier when the preferred one is unavailable", "[memory][manager]") {
  MemoryManager mgr(ramDiskSpecs(0, 0, scratchDir("mgr_fallback")));

  // GpuVram is not configured (pre-Phase-4): placement falls through to HostRam.
  TensorRef t = mgr.create("t", 1 * kKiB, Tier::GpuVram);
  REQUIRE(t.valid());
  REQUIRE(t.tier() == Tier::HostRam);
}
