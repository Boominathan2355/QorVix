#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <filesystem>
#include <map>
#include <memory>

#include "qorvix/memory/disk_allocator.hpp"
#include "qorvix/memory/memory_manager.hpp"
#include "qorvix/memory/slab_allocator.hpp"
#include "qorvix/memory/transfer.hpp"

using namespace qorvix::memory;

// A stand-in for the CUDA transfer engine: records which tier pairs it was asked to move, then
// falls back to memcpy. Proves the MemoryManager routes migration through the engine and passes
// the correct source/target tiers — the exact contract the real CudaTransferEngine relies on.
namespace {
struct RecordingEngine final : ITransferEngine {
  Tier lastDst = Tier::HostRam;
  Tier lastSrc = Tier::HostRam;
  int copies = 0;
  void copy(void* dst, Tier dstTier, const void* src, Tier srcTier, std::size_t bytes) override {
    lastDst = dstTier;
    lastSrc = srcTier;
    ++copies;
    std::memcpy(dst, src, bytes);
  }
};
}  // namespace

TEST_CASE("HostTransferEngine copies bytes", "[memory][transfer]") {
  HostTransferEngine eng;
  char src[] = "qorvix";
  char dst[sizeof(src)] = {};
  eng.copy(dst, Tier::HostRam, src, Tier::DiskNvme, sizeof(src));
  REQUIRE(std::string(dst) == "qorvix");
}

TEST_CASE("MemoryManager routes migration through the transfer engine with correct tiers",
          "[memory][transfer]") {
  auto engine = std::make_unique<RecordingEngine>();
  RecordingEngine* probe = engine.get();

  std::map<Tier, MemoryManager::TierSpec> tiers;
  tiers[Tier::HostRam] = {std::make_unique<HostSlabAllocator>(), PoolConfig{}};
  tiers[Tier::DiskNvme] = {std::make_unique<DiskSlabAllocator>(
                               std::filesystem::temp_directory_path() / "qorvix_xfer_spool"),
                           PoolConfig{}};
  MemoryManager mgr(std::move(tiers), std::move(engine));

  { auto t = mgr.create("t", 1024, Tier::HostRam); }  // unpinned
  REQUIRE(mgr.migrate("t", Tier::DiskNvme));

  REQUIRE(probe->copies == 1);
  REQUIRE(probe->lastSrc == Tier::HostRam);
  REQUIRE(probe->lastDst == Tier::DiskNvme);
}
