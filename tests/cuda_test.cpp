#include <catch2/catch_test_macros.hpp>

#include "qorvix/cuda/backend.hpp"
#include "qorvix/cuda/gpu_memory.hpp"
#include "qorvix/memory/memory_manager.hpp"

using namespace qorvix;

// These tests run in every build. The device-dependent assertions are guarded on deviceCount()
// so a GPU-less host (CPU build, or a CUDA build with no card — e.g. CI) exercises the "skip"
// paths instead of failing.

TEST_CASE("CUDA facade is callable without a device", "[cuda]") {
  // Never throws, regardless of build flavor.
  const int count = cuda::deviceCount();
  REQUIRE(count >= 0);
  REQUIRE(cuda::enumerateDevices().size() == static_cast<std::size_t>(count));

  if (!cuda::builtWithCuda()) {
    REQUIRE(count == 0);
    REQUIRE(cuda::makeGpuSlabAllocator() == nullptr);
    REQUIRE(cuda::makeGpuMemoryManager(0, 0, 0, "unused") == nullptr);
    auto self = cuda::selfTest();
    REQUIRE_FALSE(self.ran);  // not built in -> reported as skipped, not failed
  }
}

TEST_CASE("self-tests either pass on a device or report skipped", "[cuda]") {
  const auto self = cuda::selfTest();
  const auto gemm = cuda::gemmSelfTest();
  if (cuda::deviceCount() > 0) {
    REQUIRE(self.ran);
    REQUIRE(self.passed);
    REQUIRE(gemm.ran);
    REQUIRE(gemm.passed);
  } else {
    REQUIRE_FALSE(self.ran);   // ran==false == "skipped", the correct outcome with no GPU
    REQUIRE_FALSE(gemm.ran);
  }
}

TEST_CASE("GPU memory manager works when a device is present", "[cuda]") {
  auto mgr = cuda::makeGpuMemoryManager(64 * memory::kMiB, 64 * memory::kMiB, 0,
                                        std::filesystem::temp_directory_path() / "qorvix_gpu_spool");
  if (cuda::deviceCount() == 0) {
    REQUIRE(mgr == nullptr);
    return;
  }
  REQUIRE(mgr != nullptr);
  // A tensor requested on the GPU tier lands there; migrating to host and back preserves it
  // (round-trips through cudaMemcpy in the transfer engine).
  {
    auto t = mgr->create("g", 4 * memory::kKiB, memory::Tier::GpuVram);
    REQUIRE(t.valid());
    REQUIRE(t.tier() == memory::Tier::GpuVram);
  }
  REQUIRE(mgr->migrate("g", memory::Tier::HostRam));
  REQUIRE(mgr->tierOf("g") == memory::Tier::HostRam);
}
