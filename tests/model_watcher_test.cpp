#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

#include "qorvix/model_watcher.hpp"

namespace fs = std::filesystem;

namespace {

fs::path makeCleanDir(const std::string& name) {
  fs::path dir = fs::temp_directory_path() / ("qorvix_test_" + name);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

void writeFile(const fs::path& path) {
  std::ofstream out(path, std::ios::binary);
  out << "gguf";
}

}  // namespace

// Drives the watcher deterministically via pollOnce() rather than sleeping on the polling thread,
// so the test is fast and not timing-dependent.
TEST_CASE("ModelWatcher reports additions and removals across polls", "[model_watcher]") {
  const fs::path dir = makeCleanDir("watcher_diff");

  std::vector<std::string> added;
  std::vector<std::string> removed;

  qorvix::ModelWatcher watcher(dir, std::chrono::milliseconds(50));
  watcher.onAdded([&](const qorvix::ModelInfo& m) { added.push_back(m.name); });
  watcher.onRemoved([&](const qorvix::ModelInfo& m) { removed.push_back(m.name); });

  watcher.pollOnce();  // establish empty baseline
  REQUIRE(added.empty());

  writeFile(dir / "gemma.gguf");
  watcher.pollOnce();
  REQUIRE(added.size() == 1);
  REQUIRE(added[0] == "gemma");
  REQUIRE(removed.empty());

  // No change between polls: no callbacks.
  watcher.pollOnce();
  REQUIRE(added.size() == 1);

  fs::remove(dir / "gemma.gguf");
  watcher.pollOnce();
  REQUIRE(removed.size() == 1);
  REQUIRE(removed[0] == "gemma");

  fs::remove_all(dir);
}

TEST_CASE("ModelWatcher start/stop runs the polling thread without deadlock", "[model_watcher]") {
  const fs::path dir = makeCleanDir("watcher_thread");
  qorvix::ModelWatcher watcher(dir, std::chrono::milliseconds(10));
  watcher.start();
  REQUIRE(watcher.running());
  watcher.stop();
  REQUIRE_FALSE(watcher.running());
  fs::remove_all(dir);
}
