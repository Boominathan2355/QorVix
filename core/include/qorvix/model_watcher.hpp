#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "qorvix/model_info.hpp"

namespace qorvix {

// Watches a model directory and reports files appearing or disappearing, so models can be
// registered without restarting the process (spec: "no restart required"). Polling-based for
// portability — a good-enough default that works identically on Linux and Windows; can be
// swapped for inotify/ReadDirectoryChangesW later without touching callers.
//
// Callbacks run on the watcher's own thread; keep them short and thread-safe.
class ModelWatcher {
 public:
  using Callback = std::function<void(const ModelInfo&)>;

  ModelWatcher(std::filesystem::path directory, std::chrono::milliseconds interval);
  ~ModelWatcher();

  ModelWatcher(const ModelWatcher&) = delete;
  ModelWatcher& operator=(const ModelWatcher&) = delete;

  void onAdded(Callback cb) { onAdded_ = std::move(cb); }
  void onRemoved(Callback cb) { onRemoved_ = std::move(cb); }

  void start();
  void stop();
  bool running() const noexcept { return running_.load(); }

  // Runs a single scan synchronously and fires callbacks for any changes. Called on the polling
  // thread; exposed so tests (and start()) can perform a deterministic first pass.
  void pollOnce();

 private:
  void run();

  std::filesystem::path directory_;
  std::chrono::milliseconds interval_;

  Callback onAdded_;
  Callback onRemoved_;

  std::unordered_map<std::string, ModelInfo> known_;  // keyed by absolute path

  std::thread thread_;
  std::atomic<bool> running_{false};
  std::mutex mutex_;
  std::condition_variable cv_;
};

}  // namespace qorvix
