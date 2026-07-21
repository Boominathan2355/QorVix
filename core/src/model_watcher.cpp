#include "qorvix/model_watcher.hpp"

#include <system_error>
#include <utility>

#include "qorvix/model_registry.hpp"

namespace qorvix {

ModelWatcher::ModelWatcher(std::filesystem::path directory, std::chrono::milliseconds interval)
    : directory_(std::move(directory)), interval_(interval) {}

ModelWatcher::~ModelWatcher() { stop(); }

void ModelWatcher::start() {
  if (running_.exchange(true)) return;
  thread_ = std::thread(&ModelWatcher::run, this);
}

void ModelWatcher::stop() {
  if (!running_.exchange(false)) return;
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void ModelWatcher::run() {
  pollOnce();
  std::unique_lock<std::mutex> lock(mutex_);
  while (running_.load()) {
    // Wake early on stop() instead of sleeping the full interval.
    if (cv_.wait_for(lock, interval_, [this] { return !running_.load(); })) break;
    lock.unlock();
    pollOnce();
    lock.lock();
  }
}

void ModelWatcher::pollOnce() {
  std::unordered_map<std::string, ModelInfo> current;

  std::error_code ec;
  if (std::filesystem::is_directory(directory_, ec)) {
    for (const auto& entry : std::filesystem::directory_iterator(directory_, ec)) {
      if (ec) break;
      if (!entry.is_regular_file()) continue;
      if (!isModelFile(entry.path())) continue;
      ModelInfo info;
      if (describeModel(entry.path(), info)) current.emplace(info.path.string(), std::move(info));
    }
  }

  for (const auto& [key, info] : current) {
    if (!known_.count(key)) {
      if (onAdded_) onAdded_(info);
    }
  }
  for (const auto& [key, info] : known_) {
    if (!current.count(key)) {
      if (onRemoved_) onRemoved_(info);
    }
  }

  known_ = std::move(current);
}

}  // namespace qorvix
