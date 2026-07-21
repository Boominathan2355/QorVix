#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "qorvix/dynamic_library.hpp"
#include "qorvix/plugin.hpp"

namespace qorvix {

// Owns dynamically loaded architecture plugins, keyed by the architecture string they report.
// Supports hot load (a single library) and hot unload. Not thread-safe; callers serialize
// access (the scheduler will own a single registry instance in later phases).
class PluginRegistry {
 public:
  PluginRegistry() = default;
  ~PluginRegistry();

  PluginRegistry(const PluginRegistry&) = delete;
  PluginRegistry& operator=(const PluginRegistry&) = delete;

  // Loads one plugin library and registers its instance. Returns the architecture on success,
  // or std::nullopt on failure (see lastError()).
  std::optional<std::string> load(const std::filesystem::path& libraryPath);

  // Unloads a previously registered architecture. Returns false if it wasn't registered.
  bool unload(const std::string& architecture);

  // Loads every shared library in a directory (non-recursive). Returns the architectures that
  // were newly registered by this call.
  std::vector<std::string> scan(const std::filesystem::path& pluginsDir);

  IPlugin* get(const std::string& architecture) const;
  std::vector<std::string> architectures() const;
  bool contains(const std::string& architecture) const;

  const std::string& lastError() const noexcept { return lastError_; }

 private:
  struct Entry {
    std::filesystem::path path;
    DynamicLibrary library;
    IPlugin* instance = nullptr;
    QorvixDestroyPluginFn destroy = nullptr;
  };

  void destroyEntry(Entry& entry) noexcept;

  std::map<std::string, Entry> entries_;
  std::string lastError_;
};

}  // namespace qorvix
