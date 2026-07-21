#include "qorvix/plugin_registry.hpp"

#include <system_error>
#include <utility>

namespace qorvix {

PluginRegistry::~PluginRegistry() {
  for (auto& [arch, entry] : entries_) destroyEntry(entry);
}

void PluginRegistry::destroyEntry(Entry& entry) noexcept {
  if (entry.instance && entry.destroy) {
    entry.destroy(entry.instance);
    entry.instance = nullptr;
  }
  // entry.library closes on destruction — the instance must be freed first (above), since it was
  // allocated by code living inside that library.
}

std::optional<std::string> PluginRegistry::load(const std::filesystem::path& libraryPath) {
  DynamicLibrary library(libraryPath);
  if (!library.valid()) {
    lastError_ = "failed to open '" + libraryPath.string() + "': " + library.error();
    return std::nullopt;
  }

  auto create = reinterpret_cast<QorvixCreatePluginFn>(library.symbol(kCreatePluginSymbol));
  auto destroy = reinterpret_cast<QorvixDestroyPluginFn>(library.symbol(kDestroyPluginSymbol));
  if (!create || !destroy) {
    lastError_ = "'" + libraryPath.string() + "' is missing qorvix plugin entry points";
    return std::nullopt;
  }

  IPlugin* instance = create();
  if (!instance) {
    lastError_ = "'" + libraryPath.string() + "' factory returned null";
    return std::nullopt;
  }

  std::string architecture = instance->architecture();
  if (architecture.empty()) {
    destroy(instance);
    lastError_ = "'" + libraryPath.string() + "' reported an empty architecture";
    return std::nullopt;
  }
  if (entries_.count(architecture)) {
    destroy(instance);
    lastError_ = "architecture '" + architecture + "' is already registered";
    return std::nullopt;
  }

  Entry entry;
  entry.path = libraryPath;
  entry.library = std::move(library);
  entry.instance = instance;
  entry.destroy = destroy;
  entries_.emplace(architecture, std::move(entry));
  return architecture;
}

bool PluginRegistry::unload(const std::string& architecture) {
  auto it = entries_.find(architecture);
  if (it == entries_.end()) {
    lastError_ = "architecture '" + architecture + "' is not registered";
    return false;
  }
  destroyEntry(it->second);
  entries_.erase(it);
  return true;
}

std::vector<std::string> PluginRegistry::scan(const std::filesystem::path& pluginsDir) {
  std::vector<std::string> loaded;
  std::error_code ec;
  if (!std::filesystem::is_directory(pluginsDir, ec)) {
    lastError_ = "plugins directory not found: " + pluginsDir.string();
    return loaded;
  }

  const std::string ext = DynamicLibrary::nativeExtension();
  for (const auto& entry : std::filesystem::directory_iterator(pluginsDir, ec)) {
    if (ec) break;
    if (!entry.is_regular_file()) continue;
    if (entry.path().extension() != ext) continue;
    if (auto arch = load(entry.path())) loaded.push_back(*arch);
  }
  return loaded;
}

IPlugin* PluginRegistry::get(const std::string& architecture) const {
  auto it = entries_.find(architecture);
  return it == entries_.end() ? nullptr : it->second.instance;
}

std::vector<std::string> PluginRegistry::architectures() const {
  std::vector<std::string> out;
  out.reserve(entries_.size());
  for (const auto& [arch, entry] : entries_) out.push_back(arch);
  return out;
}

bool PluginRegistry::contains(const std::string& architecture) const {
  return entries_.count(architecture) != 0;
}

}  // namespace qorvix
