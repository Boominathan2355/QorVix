#pragma once

#include <filesystem>
#include <string>

namespace qorvix {

// Move-only RAII wrapper over the platform dynamic-loader (dlopen/LoadLibrary). Owns the module
// handle and closes it on destruction. Platform headers are confined to the .cpp.
class DynamicLibrary {
 public:
  DynamicLibrary() noexcept = default;
  explicit DynamicLibrary(const std::filesystem::path& path);
  ~DynamicLibrary();

  DynamicLibrary(DynamicLibrary&& other) noexcept;
  DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;
  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  bool valid() const noexcept { return handle_ != nullptr; }

  // Resolves an exported symbol, or nullptr if absent (sets error()).
  void* symbol(const char* name) const;

  const std::string& error() const noexcept { return error_; }

  // Platform-native shared-library suffix, including the dot: ".dll", ".so", or ".dylib".
  static const char* nativeExtension() noexcept;

 private:
  void close() noexcept;

  void* handle_ = nullptr;
  mutable std::string error_;
};

}  // namespace qorvix
