#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>

namespace qorvix::gguf {

// Read-only memory map of a whole file. Only the pages actually touched are faulted in, so
// mapping a multi-GB GGUF to parse its (small) header region does not read the tensor data.
// Move-only RAII; platform handles live in the .cpp. A generalized version graduates to the
// memory manager's NVMe tier in Phase 3.
class MappedFile {
 public:
  MappedFile() = default;
  ~MappedFile();

  MappedFile(MappedFile&&) noexcept;
  MappedFile& operator=(MappedFile&&) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  // Maps `path`. Returns false and sets error() on failure (missing file, empty file, mmap error).
  bool open(const std::filesystem::path& path);
  void close() noexcept;

  bool valid() const noexcept { return data_ != nullptr; }
  std::span<const std::byte> bytes() const noexcept {
    return {static_cast<const std::byte*>(data_), size_};
  }
  std::size_t size() const noexcept { return size_; }
  const std::string& error() const noexcept { return error_; }

 private:
  void* data_ = nullptr;
  std::size_t size_ = 0;
  std::string error_;

#if defined(_WIN32)
  void* fileHandle_ = nullptr;    // HANDLE
  void* mappingHandle_ = nullptr;  // HANDLE
#else
  int fd_ = -1;
#endif
};

}  // namespace qorvix::gguf
