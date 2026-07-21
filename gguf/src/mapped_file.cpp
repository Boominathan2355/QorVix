#include "qorvix/gguf/mapped_file.hpp"

#include <system_error>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace qorvix::gguf {

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      error_(std::move(other.error_)) {
#if defined(_WIN32)
  fileHandle_ = std::exchange(other.fileHandle_, nullptr);
  mappingHandle_ = std::exchange(other.mappingHandle_, nullptr);
#else
  fd_ = std::exchange(other.fd_, -1);
#endif
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    close();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    error_ = std::move(other.error_);
#if defined(_WIN32)
    fileHandle_ = std::exchange(other.fileHandle_, nullptr);
    mappingHandle_ = std::exchange(other.mappingHandle_, nullptr);
#else
    fd_ = std::exchange(other.fd_, -1);
#endif
  }
  return *this;
}

#if defined(_WIN32)

bool MappedFile::open(const std::filesystem::path& path) {
  close();
  HANDLE file = ::CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    error_ = "CreateFile failed for " + path.string();
    return false;
  }

  LARGE_INTEGER fileSize;
  if (!::GetFileSizeEx(file, &fileSize) || fileSize.QuadPart == 0) {
    error_ = "empty or unsizable file: " + path.string();
    ::CloseHandle(file);
    return false;
  }

  HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mapping) {
    error_ = "CreateFileMapping failed for " + path.string();
    ::CloseHandle(file);
    return false;
  }

  void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (!view) {
    error_ = "MapViewOfFile failed for " + path.string();
    ::CloseHandle(mapping);
    ::CloseHandle(file);
    return false;
  }

  fileHandle_ = file;
  mappingHandle_ = mapping;
  data_ = view;
  size_ = static_cast<std::size_t>(fileSize.QuadPart);
  return true;
}

void MappedFile::close() noexcept {
  if (data_) ::UnmapViewOfFile(data_);
  if (mappingHandle_) ::CloseHandle(static_cast<HANDLE>(mappingHandle_));
  if (fileHandle_) ::CloseHandle(static_cast<HANDLE>(fileHandle_));
  data_ = nullptr;
  size_ = 0;
  mappingHandle_ = nullptr;
  fileHandle_ = nullptr;
}

#else

bool MappedFile::open(const std::filesystem::path& path) {
  close();
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    error_ = "open failed for " + path.string();
    return false;
  }

  struct stat st{};
  if (::fstat(fd, &st) != 0 || st.st_size == 0) {
    error_ = "empty or unstattable file: " + path.string();
    ::close(fd);
    return false;
  }

  const std::size_t size = static_cast<std::size_t>(st.st_size);
  void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
    error_ = "mmap failed for " + path.string();
    ::close(fd);
    return false;
  }

  fd_ = fd;
  data_ = addr;
  size_ = size;
  return true;
}

void MappedFile::close() noexcept {
  if (data_) ::munmap(data_, size_);
  if (fd_ >= 0) ::close(fd_);
  data_ = nullptr;
  size_ = 0;
  fd_ = -1;
}

#endif

}  // namespace qorvix::gguf
