#include "qorvix/memory/disk_allocator.hpp"

#include <system_error>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace qorvix::memory {

DiskSlabAllocator::DiskSlabAllocator(std::filesystem::path spoolDir) : dir_(std::move(spoolDir)) {
  std::error_code ec;
  std::filesystem::create_directories(dir_, ec);
  if (ec) error_ = "cannot create spool directory " + dir_.string();
}

DiskSlabAllocator::~DiskSlabAllocator() {
  while (!slabs_.empty()) {
    auto it = slabs_.begin();
    release(it->first, it->second);
    slabs_.erase(it);
  }
  std::error_code ec;
  std::filesystem::remove(dir_, ec);  // only succeeds if empty; best-effort
}

#if defined(_WIN32)

void* DiskSlabAllocator::allocate(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  Slab slab;
  slab.size = bytes;
  slab.path = dir_ / ("slab_" + std::to_string(nextId_++) + ".bin");

  HANDLE file = ::CreateFileW(slab.path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0,
                              nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    error_ = "CreateFile failed for " + slab.path.string();
    return nullptr;
  }

  // CreateFileMapping with an explicit size extends the file to `bytes`.
  HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READWRITE,
                                        static_cast<DWORD>(static_cast<std::uint64_t>(bytes) >> 32),
                                        static_cast<DWORD>(bytes & 0xFFFFFFFFu), nullptr);
  if (!mapping) {
    error_ = "CreateFileMapping failed for " + slab.path.string();
    ::CloseHandle(file);
    ::DeleteFileW(slab.path.wstring().c_str());
    return nullptr;
  }

  void* view = ::MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, bytes);
  if (!view) {
    error_ = "MapViewOfFile failed for " + slab.path.string();
    ::CloseHandle(mapping);
    ::CloseHandle(file);
    ::DeleteFileW(slab.path.wstring().c_str());
    return nullptr;
  }

  slab.fileHandle = file;
  slab.mappingHandle = mapping;
  slabs_.emplace(view, std::move(slab));
  return view;
}

void DiskSlabAllocator::release(void* ptr, Slab& slab) noexcept {
  ::UnmapViewOfFile(ptr);
  if (slab.mappingHandle) ::CloseHandle(static_cast<HANDLE>(slab.mappingHandle));
  if (slab.fileHandle) ::CloseHandle(static_cast<HANDLE>(slab.fileHandle));
  ::DeleteFileW(slab.path.wstring().c_str());
}

#else

void* DiskSlabAllocator::allocate(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  Slab slab;
  slab.size = bytes;
  slab.path = dir_ / ("slab_" + std::to_string(nextId_++) + ".bin");

  int fd = ::open(slab.path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) {
    error_ = "open failed for " + slab.path.string();
    return nullptr;
  }
  if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    error_ = "ftruncate failed for " + slab.path.string();
    ::close(fd);
    ::unlink(slab.path.c_str());
    return nullptr;
  }

  void* addr = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    error_ = "mmap failed for " + slab.path.string();
    ::close(fd);
    ::unlink(slab.path.c_str());
    return nullptr;
  }

  slab.fd = fd;
  slabs_.emplace(addr, std::move(slab));
  return addr;
}

void DiskSlabAllocator::release(void* ptr, Slab& slab) noexcept {
  ::munmap(ptr, slab.size);
  if (slab.fd >= 0) ::close(slab.fd);
  ::unlink(slab.path.c_str());
}

#endif

void DiskSlabAllocator::deallocate(void* ptr, std::size_t /*bytes*/) noexcept {
  auto it = slabs_.find(ptr);
  if (it == slabs_.end()) return;
  release(ptr, it->second);
  slabs_.erase(it);
}

}  // namespace qorvix::memory
