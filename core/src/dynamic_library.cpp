#include "qorvix/dynamic_library.hpp"

#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace qorvix {

namespace {

#if defined(_WIN32)
std::string lastSystemError() {
  const DWORD code = ::GetLastError();
  if (code == 0) return {};
  LPSTR buffer = nullptr;
  const DWORD len = ::FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPSTR>(&buffer),
      0, nullptr);
  std::string message(buffer, len);
  if (buffer) ::LocalFree(buffer);
  while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) message.pop_back();
  return message;
}
#endif

}  // namespace

DynamicLibrary::DynamicLibrary(const std::filesystem::path& path) {
#if defined(_WIN32)
  handle_ = ::LoadLibraryW(path.wstring().c_str());
  if (!handle_) error_ = lastSystemError();
#else
  handle_ = ::dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle_) {
    const char* err = ::dlerror();
    error_ = err ? err : "dlopen failed";
  }
#endif
}

DynamicLibrary::~DynamicLibrary() { close(); }

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)), error_(std::move(other.error_)) {}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
  if (this != &other) {
    close();
    handle_ = std::exchange(other.handle_, nullptr);
    error_ = std::move(other.error_);
  }
  return *this;
}

void DynamicLibrary::close() noexcept {
  if (!handle_) return;
#if defined(_WIN32)
  ::FreeLibrary(static_cast<HMODULE>(handle_));
#else
  ::dlclose(handle_);
#endif
  handle_ = nullptr;
}

void* DynamicLibrary::symbol(const char* name) const {
  if (!handle_) {
    error_ = "library not loaded";
    return nullptr;
  }
#if defined(_WIN32)
  void* sym = reinterpret_cast<void*>(::GetProcAddress(static_cast<HMODULE>(handle_), name));
  if (!sym) error_ = lastSystemError();
  return sym;
#else
  ::dlerror();  // clear stale error
  void* sym = ::dlsym(handle_, name);
  if (!sym) {
    const char* err = ::dlerror();
    if (err) error_ = err;
  }
  return sym;
#endif
}

const char* DynamicLibrary::nativeExtension() noexcept {
#if defined(_WIN32)
  return ".dll";
#elif defined(__APPLE__)
  return ".dylib";
#else
  return ".so";
#endif
}

}  // namespace qorvix
