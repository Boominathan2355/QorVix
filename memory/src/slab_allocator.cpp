#include "qorvix/memory/slab_allocator.hpp"

#include <new>

namespace qorvix::memory {

void* HostSlabAllocator::allocate(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  return ::operator new(bytes, std::align_val_t{kSlabAlign}, std::nothrow);
}

void HostSlabAllocator::deallocate(void* ptr, std::size_t /*bytes*/) noexcept {
  ::operator delete(ptr, std::align_val_t{kSlabAlign});
}

}  // namespace qorvix::memory
