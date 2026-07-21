#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

#include "qorvix/memory/memory_manager.hpp"
#include "qorvix/memory/slab_allocator.hpp"
#include "qorvix/memory/transfer.hpp"

// Bridges the CUDA backend into the Phase 3 memory manager: a GpuVram slab allocator and a
// transfer engine that routes GPU tiers through cudaMemcpy. Both factories return nullptr in a
// CPU build (no CUDA), so a caller can attempt a GPU tier and cleanly fall back.
namespace qorvix::cuda {

// cudaMalloc/cudaFree slabs for the GpuVram tier. Slabs are device pointers: MemoryPage does
// pointer arithmetic on them (never dereferences), so page management works unchanged; only the
// transfer engine ever touches the bytes. Returns nullptr if CUDA isn't built in or no device.
std::unique_ptr<memory::ISlabAllocator> makeGpuSlabAllocator();

// Transfer engine that memcpys between host tiers and cudaMemcpys whenever a GpuVram tier is one
// end. Returns nullptr in a CPU build. Pass to MemoryManager to enable GPU migration.
std::unique_ptr<memory::ITransferEngine> makeCudaTransferEngine();

// Convenience: a MemoryManager with GpuVram + HostRam + DiskNvme tiers wired with the CUDA
// transfer engine. Returns nullptr (and leaves *why empty or set) when CUDA/device is absent —
// callers fall back to MemoryManager::makeHostAndDisk.
std::unique_ptr<memory::MemoryManager> makeGpuMemoryManager(std::size_t vramBudgetBytes,
                                                            std::size_t ramBudgetBytes,
                                                            std::size_t diskBudgetBytes,
                                                            const std::filesystem::path& spoolDir);

}  // namespace qorvix::cuda
