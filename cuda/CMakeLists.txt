# The cuda module always exists so the rest of the tree can call the backend facade
# unconditionally. It compiles ONE of two implementations:
#   * cuda_backend.cu       — real, via nvcc, when CUDA is enabled and a toolkit was found
#   * cuda_backend_stub.cpp — CPU stub otherwise (reports "no CUDA", returns null factories)
add_library(qorvix_cuda STATIC "")

# Tensor-parallel sharding math (Phase 10) is plain C++ with no CUDA dependency, so it compiles
# into BOTH configurations: the split logic is where multi-GPU bugs actually live, and keeping it
# toolkit-free means it stays unit-testable on a machine with no GPU.
target_sources(qorvix_cuda PRIVATE src/tensor_parallel.cpp)

target_include_directories(qorvix_cuda PUBLIC include)
target_compile_features(qorvix_cuda PUBLIC cxx_std_23)
target_link_libraries(qorvix_cuda PUBLIC qorvix_memory)

# QORVIX_ENABLE_CUDA is the top-level option; CMAKE_CUDA_COMPILER is set there via
# enable_language(CUDA) only when an actual toolkit is present (otherwise it degrades to OFF).
if(QORVIX_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  find_package(CUDAToolkit REQUIRED)
  target_sources(qorvix_cuda PRIVATE src/cuda_backend.cu)
  target_link_libraries(qorvix_cuda PUBLIC CUDA::cudart CUDA::cublas)
  set_target_properties(qorvix_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  # SASS for the datacenter/consumer arches the project targets (Turing..Hopper). Compiling for
  # these needs no local GPU; it only needs nvcc.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(qorvix_cuda PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90")
  endif()
  message(STATUS "qorvix_cuda: building the real CUDA backend (nvcc).")
else()
  target_sources(qorvix_cuda PRIVATE src/cuda_backend_stub.cpp)
  message(STATUS "qorvix_cuda: building the CPU stub (no CUDA toolkit / QORVIX_ENABLE_CUDA=OFF).")
endif()
