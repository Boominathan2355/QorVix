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

  # NCCL: optional inside the CUDA build. It is the transport that actually uses NVLink/PCIe P2P
  # bandwidth for the all-reduce; when it is absent the seam falls back to staging through host
  # memory, which is correct everywhere (and is also what lets several ranks share one GPU for
  # verification). So a missing NCCL costs speed on multi-GPU hosts, not capability — which is why
  # this is a warning and not an error.
  if(QORVIX_ENABLE_NCCL)
    find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${CUDAToolkit_INCLUDE_DIRS} ENV NCCL_ROOT
              PATH_SUFFIXES include)
    find_library(NCCL_LIBRARY NAMES nccl HINTS ${CUDAToolkit_LIBRARY_DIR} ENV NCCL_ROOT
                 PATH_SUFFIXES lib lib64)
    if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
      target_include_directories(qorvix_cuda PRIVATE ${NCCL_INCLUDE_DIR})
      target_link_libraries(qorvix_cuda PUBLIC ${NCCL_LIBRARY})
      # PRIVATE, but it reaches every source in this target - including tensor_parallel.cpp, whose
      # #ifndef QORVIX_WITH_NCCL fallback must disappear when the real one is compiled in.
      target_compile_definitions(qorvix_cuda PRIVATE QORVIX_WITH_NCCL=1)
      message(STATUS "qorvix_cuda: NCCL found (${NCCL_LIBRARY}) - building the NCCL collective.")
    else()
      message(WARNING "QORVIX_ENABLE_NCCL is ON but nccl.h / libnccl were not found; "
                      "multi-GPU will use the host-staged collective instead.")
    endif()
  endif()
else()
  target_sources(qorvix_cuda PRIVATE src/cuda_backend_stub.cpp)
  message(STATUS "qorvix_cuda: building the CPU stub (no CUDA toolkit / QORVIX_ENABLE_CUDA=OFF).")
endif()
