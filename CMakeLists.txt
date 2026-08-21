cmake_minimum_required(VERSION 3.28)
project(qorvix VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

option(QORVIX_ENABLE_CUDA "Build the CUDA backend (requires CUDA 12+ toolkit)" OFF)
option(QORVIX_ENABLE_VULKAN "Build the Vulkan compute backend (runs on any vendor; needs the Vulkan loader + glslang)" OFF)
# NCCL is the fast multi-GPU collective transport. Optional even inside a CUDA build: without it
# the host-staged transport still works (correctly, just slower), so this changes throughput on
# multi-GPU hosts, never whether tensor parallelism runs. Needs QORVIX_ENABLE_CUDA.
option(QORVIX_ENABLE_NCCL "Build the NCCL collective transport for multi-GPU (requires CUDA + NCCL)" OFF)
option(QORVIX_BUILD_TESTS "Build unit tests" ON)

if(QORVIX_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  else()
    message(WARNING "QORVIX_ENABLE_CUDA is ON but no CUDA compiler was found; building without CUDA.")
    set(QORVIX_ENABLE_CUDA OFF)
  endif()
endif()

# NCCL lives inside the CUDA build (it needs cudaStream_t and a device per rank), so asking for it
# without CUDA is a configuration mistake worth saying out loud rather than silently ignoring.
if(QORVIX_ENABLE_NCCL AND NOT QORVIX_ENABLE_CUDA)
  message(WARNING "QORVIX_ENABLE_NCCL needs QORVIX_ENABLE_CUDA; turning it off.")
  set(QORVIX_ENABLE_NCCL OFF)
endif()

add_subdirectory(gguf)
add_subdirectory(memory)
add_subdirectory(cuda)
add_subdirectory(vulkan)
add_subdirectory(tokenizer)
add_subdirectory(runtime)
add_subdirectory(embeddings)
add_subdirectory(vision)
add_subdirectory(audio)
add_subdirectory(image)
add_subdirectory(rag)
add_subdirectory(agents)
add_subdirectory(scheduler)
add_subdirectory(api)
add_subdirectory(core)
add_subdirectory(plugins)

if(QORVIX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Modules are added here as their roadmap phase begins (see ROADMAP.md) — not before,
# so the build never carries libraries with no real content behind them.
# add_subdirectory(api)       # Phase 9
