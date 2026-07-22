add_library(qorvix_runtime STATIC
  src/ops.cpp
  src/dequant.cpp
  src/qmatmul.cpp
  src/model_config.cpp
  src/weights.cpp
  src/text_model.cpp
  src/sampler.cpp
  src/generator.cpp
)

target_include_directories(qorvix_runtime PUBLIC include)
target_compile_features(qorvix_runtime PUBLIC cxx_std_23)

# dequant references the GGUF ggml_type enum; the generator drives the tokenizer.
target_link_libraries(qorvix_runtime PUBLIC qorvix_gguf qorvix_tokenizer)

# Multithread the CPU reference matmul (the forward pass's dominant cost) via OpenMP when
# available. Optional: without it the #pragma omp lines are simply ignored.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(qorvix_runtime PUBLIC OpenMP::OpenMP_CXX)
  message(STATUS "qorvix_runtime: OpenMP enabled")
else()
  message(STATUS "qorvix_runtime: OpenMP not found — matmul runs single-threaded")
endif()

# -march=native unlocks the host's SIMD (AVX2/FMA) for the auto-vectorized dot products. Off by
# default (non-portable binaries); enable for local/benchmark builds via -DQORVIX_NATIVE=ON.
if(QORVIX_NATIVE)
  if(MSVC)
    target_compile_options(qorvix_runtime PRIVATE /arch:AVX2)
  else()
    target_compile_options(qorvix_runtime PRIVATE -march=native)
  endif()
endif()
