add_library(qorvix_vision STATIC
  src/inflate.cpp
  src/image.cpp
  src/png_encode.cpp
  src/preprocess.cpp
  src/clip_weights.cpp
  src/clip_model.cpp
  src/vision_check.cpp
)

target_include_directories(qorvix_vision PUBLIC include)
target_compile_features(qorvix_vision PUBLIC cxx_std_23)

# No third-party dependencies: DEFLATE, PNG and the ViT are all in-tree, so this builds under the
# `quick` preset. The vision tower reuses the runtime's quantized GEMV, ops and tensor loaders.
target_link_libraries(qorvix_vision PUBLIC qorvix_runtime qorvix_gguf)
