add_library(qorvix_runtime STATIC
  src/ops.cpp
  src/dequant.cpp
  src/model_config.cpp
  src/weights.cpp
  src/text_model.cpp
)

target_include_directories(qorvix_runtime PUBLIC include)
target_compile_features(qorvix_runtime PUBLIC cxx_std_23)

# dequant references the GGUF ggml_type enum for block layouts.
target_link_libraries(qorvix_runtime PUBLIC qorvix_gguf)
