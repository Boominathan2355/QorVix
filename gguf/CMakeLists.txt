add_library(qorvix_gguf STATIC
  src/gguf_types.cpp
  src/mapped_file.cpp
  src/gguf_file.cpp
)

target_include_directories(qorvix_gguf PUBLIC include)
target_compile_features(qorvix_gguf PUBLIC cxx_std_23)
