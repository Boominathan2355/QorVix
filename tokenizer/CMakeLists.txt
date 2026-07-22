add_library(qorvix_tokenizer STATIC
  src/tokenizer.cpp
)

target_include_directories(qorvix_tokenizer PUBLIC include)
target_compile_features(qorvix_tokenizer PUBLIC cxx_std_23)
target_link_libraries(qorvix_tokenizer PUBLIC qorvix_gguf)
