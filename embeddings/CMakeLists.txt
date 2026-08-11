add_library(qorvix_embeddings STATIC
  src/bert_model.cpp
)

target_include_directories(qorvix_embeddings PUBLIC include)
target_compile_features(qorvix_embeddings PUBLIC cxx_std_23)

# The encoder reuses the runtime's quantized GEMV, dequant, ops and encoder weight loader; it owns
# only the forward pass and the seam. No tokenizer dependency: callers supply token ids.
target_link_libraries(qorvix_embeddings PUBLIC qorvix_runtime qorvix_gguf)
