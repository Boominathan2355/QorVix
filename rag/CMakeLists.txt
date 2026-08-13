add_library(qorvix_rag STATIC
  src/chunker.cpp
  src/loaders.cpp
  src/vector_store.cpp
  src/bm25.cpp
  src/hybrid_search.cpp
  src/pipeline.cpp
)

target_include_directories(qorvix_rag PUBLIC include)
target_compile_features(qorvix_rag PUBLIC cxx_std_23)

# No third-party dependencies, so this builds under the `quick` preset like qorvix_api does.
# The chunker needs the embedding model's own tokenizer; the pipeline drives the encoder seam.
target_link_libraries(qorvix_rag PUBLIC qorvix_embeddings qorvix_runtime qorvix_tokenizer)
