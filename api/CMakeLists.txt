# Protocol layer only (JSON + OpenAI schema) — no external dependencies, so it builds in every
# preset. The HTTP transport (Boost.Beast) is added as a separate, optional piece in Phase 9b.
add_library(qorvix_api STATIC
  src/json.cpp
  src/openai.cpp
)

target_include_directories(qorvix_api PUBLIC include)
target_compile_features(qorvix_api PUBLIC cxx_std_23)
