# Protocol layer only (JSON + OpenAI schema) — no external dependencies, so it builds in every
# preset. The HTTP transport (Boost.Beast) is added as a separate, optional piece in Phase 9b.
add_library(qorvix_api STATIC
  src/json.cpp
  src/openai.cpp
  src/http_server.cpp
)

target_include_directories(qorvix_api PUBLIC include)
target_compile_features(qorvix_api PUBLIC cxx_std_23)

# Winsock for the from-scratch HTTP server on Windows (POSIX needs nothing extra).
if(WIN32)
  target_link_libraries(qorvix_api PUBLIC ws2_32)
endif()
