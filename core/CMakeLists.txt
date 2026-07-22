find_package(Threads REQUIRED)

add_library(qorvix_core STATIC
  src/version.cpp
  src/dynamic_library.cpp
  src/plugin_registry.cpp
  src/model_registry.cpp
  src/model_watcher.cpp
)

target_include_directories(qorvix_core PUBLIC include)
target_compile_features(qorvix_core PUBLIC cxx_std_23)

# Threads: ModelWatcher's polling thread. ${CMAKE_DL_LIBS}: dlopen/dlclose on Linux (empty on
# Windows, where LoadLibrary is in the base API).
target_link_libraries(qorvix_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

add_executable(qorvix src/main.cpp)
target_link_libraries(qorvix PRIVATE qorvix_core qorvix_gguf qorvix_cuda qorvix_runtime
  qorvix_tokenizer qorvix_scheduler qorvix_api)
