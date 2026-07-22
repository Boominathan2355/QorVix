find_package(Catch2 3 CONFIG QUIET)

if(Catch2_FOUND)
  add_executable(qorvix_tests
    core_smoke_test.cpp
    model_registry_test.cpp
    model_watcher_test.cpp
    plugin_registry_test.cpp
    gguf_file_test.cpp
    memory_test.cpp
    transfer_test.cpp
    cuda_test.cpp
    ops_test.cpp
    dequant_test.cpp
    model_config_test.cpp
    text_model_test.cpp
    tokenizer_test.cpp
    sampler_test.cpp
    generator_test.cpp
  )
  target_link_libraries(qorvix_tests PRIVATE
    qorvix_core qorvix_gguf qorvix_memory qorvix_cuda qorvix_runtime qorvix_tokenizer
    Catch2::Catch2WithMain)

  # The plugin test dlopen's the real example plugin; hand it the built library path and make
  # sure that library is built before the tests run.
  add_dependencies(qorvix_tests qorvix_plugin_example)
  target_compile_definitions(qorvix_tests PRIVATE
    QORVIX_EXAMPLE_PLUGIN_PATH="$<TARGET_FILE:qorvix_plugin_example>")

  include(CTest)
  include(Catch)
  catch_discover_tests(qorvix_tests)
else()
  message(STATUS "Catch2 not found (vcpkg not bootstrapped yet) — skipping qorvix_tests. See docs/BUILDING.md.")
endif()
