# Reference plugin. Built as a MODULE library (loaded via dlopen/LoadLibrary, never linked).
add_library(qorvix_plugin_example MODULE example_plugin.cpp)

# Only needs the interface header, not the core static library.
target_include_directories(qorvix_plugin_example PRIVATE ${CMAKE_SOURCE_DIR}/core/include)
target_compile_features(qorvix_plugin_example PRIVATE cxx_std_23)

set_target_properties(qorvix_plugin_example PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
