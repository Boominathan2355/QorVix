add_library(qorvix_core STATIC
  src/version.cpp
)

target_include_directories(qorvix_core PUBLIC include)
target_compile_features(qorvix_core PUBLIC cxx_std_23)

add_executable(qorvix src/main.cpp)
target_link_libraries(qorvix PRIVATE qorvix_core)
