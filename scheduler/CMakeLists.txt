add_library(qorvix_scheduler STATIC
  src/scheduler.cpp
)

target_include_directories(qorvix_scheduler PUBLIC include)
target_compile_features(qorvix_scheduler PUBLIC cxx_std_23)
target_link_libraries(qorvix_scheduler PUBLIC qorvix_runtime qorvix_tokenizer qorvix_memory)
