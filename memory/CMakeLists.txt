add_library(qorvix_memory STATIC
  src/slab_allocator.cpp
  src/disk_allocator.cpp
  src/page.cpp
  src/page_pool.cpp
  src/memory_manager.cpp
  src/kv_cache.cpp
)

target_include_directories(qorvix_memory PUBLIC include)
target_compile_features(qorvix_memory PUBLIC cxx_std_23)
