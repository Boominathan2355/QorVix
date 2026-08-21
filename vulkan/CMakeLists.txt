# The vulkan module always exists so the rest of the tree can call the backend facade
# unconditionally. It compiles ONE of two implementations:
#   * vulkan_backend.cpp      — real, when QORVIX_ENABLE_VULKAN is ON and the Vulkan loader + a
#                               GLSL->SPIR-V compiler (glslangValidator/glslc) are found
#   * vulkan_backend_stub.cpp — CPU stub otherwise (reports "no Vulkan")
add_library(qorvix_vulkan STATIC "")

target_include_directories(qorvix_vulkan PUBLIC include)
target_compile_features(qorvix_vulkan PUBLIC cxx_std_23)
target_link_libraries(qorvix_vulkan PUBLIC qorvix_memory)

set(_vk_real OFF)
if(QORVIX_ENABLE_VULKAN)
  find_package(Vulkan)
  # A GLSL->SPIR-V compiler is required to embed the compute shaders. Prefer the one Vulkan::glslang
  # exposes; fall back to a glslangValidator/glslc on PATH.
  find_program(QORVIX_GLSLANG NAMES glslangValidator glslang)
  find_program(QORVIX_GLSLC NAMES glslc)
  if(Vulkan_FOUND AND (QORVIX_GLSLANG OR QORVIX_GLSLC))
    set(_vk_real ON)
  else()
    message(WARNING "QORVIX_ENABLE_VULKAN is ON but Vulkan loader or a GLSL compiler was not found; "
                    "building the Vulkan stub. (need libvulkan-dev + glslang-tools or shaderc)")
  endif()
endif()

if(_vk_real)
  set(_shader_dir "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
  set(_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
  file(MAKE_DIRECTORY "${_gen_dir}")

  # Compile each .comp to a SPIR-V C header (uint32_t array named <stem>_spv) so the shader bytes
  # are embedded in the binary — no shader files to ship or locate at runtime.
  set(_spv_headers "")
  foreach(_shader qmatmul_q8_0 qmatmul_q4_k qmatmul_q6_k rmsnorm rope swiglu add attention
                  matmul_f32 embed_row kv_store)
    set(_src "${_shader_dir}/${_shader}.comp")
    set(_hdr "${_gen_dir}/${_shader}_spv.h")
    if(QORVIX_GLSLANG)
      add_custom_command(
        OUTPUT "${_hdr}"
        COMMAND "${QORVIX_GLSLANG}" -V --target-env vulkan1.1
                --vn ${_shader}_spv -o "${_hdr}" "${_src}"
        DEPENDS "${_src}"
        COMMENT "glslang: ${_shader}.comp -> ${_shader}_spv.h"
        VERBATIM)
    else()
      # glslc emits a raw .spv; wrap it into a header with a tiny CMake step.
      set(_spv "${_gen_dir}/${_shader}.spv")
      add_custom_command(
        OUTPUT "${_hdr}"
        COMMAND "${QORVIX_GLSLC}" --target-env=vulkan1.1 -mfmt=c -o "${_hdr}.tmp" "${_src}"
        COMMAND ${CMAKE_COMMAND} -DIN=${_hdr}.tmp -DOUT=${_hdr} -DVN=${_shader}_spv
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/wrap_spv.cmake"
        DEPENDS "${_src}"
        COMMENT "glslc: ${_shader}.comp -> ${_shader}_spv.h"
        VERBATIM)
    endif()
    list(APPEND _spv_headers "${_hdr}")
  endforeach()
  add_custom_target(qorvix_vulkan_shaders DEPENDS ${_spv_headers})

  target_sources(qorvix_vulkan PRIVATE src/vulkan_backend.cpp src/embedding_model.cpp src/clip_model.cpp)
  add_dependencies(qorvix_vulkan qorvix_vulkan_shaders)
  target_include_directories(qorvix_vulkan PRIVATE "${_gen_dir}")
  target_link_libraries(qorvix_vulkan PUBLIC Vulkan::Vulkan)
  target_compile_definitions(qorvix_vulkan PRIVATE QORVIX_HAVE_VULKAN=1)
  message(STATUS "qorvix_vulkan: building the real Vulkan backend (loader + glslang).")
else()
  target_sources(qorvix_vulkan PRIVATE src/vulkan_backend_stub.cpp src/embedding_model.cpp src/clip_model.cpp)
  message(STATUS "qorvix_vulkan: building the CPU stub (no Vulkan / QORVIX_ENABLE_VULKAN=OFF).")
endif()
