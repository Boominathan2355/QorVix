add_library(qorvix_image STATIC
  src/nn.cpp
  src/blocks.cpp
  src/clip_tokenizer.cpp
  src/sd_weights.cpp
  src/text_encoder.cpp
  src/unet.cpp
  src/vae.cpp
  src/scheduler.cpp
  src/diffusion.cpp
  src/sd_check.cpp
)

target_include_directories(qorvix_image PUBLIC include)
target_compile_features(qorvix_image PUBLIC cxx_std_23)

# No third-party dependencies, like every other model module here: the UNet, the VAE decoder, the
# CLIP text encoder and the samplers are all in-tree, so this builds under the `quick` preset.
# qorvix_vision supplies the Image struct and the PNG writer — the decoder's other half — rather
# than this module growing a second image type.
target_link_libraries(qorvix_image PUBLIC qorvix_runtime qorvix_gguf qorvix_vision)

# Unlike the other model modules, this one asks for -march=native explicitly rather than only
# inheriting OpenMP from qorvix_runtime. The reason is where the time goes: a diffusion sample is
# dominated by im2col convolutions and 4096-token self-attention whose inner loops live HERE, not
# in the runtime's GEMV, so leaving them at baseline SSE2 while the runtime runs AVX2 would
# vectorize the smaller half of the work. Off by default like everywhere else (-DQORVIX_NATIVE=ON).
if(QORVIX_NATIVE)
  if(MSVC)
    target_compile_options(qorvix_image PRIVATE /arch:AVX2)
  else()
    target_compile_options(qorvix_image PRIVATE -march=native)
  endif()
endif()
