add_library(qorvix_audio STATIC
  src/fft.cpp
  src/audio_file.cpp
  src/mel.cpp
  src/audio_check.cpp
  src/whisper_weights.cpp
  src/whisper_model.cpp
  src/whisper_check.cpp
)

target_include_directories(qorvix_audio PUBLIC include)
target_compile_features(qorvix_audio PUBLIC cxx_std_23)

# No third-party dependencies: the WAV reader, the FFT, the mel filter bank and the Whisper
# encoder/decoder are all in-tree, so this builds under the `quick` preset. Phase 11b-3a was the
# front end alone and linked nothing; 11b-3b reuses the runtime's quantized GEMV, ops and tensor
# loaders, the GGUF reader, and the byte-level BPE tokenizer — Whisper needed no new tokenizer.
target_link_libraries(qorvix_audio PUBLIC qorvix_runtime qorvix_gguf qorvix_tokenizer)
