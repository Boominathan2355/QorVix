add_library(qorvix_audio STATIC
  src/fft.cpp
  src/audio_file.cpp
  src/mel.cpp
  src/audio_check.cpp
)

target_include_directories(qorvix_audio PUBLIC include)
target_compile_features(qorvix_audio PUBLIC cxx_std_23)

# No third-party dependencies and — for now — no dependency on the runtime either: Phase 11b-3a is
# the front end alone (WAV, FFT, log-mel), which is pure signal processing over std::vector. The
# Whisper encoder/decoder in 11b-3b will link qorvix_runtime and qorvix_gguf the way qorvix_vision
# does; the link line grows when there is something behind it, not before.
