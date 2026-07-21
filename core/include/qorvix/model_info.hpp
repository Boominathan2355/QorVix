#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace qorvix {

// A model file discovered on disk. Deep inspection (GGUF header, architecture, quantization) is
// deferred to the parser in Phase 2 — this is what the discovery layer can know from the file
// itself without opening it.
struct ModelInfo {
  std::string name;                // filename stem, e.g. "llama3"
  std::filesystem::path path;      // absolute path to the file
  std::uintmax_t sizeBytes = 0;    // file size at discovery time
  std::string format;             // lowercase extension without dot, e.g. "gguf"

  bool operator==(const ModelInfo& other) const noexcept { return path == other.path; }
};

}  // namespace qorvix
