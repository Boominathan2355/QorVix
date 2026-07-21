#pragma once

#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

#include "qorvix/model_info.hpp"

namespace qorvix {

// File extensions treated as loadable models during discovery. GGUF is the Phase 2 target;
// others are placeholders for the "future model formats" objective in the spec.
inline constexpr std::string_view kModelExtensions[] = {".gguf"};

bool isModelFile(const std::filesystem::path& path);

// Builds a ModelInfo for a single file (does not check the extension). Returns false if the
// file cannot be stat'd.
bool describeModel(const std::filesystem::path& path, ModelInfo& out);

// One-shot, non-recursive discovery of model files in a directory. Results are sorted by name.
// Missing directory yields an empty list, not an error (the models dir may not exist yet).
class ModelRegistry {
 public:
  const std::vector<ModelInfo>& scan(const std::filesystem::path& modelsDir);

  const std::vector<ModelInfo>& models() const noexcept { return models_; }
  bool empty() const noexcept { return models_.empty(); }
  std::size_t size() const noexcept { return models_.size(); }

 private:
  std::vector<ModelInfo> models_;
};

}  // namespace qorvix
