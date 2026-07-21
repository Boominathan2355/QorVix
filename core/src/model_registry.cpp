#include "qorvix/model_registry.hpp"

#include <algorithm>
#include <cctype>
#include <system_error>

namespace qorvix {

namespace {

std::string toLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

bool isModelFile(const std::filesystem::path& path) {
  const std::string ext = toLower(path.extension().string());
  for (std::string_view known : kModelExtensions) {
    if (ext == known) return true;
  }
  return false;
}

bool describeModel(const std::filesystem::path& path, ModelInfo& out) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) return false;

  out.name = path.stem().string();
  out.path = std::filesystem::absolute(path, ec);
  if (ec) out.path = path;
  out.sizeBytes = size;
  std::string ext = toLower(path.extension().string());
  out.format = ext.empty() ? std::string{} : ext.substr(1);
  return true;
}

const std::vector<ModelInfo>& ModelRegistry::scan(const std::filesystem::path& modelsDir) {
  models_.clear();

  std::error_code ec;
  if (!std::filesystem::is_directory(modelsDir, ec)) return models_;

  for (const auto& entry : std::filesystem::directory_iterator(modelsDir, ec)) {
    if (ec) break;
    if (!entry.is_regular_file()) continue;
    if (!isModelFile(entry.path())) continue;
    ModelInfo info;
    if (describeModel(entry.path(), info)) models_.push_back(std::move(info));
  }

  std::sort(models_.begin(), models_.end(),
            [](const ModelInfo& a, const ModelInfo& b) { return a.name < b.name; });
  return models_;
}

}  // namespace qorvix
