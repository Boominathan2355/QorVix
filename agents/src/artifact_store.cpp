#include "qorvix/agents/artifact_store.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace qorvix::agents {

ArtifactStore::ArtifactStore(ArtifactStore&& other) noexcept {
  std::lock_guard<std::mutex> lock(other.mutex_);
  artifacts_ = std::move(other.artifacts_);
  order_ = std::move(other.order_);
}

ArtifactStore& ArtifactStore::operator=(ArtifactStore&& other) noexcept {
  if (this != &other) {
    std::scoped_lock lock(mutex_, other.mutex_);
    artifacts_ = std::move(other.artifacts_);
    order_ = std::move(other.order_);
  }
  return *this;
}

bool ArtifactStore::createArtifact(Artifact artifact) {
  if (artifact.name().empty()) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  const auto& name = artifact.name();
  if (artifacts_.find(name) == artifacts_.end()) {
    order_.push_back(name);
  }
  artifacts_[name] = std::move(artifact);
  return true;
}

bool ArtifactStore::updateArtifact(const std::string& name,
                                  const std::string& newContent,
                                  const std::string& author,
                                  const std::string& changeSummary) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = artifacts_.find(name);
  if (it == artifacts_.end()) return false;
  it->second.updateContent(newContent, author, changeSummary);
  return true;
}

std::optional<Artifact> ArtifactStore::getArtifact(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = artifacts_.find(name);
  if (it != artifacts_.end()) return it->second;
  return std::nullopt;
}

bool ArtifactStore::hasArtifact(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return artifacts_.find(name) != artifacts_.end();
}

bool ArtifactStore::removeArtifact(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = artifacts_.find(name);
  if (it == artifacts_.end()) return false;
  artifacts_.erase(it);
  order_.erase(std::remove(order_.begin(), order_.end(), name), order_.end());
  return true;
}

void ArtifactStore::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  artifacts_.clear();
  order_.clear();
}

std::vector<std::string> ArtifactStore::artifactNames() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return order_;
}

std::vector<Artifact> ArtifactStore::listArtifacts() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Artifact> res;
  res.reserve(order_.size());
  for (const auto& name : order_) {
    auto it = artifacts_.find(name);
    if (it != artifacts_.end()) res.push_back(it->second);
  }
  return res;
}

std::vector<Artifact> ArtifactStore::filterArtifacts(const std::string& tag,
                                                    std::optional<ArtifactType> type,
                                                    const std::string& author) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Artifact> res;
  for (const auto& name : order_) {
    auto it = artifacts_.find(name);
    if (it == artifacts_.end()) continue;
    const auto& art = it->second;

    if (type.has_value() && art.type() != *type) continue;
    if (!author.empty() && art.metadata().author != author) continue;
    if (!tag.empty()) {
      const auto& tags = art.metadata().tags;
      if (std::find(tags.begin(), tags.end(), tag) == tags.end()) continue;
    }
    res.push_back(art);
  }
  return res;
}

bool ArtifactStore::syncToDirectory(const std::filesystem::path& dirPath, std::string& error) const {
  try {
    std::filesystem::create_directories(dirPath);
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& [name, art] : artifacts_) {
      std::string ext = ".txt";
      if (art.type() == ArtifactType::Code) {
        std::string lang = art.metadata().language;
        if (lang == "cpp" || lang == "c++") ext = ".cpp";
        else if (lang == "hpp" || lang == "h") ext = ".hpp";
        else if (lang == "python" || lang == "py") ext = ".py";
        else if (lang == "typescript" || lang == "ts") ext = ".ts";
        else if (lang == "javascript" || lang == "js") ext = ".js";
        else if (lang == "json") ext = ".json";
      } else if (art.type() == ArtifactType::Markdown || art.type() == ArtifactType::Document) {
        ext = ".md";
      } else if (art.type() == ArtifactType::Json) {
        ext = ".json";
      }

      std::filesystem::path target = dirPath / (name + ext);
      std::ofstream fout(target, std::ios::binary | std::ios::trunc);
      if (fout) {
        fout << art.content();
      }
    }
    return true;
  } catch (const std::exception& ex) {
    error = std::string("Sync to directory failed: ") + ex.what();
    return false;
  }
}

std::size_t ArtifactStore::loadFromDirectory(const std::filesystem::path& dirPath, std::string& error) {
  if (!std::filesystem::exists(dirPath)) {
    error = "Directory does not exist: " + dirPath.string();
    return 0;
  }

  std::size_t loadedCount = 0;
  try {
    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
      if (!entry.is_regular_file()) continue;

      std::string name = entry.path().stem().string();
      std::string ext = entry.path().extension().string();

      std::ifstream fin(entry.path(), std::ios::binary);
      if (!fin) continue;

      std::ostringstream ss;
      ss << fin.rdbuf();
      std::string content = ss.str();

      ArtifactType type = ArtifactType::Document;
      ArtifactMetadata meta;
      meta.title = name;
      meta.author = "filesystem";

      if (ext == ".cpp" || ext == ".hpp" || ext == ".py" || ext == ".ts" || ext == ".js") {
        type = ArtifactType::Code;
        meta.language = ext.substr(1);
      } else if (ext == ".md") {
        type = ArtifactType::Markdown;
      } else if (ext == ".json") {
        type = ArtifactType::Json;
      }

      Artifact art(name, type, content, meta);
      if (createArtifact(std::move(art))) {
        ++loadedCount;
      }
    }
  } catch (const std::exception& ex) {
    error = std::string("Failed to load artifacts from directory: ") + ex.what();
  }

  return loadedCount;
}

api::json::Value ArtifactStore::toJson() const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto arr = api::json::Value::array();
  for (const auto& name : order_) {
    auto it = artifacts_.find(name);
    if (it != artifacts_.end()) {
      arr.push(it->second.toJson());
    }
  }
  return arr;
}

bool ArtifactStore::fromJson(const api::json::Value& v, std::string& error) {
  if (!v.isArray()) {
    error = "ArtifactStore expects a JSON array";
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  clear();

  for (const auto& item : v.items()) {
    Artifact art;
    std::string err;
    if (Artifact::fromJson(item, art, err)) {
      order_.push_back(art.name());
      artifacts_[art.name()] = std::move(art);
    }
  }
  return true;
}

}  // namespace qorvix::agents
