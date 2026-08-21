#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "qorvix/agents/artifact.hpp"

namespace qorvix::agents {

// Thread-safe centralized repository for artifacts created and managed by agents
class ArtifactStore {
 public:
  ArtifactStore() = default;
  ~ArtifactStore() = default;

  // Non-copyable, movable
  ArtifactStore(const ArtifactStore&) = delete;
  ArtifactStore& operator=(const ArtifactStore&) = delete;
  ArtifactStore(ArtifactStore&&) noexcept;
  ArtifactStore& operator=(ArtifactStore&&) noexcept;

  // CRUD
  bool createArtifact(Artifact artifact);
  bool updateArtifact(const std::string& name,
                      const std::string& newContent,
                      const std::string& author = "",
                      const std::string& changeSummary = "");
  std::optional<Artifact> getArtifact(const std::string& name) const;
  bool hasArtifact(const std::string& name) const;
  bool removeArtifact(const std::string& name);
  void clear();

  // Queries
  std::vector<std::string> artifactNames() const;
  std::vector<Artifact> listArtifacts() const;
  std::vector<Artifact> filterArtifacts(const std::string& tag = "",
                                       std::optional<ArtifactType> type = std::nullopt,
                                       const std::string& author = "") const;

  // Filesystem persistence and directory synchronization
  bool syncToDirectory(const std::filesystem::path& dirPath, std::string& error) const;
  std::size_t loadFromDirectory(const std::filesystem::path& dirPath, std::string& error);

  // Serialization
  api::json::Value toJson() const;
  bool fromJson(const api::json::Value& v, std::string& error);

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, Artifact> artifacts_;
  std::vector<std::string> order_;
};

}  // namespace qorvix::agents
