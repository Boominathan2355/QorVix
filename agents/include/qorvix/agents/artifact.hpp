#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "qorvix/api/json.hpp"

namespace qorvix::agents {

enum class ArtifactType {
  Code,
  Markdown,
  Document,
  Diff,
  Table,
  Json,
  Image,
  Binary,
  Custom
};

std::string_view artifactTypeName(ArtifactType type);
ArtifactType parseArtifactType(std::string_view name);

struct ArtifactMetadata {
  std::string title;
  std::string summary;
  std::string language;  // "cpp", "typescript", "python", "json", "markdown", etc.
  std::vector<std::string> tags;
  std::string author;
  bool userFacing = true;
  bool isExecutable = false;
  std::int64_t createdAt = 0;
  std::int64_t updatedAt = 0;
  std::string mimeType;

  api::json::Value toJson() const;
  static bool fromJson(const api::json::Value& v, ArtifactMetadata& out, std::string& error);
};

struct ArtifactVersion {
  int versionNumber = 1;
  std::string content;
  std::string author;
  std::string changeSummary;
  std::int64_t timestamp = 0;

  api::json::Value toJson() const;
  static bool fromJson(const api::json::Value& v, ArtifactVersion& out, std::string& error);
};

struct ArtifactDiffLine {
  enum class Type { Context, Added, Removed };
  Type type = Type::Context;
  int oldLineNumber = 0;
  int newLineNumber = 0;
  std::string line;
};

struct ArtifactDiff {
  int fromVersion = 0;
  int toVersion = 0;
  std::vector<ArtifactDiffLine> lines;
  int additions = 0;
  int deletions = 0;

  std::string toString() const;
  api::json::Value toJson() const;
};

class Artifact {
 public:
  Artifact() = default;
  Artifact(std::string name, ArtifactType type, std::string content, ArtifactMetadata metadata = {});

  const std::string& id() const { return id_; }
  void setId(std::string id) { id_ = std::move(id); }

  const std::string& name() const { return name_; }
  void setName(std::string name) { name_ = std::move(name); }

  ArtifactType type() const { return type_; }
  void setType(ArtifactType type) { type_ = type; }

  const std::string& content() const { return content_; }
  const ArtifactMetadata& metadata() const { return metadata_; }
  ArtifactMetadata& mutableMetadata() { return metadata_; }

  int currentVersion() const { return currentVersion_; }
  const std::vector<ArtifactVersion>& history() const { return history_; }

  // Updates content, creating a new historical version snapshot with a diff record
  void updateContent(std::string newContent, const std::string& author = "", const std::string& changeSummary = "");

  // Retrieves content from a specific past version
  std::optional<std::string> getContentAtVersion(int version) const;

  // Computes unified diff between two versions (or between past version and current)
  ArtifactDiff diff(int fromVersion, int toVersion = -1) const;

  // Serialization
  api::json::Value toJson() const;
  static bool fromJson(const api::json::Value& v, Artifact& out, std::string& error);

  // Markdown representation (includes metadata header, language fenced block, and change notes)
  std::string toMarkdown() const;

 private:
  std::string id_;
  std::string name_;
  ArtifactType type_ = ArtifactType::Document;
  std::string content_;
  ArtifactMetadata metadata_;
  int currentVersion_ = 1;
  std::vector<ArtifactVersion> history_;
};

}  // namespace qorvix::agents
