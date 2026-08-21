#include "qorvix/agents/artifact.hpp"

#include <chrono>
#include <sstream>

namespace qorvix::agents {

namespace {
std::int64_t currentEpochMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::vector<std::string> splitLines(std::string_view s) {
  std::vector<std::string> lines;
  std::string cur;
  for (char c : s) {
    if (c == '\r') continue;
    if (c == '\n') {
      lines.push_back(std::move(cur));
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  lines.push_back(std::move(cur));
  return lines;
}
}  // namespace

std::string_view artifactTypeName(ArtifactType type) {
  switch (type) {
    case ArtifactType::Code: return "code";
    case ArtifactType::Markdown: return "markdown";
    case ArtifactType::Document: return "document";
    case ArtifactType::Diff: return "diff";
    case ArtifactType::Table: return "table";
    case ArtifactType::Json: return "json";
    case ArtifactType::Image: return "image";
    case ArtifactType::Binary: return "binary";
    case ArtifactType::Custom: return "custom";
  }
  return "document";
}

ArtifactType parseArtifactType(std::string_view name) {
  if (name == "code") return ArtifactType::Code;
  if (name == "markdown") return ArtifactType::Markdown;
  if (name == "diff") return ArtifactType::Diff;
  if (name == "table") return ArtifactType::Table;
  if (name == "json") return ArtifactType::Json;
  if (name == "image") return ArtifactType::Image;
  if (name == "binary") return ArtifactType::Binary;
  if (name == "custom") return ArtifactType::Custom;
  return ArtifactType::Document;
}

api::json::Value ArtifactMetadata::toJson() const {
  auto v = api::json::Value::object();
  v["title"] = title;
  v["summary"] = summary;
  v["language"] = language;
  v["author"] = author;
  v["user_facing"] = userFacing;
  v["is_executable"] = isExecutable;
  v["created_at"] = createdAt;
  v["updated_at"] = updatedAt;
  if (!mimeType.empty()) v["mime_type"] = mimeType;

  auto t = api::json::Value::array();
  for (const auto& tag : tags) t.push(tag);
  v["tags"] = std::move(t);
  return v;
}

bool ArtifactMetadata::fromJson(const api::json::Value& v, ArtifactMetadata& out, std::string& error) {
  if (!v.isObject()) {
    error = "ArtifactMetadata expects a JSON object";
    return false;
  }
  if (const auto* t = v.get("title"); t && t->isString()) out.title = t->asString();
  if (const auto* s = v.get("summary"); s && s->isString()) out.summary = s->asString();
  if (const auto* l = v.get("language"); l && l->isString()) out.language = l->asString();
  if (const auto* a = v.get("author"); a && a->isString()) out.author = a->asString();
  if (const auto* uf = v.get("user_facing"); uf && uf->isBool()) out.userFacing = uf->asBool();
  if (const auto* ex = v.get("is_executable"); ex && ex->isBool()) out.isExecutable = ex->asBool();
  if (const auto* c = v.get("created_at"); c && c->isNumber()) out.createdAt = static_cast<std::int64_t>(c->asNumber());
  if (const auto* u = v.get("updated_at"); u && u->isNumber()) out.updatedAt = static_cast<std::int64_t>(u->asNumber());
  if (const auto* m = v.get("mime_type"); m && m->isString()) out.mimeType = m->asString();

  out.tags.clear();
  if (const auto* tg = v.get("tags"); tg && tg->isArray()) {
    for (const auto& item : tg->items()) {
      if (item.isString()) out.tags.push_back(item.asString());
    }
  }
  return true;
}

api::json::Value ArtifactVersion::toJson() const {
  auto v = api::json::Value::object();
  v["version"] = versionNumber;
  v["content"] = content;
  v["author"] = author;
  v["change_summary"] = changeSummary;
  v["timestamp"] = timestamp;
  return v;
}

bool ArtifactVersion::fromJson(const api::json::Value& v, ArtifactVersion& out, std::string& error) {
  if (!v.isObject()) {
    error = "ArtifactVersion expects a JSON object";
    return false;
  }
  if (const auto* ver = v.get("version"); ver && ver->isNumber()) out.versionNumber = ver->asInt();
  if (const auto* c = v.get("content"); c && c->isString()) out.content = c->asString();
  if (const auto* a = v.get("author"); a && a->isString()) out.author = a->asString();
  if (const auto* cs = v.get("change_summary"); cs && cs->isString()) out.changeSummary = cs->asString();
  if (const auto* t = v.get("timestamp"); t && t->isNumber()) out.timestamp = static_cast<std::int64_t>(t->asNumber());
  return true;
}

std::string ArtifactDiff::toString() const {
  std::ostringstream ss;
  ss << "--- v" << fromVersion << "\n"
     << "+++ v" << toVersion << "\n";
  for (const auto& dl : lines) {
    if (dl.type == ArtifactDiffLine::Type::Added) {
      ss << "+" << dl.line << "\n";
    } else if (dl.type == ArtifactDiffLine::Type::Removed) {
      ss << "-" << dl.line << "\n";
    } else {
      ss << " " << dl.line << "\n";
    }
  }
  return ss.str();
}

api::json::Value ArtifactDiff::toJson() const {
  auto v = api::json::Value::object();
  v["from_version"] = fromVersion;
  v["to_version"] = toVersion;
  v["additions"] = additions;
  v["deletions"] = deletions;

  auto diffLines = api::json::Value::array();
  for (const auto& dl : lines) {
    auto item = api::json::Value::object();
    item["type"] = dl.type == ArtifactDiffLine::Type::Added ? "add" : (dl.type == ArtifactDiffLine::Type::Removed ? "remove" : "context");
    item["line"] = dl.line;
    diffLines.push(std::move(item));
  }
  v["lines"] = std::move(diffLines);
  return v;
}

// Artifact implementation
Artifact::Artifact(std::string name, ArtifactType type, std::string content, ArtifactMetadata metadata)
    : name_(std::move(name)),
      type_(type),
      content_(std::move(content)),
      metadata_(std::move(metadata)),
      currentVersion_(1) {
  if (id_.empty()) {
    id_ = "art_" + std::to_string(currentEpochMs());
  }
  if (metadata_.createdAt == 0) metadata_.createdAt = currentEpochMs();
  metadata_.updatedAt = metadata_.createdAt;

  // Snapshot initial version
  ArtifactVersion v1;
  v1.versionNumber = 1;
  v1.content = content_;
  v1.author = metadata_.author;
  v1.changeSummary = "Initial creation";
  v1.timestamp = metadata_.createdAt;
  history_.push_back(std::move(v1));
}

void Artifact::updateContent(std::string newContent, const std::string& author, const std::string& changeSummary) {
  ++currentVersion_;
  content_ = std::move(newContent);
  metadata_.updatedAt = currentEpochMs();
  if (!author.empty()) metadata_.author = author;

  ArtifactVersion ver;
  ver.versionNumber = currentVersion_;
  ver.content = content_;
  ver.author = author.empty() ? metadata_.author : author;
  ver.changeSummary = changeSummary.empty() ? ("Updated to v" + std::to_string(currentVersion_)) : changeSummary;
  ver.timestamp = metadata_.updatedAt;
  history_.push_back(std::move(ver));
}

std::optional<std::string> Artifact::getContentAtVersion(int version) const {
  if (version == currentVersion_) return content_;
  for (const auto& v : history_) {
    if (v.versionNumber == version) return v.content;
  }
  return std::nullopt;
}

ArtifactDiff Artifact::diff(int fromVersion, int toVersion) const {
  if (toVersion <= 0) toVersion = currentVersion_;

  auto contentFromOpt = getContentAtVersion(fromVersion);
  auto contentToOpt = getContentAtVersion(toVersion);

  std::string fromStr = contentFromOpt.value_or("");
  std::string toStr = contentToOpt.value_or("");

  auto fromLines = splitLines(fromStr);
  auto toLines = splitLines(toStr);

  ArtifactDiff result;
  result.fromVersion = fromVersion;
  result.toVersion = toVersion;

  std::size_t i = 0, j = 0;
  while (i < fromLines.size() || j < toLines.size()) {
    if (i < fromLines.size() && j < toLines.size() && fromLines[i] == toLines[j]) {
      ArtifactDiffLine line;
      line.type = ArtifactDiffLine::Type::Context;
      line.oldLineNumber = static_cast<int>(i + 1);
      line.newLineNumber = static_cast<int>(j + 1);
      line.line = fromLines[i];
      result.lines.push_back(std::move(line));
      ++i;
      ++j;
    } else {
      if (i < fromLines.size()) {
        ArtifactDiffLine line;
        line.type = ArtifactDiffLine::Type::Removed;
        line.oldLineNumber = static_cast<int>(i + 1);
        line.line = fromLines[i];
        result.lines.push_back(std::move(line));
        ++result.deletions;
        ++i;
      }
      if (j < toLines.size()) {
        ArtifactDiffLine line;
        line.type = ArtifactDiffLine::Type::Added;
        line.newLineNumber = static_cast<int>(j + 1);
        line.line = toLines[j];
        result.lines.push_back(std::move(line));
        ++result.additions;
        ++j;
      }
    }
  }

  return result;
}

api::json::Value Artifact::toJson() const {
  auto v = api::json::Value::object();
  v["id"] = id_;
  v["name"] = name_;
  v["type"] = std::string(artifactTypeName(type_));
  v["content"] = content_;
  v["current_version"] = currentVersion_;
  v["metadata"] = metadata_.toJson();

  auto hist = api::json::Value::array();
  for (const auto& ver : history_) {
    hist.push(ver.toJson());
  }
  v["history"] = std::move(hist);
  return v;
}

bool Artifact::fromJson(const api::json::Value& v, Artifact& out, std::string& error) {
  if (!v.isObject()) {
    error = "Artifact expects a JSON object";
    return false;
  }
  if (const auto* i = v.get("id"); i && i->isString()) out.id_ = i->asString();
  if (const auto* n = v.get("name"); n && n->isString()) out.name_ = n->asString();
  else { error = "Missing artifact name"; return false; }
  if (const auto* t = v.get("type"); t && t->isString()) out.type_ = parseArtifactType(t->asString());
  if (const auto* c = v.get("content"); c && c->isString()) out.content_ = c->asString();
  if (const auto* cv = v.get("current_version"); cv && cv->isNumber()) out.currentVersion_ = cv->asInt();

  if (const auto* m = v.get("metadata"); m) {
    ArtifactMetadata::fromJson(*m, out.metadata_, error);
  }

  out.history_.clear();
  if (const auto* h = v.get("history"); h && h->isArray()) {
    for (const auto& item : h->items()) {
      ArtifactVersion ver;
      std::string err;
      if (ArtifactVersion::fromJson(item, ver, err)) {
        out.history_.push_back(std::move(ver));
      }
    }
  }

  return true;
}

std::string Artifact::toMarkdown() const {
  std::ostringstream ss;
  ss << "# Artifact: " << name_ << " (v" << currentVersion_ << ")\n\n"
     << "> **Type**: " << artifactTypeName(type_) << " | **Author**: " << metadata_.author
     << " | **Created**: " << metadata_.createdAt << "\n\n";

  if (!metadata_.summary.empty()) {
    ss << "### Summary\n" << metadata_.summary << "\n\n";
  }

  ss << "### Content\n";
  if (type_ == ArtifactType::Code) {
    ss << "```" << (metadata_.language.empty() ? "text" : metadata_.language) << "\n"
       << content_ << "\n```\n";
  } else {
    ss << content_ << "\n";
  }

  if (history_.size() > 1) {
    ss << "\n### Revision History\n";
    for (const auto& ver : history_) {
      ss << "- **v" << ver.versionNumber << "** (" << ver.author << "): " << ver.changeSummary << "\n";
    }
  }
  return ss.str();
}

}  // namespace qorvix::agents
