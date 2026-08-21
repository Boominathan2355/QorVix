#include "qorvix/agents/artifact_tools.hpp"

#include <sstream>

namespace qorvix::agents {

std::shared_ptr<ITool> createArtifactCreateTool(std::shared_ptr<ArtifactStore> store) {
  ToolDefinition def;
  def.name = "artifact_create";
  def.description = "Creates a persistent digital artifact (e.g. code file, markdown report, data matrix, diff).";
  def.category = "artifacts";
  def.parameters = {
      {"name", ToolParamType::String, "Unique identifier/name for the artifact (e.g. 'auth_middleware')", true, {}},
      {"content", ToolParamType::String, "The full content of the artifact", true, {}},
      {"type", ToolParamType::String, "Type: 'code', 'markdown', 'document', 'diff', 'json', 'table'", false, {"code", "markdown", "document", "diff", "json", "table"}},
      {"language", ToolParamType::String, "Language (for code artifacts, e.g. 'cpp', 'typescript', 'python')", false, {}},
      {"summary", ToolParamType::String, "Brief summary of what this artifact contains", false, {}},
      {"author", ToolParamType::String, "Author or agent name", false, {}}
  };

  return std::make_shared<FunctionTool>(std::move(def), [store](const api::json::Value& args) -> ToolResult {
    if (!store) return ToolResult::makeError("ArtifactStore is uninitialized");

    const auto* name = args.get("name");
    const auto* content = args.get("content");
    if (!name || !name->isString() || !content || !content->isString()) {
      return ToolResult::makeError("Missing required parameters 'name' or 'content'");
    }

    ArtifactType type = ArtifactType::Document;
    if (const auto* t = args.get("type"); t && t->isString()) {
      type = parseArtifactType(t->asString());
    }

    ArtifactMetadata meta;
    if (const auto* l = args.get("language"); l && l->isString()) meta.language = l->asString();
    if (const auto* s = args.get("summary"); s && s->isString()) meta.summary = s->asString();
    if (const auto* a = args.get("author"); a && a->isString()) meta.author = a->asString();

    Artifact art(name->asString(), type, content->asString(), meta);
    if (!store->createArtifact(std::move(art))) {
      return ToolResult::makeError("Failed to create artifact: " + name->asString());
    }

    auto data = api::json::Value::object();
    data["name"] = name->asString();
    data["version"] = 1;
    return ToolResult::makeSuccess("Successfully created artifact '" + name->asString() + "' (v1)", data);
  });
}

std::shared_ptr<ITool> createArtifactReadTool(std::shared_ptr<ArtifactStore> store) {
  ToolDefinition def;
  def.name = "artifact_read";
  def.description = "Retrieves the content, summary, and metadata of a created artifact (optionally at a specific version).";
  def.category = "artifacts";
  def.parameters = {
      {"name", ToolParamType::String, "Name of the artifact to read", true, {}},
      {"version", ToolParamType::Integer, "Specific historical version to read (default is latest)", false, {}}
  };

  return std::make_shared<FunctionTool>(std::move(def), [store](const api::json::Value& args) -> ToolResult {
    if (!store) return ToolResult::makeError("ArtifactStore is uninitialized");

    const auto* name = args.get("name");
    if (!name || !name->isString()) return ToolResult::makeError("Missing required parameter 'name'");

    auto artOpt = store->getArtifact(name->asString());
    if (!artOpt.has_value()) {
      return ToolResult::makeError("Artifact not found: " + name->asString());
    }

    const auto& art = *artOpt;
    int version = art.currentVersion();
    if (const auto* v = args.get("version"); v && v->isNumber()) {
      version = v->asInt();
    }

    auto contentOpt = art.getContentAtVersion(version);
    if (!contentOpt.has_value()) {
      return ToolResult::makeError("Version " + std::to_string(version) + " not found for artifact " + name->asString());
    }

    auto data = art.toJson();
    return ToolResult::makeSuccess(*contentOpt, data);
  });
}

std::shared_ptr<ITool> createArtifactUpdateTool(std::shared_ptr<ArtifactStore> store) {
  ToolDefinition def;
  def.name = "artifact_update";
  def.description = "Updates an existing artifact with new content, incrementing its revision version and recording change history.";
  def.category = "artifacts";
  def.parameters = {
      {"name", ToolParamType::String, "Name of the artifact to update", true, {}},
      {"content", ToolParamType::String, "The new updated content", true, {}},
      {"change_summary", ToolParamType::String, "Description of what was changed and why", false, {}},
      {"author", ToolParamType::String, "Author or agent making the update", false, {}}
  };

  return std::make_shared<FunctionTool>(std::move(def), [store](const api::json::Value& args) -> ToolResult {
    if (!store) return ToolResult::makeError("ArtifactStore is uninitialized");

    const auto* name = args.get("name");
    const auto* content = args.get("content");
    if (!name || !name->isString() || !content || !content->isString()) {
      return ToolResult::makeError("Missing required parameters 'name' or 'content'");
    }

    std::string summary;
    if (const auto* cs = args.get("change_summary"); cs && cs->isString()) summary = cs->asString();

    std::string author;
    if (const auto* a = args.get("author"); a && a->isString()) author = a->asString();

    if (!store->updateArtifact(name->asString(), content->asString(), author, summary)) {
      return ToolResult::makeError("Artifact not found to update: " + name->asString());
    }

    auto updated = store->getArtifact(name->asString());
    int ver = updated.has_value() ? updated->currentVersion() : 1;

    auto data = api::json::Value::object();
    data["name"] = name->asString();
    data["version"] = ver;
    return ToolResult::makeSuccess("Successfully updated artifact '" + name->asString() + "' to v" + std::to_string(ver), data);
  });
}

std::shared_ptr<ITool> createArtifactListTool(std::shared_ptr<ArtifactStore> store) {
  ToolDefinition def;
  def.name = "artifact_list";
  def.description = "Lists all stored artifacts, their versions, types, and summaries.";
  def.category = "artifacts";

  return std::make_shared<FunctionTool>(std::move(def), [store](const api::json::Value&) -> ToolResult {
    if (!store) return ToolResult::makeError("ArtifactStore is uninitialized");

    auto list = store->listArtifacts();
    if (list.empty()) {
      return ToolResult::makeSuccess("No artifacts stored.");
    }

    std::ostringstream ss;
    ss << "Stored Artifacts (" << list.size() << "):\n";
    for (const auto& a : list) {
      ss << "- **" << a.name() << "** [v" << a.currentVersion() << " | " << artifactTypeName(a.type()) << "]: "
         << (a.metadata().summary.empty() ? "(no summary)" : a.metadata().summary) << "\n";
    }

    return ToolResult::makeSuccess(ss.str(), store->toJson());
  });
}

std::shared_ptr<ITool> createArtifactDiffTool(std::shared_ptr<ArtifactStore> store) {
  ToolDefinition def;
  def.name = "artifact_diff";
  def.description = "Computes unified line-by-line diff between two versions of an artifact.";
  def.category = "artifacts";
  def.parameters = {
      {"name", ToolParamType::String, "Name of the artifact", true, {}},
      {"from_version", ToolParamType::Integer, "Starting version number", true, {}},
      {"to_version", ToolParamType::Integer, "Target version number (optional, default latest)", false, {}}
  };

  return std::make_shared<FunctionTool>(std::move(def), [store](const api::json::Value& args) -> ToolResult {
    if (!store) return ToolResult::makeError("ArtifactStore is uninitialized");

    const auto* name = args.get("name");
    const auto* fromV = args.get("from_version");
    if (!name || !name->isString() || !fromV || !fromV->isNumber()) {
      return ToolResult::makeError("Missing required parameters 'name' or 'from_version'");
    }

    auto artOpt = store->getArtifact(name->asString());
    if (!artOpt.has_value()) {
      return ToolResult::makeError("Artifact not found: " + name->asString());
    }

    int toV = -1;
    if (const auto* tv = args.get("to_version"); tv && tv->isNumber()) {
      toV = tv->asInt();
    }

    ArtifactDiff diff = artOpt->diff(fromV->asInt(), toV);
    return ToolResult::makeSuccess(diff.toString(), diff.toJson());
  });
}

void registerArtifactTools(ToolRegistry& registry, std::shared_ptr<ArtifactStore> store) {
  if (!store) return;
  registry.registerTool(createArtifactCreateTool(store));
  registry.registerTool(createArtifactReadTool(store));
  registry.registerTool(createArtifactUpdateTool(store));
  registry.registerTool(createArtifactListTool(store));
  registry.registerTool(createArtifactDiffTool(store));
}

}  // namespace qorvix::agents
