#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "qorvix/api/json.hpp"

namespace qorvix::agents {

struct SkillParameter {
  std::string name;
  std::string type = "string";  // "string", "number", "boolean", "array", "object"
  std::string description;
  bool required = false;
  std::string defaultValue;

  api::json::Value toJson() const;
  static bool fromJson(const api::json::Value& v, SkillParameter& out, std::string& error);
};

struct SkillExample {
  std::string title;
  std::string input;
  std::string output;
  std::string explanation;

  api::json::Value toJson() const;
  static bool fromJson(const api::json::Value& v, SkillExample& out, std::string& error);
};

struct SkillDefinition {
  std::string name;
  std::string version = "1.0.0";
  std::string description;
  std::string category = "general";  // "coding", "research", "math", "workflow", "system"
  std::vector<std::string> tags;
  std::string author = "QorVix";

  std::string instructions;                  // Core procedural playbook / markdown prompt
  std::vector<std::string> requiredTools;    // Tools this skill utilizes (e.g. "calculator", "file_ops")
  std::vector<SkillParameter> parameters;    // Input parameters if invoked directly
  std::vector<SkillExample> examples;        // Few-shot demonstrations
  std::string sourcePath;                    // Filesystem path if loaded from disk

  // Formats instructions and metadata into prompt text for agent injection
  std::string toPromptSection() const;

  // Markdown (SKILL.md with frontmatter) format
  std::string toMarkdown() const;
  static bool fromMarkdown(std::string_view markdown, SkillDefinition& out, std::string& error);

  // JSON format
  api::json::Value toJson() const;
  static bool fromJson(const api::json::Value& v, SkillDefinition& out, std::string& error);
};

}  // namespace qorvix::agents
