#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "qorvix/agents/skill.hpp"
#include "qorvix/agents/tool.hpp"
#include "qorvix/agents/tool_registry.hpp"

namespace qorvix::agents {

// Central catalog and manager for Agent Skills in QorVix
class SkillRegistry {
 public:
  SkillRegistry() = default;
  ~SkillRegistry() = default;

  // Non-copyable, movable
  SkillRegistry(const SkillRegistry&) = delete;
  SkillRegistry& operator=(const SkillRegistry&) = delete;
  SkillRegistry(SkillRegistry&&) noexcept;
  SkillRegistry& operator=(SkillRegistry&&) noexcept;

  // Registration & mutation
  bool registerSkill(SkillDefinition skill);
  bool unregisterSkill(const std::string& name);
  void clear();

  // Queries
  bool hasSkill(const std::string& name) const;
  std::optional<SkillDefinition> getSkill(const std::string& name) const;
  std::vector<std::string> skillNames() const;
  std::vector<SkillDefinition> listSkills() const;
  std::vector<SkillDefinition> searchSkills(const std::string& query = "",
                                           const std::string& tag = "",
                                           const std::string& category = "") const;

  // Prompt composition: combines active skill playbooks into markdown text for system prompts
  std::string composeSkillsPrompt(const std::vector<std::string>& activeSkills = {}) const;

  // Meta-tool bridge: wraps a skill into an executable ITool for explicit agent invocation
  std::shared_ptr<ITool> createSkillTool(const std::string& skillName) const;
  void registerAllAsTools(ToolRegistry& toolRegistry) const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, SkillDefinition> skills_;
};

}  // namespace qorvix::agents
