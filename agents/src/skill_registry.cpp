#include "qorvix/agents/skill_registry.hpp"

#include <algorithm>
#include <sstream>

namespace qorvix::agents {

SkillRegistry::SkillRegistry(SkillRegistry&& other) noexcept {
  std::lock_guard<std::mutex> lock(other.mutex_);
  skills_ = std::move(other.skills_);
}

SkillRegistry& SkillRegistry::operator=(SkillRegistry&& other) noexcept {
  if (this != &other) {
    std::scoped_lock lock(mutex_, other.mutex_);
    skills_ = std::move(other.skills_);
  }
  return *this;
}

bool SkillRegistry::registerSkill(SkillDefinition skill) {
  if (skill.name.empty()) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  skills_[skill.name] = std::move(skill);
  return true;
}

bool SkillRegistry::unregisterSkill(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  return skills_.erase(name) > 0;
}

void SkillRegistry::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  skills_.clear();
}

bool SkillRegistry::hasSkill(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return skills_.find(name) != skills_.end();
}

std::optional<SkillDefinition> SkillRegistry::getSkill(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = skills_.find(name);
  if (it != skills_.end()) return it->second;
  return std::nullopt;
}

std::vector<std::string> SkillRegistry::skillNames() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> names;
  names.reserve(skills_.size());
  for (const auto& [name, _] : skills_) names.push_back(name);
  std::sort(names.begin(), names.end());
  return names;
}

std::vector<SkillDefinition> SkillRegistry::listSkills() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<SkillDefinition> list;
  list.reserve(skills_.size());
  for (const auto& [_, s] : skills_) list.push_back(s);
  return list;
}

std::vector<SkillDefinition> SkillRegistry::searchSkills(const std::string& query,
                                                        const std::string& tag,
                                                        const std::string& category) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<SkillDefinition> results;

  for (const auto& [_, s] : skills_) {
    if (!category.empty() && s.category != category) continue;
    if (!tag.empty() && std::find(s.tags.begin(), s.tags.end(), tag) == s.tags.end()) continue;
    if (!query.empty()) {
      bool nameMatch = s.name.find(query) != std::string::npos;
      bool descMatch = s.description.find(query) != std::string::npos;
      bool instMatch = s.instructions.find(query) != std::string::npos;
      if (!nameMatch && !descMatch && !instMatch) continue;
    }
    results.push_back(s);
  }
  return results;
}

std::string SkillRegistry::composeSkillsPrompt(const std::vector<std::string>& activeSkills) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (skills_.empty()) return "";

  std::ostringstream ss;
  ss << "\n## ACTIVATED AGENT SKILLS & DOMAIN PLAYBOOKS\n"
     << "You are equipped with specialized skill playbooks. Adhere strictly to the procedures defined below:\n\n";

  int count = 0;
  for (const auto& [name, skill] : skills_) {
    if (!activeSkills.empty() && std::find(activeSkills.begin(), activeSkills.end(), name) == activeSkills.end()) {
      continue;
    }
    ++count;
    ss << skill.toPromptSection() << "\n---\n\n";
  }

  if (count == 0) return "";
  return ss.str();
}

std::shared_ptr<ITool> SkillRegistry::createSkillTool(const std::string& skillName) const {
  auto skillOpt = getSkill(skillName);
  if (!skillOpt.has_value()) return nullptr;

  const auto& skill = *skillOpt;
  ToolDefinition def;
  def.name = "skill_" + skill.name;
  def.description = "Execute the specialized playbook: " + skill.description;
  def.category = "skill";

  for (const auto& p : skill.parameters) {
    ToolParameter tp;
    tp.name = p.name;
    tp.type = parseToolParamType(p.type);
    tp.description = p.description;
    tp.required = p.required;
    def.parameters.push_back(std::move(tp));
  }

  return std::make_shared<FunctionTool>(std::move(def), [s = skill](const api::json::Value& args) -> ToolResult {
    auto data = api::json::Value::object();
    data["skill"] = s.name;
    data["version"] = s.version;
    data["instructions"] = s.instructions;
    data["arguments"] = args;
    std::string out = "Activating playbook for skill [" + s.name + "].\n" + s.instructions;
    return ToolResult::makeSuccess(out, data);
  });
}

void SkillRegistry::registerAllAsTools(ToolRegistry& toolRegistry) const {
  auto names = skillNames();
  for (const auto& name : names) {
    auto tool = createSkillTool(name);
    if (tool) {
      toolRegistry.registerTool(tool);
    }
  }
}

}  // namespace qorvix::agents
