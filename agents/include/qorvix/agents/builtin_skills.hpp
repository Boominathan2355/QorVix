#pragma once

#include <filesystem>
#include <memory>

#include "qorvix/agents/skill.hpp"
#include "qorvix/agents/skill_registry.hpp"

namespace qorvix::agents {

// Built-in standard skill definitions
SkillDefinition createCodeReviewSkill();
SkillDefinition createRagSearchSkill();
SkillDefinition createMathSolverSkill();
SkillDefinition createCppRefactorSkill();
SkillDefinition createGitWorkflowSkill();

// Populates default built-in skills and optionally scans a directory for disk packages
void registerDefaultSkills(SkillRegistry& registry, const std::filesystem::path& skillsDir = "");

}  // namespace qorvix::agents
