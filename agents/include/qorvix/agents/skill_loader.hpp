#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "qorvix/agents/skill.hpp"
#include "qorvix/agents/skill_registry.hpp"

namespace qorvix::agents {

// File and directory scanner for discovering and loading Agent Skills
class SkillLoader {
 public:
  // Loads a single skill from a markdown or JSON file
  static bool loadFromFile(const std::filesystem::path& filePath,
                           SkillDefinition& out,
                           std::string& error);

  // Scans a directory (and subdirectories) for SKILL.md and *.skill.md files, registering them
  static std::size_t loadFromDirectory(const std::filesystem::path& dirPath,
                                       SkillRegistry& registry,
                                       std::string& error);

  // Saves a skill definition to disk as a SKILL.md package
  static bool saveToDirectory(const SkillDefinition& skill,
                              const std::filesystem::path& dirPath,
                              std::string& error);
};

}  // namespace qorvix::agents
