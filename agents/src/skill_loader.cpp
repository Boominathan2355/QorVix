#include "qorvix/agents/skill_loader.hpp"

#include <fstream>
#include <sstream>

namespace qorvix::agents {

bool SkillLoader::loadFromFile(const std::filesystem::path& filePath,
                               SkillDefinition& out,
                               std::string& error) {
  if (!std::filesystem::exists(filePath)) {
    error = "Skill file does not exist: " + filePath.string();
    return false;
  }
  if (std::filesystem::is_directory(filePath)) {
    error = "Path is a directory: " + filePath.string();
    return false;
  }

  std::ifstream fin(filePath, std::ios::binary);
  if (!fin) {
    error = "Failed to open skill file: " + filePath.string();
    return false;
  }

  std::ostringstream ss;
  ss << fin.rdbuf();
  std::string content = ss.str();

  // If JSON format
  if (filePath.extension() == ".json") {
    auto parsed = api::json::parse(content);
    if (!parsed.has_value()) {
      error = "Invalid JSON in skill file: " + filePath.string();
      return false;
    }
    if (!SkillDefinition::fromJson(*parsed, out, error)) {
      return false;
    }
  } else {
    // Markdown format (SKILL.md or *.md)
    if (!SkillDefinition::fromMarkdown(content, out, error)) {
      return false;
    }
  }

  out.sourcePath = filePath.string();
  if (out.name == "unnamed_skill" || out.name.empty()) {
    out.name = filePath.stem().string();
    if (out.name == "SKILL" || out.name == "skill") {
      out.name = filePath.parent_path().filename().string();
    }
  }
  return true;
}

std::size_t SkillLoader::loadFromDirectory(const std::filesystem::path& dirPath,
                                           SkillRegistry& registry,
                                           std::string& error) {
  if (!std::filesystem::exists(dirPath)) {
    error = "Directory does not exist: " + dirPath.string();
    return 0;
  }
  if (!std::filesystem::is_directory(dirPath)) {
    error = "Path is not a directory: " + dirPath.string();
    return 0;
  }

  std::size_t count = 0;
  try {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(dirPath)) {
      if (!entry.is_regular_file()) continue;

      const auto filename = entry.path().filename().string();
      if (filename == "SKILL.md" || filename == "skill.md" ||
          entry.path().extension() == ".skill" || entry.path().extension() == ".skill.md") {
        SkillDefinition skill;
        std::string err;
        if (loadFromFile(entry.path(), skill, err)) {
          if (registry.registerSkill(std::move(skill))) {
            ++count;
          }
        }
      }
    }
  } catch (const std::exception& ex) {
    error = std::string("Filesystem iteration error: ") + ex.what();
  }

  return count;
}

bool SkillLoader::saveToDirectory(const SkillDefinition& skill,
                                  const std::filesystem::path& dirPath,
                                  std::string& error) {
  try {
    std::filesystem::path targetDir = dirPath / skill.name;
    std::filesystem::create_directories(targetDir);

    std::filesystem::path targetFile = targetDir / "SKILL.md";
    std::ofstream fout(targetFile, std::ios::binary | std::ios::trunc);
    if (!fout) {
      error = "Failed to open target file for writing: " + targetFile.string();
      return false;
    }
    fout << skill.toMarkdown();
    return true;
  } catch (const std::exception& ex) {
    error = std::string("Failed to save skill: ") + ex.what();
    return false;
  }
}

}  // namespace qorvix::agents
