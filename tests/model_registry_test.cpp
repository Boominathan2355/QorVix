#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>

#include "qorvix/model_registry.hpp"

namespace fs = std::filesystem;

namespace {

fs::path makeCleanDir(const std::string& name) {
  fs::path dir = fs::temp_directory_path() / ("qorvix_test_" + name);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

void writeFile(const fs::path& path, std::size_t bytes) {
  std::ofstream out(path, std::ios::binary);
  out << std::string(bytes, 'x');
}

}  // namespace

TEST_CASE("ModelRegistry discovers gguf files and ignores others", "[model_registry]") {
  const fs::path dir = makeCleanDir("registry_basic");
  writeFile(dir / "llama3.gguf", 10);
  writeFile(dir / "qwen3.GGUF", 20);  // case-insensitive extension
  writeFile(dir / "notes.txt", 5);
  writeFile(dir / "readme.md", 5);

  qorvix::ModelRegistry registry;
  const auto& models = registry.scan(dir);

  REQUIRE(models.size() == 2);
  // Sorted by name.
  REQUIRE(models[0].name == "llama3");
  REQUIRE(models[0].format == "gguf");
  REQUIRE(models[0].sizeBytes == 10);
  REQUIRE(models[1].name == "qwen3");

  fs::remove_all(dir);
}

TEST_CASE("ModelRegistry on a missing directory yields no models", "[model_registry]") {
  qorvix::ModelRegistry registry;
  const auto& models = registry.scan(fs::temp_directory_path() / "qorvix_does_not_exist_xyz");
  REQUIRE(models.empty());
}

TEST_CASE("isModelFile matches known extensions case-insensitively", "[model_registry]") {
  REQUIRE(qorvix::isModelFile("a/b/c.gguf"));
  REQUIRE(qorvix::isModelFile("MODEL.GGUF"));
  REQUIRE_FALSE(qorvix::isModelFile("model.safetensors"));
  REQUIRE_FALSE(qorvix::isModelFile("model"));
}
