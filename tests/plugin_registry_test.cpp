#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include "qorvix/plugin_registry.hpp"

#ifndef QORVIX_EXAMPLE_PLUGIN_PATH
#error "QORVIX_EXAMPLE_PLUGIN_PATH must be defined by the build (see tests/CMakeLists.txt)"
#endif

TEST_CASE("PluginRegistry hot-loads and unloads the example plugin", "[plugin_registry]") {
  const std::filesystem::path pluginPath = QORVIX_EXAMPLE_PLUGIN_PATH;
  REQUIRE(std::filesystem::exists(pluginPath));

  qorvix::PluginRegistry registry;

  auto arch = registry.load(pluginPath);
  REQUIRE(arch.has_value());
  REQUIRE(*arch == "example");
  REQUIRE(registry.contains("example"));

  qorvix::IPlugin* plugin = registry.get("example");
  REQUIRE(plugin != nullptr);
  REQUIRE(plugin->architecture() == "example");
  REQUIRE(plugin->load());
  REQUIRE(plugin->infer());  // true only after load()
  REQUIRE(plugin->unload());
  REQUIRE_FALSE(plugin->infer());

  // Loading the same architecture twice is rejected.
  REQUIRE_FALSE(registry.load(pluginPath).has_value());

  REQUIRE(registry.unload("example"));
  REQUIRE_FALSE(registry.contains("example"));
  REQUIRE(registry.get("example") == nullptr);
}

TEST_CASE("PluginRegistry::scan loads plugins from a directory", "[plugin_registry]") {
  const std::filesystem::path pluginDir =
      std::filesystem::path(QORVIX_EXAMPLE_PLUGIN_PATH).parent_path();

  qorvix::PluginRegistry registry;
  const auto loaded = registry.scan(pluginDir);
  REQUIRE_FALSE(loaded.empty());
  REQUIRE(registry.contains("example"));
}

TEST_CASE("PluginRegistry reports an error for a missing library", "[plugin_registry]") {
  qorvix::PluginRegistry registry;
  auto arch = registry.load("definitely_not_a_real_plugin_qorvix.so");
  REQUIRE_FALSE(arch.has_value());
  REQUIRE_FALSE(registry.lastError().empty());
}
