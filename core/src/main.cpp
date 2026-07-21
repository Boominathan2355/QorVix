#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "qorvix/model_registry.hpp"
#include "qorvix/plugin_registry.hpp"
#include "qorvix/version.hpp"

namespace {

// Minimal, dependency-free command dispatch for Phase 1. When the CLI grows real flags/options
// (later phases), migrate to CLI11 — declared in vcpkg.json for exactly that.

std::string humanSize(std::uintmax_t bytes) {
  constexpr const char* units[] = {"B", "KB", "MB", "GB", "TB"};
  double value = static_cast<double>(bytes);
  int unit = 0;
  while (value >= 1024.0 && unit < 4) {
    value /= 1024.0;
    ++unit;
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(value < 10 && unit > 0 ? 1 : 0) << value << ' '
      << units[unit];
  return out.str();
}

void printModels(const std::vector<qorvix::ModelInfo>& models) {
  if (models.empty()) {
    std::cout << "  (none)\n";
    return;
  }
  std::size_t nameWidth = 4;
  for (const auto& m : models) nameWidth = std::max(nameWidth, m.name.size());
  for (const auto& m : models) {
    std::cout << "  " << std::left << std::setw(static_cast<int>(nameWidth)) << m.name << "  "
              << std::setw(6) << m.format << "  " << std::right << std::setw(10)
              << humanSize(m.sizeBytes) << "  " << m.path.string() << '\n';
  }
}

int cmdScanModels(const std::string& dir) {
  std::cout << "Scanning '" << dir << "' for models...\n";
  qorvix::ModelRegistry registry;
  const auto& models = registry.scan(dir);
  printModels(models);
  std::cout << "Found " << models.size() << " model" << (models.size() == 1 ? "" : "s") << ".\n";
  return 0;
}

int cmdList(const std::string& dir) {
  qorvix::ModelRegistry registry;
  const auto& models = registry.scan(dir);
  std::cout << "Models in '" << dir << "':\n";
  printModels(models);
  return 0;
}

int cmdPlugins(const std::string& dir) {
  std::cout << "Scanning '" << dir << "' for plugins...\n";
  qorvix::PluginRegistry registry;
  const auto loaded = registry.scan(dir);
  if (loaded.empty()) {
    std::cout << "  (none)\n";
    if (!registry.lastError().empty()) std::cout << "  note: " << registry.lastError() << '\n';
  } else {
    for (const auto& arch : loaded) std::cout << "  " << arch << '\n';
  }
  std::cout << "Loaded " << loaded.size() << " plugin" << (loaded.size() == 1 ? "" : "s") << ".\n";
  return 0;
}

int printUsage() {
  std::cout << qorvix::startupBanner() << "\n\n"
            << "Usage: qorvix <command> [args]\n\n"
            << "Commands:\n"
            << "  scan-models [dir]   Scan a directory for model files (default: models)\n"
            << "  list [dir]          List discovered models (default: models)\n"
            << "  plugins [dir]       Load and list architecture plugins in a directory\n"
            << "  version             Print the version\n"
            << "  help                Show this help\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string_view> args(argv + 1, argv + argc);
  if (args.empty()) return printUsage();

  const std::string_view command = args[0];
  const std::string arg1 = args.size() > 1 ? std::string(args[1]) : std::string();

  if (command == "help" || command == "-h" || command == "--help") return printUsage();
  if (command == "version" || command == "--version") {
    std::cout << qorvix::kVersionString << '\n';
    return 0;
  }
  if (command == "scan-models") return cmdScanModels(arg1.empty() ? "models" : arg1);
  if (command == "list") return cmdList(arg1.empty() ? "models" : arg1);
  if (command == "plugins") return cmdPlugins(arg1.empty() ? "plugins" : arg1);

  std::cerr << "Unknown command: " << command << "\n\n";
  printUsage();
  return 1;
}
