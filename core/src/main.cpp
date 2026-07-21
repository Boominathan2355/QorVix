#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "qorvix/cuda/backend.hpp"
#include "qorvix/gguf/gguf_file.hpp"
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

std::string metaValuePreview(const qorvix::gguf::GgufValue& v) {
  using qorvix::gguf::MetaType;
  if (v.isArray()) {
    std::ostringstream out;
    out << "[" << qorvix::gguf::metaTypeName(v.arrayElementType()) << " x " << v.array().size()
        << "]";
    return out.str();
  }
  if (const std::string* s = v.asString()) {
    std::string preview = *s;
    if (preview.size() > 48) preview = preview.substr(0, 45) + "...";
    return preview;
  }
  if (auto b = v.asBool()) return *b ? "true" : "false";
  if (auto i = v.asI64()) return std::to_string(*i);
  if (auto f = v.asF64()) {
    std::ostringstream out;
    out << *f;
    return out.str();
  }
  return "?";
}

int cmdGgufInfo(const std::string& path) {
  if (path.empty()) {
    std::cerr << "usage: qorvix gguf-info <file.gguf>\n";
    return 1;
  }
  try {
    const auto file = qorvix::gguf::GgufFile::open(path);
    const auto& h = file.header();
    std::cout << "File:         " << path << "\n"
              << "GGUF version: " << h.version << "\n"
              << "Architecture: " << (file.architecture().empty() ? "(unknown)" : file.architecture())
              << "\n";
    if (auto name = file.name()) std::cout << "Name:         " << *name << "\n";
    if (auto ft = file.fileType()) std::cout << "File type:    " << *ft << "\n";
    std::cout << "Alignment:    " << file.alignment() << "\n"
              << "Data offset:  " << file.dataOffset() << "\n"
              << "Metadata KVs: " << h.metadataCount << "\n"
              << "Tensors:      " << h.tensorCount << "\n";

    const auto rope = file.rope();
    if (rope.dimensionCount || rope.freqBase || rope.scalingType) {
      std::cout << "RoPE:         ";
      if (rope.dimensionCount) std::cout << "dim=" << *rope.dimensionCount << " ";
      if (rope.freqBase) std::cout << "freq_base=" << *rope.freqBase << " ";
      if (rope.scalingType) std::cout << "scaling=" << *rope.scalingType << " ";
      std::cout << "\n";
    }

    std::cout << "\nMetadata:\n";
    for (const auto& [key, value] : file.metadata()) {
      std::cout << "  " << std::left << std::setw(40) << key << "  " << metaValuePreview(value)
                << "\n";
    }

    std::cout << "\nTensors:\n";
    for (const auto& t : file.tensors()) {
      std::ostringstream dims;
      dims << "[";
      for (std::size_t i = 0; i < t.dimensions.size(); ++i) {
        dims << (i ? "," : "") << t.dimensions[i];
      }
      dims << "]";
      std::cout << "  " << std::left << std::setw(40) << t.name << "  " << std::setw(8)
                << t.typeName() << "  " << std::setw(18) << dims.str() << "  " << t.nBytes
                << " bytes\n";
    }
    return 0;
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

std::string humanBytes(std::size_t bytes) {
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

int cmdGpu() {
  if (!qorvix::cuda::builtWithCuda()) {
    std::cout << "CUDA support: not built in.\n"
              << "Rebuild with -DQORVIX_ENABLE_CUDA=ON (needs a CUDA 12+ toolkit) to enable the "
                 "GPU backend.\n";
    return 0;
  }

  const int count = qorvix::cuda::deviceCount();
  std::cout << "CUDA support: built in.\n" << "Devices: " << count << "\n";
  if (count == 0) {
    std::cout << "No CUDA devices detected on this host.\n";
    return 0;
  }

  for (const auto& d : qorvix::cuda::enumerateDevices()) {
    std::cout << "\n  [" << d.index << "] " << d.name << "\n"
              << "      compute capability : " << d.computeMajor << "." << d.computeMinor << "\n"
              << "      SMs                : " << d.multiProcessorCount << "\n"
              << "      memory (free/total): " << humanBytes(d.freeMem) << " / "
              << humanBytes(d.totalGlobalMem) << "\n";
  }

  const auto self = qorvix::cuda::selfTest();
  std::cout << "\nSelf-test (scale kernel): " << (self.passed ? "PASS" : (self.ran ? "FAIL" : "skip"))
            << " — " << self.message << "\n";
  const auto gemm = qorvix::cuda::gemmSelfTest();
  std::cout << "Self-test (cuBLAS GEMM):  " << (gemm.passed ? "PASS" : (gemm.ran ? "FAIL" : "skip"))
            << " — " << gemm.message << "\n";
  return (self.ran && !self.passed) || (gemm.ran && !gemm.passed) ? 1 : 0;
}

int printUsage() {
  std::cout << qorvix::startupBanner() << "\n\n"
            << "Usage: qorvix <command> [args]\n\n"
            << "Commands:\n"
            << "  scan-models [dir]   Scan a directory for model files (default: models)\n"
            << "  list [dir]          List discovered models (default: models)\n"
            << "  gguf-info <file>    Parse a GGUF file and print its header, metadata, tensors\n"
            << "  gpu                 Show CUDA devices and run backend self-tests\n"
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
  if (command == "gguf-info") return cmdGgufInfo(arg1);
  if (command == "gpu") return cmdGpu();
  if (command == "plugins") return cmdPlugins(arg1.empty() ? "plugins" : arg1);

  std::cerr << "Unknown command: " << command << "\n\n";
  printUsage();
  return 1;
}
