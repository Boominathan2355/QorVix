#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "qorvix/api/http_server.hpp"
#include "qorvix/api/openai.hpp"
#include "qorvix/cuda/backend.hpp"
#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/model_registry.hpp"
#include "qorvix/plugin_registry.hpp"
#include "qorvix/runtime/generator.hpp"
#include "qorvix/runtime/model_config.hpp"
#include "qorvix/runtime/text_model.hpp"
#include "qorvix/scheduler/scheduler.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"
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

int cmdModelInfo(const std::string& path) {
  if (path.empty()) {
    std::cerr << "usage: qorvix model-info <file.gguf>\n";
    return 1;
  }
  try {
    const auto file = qorvix::gguf::GgufFile::open(path);
    std::string err;
    const auto cfg = qorvix::runtime::configFromGguf(file, err);
    if (!cfg.valid()) {
      std::cerr << "error: " << (err.empty() ? "could not derive model config" : err) << "\n";
      return 1;
    }
    std::cout << "Architecture:      " << cfg.architecture << "\n"
              << "Vocab size:        " << cfg.vocabSize << "\n"
              << "Context length:    " << cfg.contextLength << "\n"
              << "Embedding (d_model): " << cfg.embeddingLength << "\n"
              << "Layers:            " << cfg.blockCount << "\n"
              << "FFN hidden:        " << cfg.feedForwardLength << "\n"
              << "Attention heads:   " << cfg.headCount << " (kv " << cfg.headCountKv << ", head_dim "
              << cfg.headDim() << ")\n"
              << "RoPE:              dim=" << cfg.ropeDimensionCount << " freq_base="
              << cfg.ropeFreqBase << "\n"
              << "RMSNorm eps:       " << cfg.normEpsilon << "\n";
    return 0;
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

// Parses `--flag value` style options after the positional model path. Returns "" if absent.
std::string flagValue(const std::vector<std::string_view>& args, std::string_view flag) {
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == flag) return std::string(args[i + 1]);
  }
  return {};
}

int cmdGenerate(const std::vector<std::string_view>& args) {
  const std::string path = args.size() > 1 ? std::string(args[1]) : std::string();
  const std::string prompt = flagValue(args, "--prompt");
  if (path.empty() || prompt.empty()) {
    std::cerr << "usage: qorvix generate <file.gguf> --prompt \"...\" "
                 "[--max N] [--temp T] [--top-k K] [--top-p P] [--seed S]\n";
    return 1;
  }

  qorvix::runtime::GenerationConfig cfg;
  if (auto v = flagValue(args, "--max"); !v.empty()) cfg.maxNewTokens = std::stoi(v);
  if (auto v = flagValue(args, "--temp"); !v.empty()) cfg.sampling.temperature = std::stof(v);
  if (auto v = flagValue(args, "--top-k"); !v.empty()) cfg.sampling.topK = std::stoi(v);
  if (auto v = flagValue(args, "--top-p"); !v.empty()) cfg.sampling.topP = std::stof(v);
  if (auto v = flagValue(args, "--seed"); !v.empty()) cfg.seed = std::stoull(v);

  try {
    using clock = std::chrono::steady_clock;
    const auto tLoad0 = clock::now();
    auto file = qorvix::gguf::GgufFile::open(path);
    std::string err;
    // Build the tokenizer first (it copies the vocab out of the file), then hand the file to the
    // model, which takes ownership so its quantized weights can keep aliasing the mmap.
    auto tok = qorvix::tokenizer::Tokenizer::fromGguf(file, err);
    if (!tok) {
      std::cerr << "error: tokenizer: " << err << "\n";
      return 1;
    }
    auto model = qorvix::runtime::TextModel::fromGguf(std::move(file), err);
    if (!model) {
      std::cerr << "error: model: " << err << "\n";
      return 1;
    }
    const double loadSec = std::chrono::duration<double>(clock::now() - tLoad0).count();

    qorvix::runtime::Generator gen(*model, *tok);
    std::cout << prompt << std::flush;
    const auto tGen0 = clock::now();
    auto result = gen.generate(prompt, cfg,
                               [](const std::string& piece) { std::cout << piece << std::flush; });
    const double genSec = std::chrono::duration<double>(clock::now() - tGen0).count();

    const int forwards = result.promptTokens + static_cast<int>(result.tokens.size());
    std::cout << "\n\n[" << result.promptTokens << " prompt tokens, " << result.tokens.size()
              << " generated" << (result.hitEos ? ", eos" : "") << "]\n";
    std::cout << "[load " << std::fixed << std::setprecision(1) << loadSec << "s | "
              << forwards << " forwards in " << genSec << "s = " << std::setprecision(2)
              << (genSec > 0 ? forwards / genSec : 0.0) << " tok/s]\n";
    return 0;
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

namespace {
qorvix::scheduler::RequestParams toRequestParams(const qorvix::api::SamplingRequest& s) {
  qorvix::scheduler::RequestParams rp;
  rp.maxNewTokens = s.maxTokens;
  rp.sampling.temperature = s.temperature;
  rp.sampling.topP = s.topP;
  rp.sampling.topK = s.topK;
  rp.sampling.minP = s.minP;
  rp.sampling.frequencyPenalty = s.frequencyPenalty;
  rp.sampling.presencePenalty = s.presencePenalty;
  rp.sampling.repetitionPenalty = s.repetitionPenalty;
  rp.seed = s.seed;
  rp.addBos = true;
  return rp;
}
}  // namespace

int cmdServe(const std::vector<std::string_view>& args) {
  namespace api = qorvix::api;
  const std::string path = args.size() > 1 ? std::string(args[1]) : std::string();
  if (path.empty()) {
    std::cerr << "usage: qorvix serve <file.gguf> [--port N] [--max-concurrent N] [--ctx N]\n";
    return 1;
  }
  int port = 8080, maxConcurrent = 4, ctx = 4096;
  if (auto v = flagValue(args, "--port"); !v.empty()) port = std::stoi(v);
  if (auto v = flagValue(args, "--max-concurrent"); !v.empty()) maxConcurrent = std::stoi(v);
  if (auto v = flagValue(args, "--ctx"); !v.empty()) ctx = std::stoi(v);

  std::string err;
  std::optional<qorvix::tokenizer::Tokenizer> tok;
  std::optional<qorvix::runtime::TextModel> model;
  try {
    auto file = qorvix::gguf::GgufFile::open(path);
    tok = qorvix::tokenizer::Tokenizer::fromGguf(file, err);
    if (!tok) {
      std::cerr << "error: tokenizer: " << err << "\n";
      return 1;
    }
    model = qorvix::runtime::TextModel::fromGguf(std::move(file), err,
                                                 static_cast<std::uint32_t>(ctx),
                                                 static_cast<std::uint32_t>(maxConcurrent));
    if (!model) {
      std::cerr << "error: model: " << err << "\n";
      return 1;
    }
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  const std::string modelId = model->config().architecture + "/" + path;
  qorvix::scheduler::Scheduler sched(*model, *tok, {maxConcurrent});
  long long idCounter = 0;

  api::HttpServer server(port);
  if (!server.start(err)) {
    std::cerr << "error: " << err << "\n";
    return 1;
  }
  std::cout << "qorvix serving " << path << " on http://0.0.0.0:" << port << "\n"
            << "  POST /v1/chat/completions   POST /v1/completions   GET /v1/models\n"
            << "  (Ctrl-C to stop)\n";

  auto handler = [&](const api::HttpRequest& req, api::HttpResponder& res) {
    if (req.method == "OPTIONS") {
      res.send(200, "text/plain", "");
      return;
    }
    if (req.method == "GET" && req.target == "/v1/models") {
      res.send(200, "application/json", api::modelsResponse({modelId}).dump());
      return;
    }
    if (req.method == "GET" && (req.target == "/" || req.target == "/health")) {
      res.send(200, "application/json", R"({"status":"ok","service":"qorvix"})");
      return;
    }

    const bool isChat = req.target == "/v1/chat/completions";
    const bool isCompletion = req.target == "/v1/completions";
    if (req.method != "POST" || (!isChat && !isCompletion)) {
      res.send(404, "application/json",
               api::errorResponse("unknown route " + req.method + " " + req.target, "not_found").dump());
      return;
    }

    std::string perr;
    auto body = api::json::parse(req.body, &perr);
    if (!body) {
      res.send(400, "application/json", api::errorResponse("invalid JSON: " + perr).dump());
      return;
    }

    std::string prompt, respModel = modelId;
    bool stream = false;
    qorvix::scheduler::RequestParams rp;
    if (isChat) {
      auto cr = api::parseChatRequest(*body, perr);
      if (!cr.valid) {
        res.send(400, "application/json", api::errorResponse(perr).dump());
        return;
      }
      prompt = api::buildChatPrompt(cr.messages);
      stream = cr.stream;
      rp = toRequestParams(cr.sampling);
      if (!cr.model.empty()) respModel = cr.model;
    } else {
      auto cr = api::parseCompletionRequest(*body, perr);
      if (!cr.valid) {
        res.send(400, "application/json", api::errorResponse(perr).dump());
        return;
      }
      prompt = cr.prompt;
      stream = cr.stream;
      rp = toRequestParams(cr.sampling);
      if (!cr.model.empty()) respModel = cr.model;
    }

    const std::string id = (isChat ? "chatcmpl-" : "cmpl-") + std::to_string(++idCounter);

    if (stream) {
      res.beginStream(200, "text/event-stream");
      if (isChat) res.writeChunk(api::sseData(api::chatChunk(id, respModel, "", true, "")));
      sched.submit(prompt, rp, [&](qorvix::scheduler::RequestId, const std::string& piece) {
        res.writeChunk(api::sseData(isChat ? api::chatChunk(id, respModel, piece, false, "")
                                           : api::completionChunk(id, respModel, piece, "")));
      });
      auto results = sched.runToCompletion();
      const bool eos = !results.empty() && results.front().hitEos;
      const std::string finish = eos ? "stop" : "length";
      res.writeChunk(api::sseData(isChat ? api::chatChunk(id, respModel, "", false, finish)
                                         : api::completionChunk(id, respModel, "", finish)));
      res.writeChunk(api::sseDone());
      res.endStream();
    } else {
      sched.submit(prompt, rp);
      auto results = sched.runToCompletion();
      if (results.empty()) {
        res.send(500, "application/json", api::errorResponse("generation produced no result", "server_error").dump());
        return;
      }
      const auto& r = results.front();
      const std::string finish = r.hitEos ? "stop" : "length";
      const int completion = static_cast<int>(r.tokens.size());
      auto json = isChat ? api::chatCompletion(id, respModel, r.text, r.promptTokens, completion, finish)
                         : api::completion(id, respModel, r.text, r.promptTokens, completion, finish);
      res.send(200, "application/json", json.dump());
    }
  };

  server.run(handler);
  return 0;
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
            << " - " << self.message << "\n";
  const auto gemm = qorvix::cuda::gemmSelfTest();
  std::cout << "Self-test (cuBLAS GEMM):  " << (gemm.passed ? "PASS" : (gemm.ran ? "FAIL" : "skip"))
            << " - " << gemm.message << "\n";
  const auto qmm = qorvix::cuda::qmatmulSelfTest();
  std::cout << "Self-test (Q8_0 matmul):  " << (qmm.passed ? "PASS" : (qmm.ran ? "FAIL" : "skip"))
            << " - " << qmm.message << "\n";
  const auto ops = qorvix::cuda::opsSelfTest();
  std::cout << "Self-test (forward ops):  " << (ops.passed ? "PASS" : (ops.ran ? "FAIL" : "skip"))
            << " - " << ops.message << "\n";
  const auto attn = qorvix::cuda::attentionSelfTest();
  std::cout << "Self-test (attention):    " << (attn.passed ? "PASS" : (attn.ran ? "FAIL" : "skip"))
            << " - " << attn.message << "\n";
  return (self.ran && !self.passed) || (gemm.ran && !gemm.passed) || (qmm.ran && !qmm.passed) ||
                 (ops.ran && !ops.passed) || (attn.ran && !attn.passed)
             ? 1
             : 0;
}

int printUsage() {
  std::cout << qorvix::startupBanner() << "\n\n"
            << "Usage: qorvix <command> [args]\n\n"
            << "Commands:\n"
            << "  scan-models [dir]   Scan a directory for model files (default: models)\n"
            << "  list [dir]          List discovered models (default: models)\n"
            << "  gguf-info <file>    Parse a GGUF file and print its header, metadata, tensors\n"
            << "  model-info <file>   Derive and print the model config from a GGUF file\n"
            << "  generate <file> --prompt \"...\"   Generate text from a GGUF model\n"
            << "  serve <file> [--port N]         Start the OpenAI-compatible HTTP server\n"
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
  if (command == "model-info") return cmdModelInfo(arg1);
  if (command == "generate") return cmdGenerate(args);
  if (command == "serve") return cmdServe(args);
  if (command == "gpu") return cmdGpu();
  if (command == "plugins") return cmdPlugins(arg1.empty() ? "plugins" : arg1);

  std::cerr << "Unknown command: " << command << "\n\n";
  printUsage();
  return 1;
}
