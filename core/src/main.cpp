#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "qorvix/api/http_server.hpp"
#include "qorvix/api/openai.hpp"
#include "qorvix/cuda/backend.hpp"
#include "qorvix/cuda/gpu_model.hpp"
#include "qorvix/cuda/multi_gpu.hpp"
#include "qorvix/vulkan/backend.hpp"
#include "qorvix/vulkan/vulkan_model.hpp"
#include "qorvix/backend.hpp"
#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/gpu_engine.hpp"
#include "qorvix/model_registry.hpp"
#include "qorvix/plugin_registry.hpp"
#include "qorvix/ports.hpp"
#include "qorvix/embeddings/embed_benchmark.hpp"
#include "qorvix/rag/pipeline.hpp"
#include "qorvix/vision/clip_model.hpp"
#include "qorvix/vision/image.hpp"
#include "qorvix/vision/vision_check.hpp"
#include "qorvix/runtime/benchmark.hpp"
#include "qorvix/runtime/cpu_features.hpp"
#include "qorvix/runtime/dequant.hpp"
#include "qorvix/runtime/generator.hpp"
#include "qorvix/runtime/model_config.hpp"
#include "qorvix/runtime/pooling.hpp"
#include "qorvix/runtime/text_model.hpp"
#include "qorvix/runtime/weights.hpp"
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
    std::cout << "Architecture:      " << cfg.architecture
              << (cfg.isEncoder() ? "  (encoder / embeddings)" : "  (decoder / generation)") << "\n"
              << "Vocab size:        " << cfg.vocabSize << "\n"
              << "Context length:    " << cfg.contextLength << "\n"
              << "Embedding (d_model): " << cfg.embeddingLength << "\n"
              << "Layers:            " << cfg.blockCount << "\n"
              << "FFN hidden:        " << cfg.feedForwardLength << "\n"
              << "Attention heads:   " << cfg.headCount << " (kv " << cfg.headCountKv << ", head_dim "
              << cfg.headDim() << ")\n";
    if (cfg.isEncoder()) {
      std::cout << "Attention:         bidirectional, "
                << (cfg.attnBias ? "q/k/v/o carry bias" : "no bias") << "\n"
                << "Position:          "
                << (cfg.hasPositionEmbd ? "learned table" : "rope") << "\n"
                << "FFN:               " << (cfg.ffnGated ? "gated" : "single") << ", GELU\n"
                << "Token types:       " << cfg.tokenTypeCount << "\n"
                << "Pooling:           " << qorvix::runtime::poolingName(cfg.pooling) << "\n"
                << "LayerNorm eps:     " << cfg.normEpsilon << "\n";
    } else {
      std::cout << "RoPE:              dim=" << cfg.ropeDimensionCount << " freq_base="
                << cfg.ropeFreqBase << "\n"
                << "RMSNorm eps:       " << cfg.normEpsilon << "\n";
    }
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

// The weight-bridge + model builders (buildGpuModel/buildVulkanModel) and the createEngine factory
// now live in qorvix/backend.hpp — the single unified backend layer. main.cpp only ever talks to
// runtime::IInferenceEngine via createEngine(), never to a concrete backend.
int argmaxOf(const std::vector<float>& v) {
  int best = 0;
  for (int i = 1; i < static_cast<int>(v.size()); ++i)
    if (v[i] > v[best]) best = i;
  return best;
}
bool hasFlag(const std::vector<std::string_view>& args, std::string_view flag) {
  for (const auto& a : args)
    if (a == flag) return true;
  return false;
}

// Chooses the backend from CLI flags: --gpu/--cuda -> cuda, --vulkan -> vulkan, --auto -> best
// available, --cpu / (nothing) -> cpu. One place, used by generate, serve, and bench.
qorvix::Backend backendFromArgs(const std::vector<std::string_view>& args) {
  if (hasFlag(args, "--gpu") || hasFlag(args, "--cuda")) return qorvix::Backend::Cuda;
  if (hasFlag(args, "--vulkan")) return qorvix::Backend::Vulkan;
  if (hasFlag(args, "--auto")) return qorvix::selectBestBackend();
  return qorvix::Backend::Cpu;  // explicit --cpu or no flag
}

// THE generation loop — backend-agnostic. Drives any IInferenceEngine through the seam
// (openSession -> forward -> sample -> decode), so CPU/CUDA/Vulkan run byte-identical host code and
// the only difference is which engine createEngine() handed back. Splits per-token wall time into
// forward / sampling / stdout so the bottleneck is visible on any backend.
int runGenerate(qorvix::runtime::IInferenceEngine& engine, qorvix::tokenizer::Tokenizer& tok,
                const std::string& prompt, const qorvix::runtime::GenerationConfig& cfg,
                double loadSec) {
  using clock = std::chrono::steady_clock;
  const auto session = engine.openSession();
  if (session == qorvix::memory::kInvalidSession) {
    std::cerr << "error: could not open a session on the " << engine.backendName() << " engine\n";
    return 1;
  }
  const int maxSeq = static_cast<int>(engine.maxSeqLen());
  qorvix::runtime::Sampler sampler(cfg.sampling, cfg.seed);
  const auto promptIds = tok.encode(prompt, cfg.addBos);
  std::vector<int> history = promptIds;
  const int eos = tok.special().eos;
  auto since = [](const clock::time_point& t0) {
    return std::chrono::duration<double>(clock::now() - t0).count();
  };

  std::cout << prompt << std::flush;
  const auto tGen0 = clock::now();
  double tFwd = 0, tSample = 0, tIo = 0;
  int pos = 0;
  std::vector<float> logits;
  for (std::size_t i = 0; i < promptIds.size() && pos < maxSeq; ++i, ++pos) {
    const auto t = clock::now();
    logits = engine.forward(session, promptIds[i], pos);
    tFwd += since(t);
  }
  int next;
  { const auto t = clock::now(); next = sampler.sample(logits, history); tSample += since(t); }

  int generated = 0;
  bool hitEos = false;
  for (int n = 0; n < cfg.maxNewTokens && pos < maxSeq; ++n) {
    if (next == eos) { hitEos = true; break; }
    { const auto t = clock::now(); std::cout << tok.decodeToken(next) << std::flush; tIo += since(t); }
    history.push_back(next);
    ++generated;
    { const auto t = clock::now(); logits = engine.forward(session, next, pos++); tFwd += since(t); }
    { const auto t = clock::now(); next = sampler.sample(logits, history); tSample += since(t); }
  }
  engine.closeSession(session);

  const double genSec = since(tGen0);
  const int forwards = static_cast<int>(promptIds.size()) + generated;
  std::cout << "\n\n[" << engine.backendName() << " | " << promptIds.size() << " prompt tokens, "
            << generated << " generated" << (hitEos ? ", eos" : "") << "]\n"
            << "[load " << std::fixed << std::setprecision(1) << loadSec << "s | " << forwards
            << " forwards in " << std::setprecision(2) << genSec << "s = "
            << (genSec > 0 ? forwards / genSec : 0.0) << " tok/s]\n";
  const double perFwdMs = forwards > 0 ? 1000.0 * tFwd / forwards : 0.0;
  std::cout << "[time split: forward " << std::setprecision(1) << 100.0 * tFwd / genSec << "% ("
            << std::setprecision(2) << perFwdMs << " ms/fwd) | sampling " << std::setprecision(1)
            << 100.0 * tSample / genSec << "% | stdout " << 100.0 * tIo / genSec << "%]\n";
  return 0;
}


int cmdGenerate(const std::vector<std::string_view>& args) {
  const std::string path = args.size() > 1 ? std::string(args[1]) : std::string();
  const std::string prompt = flagValue(args, "--prompt");
  if (path.empty() || prompt.empty()) {
    std::cerr << "usage: qorvix generate <file.gguf> --prompt \"...\" "
                 "[--gpu|--vulkan] [--max N] [--temp T] [--top-k K] [--top-p P] [--seed S]\n";
    return 1;
  }

  qorvix::runtime::GenerationConfig cfg;
  if (auto v = flagValue(args, "--max"); !v.empty()) cfg.maxNewTokens = std::stoi(v);
  if (auto v = flagValue(args, "--temp"); !v.empty()) cfg.sampling.temperature = std::stof(v);
  if (auto v = flagValue(args, "--top-k"); !v.empty()) cfg.sampling.topK = std::stoi(v);
  if (auto v = flagValue(args, "--top-p"); !v.empty()) cfg.sampling.topP = std::stof(v);
  if (auto v = flagValue(args, "--seed"); !v.empty()) cfg.seed = std::stoull(v);

  // One path for every backend: pick it, build one engine through the unified factory, run one loop.
  const qorvix::Backend backend = backendFromArgs(args);
  if (hasFlag(args, "--auto"))
    std::cerr << "[auto: using " << qorvix::backendName(backend) << " backend]\n";

  try {
    using clock = std::chrono::steady_clock;
    const auto tLoad0 = clock::now();
    auto file = qorvix::gguf::GgufFile::open(path);
    std::string err;
    // Tokenizer first (it copies the vocab out), then the file is moved into createEngine — the CPU
    // engine keeps it mapped for its borrowed weights; device engines upload and release it.
    auto tok = qorvix::tokenizer::Tokenizer::fromGguf(file, err);
    if (!tok) {
      std::cerr << "error: tokenizer: " << err << "\n";
      return 1;
    }
    const int maxSeq = static_cast<int>(tok->encode(prompt, cfg.addBos).size()) + cfg.maxNewTokens + 4;
    auto engine = qorvix::createEngine(backend, std::move(file), static_cast<std::uint32_t>(maxSeq),
                                       1, err);
    if (!engine) {
      std::cerr << "error: " << qorvix::backendName(backend) << " engine: " << err << "\n";
      return 1;
    }
    const double loadSec = std::chrono::duration<double>(clock::now() - tLoad0).count();
    return runGenerate(*engine, *tok, prompt, cfg, loadSec);
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

// Truncates a token sequence to the encoder's limit, keeping the closing [SEP]. Dropping the tail
// outright would leave the sequence unterminated, which shifts the vector on a CLS-pooled model
// (every layer attends to [SEP]) — so the last slot is overwritten rather than lost.
std::vector<int> truncateForEncoder(std::vector<int> ids, int maxSeq, int sepId) {
  if (static_cast<int>(ids.size()) <= maxSeq) return ids;
  ids.resize(maxSeq);
  if (sepId >= 0) ids.back() = sepId;
  return ids;
}

// ---- embed-check: the Phase 11a correctness gate -------------------------------------------
//
// Every prior phase diffed a new implementation against an existing one (gpu-check compares CUDA
// to the CPU reference). There is no second embedding implementation, so ground truth has to be
// imported: a fixture captured once from sentence-transformers by scripts/capture_embed_reference.py.
//
// The gate is split into tiers because a single "cosine > 0.99" says THAT something is wrong and
// nothing about WHAT:
//   tokens exact + cosine 0.9999 -> correct
//   tokens exact + cosine 0.7    -> encoder math (pooling, eps, GELU variant, mask, positions)
//   tokens mismatched            -> tokenizer; fix that first, then re-read the vector tier
// Without the token tier those are indistinguishable — and a WordPiece convention bug produced
// exactly the second signature during this phase's development.
struct EmbedReference {
  std::string model;
  int dim = 0;
  std::string pooling = "cls";
  bool normalize = true;
  // The reference implementation's own truncation limit, which is NOT always the GGUF's context
  // length: sentence-transformers caps all-MiniLM-L6-v2 at 256 in its config while the model (and
  // the GGUF) support 512. Comparing a 512-token vector against a 256-token one would report a
  // difference that is a configuration mismatch, not an encoder bug.
  int maxSeqLen = 0;
  std::vector<std::string> texts;
  std::vector<std::vector<int>> ids;
  std::vector<std::vector<float>> vecs;
};

bool loadEmbedReference(const std::string& path, EmbedReference& ref, std::string& error) {
  std::ifstream in(path);
  if (!in) {
    error = "cannot open '" + path + "'";
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    if (key == "model") {
      ls >> ref.model;
    } else if (key == "dim") {
      ls >> ref.dim;
    } else if (key == "pooling") {
      ls >> ref.pooling;
    } else if (key == "max_seq_len") {
      ls >> ref.maxSeqLen;
    } else if (key == "normalize") {
      int n = 1;
      ls >> n;
      ref.normalize = n != 0;
    } else if (key == "text") {
      std::string rest;
      std::getline(ls, rest);
      if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
      ref.texts.push_back(rest);
    } else if (key == "ids") {
      std::vector<int> v;
      int id = 0;
      while (ls >> id) v.push_back(id);
      ref.ids.push_back(std::move(v));
    } else if (key == "vec") {
      std::vector<float> v;
      float f = 0.0f;
      while (ls >> f) v.push_back(f);
      ref.vecs.push_back(std::move(v));
    }
  }
  if (ref.texts.empty() || ref.texts.size() != ref.ids.size() ||
      ref.texts.size() != ref.vecs.size()) {
    error = "malformed fixture: " + std::to_string(ref.texts.size()) + " texts, " +
            std::to_string(ref.ids.size()) + " id rows, " + std::to_string(ref.vecs.size()) +
            " vec rows";
    return false;
  }
  return true;
}

int cmdEmbedCheck(const std::vector<std::string_view>& args) {
  const std::string path = args.size() > 1 ? std::string(args[1]) : std::string();
  if (path.empty()) {
    std::cerr << "usage: qorvix embed-check <file.gguf> [--ref <fixture>] [--min-cos 0.999]\n";
    return 1;
  }
  const std::string refPath = flagValue(args, "--ref");
  float minCos = 0.999f;
  if (auto v = flagValue(args, "--min-cos"); !v.empty()) minCos = std::stof(v);

  try {
    auto file = qorvix::gguf::GgufFile::open(path);
    std::string err;
    auto tok = qorvix::tokenizer::Tokenizer::fromGguf(file, err);
    if (!tok) {
      std::cerr << "error: tokenizer: " << err << "\n";
      return 1;
    }
    auto engine = qorvix::createEmbeddingEngine(qorvix::Backend::Cpu, std::move(file), 0, err);
    if (!engine) {
      std::cerr << "error: " << err << "\n";
      return 1;
    }
    const int d = static_cast<int>(engine->dim());
    std::cout << "model:   " << path << "\n"
              << "config:  " << engine->config().architecture << ", dim " << d << ", pooling "
              << qorvix::runtime::poolingName(engine->defaultPooling()) << ", ctx "
              << engine->maxSeqLen() << "\n\n";

    // Truncation limit for this run. The reference's own limit wins when it is lower, so the two
    // implementations see the same token sequence.
    int seqCap = static_cast<int>(engine->maxSeqLen());

    auto embedTextAt = [&](const std::string& text, int cap, std::vector<float>& out) {
      const std::vector<int> ids =
          truncateForEncoder(tok->encode(text, true), cap, tok->special().sep);
      return engine->embed(ids, out, err);
    };
    auto embedText = [&](const std::string& text, std::vector<float>& out) {
      return embedTextAt(text, seqCap, out);
    };

    bool pass = true;

    // ---- tier 1+2: parity against the captured reference ----
    EmbedReference ref;
    bool haveRef = false;
    if (!refPath.empty()) {
      if (!loadEmbedReference(refPath, ref, err)) {
        std::cerr << "error: reference: " << err << "\n";
        return 1;
      }
      haveRef = true;
    }

    if (haveRef) {
      if (ref.dim != d) {
        std::cout << "Reference dim " << ref.dim << " != model dim " << d
                  << " — wrong fixture for this model.\n\nRESULT: MISMATCH\n";
        return 1;
      }
      const std::string ourPooling = qorvix::runtime::poolingName(engine->defaultPooling());
      if (ref.pooling != ourPooling) {
        std::cout << "Reference pooling '" << ref.pooling << "' != model pooling '" << ourPooling
                  << "' — comparing these would report an encoder bug that does not exist.\n\n"
                  << "RESULT: MISMATCH\n";
        return 1;
      }
      if (ref.maxSeqLen > 0 && ref.maxSeqLen < seqCap) {
        std::cout << "note: truncating to the reference's " << ref.maxSeqLen
                  << "-token limit (model allows " << seqCap << ")\n\n";
        seqCap = ref.maxSeqLen;
      }
      int tokOk = 0;
      std::string tokWorst;
      for (std::size_t i = 0; i < ref.texts.size(); ++i) {
        // Truncate the same way the capture did, or the long probe reports a mismatch that is
        // an artefact of where each side cut the sequence.
        const std::vector<int> got =
            truncateForEncoder(tok->encode(ref.texts[i], true), seqCap, tok->special().sep);
        if (got == ref.ids[i]) {
          ++tokOk;
        } else if (tokWorst.empty()) {
          tokWorst = ref.texts[i];
        }
      }
      std::cout << "Tokenizer parity (" << ref.texts.size() << " strings):     " << tokOk << "/"
                << ref.texts.size() << " exact";
      if (tokOk != static_cast<int>(ref.texts.size())) std::cout << "  (first mismatch: \"" << tokWorst << "\")";
      std::cout << "\n";
      if (tokOk != static_cast<int>(ref.texts.size())) pass = false;

      float worstCos = 2.0f;
      std::string worstText;
      std::vector<std::pair<std::string, float>> perProbe;
      for (std::size_t i = 0; i < ref.texts.size(); ++i) {
        std::vector<float> v;
        if (!embedText(ref.texts[i], v)) {
          std::cout << "  embed failed on \"" << ref.texts[i] << "\": " << err << "\n";
          pass = false;
          continue;
        }
        const float c = qorvix::runtime::ops::cosineSimilarity(v.data(), ref.vecs[i].data(), d);
        perProbe.emplace_back(ref.texts[i], c);
        if (c < worstCos) {
          worstCos = c;
          worstText = ref.texts[i];
        }
      }
      const bool vecPass = worstCos >= minCos;
      std::cout << std::fixed << std::setprecision(5)
                << "Vector parity vs reference:      min cos " << worstCos << "  (worst: \""
                << worstText << "\")\n";
      // On failure, show every probe. The fixture is built so each string isolates one cause, and
      // that only pays off if a failure names which one — "min cos 0.976" alone does not say
      // whether one degenerate input or the whole encoder is off.
      if (!vecPass) {
        for (const auto& [text, c] : perProbe) {
          std::string label = text.size() > 44 ? text.substr(0, 41) + "..." : text;
          if (label.empty()) label = "(empty string)";
          std::cout << "    " << (c < minCos ? "FAIL " : "ok   ") << std::setw(8) << c << "  \""
                    << label << "\" (" << tok->encode(text, true).size() << " tok)\n";
        }
      }
      std::cout << std::defaultfloat;
      if (!vecPass) pass = false;
    } else {
      std::cout << "Tokenizer parity:                SKIPPED (no --ref fixture)\n"
                << "Vector parity vs reference:      SKIPPED (no --ref fixture)\n";
    }

    // ---- tier 3: invariants, which need no fixture and so always run ----
    std::vector<float> a, b, a2;
    const bool okA = embedText("A dog runs in the park", a);
    const bool okB = embedText("A canine sprints across the grass", b);
    const bool okA2 = embedText("A dog runs in the park", a2);
    if (!okA || !okB || !okA2) {
      std::cout << "Invariants:                      FAILED to embed (" << err << ")\n";
      pass = false;
    } else {
      const float norm = qorvix::runtime::ops::l2Norm(a.data(), d);
      const bool unit = !engine->defaultNormalize() || std::abs(norm - 1.0f) < 1e-4f;
      const bool self = qorvix::runtime::ops::cosineSimilarity(a.data(), a.data(), d) > 0.9999f;
      const bool sym = std::abs(qorvix::runtime::ops::cosineSimilarity(a.data(), b.data(), d) -
                                qorvix::runtime::ops::cosineSimilarity(b.data(), a.data(), d)) < 1e-6f;
      bool finite = true;
      for (float x : a) finite = finite && std::isfinite(x);
      const bool deterministic = a == a2;
      std::cout << "Invariants:                      "
                << (unit ? "unit norm OK" : "UNIT NORM FAIL") << " · "
                << (self ? "self-cos OK" : "SELF-COS FAIL") << " · "
                << (sym ? "symmetry OK" : "SYMMETRY FAIL") << " · "
                << (finite ? "finite OK" : "NON-FINITE") << " · "
                << (deterministic ? "deterministic OK" : "NONDETERMINISTIC") << "\n";
      pass = pass && unit && self && sym && finite && deterministic;
    }

    // ---- tier 4: triplet ordering. Catches gross breakage with no fixture at all. ----
    struct Triplet {
      const char* anchor;
      const char* positive;
      const char* negative;
    };
    const Triplet triplets[] = {
        {"A dog runs in the park", "A canine sprints across the grass",
         "The stock market closed lower today"},
        {"How do I install the software?", "What are the setup instructions?",
         "The cat slept on the windowsill"},
        {"Paris is the capital of France", "France's capital city is Paris",
         "Photosynthesis occurs in chloroplasts"},
    };
    int trip = 0;
    float minMargin = 1e9f;
    for (const auto& t : triplets) {
      std::vector<float> va, vp, vn;
      if (!embedText(t.anchor, va) || !embedText(t.positive, vp) || !embedText(t.negative, vn)) {
        pass = false;
        continue;
      }
      const float cp = qorvix::runtime::ops::cosineSimilarity(va.data(), vp.data(), d);
      const float cn = qorvix::runtime::ops::cosineSimilarity(va.data(), vn.data(), d);
      if (cp > cn + 0.10f) ++trip;
      minMargin = std::min(minMargin, cp - cn);
    }
    std::cout << std::fixed << std::setprecision(3) << "Triplet ordering:                " << trip
              << "/3" << " (min margin " << minMargin << ")\n"
              << std::defaultfloat;
    if (trip != 3) pass = false;

    std::cout << "\nRESULT: " << (pass ? "PASS" : "MISMATCH");
    if (pass && !haveRef) {
      std::cout << " (invariants only — no reference fixture)\n"
                << "note: capture one with scripts/capture_embed_reference.py and pass --ref to\n"
                << "      gate on vectors, which is the tier that catches subtle encoder bugs.\n";
      // Returning 0 without a fixture matches gpu-check, which also exits 0 when its reference
      // simply isn't available rather than treating "cannot check" as "failed".
      return 0;
    }
    std::cout << "\n";
    return pass ? 0 : 1;
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

int cmdEmbed(const std::vector<std::string_view>& args) {
  const std::string path = args.size() > 1 ? std::string(args[1]) : std::string();
  const std::string text = flagValue(args, "--text");
  if (path.empty() || text.empty()) {
    std::cerr << "usage: qorvix embed <file.gguf> --text \"...\" "
                 "[--pooling mean|cls|last] [--no-normalize] [--dims N] [--json]\n";
    return 1;
  }
  const bool json = hasFlag(args, "--json");
  const bool normalize = !hasFlag(args, "--no-normalize");
  int dims = 0;
  if (auto v = flagValue(args, "--dims"); !v.empty()) dims = std::stoi(v);

  const qorvix::Backend backend = backendFromArgs(args);
  try {
    using clock = std::chrono::steady_clock;
    const auto tLoad0 = clock::now();
    auto file = qorvix::gguf::GgufFile::open(path);
    std::string err;
    // Tokenizer first (it copies the vocab out), then the file moves into the engine, which keeps
    // it mapped for the borrowed quantized weights.
    auto tok = qorvix::tokenizer::Tokenizer::fromGguf(file, err);
    if (!tok) {
      std::cerr << "error: tokenizer: " << err << "\n";
      return 1;
    }
    auto engine = qorvix::createEmbeddingEngine(backend, std::move(file), 0, err);
    if (!engine) {
      std::cerr << "error: " << qorvix::backendName(backend) << " embedding engine: " << err << "\n";
      return 1;
    }
    const double loadSec = std::chrono::duration<double>(clock::now() - tLoad0).count();

    qorvix::runtime::PoolingType pooling = engine->defaultPooling();
    if (auto v = flagValue(args, "--pooling"); !v.empty()) {
      if (!qorvix::runtime::parsePooling(v, pooling)) {
        std::cerr << "error: unknown pooling '" << v << "' (expected mean, cls, or last)\n";
        return 1;
      }
    }

    const std::vector<int> ids = truncateForEncoder(
        tok->encode(text, true), static_cast<int>(engine->maxSeqLen()), tok->special().sep);

    const auto tRun0 = clock::now();
    std::vector<float> vec;
    if (!engine->embedWith(ids, pooling, normalize, vec, err)) {
      std::cerr << "error: " << err << "\n";
      return 1;
    }
    const double runSec = std::chrono::duration<double>(clock::now() - tRun0).count();

    // Matryoshka truncation: only meaningful for models trained for it (bge-m3, nomic v1.5), but
    // clients send it regardless, so honour it rather than erroring.
    if (dims > 0 && dims < static_cast<int>(vec.size())) {
      vec.resize(dims);
      if (normalize) qorvix::runtime::ops::l2Normalize(vec.data(), static_cast<int>(vec.size()));
    }

    if (json) {
      std::cout << "[";
      for (std::size_t i = 0; i < vec.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << vec[i];
      }
      std::cout << "]\n";
      return 0;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "model:    " << engine->config().architecture << " (" << engine->backendName()
              << ")\n"
              << "tokens:   " << ids.size() << " / " << engine->maxSeqLen() << "\n"
              << "pooling:  " << qorvix::runtime::poolingName(pooling)
              << (normalize ? ", L2-normalized" : ", unnormalized") << "\n"
              << "dim:      " << vec.size() << "\n"
              << "L2 norm:  " << qorvix::runtime::ops::l2Norm(vec.data(),
                                                              static_cast<int>(vec.size()))
              << "\n"
              << "first 8:  [";
    for (std::size_t i = 0; i < vec.size() && i < 8; ++i) {
      if (i) std::cout << ", ";
      std::cout << vec[i];
    }
    std::cout << (vec.size() > 8 ? ", ...]\n" : "]\n");
    std::cout << std::defaultfloat << "[load " << loadSec << "s | embed " << runSec << "s]\n";
    return 0;
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

// ---- vision: CLIP tower + LLaVA projector ---------------------------------------------------

int cmdImageEmbed(const std::vector<std::string_view>& args) {
  namespace vis = qorvix::vision;
  const std::string mm = args.size() > 1 ? std::string(args[1]) : std::string();
  const std::string imagePath = flagValue(args, "--image");
  if (mm.empty() || imagePath.empty()) {
    std::cerr << "usage: qorvix image-embed <mmproj.gguf> --image <file.png> [--project] [--json]\n";
    return 1;
  }
  const bool json = hasFlag(args, "--json");
  const bool project = hasFlag(args, "--project");

  try {
    using clock = std::chrono::steady_clock;
    const auto tLoad0 = clock::now();
    std::string err;
    auto model = vis::ClipVisionModel::fromGguf(qorvix::gguf::GgufFile::open(mm), err);
    if (!model) {
      std::cerr << "error: vision tower: " << err << "\n";
      return 1;
    }
    const double loadSec = std::chrono::duration<double>(clock::now() - tLoad0).count();

    vis::Image img;
    if (!vis::loadImage(imagePath, img, err)) {
      std::cerr << "error: " << err << "\n";
      return 1;
    }

    const auto tRun0 = clock::now();
    std::vector<float> hidden;
    if (!model->encodeImage(img, hidden, err)) {
      std::cerr << "error: " << err << "\n";
      return 1;
    }
    std::vector<float> projected;
    if (project) {
      if (!model->project(hidden, projected, err)) {
        std::cerr << "error: " << err << "\n";
        return 1;
      }
    }
    const double runSec = std::chrono::duration<double>(clock::now() - tRun0).count();

    const std::vector<float>& outv = project ? projected : hidden;
    const int tokens = static_cast<int>(model->patchTokens());
    const int dim = project ? model->projectedDim() : static_cast<int>(model->embeddingLength());

    if (json) {
      std::cout << std::fixed << std::setprecision(6) << "{\"tokens\":" << tokens
                << ",\"dim\":" << dim << ",\"projected\":" << (project ? "true" : "false")
                << ",\"features\":[";
      for (std::size_t i = 0; i < outv.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << outv[i];
      }
      std::cout << "]}\n";
      return 0;
    }

    std::cout << std::fixed << std::setprecision(6) << "model:    " << model->config().name
              << " (cpu)\n"
              << "image:    " << imagePath << "  " << img.width << "x" << img.height << " -> "
              << model->config().imageSize << "x" << model->config().imageSize << "\n"
              << "tower:    " << model->config().blockCount << " layers, d "
              << model->config().embeddingLength << ", patch " << model->config().patchSize << ", "
              << (model->config().useGelu ? "gelu" : "quick-gelu") << "\n"
              << "features: " << tokens << " patch tokens x " << dim
              << (project ? "  (projected for the language model)" : "  (vision hidden states)")
              << "\n"
              << "first 8:  [";
    for (int i = 0; i < 8 && i < static_cast<int>(outv.size()); ++i) {
      if (i) std::cout << ", ";
      std::cout << outv[static_cast<std::size_t>(i)];
    }
    std::cout << ", ...]\n"
              << std::defaultfloat << "[load " << loadSec << "s | encode " << runSec << "s]\n";
    return 0;
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

// The Phase 11b analogue of embed-check. Tiered the same way and for the same reason: the vision
// path has two independently-wrong halves, and one aggregate verdict would not say which.
int cmdVisionCheck(const std::vector<std::string_view>& args) {
  namespace vis = qorvix::vision;
  const std::string mm = args.size() > 1 ? std::string(args[1]) : std::string();
  const std::string refPath = flagValue(args, "--ref");
  std::string imagePath = flagValue(args, "--image");
  if (mm.empty() || refPath.empty()) {
    std::cerr << "usage: qorvix vision-check <mmproj.gguf> --ref <fixture> [--image <file.png>] "
                 "[--min-cos 0.999]\n";
    return 1;
  }
  if (imagePath.empty()) imagePath = "tests/data/vision_probe.png";
  float minCos = 0.999f;
  if (auto v = flagValue(args, "--min-cos"); !v.empty()) minCos = std::stof(v);

  try {
    std::string err;
    vis::VisionReference ref;
    if (!ref.load(refPath, err)) {
      std::cerr << "error: reference: " << err << "\n";
      return 1;
    }
    vis::Image img;
    if (!vis::loadImage(imagePath, img, err)) {
      std::cerr << "error: " << err << "\n";
      return 1;
    }
    auto model = vis::ClipVisionModel::fromGguf(qorvix::gguf::GgufFile::open(mm), err);
    if (!model) {
      std::cerr << "error: vision tower: " << err << "\n";
      return 1;
    }

    std::cout << "tower:   " << mm << "\n"
              << "config:  " << model->config().blockCount << " layers, d "
              << model->config().embeddingLength << ", " << model->patchTokens() << " patches, "
              << (model->config().useGelu ? "gelu" : "quick-gelu") << "\n"
              << "image:   " << imagePath << " (" << img.width << "x" << img.height << ")\n"
              << "ref:     " << ref.model << "\n\n";

    if (static_cast<int>(model->config().imageSize) != ref.imageSize) {
      std::cout << "Reference image size " << ref.imageSize << " != tower's "
                << model->config().imageSize << " — wrong fixture.\n\nRESULT: MISMATCH\n";
      return 1;
    }

    bool pass = true;

    // ---- tier 1: preprocessing ----
    // Separated because it is the half most likely to be subtly wrong and the half whose errors
    // are invisible: a different resize filter or crop origin moves every downstream value and
    // reports nothing. Both faults found while building this were here, not in the transformer.
    std::vector<float> chw;
    if (!vis::preprocessClip(img, model->preprocessConfig(), chw, err)) {
      std::cerr << "error: preprocess: " << err << "\n";
      return 1;
    }
    double mean = 0.0, absMean = 0.0;
    for (float v : chw) {
      mean += v;
      absMean += std::abs(v);
    }
    mean /= static_cast<double>(chw.size());
    absMean /= static_cast<double>(chw.size());
    double worstPixel = std::max(std::abs(mean - ref.pixelMean), std::abs(absMean - ref.pixelAbsMean));
    const int S = ref.imageSize;
    for (const auto& p : ref.pixelProbes) {
      const float got = chw[(static_cast<std::size_t>(p.c) * S + p.y) * S + p.x];
      worstPixel = std::max(worstPixel, static_cast<double>(std::abs(got - p.value)));
    }
    std::cout << std::scientific << std::setprecision(2)
              << "Preprocessing vs reference:      max |diff| " << worstPixel << "  ("
              << ref.pixelProbes.size() << " pixel probes + mean/abs-mean)\n";
    if (worstPixel > 1e-3) pass = false;

    // ---- tier 2: the transformer ----
    std::vector<float> hidden;
    if (!model->encodePixels(chw, hidden, err)) {
      std::cerr << "error: encode: " << err << "\n";
      return 1;
    }
    const int P = static_cast<int>(model->patchTokens());
    const int D = static_cast<int>(model->embeddingLength());
    if (P != ref.patches || D != ref.dim) {
      std::cout << "Feature shape " << P << "x" << D << " != reference " << ref.patches << "x"
                << ref.dim << "\n\nRESULT: MISMATCH\n";
      return 1;
    }

    const float cosRow0 = qorvix::runtime::ops::cosineSimilarity(hidden.data(), ref.row0.data(), D);
    double maxRow0 = 0.0;
    for (int i = 0; i < D; ++i) {
      maxRow0 = std::max(maxRow0, static_cast<double>(std::abs(hidden[i] - ref.row0[i])));
    }

    std::vector<float> means(static_cast<std::size_t>(P));
    for (int p = 0; p < P; ++p) {
      double m = 0.0;
      for (int i = 0; i < D; ++i) m += hidden[static_cast<std::size_t>(p) * D + i];
      means[static_cast<std::size_t>(p)] = static_cast<float>(m / D);
    }
    const float cosMeans =
        qorvix::runtime::ops::cosineSimilarity(means.data(), ref.rowMeans.data(), P);

    std::cout << std::fixed << std::setprecision(7)
              << "Patch-token parity (row 0):      cos " << cosRow0 << std::scientific
              << std::setprecision(2) << "   max |diff| " << maxRow0 << "\n"
              << std::fixed << std::setprecision(7)
              << "All-patch parity (row means):    cos " << cosMeans << "\n"
              << std::defaultfloat;
    if (cosRow0 < minCos || cosMeans < minCos) pass = false;

    // ---- tier 3: the projector, when the tower carries one ----
    if (model->hasProjector()) {
      std::vector<float> projected;
      if (!model->project(hidden, projected, err)) {
        std::cerr << "error: project: " << err << "\n";
        return 1;
      }
      bool finite = true;
      for (float v : projected) finite = finite && std::isfinite(v);
      std::cout << "LLaVA projector:                 " << P << " x " << model->projectedDim()
                << (finite ? "  finite OK" : "  NON-FINITE") << "\n";
      pass = pass && finite;
    } else {
      std::cout << "LLaVA projector:                 absent in this tower\n";
    }

    std::cout << "\nRESULT: " << (pass ? "PASS" : "MISMATCH") << "\n";
    return pass ? 0 : 1;
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

// ---- rag: index and search over a document directory ----------------------------------------

// Loads an embedding model plus its tokenizer, the pair every RAG command needs. Keeps the
// tokenizer-first-then-move-the-file invariant in one place instead of both subcommands.
bool openEmbedder(const std::string& path, std::optional<qorvix::tokenizer::Tokenizer>& tok,
                  std::unique_ptr<qorvix::embeddings::IEmbeddingEngine>& engine) {
  try {
    auto file = qorvix::gguf::GgufFile::open(path);
    std::string err;
    tok = qorvix::tokenizer::Tokenizer::fromGguf(file, err);
    if (!tok) {
      std::cerr << "error: tokenizer: " << err << "\n";
      return false;
    }
    engine = qorvix::createEmbeddingEngine(qorvix::Backend::Cpu, std::move(file), 0, err);
    if (!engine) {
      std::cerr << "error: embedding engine: " << err << "\n";
      return false;
    }
    return true;
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return false;
  }
}

int cmdRagIndex(const std::vector<std::string_view>& args) {
  namespace rag = qorvix::rag;
  const std::string dir = args.size() > 2 ? std::string(args[2]) : std::string();
  const std::string modelPath = flagValue(args, "--embed-model");
  const std::string storePath = flagValue(args, "--store");
  if (dir.empty() || modelPath.empty() || storePath.empty()) {
    std::cerr << "usage: qorvix rag index <dir> --embed-model <f.gguf> --store <out.qvx> "
                 "[--chunk-tokens N] [--overlap N]\n";
    return 1;
  }
  rag::ChunkOptions opt;
  if (auto v = flagValue(args, "--chunk-tokens"); !v.empty()) opt.maxTokens = std::stoi(v);
  if (auto v = flagValue(args, "--overlap"); !v.empty()) opt.overlapTokens = std::stoi(v);

  std::optional<qorvix::tokenizer::Tokenizer> tok;
  std::unique_ptr<qorvix::embeddings::IEmbeddingEngine> engine;
  if (!openEmbedder(modelPath, tok, engine)) return 1;

  std::cout << "indexing " << dir << "\n"
            << "  model: " << modelPath << " (dim " << engine->dim() << ", pooling "
            << qorvix::runtime::poolingName(engine->defaultPooling()) << ")\n"
            << "  chunks: " << opt.maxTokens << " tokens, " << opt.overlapTokens << " overlap\n";

  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  rag::Index index;
  rag::IndexStats stats;
  std::string err;
  // Indexing a directory is minutes of CPU-encoder work; a silent process looks hung.
  auto progress = [](int done, int total, const std::string& source) {
    if (!source.empty()) {
      std::cout << "  [" << done + 1 << "/" << total << "] " << source << "\n" << std::flush;
    }
  };
  if (!rag::buildIndex(dir, *engine, *tok, opt, index, stats, err, progress)) {
    std::cerr << "error: " << err << "\n";
    return 1;
  }
  const double sec = std::chrono::duration<double>(clock::now() - t0).count();

  if (!index.save(storePath, err)) {
    std::cerr << "error: " << err << "\n";
    return 1;
  }
  std::cout << std::fixed << std::setprecision(1) << "\nindexed " << stats.documents
            << " documents into " << stats.chunks << " chunks (" << stats.tokens << " tokens) in "
            << sec << "s\n"
            << "stored: " << storePath << "\n";
  for (const auto& s : stats.skipped) std::cout << "  skipped " << s << "\n";
  return 0;
}

int cmdRagSearch(const std::vector<std::string_view>& args) {
  namespace rag = qorvix::rag;
  const std::string storePath = flagValue(args, "--store");
  const std::string modelPath = flagValue(args, "--embed-model");
  const std::string query = flagValue(args, "--query");
  if (storePath.empty() || modelPath.empty() || query.empty()) {
    std::cerr << "usage: qorvix rag search --store <x.qvx> --embed-model <f.gguf> "
                 "--query \"...\" [--k 5] [--alpha 0.5]\n";
    return 1;
  }
  rag::HybridOptions opt;
  if (auto v = flagValue(args, "--k"); !v.empty()) opt.k = std::stoi(v);
  if (auto v = flagValue(args, "--alpha"); !v.empty()) opt.alpha = std::stof(v);

  std::string err;
  rag::Index index;
  if (!rag::Index::load(storePath, index, err)) {
    std::cerr << "error: " << err << "\n";
    return 1;
  }

  std::optional<qorvix::tokenizer::Tokenizer> tok;
  std::unique_ptr<qorvix::embeddings::IEmbeddingEngine> engine;
  if (!openEmbedder(modelPath, tok, engine)) return 1;

  std::vector<rag::SearchHit> hits;
  if (!rag::queryIndex(index, *engine, *tok, query, opt, hits, err)) {
    std::cerr << "error: " << err << "\n";
    return 1;
  }

  std::cout << "query: \"" << query << "\"\n"
            << index.store.size() << " chunks | alpha " << opt.alpha << " ("
            << (opt.alpha >= 1.0f  ? "dense only"
                : opt.alpha <= 0.0f ? "lexical only"
                                    : "hybrid RRF")
            << ")\n\n";
  if (hits.empty()) {
    std::cout << "no results\n";
    return 0;
  }
  for (std::size_t i = 0; i < hits.size(); ++i) {
    const auto& c = index.store.chunk(hits[i].index);
    std::string snippet = c.text.substr(0, 160);
    for (char& ch : snippet) {
      if (ch == '\n' || ch == '\t') ch = ' ';
    }
    std::cout << std::fixed << std::setprecision(4) << "[" << i + 1 << "] " << hits[i].score << "  "
              << c.source << " #" << c.index << " (bytes " << c.byteStart << "-" << c.byteEnd
              << ")\n      " << snippet << (c.text.size() > 160 ? "..." : "") << "\n";
  }
  return 0;
}

int cmdRag(const std::vector<std::string_view>& args) {
  const std::string sub = args.size() > 1 ? std::string(args[1]) : std::string();
  if (sub == "index") return cmdRagIndex(args);
  if (sub == "search" || sub == "query") return cmdRagSearch(args);
  std::cerr << "usage: qorvix rag index|search ...\n"
            << "  qorvix rag index <dir> --embed-model <f.gguf> --store <out.qvx>\n"
            << "  qorvix rag search --store <x.qvx> --embed-model <f.gguf> --query \"...\"\n";
  return 1;
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
    std::cerr << "usage: qorvix serve <file.gguf> [--gpu|--vulkan|--auto] [--port N] "
                 "[--max-concurrent N] [--ctx N] [--embed-model <file.gguf>] [--max-batch N]\n";
    return 1;
  }
  // SPEC: "Everything must run from ONE server process." The positional model stays the generation
  // model and embeddings are opt-in, so this is backwards compatible. A general --model registry
  // with per-request name routing is the right long-term shape — it is what port 2006 (Gateway) is
  // reserved for — but it needs a loaded-engine registry, name->engine dispatch, and an eviction
  // policy; --embed-model is the seam that grows into it.
  const std::string embedPath = flagValue(args, "--embed-model");
  // Same backend selection as generate: --gpu / --vulkan / --auto / (default) CPU. serve reaches
  // every backend through the ONE createEngine() factory + the IInferenceEngine seam.
  const qorvix::Backend backend = backendFromArgs(args);
  int port = qorvix::ports::kRuntime, maxConcurrent = 4, ctx = 4096, maxBatch = 64;
  if (auto v = flagValue(args, "--port"); !v.empty()) port = std::stoi(v);
  if (auto v = flagValue(args, "--max-concurrent"); !v.empty()) maxConcurrent = std::stoi(v);
  if (auto v = flagValue(args, "--ctx"); !v.empty()) ctx = std::stoi(v);
  if (auto v = flagValue(args, "--max-batch"); !v.empty()) maxBatch = std::stoi(v);

  if (!qorvix::backendAvailable(backend)) {
    std::cerr << "error: " << qorvix::backendName(backend)
              << " backend requested but unavailable (not built in, or no device)\n";
    return 1;
  }

  std::string err;
  std::optional<qorvix::tokenizer::Tokenizer> tok;
  std::unique_ptr<qorvix::runtime::IInferenceEngine> engine;
  std::string chatTemplate;
  try {
    auto file = qorvix::gguf::GgufFile::open(path);
    tok = qorvix::tokenizer::Tokenizer::fromGguf(file, err);
    if (!tok) {
      std::cerr << "error: tokenizer: " << err << "\n";
      return 1;
    }
    // The model's own chat template. Must be read before `file` is moved into createEngine (the CPU
    // engine takes ownership). Without it every model got the generic "role:\n" prompt, which
    // instruction-tuned models were not trained on — the biggest cause of poor chat output.
    chatTemplate = file.getString("tokenizer.chat_template").value_or("");
    // One KV slot per concurrent request (device engines size their cache to maxConcurrent). For
    // maxConcurrent applies to every backend now (CPU, CUDA, and multi-session Vulkan).
    engine = qorvix::createEngine(backend, std::move(file), static_cast<std::uint32_t>(ctx),
                                  static_cast<std::uint32_t>(maxConcurrent), err);
    if (!engine) {
      std::cerr << "error: " << qorvix::backendName(backend) << " engine: " << err << "\n";
      return 1;
    }
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  // Second engine, second seam, same process. It carries its OWN tokenizer — an embedding model
  // is WordPiece while the chat model is SPM or BPE, so they cannot be shared.
  std::optional<qorvix::tokenizer::Tokenizer> embedTok;
  std::unique_ptr<qorvix::embeddings::IEmbeddingEngine> embedEngine;
  if (!embedPath.empty()) {
    try {
      auto ef = qorvix::gguf::GgufFile::open(embedPath);
      embedTok = qorvix::tokenizer::Tokenizer::fromGguf(ef, err);
      if (!embedTok) {
        std::cerr << "error: embedding tokenizer: " << err << "\n";
        return 1;
      }
      // CPU only for now (see createEmbeddingEngine); the flag deliberately does not follow the
      // chat backend, so `serve --gpu --embed-model x` does not silently claim a GPU encoder.
      embedEngine = qorvix::createEmbeddingEngine(qorvix::Backend::Cpu, std::move(ef), 0, err);
      if (!embedEngine) {
        std::cerr << "error: embedding engine: " << err << "\n";
        return 1;
      }
    } catch (const qorvix::gguf::GgufParseError& e) {
      std::cerr << "error: embedding model: " << e.what() << "\n";
      return 1;
    }
  }

  const std::string modelId = engine->config().architecture + "/" + path;
  const std::string embedModelId =
      embedEngine ? embedEngine->config().architecture + "/" + embedPath : std::string();
  qorvix::scheduler::Scheduler sched(*engine, *tok, {maxConcurrent});
  std::atomic<long long> idCounter{0};

  // The HTTP server now handles each connection on its own detached thread, but Scheduler is
  // explicitly single-threaded (priority queue, active-request vector) and every engine hands
  // back a reference to ONE reused logits buffer. Concurrent handlers would corrupt both. This
  // mutex serializes the scheduler section so requests queue instead of racing.
  //
  // Note this bounds real concurrency at one in-flight generation: the accept loop no longer
  // blocks, but generations still run one at a time. Overlapping them needs the scheduler to
  // batch across requests (IInferenceEngine::forwardBatch) rather than draining per request.
  std::mutex schedMutex;

  // A SEPARATE lock, not schedMutex. That one exists for two specific objects: the single-threaded
  // Scheduler and the generation engine's one reused logits buffer. BertModel is a different
  // object with its own scratch — sharing schedMutex would queue every embedding behind every
  // in-flight generation for no reason, while sharing nothing would corrupt scratch under the
  // one-detached-thread-per-connection model.
  std::mutex embedMutex;

  api::HttpServer server(port);
  if (!server.start(err)) {
    std::cerr << "error: " << err << "\n";
    // A bind failure on one of our own reserved ports is nearly always a second Qorvix service
    // already running, so name it rather than leaving the operator to look the number up.
    if (const auto svc = qorvix::ports::serviceName(port); !svc.empty())
      std::cerr << "note: port " << port << " is the default for " << svc
                << " — is one already running? Override with --port N.\n";
    return 1;
  }
  // Zephyr-style templates interpolate the model's EOS piece between turns, so resolve it once.
  const std::string eosPiece =
      tok->special().eos >= 0 ? tok->decodeToken(tok->special().eos) : std::string("</s>");
  const std::string chatFamily = api::detectChatTemplateFamily(chatTemplate);

  std::cout << "qorvix serving " << path << " on http://0.0.0.0:" << port << "\n"
            << "  backend: " << engine->backendName() << " | max-concurrent: " << maxConcurrent
            << " | ctx: " << ctx << "\n"
            << "  chat template: " << chatFamily
            << (chatTemplate.empty() ? " (model has none; using generic prompt)" : "") << "\n";
  if (embedEngine) {
    std::cout << "  embeddings: " << embedPath << " | dim " << embedEngine->dim() << " | pooling "
              << qorvix::runtime::poolingName(embedEngine->defaultPooling()) << " | max-batch "
              << maxBatch << "\n";
  } else {
    std::cout << "  embeddings: none (pass --embed-model <file.gguf> to enable /v1/embeddings)\n";
  }
  std::cout << "  POST /v1/chat/completions   POST /v1/completions   POST /v1/embeddings\n"
            << "  GET /v1/models\n"
            << "  (Ctrl-C to stop)\n";

  auto handler = [&](const api::HttpRequest& req, api::HttpResponder& res) {
    if (req.method == "OPTIONS") {
      res.send(200, "text/plain", "");
      return;
    }
    if (req.method == "GET" && req.target == "/v1/models") {
      std::vector<std::string> ids{modelId};
      if (embedEngine) ids.push_back(embedModelId);
      res.send(200, "application/json", api::modelsResponse(ids).dump());
      return;
    }

    if (req.target == "/v1/embeddings") {
      if (req.method != "POST") {
        res.send(405, "application/json",
                 api::errorResponse("use POST for /v1/embeddings", "invalid_request_error").dump());
        return;
      }
      if (!embedEngine) {
        // 501, not 404: the route exists, this process just has no encoder loaded. 404 would tell
        // a client to stop trying rather than to start the server differently.
        res.send(501, "application/json",
                 api::errorResponse(
                     "no embedding model loaded — restart with --embed-model <file.gguf>",
                     "not_implemented")
                     .dump());
        return;
      }
      std::string perr;
      auto ebody = api::json::parse(req.body, &perr);
      if (!ebody) {
        res.send(400, "application/json", api::errorResponse("invalid JSON: " + perr).dump());
        return;
      }
      auto er = api::parseEmbeddingsRequest(*ebody, perr);
      if (!er.valid) {
        res.send(400, "application/json", api::errorResponse(perr).dump());
        return;
      }
      if (static_cast<int>(er.count()) > maxBatch) {
        // A 384-dim vector is ~5 KB of JSON, so an unbounded batch becomes a multi-megabyte
        // single send() on a server that has no chunked encoding.
        res.send(400, "application/json",
                 api::errorResponse("batch of " + std::to_string(er.count()) +
                                        " exceeds --max-batch " + std::to_string(maxBatch))
                     .dump());
        return;
      }

      // Tokenize outside the lock; only the forward pass needs exclusion.
      std::vector<std::vector<int>> batch;
      batch.reserve(er.count());
      const int cap = static_cast<int>(embedEngine->maxSeqLen());
      for (const auto& text : er.input) {
        batch.push_back(truncateForEncoder(embedTok->encode(text, true), cap,
                                           embedTok->special().sep));
      }
      for (const auto& ids : er.inputTokens) {
        batch.push_back(truncateForEncoder(ids, cap, embedTok->special().sep));
      }

      int promptTokens = 0;
      for (const auto& ids : batch) promptTokens += static_cast<int>(ids.size());

      std::vector<std::vector<float>> vecs;
      {
        std::lock_guard<std::mutex> lock(embedMutex);
        if (!embedEngine->embedBatch(batch, vecs, perr)) {
          res.send(400, "application/json", api::errorResponse(perr).dump());
          return;
        }
      }

      // Matryoshka truncation. Only meaningful for models trained for it, but clients send it
      // reflexively, so honour it rather than erroring.
      if (er.dimensions > 0 && er.dimensions < static_cast<int>(embedEngine->dim())) {
        for (auto& v : vecs) {
          v.resize(er.dimensions);
          if (embedEngine->defaultNormalize()) {
            qorvix::runtime::ops::l2Normalize(v.data(), static_cast<int>(v.size()));
          }
        }
      }

      const std::string respId = er.model.empty() ? embedModelId : er.model;
      res.send(200, "application/json",
               api::embeddingsResponse(respId, vecs, promptTokens, er.encodingFormat == "base64")
                   .dump());
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
      prompt = api::buildChatPromptWithTemplate(cr.messages, chatTemplate, eosPiece);
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
      std::vector<qorvix::scheduler::RequestResult> results;
      {
        std::lock_guard<std::mutex> lock(schedMutex);
        sched.submit(prompt, rp, [&](qorvix::scheduler::RequestId, const std::string& piece) {
          res.writeChunk(api::sseData(isChat ? api::chatChunk(id, respModel, piece, false, "")
                                             : api::completionChunk(id, respModel, piece, "")));
        });
        results = sched.runToCompletion();
      }
      const bool eos = !results.empty() && results.front().hitEos;
      const std::string finish = eos ? "stop" : "length";
      res.writeChunk(api::sseData(isChat ? api::chatChunk(id, respModel, "", false, finish)
                                         : api::completionChunk(id, respModel, "", finish)));
      res.writeChunk(api::sseDone());
      res.endStream();
    } else {
      std::vector<qorvix::scheduler::RequestResult> results;
      {
        std::lock_guard<std::mutex> lock(schedMutex);
        sched.submit(prompt, rp);
        results = sched.runToCompletion();
      }
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

// Correctness gate for GPU inference: runs one forward pass on both the CPU TextModel and the GPU
// model over a short prompt and compares logits.
int cmdGpuCheck(const std::string& path) {
  namespace rt = qorvix::runtime;
  namespace cu = qorvix::cuda;
  if (!cu::builtWithCuda()) {
    std::cout << "CUDA not built in — rebuild with -DQORVIX_ENABLE_CUDA=ON (needs a GPU host).\n";
    return 0;
  }
  if (path.empty()) {
    std::cerr << "usage: qorvix gpu-check <file.gguf>\n";
    return 1;
  }
  try {
    auto file = qorvix::gguf::GgufFile::open(path);  // kept alive: CPU weights borrow its mmap
    std::string err;
    auto tok = qorvix::tokenizer::Tokenizer::fromGguf(file, err);
    if (!tok) { std::cerr << "tokenizer: " << err << "\n"; return 1; }
    const auto cfg = rt::configFromGguf(file, err);
    if (!cfg.valid()) { std::cerr << "config: " << err << "\n"; return 1; }
    auto weights = rt::loadWeights(file, cfg, err);
    if (!weights) { std::cerr << "weights: " << err << "\n"; return 1; }

    const int vocab = static_cast<int>(cfg.vocabSize);
    const int maxSeq = 64;

    std::string gerr;
    std::cout << "Uploading weights to VRAM and building GPU model...\n";
    auto gpu = qorvix::buildGpuModel(cfg, *weights, maxSeq, gerr);
    if (!gpu) { std::cerr << "GPU model: " << gerr << "\n"; return 1; }

    rt::TextModel cpu(cfg, std::move(*weights), maxSeq);  // in-memory ctor; borrows the live file

    const auto ids = tok->encode("The capital of France is", true);
    std::cout << "Comparing GPU vs CPU logits over " << ids.size() << " prompt tokens...\n";
    float maxErr = 0.0f, maxRef = 1e-6f;
    bool argmaxMatch = true;
    for (int pos = 0; pos < static_cast<int>(ids.size()) && pos < maxSeq; ++pos) {
      const auto& cl = cpu.forward(ids[pos], pos);
      const auto& glog = gpu->forward(ids[pos], pos);
      for (int i = 0; i < vocab; ++i) {
        maxErr = std::max(maxErr, std::fabs(cl[i] - glog[i]));
        maxRef = std::max(maxRef, std::fabs(cl[i]));
      }
      if (argmaxOf(cl) != argmaxOf(glog)) argmaxMatch = false;
    }
    const float relErr = maxErr / maxRef;
    std::cout << "\nGPU vs CPU logits:  max abs err " << maxErr << ", rel err " << relErr << "\n"
              << "Argmax agrees at every position: " << (argmaxMatch ? "yes" : "NO") << "\n"
              << (argmaxMatch && relErr < 5e-2f ? "RESULT: PASS - GPU forward matches the CPU runtime.\n"
                                                : "RESULT: MISMATCH - see errors above.\n");
    return (argmaxMatch && relErr < 5e-2f) ? 0 : 1;
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

// Vulkan twin of gpu-check: builds the CPU model and the Vulkan model from the same GGUF, runs both
// over a short prompt, and checks argmax parity. Works on any Vulkan device — including Mesa's
// software device (lavapipe) — so it is meaningful without a discrete GPU.
int cmdVulkanCheck(const std::string& path) {
  namespace rt = qorvix::runtime;
  namespace vk = qorvix::vulkan;
  if (!vk::builtWithVulkan()) {
    std::cout << "Vulkan not built in — rebuild with -DQORVIX_ENABLE_VULKAN=ON.\n";
    return 0;
  }
  if (path.empty()) {
    std::cerr << "usage: qorvix vulkan-check <file.gguf>\n";
    return 1;
  }
  try {
    auto file = qorvix::gguf::GgufFile::open(path);  // kept alive: CPU weights borrow its mmap
    std::string err;
    auto tok = qorvix::tokenizer::Tokenizer::fromGguf(file, err);
    if (!tok) { std::cerr << "tokenizer: " << err << "\n"; return 1; }
    const auto cfg = rt::configFromGguf(file, err);
    if (!cfg.valid()) { std::cerr << "config: " << err << "\n"; return 1; }
    auto weights = rt::loadWeights(file, cfg, err);
    if (!weights) { std::cerr << "weights: " << err << "\n"; return 1; }

    const int vocab = static_cast<int>(cfg.vocabSize);
    const int maxSeq = 64;

    std::string gerr;
    std::cout << "Uploading weights to device buffers and building Vulkan model...\n";
    auto gpu = qorvix::buildVulkanModel(cfg, *weights, maxSeq, gerr);
    if (!gpu) { std::cerr << "Vulkan model: " << gerr << "\n"; return 1; }

    rt::TextModel cpu(cfg, std::move(*weights), maxSeq);  // in-memory ctor; borrows the live file

    const auto ids = tok->encode("The capital of France is", true);
    std::cout << "Comparing Vulkan vs CPU logits over " << ids.size() << " prompt tokens...\n";
    float maxErr = 0.0f, maxRef = 1e-6f;
    bool argmaxMatch = true;
    for (int pos = 0; pos < static_cast<int>(ids.size()) && pos < maxSeq; ++pos) {
      const auto& cl = cpu.forward(ids[pos], pos);
      const auto& glog = gpu->forward(ids[pos], pos);
      for (int i = 0; i < vocab; ++i) {
        maxErr = std::max(maxErr, std::fabs(cl[i] - glog[i]));
        maxRef = std::max(maxRef, std::fabs(cl[i]));
      }
      if (argmaxOf(cl) != argmaxOf(glog)) argmaxMatch = false;
    }
    const float relErr = maxErr / maxRef;
    std::cout << "\nVulkan vs CPU logits:  max abs err " << maxErr << ", rel err " << relErr << "\n"
              << "Argmax agrees at every position: " << (argmaxMatch ? "yes" : "NO") << "\n"
              << (argmaxMatch && relErr < 5e-2f
                      ? "RESULT: PASS - Vulkan forward matches the CPU runtime.\n"
                      : "RESULT: MISMATCH - see errors above.\n");
    return (argmaxMatch && relErr < 5e-2f) ? 0 : 1;
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
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
  const auto q4k = qorvix::cuda::qmatmulQ4_KSelfTest();
  std::cout << "Self-test (Q4_K matmul):  " << (q4k.passed ? "PASS" : (q4k.ran ? "FAIL" : "skip"))
            << " - " << q4k.message << "\n";
  const auto q6k = qorvix::cuda::qmatmulQ6_KSelfTest();
  std::cout << "Self-test (Q6_K matmul):  " << (q6k.passed ? "PASS" : (q6k.ran ? "FAIL" : "skip"))
            << " - " << q6k.message << "\n";
  const auto ops = qorvix::cuda::opsSelfTest();
  std::cout << "Self-test (forward ops):  " << (ops.passed ? "PASS" : (ops.ran ? "FAIL" : "skip"))
            << " - " << ops.message << "\n";
  const auto attn = qorvix::cuda::attentionSelfTest();
  std::cout << "Self-test (attention):    " << (attn.passed ? "PASS" : (attn.ran ? "FAIL" : "skip"))
            << " - " << attn.message << "\n";
  const auto fwd = qorvix::cuda::gpuForwardSelfTest();
  std::cout << "Self-test (GPU forward):  " << (fwd.passed ? "PASS" : (fwd.ran ? "FAIL" : "skip"))
            << " - " << fwd.message << "\n";
  const auto tp = qorvix::cuda::tensorParallelSelfTest();
  std::cout << "Self-test (tensor-par):   " << (tp.passed ? "PASS" : (tp.ran ? "FAIL" : "skip"))
            << " - " << tp.message << "\n";

  // Multi-GPU topology (Phase 10). With one device this reports a world size of 1, which is the
  // honest answer — tensor parallelism needs >= 2 devices to do anything, but the sharding math
  // above is verified regardless.
  const auto topo = qorvix::cuda::queryTopology();
  std::cout << "\nTopology: " << topo.deviceCount << " device(s)"
            << ", NCCL " << (qorvix::cuda::builtWithNccl() ? "built in" : "not built in") << "\n";
  if (topo.deviceCount > 1) {
    std::cout << "  peer access matrix (rows = src, cols = dst; . none, P pcie, N nvlink):\n";
    for (int a = 0; a < topo.deviceCount; ++a) {
      std::cout << "    [" << a << "] ";
      for (int b = 0; b < topo.deviceCount; ++b) {
        const auto l = topo.link(a, b);
        std::cout << (l == qorvix::cuda::PeerLink::Nvlink   ? 'N'
                      : l == qorvix::cuda::PeerLink::Pcie   ? 'P'
                                                            : '.');
      }
      std::cout << "\n";
    }
    std::cout << "  fully connected: " << (topo.fullyConnected(topo.deviceCount) ? "yes" : "no")
              << " | min free VRAM: " << humanBytes(topo.minFreeMem)
              << " | aggregate free: " << humanBytes(topo.totalFreeMem) << "\n";
  }

  return (self.ran && !self.passed) || (gemm.ran && !gemm.passed) || (qmm.ran && !qmm.passed) ||
                 (q4k.ran && !q4k.passed) || (q6k.ran && !q6k.passed) || (ops.ran && !ops.passed) ||
                 (attn.ran && !attn.passed) || (fwd.ran && !fwd.passed) || (tp.ran && !tp.passed)
             ? 1
             : 0;
}

// Vulkan compute backend: devices + self-tests. The Vulkan analogue of `qorvix gpu`. This backend
// runs on any vendor (NVIDIA, AMD, Apple via MoltenVK, Intel) and its correctness is checked
// headlessly on Mesa's software device (lavapipe), so this command is meaningful even on a host
// with no discrete GPU.
int cmdVulkan() {
  if (!qorvix::vulkan::builtWithVulkan()) {
    std::cout << "Vulkan support: not built in.\n"
              << "Rebuild with -DQORVIX_ENABLE_VULKAN=ON (needs the Vulkan loader + glslang) to "
                 "enable the cross-vendor GPU backend.\n";
    return 0;
  }

  const int count = qorvix::vulkan::deviceCount();
  std::cout << "Vulkan support: built in.\n";
  const auto devs = qorvix::vulkan::enumerateDevices();
  std::cout << "Physical devices: " << devs.size() << " (usable compute: " << count << ")\n";
  static const char* kType[] = {"other", "integrated-gpu", "discrete-gpu", "virtual-gpu", "cpu"};
  for (const auto& d : devs) {
    const char* t = (d.deviceType >= 0 && d.deviceType <= 4) ? kType[d.deviceType] : "other";
    // Decode the packed VkVersion (major:10 | minor:10 | patch:12) without pulling in vulkan.h,
    // which the facade keeps out of this translation unit so the no-Vulkan build still compiles.
    const unsigned vMaj = d.apiVersion >> 22, vMin = (d.apiVersion >> 12) & 0x3FF,
                   vPat = d.apiVersion & 0xFFF;
    std::cout << "\n  [" << d.index << "] " << d.name << "\n"
              << "      type               : " << t << "\n"
              << "      Vulkan API         : " << vMaj << "." << vMin << "." << vPat << "\n"
              << "      vendor/device ID   : 0x" << std::hex << d.vendorID << " / 0x" << d.deviceID
              << std::dec << "\n"
              << "      device-local mem   : " << humanBytes(d.deviceLocalMem) << "\n";
  }
  if (devs.empty()) {
    std::cout << "No Vulkan devices detected on this host.\n";
    return 0;
  }

  const auto self = qorvix::vulkan::selfTest();
  std::cout << "\nSelf-test (compute):      " << (self.passed ? "PASS" : (self.ran ? "FAIL" : "skip"))
            << " - " << self.message << "\n";
  const auto q8 = qorvix::vulkan::qmatmulQ8_0SelfTest();
  std::cout << "Self-test (Q8_0 matmul):  " << (q8.passed ? "PASS" : (q8.ran ? "FAIL" : "skip"))
            << " - " << q8.message << "\n";
  const auto q4k = qorvix::vulkan::qmatmulQ4_KSelfTest();
  std::cout << "Self-test (Q4_K matmul):  " << (q4k.passed ? "PASS" : (q4k.ran ? "FAIL" : "skip"))
            << " - " << q4k.message << "\n";
  const auto q6k = qorvix::vulkan::qmatmulQ6_KSelfTest();
  std::cout << "Self-test (Q6_K matmul):  " << (q6k.passed ? "PASS" : (q6k.ran ? "FAIL" : "skip"))
            << " - " << q6k.message << "\n";
  const auto ops = qorvix::vulkan::opsSelfTest();
  std::cout << "Self-test (forward ops):  " << (ops.passed ? "PASS" : (ops.ran ? "FAIL" : "skip"))
            << " - " << ops.message << "\n";
  const auto attn = qorvix::vulkan::attentionSelfTest();
  std::cout << "Self-test (attention):    " << (attn.passed ? "PASS" : (attn.ran ? "FAIL" : "skip"))
            << " - " << attn.message << "\n";
  const auto fwd = qorvix::vulkan::forwardSelfTest();
  std::cout << "Self-test (forward pass): " << (fwd.passed ? "PASS" : (fwd.ran ? "FAIL" : "skip"))
            << " - " << fwd.message << "\n";
  const auto ms = qorvix::vulkan::multiSessionSelfTest();
  std::cout << "Self-test (multi-session):" << (ms.passed ? "PASS" : (ms.ran ? "FAIL" : "skip"))
            << " - " << ms.message << "\n";

  return (self.ran && !self.passed) || (q8.ran && !q8.passed) || (q4k.ran && !q4k.passed) ||
                 (q6k.ran && !q6k.passed) || (ops.ran && !ops.passed) || (attn.ran && !attn.passed) ||
                 (fwd.ran && !fwd.passed) || (ms.ran && !ms.passed)
             ? 1
             : 0;
}

// Reports every compute backend and whether it is usable, plus which one `--auto` would pick.
// Availability/selection logic lives once in qorvix/backend.hpp (backendAvailable/selectBestBackend).
int cmdBackends() {
  const bool cudaBuilt = qorvix::cuda::builtWithCuda();
  const int cudaDevs = cudaBuilt ? qorvix::cuda::deviceCount() : 0;
  const bool vkBuilt = qorvix::vulkan::builtWithVulkan();
  const int vkDevs = vkBuilt ? qorvix::vulkan::deviceCount() : 0;
  auto row = [](const char* name, bool built, int devs, const char* note) {
    std::cout << "  " << std::left << std::setw(8) << name
              << (!built ? "not built in" : devs > 0 ? "available" : "built in, no device")
              << "   " << note << "\n";
  };
  std::cout << "Compute backends (one unified IInferenceEngine seam; createEngine picks one):\n";
  row("cpu", true, 1, "all CPUs           (generalized reference)");
  row("vulkan", vkBuilt, vkDevs, "all GPUs           (NVIDIA / AMD / Apple / Intel — one backend)");
  row("cuda", cudaBuilt, cudaDevs, "NVIDIA GPUs only   (the fast path where present)");
  std::cout << "\nModel: one universal CPU backend + one universal Vulkan GPU backend, plus CUDA as\n"
               "the NVIDIA-only fast path. `--auto` selects the FASTEST available for this hardware\n"
               "(CUDA > Vulkan > CPU) -> "
            << qorvix::backendName(qorvix::selectBestBackend())
            << " here. Force with --cuda/--gpu, --vulkan, or --cpu.\n";
  return 0;
}

// Reproducible throughput benchmark over the unified engine seam — the single source of truth for
// every perf claim. Warmup + timed runs -> median, on whichever backend createEngine() picks.
// Encoder half of `qorvix bench`. Same warmup/median discipline and the same --json contract, but
// a different measurement core because the seams differ (see embed_benchmark.hpp).
int benchEncoder(const std::vector<std::string_view>& args, const std::string& path, bool json) {
  namespace emb = qorvix::embeddings;
  emb::EmbedBenchConfig bc;
  if (auto v = flagValue(args, "--seq"); !v.empty()) bc.seqTokens = std::stoi(v);
  if (auto v = flagValue(args, "--batch"); !v.empty()) bc.batch = std::stoi(v);
  if (auto v = flagValue(args, "--warmup"); !v.empty()) bc.warmupRuns = std::stoi(v);
  if (auto v = flagValue(args, "--runs"); !v.empty()) bc.timedRuns = std::stoi(v);

  using clock = std::chrono::steady_clock;
  const auto tLoad0 = clock::now();
  auto file = qorvix::gguf::GgufFile::open(path);
  std::string err;
  auto engine = qorvix::createEmbeddingEngine(qorvix::Backend::Cpu, std::move(file), 0, err);
  if (!engine) {
    std::cerr << "error: embedding engine: " << err << "\n";
    return 1;
  }
  const double loadSec = std::chrono::duration<double>(clock::now() - tLoad0).count();

  if (!json) {
    std::cerr << "benchmarking cpu encoder | seq " << bc.seqTokens << " batch " << bc.batch
              << " | warmup " << bc.warmupRuns << " runs " << bc.timedRuns << " ...\n";
  }
  auto r = emb::runEmbedBenchmark(*engine, bc);
  r.loadSec = loadSec;
  if (!r.ran) {
    std::cerr << "error: benchmark did not run\n";
    return 1;
  }

  if (json) {
    std::cout << std::fixed << std::setprecision(4) << "{"
              << "\"kind\":\"embed\",\"backend\":\"" << r.backend << "\",\"model\":\"" << path
              << "\",\"seq_tokens\":" << r.seqTokens << ",\"batch\":" << r.batch
              << ",\"runs\":" << r.timedRuns << ",\"load_sec\":" << r.loadSec
              << ",\"embed_seq_per_sec\":" << r.embedSeqPerSec
              << ",\"embed_tok_per_sec\":" << r.embedTokPerSec
              << ",\"ms_per_seq_median\":" << r.msPerSeqMedian
              << ",\"ms_per_seq_min\":" << r.msPerSeqMin
              << ",\"ms_per_seq_max\":" << r.msPerSeqMax << "}\n";
    return 0;
  }
  std::cout << std::fixed << std::setprecision(2) << "\n"
            << "  backend             " << r.backend << " (encoder)\n"
            << "  model               " << path << "\n"
            << "  sequence            " << r.seqTokens << " tokens x " << r.batch << "\n"
            << "  load                " << r.loadSec << " s\n"
            << "  embed_seq_per_sec   " << r.embedSeqPerSec << "\n"
            << "  embed_tok_per_sec   " << r.embedTokPerSec << "\n"
            << "  ms/seq (med/min/max)" << "  " << r.msPerSeqMedian << " / " << r.msPerSeqMin
            << " / " << r.msPerSeqMax << "\n";
  return 0;
}

int cmdBench(const std::vector<std::string_view>& args) {
  namespace rt = qorvix::runtime;
  const std::string path = args.size() > 1 ? std::string(args[1]) : std::string();
  if (path.empty()) {
    std::cerr << "usage: qorvix bench <file.gguf> [--gpu|--vulkan|--auto] [--prompt N] [--gen N] "
                 "[--warmup N] [--runs N] [--json]\n"
                 "       encoder models instead take [--seq N] [--batch N]\n";
    return 1;
  }
  // One CLI, two cores: dispatch on the model family so a user never has to know which is which.
  try {
    const auto probe = qorvix::gguf::GgufFile::open(path);
    std::string perr;
    const auto cfg = qorvix::runtime::configFromGguf(probe, perr);
    if (cfg.valid() && cfg.isEncoder()) {
      return benchEncoder(args, path, hasFlag(args, "--json"));
    }
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  const qorvix::Backend backend = backendFromArgs(args);
  const bool json = hasFlag(args, "--json");
  rt::BenchConfig bc;
  if (auto v = flagValue(args, "--prompt"); !v.empty()) bc.promptTokens = std::stoi(v);
  if (auto v = flagValue(args, "--gen"); !v.empty()) bc.genTokens = std::stoi(v);
  if (auto v = flagValue(args, "--warmup"); !v.empty()) bc.warmupRuns = std::stoi(v);
  if (auto v = flagValue(args, "--runs"); !v.empty()) bc.timedRuns = std::stoi(v);

  if (!qorvix::backendAvailable(backend)) {
    std::cerr << "error: " << qorvix::backendName(backend)
              << " backend requested but unavailable (not built in, or no device)\n";
    return 1;
  }

  try {
    using clock = std::chrono::steady_clock;
    const auto tLoad0 = clock::now();
    auto file = qorvix::gguf::GgufFile::open(path);
    std::string err;
    const std::uint32_t maxSeq = static_cast<std::uint32_t>(bc.promptTokens + bc.genTokens + 4);
    auto engine = qorvix::createEngine(backend, std::move(file), maxSeq, 1, err);
    if (!engine) {
      std::cerr << "error: " << qorvix::backendName(backend) << " engine: " << err << "\n";
      return 1;
    }
    const double loadSec = std::chrono::duration<double>(clock::now() - tLoad0).count();

    if (!json)
      std::cerr << "benchmarking " << qorvix::backendName(backend) << " | prompt " << bc.promptTokens
                << " gen " << bc.genTokens << " | warmup " << bc.warmupRuns << " runs " << bc.timedRuns
                << " ...\n";
    auto r = rt::runBenchmark(*engine, bc);
    r.loadSec = loadSec;
    if (!r.ran) {
      std::cerr << "error: benchmark did not run (engine capacity or empty config)\n";
      return 1;
    }

    if (json) {
      std::cout << "{\"backend\":\"" << r.backend << "\",\"prompt_tokens\":" << r.promptTokens
                << ",\"gen_tokens\":" << r.genTokens << ",\"timed_runs\":" << r.timedRuns
                << ",\"load_sec\":" << r.loadSec << ",\"prefill_tok_per_sec\":" << r.prefillTokPerSec
                << ",\"decode_tok_per_sec\":" << r.decodeTokPerSec
                << ",\"decode_ms_per_tok_median\":" << r.decodeMsPerTokMedian
                << ",\"decode_ms_per_tok_min\":" << r.decodeMsPerTokMin
                << ",\"decode_ms_per_tok_max\":" << r.decodeMsPerTokMax << "}\n";
    } else {
      std::cout << std::fixed;
      std::cout << "\n==== qorvix bench: " << r.backend << " ====\n"
                << "  model          : " << path << "\n"
                << "  load           : " << std::setprecision(2) << r.loadSec << " s\n"
                << "  prompt / gen   : " << r.promptTokens << " / " << r.genTokens << " tokens\n"
                << "  prefill        : " << std::setprecision(1) << r.prefillTokPerSec << " tok/s\n"
                << "  decode         : " << std::setprecision(1) << r.decodeTokPerSec << " tok/s ("
                << std::setprecision(2) << r.decodeMsPerTokMedian << " ms/tok, min "
                << r.decodeMsPerTokMin << " max " << r.decodeMsPerTokMax << ")\n"
                << "  timed runs     : " << r.timedRuns << " (median reported)\n";
    }
    return 0;
  } catch (const qorvix::gguf::GgufParseError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

// Reports the CPU's detected ISA + which SIMD dot-product kernel the runtime dispatches to. The CPU
// analogue of `backends`/`vulkan`: one generalized backend, best kernel chosen at runtime.
int cmdCpuInfo() {
  const auto& f = qorvix::runtime::cpu::features();
  std::cout << "CPU backend (generalized — one binary, best SIMD kernel chosen at runtime):\n"
            << "  architecture     : " << f.arch << "\n"
            << "  hardware threads : " << f.hardwareThreads << "\n";
  std::cout << "  SIMD features    : ";
  if (f.arch == "x86_64") {
    std::cout << (f.sse2 ? "sse2 " : "") << (f.avx ? "avx " : "") << (f.avx2 ? "avx2 " : "")
              << (f.fma ? "fma " : "") << (f.avx512f ? "avx512f " : "");
  } else if (f.arch == "aarch64") {
    std::cout << (f.neon ? "neon " : "") << (f.sve ? "sve " : "");
  } else {
    std::cout << "(scalar only on this arch)";
  }
  std::cout << "\n  active dot kernel: " << qorvix::runtime::cpu::activeDotKernel() << "\n"
            << "\nOne portable build runs optimally on any CPU (no -march=native). x86 AVX2 is\n"
            << "active here; NEON/SVE (ARM/Apple), RISC-V vector, NUMA, and thread affinity are\n"
            << "architected but need their own hardware to verify (see ROADMAP).\n";
  return 0;
}

int printUsage() {
  std::cout << qorvix::startupBanner() << "\n\n"
            << "Usage: qorvix <command> [args]\n\n"
            << "Commands:\n"
            << "  scan-models [dir]   Scan a directory for model files (default: models)\n"
            << "  list [dir]          List discovered models (default: models)\n"
            << "  gguf-info <file>    Parse a GGUF file and print its header, metadata, tensors\n"
            << "  model-info <file>   Derive and print the model config from a GGUF file\n"
            << "  generate <file> --prompt \"...\" [--gpu|--vulkan|--auto]  Generate text\n"
            << "  embed <file> --text \"...\"       Embed text with an encoder model (bert)\n"
            << "                                  [--pooling mean|cls|last] [--no-normalize]\n"
            << "                                  [--dims N] [--json]\n"
            << "  serve <file> [--gpu|--vulkan|--auto] [--port N] [--embed-model <f.gguf>]\n"
            << "                                  OpenAI-compatible HTTP server\n"
            << "                                  (default port " << qorvix::ports::kRuntime
            << "; Qorvix reserves " << qorvix::ports::kRangeFirst << "-"
            << qorvix::ports::kRangeLast << ")\n"
            << "  backends            List compute backends (cpu/cuda/vulkan) and the auto default\n"
            << "  cpuinfo             Show CPU arch + SIMD features + active dot kernel\n"
            << "  bench <file> [--gpu|--vulkan|--auto] [--json]   Benchmark throughput\n"
            << "                      (decode for a decoder; --seq/--batch for an encoder)\n"
            << "  gpu                 Show CUDA devices and run backend self-tests\n"
            << "  vulkan              Show Vulkan devices and run compute-backend self-tests\n"
            << "  gpu-check <file>    Compare GPU vs CPU forward-pass logits for a GGUF model\n"
            << "  vulkan-check <file> Compare Vulkan vs CPU forward-pass logits for a GGUF model\n"
            << "  embed-check <file> [--ref F] [--min-cos C]   Gate embeddings against a\n"
            << "                      captured sentence-transformers reference\n"
            << "  rag index <dir> --embed-model <f.gguf> --store <out.qvx>\n"
            << "                      Chunk, embed and index documents (.txt/.md/.csv/.tsv)\n"
            << "                      [--chunk-tokens N] [--overlap N]\n"
            << "  rag search --store <x.qvx> --embed-model <f.gguf> --query \"...\"\n"
            << "                      Hybrid dense + BM25 retrieval  [--k N] [--alpha A]\n"
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
  if (command == "embed") return cmdEmbed(args);
  if (command == "embed-check") return cmdEmbedCheck(args);
  if (command == "image-embed") return cmdImageEmbed(args);
  if (command == "vision-check") return cmdVisionCheck(args);
  if (command == "rag") return cmdRag(args);
  if (command == "serve") return cmdServe(args);
  if (command == "gpu") return cmdGpu();
  if (command == "vulkan") return cmdVulkan();
  if (command == "backends") return cmdBackends();
  if (command == "cpuinfo") return cmdCpuInfo();
  if (command == "bench") return cmdBench(args);
  if (command == "gpu-check") return cmdGpuCheck(arg1);
  if (command == "vulkan-check") return cmdVulkanCheck(arg1);
  if (command == "plugins") return cmdPlugins(arg1.empty() ? "plugins" : arg1);

  std::cerr << "Unknown command: " << command << "\n\n";
  printUsage();
  return 1;
}
