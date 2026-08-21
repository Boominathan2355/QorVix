#include "qorvix/agents/builtin_tools.hpp"

#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#include "qorvix/agents/artifact_tools.hpp"
#include "qorvix/agents/blackboard.hpp"
#include "qorvix/rag/pipeline.hpp"
#include "qorvix/runtime/cpu_features.hpp"
#include "qorvix/version.hpp"

namespace qorvix::agents {

namespace {

// Safe recursive-descent math parser for calculator
class ExprParser {
 public:
  explicit ExprParser(std::string_view s) : src_(s), pos_(0) {}

  double parse(std::string& err) {
    skipWs();
    if (pos_ >= src_.size()) {
      err = "Empty expression";
      return 0.0;
    }
    double res = parseExpr(err);
    if (!err.empty()) return 0.0;
    skipWs();
    if (pos_ < src_.size()) {
      err = "Unexpected character: '" + std::string(1, src_[pos_]) + "'";
      return 0.0;
    }
    return res;
  }

 private:
  void skipWs() {
    while (pos_ < src_.size() && std::isspace(static_cast<unsigned char>(src_[pos_]))) {
      ++pos_;
    }
  }

  double parseExpr(std::string& err) {
    double v = parseTerm(err);
    if (!err.empty()) return 0.0;
    while (true) {
      skipWs();
      if (pos_ < src_.size() && src_[pos_] == '+') {
        ++pos_;
        double rhs = parseTerm(err);
        if (!err.empty()) return 0.0;
        v += rhs;
      } else if (pos_ < src_.size() && src_[pos_] == '-') {
        ++pos_;
        double rhs = parseTerm(err);
        if (!err.empty()) return 0.0;
        v -= rhs;
      } else {
        break;
      }
    }
    return v;
  }

  double parseTerm(std::string& err) {
    double v = parsePower(err);
    if (!err.empty()) return 0.0;
    while (true) {
      skipWs();
      if (pos_ < src_.size() && src_[pos_] == '*') {
        ++pos_;
        double rhs = parsePower(err);
        if (!err.empty()) return 0.0;
        v *= rhs;
      } else if (pos_ < src_.size() && src_[pos_] == '/') {
        ++pos_;
        double rhs = parsePower(err);
        if (!err.empty()) return 0.0;
        if (std::abs(rhs) < 1e-15) {
          err = "Division by zero";
          return 0.0;
        }
        v /= rhs;
      } else if (pos_ < src_.size() && src_[pos_] == '%') {
        ++pos_;
        double rhs = parsePower(err);
        if (!err.empty()) return 0.0;
        if (std::abs(rhs) < 1e-15) {
          err = "Modulo by zero";
          return 0.0;
        }
        v = std::fmod(v, rhs);
      } else {
        break;
      }
    }
    return v;
  }

  double parsePower(std::string& err) {
    double v = parseFactor(err);
    if (!err.empty()) return 0.0;
    skipWs();
    if (pos_ < src_.size() && src_[pos_] == '^') {
      ++pos_;
      double exp = parsePower(err);
      if (!err.empty()) return 0.0;
      v = std::pow(v, exp);
    }
    return v;
  }

  double parseFactor(std::string& err) {
    skipWs();
    if (pos_ >= src_.size()) {
      err = "Unexpected end of expression";
      return 0.0;
    }

    if (src_[pos_] == '+') {
      ++pos_;
      return parseFactor(err);
    }
    if (src_[pos_] == '-') {
      ++pos_;
      return -parseFactor(err);
    }

    if (src_[pos_] == '(') {
      ++pos_;
      double v = parseExpr(err);
      if (!err.empty()) return 0.0;
      skipWs();
      if (pos_ >= src_.size() || src_[pos_] != ')') {
        err = "Missing closing parenthesis";
        return 0.0;
      }
      ++pos_;
      return v;
    }

    if (std::isdigit(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '.') {
      std::size_t start = pos_;
      while (pos_ < src_.size() && (std::isdigit(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '.')) {
        ++pos_;
      }
      try {
        return std::stod(std::string(src_.substr(start, pos_ - start)));
      } catch (...) {
        err = "Invalid number format";
        return 0.0;
      }
    }

    if (std::isalpha(static_cast<unsigned char>(src_[pos_]))) {
      std::size_t start = pos_;
      while (pos_ < src_.size() && std::isalpha(static_cast<unsigned char>(src_[pos_]))) {
        ++pos_;
      }
      std::string fn(src_.substr(start, pos_ - start));
      if (fn == "pi") return 3.14159265358979323846;
      if (fn == "e") return 2.71828182845904523536;

      skipWs();
      if (pos_ < src_.size() && src_[pos_] == '(') {
        ++pos_;
        double arg = parseExpr(err);
        if (!err.empty()) return 0.0;
        skipWs();
        if (pos_ >= src_.size() || src_[pos_] != ')') {
          err = "Missing ')' after function argument";
          return 0.0;
        }
        ++pos_;
        if (fn == "sqrt") {
          if (arg < 0) { err = "sqrt of negative number"; return 0.0; }
          return std::sqrt(arg);
        }
        if (fn == "abs") return std::abs(arg);
        if (fn == "sin") return std::sin(arg);
        if (fn == "cos") return std::cos(arg);
        if (fn == "tan") return std::tan(arg);
        if (fn == "log") {
          if (arg <= 0) { err = "log of non-positive number"; return 0.0; }
          return std::log(arg);
        }
        if (fn == "exp") return std::exp(arg);
        if (fn == "floor") return std::floor(arg);
        if (fn == "ceil") return std::ceil(arg);
        err = "Unknown function: " + fn;
        return 0.0;
      }
      err = "Unknown constant or function: " + fn;
      return 0.0;
    }

    err = "Unexpected character in expression: '" + std::string(1, src_[pos_]) + "'";
    return 0.0;
  }

  std::string_view src_;
  std::size_t pos_;
};

}  // namespace

std::shared_ptr<ITool> createCalculatorTool() {
  ToolDefinition def;
  def.name = "calculator";
  def.description = "Evaluates a mathematical or arithmetic expression safely. Supports +, -, *, /, ^, %, sqrt, abs, sin, cos, tan, log, exp, pi, e.";
  def.category = "math";
  def.parameters = {
      {"expression", ToolParamType::String, "The mathematical expression to evaluate (e.g. 'sqrt(144) + 2 * (3.5 + 4)')", true, {}}
  };

  return std::make_shared<FunctionTool>(std::move(def), [](const api::json::Value& args) -> ToolResult {
    const auto* expr = args.get("expression");
    if (!expr || !expr->isString()) {
      return ToolResult::makeError("Missing required string parameter 'expression'");
    }
    std::string err;
    ExprParser parser(expr->asString());
    double val = parser.parse(err);
    if (!err.empty()) {
      return ToolResult::makeError(err);
    }
    std::ostringstream ss;
    if (std::floor(val) == val && std::abs(val) < 1e15) {
      ss << static_cast<long long>(val);
    } else {
      ss << val;
    }
    auto d = api::json::Value::object();
    d["result"] = val;
    d["formatted"] = ss.str();
    return ToolResult::makeSuccess(ss.str(), d);
  });
}

std::shared_ptr<ITool> createFileOpsTool(std::filesystem::path workspaceRoot) {
  if (workspaceRoot.empty()) {
    workspaceRoot = std::filesystem::current_path();
  }

  ToolDefinition def;
  def.name = "file_ops";
  def.description = "Perform file operations within the workspace: read, write, or list files.";
  def.category = "system";
  def.parameters = {
      {"action", ToolParamType::String, "Operation to perform: 'read', 'write', 'list'", true, {"read", "write", "list"}},
      {"path", ToolParamType::String, "Relative path to target file or directory", true, {}},
      {"content", ToolParamType::String, "File content to write (required for action='write')", false, {}}
  };

  return std::make_shared<FunctionTool>(std::move(def), [root = workspaceRoot](const api::json::Value& args) -> ToolResult {
    const auto* act = args.get("action");
    const auto* relPath = args.get("path");
    if (!act || !act->isString() || !relPath || !relPath->isString()) {
      return ToolResult::makeError("Missing required parameters 'action' or 'path'");
    }
    std::string action = act->asString();
    std::filesystem::path target = root / std::filesystem::path(relPath->asString()).relative_path();

    if (action == "read") {
      if (!std::filesystem::exists(target)) {
        return ToolResult::makeError("File does not exist: " + relPath->asString());
      }
      if (std::filesystem::is_directory(target)) {
        return ToolResult::makeError("Path is a directory, not a file: " + relPath->asString());
      }
      std::ifstream fin(target, std::ios::binary);
      if (!fin) {
        return ToolResult::makeError("Failed to open file for reading: " + relPath->asString());
      }
      std::ostringstream ss;
      ss << fin.rdbuf();
      auto d = api::json::Value::object();
      d["size_bytes"] = static_cast<int>(ss.str().size());
      return ToolResult::makeSuccess(ss.str(), d);
    }

    if (action == "write") {
      const auto* cont = args.get("content");
      if (!cont || !cont->isString()) {
        return ToolResult::makeError("Missing required string parameter 'content' for write action");
      }
      try {
        if (target.has_parent_path()) {
          std::filesystem::create_directories(target.parent_path());
        }
        std::ofstream fout(target, std::ios::binary | std::ios::trunc);
        if (!fout) {
          return ToolResult::makeError("Failed to open file for writing: " + relPath->asString());
        }
        fout << cont->asString();
        auto d = api::json::Value::object();
        d["written_bytes"] = static_cast<int>(cont->asString().size());
        return ToolResult::makeSuccess("Successfully wrote " + std::to_string(cont->asString().size()) + " bytes to " + relPath->asString(), d);
      } catch (const std::exception& ex) {
        return ToolResult::makeError(std::string("Filesystem error: ") + ex.what());
      }
    }

    if (action == "list") {
      if (!std::filesystem::exists(target)) {
        return ToolResult::makeError("Directory does not exist: " + relPath->asString());
      }
      if (!std::filesystem::is_directory(target)) {
        return ToolResult::makeError("Path is not a directory: " + relPath->asString());
      }
      auto files = api::json::Value::array();
      std::ostringstream ss;
      for (const auto& entry : std::filesystem::directory_iterator(target)) {
        auto item = api::json::Value::object();
        item["name"] = entry.path().filename().string();
        item["is_directory"] = entry.is_directory();
        if (!entry.is_directory()) {
          item["size_bytes"] = static_cast<int>(entry.file_size());
        }
        files.push(std::move(item));
        ss << entry.path().filename().string() << (entry.is_directory() ? "/" : "") << "\n";
      }
      auto d = api::json::Value::object();
      d["entries"] = std::move(files);
      return ToolResult::makeSuccess(ss.str(), d);
    }

    return ToolResult::makeError("Unknown action: '" + action + "'");
  });
}

std::shared_ptr<ITool> createBlackboardTool(std::shared_ptr<Blackboard> blackboard) {
  ToolDefinition def;
  def.name = "blackboard";
  def.description = "Inspect or update key-value state and task progress on the shared multi-agent blackboard.";
  def.category = "agent_state";
  def.parameters = {
      {"action", ToolParamType::String, "Action: 'get', 'set', 'list_keys', 'post_message'", true, {"get", "set", "list_keys", "post_message"}},
      {"key", ToolParamType::String, "State key name (for 'get'/'set')", false, {}},
      {"value", ToolParamType::String, "State value string or JSON (for 'set')", false, {}},
      {"message", ToolParamType::String, "Message text to broadcast to agents (for 'post_message')", false, {}}
  };

  return std::make_shared<FunctionTool>(std::move(def), [bb = blackboard](const api::json::Value& args) -> ToolResult {
    if (!bb) {
      return ToolResult::makeError("No active blackboard attached to workflow");
    }
    const auto* act = args.get("action");
    if (!act || !act->isString()) {
      return ToolResult::makeError("Missing required parameter 'action'");
    }
    std::string action = act->asString();

    if (action == "get") {
      const auto* k = args.get("key");
      if (!k || !k->isString()) return ToolResult::makeError("Missing 'key' parameter for get action");
      auto val = bb->getState(k->asString());
      if (val.isNull()) {
        return ToolResult::makeError("Key not found: " + k->asString());
      }
      return ToolResult::makeSuccess(val.isString() ? val.asString() : val.dump(), val);
    }

    if (action == "set") {
      const auto* k = args.get("key");
      const auto* v = args.get("value");
      if (!k || !k->isString() || !v) return ToolResult::makeError("Missing 'key' or 'value' parameter for set action");
      
      api::json::Value parsedVal;
      if (v->isString()) {
        auto parsed = api::json::parse(v->asString());
        parsedVal = parsed.has_value() ? *parsed : *v;
      } else {
        parsedVal = *v;
      }
      bb->setState(k->asString(), parsedVal);
      return ToolResult::makeSuccess("State updated for key: " + k->asString());
    }

    if (action == "list_keys") {
      auto keys = bb->listKeys();
      auto arr = api::json::Value::array();
      std::ostringstream ss;
      for (const auto& k : keys) {
        arr.push(k);
        ss << "- " << k << "\n";
      }
      auto d = api::json::Value::object();
      d["keys"] = std::move(arr);
      return ToolResult::makeSuccess(ss.str(), d);
    }

    if (action == "post_message") {
      const auto* msg = args.get("message");
      if (!msg || !msg->isString()) return ToolResult::makeError("Missing 'message' parameter");
      bb->postMessage("agent", msg->asString());
      return ToolResult::makeSuccess("Message posted to blackboard timeline");
    }

    return ToolResult::makeError("Unknown action: " + action);
  });
}

std::shared_ptr<ITool> createRagSearchTool(const rag::Index* index,
                                          embeddings::IEmbeddingEngine* engine,
                                          const tokenizer::Tokenizer* tok) {
  ToolDefinition def;
  def.name = "rag_search";
  def.description = "Perform hybrid lexical (BM25) and dense semantic search over indexed documents in the knowledge base.";
  def.category = "retrieval";
  def.parameters = {
      {"query", ToolParamType::String, "The search query or factual question", true, {}},
      {"top_k", ToolParamType::Integer, "Maximum number of chunks to return (default 5)", false, {}}
  };

  return std::make_shared<FunctionTool>(std::move(def), [index, engine, tok](const api::json::Value& args) -> ToolResult {
    if (!index || !engine || !tok) {
      return ToolResult::makeError("RAG search engine or index is uninitialized");
    }
    const auto* q = args.get("query");
    if (!q || !q->isString()) {
      return ToolResult::makeError("Missing required parameter 'query'");
    }
    int topK = 5;
    if (const auto* k = args.get("top_k"); k && k->isNumber()) {
      topK = std::max(1, k->asInt());
    }

    rag::HybridOptions opt;
    opt.topK = topK;
    std::vector<rag::SearchHit> hits;
    std::string err;
    if (!rag::queryIndex(*index, *engine, *tok, q->asString(), opt, hits, err)) {
      return ToolResult::makeError("RAG query failed: " + err);
    }

    std::ostringstream ss;
    auto dataHits = api::json::Value::array();
    for (std::size_t i = 0; i < hits.size(); ++i) {
      const auto& hit = hits[i];
      if (hit.index >= index->store.chunks().size()) continue;
      const auto& chunk = index->store.chunks()[hit.index];

      auto item = api::json::Value::object();
      item["score"] = static_cast<double>(hit.score);
      item["source"] = chunk.source;
      item["text"] = chunk.text;
      dataHits.push(std::move(item));

      ss << "[" << (i + 1) << "] (Score: " << hit.score << " | Source: " << chunk.source << ")\n"
         << chunk.text << "\n\n";
    }

    auto d = api::json::Value::object();
    d["hits"] = std::move(dataHits);
    return ToolResult::makeSuccess(ss.str().empty() ? "No matching documents found." : ss.str(), d);
  });
}

std::shared_ptr<ITool> createSystemInfoTool() {
  ToolDefinition def;
  def.name = "system_info";
  def.description = "Inspect system telemetry, CPU features, hardware threads, and QorVix runtime version.";
  def.category = "system";

  return std::make_shared<FunctionTool>(std::move(def), [](const api::json::Value&) -> ToolResult {
    auto d = api::json::Value::object();
    d["qorvix_version"] = std::string(kVersionString);
    d["hardware_concurrency"] = static_cast<int>(std::thread::hardware_concurrency());
    
    auto cpu = api::json::Value::object();
    cpu["avx2"] = runtime::hasAvx2();
    cpu["avx512"] = runtime::hasAvx512F();
    d["cpu_features"] = std::move(cpu);

    std::ostringstream ss;
    ss << "QorVix Runtime v" << kVersionString << "\n"
       << "CPU Hardware Threads: " << std::thread::hardware_concurrency() << "\n"
       << "AVX2: " << (runtime::hasAvx2() ? "yes" : "no") << " | AVX-512: " << (runtime::hasAvx512F() ? "yes" : "no");

    return ToolResult::makeSuccess(ss.str(), d);
  });
}

std::shared_ptr<ITool> createMcpBridgeTool(api::mcp::McpTool mcpTool,
                                          std::function<api::mcp::McpToolCallResult(const api::mcp::McpToolCallRequest&)> dispatcher) {
  ToolDefinition def;
  def.name = mcpTool.name;
  def.description = mcpTool.description;
  def.category = "mcp_extension";

  return std::make_shared<FunctionTool>(std::move(def), [name = mcpTool.name, disp = std::move(dispatcher)](const api::json::Value& args) -> ToolResult {
    if (!disp) {
      return ToolResult::makeError("MCP dispatcher callback is not connected");
    }
    api::mcp::McpToolCallRequest req;
    req.name = name;
    req.arguments = args;
    api::mcp::McpToolCallResult res = disp(req);
    if (res.isError) {
      std::string errText = "MCP Tool Error";
      if (!res.content.empty() && !res.content[0].text.empty()) {
        errText = res.content[0].text;
      }
      return ToolResult::makeError(errText);
    }
    std::ostringstream ss;
    for (const auto& c : res.content) {
      if (c.type == "text" || c.type.empty()) {
        ss << c.text;
      }
    }
    return ToolResult::makeSuccess(ss.str());
  });
}

void registerDefaultTools(ToolRegistry& registry,
                         const std::filesystem::path& workspaceRoot,
                         std::shared_ptr<Blackboard> blackboard) {
  registry.registerTool(createCalculatorTool());
  registry.registerTool(createFileOpsTool(workspaceRoot));
  registry.registerTool(createSystemInfoTool());
  if (blackboard) {
    registry.registerTool(createBlackboardTool(blackboard));
    if (blackboard->artifactStore()) {
      registerArtifactTools(registry, blackboard->artifactStore());
    }
  }
}

}  // namespace qorvix::agents
