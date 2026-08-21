#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "qorvix/api/json.hpp"
#include "qorvix/api/openai.hpp"

namespace qorvix::agents {

enum class ToolParamType {
  String,
  Number,
  Integer,
  Boolean,
  Array,
  Object
};

std::string_view toolParamTypeName(ToolParamType type);
ToolParamType parseToolParamType(std::string_view name);

struct ToolParameter {
  std::string name;
  ToolParamType type = ToolParamType::String;
  std::string description;
  bool required = false;
  std::vector<std::string> enumValues;

  api::json::Value toJsonSchema() const;
};

struct ToolDefinition {
  std::string name;
  std::string description;
  std::string category = "general";
  std::vector<ToolParameter> parameters;

  api::json::Value toJsonSchema() const;
  api::ToolDefinition toOpenAiTool() const;
};

enum class ToolStatus {
  Success,
  Error,
  Blocked
};

struct ToolResult {
  ToolStatus status = ToolStatus::Success;
  std::string output;
  api::json::Value data;
  std::string error;
  double durationMs = 0.0;

  bool ok() const { return status == ToolStatus::Success && error.empty(); }

  api::json::Value toJson() const;
  static ToolResult makeSuccess(std::string output, api::json::Value data = api::json::Value(), double durationMs = 0.0);
  static ToolResult makeError(std::string error, double durationMs = 0.0);
};

// Abstract interface for all tools invocable by agents
class ITool {
 public:
  virtual ~ITool() = default;

  virtual const ToolDefinition& definition() const = 0;
  virtual ToolResult execute(const api::json::Value& arguments) = 0;
};

// Generic functional wrapper for lightweight tool creation
class FunctionTool : public ITool {
 public:
  using Handler = std::function<ToolResult(const api::json::Value&)>;

  FunctionTool(ToolDefinition def, Handler handler)
      : def_(std::move(def)), handler_(std::move(handler)) {}

  const ToolDefinition& definition() const override { return def_; }
  ToolResult execute(const api::json::Value& arguments) override {
    if (!handler_) {
      return ToolResult::makeError("Tool handler unassigned");
    }
    auto start = std::chrono::steady_clock::now();
    ToolResult res = handler_(arguments);
    auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    res.durationMs = elapsed;
    return res;
  }

 private:
  ToolDefinition def_;
  Handler handler_;
};

}  // namespace qorvix::agents
