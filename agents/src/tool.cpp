#include "qorvix/agents/tool.hpp"

namespace qorvix::agents {

std::string_view toolParamTypeName(ToolParamType type) {
  switch (type) {
    case ToolParamType::String: return "string";
    case ToolParamType::Number: return "number";
    case ToolParamType::Integer: return "integer";
    case ToolParamType::Boolean: return "boolean";
    case ToolParamType::Array: return "array";
    case ToolParamType::Object: return "object";
  }
  return "string";
}

ToolParamType parseToolParamType(std::string_view name) {
  if (name == "number") return ToolParamType::Number;
  if (name == "integer") return ToolParamType::Integer;
  if (name == "boolean") return ToolParamType::Boolean;
  if (name == "array") return ToolParamType::Array;
  if (name == "object") return ToolParamType::Object;
  return ToolParamType::String;
}

api::json::Value ToolParameter::toJsonSchema() const {
  auto schema = api::json::Value::object();
  schema["type"] = std::string(toolParamTypeName(type));
  if (!description.empty()) {
    schema["description"] = description;
  }
  if (!enumValues.empty()) {
    auto ev = api::json::Value::array();
    for (const auto& e : enumValues) {
      ev.push(e);
    }
    schema["enum"] = std::move(ev);
  }
  return schema;
}

api::json::Value ToolDefinition::toJsonSchema() const {
  auto root = api::json::Value::object();
  root["type"] = "object";

  auto props = api::json::Value::object();
  auto reqs = api::json::Value::array();

  for (const auto& p : parameters) {
    props[p.name] = p.toJsonSchema();
    if (p.required) {
      reqs.push(p.name);
    }
  }

  root["properties"] = std::move(props);
  if (reqs.size() > 0) {
    root["required"] = std::move(reqs);
  }

  return root;
}

api::ToolDefinition ToolDefinition::toOpenAiTool() const {
  api::ToolDefinition t;
  t.type = "function";
  t.function.name = name;
  t.function.description = description;
  t.function.parameters = toJsonSchema();
  return t;
}

api::json::Value ToolResult::toJson() const {
  auto v = api::json::Value::object();
  std::string st = "success";
  if (status == ToolStatus::Error) st = "error";
  else if (status == ToolStatus::Blocked) st = "blocked";

  v["status"] = st;
  v["output"] = output;
  v["duration_ms"] = durationMs;
  if (!error.empty()) {
    v["error"] = error;
  }
  if (!data.isNull()) {
    v["data"] = data;
  }
  return v;
}

ToolResult ToolResult::makeSuccess(std::string output, api::json::Value data, double durationMs) {
  ToolResult r;
  r.status = ToolStatus::Success;
  r.output = std::move(output);
  r.data = std::move(data);
  r.durationMs = durationMs;
  return r;
}

ToolResult ToolResult::makeError(std::string error, double durationMs) {
  ToolResult r;
  r.status = ToolStatus::Error;
  r.error = error;
  r.output = "Error: " + error;
  r.durationMs = durationMs;
  return r;
}

}  // namespace qorvix::agents
