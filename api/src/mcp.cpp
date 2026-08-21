#include "qorvix/api/mcp.hpp"

namespace qorvix::api::mcp {

json::Value makeJsonRpcRequest(std::uint64_t id, const std::string& method, const json::Value& params) {
  json::Value req = json::Value::object();
  req.set("jsonrpc", "2.0");
  req.set("id", static_cast<double>(id));
  req.set("method", method);
  if (!params.isNull()) {
    req.set("params", params);
  }
  return req;
}

json::Value makeJsonRpcSuccess(std::uint64_t id, const json::Value& result) {
  json::Value res = json::Value::object();
  res.set("jsonrpc", "2.0");
  res.set("id", static_cast<double>(id));
  res.set("result", result);
  return res;
}

json::Value makeJsonRpcError(std::uint64_t id, int code, const std::string& message, const json::Value& data) {
  json::Value res = json::Value::object();
  res.set("jsonrpc", "2.0");
  res.set("id", static_cast<double>(id));
  json::Value err = json::Value::object();
  err.set("code", static_cast<double>(code));
  err.set("message", message);
  if (!data.isNull()) err.set("data", data);
  res.set("error", err);
  return res;
}

json::Value toolsToOpenAiSchema(const std::vector<McpTool>& tools) {
  json::Value arr = json::Value::array();
  for (const auto& t : tools) {
    json::Value def = json::Value::object();
    def.set("type", "function");
    json::Value fn = json::Value::object();
    fn.set("name", t.name);
    fn.set("description", t.description);
    fn.set("parameters", t.inputSchema);
    def.set("function", fn);
    arr.push_back(std::move(def));
  }
  return arr;
}

bool parseInitializeResult(const json::Value& result, std::string& serverName, std::string& serverVersion) {
  if (!result.isObject()) return false;
  if (const auto* info = result.get("serverInfo")) {
    if (info->isObject()) {
      if (const auto* n = info->get("name")) serverName = n->asString();
      if (const auto* v = info->get("version")) serverVersion = v->asString();
      return true;
    }
  }
  return false;
}

std::vector<McpTool> parseToolsListResult(const json::Value& result) {
  std::vector<McpTool> tools;
  if (!result.isObject()) return tools;
  const auto* list = result.get("tools");
  if (!list || !list->isArray()) return tools;

  for (const auto& item : list->items()) {
    if (!item.isObject()) continue;
    McpTool tool;
    if (const auto* name = item.get("name")) tool.name = name->asString();
    if (const auto* desc = item.get("description")) tool.description = desc->asString();
    if (const auto* schema = item.get("inputSchema")) tool.inputSchema = *schema;
    tools.push_back(std::move(tool));
  }
  return tools;
}

McpToolCallResult parseToolCallResult(const json::Value& result) {
  McpToolCallResult res;
  if (!result.isObject()) return res;
  if (const auto* err = result.get("isError")) res.isError = err->asBool(false);
  const auto* content = result.get("content");
  if (content && content->isArray()) {
    for (const auto& c : content->items()) {
      if (!c.isObject()) continue;
      McpContentPart part;
      if (const auto* t = c.get("type")) part.type = t->asString();
      if (const auto* txt = c.get("text")) part.text = txt->asString();
      if (const auto* d = c.get("data")) part.data = d->asString();
      if (const auto* mime = c.get("mimeType")) part.mimeType = mime->asString();
      res.content.push_back(std::move(part));
    }
  }
  return res;
}

}  // namespace qorvix::api::mcp
