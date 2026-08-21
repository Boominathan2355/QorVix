#include "qorvix/agents/tool_registry.hpp"

#include <algorithm>

namespace qorvix::agents {

ToolRegistry::ToolRegistry(ToolRegistry&& other) noexcept {
  std::lock_guard<std::mutex> lock(other.mutex_);
  tools_ = std::move(other.tools_);
}

ToolRegistry& ToolRegistry::operator=(ToolRegistry&& other) noexcept {
  if (this != &other) {
    std::scoped_lock lock(mutex_, other.mutex_);
    tools_ = std::move(other.tools_);
  }
  return *this;
}

bool ToolRegistry::registerTool(std::shared_ptr<ITool> tool) {
  if (!tool) return false;
  const auto& name = tool->definition().name;
  if (name.empty()) return false;

  std::lock_guard<std::mutex> lock(mutex_);
  tools_[name] = std::move(tool);
  return true;
}

bool ToolRegistry::registerFunction(ToolDefinition def, FunctionTool::Handler handler) {
  return registerTool(std::make_shared<FunctionTool>(std::move(def), std::move(handler)));
}

bool ToolRegistry::unregisterTool(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  return tools_.erase(name) > 0;
}

void ToolRegistry::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  tools_.clear();
}

bool ToolRegistry::hasTool(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return tools_.find(name) != tools_.end();
}

std::shared_ptr<ITool> ToolRegistry::getTool(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = tools_.find(name);
  if (it != tools_.end()) return it->second;
  return nullptr;
}

std::vector<std::string> ToolRegistry::toolNames() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> names;
  names.reserve(tools_.size());
  for (const auto& [name, _] : tools_) {
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::vector<ToolDefinition> ToolRegistry::definitions() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<ToolDefinition> defs;
  defs.reserve(tools_.size());
  for (const auto& [_, tool] : tools_) {
    defs.push_back(tool->definition());
  }
  return defs;
}

std::vector<api::ToolDefinition> ToolRegistry::toOpenAiTools(const std::vector<std::string>& allowed) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<api::ToolDefinition> res;
  for (const auto& [name, tool] : tools_) {
    if (!allowed.empty() && std::find(allowed.begin(), allowed.end(), name) == allowed.end()) {
      continue;
    }
    res.push_back(tool->definition().toOpenAiTool());
  }
  return res;
}

std::vector<api::mcp::McpTool> ToolRegistry::toMcpTools(const std::vector<std::string>& allowed) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<api::mcp::McpTool> res;
  for (const auto& [name, tool] : tools_) {
    if (!allowed.empty() && std::find(allowed.begin(), allowed.end(), name) == allowed.end()) {
      continue;
    }
    const auto& def = tool->definition();
    api::mcp::McpTool mcp;
    mcp.name = def.name;
    mcp.description = def.description;
    mcp.inputSchema = def.toJsonSchema();
    res.push_back(std::move(mcp));
  }
  return res;
}

ToolResult ToolRegistry::execute(const std::string& name, const api::json::Value& arguments) const {
  std::shared_ptr<ITool> tool;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tools_.find(name);
    if (it == tools_.end()) {
      return ToolResult::makeError("Unknown tool: '" + name + "'");
    }
    tool = it->second;
  }
  return tool->execute(arguments);
}

}  // namespace qorvix::agents
