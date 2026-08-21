#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "qorvix/agents/tool.hpp"
#include "qorvix/api/mcp.hpp"
#include "qorvix/api/openai.hpp"

namespace qorvix::agents {

// Registry managing all tools accessible to agents in QorVix
class ToolRegistry {
 public:
  ToolRegistry() = default;
  ~ToolRegistry() = default;

  // Non-copyable, movable
  ToolRegistry(const ToolRegistry&) = delete;
  ToolRegistry& operator=(const ToolRegistry&) = delete;
  ToolRegistry(ToolRegistry&&) noexcept;
  ToolRegistry& operator=(ToolRegistry&&) noexcept;

  // Tool registration
  bool registerTool(std::shared_ptr<ITool> tool);
  bool registerFunction(ToolDefinition def, FunctionTool::Handler handler);
  bool unregisterTool(const std::string& name);
  void clear();

  // Queries
  bool hasTool(const std::string& name) const;
  std::shared_ptr<ITool> getTool(const std::string& name) const;
  std::vector<std::string> toolNames() const;
  std::vector<ToolDefinition> definitions() const;

  // Schema exports
  std::vector<api::ToolDefinition> toOpenAiTools(const std::vector<std::string>& allowed = {}) const;
  std::vector<api::mcp::McpTool> toMcpTools(const std::vector<std::string>& allowed = {}) const;

  // Execution dispatch
  ToolResult execute(const std::string& name, const api::json::Value& arguments) const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<ITool>> tools_;
};

}  // namespace qorvix::agents
