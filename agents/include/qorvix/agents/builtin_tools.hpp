#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "qorvix/agents/tool.hpp"
#include "qorvix/agents/tool_registry.hpp"
#include "qorvix/api/mcp.hpp"

namespace qorvix::rag {
struct Index;
}
namespace qorvix::embeddings {
class IEmbeddingEngine;
}
namespace qorvix::tokenizer {
class Tokenizer;
}

namespace qorvix::agents {

class Blackboard;

// Calculator / Math evaluation tool
std::shared_ptr<ITool> createCalculatorTool();

// Safe workspace file operations tool (read, write, list)
std::shared_ptr<ITool> createFileOpsTool(std::filesystem::path workspaceRoot = "");

// Direct Blackboard inspector and mutator tool for agents
std::shared_ptr<ITool> createBlackboardTool(std::shared_ptr<Blackboard> blackboard);

// Semantic RAG search tool bridging QorVix hybrid retrieval
std::shared_ptr<ITool> createRagSearchTool(const rag::Index* index,
                                          embeddings::IEmbeddingEngine* engine,
                                          const tokenizer::Tokenizer* tok);

// System telemetry and hardware info tool
std::shared_ptr<ITool> createSystemInfoTool();

// Bridge for MCP protocol tools
std::shared_ptr<ITool> createMcpBridgeTool(api::mcp::McpTool mcpTool,
                                          std::function<api::mcp::McpToolCallResult(const api::mcp::McpToolCallRequest&)> dispatcher);

// Factory populating standard default tools into a registry
void registerDefaultTools(ToolRegistry& registry,
                         const std::filesystem::path& workspaceRoot = "",
                         std::shared_ptr<Blackboard> blackboard = nullptr);

}  // namespace qorvix::agents
