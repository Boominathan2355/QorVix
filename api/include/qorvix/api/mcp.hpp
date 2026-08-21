#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "qorvix/api/json.hpp"

namespace qorvix::api::mcp {

// Model Context Protocol (MCP) specification 2024-11-05 (Anthropic Open Standard)
// JSON-RPC 2.0 transport for external tools, resources, and prompt templates.

struct McpTool {
  std::string name;
  std::string description;
  json::Value inputSchema;  // JSON Schema for tool parameters
};

struct McpResource {
  std::string uri;
  std::string name;
  std::string description;
  std::string mimeType;
};

struct McpPromptArgument {
  std::string name;
  std::string description;
  bool required = false;
};

struct McpPrompt {
  std::string name;
  std::string description;
  std::vector<McpPromptArgument> arguments;
};

struct McpToolCallRequest {
  std::string name;
  json::Value arguments;
};

struct McpContentPart {
  std::string type;  // "text" | "image" | "resource"
  std::string text;
  std::string data;  // base64 data for binary / image
  std::string mimeType;
};

struct McpToolCallResult {
  bool isError = false;
  std::vector<McpContentPart> content;
};

// JSON-RPC 2.0 formatting helpers
json::Value makeJsonRpcRequest(std::uint64_t id, const std::string& method, const json::Value& params = json::Value::object());
json::Value makeJsonRpcSuccess(std::uint64_t id, const json::Value& result);
json::Value makeJsonRpcError(std::uint64_t id, int code, const std::string& message, const json::Value& data = json::Value());

// MCP tools serialization to OpenAI function definitions
json::Value toolsToOpenAiSchema(const std::vector<McpTool>& tools);

// Parses an MCP initialize response
bool parseInitializeResult(const json::Value& result, std::string& serverName, std::string& serverVersion);

// Parses an MCP tools/list response
std::vector<McpTool> parseToolsListResult(const json::Value& result);

// Parses an MCP tools/call response
McpToolCallResult parseToolCallResult(const json::Value& result);

}  // namespace qorvix::api::mcp
