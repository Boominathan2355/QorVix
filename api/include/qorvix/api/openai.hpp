#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "qorvix/api/json.hpp"

// OpenAI-compatible request/response schema mapping (SPEC "OpenAI Compatible API"). Pure
// JSON <-> struct translation with no runtime coupling — the server (core) bridges these structs
// to the scheduler. Covers /v1/models, /v1/chat/completions, /v1/completions (streaming + not).
namespace qorvix::api {

struct ChatMessage {
  std::string role;
  std::string content;
};

// Sampling/decoding parameters common to chat and text completions (OpenAI names).
struct SamplingRequest {
  int maxTokens = 128;
  float temperature = 0.8f;
  float topP = 0.95f;
  int topK = 40;          // OpenAI extension (also used by llama.cpp servers)
  float minP = 0.0f;      // extension
  float frequencyPenalty = 0.0f;
  float presencePenalty = 0.0f;
  float repetitionPenalty = 1.0f;  // extension
  std::uint64_t seed = 0;
  std::vector<std::string> stop;
};

struct ChatRequest {
  bool valid = false;
  std::string model;
  std::vector<ChatMessage> messages;
  bool stream = false;
  SamplingRequest sampling;
};

struct CompletionRequest {
  bool valid = false;
  std::string model;
  std::string prompt;
  bool stream = false;
  SamplingRequest sampling;
};

// Parse requests from a decoded JSON body. On failure, `error` is set and .valid stays false.
ChatRequest parseChatRequest(const json::Value& body, std::string& error);
CompletionRequest parseCompletionRequest(const json::Value& body, std::string& error);

// Flattens chat messages into a single prompt. This is a simple, generic template
// ("<role>:\n<content>\n\n" then an "assistant:" turn); exact per-model chat templates
// (from GGUF metadata) are a later refinement — use /v1/completions for full prompt control.
std::string buildChatPrompt(const std::vector<ChatMessage>& messages);
std::string buildChatPromptWithTemplate(const std::vector<ChatMessage>& messages, const std::string& chatTemplate = "");

// ---- responses (return JSON values; the server serializes + frames them) -------------------

json::Value modelsResponse(const std::vector<std::string>& modelIds);

json::Value chatCompletion(const std::string& id, const std::string& model,
                           const std::string& content, int promptTokens, int completionTokens,
                           const std::string& finishReason);
// Streaming chunk: `role` emits the opening {"role":"assistant"} delta; otherwise a content delta.
// A non-empty finishReason marks the terminal chunk (empty content delta).
json::Value chatChunk(const std::string& id, const std::string& model, const std::string& delta,
                      bool role, const std::string& finishReason);

json::Value completion(const std::string& id, const std::string& model, const std::string& text,
                       int promptTokens, int completionTokens, const std::string& finishReason);
json::Value completionChunk(const std::string& id, const std::string& model,
                            const std::string& text, const std::string& finishReason);

json::Value errorResponse(const std::string& message, const std::string& type = "invalid_request_error");

// SSE framing.
std::string sseData(const json::Value& v);  // "data: <json>\n\n"
std::string sseDone();                        // "data: [DONE]\n\n"

}  // namespace qorvix::api
