#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "qorvix/api/json.hpp"

// OpenAI-compatible request/response schema mapping (SPEC "OpenAI Compatible API"). Pure
// JSON <-> struct translation with no runtime coupling — the server (core) bridges these structs
// to the scheduler. Covers /v1/models, /v1/chat/completions, /v1/completions (streaming + not),
// and /v1/embeddings.
namespace qorvix::api {

// One element of OpenAI's multimodal `content` array. A message whose content is a plain string
// has no parts; a message whose content is an array has one part per element.
struct ContentPart {
  enum class Type { Text, ImageUrl, InputAudio, VideoUrl };
  Type type = Type::Text;
  std::string text;         // Type::Text
  std::string imageUrl;     // Type::ImageUrl — raw url field, usually a data: URI
  std::string videoUrl;     // Type::VideoUrl — video data URI or URL
  std::string audioData;    // Type::InputAudio — base64 encoded audio bytes
  std::string audioFormat;  // Type::InputAudio — "wav", "mp3", "ogg", "flac"
};

// Model Context Protocol (MCP) and OpenAI Tool schemas
struct ToolFunction {
  std::string name;
  std::string description;
  json::Value parameters;  // JSON Schema definition of input parameters
};

struct ToolDefinition {
  std::string type = "function";
  ToolFunction function;
};

struct ToolCall {
  std::string id;
  std::string type = "function";
  std::string name;
  std::string arguments;  // JSON stringified arguments
};

struct ChatMessage {
  std::string role;
  // The message as text. For an array content this is the flattened form
  std::string content;
  std::string reasoningContent;     // Chain-of-thought / <think> reasoning tokens
  std::vector<ContentPart> parts;   // empty when `content` arrived as a plain string
  std::vector<ToolCall> toolCalls;  // tool calls emitted by assistant
  std::string toolCallId;           // for role="tool", the id of the call this answers

  bool hasImages() const {
    for (const auto& p : parts)
      if (p.type == ContentPart::Type::ImageUrl) return true;
    return false;
  }

  bool hasAudio() const {
    for (const auto& p : parts)
      if (p.type == ContentPart::Type::InputAudio) return true;
    return false;
  }
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
  std::vector<ToolDefinition> tools;  // MCP tools exposed to model
  std::string toolChoice = "auto";
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
// Renders `messages` using the model's own chat template, read from the GGUF
// `tokenizer.chat_template` metadata key.
//
// The stored template is Jinja2, which this does NOT evaluate. Instead it identifies the family
// by the marker tokens the template contains and emits that family's format directly — enough to
// cover the templates real GGUF chat models ship, without embedding a Jinja engine. An unknown
// template falls back to buildChatPrompt's generic "role:\n" form.
//
// `eosToken` is the model's EOS piece (e.g. "</s>"); Zephyr/TinyLlama-style templates interpolate
// it after every message, so passing it wrong truncates or runs messages together.
std::string buildChatPromptWithTemplate(const std::vector<ChatMessage>& messages,
                                        const std::string& chatTemplate = "",
                                        const std::string& eosToken = "");

// Which family buildChatPromptWithTemplate detected — for logging and tests.
// One of: "chatml", "llama3", "gemma", "phi3", "zephyr", "mistral", "generic".
std::string detectChatTemplateFamily(const std::string& chatTemplate);

// ---- multimodal content (Phase 11b-2) --------------------------------------------------------

// Rewrites each message's `content` from its parts: text parts verbatim, image parts replaced by
// `imageMarker`. Returns every image URL across all messages, in the order the markers appear —
// so the Nth returned URL belongs to the Nth marker in the rendered prompt. Messages with no
// parts are left untouched.
//
// The marker is a PARAMETER rather than a constant here so this layer stays free of any runtime
// dependency; the server passes runtime::kImageMarker, the one definition of the token.
std::vector<std::string> flattenContentParts(std::vector<ChatMessage>& messages,
                                             const std::string& imageMarker);

// Standard base64 of arbitrary bytes. Extracted from embeddingsBase64 when /v1/images needed the
// same encoder for PNG bytes — an image and a float buffer differ in what they mean, not in how
// they are spelled over JSON.
std::string encodeBase64(const std::uint8_t* data, std::size_t size);

// Decodes standard base64 (with or without padding). Returns false on an invalid character or a
// truncated final group.
bool decodeBase64(std::string_view text, std::vector<std::uint8_t>& out, std::string& error);

// Decodes a `data:[<mediatype>][;base64],<data>` URI into raw bytes.
//
// A plain http(s):// URL is REFUSED with an explanatory error rather than fetched. Server-side
// fetching of a client-supplied URL is an SSRF primitive — it would let any caller make this
// process issue requests to hosts only it can reach — so the capability is declined outright
// instead of being added with a blocklist. Clients inline the image as a data: URI, which every
// OpenAI SDK already supports.
bool decodeDataUrl(std::string_view url, std::vector<std::uint8_t>& out, std::string& error);

// POST /v1/embeddings. `input` accepts four shapes in the OpenAI schema — a string, an array of
// strings, an array of ints (ONE pre-tokenized sequence), and an array of int arrays — so text and
// token inputs are collected separately rather than forced into one field.
struct EmbeddingsRequest {
  bool valid = false;
  std::string model;
  std::vector<std::string> input;             // text inputs
  std::vector<std::vector<int>> inputTokens;  // pre-tokenized inputs
  std::string encodingFormat = "float";       // "float" | "base64"
  int dimensions = 0;                         // 0 = model default (Matryoshka truncation)
  std::string user;

  // Total inputs across both forms, in request order (texts first, then token sequences).
  std::size_t count() const { return input.size() + inputTokens.size(); }
};

EmbeddingsRequest parseEmbeddingsRequest(const json::Value& body, std::string& error);

// ---- responses (return JSON values; the server serializes + frames them) -------------------

// POST /v1/images/generations.
//
// OpenAI's own fields (prompt, n, size, response_format, model, user) plus the ones a local
// diffusion runtime cannot do without and their API has no room for: negative_prompt, steps,
// guidance_scale, sampler, seed, clip_skip. The extensions are named the way the rest of the local
// ecosystem names them rather than invented here.
//
// Unset numeric fields stay at their sentinel (0 / -1) so the server can tell "the client did not
// say" from "the client asked for zero" — a `steps: 0` that silently became 20 would answer a
// malformed request with a picture.
struct ImagesRequest {
  bool valid = false;
  std::string model;
  std::string prompt;
  std::string negativePrompt;
  std::string responseFormat = "b64_json";
  std::string sampler;          // empty = the server's default
  int n = 1;
  int width = 0, height = 0;    // 0 = the model's native size
  int steps = 0;                // 0 = the server's default
  float guidance = -1.0f;       // < 0 = the server's default
  int clipSkip = 0;             // 0 = the server's default
  unsigned long long seed = 0;
  bool hasSeed = false;
};

ImagesRequest parseImagesRequest(const json::Value& body, std::string& error);

// `images` are base64-encoded PNG payloads, one per generated picture.
json::Value imagesResponse(const std::vector<std::string>& images, long long created);

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

// One "data" entry per vector, in request order. Embeddings usage carries only prompt_tokens and
// total_tokens — there is no completion half — so it does not reuse the chat usage object.
// `base64` emits each vector as little-endian float32 then base64, which is what OpenAI's own
// Python SDK requests BY DEFAULT; without it the most common client gets numbers it cannot read.
json::Value embeddingsResponse(const std::string& model,
                               const std::vector<std::vector<float>>& vectors, int promptTokens,
                               bool base64 = false);

// Little-endian float32 bytes, base64-encoded. Exposed for tests and for the server's own use.
std::string embeddingsBase64(const std::vector<float>& v);

json::Value errorResponse(const std::string& message, const std::string& type = "invalid_request_error");

// SSE framing.
std::string sseData(const json::Value& v);  // "data: <json>\n\n"
std::string sseDone();                        // "data: [DONE]\n\n"

}  // namespace qorvix::api
