#include <catch2/catch_test_macros.hpp>

#include <string>

#include "qorvix/api/json.hpp"
#include "qorvix/api/openai.hpp"

using namespace qorvix::api;

TEST_CASE("parses a chat completion request", "[openai]") {
  auto body = json::parse(R"({
    "model": "tinyllama",
    "messages": [{"role":"system","content":"be terse"},{"role":"user","content":"hi"}],
    "stream": true,
    "max_tokens": 32,
    "temperature": 0.2,
    "top_k": 10,
    "stop": ["</s>", "\n\n"]
  })");
  REQUIRE(body.has_value());

  std::string err;
  ChatRequest req = parseChatRequest(*body, err);
  REQUIRE(req.valid);
  REQUIRE(err.empty());
  REQUIRE(req.model == "tinyllama");
  REQUIRE(req.messages.size() == 2);
  REQUIRE(req.messages[1].role == "user");
  REQUIRE(req.messages[1].content == "hi");
  REQUIRE(req.stream == true);
  REQUIRE(req.sampling.maxTokens == 32);
  REQUIRE(req.sampling.temperature == 0.2f);
  REQUIRE(req.sampling.topK == 10);
  REQUIRE(req.sampling.stop.size() == 2);
  REQUIRE(req.sampling.stop[0] == "</s>");
}

TEST_CASE("rejects a chat request without messages", "[openai]") {
  auto body = json::parse(R"({"model":"x"})");
  std::string err;
  ChatRequest req = parseChatRequest(*body, err);
  REQUIRE_FALSE(req.valid);
  REQUIRE_FALSE(err.empty());
}

TEST_CASE("parses a text completion request", "[openai]") {
  auto body = json::parse(R"({"model":"m","prompt":"The capital","max_tokens":5})");
  std::string err;
  CompletionRequest req = parseCompletionRequest(*body, err);
  REQUIRE(req.valid);
  REQUIRE(req.prompt == "The capital");
  REQUIRE(req.sampling.maxTokens == 5);
}

TEST_CASE("buildChatPrompt flattens messages", "[openai]") {
  std::vector<ChatMessage> msgs = {{"user", "hello"}};
  const std::string p = buildChatPrompt(msgs);
  REQUIRE(p.find("user:") != std::string::npos);
  REQUIRE(p.find("hello") != std::string::npos);
  REQUIRE(p.find("assistant:") != std::string::npos);
}

TEST_CASE("models response shape", "[openai]") {
  auto v = modelsResponse({"a", "b"});
  REQUIRE(v.get("object")->asString() == "list");
  REQUIRE(v.get("data")->size() == 2);
  REQUIRE(v.get("data")->at(0).get("id")->asString() == "a");
  REQUIRE(v.get("data")->at(0).get("owned_by")->asString() == "qorvix");
}

TEST_CASE("chat completion and chunk shapes", "[openai]") {
  auto full = chatCompletion("chatcmpl-1", "m", "Paris", 3, 1, "stop");
  REQUIRE(full.get("object")->asString() == "chat.completion");
  REQUIRE(full.get("choices")->at(0).get("message")->get("content")->asString() == "Paris");
  REQUIRE(full.get("choices")->at(0).get("finish_reason")->asString() == "stop");
  REQUIRE(full.get("usage")->get("total_tokens")->asInt() == 4);

  auto roleChunk = chatChunk("chatcmpl-1", "m", "", true, "");
  REQUIRE(roleChunk.get("object")->asString() == "chat.completion.chunk");
  REQUIRE(roleChunk.get("choices")->at(0).get("delta")->get("role")->asString() == "assistant");
  REQUIRE(roleChunk.get("choices")->at(0).get("finish_reason")->isNull());

  auto contentChunk = chatChunk("chatcmpl-1", "m", "Pa", false, "");
  REQUIRE(contentChunk.get("choices")->at(0).get("delta")->get("content")->asString() == "Pa");

  auto finalChunk = chatChunk("chatcmpl-1", "m", "", false, "stop");
  REQUIRE(finalChunk.get("choices")->at(0).get("finish_reason")->asString() == "stop");
}

TEST_CASE("SSE framing", "[openai]") {
  auto v = chatChunk("id", "m", "hi", false, "");
  const std::string line = sseData(v);
  REQUIRE(line.rfind("data: ", 0) == 0);
  REQUIRE(line.substr(line.size() - 2) == "\n\n");
  REQUIRE(sseDone() == "data: [DONE]\n\n");
}
