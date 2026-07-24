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

// ---- chat templates (GGUF tokenizer.chat_template) -------------------------------------------
// The stored template is Jinja2, which we do not evaluate; we identify the family by its marker
// tokens and emit that format. These pin the detection, because several families share the
// "<|user|>" marker and only differ in how a turn is CLOSED — getting that wrong runs messages
// together or truncates them, which looks like a model quality problem, not a prompt bug.

TEST_CASE("detects chat template families by their marker tokens", "[openai][chat]") {
  REQUIRE(detectChatTemplateFamily("{% for m %}<|im_start|>{{m.role}}") == "chatml");
  REQUIRE(detectChatTemplateFamily("<|start_header_id|>{{role}}<|end_header_id|>") == "llama3");
  REQUIRE(detectChatTemplateFamily("{{ '<start_of_turn>' + role }}") == "gemma");
  REQUIRE(detectChatTemplateFamily("[INST] {{ content }} [/INST]") == "mistral");
  REQUIRE(detectChatTemplateFamily("") == "generic");
  REQUIRE(detectChatTemplateFamily("something entirely unknown") == "generic");

  // The discriminating case: both close a "<|user|>" turn, but differently.
  REQUIRE(detectChatTemplateFamily("<|user|>\n{{content}}<|end|>\n") == "phi3");
  REQUIRE(detectChatTemplateFamily("<|user|>\n{{content}}{{eos_token}}\n") == "zephyr");
}

TEST_CASE("renders each family in its own format", "[openai][chat]") {
  const std::vector<ChatMessage> msgs{{"user", "hi"}, {"assistant", "hello"}, {"user", "bye"}};

  const std::string chatml = buildChatPromptWithTemplate(msgs, "<|im_start|>");
  REQUIRE(chatml.rfind("<|im_start|>user\nhi<|im_end|>\n", 0) == 0);
  REQUIRE(chatml.substr(chatml.size() - 22) == "<|im_start|>assistant\n");

  const std::string l3 = buildChatPromptWithTemplate(msgs, "<|start_header_id|>");
  REQUIRE(l3.find("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>") == 0);

  const std::string mistral = buildChatPromptWithTemplate(msgs, "[INST]");
  REQUIRE(mistral == "[INST] hi [/INST]hello[INST] bye [/INST]");  // assistant turns unbracketed

  // Gemma renames the assistant turn to "model".
  const std::string gemma = buildChatPromptWithTemplate(msgs, "<start_of_turn>");
  REQUIRE(gemma.find("<start_of_turn>model\nhello<end_of_turn>") != std::string::npos);
  REQUIRE(gemma.find("<start_of_turn>assistant") == std::string::npos);
}

TEST_CASE("zephyr templates close turns with the model's own eos piece", "[openai][chat]") {
  const std::vector<ChatMessage> msgs{{"user", "hi"}};
  // TinyLlama is this family; passing the wrong terminator is what made its chat output poor.
  const std::string out = buildChatPromptWithTemplate(msgs, "<|user|>{{eos_token}}", "</s>");
  REQUIRE(out == "<|user|>\nhi</s>\n<|assistant|>\n");

  // A different model's eos must be honoured, not hard-coded.
  const std::string other = buildChatPromptWithTemplate(msgs, "<|user|>{{eos_token}}", "<|endoftext|>");
  REQUIRE(other == "<|user|>\nhi<|endoftext|>\n<|assistant|>\n");

  // Empty eos falls back to the SPM default rather than emitting nothing.
  REQUIRE(buildChatPromptWithTemplate(msgs, "<|user|>{{eos_token}}", "") ==
          "<|user|>\nhi</s>\n<|assistant|>\n");
}

TEST_CASE("an unknown template falls back to the generic prompt", "[openai][chat]") {
  const std::vector<ChatMessage> msgs{{"user", "hi"}};
  REQUIRE(buildChatPromptWithTemplate(msgs, "") == buildChatPrompt(msgs));
  REQUIRE(buildChatPromptWithTemplate(msgs, "no markers here") == buildChatPrompt(msgs));
}

TEST_CASE("empty role defaults to user in every family", "[openai][chat]") {
  const std::vector<ChatMessage> msgs{{"", "hi"}};
  REQUIRE(buildChatPromptWithTemplate(msgs, "<|im_start|>").find("<|im_start|>user\n") == 0);
  REQUIRE(buildChatPromptWithTemplate(msgs, "<|user|>{{eos_token}}").find("<|user|>\n") == 0);
}
