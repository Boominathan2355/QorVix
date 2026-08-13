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

// ---- /v1/embeddings (Phase 11a) --------------------------------------------------------------

namespace {
qorvix::api::EmbeddingsRequest parseEmb(const std::string& body, std::string& err) {
  const auto v = json::parse(body);
  REQUIRE(v.has_value());
  return parseEmbeddingsRequest(*v, err);
}
}  // namespace

TEST_CASE("parse embeddings request accepts a bare string", "[openai]") {
  std::string err;
  const auto req = parseEmb(R"({"model":"bge","input":"hello world"})", err);
  REQUIRE(req.valid);
  REQUIRE(err.empty());
  REQUIRE(req.model == "bge");
  REQUIRE(req.input.size() == 1);
  REQUIRE(req.input[0] == "hello world");
  REQUIRE(req.inputTokens.empty());
  REQUIRE(req.count() == 1);
  REQUIRE(req.encodingFormat == "float");
  REQUIRE(req.dimensions == 0);
}

TEST_CASE("parse embeddings request accepts an array of strings", "[openai]") {
  std::string err;
  const auto req = parseEmb(R"({"input":["a","b","c"]})", err);
  REQUIRE(req.valid);
  REQUIRE(req.input.size() == 3);
  REQUIRE(req.input[2] == "c");
}

TEST_CASE("parse embeddings request accepts one pre-tokenized sequence", "[openai]") {
  // OpenAI allows a flat int array, meaning ONE sequence — not N single-token sequences. Getting
  // that wrong would silently turn a 5-token document into 5 one-token embeddings.
  std::string err;
  const auto req = parseEmb(R"({"input":[101,7592,2088,102]})", err);
  REQUIRE(req.valid);
  REQUIRE(req.input.empty());
  REQUIRE(req.inputTokens.size() == 1);
  REQUIRE(req.inputTokens[0] == std::vector<int>{101, 7592, 2088, 102});
  REQUIRE(req.count() == 1);
}

TEST_CASE("parse embeddings request accepts many pre-tokenized sequences", "[openai]") {
  std::string err;
  const auto req = parseEmb(R"({"input":[[101,102],[101,7592,102]]})", err);
  REQUIRE(req.valid);
  REQUIRE(req.inputTokens.size() == 2);
  REQUIRE(req.inputTokens[1].size() == 3);
}

TEST_CASE("parse embeddings request rejects a missing, empty, or mixed input", "[openai]") {
  std::string err;
  REQUIRE_FALSE(parseEmb(R"({"model":"bge"})", err).valid);
  REQUIRE(err.find("input") != std::string::npos);

  REQUIRE_FALSE(parseEmb(R"({"input":[]})", err).valid);
  // A mixed array is rejected rather than silently dropping the odd element, which is what the
  // `stop` field does — acceptable there, not here, where a dropped element shifts every later
  // vector's index in the response.
  REQUIRE_FALSE(parseEmb(R"({"input":["a",5]})", err).valid);
  REQUIRE_FALSE(parseEmb(R"({"input":true})", err).valid);
}

TEST_CASE("parse embeddings request reads encoding_format and dimensions", "[openai]") {
  std::string err;
  const auto req = parseEmb(R"({"input":"x","encoding_format":"base64","dimensions":128})", err);
  REQUIRE(req.valid);
  REQUIRE(req.encodingFormat == "base64");
  REQUIRE(req.dimensions == 128);

  REQUIRE_FALSE(parseEmb(R"({"input":"x","encoding_format":"protobuf"})", err).valid);
}

TEST_CASE("embeddings response indexes each vector in request order", "[openai]") {
  const std::vector<std::vector<float>> vecs{{1.0f, 2.0f}, {3.0f, 4.0f}};
  const auto v = embeddingsResponse("bge", vecs, 7);
  REQUIRE(v.get("object")->asString() == "list");
  REQUIRE(v.get("model")->asString() == "bge");

  const auto* data = v.get("data");
  REQUIRE(data->size() == 2);
  for (int i = 0; i < 2; ++i) {
    const auto& e = data->at(static_cast<std::size_t>(i));
    REQUIRE(e.get("object")->asString() == "embedding");
    REQUIRE(e.get("index")->asInt() == i);
    REQUIRE(e.get("embedding")->size() == 2);
  }
  REQUIRE(data->at(1).get("embedding")->at(0).asNumber() == 3.0);
}

TEST_CASE("embeddings usage reports prompt and total but no completion tokens", "[openai]") {
  // The chat usage object carries completion_tokens; the embeddings schema does not define it.
  const auto v = embeddingsResponse("bge", {{1.0f}}, 7);
  const auto* u = v.get("usage");
  REQUIRE(u->get("prompt_tokens")->asInt() == 7);
  REQUIRE(u->get("total_tokens")->asInt() == 7);
  REQUIRE(u->get("completion_tokens") == nullptr);
}

TEST_CASE("base64 embeddings encode little-endian float32", "[openai]") {
  // 1.0f is 0x3F800000, little-endian bytes 00 00 80 3F -> "AACAPw==". OpenAI's Python SDK
  // requests base64 by default, so this is the format most clients actually receive.
  REQUIRE(embeddingsBase64({1.0f}) == "AACAPw==");
  REQUIRE(embeddingsBase64({0.0f}) == "AAAAAA==");
  // Padding follows the byte count, not the float count: 2 floats are 8 bytes (8 % 3 == 2, so one
  // '='), while 3 floats are 12 bytes and divide evenly.
  const std::string two = embeddingsBase64({1.0f, 2.0f});
  REQUIRE(two.size() == 12);
  REQUIRE(two.find('=') == 11);

  const std::string three = embeddingsBase64({1.0f, 2.0f, 3.0f});
  REQUIRE(three.size() == 16);
  REQUIRE(three.find('=') == std::string::npos);

  const auto v = embeddingsResponse("bge", {{1.0f}}, 1, /*base64=*/true);
  REQUIRE(v.get("data")->at(0).get("embedding")->isString());
  REQUIRE(v.get("data")->at(0).get("embedding")->asString() == "AACAPw==");
}

// ---- multimodal content parts (Phase 11b-2) --------------------------------------------------

TEST_CASE("parses image_url content parts", "[openai]") {
  auto body = json::parse(R"({
    "messages": [{"role":"user","content":[
      {"type":"text","text":"what is this? "},
      {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},
      {"type":"text","text":" be terse"}
    ]}]
  })");
  REQUIRE(body.has_value());

  std::string err;
  ChatRequest req = parseChatRequest(*body, err);
  REQUIRE(req.valid);
  REQUIRE(req.messages.size() == 1);
  REQUIRE(req.messages[0].parts.size() == 3);
  REQUIRE(req.messages[0].hasImages());
  REQUIRE(req.messages[0].parts[1].type == ContentPart::Type::ImageUrl);
  REQUIRE(req.messages[0].parts[1].imageUrl == "data:image/png;base64,AAAA");
  // Default flattening drops the images, so a caller that never calls flattenContentParts still
  // gets usable text rather than an empty message.
  REQUIRE(req.messages[0].content == "what is this?  be terse");
}

TEST_CASE("accepts the flat image_url string form", "[openai]") {
  // Not in OpenAI's schema, but several clients send it and the intent is unambiguous.
  auto body = json::parse(
      R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":"data:image/png;base64,AA"}]}]})");
  std::string err;
  ChatRequest req = parseChatRequest(*body, err);
  REQUIRE(req.valid);
  REQUIRE(req.messages[0].parts[0].imageUrl == "data:image/png;base64,AA");
}

TEST_CASE("rejects unsupported and malformed content parts", "[openai]") {
  std::string err;
  // An unknown part type is named rather than skipped — silently dropping audio would answer a
  // question about media the model never received.
  auto audio = json::parse(
      R"({"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{}}]}]})");
  REQUIRE_FALSE(parseChatRequest(*audio, err).valid);
  REQUIRE(err.find("input_audio") != std::string::npos);

  auto noUrl = json::parse(
      R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{}}]}]})");
  REQUIRE_FALSE(parseChatRequest(*noUrl, err).valid);
  REQUIRE(err.find("url") != std::string::npos);
}

TEST_CASE("flattenContentParts places markers where the images were", "[openai]") {
  auto body = json::parse(R"({
    "messages": [
      {"role":"system","content":"be terse"},
      {"role":"user","content":[
        {"type":"text","text":"left "},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,AA"}},
        {"type":"text","text":" right"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,BB"}}
      ]}
    ]
  })");
  std::string err;
  ChatRequest req = parseChatRequest(*body, err);
  REQUIRE(req.valid);

  const auto urls = flattenContentParts(req.messages, "<image>");
  REQUIRE(urls.size() == 2);
  // Marker order must match URL order — that pairing is what puts each image in its own slot.
  REQUIRE(urls[0] == "data:image/png;base64,AA");
  REQUIRE(urls[1] == "data:image/png;base64,BB");
  REQUIRE(req.messages[1].content == "left <image> right<image>");
  // A plain-string message has no parts and must be left exactly as it was.
  REQUIRE(req.messages[0].content == "be terse");
}

TEST_CASE("decodeBase64 round-trips and rejects bad input", "[openai]") {
  std::vector<std::uint8_t> out;
  std::string err;

  REQUIRE(decodeBase64("AACAPw==", out, err));
  REQUIRE(out == std::vector<std::uint8_t>{0x00, 0x00, 0x80, 0x3F});

  // Unpadded input is legal and common in hand-built data URIs.
  REQUIRE(decodeBase64("AACAPw", out, err));
  REQUIRE(out.size() == 4);

  // Clients that wrap long data URIs across lines must still work.
  REQUIRE(decodeBase64("AACA\nPw==", out, err));
  REQUIRE(out.size() == 4);

  REQUIRE_FALSE(decodeBase64("AA*A", out, err));
  REQUIRE(err.find("invalid") != std::string::npos);
  // One leftover character carries 6 bits — not a byte, so the payload was cut short.
  REQUIRE_FALSE(decodeBase64("AAAAA", out, err));
  REQUIRE(err.find("truncated") != std::string::npos);
}

TEST_CASE("decodeDataUrl accepts data URIs and refuses remote fetches", "[openai]") {
  std::vector<std::uint8_t> out;
  std::string err;

  REQUIRE(decodeDataUrl("data:image/png;base64,AACAPw==", out, err));
  REQUIRE(out.size() == 4);

  // Refused rather than fetched: server-side retrieval of a client-supplied URL is an SSRF
  // primitive, so the capability is declined instead of filtered.
  REQUIRE_FALSE(decodeDataUrl("https://example.com/cat.png", out, err));
  REQUIRE(err.find("data:") != std::string::npos);
  REQUIRE_FALSE(decodeDataUrl("HTTP://example.com/cat.png", out, err));

  REQUIRE_FALSE(decodeDataUrl("data:image/png,notbase64", out, err));
  REQUIRE_FALSE(decodeDataUrl("data:image/png;base64", out, err));
  REQUIRE_FALSE(decodeDataUrl("data:image/png;base64,", out, err));
}
