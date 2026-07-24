#include "qorvix/api/openai.hpp"

namespace qorvix::api {

namespace {

// Reads sampling fields shared by chat and text completions from a request object.
SamplingRequest parseSampling(const json::Value& body) {
  SamplingRequest s;
  if (const auto* v = body.get("max_tokens")) s.maxTokens = v->asInt(s.maxTokens);
  if (const auto* v = body.get("temperature")) s.temperature = static_cast<float>(v->asNumber(s.temperature));
  if (const auto* v = body.get("top_p")) s.topP = static_cast<float>(v->asNumber(s.topP));
  if (const auto* v = body.get("top_k")) s.topK = v->asInt(s.topK);
  if (const auto* v = body.get("min_p")) s.minP = static_cast<float>(v->asNumber(s.minP));
  if (const auto* v = body.get("frequency_penalty"))
    s.frequencyPenalty = static_cast<float>(v->asNumber(s.frequencyPenalty));
  if (const auto* v = body.get("presence_penalty"))
    s.presencePenalty = static_cast<float>(v->asNumber(s.presencePenalty));
  if (const auto* v = body.get("repetition_penalty"))
    s.repetitionPenalty = static_cast<float>(v->asNumber(s.repetitionPenalty));
  if (const auto* v = body.get("seed")) s.seed = static_cast<std::uint64_t>(v->asNumber(0));
  if (const auto* v = body.get("stop")) {
    if (v->isString()) {
      s.stop.push_back(v->asString());
    } else if (v->isArray()) {
      for (const auto& e : v->items())
        if (e.isString()) s.stop.push_back(e.asString());
    }
  }
  return s;
}

}  // namespace

ChatRequest parseChatRequest(const json::Value& body, std::string& error) {
  ChatRequest req;
  if (!body.isObject()) {
    error = "request body must be a JSON object";
    return req;
  }
  const auto* messages = body.get("messages");
  if (!messages || !messages->isArray() || messages->size() == 0) {
    error = "'messages' must be a non-empty array";
    return req;
  }
  for (const auto& m : messages->items()) {
    if (!m.isObject()) {
      error = "each message must be an object";
      return req;
    }
    ChatMessage cm;
    if (const auto* r = m.get("role")) cm.role = r->asString();
    if (const auto* c = m.get("content")) cm.content = c->asString();
    req.messages.push_back(std::move(cm));
  }
  if (const auto* model = body.get("model")) req.model = model->asString();
  if (const auto* stream = body.get("stream")) req.stream = stream->asBool();
  req.sampling = parseSampling(body);
  req.valid = true;
  return req;
}

CompletionRequest parseCompletionRequest(const json::Value& body, std::string& error) {
  CompletionRequest req;
  if (!body.isObject()) {
    error = "request body must be a JSON object";
    return req;
  }
  const auto* prompt = body.get("prompt");
  if (!prompt || !prompt->isString()) {
    error = "'prompt' must be a string";
    return req;
  }
  req.prompt = prompt->asString();
  if (const auto* model = body.get("model")) req.model = model->asString();
  if (const auto* stream = body.get("stream")) req.stream = stream->asBool();
  req.sampling = parseSampling(body);
  req.valid = true;
  return req;
}

std::string buildChatPrompt(const std::vector<ChatMessage>& messages) {
  std::string out;
  for (const auto& m : messages) {
    out += m.role.empty() ? "user" : m.role;
    out += ":\n";
    out += m.content;
    out += "\n\n";
  }
  out += "assistant:\n";
  return out;
}

std::string detectChatTemplateFamily(const std::string& t) {
  auto has = [&](const char* n) { return t.find(n) != std::string::npos; };
  // Ordered most-specific first: several families share the "<|user|>" marker and differ only in
  // how a turn is CLOSED, so the discriminating token must be tested before the shared one.
  if (has("<|im_start|>")) return "chatml";
  if (has("start_header_id") || has("eot_id")) return "llama3";
  if (has("start_of_turn")) return "gemma";
  if (has("<|end|>")) return "phi3";     // Phi-3: "<|user|>" closed by "<|end|>"
  if (has("<|user|>")) return "zephyr";  // Zephyr/TinyLlama: "<|user|>" closed by eos_token
  if (has("[INST]")) return "mistral";
  return "generic";
}

std::string buildChatPromptWithTemplate(const std::vector<ChatMessage>& messages,
                                        const std::string& chatTemplate,
                                        const std::string& eosToken) {
  const std::string family = detectChatTemplateFamily(chatTemplate);
  auto roleOf = [](const ChatMessage& m) { return m.role.empty() ? std::string("user") : m.role; };
  std::string out;

  if (family == "chatml") {
    for (const auto& m : messages)
      out += "<|im_start|>" + roleOf(m) + "\n" + m.content + "<|im_end|>\n";
    out += "<|im_start|>assistant\n";
    return out;
  }
  if (family == "llama3") {
    for (const auto& m : messages)
      out += "<|start_header_id|>" + roleOf(m) + "<|end_header_id|>\n\n" + m.content + "<|eot_id|>";
    out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return out;
  }
  if (family == "gemma") {
    // Gemma names the assistant turn "model", not "assistant".
    for (const auto& m : messages) {
      const std::string r = roleOf(m) == "assistant" ? std::string("model") : roleOf(m);
      out += "<start_of_turn>" + r + "\n" + m.content + "<end_of_turn>\n";
    }
    out += "<start_of_turn>model\n";
    return out;
  }
  if (family == "phi3") {
    for (const auto& m : messages) out += "<|" + roleOf(m) + "|>\n" + m.content + "<|end|>\n";
    out += "<|assistant|>\n";
    return out;
  }
  if (family == "zephyr") {
    // TinyLlama/Zephyr close each turn with the model's EOS piece, not a literal marker.
    const std::string eos = eosToken.empty() ? std::string("</s>") : eosToken;
    for (const auto& m : messages) out += "<|" + roleOf(m) + "|>\n" + m.content + eos + "\n";
    out += "<|assistant|>\n";
    return out;
  }
  if (family == "mistral") {
    // Only user turns are bracketed; assistant turns are emitted bare.
    for (const auto& m : messages) {
      if (roleOf(m) == "assistant") out += m.content;
      else out += "[INST] " + m.content + " [/INST]";
    }
    return out;
  }
  return buildChatPrompt(messages);
}

json::Value modelsResponse(const std::vector<std::string>& modelIds) {
  json::Value root = json::Value::object();
  root["object"] = "list";
  json::Value data = json::Value::array();
  for (const auto& id : modelIds) {
    json::Value m = json::Value::object();
    m["id"] = id;
    m["object"] = "model";
    m["owned_by"] = "qorvix";
    data.push(std::move(m));
  }
  root["data"] = std::move(data);
  return root;
}

namespace {
json::Value usage(int prompt, int completion) {
  json::Value u = json::Value::object();
  u["prompt_tokens"] = prompt;
  u["completion_tokens"] = completion;
  u["total_tokens"] = prompt + completion;
  return u;
}
}  // namespace

json::Value chatCompletion(const std::string& id, const std::string& model,
                           const std::string& content, int promptTokens, int completionTokens,
                           const std::string& finishReason) {
  json::Value root = json::Value::object();
  root["id"] = id;
  root["object"] = "chat.completion";
  root["model"] = model;
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  json::Value msg = json::Value::object();
  msg["role"] = "assistant";
  msg["content"] = content;
  choice["message"] = std::move(msg);
  choice["finish_reason"] = finishReason;
  json::Value choices = json::Value::array();
  choices.push(std::move(choice));
  root["choices"] = std::move(choices);
  root["usage"] = usage(promptTokens, completionTokens);
  return root;
}

json::Value chatChunk(const std::string& id, const std::string& model, const std::string& delta,
                      bool role, const std::string& finishReason) {
  json::Value root = json::Value::object();
  root["id"] = id;
  root["object"] = "chat.completion.chunk";
  root["model"] = model;
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  json::Value d = json::Value::object();
  if (role) d["role"] = "assistant";
  if (!delta.empty()) d["content"] = delta;
  choice["delta"] = std::move(d);
  choice["finish_reason"] = finishReason.empty() ? json::Value(nullptr) : json::Value(finishReason);
  json::Value choices = json::Value::array();
  choices.push(std::move(choice));
  root["choices"] = std::move(choices);
  return root;
}

json::Value completion(const std::string& id, const std::string& model, const std::string& text,
                       int promptTokens, int completionTokens, const std::string& finishReason) {
  json::Value root = json::Value::object();
  root["id"] = id;
  root["object"] = "text_completion";
  root["model"] = model;
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  choice["text"] = text;
  choice["finish_reason"] = finishReason;
  json::Value choices = json::Value::array();
  choices.push(std::move(choice));
  root["choices"] = std::move(choices);
  root["usage"] = usage(promptTokens, completionTokens);
  return root;
}

json::Value completionChunk(const std::string& id, const std::string& model,
                            const std::string& text, const std::string& finishReason) {
  json::Value root = json::Value::object();
  root["id"] = id;
  root["object"] = "text_completion";
  root["model"] = model;
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  choice["text"] = text;
  choice["finish_reason"] = finishReason.empty() ? json::Value(nullptr) : json::Value(finishReason);
  json::Value choices = json::Value::array();
  choices.push(std::move(choice));
  root["choices"] = std::move(choices);
  return root;
}

json::Value errorResponse(const std::string& message, const std::string& type) {
  json::Value root = json::Value::object();
  json::Value err = json::Value::object();
  err["message"] = message;
  err["type"] = type;
  root["error"] = std::move(err);
  return root;
}

std::string sseData(const json::Value& v) { return "data: " + v.dump() + "\n\n"; }
std::string sseDone() { return "data: [DONE]\n\n"; }

}  // namespace qorvix::api
