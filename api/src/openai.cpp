#include "qorvix/api/openai.hpp"

#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>

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
    if (const auto* c = m.get("content")) {
      if (c->isArray()) {
        // OpenAI's multimodal shape: content is an array of typed parts. Order is significant —
        // it is what places the image relative to the words about it.
        for (const auto& part : c->items()) {
          if (!part.isObject()) {
            error = "each content part must be an object";
            return req;
          }
          const auto* t = part.get("type");
          const std::string type = t ? t->asString() : std::string();
          if (type == "text") {
            ContentPart p;
            p.type = ContentPart::Type::Text;
            if (const auto* v = part.get("text")) p.text = v->asString();
            cm.parts.push_back(std::move(p));
          } else if (type == "image_url") {
            ContentPart p;
            p.type = ContentPart::Type::ImageUrl;
            // Accept both the nested {"image_url":{"url":...}} object and the flat string form
            // some clients send; rejecting the latter would fail on requests that are otherwise
            // unambiguous.
            if (const auto* v = part.get("image_url")) {
              if (v->isString()) {
                p.imageUrl = v->asString();
              } else if (const auto* u = v->get("url")) {
                p.imageUrl = u->asString();
              }
            }
            if (p.imageUrl.empty()) {
              error = "an image_url content part is missing its 'url'";
              return req;
            }
            cm.parts.push_back(std::move(p));
          } else {
            // Named rather than ignored: silently dropping an input_audio part would answer a
            // question about media the model never received.
            error = "unsupported content part type '" + type + "' (expected 'text' or 'image_url')";
            return req;
          }
        }
        // Default flattening, so a caller that never calls flattenContentParts still sees text.
        for (const auto& p : cm.parts)
          if (p.type == ContentPart::Type::Text) cm.content += p.text;
      } else {
        cm.content = c->asString();
      }
    }
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

EmbeddingsRequest parseEmbeddingsRequest(const json::Value& body, std::string& error) {
  EmbeddingsRequest req;
  if (!body.isObject()) {
    error = "request body must be a JSON object";
    return req;
  }
  const auto* input = body.get("input");
  if (!input) {
    error = "'input' is required";
    return req;
  }

  if (input->isString()) {
    req.input.push_back(input->asString());
  } else if (input->isArray() && input->size() > 0) {
    // Disambiguate the three array shapes by the first element's type.
    const auto& first = input->items()[0];
    if (first.isString()) {
      for (const auto& e : input->items()) {
        if (!e.isString()) {
          error = "'input' array must be all strings";
          return req;
        }
        req.input.push_back(e.asString());
      }
    } else if (first.isNumber()) {
      std::vector<int> ids;
      for (const auto& e : input->items()) {
        if (!e.isNumber()) {
          error = "'input' array must be all token ids";
          return req;
        }
        ids.push_back(e.asInt(0));
      }
      req.inputTokens.push_back(std::move(ids));
    } else if (first.isArray()) {
      for (const auto& row : input->items()) {
        if (!row.isArray()) {
          error = "'input' array must be all token-id arrays";
          return req;
        }
        std::vector<int> ids;
        for (const auto& e : row.items()) ids.push_back(e.asInt(0));
        req.inputTokens.push_back(std::move(ids));
      }
    } else {
      error = "'input' must be a string, array of strings, or token ids";
      return req;
    }
  }

  if (req.count() == 0) {
    error = "'input' must be a non-empty string, array of strings, or array of token ids";
    return req;
  }

  if (const auto* v = body.get("model")) req.model = v->asString();
  if (const auto* v = body.get("user")) req.user = v->asString();
  if (const auto* v = body.get("dimensions")) req.dimensions = v->asInt(0);
  if (const auto* v = body.get("encoding_format")) {
    const std::string fmt = v->asString();
    if (fmt != "float" && fmt != "base64") {
      error = "'encoding_format' must be \"float\" or \"base64\"";
      return req;
    }
    req.encodingFormat = fmt;
  }
  req.valid = true;
  return req;
}

namespace {

json::Value usage(int prompt, int completion) {
  json::Value u = json::Value::object();
  u["prompt_tokens"] = prompt;
  u["completion_tokens"] = completion;
  u["total_tokens"] = prompt + completion;
  return u;
}

// Embeddings usage has no completion half, so this is a separate object rather than usage(n, 0) —
// emitting "completion_tokens": 0 would be a field the OpenAI schema does not define here.
json::Value embeddingUsage(int prompt) {
  json::Value u = json::Value::object();
  u["prompt_tokens"] = prompt;
  u["total_tokens"] = prompt;
  return u;
}

}  // namespace

std::vector<std::string> flattenContentParts(std::vector<ChatMessage>& messages,
                                             const std::string& imageMarker) {
  std::vector<std::string> urls;
  for (auto& m : messages) {
    if (m.parts.empty()) continue;
    std::string text;
    for (const auto& p : m.parts) {
      if (p.type == ContentPart::Type::Text) {
        text += p.text;
      } else {
        // The marker goes exactly where the part was, so the image lands in the same place in the
        // rendered prompt that the client put it in the message.
        text += imageMarker;
        urls.push_back(p.imageUrl);
      }
    }
    m.content = std::move(text);
  }
  return urls;
}

bool decodeBase64(std::string_view text, std::vector<std::uint8_t>& out, std::string& error) {
  error.clear();
  out.clear();

  // Reverse alphabet, built once. -1 = not a base64 character.
  static const auto kReverse = [] {
    std::array<signed char, 256> table{};
    table.fill(-1);
    const std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (std::size_t i = 0; i < alphabet.size(); ++i)
      table[static_cast<unsigned char>(alphabet[i])] = static_cast<signed char>(i);
    return table;
  }();

  std::uint32_t accum = 0;
  int bits = 0;
  out.reserve(text.size() / 4 * 3);
  for (char c : text) {
    const auto uc = static_cast<unsigned char>(c);
    // Data URIs are routinely wrapped across lines by clients that build them by hand.
    if (uc == '\n' || uc == '\r' || uc == ' ' || uc == '\t') continue;
    if (uc == '=') break;  // padding: everything after it is padding too
    const signed char v = kReverse[uc];
    if (v < 0) {
      error = "invalid base64 character";
      return false;
    }
    accum = (accum << 6) | static_cast<std::uint32_t>(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<std::uint8_t>((accum >> bits) & 0xFF));
    }
  }
  // A trailing group of exactly one character carries 6 bits — not enough for a byte, so the
  // input was truncated. Leftover bits from a 2- or 3-character group are the normal padding case.
  if (bits >= 6) {
    error = "truncated base64 input";
    return false;
  }
  return true;
}

bool decodeDataUrl(std::string_view url, std::vector<std::uint8_t>& out, std::string& error) {
  error.clear();
  out.clear();

  auto startsWithNoCase = [&](std::string_view prefix) {
    if (url.size() < prefix.size()) return false;
    for (std::size_t i = 0; i < prefix.size(); ++i) {
      if (std::tolower(static_cast<unsigned char>(url[i])) != prefix[i]) return false;
    }
    return true;
  };

  if (startsWithNoCase("http://") || startsWithNoCase("https://")) {
    error =
        "remote image URLs are not fetched by this server — inline the image as a data: URI "
        "(data:image/png;base64,...), which every OpenAI SDK supports";
    return false;
  }
  if (!startsWithNoCase("data:")) {
    error = "image_url must be a data: URI (data:image/png;base64,...)";
    return false;
  }

  const std::size_t comma = url.find(',');
  if (comma == std::string_view::npos) {
    error = "malformed data: URI (no comma separating the header from the payload)";
    return false;
  }
  const std::string_view header = url.substr(5, comma - 5);
  if (header.find("base64") == std::string_view::npos) {
    // Percent-encoded data: URIs are legal but no client sends images that way.
    error = "only base64-encoded data: URIs are supported";
    return false;
  }
  if (!decodeBase64(url.substr(comma + 1), out, error)) return false;
  if (out.empty()) {
    error = "data: URI decoded to zero bytes";
    return false;
  }
  return true;
}

std::string embeddingsBase64(const std::vector<float>& v) {
  static const char* kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  // OpenAI's base64 form is the raw little-endian float32 buffer. Serialize explicitly rather
  // than memcpy'ing the vector, so the output is byte-identical on a big-endian host.
  std::string raw;
  raw.reserve(v.size() * 4);
  for (float f : v) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    raw.push_back(static_cast<char>(bits & 0xFF));
    raw.push_back(static_cast<char>((bits >> 8) & 0xFF));
    raw.push_back(static_cast<char>((bits >> 16) & 0xFF));
    raw.push_back(static_cast<char>((bits >> 24) & 0xFF));
  }

  std::string out;
  out.reserve(((raw.size() + 2) / 3) * 4);
  for (std::size_t i = 0; i < raw.size(); i += 3) {
    const std::uint32_t b0 = static_cast<unsigned char>(raw[i]);
    const std::uint32_t b1 = i + 1 < raw.size() ? static_cast<unsigned char>(raw[i + 1]) : 0;
    const std::uint32_t b2 = i + 2 < raw.size() ? static_cast<unsigned char>(raw[i + 2]) : 0;
    const std::uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
    out.push_back(kAlphabet[(triple >> 18) & 0x3F]);
    out.push_back(kAlphabet[(triple >> 12) & 0x3F]);
    out.push_back(i + 1 < raw.size() ? kAlphabet[(triple >> 6) & 0x3F] : '=');
    out.push_back(i + 2 < raw.size() ? kAlphabet[triple & 0x3F] : '=');
  }
  return out;
}

json::Value embeddingsResponse(const std::string& model,
                               const std::vector<std::vector<float>>& vectors, int promptTokens,
                               bool base64) {
  json::Value root = json::Value::object();
  root["object"] = "list";
  json::Value data = json::Value::array();
  for (std::size_t i = 0; i < vectors.size(); ++i) {
    json::Value e = json::Value::object();
    e["object"] = "embedding";
    e["index"] = static_cast<int>(i);
    if (base64) {
      e["embedding"] = embeddingsBase64(vectors[i]);
    } else {
      json::Value arr = json::Value::array();
      // json::Value has no float ctor; float promotes to double, and dump()'s %.10g round-trips
      // float32 exactly (float needs 9 significant digits).
      for (float x : vectors[i]) arr.push(json::Value(static_cast<double>(x)));
      e["embedding"] = std::move(arr);
    }
    data.push(std::move(e));
  }
  root["data"] = std::move(data);
  root["model"] = model;
  root["usage"] = embeddingUsage(promptTokens);
  return root;
}

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
