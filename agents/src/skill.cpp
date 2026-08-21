#include "qorvix/agents/skill.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace qorvix::agents {

namespace {

std::string trim(std::string_view s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
  return std::string(s);
}

std::vector<std::string> parseList(std::string_view s) {
  std::vector<std::string> res;
  std::string str = trim(s);
  if (str.empty()) return res;

  if (str.front() == '[' && str.back() == ']') {
    str = str.substr(1, str.size() - 2);
  }

  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, ',')) {
    std::string trimmed = trim(item);
    if (!trimmed.empty()) {
      if ((trimmed.front() == '"' && trimmed.back() == '"') ||
          (trimmed.front() == '\'' && trimmed.back() == '\'')) {
        trimmed = trimmed.substr(1, trimmed.size() - 2);
      }
      res.push_back(trimmed);
    }
  }
  return res;
}

}  // namespace

api::json::Value SkillParameter::toJson() const {
  auto v = api::json::Value::object();
  v["name"] = name;
  v["type"] = type;
  v["description"] = description;
  v["required"] = required;
  if (!defaultValue.empty()) v["default"] = defaultValue;
  return v;
}

bool SkillParameter::fromJson(const api::json::Value& v, SkillParameter& out, std::string& error) {
  if (!v.isObject()) {
    error = "SkillParameter expects a JSON object";
    return false;
  }
  if (const auto* n = v.get("name"); n && n->isString()) out.name = n->asString();
  else { error = "Missing parameter name"; return false; }
  if (const auto* t = v.get("type"); t && t->isString()) out.type = t->asString();
  if (const auto* d = v.get("description"); d && d->isString()) out.description = d->asString();
  if (const auto* r = v.get("required"); r && r->isBool()) out.required = r->asBool();
  if (const auto* df = v.get("default"); df && df->isString()) out.defaultValue = df->asString();
  return true;
}

api::json::Value SkillExample::toJson() const {
  auto v = api::json::Value::object();
  v["title"] = title;
  v["input"] = input;
  v["output"] = output;
  if (!explanation.empty()) v["explanation"] = explanation;
  return v;
}

bool SkillExample::fromJson(const api::json::Value& v, SkillExample& out, std::string& error) {
  if (!v.isObject()) {
    error = "SkillExample expects a JSON object";
    return false;
  }
  if (const auto* t = v.get("title"); t && t->isString()) out.title = t->asString();
  if (const auto* i = v.get("input"); i && i->isString()) out.input = i->asString();
  if (const auto* o = v.get("output"); o && o->isString()) out.output = o->asString();
  if (const auto* e = v.get("explanation"); e && e->isString()) out.explanation = e->asString();
  return true;
}

std::string SkillDefinition::toPromptSection() const {
  std::ostringstream ss;
  ss << "### Skill: " << name << " (v" << version << ", Category: " << category << ")\n"
     << description << "\n";
  if (!requiredTools.empty()) {
    ss << "Required Tools: ";
    for (std::size_t i = 0; i < requiredTools.size(); ++i) {
      ss << (i > 0 ? ", " : "") << requiredTools[i];
    }
    ss << "\n";
  }
  ss << "\n#### Playbook & Step-by-Step Procedure:\n"
     << instructions << "\n";

  if (!examples.empty()) {
    ss << "\n#### Demonstrations & Examples:\n";
    for (const auto& ex : examples) {
      ss << "**Example: " << ex.title << "**\n"
         << "- Input: " << ex.input << "\n"
         << "- Output / Action: " << ex.output << "\n";
      if (!ex.explanation.empty()) {
        ss << "- Note: " << ex.explanation << "\n";
      }
      ss << "\n";
    }
  }
  return ss.str();
}

std::string SkillDefinition::toMarkdown() const {
  std::ostringstream ss;
  ss << "---\n"
     << "name: " << name << "\n"
     << "version: " << version << "\n"
     << "description: " << description << "\n"
     << "category: " << category << "\n"
     << "author: " << author << "\n";

  if (!tags.empty()) {
    ss << "tags: [";
    for (std::size_t i = 0; i < tags.size(); ++i) {
      ss << (i > 0 ? ", " : "") << "\"" << tags[i] << "\"";
    }
    ss << "]\n";
  }

  if (!requiredTools.empty()) {
    ss << "required_tools: [";
    for (std::size_t i = 0; i < requiredTools.size(); ++i) {
      ss << (i > 0 ? ", " : "") << "\"" << requiredTools[i] << "\"";
    }
    ss << "]\n";
  }

  ss << "---\n\n"
     << instructions << "\n";
  return ss.str();
}

bool SkillDefinition::fromMarkdown(std::string_view markdown, SkillDefinition& out, std::string& error) {
  std::string_view text = markdown;
  // Trim leading whitespace
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
    text.remove_prefix(1);
  }

  if (text.rfind("---", 0) == 0) {
    auto endFm = text.find("\n---", 3);
    if (endFm != std::string_view::npos) {
      std::string_view frontmatter = text.substr(3, endFm - 3);
      std::string_view body = text.substr(endFm + 4);
      while (!body.empty() && (body.front() == '\n' || body.front() == '\r')) {
        body.remove_prefix(1);
      }
      out.instructions = std::string(body);

      // Parse YAML lines
      std::stringstream ss(std::string(frontmatter));
      std::string line;
      while (std::getline(ss, line)) {
        auto colon = line.find(':');
        if (colon != std::string::npos) {
          std::string key = trim(line.substr(0, colon));
          std::string val = trim(line.substr(colon + 1));
          if ((val.front() == '"' && val.back() == '"') ||
              (val.front() == '\'' && val.back() == '\'')) {
            val = val.substr(1, val.size() - 2);
          }

          if (key == "name") out.name = val;
          else if (key == "version") out.version = val;
          else if (key == "description") out.description = val;
          else if (key == "category") out.category = val;
          else if (key == "author") out.author = val;
          else if (key == "tags") out.tags = parseList(val);
          else if (key == "required_tools") out.requiredTools = parseList(val);
        }
      }
      if (out.name.empty()) {
        error = "Missing 'name' in skill frontmatter";
        return false;
      }
      return true;
    }
  }

  // Fallback: entire text is instructions if no frontmatter
  out.instructions = std::string(text);
  if (out.name.empty()) out.name = "unnamed_skill";
  return true;
}

api::json::Value SkillDefinition::toJson() const {
  auto v = api::json::Value::object();
  v["name"] = name;
  v["version"] = version;
  v["description"] = description;
  v["category"] = category;
  v["author"] = author;
  v["instructions"] = instructions;

  auto t = api::json::Value::array();
  for (const auto& tag : tags) t.push(tag);
  v["tags"] = std::move(t);

  auto rt = api::json::Value::array();
  for (const auto& tool : requiredTools) rt.push(tool);
  v["required_tools"] = std::move(rt);

  auto params = api::json::Value::array();
  for (const auto& p : parameters) params.push(p.toJson());
  v["parameters"] = std::move(params);

  auto exs = api::json::Value::array();
  for (const auto& e : examples) exs.push(e.toJson());
  v["examples"] = std::move(exs);

  if (!sourcePath.empty()) v["source_path"] = sourcePath;
  return v;
}

bool SkillDefinition::fromJson(const api::json::Value& v, SkillDefinition& out, std::string& error) {
  if (!v.isObject()) {
    error = "SkillDefinition expects a JSON object";
    return false;
  }
  if (const auto* n = v.get("name"); n && n->isString()) out.name = n->asString();
  else { error = "Missing skill name"; return false; }

  if (const auto* ver = v.get("version"); ver && ver->isString()) out.version = ver->asString();
  if (const auto* d = v.get("description"); d && d->isString()) out.description = d->asString();
  if (const auto* c = v.get("category"); c && c->isString()) out.category = c->asString();
  if (const auto* a = v.get("author"); a && a->isString()) out.author = a->asString();
  if (const auto* inst = v.get("instructions"); inst && inst->isString()) out.instructions = inst->asString();
  if (const auto* sp = v.get("source_path"); sp && sp->isString()) out.sourcePath = sp->asString();

  out.tags.clear();
  if (const auto* tg = v.get("tags"); tg && tg->isArray()) {
    for (const auto& item : tg->items()) {
      if (item.isString()) out.tags.push_back(item.asString());
    }
  }

  out.requiredTools.clear();
  if (const auto* rt = v.get("required_tools"); rt && rt->isArray()) {
    for (const auto& item : rt->items()) {
      if (item.isString()) out.requiredTools.push_back(item.asString());
    }
  }

  out.parameters.clear();
  if (const auto* params = v.get("parameters"); params && params->isArray()) {
    for (const auto& item : params->items()) {
      SkillParameter p;
      std::string err;
      if (SkillParameter::fromJson(item, p, err)) out.parameters.push_back(std::move(p));
    }
  }

  out.examples.clear();
  if (const auto* exs = v.get("examples"); exs && exs->isArray()) {
    for (const auto& item : exs->items()) {
      SkillExample e;
      std::string err;
      if (SkillExample::fromJson(item, e, err)) out.examples.push_back(std::move(e));
    }
  }

  return true;
}

}  // namespace qorvix::agents
