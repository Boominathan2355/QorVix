#include "qorvix/api/json.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace qorvix::api::json {

// ---- serialization -------------------------------------------------------------------------

namespace {

void escapeTo(std::string& out, const std::string& s) {
  out.push_back('"');
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
}

void dumpTo(std::string& out, const Value& v) {
  switch (v.type()) {
    case Value::Type::Null: out += "null"; break;
    case Value::Type::Bool: out += v.asBool() ? "true" : "false"; break;
    case Value::Type::Number: {
      const double n = v.asNumber();
      if (n == std::floor(n) && std::abs(n) < 1e15) {
        out += std::to_string(static_cast<std::int64_t>(n));
      } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.10g", n);
        out += buf;
      }
      break;
    }
    case Value::Type::String: escapeTo(out, v.asString()); break;
    case Value::Type::Array: {
      out.push_back('[');
      for (std::size_t i = 0; i < v.items().size(); ++i) {
        if (i) out.push_back(',');
        dumpTo(out, v.items()[i]);
      }
      out.push_back(']');
      break;
    }
    case Value::Type::Object: {
      out.push_back('{');
      bool first = true;
      for (const auto& [k, val] : v.members()) {
        if (!first) out.push_back(',');
        first = false;
        escapeTo(out, k);
        out.push_back(':');
        dumpTo(out, val);
      }
      out.push_back('}');
      break;
    }
  }
}

}  // namespace

std::string Value::dump() const {
  std::string out;
  dumpTo(out, *this);
  return out;
}

// ---- parsing (recursive descent) -----------------------------------------------------------

namespace {

struct Parser {
  const std::string& s;
  std::size_t i = 0;
  std::string err;

  explicit Parser(const std::string& src) : s(src) {}

  void skipWs() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
  }
  bool fail(const std::string& m) {
    if (err.empty()) err = m + " at offset " + std::to_string(i);
    return false;
  }

  bool parseValue(Value& out);

  void encodeUtf8(std::string& str, unsigned cp) {
    if (cp < 0x80) {
      str.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      str.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      str.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      str.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      str.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      str.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  bool parseHex4(unsigned& out) {
    if (i + 4 > s.size()) return fail("bad \\u escape");
    out = 0;
    for (int k = 0; k < 4; ++k) {
      char c = s[i++];
      out <<= 4;
      if (c >= '0' && c <= '9') out |= c - '0';
      else if (c >= 'a' && c <= 'f') out |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') out |= c - 'A' + 10;
      else return fail("bad hex digit");
    }
    return true;
  }

  bool parseString(std::string& out) {
    if (s[i] != '"') return fail("expected string");
    ++i;
    while (i < s.size()) {
      char c = s[i++];
      if (c == '"') return true;
      if (c == '\\') {
        if (i >= s.size()) return fail("unterminated escape");
        char e = s[i++];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'n': out.push_back('\n'); break;
          case 't': out.push_back('\t'); break;
          case 'r': out.push_back('\r'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'u': {
            unsigned cp;
            if (!parseHex4(cp)) return false;
            // Surrogate pair.
            if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
              i += 2;
              unsigned lo;
              if (!parseHex4(lo)) return false;
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              // Encode as 4-byte UTF-8.
              out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
              out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
              out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
              out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
              encodeUtf8(out, cp);
            }
            break;
          }
          default: return fail("bad escape");
        }
      } else {
        out.push_back(c);
      }
    }
    return fail("unterminated string");
  }

  bool literal(const char* lit, Value v, Value& out) {
    const std::size_t n = std::char_traits<char>::length(lit);
    if (s.compare(i, n, lit) != 0) return fail("invalid literal");
    i += n;
    out = std::move(v);
    return true;
  }
};

bool Parser::parseValue(Value& out) {
  skipWs();
  if (i >= s.size()) return fail("unexpected end");
  char c = s[i];
  switch (c) {
    case '{': {
      ++i;
      out = Value::object();
      skipWs();
      if (i < s.size() && s[i] == '}') {
        ++i;
        return true;
      }
      while (true) {
        skipWs();
        std::string key;
        if (!parseString(key)) return false;
        skipWs();
        if (i >= s.size() || s[i] != ':') return fail("expected ':'");
        ++i;
        Value val;
        if (!parseValue(val)) return false;
        out[key] = std::move(val);
        skipWs();
        if (i >= s.size()) return fail("unterminated object");
        if (s[i] == ',') {
          ++i;
          continue;
        }
        if (s[i] == '}') {
          ++i;
          return true;
        }
        return fail("expected ',' or '}'");
      }
    }
    case '[': {
      ++i;
      out = Value::array();
      skipWs();
      if (i < s.size() && s[i] == ']') {
        ++i;
        return true;
      }
      while (true) {
        Value val;
        if (!parseValue(val)) return false;
        out.push(std::move(val));
        skipWs();
        if (i >= s.size()) return fail("unterminated array");
        if (s[i] == ',') {
          ++i;
          continue;
        }
        if (s[i] == ']') {
          ++i;
          return true;
        }
        return fail("expected ',' or ']'");
      }
    }
    case '"': {
      std::string str;
      if (!parseString(str)) return false;
      out = Value(std::move(str));
      return true;
    }
    case 't': return literal("true", Value(true), out);
    case 'f': return literal("false", Value(false), out);
    case 'n': return literal("null", Value(nullptr), out);
    default: {
      if (c == '-' || (c >= '0' && c <= '9')) {
        const std::size_t start = i;
        if (s[i] == '-') ++i;
        while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' ||
                                s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-'))
          ++i;
        try {
          out = Value(std::stod(s.substr(start, i - start)));
        } catch (...) {
          return fail("bad number");
        }
        return true;
      }
      return fail("unexpected character");
    }
  }
}

}  // namespace

std::optional<Value> parse(const std::string& text, std::string* error) {
  Parser p(text);
  Value v;
  if (!p.parseValue(v)) {
    if (error) *error = p.err;
    return std::nullopt;
  }
  p.skipWs();
  if (p.i != text.size()) {
    if (error) *error = "trailing characters at offset " + std::to_string(p.i);
    return std::nullopt;
  }
  return v;
}

}  // namespace qorvix::api::json
