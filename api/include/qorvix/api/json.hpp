#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// A small, dependency-free JSON value with a standard-conforming parser and serializer — enough
// for the OpenAI-compatible API schema (objects, arrays, strings with escapes, numbers, bools,
// null). Objects preserve insertion order (a vector of pairs; fine for the small API payloads).
// Kept in-tree so the protocol layer compiles and is testable without vcpkg/third-party JSON.
namespace qorvix::api::json {

class Value {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Value() : type_(Type::Null) {}
  Value(std::nullptr_t) : type_(Type::Null) {}
  Value(bool b) : type_(Type::Bool), bool_(b) {}
  Value(double n) : type_(Type::Number), num_(n) {}
  Value(int n) : type_(Type::Number), num_(static_cast<double>(n)) {}
  Value(std::int64_t n) : type_(Type::Number), num_(static_cast<double>(n)) {}
  Value(const char* s) : type_(Type::String), str_(s) {}
  Value(std::string s) : type_(Type::String), str_(std::move(s)) {}

  static Value array() {
    Value v;
    v.type_ = Type::Array;
    return v;
  }
  static Value object() {
    Value v;
    v.type_ = Type::Object;
    return v;
  }

  Type type() const noexcept { return type_; }
  bool isNull() const noexcept { return type_ == Type::Null; }
  bool isBool() const noexcept { return type_ == Type::Bool; }
  bool isNumber() const noexcept { return type_ == Type::Number; }
  bool isString() const noexcept { return type_ == Type::String; }
  bool isArray() const noexcept { return type_ == Type::Array; }
  bool isObject() const noexcept { return type_ == Type::Object; }

  bool asBool(bool fallback = false) const { return isBool() ? bool_ : fallback; }
  double asNumber(double fallback = 0.0) const { return isNumber() ? num_ : fallback; }
  int asInt(int fallback = 0) const {
    return isNumber() ? static_cast<int>(num_) : fallback;
  }
  const std::string& asString(const std::string& fallback = kEmpty()) const {
    return isString() ? str_ : fallback;
  }

  // Array access/mutation.
  const std::vector<Value>& items() const { return arr_; }
  std::size_t size() const { return isArray() ? arr_.size() : (isObject() ? obj_.size() : 0); }
  void push(Value v) {
    type_ = Type::Array;
    arr_.push_back(std::move(v));
  }
  const Value& at(std::size_t i) const { return arr_[i]; }

  // Object access/mutation. operator[] on a non-object turns it into one (builder convenience).
  bool contains(const std::string& key) const { return find(key) != nullptr; }
  const Value* get(const std::string& key) const { return find(key); }
  Value& operator[](const std::string& key) {
    if (type_ != Type::Object) type_ = Type::Object;
    for (auto& [k, v] : obj_) {
      if (k == key) return v;
    }
    obj_.emplace_back(key, Value{});
    return obj_.back().second;
  }
  const std::vector<std::pair<std::string, Value>>& members() const { return obj_; }

  std::string dump() const;  // serialize

 private:
  static const std::string& kEmpty() {
    static const std::string e;
    return e;
  }
  const Value* find(const std::string& key) const {
    if (type_ != Type::Object) return nullptr;
    for (const auto& [k, v] : obj_) {
      if (k == key) return &v;
    }
    return nullptr;
  }

  Type type_;
  bool bool_ = false;
  double num_ = 0.0;
  std::string str_;
  std::vector<Value> arr_;
  std::vector<std::pair<std::string, Value>> obj_;
};

// Parses a JSON document. Returns nullopt on malformed input (with an optional error message).
std::optional<Value> parse(const std::string& text, std::string* error = nullptr);

}  // namespace qorvix::api::json
