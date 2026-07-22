#include <catch2/catch_test_macros.hpp>

#include "qorvix/api/json.hpp"

using namespace qorvix::api::json;

TEST_CASE("parses scalars and round-trips", "[json]") {
  REQUIRE(parse("true")->asBool() == true);
  REQUIRE(parse("false")->asBool() == false);
  REQUIRE(parse("null")->isNull());
  REQUIRE(parse("42")->asInt() == 42);
  REQUIRE(parse("-3.5")->asNumber() == -3.5);
  REQUIRE(parse("\"hi\"")->asString() == "hi");
}

TEST_CASE("parses objects and arrays", "[json]") {
  auto v = parse(R"({"a":1,"b":[1,2,3],"c":{"d":true}})");
  REQUIRE(v.has_value());
  REQUIRE(v->isObject());
  REQUIRE(v->get("a")->asInt() == 1);
  REQUIRE(v->get("b")->isArray());
  REQUIRE(v->get("b")->size() == 3);
  REQUIRE(v->get("b")->at(2).asInt() == 3);
  REQUIRE(v->get("c")->get("d")->asBool() == true);
  REQUIRE(v->get("missing") == nullptr);
}

TEST_CASE("handles string escapes and unicode", "[json]") {
  auto v = parse(R"("line1\nline2\t\"quoted\" A")");
  REQUIRE(v.has_value());
  REQUIRE(v->asString() == "line1\nline2\t\"quoted\" A");
}

TEST_CASE("rejects malformed input", "[json]") {
  REQUIRE_FALSE(parse("{").has_value());
  REQUIRE_FALSE(parse("[1,2").has_value());
  REQUIRE_FALSE(parse("{\"a\":}").has_value());
  REQUIRE_FALSE(parse("nul").has_value());
  REQUIRE_FALSE(parse("1 2").has_value());  // trailing
}

TEST_CASE("serializes with escaping and preserves object order", "[json]") {
  Value root = Value::object();
  root["z"] = 1;
  root["a"] = "he\"llo";
  root["arr"] = Value::array();
  root["arr"].push(true);
  root["arr"].push(nullptr);
  REQUIRE(root.dump() == R"({"z":1,"a":"he\"llo","arr":[true,null]})");
}

TEST_CASE("dump then parse is stable", "[json]") {
  auto v = parse(R"({"n":3.14159,"s":"x","b":false,"a":[1,2]})");
  REQUIRE(v.has_value());
  auto again = parse(v->dump());
  REQUIRE(again.has_value());
  REQUIRE(again->get("n")->asNumber() == 3.14159);
  REQUIRE(again->get("a")->at(1).asInt() == 2);
}
