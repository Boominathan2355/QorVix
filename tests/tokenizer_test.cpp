#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <string>
#include <vector>

#include "qorvix/tokenizer/tokenizer.hpp"

using namespace qorvix::tokenizer;

namespace {

// SPM vocab where pairwise-greedy merging is unambiguous: "ab" (score 1) then "▁ab" (score 2)
// beat the single characters, so "ab" tokenizes to the one piece "▁ab".
Tokenizer spmToy() {
  std::vector<std::string> tokens = {
      "<unk>", "<s>", "</s>",                 // 0,1,2
      "\xE2\x96\x81", "a", "b", "ab", "\xE2\x96\x81" "ab",  // ▁ a b ab ▁ab
  };
  std::vector<float> scores(tokens.size(), 0.0f);
  scores[6] = 1.0f;  // ab
  scores[7] = 2.0f;  // ▁ab (preferred)
  SpecialTokens sp;
  sp.bos = 1;
  sp.eos = 2;
  sp.unk = 0;
  return Tokenizer(TokenizerModel::Spm, tokens, scores, {}, sp);
}

}  // namespace

TEST_CASE("SPM encode merges by score and prepends the space marker", "[tokenizer]") {
  auto t = spmToy();
  auto ids = t.encode("ab", /*addBos=*/false);
  REQUIRE(ids.size() == 1);
  REQUIRE(t.idToToken(ids[0]) == "\xE2\x96\x81" "ab");  // ▁ab
}

TEST_CASE("SPM decode turns the space marker back into a space", "[tokenizer]") {
  auto t = spmToy();
  REQUIRE(t.decode(t.encode("ab", false)) == " ab");  // leading space from the ▁ prefix
}

TEST_CASE("encode adds BOS when requested", "[tokenizer]") {
  auto t = spmToy();
  auto ids = t.encode("ab", /*addBos=*/true);
  REQUIRE(ids.front() == 1);  // <s>
  REQUIRE(t.decode(ids, /*skipSpecial=*/true) == " ab");
}

TEST_CASE("byte fallback for out-of-vocab characters", "[tokenizer]") {
  std::vector<std::string> tokens = {"<unk>", "\xE2\x96\x81", "a"};
  for (int b = 0; b < 256; ++b) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
    tokens.emplace_back(buf);
  }
  SpecialTokens sp;
  sp.unk = 0;
  Tokenizer t(TokenizerModel::Spm, tokens, {}, {}, sp);

  auto ids = t.encode("a?", false);  // 'a' in vocab, '?' via byte token
  REQUIRE(t.decode(ids) == " a?");   // round-trips through the byte fallback
}

TEST_CASE("BPE encode applies merges by rank and round-trips", "[tokenizer]") {
  // Byte-level: 'a','b','c' are printable (map to themselves); space maps to Ġ (0xC4 0xA0).
  std::vector<std::string> tokens = {"a", "b", "c", "ab", "abc", "\xC4\xA0", "\xC4\xA0" "a"};
  std::vector<std::string> merges = {"a b", "ab c", "\xC4\xA0 a"};  // ranks 0,1,2
  SpecialTokens sp;
  Tokenizer t(TokenizerModel::Bpe, tokens, {}, merges, sp);

  auto ids = t.encode("abc", false);
  REQUIRE(ids.size() == 1);
  REQUIRE(t.idToToken(ids[0]) == "abc");  // a+b -> ab, ab+c -> abc
  REQUIRE(t.decode(ids) == "abc");

  auto ids2 = t.encode(" a", false);  // leading space -> Ġ, merges with 'a'
  REQUIRE(t.decode(ids2) == " a");
}
