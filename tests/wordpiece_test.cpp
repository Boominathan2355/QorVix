#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"

#include "gguf_builder.hpp"

using namespace qorvix::tokenizer;
using qorvix::gguf::test::GgufBuilder;

namespace {

// A miniature uncased BERT vocabulary. The pieces are chosen so each test can distinguish a
// specific failure: "play"/"##ing"/"##ed" exercise continuation, "cafe" exercises accent folding
// (the vocab has no "café"), and "zzz" is absent entirely so it must fall to [UNK].
Tokenizer tinyWpm(bool lowercase = true) {
  std::vector<std::string> vocab{
      "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",  // 0..4
      "hello", "world", "play",  "##ing", "##ed",    // 5..9
      "cafe",  "the",   ",",     ".",     "?",       // 10..14
      "un",    "##aff", "##able", "a",    "##b",     // 15..19
  };
  SpecialTokens sp;
  sp.pad = 0;
  sp.unk = 1;
  sp.cls = 2;
  sp.sep = 3;
  sp.mask = 4;
  sp.addBos = true;
  sp.addEos = true;
  return Tokenizer(TokenizerModel::WordPiece, vocab, {}, {}, sp, lowercase);
}

std::vector<int> body(const Tokenizer& t, const std::string& text) {
  // encode() wraps in [CLS]/[SEP]; strip them so a case can assert on the pieces alone.
  std::vector<int> ids = t.encode(text, true);
  return std::vector<int>(ids.begin() + 1, ids.end() - 1);
}

}  // namespace

TEST_CASE("wordpiece splits a known word into continuation pieces", "[tokenizer]") {
  const Tokenizer t = tinyWpm();
  REQUIRE(body(t, "playing") == std::vector<int>{7, 8});   // play + ##ing
  REQUIRE(body(t, "played") == std::vector<int>{7, 9});    // play + ##ed
  REQUIRE(body(t, "unaffable") == std::vector<int>{15, 16, 17});
}

TEST_CASE("wordpiece prefers the longest matching prefix", "[tokenizer]") {
  // "a" and "##b" both exist, so a shortest-first matcher would produce [a, ##b] for "ab" — but
  // so does longest-first here, since "ab" is not in the vocab. The distinguishing case is
  // "playing": a shortest-first matcher would try "p" (absent) and fall to [UNK] immediately.
  const Tokenizer t = tinyWpm();
  REQUIRE(body(t, "ab") == std::vector<int>{18, 19});
  REQUIRE(body(t, "playing").size() == 2);
}

TEST_CASE("wordpiece falls back to unk for the whole word, not the pieces found so far",
          "[tokenizer]") {
  // "playzzz" starts with a coverable prefix ("play") but cannot be completed. Emitting [play]
  // and stopping would change the token count — and for a mean-pooled model, the token count
  // changes the embedding itself. HuggingFace emits a single [UNK]; so do we.
  const Tokenizer t = tinyWpm();
  REQUIRE(body(t, "zzz") == std::vector<int>{1});
  REQUIRE(body(t, "playzzz") == std::vector<int>{1});
}

TEST_CASE("wordpiece lowercases and strips accents when the vocabulary is uncased",
          "[tokenizer]") {
  const Tokenizer t = tinyWpm(/*lowercase=*/true);
  // The vocab holds only "cafe" — every one of these must fold onto it.
  REQUIRE(body(t, "cafe") == std::vector<int>{10});
  REQUIRE(body(t, "CAFE") == std::vector<int>{10});
  REQUIRE(body(t, "Café") == std::vector<int>{10});
  REQUIRE(body(t, "CAFÉ") == std::vector<int>{10});

  // With folding off, the accented and capitalized forms are genuinely out of vocabulary. This is
  // the branch that proves the folding above is doing the work, not the matcher.
  const Tokenizer cased = tinyWpm(/*lowercase=*/false);
  REQUIRE(body(cased, "cafe") == std::vector<int>{10});
  REQUIRE(body(cased, "Café") == std::vector<int>{1});
}

TEST_CASE("wordpiece splits punctuation into standalone tokens", "[tokenizer]") {
  const Tokenizer t = tinyWpm();
  // Without punctuation splitting, "world." is one word, is not in the vocab, and becomes [UNK] —
  // so this is not a cosmetic step.
  REQUIRE(body(t, "hello, world.") == std::vector<int>{5, 12, 6, 13});
  REQUIRE(body(t, "hello?") == std::vector<int>{5, 14});
}

TEST_CASE("wordpiece collapses arbitrary whitespace into word boundaries", "[tokenizer]") {
  const Tokenizer t = tinyWpm();
  const std::vector<int> expected{5, 6};
  REQUIRE(body(t, "hello world") == expected);
  REQUIRE(body(t, "  hello \t\n world  ") == expected);
  REQUIRE(body(t, "hello\xC2\xA0world") == expected);  // U+00A0 no-break space is whitespace
}

TEST_CASE("wordpiece deletes control characters without splitting the word", "[tokenizer]") {
  // BERT's _clean_text drops controls with `continue` — it does NOT substitute a separator, so
  // the surrounding text joins into one word. Here that word ("helloworld") is out of vocabulary
  // and becomes [UNK]. Treating a control as a boundary instead would yield [hello, world] and
  // silently disagree with every reference implementation on any text containing one.
  const Tokenizer t = tinyWpm();
  REQUIRE(body(t, "hello\x01world") == std::vector<int>{1});
  REQUIRE(body(t, "hello\x01") == std::vector<int>{5});
}

TEST_CASE("encode wraps wordpiece output in cls and sep even when bos and eos are unset",
          "[tokenizer]") {
  // The case that today's incidental bos/eos wrapping would get wrong: a vocabulary where the
  // GGUF supplied cls/sep but no bos/eos at all.
  const Tokenizer t = tinyWpm();
  REQUIRE(t.special().bos == -1);
  REQUIRE(t.special().eos == -1);
  const std::vector<int> ids = t.encode("hello world", true);
  REQUIRE(ids == std::vector<int>{2, 5, 6, 3});
}

TEST_CASE("a bert gguf builds a wordpiece tokenizer with backfilled specials", "[tokenizer]") {
  GgufBuilder b(3);
  b.str("general.architecture", "bert")
      .str("tokenizer.ggml.model", "bert")
      .stringArray("tokenizer.ggml.tokens", {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello"});
  // Deliberately no cls/sep/unk id keys — they must be resolved from the vocabulary by piece text.
  const auto bytes = b.build();
  const auto file = qorvix::gguf::GgufFile::parse(bytes);

  std::string err;
  auto tok = Tokenizer::fromGguf(file, err);
  REQUIRE(tok.has_value());
  REQUIRE(err.empty());
  REQUIRE(tok->model() == TokenizerModel::WordPiece);
  REQUIRE(tok->special().cls == 2);
  REQUIRE(tok->special().sep == 3);
  REQUIRE(tok->special().unk == 1);
  REQUIRE(tok->special().pad == 0);
  // BERT always wraps, so both flags default on for WordPiece regardless of the add_* keys.
  REQUIRE(tok->special().addBos);
  REQUIRE(tok->special().addEos);
  REQUIRE(tok->encode("hello", true) == std::vector<int>{2, 4, 3});
}

TEST_CASE("decode rejoins wordpiece continuations and skips cls and sep", "[tokenizer]") {
  const Tokenizer t = tinyWpm();
  REQUIRE(t.decode(t.encode("hello world", true)) == "hello world");
  REQUIRE(t.decode(t.encode("playing", true)) == "playing");
  // Keeping the specials shows they were there to begin with.
  REQUIRE(t.decode(t.encode("hello", true), /*skipSpecial=*/false) == "[CLS] hello [SEP]");
}
