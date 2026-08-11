#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"

#include "gguf_builder.hpp"

using namespace qorvix::tokenizer;
using qorvix::gguf::test::GgufBuilder;

namespace {

// U+2581, the SentencePiece space marker.
const std::string kMark = "\xE2\x96\x81";

// Two miniature uncased BERT vocabularies, one per WordPiece convention.
//
//   HuggingFace:    word-initial "play",  continuation "##ing"
//   llama.cpp GGUF: word-initial "▁play", continuation bare "ing"
//
// Both exist in the wild and picking the wrong one is SILENT: every word still resolves to
// something (usually [UNK], occasionally a real-but-wrong id, since a bare continuation piece is
// itself a valid vocabulary entry), so the embedding stays finite and unit-norm while meaning
// nothing. Real bert GGUFs on disk use the marker form — verified against bge-small-en-v1.5.
//
// Pieces are chosen so each test names a specific failure: play/ing/ed exercise continuation,
// "cafe" exercises accent folding (there is no "café"), and "zzz" is absent so it must fall to
// [UNK]. Note "ing" is present BARE in the marker vocab and as "##ing" in the other, so a
// tokenizer that guesses the wrong convention still finds pieces — just the wrong ones.
//
// Punctuation carries the marker too, which is what the real vocabulary does: bge-small holds
// "▁," at 1010 (matching bert-base-uncased) and a bare "," only at 29623, as a continuation piece.
Tokenizer tinyWpm(bool spaceMarker, bool lowercase = true) {
  const std::string m = spaceMarker ? kMark : "";
  const std::string c = spaceMarker ? "" : "##";
  std::vector<std::string> vocab{
      "[PAD]",     "[UNK]",       "[CLS]",       "[SEP]",  "[MASK]",   // 0..4
      m + "hello", m + "world",   m + "play",    c + "ing", c + "ed",  // 5..9
      m + "cafe",  m + "the",     m + ",",       m + ".",   m + "?",   // 10..14
      m + "un",    c + "aff",     c + "able",    m + "a",   c + "b",   // 15..19
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

// encode() wraps in [CLS]/[SEP]; strip them so a case can assert on the pieces alone.
std::vector<int> body(const Tokenizer& t, const std::string& text) {
  std::vector<int> ids = t.encode(text, true);
  return std::vector<int>(ids.begin() + 1, ids.end() - 1);
}

}  // namespace

TEST_CASE("wordpiece splits a known word into continuation pieces", "[tokenizer]") {
  for (bool marker : {false, true}) {
    const Tokenizer t = tinyWpm(marker);
    REQUIRE(body(t, "playing") == std::vector<int>{7, 8});
    REQUIRE(body(t, "played") == std::vector<int>{7, 9});
    REQUIRE(body(t, "unaffable") == std::vector<int>{15, 16, 17});
    REQUIRE(body(t, "ab") == std::vector<int>{18, 19});
  }
}

TEST_CASE("wordpiece falls back to unk for the whole word, not the pieces found so far",
          "[tokenizer]") {
  // "playzzz" starts with a coverable prefix but cannot be completed. Emitting [play] and stopping
  // would change the token count — and for a mean-pooled model the token count changes the
  // embedding itself. HuggingFace emits a single [UNK]; so do we.
  for (bool marker : {false, true}) {
    const Tokenizer t = tinyWpm(marker);
    REQUIRE(body(t, "zzz") == std::vector<int>{1});
    REQUIRE(body(t, "playzzz") == std::vector<int>{1});
  }
}

TEST_CASE("wordpiece lowercases and strips accents when the vocabulary is uncased",
          "[tokenizer]") {
  for (bool marker : {false, true}) {
    const Tokenizer t = tinyWpm(marker, /*lowercase=*/true);
    // The vocab holds only "cafe" — every one of these must fold onto it.
    REQUIRE(body(t, "cafe") == std::vector<int>{10});
    REQUIRE(body(t, "CAFE") == std::vector<int>{10});
    REQUIRE(body(t, "Café") == std::vector<int>{10});
    REQUIRE(body(t, "CAFÉ") == std::vector<int>{10});

    // With folding off, the accented form is genuinely out of vocabulary. This is the branch that
    // proves the folding above is doing the work, not the matcher.
    const Tokenizer cased = tinyWpm(marker, /*lowercase=*/false);
    REQUIRE(body(cased, "cafe") == std::vector<int>{10});
    REQUIRE(body(cased, "Café") == std::vector<int>{1});
  }
}

TEST_CASE("wordpiece splits punctuation into standalone tokens", "[tokenizer]") {
  // Without punctuation splitting, "world." is one word, is not in the vocab, and becomes [UNK].
  for (bool marker : {false, true}) {
    const Tokenizer t = tinyWpm(marker);
    REQUIRE(body(t, "hello, world.") == std::vector<int>{5, 12, 6, 13});
    REQUIRE(body(t, "hello?") == std::vector<int>{5, 14});
  }
}

TEST_CASE("wordpiece collapses arbitrary whitespace into word boundaries", "[tokenizer]") {
  const Tokenizer t = tinyWpm(/*spaceMarker=*/true);
  const std::vector<int> expected{5, 6};
  REQUIRE(body(t, "hello world") == expected);
  REQUIRE(body(t, "  hello \t\n world  ") == expected);
  REQUIRE(body(t, "hello\xC2\xA0world") == expected);  // U+00A0 no-break space
}

TEST_CASE("wordpiece deletes control characters without splitting the word", "[tokenizer]") {
  // BERT's _clean_text drops controls with `continue` — it does NOT substitute a separator, so the
  // surrounding text joins into one word, here out of vocabulary. Treating a control as a boundary
  // would silently disagree with every reference implementation on any text containing one.
  const Tokenizer t = tinyWpm(/*spaceMarker=*/true);
  REQUIRE(body(t, "hello\x01world") == std::vector<int>{1});
  REQUIRE(body(t, "hello\x01") == std::vector<int>{5});
}

TEST_CASE("encode wraps wordpiece output in cls and sep even when bos and eos are unset",
          "[tokenizer]") {
  // The case that incidental bos/eos wrapping would get wrong: a vocabulary that supplied cls/sep
  // but no bos/eos at all.
  const Tokenizer t = tinyWpm(/*spaceMarker=*/true);
  REQUIRE(t.special().bos == -1);
  REQUIRE(t.special().eos == -1);
  REQUIRE(t.encode("hello world", true) == std::vector<int>{2, 5, 6, 3});
  REQUIRE(t.encode("", true) == std::vector<int>{2, 3});  // empty text still wraps
}

TEST_CASE("wordpiece splits CJK text one token per character", "[tokenizer]") {
  // CJK is not space-separated, so without per-character splitting a whole sentence becomes one
  // "word" and one [UNK]. Neither character is in this vocab, so the observable difference is the
  // COUNT: two [UNK]s, not one.
  const Tokenizer t = tinyWpm(/*spaceMarker=*/true);
  REQUIRE(body(t, "\xE4\xB8\xAD\xE6\x96\x87") == std::vector<int>{1, 1});
}

TEST_CASE("a wordpiece word longer than the limit becomes unk", "[tokenizer]") {
  const Tokenizer t = tinyWpm(/*spaceMarker=*/true);
  REQUIRE(body(t, std::string(150, 'a')) == std::vector<int>{1});
}

TEST_CASE("a bert gguf builds a wordpiece tokenizer with backfilled specials", "[tokenizer]") {
  GgufBuilder b(3);
  b.str("general.architecture", "bert")
      .str("tokenizer.ggml.model", "bert")
      .stringArray("tokenizer.ggml.tokens", {"[PAD]", "[UNK]", "[CLS]", "[SEP]", kMark + "hello"});
  // Deliberately no cls/sep/unk id keys — they must be resolved from the vocabulary by piece text.
  const auto file = qorvix::gguf::GgufFile::parse(b.build());

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
  for (bool marker : {false, true}) {
    const Tokenizer t = tinyWpm(marker);
    REQUIRE(t.decode(t.encode("hello world", true)) == "hello world");
    REQUIRE(t.decode(t.encode("playing", true)) == "playing");
    REQUIRE(t.decode(t.encode("hello, world.", true)) == "hello , world .");
  }
  // Keeping the specials shows they were there to begin with.
  const Tokenizer t = tinyWpm(/*spaceMarker=*/false);
  REQUIRE(t.decode(t.encode("hello", true), /*skipSpecial=*/false) == "[CLS] hello [SEP]");
}
