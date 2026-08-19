#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "qorvix/runtime/multimodal.hpp"
#include "qorvix/runtime/ops.hpp"
#include "qorvix/runtime/text_model.hpp"
#include "qorvix/runtime/weights.hpp"
#include "qorvix/scheduler/scheduler.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"

using namespace qorvix::runtime;
using qorvix::memory::SessionId;  // qorvix::memory is not pulled in by the runtime using-directive
using qorvix::tokenizer::SpecialTokens;
using qorvix::tokenizer::Tokenizer;
using qorvix::tokenizer::TokenizerModel;

// Phase 11b-2: the input-embedding seam and the prompt assembly built on it.
//
// The property that matters most is the one `qorvix vlm-check` tier 1 asserts on a real model and
// that is pinned here on a toy one: feeding a token's OWN embedding row through forwardEmbedding
// must reproduce, bit for bit, what feeding its id through forward produced. Every way the splice
// could be wrong — a stride error, a missed scale, a KV write that lands in the wrong slot —
// breaks that equality, and nothing else in the pipeline would.

namespace {

ModelConfig toyConfig() {
  ModelConfig c;
  c.architecture = "llama";
  c.vocabSize = 4;
  c.contextLength = 64;
  c.embeddingLength = 4;
  c.blockCount = 1;
  c.feedForwardLength = 4;
  c.headCount = 2;
  c.headCountKv = 2;
  c.ropeDimensionCount = 2;
  c.normEpsilon = 1e-5f;
  c.ropeMode = ops::RopeMode::Neox;
  return c;
}

// A layer with small non-zero weights: an all-zero stack would make every position produce the
// same logits, and the identity assertions below would then hold for the wrong reason.
LayerWeights toyLayer(const ModelConfig& c) {
  const int d = c.embeddingLength, kv = c.kvDim(), ffn = c.feedForwardLength;
  auto ramp = [](int rows, int cols, float scale) {
    std::vector<float> v(static_cast<std::size_t>(rows) * cols);
    for (std::size_t i = 0; i < v.size(); ++i)
      v[i] = scale * (static_cast<float>(i % 7) - 3.0f);
    return v;
  };
  LayerWeights L;
  L.attnNorm.assign(d, 1.0f);
  L.wq = WeightMat::f32(ramp(d, d, 0.10f), d, d);
  L.wk = WeightMat::f32(ramp(kv, d, 0.07f), kv, d);
  L.wv = WeightMat::f32(ramp(kv, d, 0.05f), kv, d);
  L.wo = WeightMat::f32(ramp(d, d, 0.03f), d, d);
  L.ffnNorm.assign(d, 1.0f);
  L.ffnGate = WeightMat::f32(ramp(ffn, d, 0.09f), ffn, d);
  L.ffnUp = WeightMat::f32(ramp(ffn, d, 0.04f), ffn, d);
  L.ffnDown = WeightMat::f32(ramp(d, ffn, 0.06f), d, ffn);
  return L;
}

TextModel toyModel(std::uint32_t maxSessions = 4) {
  ModelConfig c = toyConfig();
  const int d = c.embeddingLength, vocab = c.vocabSize;
  Weights w;
  std::vector<float> emb(static_cast<std::size_t>(vocab) * d);
  for (std::size_t i = 0; i < emb.size(); ++i) emb[i] = 0.1f * static_cast<float>(i + 1);
  w.tokenEmbd = WeightMat::f32(std::move(emb), vocab, d);
  w.layers = {toyLayer(c)};
  w.outputNorm.assign(d, 1.0f);
  std::vector<float> out(static_cast<std::size_t>(vocab) * d);
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = 0.2f * static_cast<float>((i % 5) + 1);
  w.output = WeightMat::f32(std::move(out), vocab, d);
  return TextModel(c, std::move(w), /*maxSeqLen=*/64, maxSessions);
}

Tokenizer toyTok() {
  SpecialTokens sp;
  sp.bos = 0;
  sp.eos = 1;
  return Tokenizer(TokenizerModel::Bpe, {"<s>", "</s>", "A", "B"}, {}, {}, sp);
}

// The embedding row for `id`, which is what forward() would have looked up internally.
std::vector<float> embRow(int id) {
  const int d = 4;
  std::vector<float> row(d);
  for (int j = 0; j < d; ++j) row[j] = 0.1f * static_cast<float>(id * d + j + 1);
  return row;
}

}  // namespace

TEST_CASE("forwardEmbedding reproduces forward exactly for a token's own row", "[multimodal]") {
  TextModel model = toyModel();
  REQUIRE(model.acceptsInputEmbeddings());

  for (int id = 0; id < 4; ++id) {
    const auto a = model.openSession();
    const auto b = model.openSession();
    const std::vector<float> byId = model.forward(a, id, 0);
    const auto row = embRow(id);
    const std::vector<float> byEmbedding = model.forwardEmbedding(b, row.data(), 0);
    model.closeSession(a);
    model.closeSession(b);

    REQUIRE(byId.size() == byEmbedding.size());
    // Bit-identical, not approximately equal: it is the same arithmetic on the same inputs.
    REQUIRE(byId == byEmbedding);
  }
}

TEST_CASE("forwardEmbedding stays exact across a full prefill", "[multimodal]") {
  // A splice that writes the right logits at pos 0 but corrupts the KV cache would pass the
  // single-step check and fail here, from pos 1 onward.
  TextModel model = toyModel();
  const std::vector<int> ids{2, 3, 2, 1};

  const auto a = model.openSession();
  const auto b = model.openSession();
  std::vector<float> lastA, lastB;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    lastA = model.forward(a, ids[i], static_cast<int>(i));
    const auto row = embRow(ids[i]);
    lastB = model.forwardEmbedding(b, row.data(), static_cast<int>(i));
  }
  model.closeSession(a);
  model.closeSession(b);
  REQUIRE(lastA == lastB);
}

TEST_CASE("an engine that does not accept input embeddings refuses loudly", "[multimodal]") {
  // The default seam implementation. A silent zero vector here would generate fluent text about
  // an image the decoder never received, which is precisely the failure worth being loud about.
  struct TextOnly final : IInferenceEngine {
    SessionId openSession() override { return 1; }
    void closeSession(SessionId) override {}
    void resetSession(SessionId) override {}
    const std::vector<float>& forward(SessionId, int, int) override { return buf_; }
    std::uint32_t maxSeqLen() const override { return 8; }
    const ModelConfig& config() const override { return cfg_; }
    std::string backendName() const override { return "text-only"; }
    std::vector<float> buf_{0.0f};
    ModelConfig cfg_{};
  } engine;

  REQUIRE_FALSE(engine.acceptsInputEmbeddings());
  const float row[4] = {0, 0, 0, 0};
  REQUIRE_THROWS_AS(engine.forwardEmbedding(1, row, 0), std::logic_error);
}

TEST_CASE("splitOnImageMarker always returns markers+1 segments", "[multimodal]") {
  REQUIRE(splitOnImageMarker("no marker here") == std::vector<std::string>{"no marker here"});
  REQUIRE(splitOnImageMarker("a<image>b") == std::vector<std::string>{"a", "b"});
  // Leading and trailing markers produce empty edge segments rather than being swallowed — the
  // count is what pairs segments with images.
  REQUIRE(splitOnImageMarker("<image>tail") == std::vector<std::string>{"", "tail"});
  REQUIRE(splitOnImageMarker("head<image>") == std::vector<std::string>{"head", ""});
  REQUIRE(splitOnImageMarker("<image><image>") == std::vector<std::string>{"", "", ""});
}

TEST_CASE("MultimodalPrompt interleaves ids and embeddings in order", "[multimodal]") {
  MultimodalPrompt mm(/*dModel=*/4);
  std::string err;
  const std::vector<float> features{1, 2, 3, 4, 5, 6, 7, 8};  // 2 tokens x 4 dims

  mm.addTokens({7, 8});
  REQUIRE(mm.addImage(features, /*tokens=*/2, /*dim=*/4, err));
  mm.addTokens({9});

  REQUIRE(mm.size() == 5);
  REQUIRE(mm.textTokens() == 3);
  REQUIRE(mm.imageTokens() == 2);
  REQUIRE(mm.imageCount() == 1);
  REQUIRE(mm.hasImages());

  const auto steps = mm.steps();
  REQUIRE(steps.size() == 5);
  REQUIRE_FALSE(steps[0].isEmbedding());
  REQUIRE(steps[0].token == 7);
  REQUIRE(steps[2].isEmbedding());
  REQUIRE(steps[2].embedding[0] == 1.0f);
  // Row 1 of the image, so the second patch must start at feature index 4 — an off-by-one in the
  // stride would hand the decoder a window straddling two patches.
  REQUIRE(steps[3].isEmbedding());
  REQUIRE(steps[3].embedding[0] == 5.0f);
  REQUIRE_FALSE(steps[4].isEmbedding());
  REQUIRE(steps[4].token == 9);

  // Image positions contribute nothing to the repetition-penalty history.
  REQUIRE(mm.textIds() == std::vector<int>{7, 8, 9});
}

TEST_CASE("MultimodalPrompt rejects a projector whose width is not the decoder's", "[multimodal]") {
  MultimodalPrompt mm(/*dModel=*/4);
  std::string err;
  const std::vector<float> features(6, 1.0f);
  REQUIRE_FALSE(mm.addImage(features, /*tokens=*/2, /*dim=*/3, err));
  REQUIRE(err.find("decoder expects 4") != std::string::npos);
  REQUIRE(mm.empty());
}

TEST_CASE("partsFromPrompt places images at the markers", "[multimodal]") {
  std::string err;
  auto image = [] { return PromptPart::fromImage(std::vector<float>(4, 1.0f), 1, 4); };

  SECTION("marker splits the text around the image") {
    std::vector<PromptPart> out;
    REQUIRE(partsFromPrompt("before <image> after", {image()}, out, err));
    REQUIRE(out.size() == 3);
    REQUIRE(out[0].text == "before ");
    REQUIRE(out[1].isImage());
    REQUIRE(out[2].text == " after");
  }

  SECTION("no marker prepends the images (the LLaVA convention for a bare prompt)") {
    std::vector<PromptPart> out;
    REQUIRE(partsFromPrompt("describe it", {image()}, out, err));
    REQUIRE(out.size() == 2);
    REQUIRE(out[0].isImage());
    REQUIRE(out[1].text == "describe it");
  }

  SECTION("a marker/image count mismatch is an error, not a silent drop") {
    std::vector<PromptPart> out;
    REQUIRE_FALSE(partsFromPrompt("<image> and <image>", {image()}, out, err));
    REQUIRE(err.find("2") != std::string::npos);
  }
}

TEST_CASE("buildPrompt keeps BOS at position 0 even when the image leads", "[multimodal]") {
  Tokenizer tok = toyTok();
  std::string err;
  std::vector<PromptPart> parts;
  REQUIRE(partsFromPrompt("A", {PromptPart::fromImage(std::vector<float>(8, 1.0f), 2, 4)}, parts,
                          err));
  REQUIRE(parts[0].isImage());  // no marker -> image first

  MultimodalPrompt mm(/*dModel=*/4);
  REQUIRE(buildPrompt(parts, tok, /*addBos=*/true, mm, err));

  const auto steps = mm.steps();
  REQUIRE(steps.size() >= 3);
  // BOS must precede the image, not follow it: a decoder that sees patches before its
  // start-of-sequence token is being fed a sequence shape it was never trained on.
  REQUIRE_FALSE(steps[0].isEmbedding());
  REQUIRE(steps[0].token == tok.special().bos);
  REQUIRE(steps[1].isEmbedding());
  REQUIRE(steps[2].isEmbedding());
}

TEST_CASE("buildPrompt matches plain encode for a text-only prompt", "[multimodal]") {
  // Phase 11b-2 rerouted every text request through this assembly, so the text-only sequence must
  // be unchanged — a shifted BOS or a doubled prefix would degrade ordinary chat silently.
  Tokenizer tok = toyTok();
  std::string err;
  MultimodalPrompt mm(/*dModel=*/4);
  REQUIRE(buildPrompt({PromptPart::fromText("AB")}, tok, /*addBos=*/true, mm, err));
  REQUIRE(mm.textIds() == tok.encode("AB", /*addBos=*/true));
  REQUIRE_FALSE(mm.hasImages());
}

TEST_CASE("scheduler prefills an image request through the embedding seam", "[multimodal]") {
  using namespace qorvix::scheduler;
  TextModel model = toyModel();
  Tokenizer tok = toyTok();
  Scheduler sched(model, tok, {/*maxConcurrent=*/2});

  RequestParams p;
  p.maxNewTokens = 2;
  p.sampling.temperature = 0.0f;  // greedy
  p.addBos = false;

  std::string err;
  const std::vector<PromptPart> parts{
      PromptPart::fromText("A"),
      PromptPart::fromImage(std::vector<float>(3 * 4, 0.25f), /*tokens=*/3, /*dim=*/4),
      PromptPart::fromText("B")};
  const RequestId id = sched.submitParts(parts, p, err);
  REQUIRE(id != 0);
  REQUIRE(err.empty());

  auto results = sched.runToCompletion();
  REQUIRE(results.size() == 1);
  REQUIRE_FALSE(results[0].rejected);
  // prompt_tokens counts PREFILL POSITIONS, so the 3 image patches are included alongside "A"
  // and "B" — reporting only the text would understate the context the model actually consumed.
  REQUIRE(results[0].promptTokens == 5);
  REQUIRE(results[0].tokens.size() == 2);
}

TEST_CASE("scheduler refuses images on an engine that cannot take embeddings", "[multimodal]") {
  using namespace qorvix::scheduler;
  struct TextOnly final : IInferenceEngine {
    SessionId openSession() override { return 1; }
    void closeSession(SessionId) override {}
    void resetSession(SessionId) override {}
    const std::vector<float>& forward(SessionId, int, int) override { return buf_; }
    std::uint32_t maxSeqLen() const override { return 8; }
    const ModelConfig& config() const override { return cfg_; }
    std::string backendName() const override { return "device"; }
    std::vector<float> buf_{0.0f};
    ModelConfig cfg_ = toyConfig();
  } engine;

  Tokenizer tok = toyTok();
  Scheduler sched(engine, tok, {/*maxConcurrent=*/1});
  std::string err;
  const std::vector<PromptPart> parts{
      PromptPart::fromImage(std::vector<float>(4, 1.0f), /*tokens=*/1, /*dim=*/4)};

  // Synchronously rejected, before admission: the HTTP layer needs a 400, not an empty completion.
  REQUIRE(sched.submitParts(parts, {}, err) == 0);
  REQUIRE(err.find("device") != std::string::npos);
  REQUIRE(sched.idle());

  // Text on the same engine is unaffected.
  REQUIRE(sched.submitParts({PromptPart::fromText("A")}, {}, err) != 0);
}

TEST_CASE("partsFromPrompt leaves a marker alone when there are no images", "[multimodal]") {
  // A text-only chat message that mentions "<image>" must reach the model verbatim — not be
  // rewritten, and not be rejected for a marker/image mismatch it never intended.
  std::string err;
  std::vector<PromptPart> out;
  REQUIRE(partsFromPrompt("what does <image> mean?", {}, out, err));
  REQUIRE(out.size() == 1);
  REQUIRE(out[0].text == "what does <image> mean?");
  REQUIRE_FALSE(out[0].isImage());
}
