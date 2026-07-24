#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "qorvix/runtime/inference_engine.hpp"

using namespace qorvix;

// Pins the IInferenceEngine::forwardBatch contract.
//
// The trap this exists for: every real engine returns a reference to ONE reused logits buffer
// (TextModel hands back its `logits_` member; GpuEngine hands back the GpuModel's host mirror).
// A forwardBatch that collects pointers across sequential forward() calls therefore ends up with
// every entry aliasing the same buffer, holding only the LAST step's logits — so every request in
// a decode round samples from the last request's distribution. It still runs, still produces
// plausible text, and single-request tests never notice. Hence this test.

namespace {

// Mimics the real engines exactly where it matters: one reused buffer, overwritten per call.
class ReusedBufferEngine final : public runtime::IInferenceEngine {
 public:
  memory::SessionId openSession() override { return ++last_; }
  void closeSession(memory::SessionId) override {}
  void resetSession(memory::SessionId) override {}

  const std::vector<float>& forward(memory::SessionId session, int token, int pos) override {
    ++calls;
    buf_.assign(3, static_cast<float>(session) * 1000.0f + static_cast<float>(token) * 10.0f +
                       static_cast<float>(pos));
    return buf_;
  }

  std::uint32_t maxSeqLen() const override { return 128; }
  const runtime::ModelConfig& config() const override { return cfg_; }
  std::string backendName() const override { return "test"; }

  int calls = 0;

 private:
  std::vector<float> buf_;
  runtime::ModelConfig cfg_{};
  memory::SessionId last_ = 0;
};

}  // namespace

TEST_CASE("forwardBatch returns one independent result per step", "[engine]") {
  ReusedBufferEngine eng;
  const std::vector<runtime::ForwardStep> steps{{1, 5, 0}, {2, 7, 1}, {3, 9, 2}};

  const auto out = eng.forwardBatch(steps);
  REQUIRE(out.size() == steps.size());
  REQUIRE(eng.calls == 3);

  // Each entry must carry ITS OWN step's logits, not the last step's.
  for (std::size_t i = 0; i < steps.size(); ++i) {
    const float want = static_cast<float>(steps[i].session) * 1000.0f +
                       static_cast<float>(steps[i].token) * 10.0f +
                       static_cast<float>(steps[i].pos);
    REQUIRE(out[i].size() == 3);
    REQUIRE(out[i][0] == want);
  }
  // And they must actually differ — the aliasing bug made all three identical.
  REQUIRE(out[0][0] != out[1][0]);
  REQUIRE(out[1][0] != out[2][0]);
}

TEST_CASE("forwardBatch results survive further engine calls", "[engine]") {
  // The results must be owned copies: a later forward() must not retroactively change them.
  ReusedBufferEngine eng;
  auto out = eng.forwardBatch({{1, 5, 0}, {2, 7, 1}});
  const float before0 = out[0][0], before1 = out[1][0];

  eng.forward(99, 99, 99);  // clobbers the shared buffer

  REQUIRE(out[0][0] == before0);
  REQUIRE(out[1][0] == before1);
}

TEST_CASE("forwardBatch handles the empty and single-step cases", "[engine]") {
  ReusedBufferEngine eng;
  REQUIRE(eng.forwardBatch({}).empty());
  REQUIRE(eng.calls == 0);

  const auto one = eng.forwardBatch({{4, 2, 6}});
  REQUIRE(one.size() == 1);
  REQUIRE(one[0][0] == 4.0f * 1000.0f + 2.0f * 10.0f + 6.0f);
}
