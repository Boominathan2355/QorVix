#include "qorvix/scheduler/scheduler.hpp"

#include <utility>

#include "qorvix/runtime/text_model.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"

namespace qorvix::scheduler {

struct Scheduler::Request {
  RequestId id = 0;
  RequestParams params;
  std::function<void(RequestId, const std::string&)> onToken;

  // Every request — text-only or multimodal — carries one MultimodalPrompt, so there is exactly
  // one prefill loop below rather than two that could drift apart. A text prompt is simply one
  // with no image chunks. Held by pointer because `inputs` points into its feature storage.
  std::unique_ptr<runtime::MultimodalPrompt> prompt;
  std::vector<runtime::InputToken> inputs;  // the prefill sequence; valid while `prompt` lives

  RequestState state = RequestState::Waiting;
  memory::SessionId session = memory::kInvalidSession;

  int pos = 0;               // next KV position to fill
  int promptCursor = 0;      // next prompt token index to prefill
  int nextToken = 0;         // token to feed at `pos`
  int generatedCount = 0;
  std::unique_ptr<runtime::Sampler> sampler;

  RequestResult result;
};

Scheduler::Scheduler(runtime::IInferenceEngine& model, const tokenizer::Tokenizer& tok,
                     SchedulerConfig config)
    : model_(model), tok_(tok), config_(config) {}

Scheduler::~Scheduler() {
  for (auto& r : active_) {
    if (r->session != memory::kInvalidSession) model_.closeSession(r->session);
  }
}

RequestId Scheduler::submit(const std::string& prompt, const RequestParams& params,
                            std::function<void(RequestId, const std::string&)> onToken) {
  std::string ignored;
  return submitParts({PromptPart::fromText(prompt)}, params, ignored, std::move(onToken));
}

RequestId Scheduler::submitParts(const std::vector<PromptPart>& parts, const RequestParams& params,
                                 std::string& error,
                                 std::function<void(RequestId, const std::string&)> onToken) {
  error.clear();
  bool anyImage = false;
  for (const auto& p : parts) anyImage = anyImage || p.isImage();
  if (anyImage && !model_.acceptsInputEmbeddings()) {
    error = "the " + model_.backendName() +
            " backend cannot take input embeddings, so it cannot serve images — run the CPU "
            "backend for vision-language chat";
    return 0;
  }

  auto req = std::make_unique<Request>();
  req->id = nextId_++;
  req->params = params;
  req->onToken = std::move(onToken);
  req->prompt = std::make_unique<runtime::MultimodalPrompt>(
      static_cast<int>(model_.config().embeddingLength));
  req->sampler = std::make_unique<runtime::Sampler>(params.sampling, params.seed);

  if (!runtime::buildPrompt(parts, tok_, params.addBos, *req->prompt, error)) return 0;

  // Built once, after every add — `steps()` hands back pointers into the prompt's feature storage,
  // which further adds would reallocate.
  req->inputs = req->prompt->steps();
  req->result.id = req->id;
  req->result.promptTokens = static_cast<int>(req->inputs.size());

  const RequestId id = req->id;
  waiting_.push(Queued{params.priority, enqueueSeq_++, std::move(req)});
  return id;
}

void Scheduler::retire(std::unique_ptr<Request> req, bool rejected,
                       std::vector<RequestResult>& out) {
  if (req->session != memory::kInvalidSession) {
    model_.closeSession(req->session);
    req->session = memory::kInvalidSession;
  }
  req->result.rejected = rejected;
  req->state = RequestState::Done;
  out.push_back(std::move(req->result));
}

void Scheduler::admitFromQueue(std::vector<RequestResult>& rejected) {
  while (!waiting_.empty() && active_.size() < static_cast<std::size_t>(config_.maxConcurrent)) {
    // priority_queue only exposes a const top(); move our own element out of it.
    auto& top = const_cast<Queued&>(waiting_.top());
    std::unique_ptr<Request> req = std::move(top.req);
    waiting_.pop();

    req->session = model_.openSession();
    if (req->session == memory::kInvalidSession || req->inputs.empty()) {
      retire(std::move(req), /*rejected=*/true, rejected);
      continue;
    }
    req->state = RequestState::Prefill;
    active_.push_back(std::move(req));
  }
}

std::vector<RequestResult> Scheduler::step() {
  std::vector<RequestResult> finished;
  admitFromQueue(finished);  // any admission rejections come back immediately

  const int eos = tok_.special().eos;
  const int maxSeq = static_cast<int>(model_.maxSeqLen());

  for (auto& req : active_) {
    // Prefill: consume every prompt position (one forward each) before decoding begins. Image
    // positions go through forwardInput's embedding branch; text positions through the id branch.
    if (req->state == RequestState::Prefill) {
      while (req->promptCursor < static_cast<int>(req->inputs.size()) && req->pos < maxSeq) {
        const auto& logits =
            model_.forwardInput(req->session, req->inputs[req->promptCursor], req->pos);
        req->promptCursor++;
        req->pos++;

        // Sample from the last position ACTUALLY run, which is not always the last position of
        // the prompt: a prompt longer than the context window exits this loop early. Leaving
        // nextToken at its initial 0 would then decode from token id 0 — a real risk now that a
        // single image contributes 576 positions. The copy is here rather than after the loop so
        // it happens once, not once per prefill step (the engine reuses one logits buffer, so it
        // cannot simply be held by reference).
        const bool endOfPrompt = req->promptCursor == static_cast<int>(req->inputs.size());
        if (endOfPrompt || req->pos >= maxSeq) {
          std::vector<float> l(logits.begin(), logits.end());
          req->nextToken = req->sampler->sample(l, req->prompt->textIds());
        }
      }
      req->state = RequestState::Decoding;
    }
  }

  // Batched decode round across all active decoding requests.
  std::vector<runtime::ForwardStep> decodeSteps;
  std::vector<Request*> decodingReqs;
  for (auto& req : active_) {
    if (req->state == RequestState::Decoding) {
      if (req->nextToken != eos && req->generatedCount < req->params.maxNewTokens && req->pos < maxSeq) {
        decodeSteps.push_back({req->session, req->nextToken, req->pos});
        decodingReqs.push_back(req.get());
      } else {
        if (req->nextToken == eos) req->result.hitEos = true;
        req->state = RequestState::Done;
      }
    }
  }

  if (!decodeSteps.empty()) {
    auto batchLogits = model_.forwardBatch(decodeSteps);
    for (std::size_t i = 0; i < decodingReqs.size(); ++i) {
      auto* req = decodingReqs[i];
      const std::string piece = tok_.decodeToken(req->nextToken);
      req->result.text += piece;
      req->result.tokens.push_back(req->nextToken);
      req->generatedCount++;
      if (req->onToken) req->onToken(req->id, piece);

      // Text ids only: image positions have no id, so nothing about them can be penalized.
      std::vector<int> history = req->prompt->textIds();
      history.insert(history.end(), req->result.tokens.begin(), req->result.tokens.end());
      req->pos++;
      req->nextToken = req->sampler->sample(batchLogits[i], history);
    }
  }

  // Retire finished requests, freeing their sessions so the queue can be admitted next round.
  std::vector<std::unique_ptr<Request>> stillActive;
  for (auto& req : active_) {
    if (req->state == RequestState::Done) {
      retire(std::move(req), /*rejected=*/false, finished);
    } else {
      stillActive.push_back(std::move(req));
    }
  }
  active_ = std::move(stillActive);
  return finished;
}

std::vector<RequestResult> Scheduler::runToCompletion() {
  std::vector<RequestResult> all;
  while (!idle()) {
    auto batch = step();
    for (auto& r : batch) all.push_back(std::move(r));
  }
  return all;
}

bool Scheduler::idle() const { return waiting_.empty() && active_.empty(); }
std::size_t Scheduler::waiting() const { return waiting_.size(); }
std::size_t Scheduler::active() const { return active_.size(); }

}  // namespace qorvix::scheduler
