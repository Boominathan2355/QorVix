#include "qorvix/scheduler/scheduler.hpp"

#include <utility>

#include "qorvix/runtime/text_model.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"

namespace qorvix::scheduler {

struct Scheduler::Request {
  RequestId id = 0;
  RequestParams params;
  std::function<void(RequestId, const std::string&)> onToken;

  std::vector<int> promptTokens;
  RequestState state = RequestState::Waiting;
  memory::SessionId session = memory::kInvalidSession;

  int pos = 0;               // next KV position to fill
  int promptCursor = 0;      // next prompt token index to prefill
  int nextToken = 0;         // token to feed at `pos`
  int generatedCount = 0;
  std::unique_ptr<runtime::Sampler> sampler;

  RequestResult result;
};

Scheduler::Scheduler(runtime::TextModel& model, const tokenizer::Tokenizer& tok,
                     SchedulerConfig config)
    : model_(model), tok_(tok), config_(config) {}

Scheduler::~Scheduler() {
  for (auto& r : active_) {
    if (r->session != memory::kInvalidSession) model_.closeSession(r->session);
  }
}

RequestId Scheduler::submit(const std::string& prompt, const RequestParams& params,
                            std::function<void(RequestId, const std::string&)> onToken) {
  auto req = std::make_unique<Request>();
  req->id = nextId_++;
  req->params = params;
  req->onToken = std::move(onToken);
  req->promptTokens = tok_.encode(prompt, params.addBos);
  req->result.id = req->id;
  req->result.promptTokens = static_cast<int>(req->promptTokens.size());
  req->sampler = std::make_unique<runtime::Sampler>(params.sampling, params.seed);

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
    if (req->session == memory::kInvalidSession || req->promptTokens.empty()) {
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
    // Prefill: consume all prompt tokens (one forward each) before decoding begins.
    if (req->state == RequestState::Prefill) {
      while (req->promptCursor < static_cast<int>(req->promptTokens.size()) && req->pos < maxSeq) {
        const auto& logits = model_.forward(req->session, req->promptTokens[req->promptCursor], req->pos);
        req->promptCursor++;
        req->pos++;
        if (req->promptCursor == static_cast<int>(req->promptTokens.size())) {
          std::vector<float> l(logits.begin(), logits.end());
          req->nextToken = req->sampler->sample(l, req->promptTokens);
        }
      }
      req->state = RequestState::Decoding;
      continue;  // emit the first decoded token on the next step
    }

    if (req->state != RequestState::Decoding) continue;

    // Decode one token.
    if (req->nextToken == eos) {
      req->result.hitEos = true;
      req->state = RequestState::Done;
      continue;
    }
    if (req->generatedCount >= req->params.maxNewTokens || req->pos >= maxSeq) {
      req->state = RequestState::Done;
      continue;
    }

    const std::string piece = tok_.decodeToken(req->nextToken);
    req->result.text += piece;
    req->result.tokens.push_back(req->nextToken);
    req->generatedCount++;
    if (req->onToken) req->onToken(req->id, piece);

    std::vector<int> history = req->promptTokens;
    history.insert(history.end(), req->result.tokens.begin(), req->result.tokens.end());
    const auto& logits = model_.forward(req->session, req->nextToken, req->pos);
    req->pos++;
    std::vector<float> l(logits.begin(), logits.end());
    req->nextToken = req->sampler->sample(l, history);
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
