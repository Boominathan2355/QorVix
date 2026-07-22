#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "qorvix/memory/kv_cache.hpp"
#include "qorvix/runtime/sampler.hpp"

namespace qorvix::runtime {
class TextModel;
}
namespace qorvix::tokenizer {
class Tokenizer;
}

namespace qorvix::scheduler {

using RequestId = std::uint64_t;

struct RequestParams {
  int maxNewTokens = 128;
  runtime::SamplingParams sampling;
  std::uint64_t seed = 0;
  bool addBos = true;
  int priority = 0;  // higher is scheduled first
};

enum class RequestState { Waiting, Prefill, Decoding, Done };

struct RequestResult {
  RequestId id = 0;
  std::string text;
  std::vector<int> tokens;  // generated token ids
  int promptTokens = 0;
  bool hitEos = false;
  bool rejected = false;    // could not be admitted (KV pool exhausted)
};

struct SchedulerConfig {
  int maxConcurrent = 8;  // max requests decoding at once (bounds the KV pool / batch)
};

// Continuous-batching scheduler over one model + tokenizer (SPEC "Scheduler"). Requests are
// queued with priority, admitted up to maxConcurrent (each gets its own KV-cache session and is
// prefilled), then advanced one token per step; finished requests free their session so waiting
// ones are admitted — requests join and leave the running set without draining it.
//
// This is the orchestration layer; each step still runs per-session forwards sequentially (fusing
// them into one batched matmul is a later GPU optimization). Single-threaded / not thread-safe.
class Scheduler {
 public:
  Scheduler(runtime::TextModel& model, const tokenizer::Tokenizer& tok, SchedulerConfig config);
  ~Scheduler();

  // Enqueues a request; returns its id. `onToken`, if set, streams each decoded piece.
  RequestId submit(const std::string& prompt, const RequestParams& params,
                   std::function<void(RequestId, const std::string&)> onToken = {});

  bool idle() const;  // nothing waiting or running

  // Advances the schedule by one round: admit+prefill up to capacity, then decode one token for
  // every active request. Returns results for requests that finished this round.
  std::vector<RequestResult> step();

  // Convenience: run step() until everything submitted so far completes; returns all results.
  std::vector<RequestResult> runToCompletion();

  std::size_t waiting() const;
  std::size_t active() const;

 private:
  struct Request;

  void admitFromQueue(std::vector<RequestResult>& rejected);
  void retire(std::unique_ptr<Request> req, bool rejected, std::vector<RequestResult>& out);

  runtime::TextModel& model_;
  const tokenizer::Tokenizer& tok_;
  SchedulerConfig config_;
  RequestId nextId_ = 1;

  struct Queued {
    int priority;
    std::uint64_t seq;  // FIFO tiebreak
    std::unique_ptr<Request> req;
  };
  struct QueueCmp {
    bool operator()(const Queued& a, const Queued& b) const {
      if (a.priority != b.priority) return a.priority < b.priority;  // higher priority first
      return a.seq > b.seq;                                          // then FIFO
    }
  };
  std::priority_queue<Queued, std::vector<Queued>, QueueCmp> waiting_;
  std::uint64_t enqueueSeq_ = 0;
  std::vector<std::unique_ptr<Request>> active_;
};

}  // namespace qorvix::scheduler
