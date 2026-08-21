#include "qorvix/agents/workflow.hpp"

#include <chrono>
#include <sstream>

namespace qorvix::agents {

namespace {
std::int64_t currentEpochMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
}  // namespace

std::string_view workflowPatternName(WorkflowPattern pattern) {
  switch (pattern) {
    case WorkflowPattern::Sequential: return "sequential";
    case WorkflowPattern::HierarchicalSupervisor: return "hierarchical_supervisor";
    case WorkflowPattern::RoundRobinBlackboard: return "round_robin_blackboard";
    case WorkflowPattern::ConsensusVoting: return "consensus_voting";
  }
  return "sequential";
}

WorkflowPattern parseWorkflowPattern(std::string_view name) {
  if (name == "hierarchical_supervisor" || name == "supervisor") return WorkflowPattern::HierarchicalSupervisor;
  if (name == "round_robin_blackboard" || name == "blackboard") return WorkflowPattern::RoundRobinBlackboard;
  if (name == "consensus_voting" || name == "consensus") return WorkflowPattern::ConsensusVoting;
  return WorkflowPattern::Sequential;
}

api::json::Value WorkflowTraceItem::toJson() const {
  auto v = api::json::Value::object();
  v["step"] = stepNumber;
  v["agent_id"] = agentId;
  v["agent_name"] = agentName;
  v["role"] = role;
  v["action"] = action;
  v["output"] = output;
  v["timestamp"] = timestamp;
  return v;
}

api::json::Value WorkflowResult::toJson() const {
  auto v = api::json::Value::object();
  v["success"] = success;
  v["final_answer"] = finalAnswer;
  v["total_steps"] = totalSteps;
  if (!error.empty()) v["error"] = error;

  auto tr = api::json::Value::array();
  for (const auto& item : trace) tr.push(item.toJson());
  v["trace"] = std::move(tr);

  auto arts = api::json::Value::array();
  for (const auto& art : artifacts) arts.push(art.toJson());
  v["artifacts"] = std::move(arts);

  return v;
}

// ---------------------------------------------------------------------------
// SequentialWorkflow
// ---------------------------------------------------------------------------
SequentialWorkflow::SequentialWorkflow(WorkflowConfig config,
                                       std::vector<std::shared_ptr<Agent>> agents,
                                       std::shared_ptr<Blackboard> blackboard)
    : config_(std::move(config)),
      agents_(std::move(agents)),
      blackboard_(std::move(blackboard)) {
  if (!blackboard_) blackboard_ = std::make_shared<Blackboard>();
  for (auto& a : agents_) {
    if (a) a->setBlackboard(blackboard_);
  }
}

WorkflowResult SequentialWorkflow::run(const std::string& goal,
                                       const std::function<void(const WorkflowTraceItem&)>& onProgress) {
  WorkflowResult res;
  res.success = false;
  if (agents_.empty()) {
    res.error = "Sequential workflow has no agents configured";
    return res;
  }

  blackboard_->postMessage("Workflow", "Starting sequential workflow for goal: " + goal, "plan");
  blackboard_->setState("goal", goal);

  std::string currentContext = "User Goal: " + goal;
  int globalStep = 0;

  for (std::size_t i = 0; i < agents_.size(); ++i) {
    auto& agent = agents_[i];
    if (!agent) continue;

    std::string prompt;
    if (i == 0) {
      prompt = currentContext;
    } else {
      prompt = "User Goal: " + goal + "\n\nPrevious Agent (" + agents_[i - 1]->name() + ") Output:\n" + currentContext +
               "\n\nPlease execute your role (" + agent->name() + ") to proceed toward the goal.";
    }

    auto agentRes = agent->run(prompt, {}, [&](const AgentStepTrace& stepTrace) {
      ++globalStep;
      WorkflowTraceItem item;
      item.stepNumber = globalStep;
      item.agentId = agent->id();
      item.agentName = agent->name();
      item.role = std::string(roleTypeName(agent->role().type));
      item.action = stepTrace.isFinal ? "Final Answer" : (stepTrace.toolCalls.empty() ? "Thought" : "Tool Call");
      item.output = stepTrace.response;
      item.timestamp = currentEpochMs();
      res.trace.push_back(item);
      if (onProgress) onProgress(item);
    });

    if (!agentRes.success && agentRes.finalAnswer.empty()) {
      res.error = "Agent " + agent->name() + " failed: " + agentRes.error;
      res.totalSteps = globalStep;
      return res;
    }

    currentContext = agentRes.finalAnswer;
    blackboard_->setState("step_" + std::to_string(i) + "_" + agent->name(), currentContext);
  }

  res.success = true;
  res.finalAnswer = currentContext;
  res.totalSteps = globalStep;
  res.artifacts = blackboard_->listArtifacts();
  blackboard_->postMessage("Workflow", "Sequential workflow completed successfully.", "info");
  blackboard_->setState("final_answer", currentContext);
  return res;
}

// ---------------------------------------------------------------------------
// SupervisorWorkflow
// ---------------------------------------------------------------------------
SupervisorWorkflow::SupervisorWorkflow(WorkflowConfig config,
                                       std::shared_ptr<Agent> supervisor,
                                       std::vector<std::shared_ptr<Agent>> workers,
                                       std::shared_ptr<Blackboard> blackboard)
    : config_(std::move(config)),
      supervisor_(std::move(supervisor)),
      workers_(std::move(workers)),
      blackboard_(std::move(blackboard)) {
  if (!blackboard_) blackboard_ = std::make_shared<Blackboard>();
  if (supervisor_) supervisor_->setBlackboard(blackboard_);
  for (auto& w : workers_) {
    if (w) w->setBlackboard(blackboard_);
  }
}

WorkflowResult SupervisorWorkflow::run(const std::string& goal,
                                       const std::function<void(const WorkflowTraceItem&)>& onProgress) {
  WorkflowResult res;
  res.success = false;
  if (!supervisor_) {
    res.error = "Supervisor workflow requires a supervisor agent";
    return res;
  }

  blackboard_->postMessage("Supervisor", "Decomposing goal: " + goal, "plan");
  blackboard_->setState("goal", goal);
  int globalStep = 0;

  // Phase 1: Supervisor creates plan and dispatches tasks to workers
  std::ostringstream planPrompt;
  planPrompt << "Goal: " << goal << "\n\nAvailable specialist workers:\n";
  for (const auto& w : workers_) {
    if (w) planPrompt << "- " << w->name() << " (" << roleTypeName(w->role().type) << "): " << w->role().description << "\n";
  }
  planPrompt << "\nFormulate the subtasks needed for each specialist worker to accomplish the goal.";

  auto planRes = supervisor_->run(planPrompt.str(), {}, [&](const AgentStepTrace& trace) {
    ++globalStep;
    WorkflowTraceItem item;
    item.stepNumber = globalStep;
    item.agentId = supervisor_->id();
    item.agentName = supervisor_->name();
    item.role = "Supervisor/Planner";
    item.action = "Plan Formulation";
    item.output = trace.response;
    item.timestamp = currentEpochMs();
    res.trace.push_back(item);
    if (onProgress) onProgress(item);
  });

  blackboard_->setState("supervisor_plan", planRes.finalAnswer);

  // Phase 2: Each worker executes its role based on supervisor plan
  std::ostringstream workerResults;
  for (auto& worker : workers_) {
    if (!worker) continue;
    std::string wPrompt = "Goal: " + goal + "\nSupervisor Plan:\n" + planRes.finalAnswer +
                          "\n\nPlease execute your specialized responsibilities (" + worker->name() + ").";

    auto wRes = worker->run(wPrompt, {}, [&](const AgentStepTrace& trace) {
      ++globalStep;
      WorkflowTraceItem item;
      item.stepNumber = globalStep;
      item.agentId = worker->id();
      item.agentName = worker->name();
      item.role = std::string(roleTypeName(worker->role().type));
      item.action = "Worker Execution";
      item.output = trace.response;
      item.timestamp = currentEpochMs();
      res.trace.push_back(item);
      if (onProgress) onProgress(item);
    });

    workerResults << "=== Worker [" << worker->name() << "] Output ===\n"
                  << wRes.finalAnswer << "\n\n";
    blackboard_->setState("worker_" + worker->name(), wRes.finalAnswer);
  }

  // Phase 3: Supervisor synthesizes the final result
  std::string synthPrompt = "Goal: " + goal + "\nSupervisor Plan:\n" + planRes.finalAnswer +
                            "\nSpecialist Worker Results:\n" + workerResults.str() +
                            "\nSynthesize a complete, definitive final answer addressing the original goal.";

  auto synthRes = supervisor_->run(synthPrompt, {}, [&](const AgentStepTrace& trace) {
    ++globalStep;
    WorkflowTraceItem item;
    item.stepNumber = globalStep;
    item.agentId = supervisor_->id();
    item.agentName = supervisor_->name();
    item.role = "Supervisor/Synthesizer";
    item.action = "Final Synthesis";
    item.output = trace.response;
    item.timestamp = currentEpochMs();
    res.trace.push_back(item);
    if (onProgress) onProgress(item);
  });

  res.success = true;
  res.finalAnswer = synthRes.finalAnswer;
  res.totalSteps = globalStep;
  res.artifacts = blackboard_->listArtifacts();
  blackboard_->setState("final_answer", synthRes.finalAnswer);
  return res;
}

// ---------------------------------------------------------------------------
// BlackboardWorkflow
// ---------------------------------------------------------------------------
BlackboardWorkflow::BlackboardWorkflow(WorkflowConfig config,
                                       std::vector<std::shared_ptr<Agent>> agents,
                                       std::shared_ptr<Blackboard> blackboard)
    : config_(std::move(config)),
      agents_(std::move(agents)),
      blackboard_(std::move(blackboard)) {
  if (!blackboard_) blackboard_ = std::make_shared<Blackboard>();
  for (auto& a : agents_) {
    if (a) a->setBlackboard(blackboard_);
  }
}

WorkflowResult BlackboardWorkflow::run(const std::string& goal,
                                       const std::function<void(const WorkflowTraceItem&)>& onProgress) {
  WorkflowResult res;
  res.success = false;
  if (agents_.empty()) {
    res.error = "Blackboard workflow has no agents";
    return res;
  }

  blackboard_->postMessage("Workflow", "Starting round-robin blackboard workflow for: " + goal, "plan");
  blackboard_->setState("goal", goal);

  int globalStep = 0;
  int idleRounds = 0;
  const int maxIdle = static_cast<int>(agents_.size()) * 2;

  while (globalStep < config_.maxTotalSteps && !blackboard_->allTasksCompleted() && idleRounds < maxIdle) {
    auto ready = blackboard_->readyTasks();
    if (ready.empty()) {
      ++idleRounds;
      continue;
    }
    idleRounds = 0;

    for (const auto& task : ready) {
      if (globalStep >= config_.maxTotalSteps) break;

      // Select next agent round-robin
      auto& agent = agents_[globalStep % agents_.size()];
      blackboard_->assignTask(task.id, agent->name());
      blackboard_->updateTaskStatus(task.id, TaskStatus::InProgress);

      std::string taskPrompt = "Execute Assigned Task: [" + task.title + "]\n" +
                               "Description: " + task.description + "\n" +
                               "Produce the final result of this task.";

      auto agentRes = agent->run(taskPrompt, {}, [&](const AgentStepTrace& trace) {
        ++globalStep;
        WorkflowTraceItem item;
        item.stepNumber = globalStep;
        item.agentId = agent->id();
        item.agentName = agent->name();
        item.role = std::string(roleTypeName(agent->role().type));
        item.action = "Task Execution: " + task.title;
        item.output = trace.response;
        item.timestamp = currentEpochMs();
        res.trace.push_back(item);
        if (onProgress) onProgress(item);
      });

      blackboard_->updateTaskStatus(task.id, TaskStatus::Done, agentRes.finalAnswer);
      blackboard_->postMessage(agent->name(), "Completed task [" + task.title + "]: " + agentRes.finalAnswer, "action");
    }
  }

  // Compile final answer from all completed tasks
  std::ostringstream ss;
  for (const auto& task : blackboard_->listTasks()) {
    if (task.status == TaskStatus::Done) {
      ss << "### " << task.title << "\n" << task.result << "\n\n";
    }
  }

  res.success = true;
  res.finalAnswer = ss.str().empty() ? "Workflow completed with no output." : ss.str();
  res.totalSteps = globalStep;
  res.artifacts = blackboard_->listArtifacts();
  blackboard_->setState("final_answer", res.finalAnswer);
  return res;
}

// ---------------------------------------------------------------------------
// ConsensusWorkflow
// ---------------------------------------------------------------------------
ConsensusWorkflow::ConsensusWorkflow(WorkflowConfig config,
                                     std::vector<std::shared_ptr<Agent>> proposers,
                                     std::shared_ptr<Agent> judge,
                                     std::shared_ptr<Blackboard> blackboard)
    : config_(std::move(config)),
      proposers_(std::move(proposers)),
      judge_(std::move(judge)),
      blackboard_(std::move(blackboard)) {
  if (!blackboard_) blackboard_ = std::make_shared<Blackboard>();
  for (auto& p : proposers_) {
    if (p) p->setBlackboard(blackboard_);
  }
  if (judge_) judge_->setBlackboard(blackboard_);
}

WorkflowResult ConsensusWorkflow::run(const std::string& goal,
                                      const std::function<void(const WorkflowTraceItem&)>& onProgress) {
  WorkflowResult res;
  res.success = false;
  if (proposers_.empty() || !judge_) {
    res.error = "Consensus workflow requires proposers and a judge agent";
    return res;
  }

  blackboard_->postMessage("Workflow", "Starting consensus workflow for: " + goal, "plan");
  blackboard_->setState("goal", goal);

  int globalStep = 0;
  std::ostringstream proposals;

  for (std::size_t i = 0; i < proposers_.size(); ++i) {
    auto& proposer = proposers_[i];
    std::string prompt = "Propose an independent, high-quality solution for goal: " + goal;

    auto propRes = proposer->run(prompt, {}, [&](const AgentStepTrace& trace) {
      ++globalStep;
      WorkflowTraceItem item;
      item.stepNumber = globalStep;
      item.agentId = proposer->id();
      item.agentName = proposer->name();
      item.role = "Proposer";
      item.action = "Drafting Proposal #" + std::to_string(i + 1);
      item.output = trace.response;
      item.timestamp = currentEpochMs();
      res.trace.push_back(item);
      if (onProgress) onProgress(item);
    });

    proposals << "Proposal #" << (i + 1) << " (by " << proposer->name() << "):\n"
              << propRes.finalAnswer << "\n\n";
    blackboard_->postMessage(proposer->name(), propRes.finalAnswer, "thought");
  }

  // Judge evaluates and synthesizes consensus
  std::string judgePrompt = "Evaluate the following competing proposals for goal: " + goal + "\n\n" +
                            proposals.str() +
                            "Synthesize the optimal consensus solution combining strengths and eliminating weaknesses.";

  auto judgeRes = judge_->run(judgePrompt, {}, [&](const AgentStepTrace& trace) {
    ++globalStep;
    WorkflowTraceItem item;
    item.stepNumber = globalStep;
    item.agentId = judge_->id();
    item.agentName = judge_->name();
    item.role = "Judge/Synthesizer";
    item.action = "Consensus Evaluation";
    item.output = trace.response;
    item.timestamp = currentEpochMs();
    res.trace.push_back(item);
    if (onProgress) onProgress(item);
  });

  res.success = true;
  res.finalAnswer = judgeRes.finalAnswer;
  res.totalSteps = globalStep;
  res.artifacts = blackboard_->listArtifacts();
  blackboard_->setState("final_answer", judgeRes.finalAnswer);
  return res;
}

}  // namespace qorvix::agents
