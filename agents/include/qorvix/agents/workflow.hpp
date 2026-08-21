#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "qorvix/agents/agent.hpp"
#include "qorvix/agents/blackboard.hpp"
#include "qorvix/agents/tool_registry.hpp"

namespace qorvix::agents {

enum class WorkflowPattern {
  Sequential,
  HierarchicalSupervisor,
  RoundRobinBlackboard,
  ConsensusVoting
};

std::string_view workflowPatternName(WorkflowPattern pattern);
WorkflowPattern parseWorkflowPattern(std::string_view name);

enum class WorkflowStatus {
  Ready,
  Running,
  Completed,
  Failed,
  StepLimitExceeded
};

struct WorkflowTraceItem {
  int stepNumber = 0;
  std::string agentId;
  std::string agentName;
  std::string role;
  std::string action;
  std::string output;
  std::int64_t timestamp = 0;

  api::json::Value toJson() const;
};

struct WorkflowResult {
  bool success = false;
  WorkflowStatus status = WorkflowStatus::Completed;
  std::string finalAnswer;
  int totalSteps = 0;
  std::vector<WorkflowTraceItem> trace;
  std::vector<Artifact> artifacts;
  std::string error;

  api::json::Value toJson() const;
};

struct WorkflowConfig {
  std::string name = "Multi-Agent Workflow";
  WorkflowPattern pattern = WorkflowPattern::Sequential;
  int maxTotalSteps = 40;
};

class IWorkflow {
 public:
  virtual ~IWorkflow() = default;

  virtual const WorkflowConfig& config() const = 0;
  virtual std::shared_ptr<Blackboard> blackboard() const = 0;
  virtual WorkflowResult run(const std::string& goal,
                             const std::function<void(const WorkflowTraceItem&)>& onProgress = {}) = 0;
};

// Sequential pipeline workflow: chains output of each agent into the next agent
class SequentialWorkflow : public IWorkflow {
 public:
  SequentialWorkflow(WorkflowConfig config,
                     std::vector<std::shared_ptr<Agent>> agents,
                     std::shared_ptr<Blackboard> blackboard);

  const WorkflowConfig& config() const override { return config_; }
  std::shared_ptr<Blackboard> blackboard() const override { return blackboard_; }

  WorkflowResult run(const std::string& goal,
                     const std::function<void(const WorkflowTraceItem&)>& onProgress = {}) override;

 private:
  WorkflowConfig config_;
  std::vector<std::shared_ptr<Agent>> agents_;
  std::shared_ptr<Blackboard> blackboard_;
};

// Hierarchical supervisor workflow: Supervisor delegates tasks to specialists and synthesizes output
class SupervisorWorkflow : public IWorkflow {
 public:
  SupervisorWorkflow(WorkflowConfig config,
                     std::shared_ptr<Agent> supervisor,
                     std::vector<std::shared_ptr<Agent>> workers,
                     std::shared_ptr<Blackboard> blackboard);

  const WorkflowConfig& config() const override { return config_; }
  std::shared_ptr<Blackboard> blackboard() const override { return blackboard_; }

  WorkflowResult run(const std::string& goal,
                     const std::function<void(const WorkflowTraceItem&)>& onProgress = {}) override;

 private:
  WorkflowConfig config_;
  std::shared_ptr<Agent> supervisor_;
  std::vector<std::shared_ptr<Agent>> workers_;
  std::shared_ptr<Blackboard> blackboard_;
};

// Round-robin blackboard workflow: agents iterate and claim ready tasks until blackboard is cleared
class BlackboardWorkflow : public IWorkflow {
 public:
  BlackboardWorkflow(WorkflowConfig config,
                     std::vector<std::shared_ptr<Agent>> agents,
                     std::shared_ptr<Blackboard> blackboard);

  const WorkflowConfig& config() const override { return config_; }
  std::shared_ptr<Blackboard> blackboard() const override { return blackboard_; }

  WorkflowResult run(const std::string& goal,
                     const std::function<void(const WorkflowTraceItem&)>& onProgress = {}) override;

 private:
  WorkflowConfig config_;
  std::vector<std::shared_ptr<Agent>> agents_;
  std::shared_ptr<Blackboard> blackboard_;
};

// Consensus voting workflow: multiple agents propose candidates, critic/synthesizer picks/merges best
class ConsensusWorkflow : public IWorkflow {
 public:
  ConsensusWorkflow(WorkflowConfig config,
                    std::vector<std::shared_ptr<Agent>> proposers,
                    std::shared_ptr<Agent> judge,
                    std::shared_ptr<Blackboard> blackboard);

  const WorkflowConfig& config() const override { return config_; }
  std::shared_ptr<Blackboard> blackboard() const override { return blackboard_; }

  WorkflowResult run(const std::string& goal,
                     const std::function<void(const WorkflowTraceItem&)>& onProgress = {}) override;

 private:
  WorkflowConfig config_;
  std::vector<std::shared_ptr<Agent>> proposers_;
  std::shared_ptr<Agent> judge_;
  std::shared_ptr<Blackboard> blackboard_;
};

}  // namespace qorvix::agents
