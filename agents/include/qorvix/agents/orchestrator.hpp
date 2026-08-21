#pragma once

#include <memory>
#include <string>
#include <vector>

#include "qorvix/agents/agent.hpp"
#include "qorvix/agents/blackboard.hpp"
#include "qorvix/agents/builtin_tools.hpp"
#include "qorvix/agents/role.hpp"
#include "qorvix/agents/skill_registry.hpp"
#include "qorvix/agents/tool_registry.hpp"
#include "qorvix/agents/workflow.hpp"

namespace qorvix::agents {

class WorkflowOrchestrator {
 public:
  WorkflowOrchestrator(std::shared_ptr<ToolRegistry> tools = nullptr,
                       std::shared_ptr<Blackboard> blackboard = nullptr,
                       std::shared_ptr<SkillRegistry> skills = nullptr);

  std::shared_ptr<ToolRegistry> toolRegistry() const { return tools_; }
  std::shared_ptr<Blackboard> blackboard() const { return blackboard_; }
  std::shared_ptr<SkillRegistry> skillRegistry() const { return skills_; }
  void setSkillRegistry(std::shared_ptr<SkillRegistry> skills) { skills_ = std::move(skills); }

  void setInferenceCallback(InferenceCallback fn) { inferenceFn_ = std::move(fn); }

  // Factory methods for pre-configured agent teams
  std::unique_ptr<IWorkflow> createSoftwareDevTeam(WorkflowPattern pattern = WorkflowPattern::Sequential);
  std::unique_ptr<IWorkflow> createResearchTeam(WorkflowPattern pattern = WorkflowPattern::HierarchicalSupervisor);
  std::unique_ptr<IWorkflow> createConsensusTeam(int numProposers = 2);

  // Custom workflow builder
  std::unique_ptr<IWorkflow> buildWorkflow(WorkflowPattern pattern,
                                          const std::vector<RoleDefinition>& roles,
                                          const std::string& workflowName = "");

  // Executes a workflow given a user goal
  WorkflowResult execute(IWorkflow& workflow,
                         const std::string& goal,
                         const std::function<void(const WorkflowTraceItem&)>& onProgress = {});

 private:
  std::shared_ptr<Agent> makeAgent(const RoleDefinition& role, const std::string& idSuffix = "");

  std::shared_ptr<ToolRegistry> tools_;
  std::shared_ptr<Blackboard> blackboard_;
  std::shared_ptr<SkillRegistry> skills_;
  InferenceCallback inferenceFn_;
  int agentCounter_ = 0;
};

}  // namespace qorvix::agents
