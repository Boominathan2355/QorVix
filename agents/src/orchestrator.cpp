#include "qorvix/agents/orchestrator.hpp"
#include "qorvix/agents/builtin_skills.hpp"

namespace qorvix::agents {

WorkflowOrchestrator::WorkflowOrchestrator(std::shared_ptr<ToolRegistry> tools,
                                           std::shared_ptr<Blackboard> blackboard,
                                           std::shared_ptr<SkillRegistry> skills)
    : tools_(std::move(tools)),
      blackboard_(std::move(blackboard)),
      skills_(std::move(skills)) {
  if (!blackboard_) {
    blackboard_ = std::make_shared<Blackboard>();
  }
  if (!skills_) {
    skills_ = std::make_shared<SkillRegistry>();
    registerDefaultSkills(*skills_);
  }
  if (!tools_) {
    tools_ = std::make_shared<ToolRegistry>();
    registerDefaultTools(*tools_, "", blackboard_);
    skills_->registerAllAsTools(*tools_);
  }
}

std::shared_ptr<Agent> WorkflowOrchestrator::makeAgent(const RoleDefinition& role, const std::string& idSuffix) {
  ++agentCounter_;
  std::string id = "agent_" + std::to_string(agentCounter_) + (idSuffix.empty() ? "" : "_" + idSuffix);
  auto agent = std::make_shared<Agent>(id, role, tools_, blackboard_, skills_);
  if (inferenceFn_) {
    agent->setInferenceCallback(inferenceFn_);
  }
  return agent;
}

std::unique_ptr<IWorkflow> WorkflowOrchestrator::createSoftwareDevTeam(WorkflowPattern pattern) {
  WorkflowConfig cfg;
  cfg.name = "Software Engineering Team";
  cfg.pattern = pattern;

  auto planner = makeAgent(createPlannerRole(), "planner");
  auto coder = makeAgent(createCoderRole(), "coder");
  auto critic = makeAgent(createCriticRole(), "critic");
  auto synth = makeAgent(createSynthesizerRole(), "synthesizer");

  if (pattern == WorkflowPattern::HierarchicalSupervisor) {
    auto supervisor = makeAgent(createCoordinatorRole(), "lead");
    return std::make_unique<SupervisorWorkflow>(cfg, supervisor, std::vector<std::shared_ptr<Agent>>{planner, coder, critic, synth}, blackboard_);
  }

  if (pattern == WorkflowPattern::RoundRobinBlackboard) {
    return std::make_unique<BlackboardWorkflow>(cfg, std::vector<std::shared_ptr<Agent>>{planner, coder, critic, synth}, blackboard_);
  }

  return std::make_unique<SequentialWorkflow>(cfg, std::vector<std::shared_ptr<Agent>>{planner, coder, critic, synth}, blackboard_);
}

std::unique_ptr<IWorkflow> WorkflowOrchestrator::createResearchTeam(WorkflowPattern pattern) {
  WorkflowConfig cfg;
  cfg.name = "Research & Analysis Team";
  cfg.pattern = pattern;

  auto researcher1 = makeAgent(createResearcherRole(), "primary_researcher");
  auto researcher2 = makeAgent(createResearcherRole(), "cross_verifier");
  auto synth = makeAgent(createSynthesizerRole(), "synthesizer");

  if (pattern == WorkflowPattern::HierarchicalSupervisor) {
    auto lead = makeAgent(createCoordinatorRole(), "research_lead");
    return std::make_unique<SupervisorWorkflow>(cfg, lead, std::vector<std::shared_ptr<Agent>>{researcher1, researcher2, synth}, blackboard_);
  }

  return std::make_unique<SequentialWorkflow>(cfg, std::vector<std::shared_ptr<Agent>>{researcher1, researcher2, synth}, blackboard_);
}

std::unique_ptr<IWorkflow> WorkflowOrchestrator::createConsensusTeam(int numProposers) {
  WorkflowConfig cfg;
  cfg.name = "Consensus & Deliberation Team";
  cfg.pattern = WorkflowPattern::ConsensusVoting;

  std::vector<std::shared_ptr<Agent>> proposers;
  for (int i = 0; i < numProposers; ++i) {
    RoleDefinition role = createResearcherRole();
    role.name = "Proposer_" + std::to_string(i + 1);
    role.sampling.temperature = 0.6f + static_cast<float>(i) * 0.15f;
    proposers.push_back(makeAgent(role, "prop_" + std::to_string(i + 1)));
  }

  auto judge = makeAgent(createCriticRole(), "judge");
  return std::make_unique<ConsensusWorkflow>(cfg, std::move(proposers), judge, blackboard_);
}

std::unique_ptr<IWorkflow> WorkflowOrchestrator::buildWorkflow(WorkflowPattern pattern,
                                                              const std::vector<RoleDefinition>& roles,
                                                              const std::string& workflowName) {
  WorkflowConfig cfg;
  cfg.name = workflowName.empty() ? "Custom Workflow" : workflowName;
  cfg.pattern = pattern;

  std::vector<std::shared_ptr<Agent>> agents;
  for (const auto& r : roles) {
    agents.push_back(makeAgent(r));
  }

  if (pattern == WorkflowPattern::HierarchicalSupervisor && !agents.empty()) {
    auto supervisor = agents.front();
    std::vector<std::shared_ptr<Agent>> workers(agents.begin() + 1, agents.end());
    return std::make_unique<SupervisorWorkflow>(cfg, supervisor, std::move(workers), blackboard_);
  }

  if (pattern == WorkflowPattern::RoundRobinBlackboard) {
    return std::make_unique<BlackboardWorkflow>(cfg, std::move(agents), blackboard_);
  }

  if (pattern == WorkflowPattern::ConsensusVoting && agents.size() >= 2) {
    auto judge = agents.back();
    std::vector<std::shared_ptr<Agent>> proposers(agents.begin(), agents.end() - 1);
    return std::make_unique<ConsensusWorkflow>(cfg, std::move(proposers), judge, blackboard_);
  }

  return std::make_unique<SequentialWorkflow>(cfg, std::move(agents), blackboard_);
}

WorkflowResult WorkflowOrchestrator::execute(IWorkflow& workflow,
                                            const std::string& goal,
                                            const std::function<void(const WorkflowTraceItem&)>& onProgress) {
  return workflow.run(goal, onProgress);
}

}  // namespace qorvix::agents
