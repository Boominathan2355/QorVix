#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "qorvix/agents/blackboard.hpp"
#include "qorvix/agents/role.hpp"
#include "qorvix/agents/skill_registry.hpp"
#include "qorvix/agents/tool.hpp"
#include "qorvix/agents/tool_registry.hpp"
#include "qorvix/api/openai.hpp"

namespace qorvix::agents {

struct AgentStepTrace {
  int stepNumber = 0;
  std::string thought;
  std::vector<api::ToolCall> toolCalls;
  std::vector<ToolResult> observations;
  std::string response;
  bool isFinal = false;
};

struct AgentExecutionResult {
  bool success = false;
  std::string finalAnswer;
  std::vector<AgentStepTrace> trace;
  int totalSteps = 0;
  std::string error;
};

// Generic LLM completion callback for the agent: given messages & tools, produces response text
using InferenceCallback = std::function<std::string(
    const std::vector<api::ChatMessage>& messages,
    const std::vector<api::ToolDefinition>& tools,
    const SamplingProfile& sampling,
    const std::function<void(const std::string&)>& onToken)>;

class Agent {
 public:
  Agent(std::string id, RoleDefinition role,
        std::shared_ptr<ToolRegistry> tools = nullptr,
        std::shared_ptr<Blackboard> blackboard = nullptr,
        std::shared_ptr<SkillRegistry> skills = nullptr);

  const std::string& id() const { return id_; }
  const std::string& name() const { return role_.name; }
  const RoleDefinition& role() const { return role_; }
  void setRole(RoleDefinition role) { role_ = std::move(role); }

  void setToolRegistry(std::shared_ptr<ToolRegistry> tools) { tools_ = std::move(tools); }
  void setBlackboard(std::shared_ptr<Blackboard> blackboard) { blackboard_ = std::move(blackboard); }
  void setSkillRegistry(std::shared_ptr<SkillRegistry> skills) { skills_ = std::move(skills); }
  std::shared_ptr<SkillRegistry> skillRegistry() const { return skills_; }
  void setInferenceCallback(InferenceCallback fn) { inferenceFn_ = std::move(fn); }

  // Runs the complete ReAct reasoning loop until a final answer is produced or max steps reached
  AgentExecutionResult run(const std::string& prompt,
                           const std::function<void(const std::string&)>& onToken = {},
                           const std::function<void(const AgentStepTrace&)>& onStep = {});

  // Executes a single reasoning/action step
  AgentStepTrace step(const std::function<void(const std::string&)>& onToken = {});

  void reset();
  void addMessage(api::ChatMessage msg);
  const std::vector<api::ChatMessage>& history() const { return history_; }

 private:
  void ensureSystemPrompt();
  bool parseToolCallsFromText(const std::string& text, std::vector<api::ToolCall>& toolCalls, std::string& thought);

  std::string id_;
  RoleDefinition role_;
  std::shared_ptr<ToolRegistry> tools_;
  std::shared_ptr<Blackboard> blackboard_;
  std::shared_ptr<SkillRegistry> skills_;
  InferenceCallback inferenceFn_;
  std::vector<api::ChatMessage> history_;
  int currentStep_ = 0;
};

}  // namespace qorvix::agents
