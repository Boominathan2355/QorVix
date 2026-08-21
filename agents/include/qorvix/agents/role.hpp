#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "qorvix/api/json.hpp"

namespace qorvix::agents {

// Roles and personas for multi-agent workflows (SPEC "Multi-Agent Workflows", Phase 11b).
enum class RoleType {
  Custom,
  Coordinator,
  Planner,
  Researcher,
  Coder,
  Critic,
  Executor,
  Synthesizer
};

std::string_view roleTypeName(RoleType type);
RoleType parseRoleType(std::string_view name);

struct SamplingProfile {
  float temperature = 0.7f;
  float topP = 0.95f;
  int topK = 40;
  float repetitionPenalty = 1.05f;
  int maxTokens = 2048;
};

struct RoleDefinition {
  std::string name;
  RoleType type = RoleType::Custom;
  std::string description;
  std::string systemPrompt;
  std::vector<std::string> allowedTools;   // tool names this role has access to; empty = all
  std::vector<std::string> allowedSkills;  // skill names active for this role; empty = all
  SamplingProfile sampling;
  int maxReasoningSteps = 10;

  // JSON serialization
  api::json::Value toJson() const;
  static bool fromJson(const api::json::Value& v, RoleDefinition& out, std::string& error);
};

// Built-in persona factories
RoleDefinition createCoordinatorRole();
RoleDefinition createPlannerRole();
RoleDefinition createResearcherRole();
RoleDefinition createCoderRole();
RoleDefinition createCriticRole();
RoleDefinition createExecutorRole();
RoleDefinition createSynthesizerRole();

}  // namespace qorvix::agents
