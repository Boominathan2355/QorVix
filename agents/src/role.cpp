#include "qorvix/agents/role.hpp"

#include <algorithm>

namespace qorvix::agents {

std::string_view roleTypeName(RoleType type) {
  switch (type) {
    case RoleType::Custom: return "custom";
    case RoleType::Coordinator: return "coordinator";
    case RoleType::Planner: return "planner";
    case RoleType::Researcher: return "researcher";
    case RoleType::Coder: return "coder";
    case RoleType::Critic: return "critic";
    case RoleType::Executor: return "executor";
    case RoleType::Synthesizer: return "synthesizer";
  }
  return "custom";
}

RoleType parseRoleType(std::string_view name) {
  if (name == "coordinator") return RoleType::Coordinator;
  if (name == "planner") return RoleType::Planner;
  if (name == "researcher") return RoleType::Researcher;
  if (name == "coder") return RoleType::Coder;
  if (name == "critic") return RoleType::Critic;
  if (name == "executor") return RoleType::Executor;
  if (name == "synthesizer") return RoleType::Synthesizer;
  return RoleType::Custom;
}

api::json::Value RoleDefinition::toJson() const {
  auto v = api::json::Value::object();
  v["name"] = name;
  v["type"] = std::string(roleTypeName(type));
  v["description"] = description;
  v["system_prompt"] = systemPrompt;
  v["max_reasoning_steps"] = maxReasoningSteps;

  auto tools = api::json::Value::array();
  for (const auto& t : allowedTools) {
    tools.push(t);
  }
  v["allowed_tools"] = std::move(tools);

  auto skills = api::json::Value::array();
  for (const auto& s : allowedSkills) {
    skills.push(s);
  }
  v["allowed_skills"] = std::move(skills);

  auto smp = api::json::Value::object();
  smp["temperature"] = sampling.temperature;
  smp["top_p"] = sampling.topP;
  smp["top_k"] = sampling.topK;
  smp["repetition_penalty"] = sampling.repetitionPenalty;
  smp["max_tokens"] = sampling.maxTokens;
  v["sampling"] = std::move(smp);

  return v;
}

bool RoleDefinition::fromJson(const api::json::Value& v, RoleDefinition& out, std::string& error) {
  if (!v.isObject()) {
    error = "RoleDefinition expects a JSON object";
    return false;
  }

  if (const auto* n = v.get("name"); n && n->isString()) {
    out.name = n->asString();
  } else {
    error = "Missing or invalid 'name' field in role";
    return false;
  }

  if (const auto* t = v.get("type"); t && t->isString()) {
    out.type = parseRoleType(t->asString());
  } else {
    out.type = RoleType::Custom;
  }

  if (const auto* d = v.get("description"); d && d->isString()) {
    out.description = d->asString();
  }

  if (const auto* sp = v.get("system_prompt"); sp && sp->isString()) {
    out.systemPrompt = sp->asString();
  }

  if (const auto* rs = v.get("max_reasoning_steps"); rs && rs->isNumber()) {
    out.maxReasoningSteps = rs->asInt();
  }

  out.allowedTools.clear();
  if (const auto* at = v.get("allowed_tools"); at && at->isArray()) {
    for (const auto& item : at->items()) {
      if (item.isString()) {
        out.allowedTools.push_back(item.asString());
      }
    }
  }

  out.allowedSkills.clear();
  if (const auto* as = v.get("allowed_skills"); as && as->isArray()) {
    for (const auto& item : as->items()) {
      if (item.isString()) {
        out.allowedSkills.push_back(item.asString());
      }
    }
  }

  if (const auto* smp = v.get("sampling"); smp && smp->isObject()) {
    if (const auto* val = smp->get("temperature"); val && val->isNumber()) out.sampling.temperature = static_cast<float>(val->asNumber());
    if (const auto* val = smp->get("top_p"); val && val->isNumber()) out.sampling.topP = static_cast<float>(val->asNumber());
    if (const auto* val = smp->get("top_k"); val && val->isNumber()) out.sampling.topK = val->asInt();
    if (const auto* val = smp->get("repetition_penalty"); val && val->isNumber()) out.sampling.repetitionPenalty = static_cast<float>(val->asNumber());
    if (const auto* val = smp->get("max_tokens"); val && val->isNumber()) out.sampling.maxTokens = val->asInt();
  }

  return true;
}

RoleDefinition createCoordinatorRole() {
  RoleDefinition r;
  r.name = "Coordinator";
  r.type = RoleType::Coordinator;
  r.description = "Oversees multi-agent execution, coordinates tasks, and ensures goal convergence.";
  r.systemPrompt =
      "You are the Workflow Coordinator. Your duty is to orchestrate specialized agents, "
      "monitor shared blackboard state, resolve deadlocks, and ensure goals are completed rigorously.";
  r.sampling.temperature = 0.3f;
  r.maxReasoningSteps = 12;
  return r;
}

RoleDefinition createPlannerRole() {
  RoleDefinition r;
  r.name = "Planner";
  r.type = RoleType::Planner;
  r.description = "Decomposes complex goals into clear, actionable, dependency-aware subtasks.";
  r.systemPrompt =
      "You are the Lead Strategic Planner. Given a goal, analyze requirements, formulate a structured "
      "step-by-step execution plan, identify necessary tools, and specify validation criteria for each subtask.";
  r.sampling.temperature = 0.2f;
  r.maxReasoningSteps = 8;
  return r;
}

RoleDefinition createResearcherRole() {
  RoleDefinition r;
  r.name = "Researcher";
  r.type = RoleType::Researcher;
  r.description = "Gathers facts, searches documentation, retrieves knowledge, and synthesizes findings.";
  r.systemPrompt =
      "You are the Senior Research Agent. Use search tools, knowledge retrieval, and document analysis "
      "to uncover factual data, cite reliable sources, and produce concise, structured findings.";
  r.sampling.temperature = 0.3f;
  r.maxReasoningSteps = 15;
  return r;
}

RoleDefinition createCoderRole() {
  RoleDefinition r;
  r.name = "Coder";
  r.type = RoleType::Coder;
  r.description = "Writes, inspects, debugs, and refactors clean, high-performance code.";
  r.systemPrompt =
      "You are an expert Software Engineer and Systems Architect. Write robust, clean, and tested code. "
      "Follow modern best practices, handle error edge-cases, and write modular components.";
  r.sampling.temperature = 0.1f;
  r.maxReasoningSteps = 15;
  return r;
}

RoleDefinition createCriticRole() {
  RoleDefinition r;
  r.name = "Critic";
  r.type = RoleType::Critic;
  r.description = "Reviews, validates, verifies logic, finds flaws, and scores agent outputs.";
  r.systemPrompt =
      "You are the Rigorous Reviewer and Quality Critic. Scrutinize proposals, code, and answers "
      "for logical flaws, edge-case regressions, inaccuracies, and performance bottlenecks. Give constructive feedback.";
  r.sampling.temperature = 0.1f;
  r.maxReasoningSteps = 8;
  return r;
}

RoleDefinition createExecutorRole() {
  RoleDefinition r;
  r.name = "Executor";
  r.type = RoleType::Executor;
  r.description = "Executes concrete tasks, runs commands, applies transformations, and logs results.";
  r.systemPrompt =
      "You are the Operational Executor. Execute instructions with exact precision, invoke appropriate tools, "
      "and document output states onto the blackboard.";
  r.sampling.temperature = 0.2f;
  r.maxReasoningSteps = 10;
  return r;
}

RoleDefinition createSynthesizerRole() {
  RoleDefinition r;
  r.name = "Synthesizer";
  r.type = RoleType::Synthesizer;
  r.description = "Consolidates findings, code, and critiques into a unified, comprehensive final response.";
  r.systemPrompt =
      "You are the Lead Synthesizer. Merge multi-agent contributions, filter redundancy, resolve conflicting notes, "
      "and deliver a polished, complete final answer.";
  r.sampling.temperature = 0.4f;
  r.maxReasoningSteps = 6;
  return r;
}

}  // namespace qorvix::agents
