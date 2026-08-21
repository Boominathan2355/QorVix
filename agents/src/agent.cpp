#include "qorvix/agents/agent.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace qorvix::agents {

Agent::Agent(std::string id, RoleDefinition role,
             std::shared_ptr<ToolRegistry> tools,
             std::shared_ptr<Blackboard> blackboard,
             std::shared_ptr<SkillRegistry> skills)
    : id_(std::move(id)),
      role_(std::move(role)),
      tools_(std::move(tools)),
      blackboard_(std::move(blackboard)),
      skills_(std::move(skills)) {
  ensureSystemPrompt();
}

void Agent::ensureSystemPrompt() {
  if (history_.empty() || history_[0].role != "system") {
    std::ostringstream ss;
    ss << role_.systemPrompt << "\n";
    if (tools_) {
      ss << "\nYou have access to the following tools:\n";
      auto allowed = role_.allowedTools;
      for (const auto& def : tools_->definitions()) {
        if (!allowed.empty() && std::find(allowed.begin(), allowed.end(), def.name) == allowed.end()) {
          continue;
        }
        ss << "- " << def.name << ": " << def.description << "\n";
      }
      ss << "\nTo use a tool, format your action as:\n"
         << "Action: <tool_name>\n"
         << "Action Input: <json_arguments>\n"
         << "When you have the final answer, output:\n"
         << "Final Answer: <your response>\n";
    }
    if (skills_) {
      std::string skillsPrompt = skills_->composeSkillsPrompt(role_.allowedSkills);
      if (!skillsPrompt.empty()) {
        ss << "\n" << skillsPrompt << "\n";
      }
    }
    if (history_.empty()) {
      history_.push_back({"system", ss.str(), "", {}, {}, ""});
    } else {
      history_[0] = {"system", ss.str(), "", {}, {}, ""};
    }
  }
}

void Agent::reset() {
  history_.clear();
  currentStep_ = 0;
  ensureSystemPrompt();
}

void Agent::addMessage(api::ChatMessage msg) {
  history_.push_back(std::move(msg));
}

bool Agent::parseToolCallsFromText(const std::string& text,
                                  std::vector<api::ToolCall>& toolCalls,
                                  std::string& thought) {
  // Check for <tool_call> ... </tool_call>
  auto startTag = text.find("<tool_call>");
  if (startTag != std::string::npos) {
    auto endTag = text.find("</tool_call>", startTag);
    if (endTag != std::string::npos) {
      thought = text.substr(0, startTag);
      std::string jsonStr = text.substr(startTag + 11, endTag - (startTag + 11));
      auto parsed = api::json::parse(jsonStr);
      if (parsed.has_value() && parsed->isObject()) {
        api::ToolCall call;
        call.id = "call_" + std::to_string(currentStep_) + "_0";
        if (const auto* n = parsed->get("name"); n && n->isString()) {
          call.name = n->asString();
        }
        if (const auto* a = parsed->get("arguments"); a) {
          call.arguments = a->isString() ? a->asString() : a->dump();
        }
        if (!call.name.empty()) {
          toolCalls.push_back(std::move(call));
          return true;
        }
      }
    }
  }

  // Check for Action: ... \n Action Input: ...
  auto actionPos = text.find("Action:");
  if (actionPos != std::string::npos) {
    thought = text.substr(0, actionPos);
    std::string rest = text.substr(actionPos + 7);
    while (!rest.empty() && (rest[0] == ' ' || rest[0] == '\t')) rest.erase(0, 1);

    auto newlinePos = rest.find('\n');
    std::string toolName = rest.substr(0, newlinePos);
    // Trim trailing whitespace from toolName
    while (!toolName.empty() && (toolName.back() == ' ' || toolName.back() == '\r' || toolName.back() == '\t')) {
      toolName.pop_back();
    }

    std::string toolArgs = "{}";
    if (newlinePos != std::string::npos) {
      std::string afterName = rest.substr(newlinePos + 1);
      auto inputPos = afterName.find("Action Input:");
      if (inputPos != std::string::npos) {
        std::string rawArgs = afterName.substr(inputPos + 13);
        while (!rawArgs.empty() && (rawArgs[0] == ' ' || rawArgs[0] == '\t' || rawArgs[0] == '\n' || rawArgs[0] == '\r')) {
          rawArgs.erase(0, 1);
        }
        // If argument is in markdown codeblock ```json ... ``` strip it
        if (rawArgs.rfind("```json", 0) == 0) {
          rawArgs = rawArgs.substr(7);
        } else if (rawArgs.rfind("```", 0) == 0) {
          rawArgs = rawArgs.substr(3);
        }
        auto blockEnd = rawArgs.find("```");
        if (blockEnd != std::string::npos) {
          rawArgs = rawArgs.substr(0, blockEnd);
        }
        while (!rawArgs.empty() && (rawArgs.back() == ' ' || rawArgs.back() == '\r' || rawArgs.back() == '\n' || rawArgs.back() == '\t')) {
          rawArgs.pop_back();
        }
        if (!rawArgs.empty()) {
          toolArgs = rawArgs;
        }
      }
    }

    if (!toolName.empty()) {
      api::ToolCall call;
      call.id = "call_" + std::to_string(currentStep_) + "_0";
      call.name = toolName;
      call.arguments = toolArgs;
      toolCalls.push_back(std::move(call));
      return true;
    }
  }

  // Check for ```json\n{"action": "...", "action_input": ...}\n```
  auto codePos = text.find("```json");
  if (codePos != std::string::npos) {
    auto codeEnd = text.find("```", codePos + 7);
    if (codeEnd != std::string::npos) {
      std::string jsonStr = text.substr(codePos + 7, codeEnd - (codePos + 7));
      auto parsed = api::json::parse(jsonStr);
      if (parsed.has_value() && parsed->isObject()) {
        const auto* act = parsed->get("action");
        if (act && act->isString()) {
          thought = text.substr(0, codePos);
          api::ToolCall call;
          call.id = "call_" + std::to_string(currentStep_) + "_0";
          call.name = act->asString();
          const auto* inp = parsed->get("action_input");
          call.arguments = inp ? (inp->isString() ? inp->asString() : inp->dump()) : "{}";
          toolCalls.push_back(std::move(call));
          return true;
        }
      }
    }
  }

  return false;
}

AgentStepTrace Agent::step(const std::function<void(const std::string&)>& onToken) {
  ensureSystemPrompt();
  ++currentStep_;

  AgentStepTrace trace;
  trace.stepNumber = currentStep_;

  std::vector<api::ToolDefinition> openAiTools;
  if (tools_) {
    openAiTools = tools_->toOpenAiTools(role_.allowedTools);
  }

  std::string responseText;
  if (inferenceFn_) {
    responseText = inferenceFn_(history_, openAiTools, role_.sampling, onToken);
  } else {
    responseText = "Final Answer: (Inference engine not attached)";
  }

  trace.response = responseText;

  std::vector<api::ToolCall> calls;
  std::string thought;
  bool hasCalls = parseToolCallsFromText(responseText, calls, thought);

  if (hasCalls && tools_) {
    trace.thought = thought;
    trace.toolCalls = calls;
    trace.isFinal = false;

    // Record assistant message with tool calls
    api::ChatMessage asstMsg;
    asstMsg.role = "assistant";
    asstMsg.content = responseText;
    asstMsg.toolCalls = calls;
    history_.push_back(asstMsg);

    for (const auto& call : calls) {
      api::json::Value argsVal = api::json::Value::object();
      auto parsed = api::json::parse(call.arguments);
      if (parsed.has_value()) {
        argsVal = *parsed;
      } else if (!call.arguments.empty() && call.arguments != "{}") {
        // Fallback: wrap raw string into { "input": ... } or expression
        if (call.name == "calculator") {
          argsVal["expression"] = call.arguments;
        } else if (call.name == "rag_search") {
          argsVal["query"] = call.arguments;
        } else {
          argsVal["input"] = call.arguments;
        }
      }

      ToolResult res = tools_->execute(call.name, argsVal);
      trace.observations.push_back(res);

      // Record observation message
      api::ChatMessage toolMsg;
      toolMsg.role = "tool";
      toolMsg.toolCallId = call.id;
      toolMsg.content = "Observation: " + res.output;
      history_.push_back(toolMsg);

      if (blackboard_) {
        blackboard_->postMessage(name(), "Executed tool [" + call.name + "] -> " + res.output, "action");
      }
    }
  } else {
    // No tool calls — final answer or monologue
    trace.isFinal = true;
    std::string finalAnswer = responseText;
    auto faPos = responseText.find("Final Answer:");
    if (faPos != std::string::npos) {
      finalAnswer = responseText.substr(faPos + 13);
      while (!finalAnswer.empty() && (finalAnswer[0] == ' ' || finalAnswer[0] == '\n' || finalAnswer[0] == '\r')) {
        finalAnswer.erase(0, 1);
      }
    }
    trace.thought = responseText;

    api::ChatMessage asstMsg;
    asstMsg.role = "assistant";
    asstMsg.content = responseText;
    history_.push_back(asstMsg);

    if (blackboard_) {
      blackboard_->postMessage(name(), finalAnswer, "thought");
    }
  }

  return trace;
}

AgentExecutionResult Agent::run(const std::string& prompt,
                               const std::function<void(const std::string&)>& onToken,
                               const std::function<void(const AgentStepTrace&)>& onStep) {
  reset();
  if (!prompt.empty()) {
    history_.push_back({"user", prompt, "", {}, {}, ""});
  }

  AgentExecutionResult result;
  result.totalSteps = 0;

  for (int stepIdx = 0; stepIdx < role_.maxReasoningSteps; ++stepIdx) {
    AgentStepTrace trace = step(onToken);
    result.trace.push_back(trace);
    result.totalSteps = currentStep_;

    if (onStep) {
      onStep(trace);
    }

    if (trace.isFinal) {
      result.success = true;
      std::string answer = trace.response;
      auto faPos = answer.find("Final Answer:");
      if (faPos != std::string::npos) {
        answer = answer.substr(faPos + 13);
        while (!answer.empty() && (answer[0] == ' ' || answer[0] == '\n' || answer[0] == '\r')) {
          answer.erase(0, 1);
        }
      }
      result.finalAnswer = answer;
      return result;
    }
  }

  result.success = false;
  result.error = "Agent reached max reasoning steps (" + std::to_string(role_.maxReasoningSteps) + ") without final answer.";
  if (!result.trace.empty()) {
    result.finalAnswer = result.trace.back().response;
  }
  return result;
}

}  // namespace qorvix::agents
