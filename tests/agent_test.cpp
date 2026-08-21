#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "qorvix/agents/agent.hpp"
#include "qorvix/agents/blackboard.hpp"
#include "qorvix/agents/builtin_skills.hpp"
#include "qorvix/agents/builtin_tools.hpp"
#include "qorvix/agents/orchestrator.hpp"
#include "qorvix/agents/role.hpp"
#include "qorvix/agents/skill.hpp"
#include "qorvix/agents/skill_loader.hpp"
#include "qorvix/agents/skill_registry.hpp"
#include "qorvix/agents/artifact.hpp"
#include "qorvix/agents/artifact_store.hpp"
#include "qorvix/agents/artifact_tools.hpp"
#include "qorvix/agents/tool.hpp"
#include "qorvix/agents/tool_registry.hpp"
#include "qorvix/agents/workflow.hpp"
#include "qorvix/api/json.hpp"

using namespace qorvix::agents;

TEST_CASE("RoleDefinition presets and JSON serialization", "[agents][role]") {
  SECTION("Role type conversions") {
    REQUIRE(roleTypeName(RoleType::Planner) == "planner");
    REQUIRE(roleTypeName(RoleType::Coder) == "coder");
    REQUIRE(roleTypeName(RoleType::Critic) == "critic");
    REQUIRE(parseRoleType("planner") == RoleType::Planner);
    REQUIRE(parseRoleType("coder") == RoleType::Coder);
    REQUIRE(parseRoleType("unknown") == RoleType::Custom);
  }

  SECTION("Preset role factories") {
    auto planner = createPlannerRole();
    REQUIRE(planner.name == "Planner");
    REQUIRE(planner.type == RoleType::Planner);
    REQUIRE_FALSE(planner.systemPrompt.empty());

    auto coder = createCoderRole();
    REQUIRE(coder.name == "Coder");
    REQUIRE(coder.type == RoleType::Coder);

    auto critic = createCriticRole();
    REQUIRE(critic.name == "Critic");
    REQUIRE(critic.type == RoleType::Critic);
  }

  SECTION("Serialization roundtrip") {
    RoleDefinition orig;
    orig.name = "Architect";
    orig.type = RoleType::Custom;
    orig.description = "System architecture designer";
    orig.systemPrompt = "Design robust distributed systems.";
    orig.allowedTools = {"file_ops", "calculator"};
    orig.sampling.temperature = 0.4f;
    orig.sampling.maxTokens = 1024;
    orig.maxReasoningSteps = 12;

    auto jsonVal = orig.toJson();
    REQUIRE(jsonVal.isObject());
    REQUIRE(jsonVal.get("name")->asString() == "Architect");
    REQUIRE(jsonVal.get("type")->asString() == "custom");

    RoleDefinition parsed;
    std::string err;
    REQUIRE(RoleDefinition::fromJson(jsonVal, parsed, err));
    REQUIRE(err.empty());
    REQUIRE(parsed.name == "Architect");
    REQUIRE(parsed.description == orig.description);
    REQUIRE(parsed.allowedTools.size() == 2);
    REQUIRE(parsed.allowedTools[0] == "file_ops");
    REQUIRE(parsed.maxReasoningSteps == 12);
  }
}

TEST_CASE("Tool and ToolRegistry management", "[agents][tool]") {
  ToolRegistry registry;

  SECTION("Register and execute custom tool") {
    ToolDefinition def;
    def.name = "string_reverser";
    def.description = "Reverses a string";
    def.parameters = {
        {"input", ToolParamType::String, "The input string", true, {}}
    };

    bool regOk = registry.registerFunction(def, [](const qorvix::api::json::Value& args) -> ToolResult {
      const auto* inp = args.get("input");
      if (!inp || !inp->isString()) {
        return ToolResult::makeError("Missing input parameter");
      }
      std::string s = inp->asString();
      std::reverse(s.begin(), s.end());
      return ToolResult::makeSuccess(s);
    });
    REQUIRE(regOk);
    REQUIRE(registry.hasTool("string_reverser"));

    qorvix::api::json::Value callArgs = qorvix::api::json::Value::object();
    callArgs["input"] = "QorVix Agents";

    ToolResult res = registry.execute("string_reverser", callArgs);
    REQUIRE(res.ok());
    REQUIRE(res.output == "stnegA xiVroQ");
  }

  SECTION("Schema generation for OpenAI and MCP") {
    ToolDefinition def;
    def.name = "test_tool";
    def.description = "Test Description";
    def.parameters = {
        {"param1", ToolParamType::String, "Param 1 description", true, {}},
        {"count", ToolParamType::Integer, "Count description", false, {}}
    };
    registry.registerFunction(def, [](const qorvix::api::json::Value&) { return ToolResult::makeSuccess("ok"); });

    auto openAiTools = registry.toOpenAiTools();
    REQUIRE(openAiTools.size() == 1);
    REQUIRE(openAiTools[0].function.name == "test_tool");

    auto mcpTools = registry.toMcpTools();
    REQUIRE(mcpTools.size() == 1);
    REQUIRE(mcpTools[0].name == "test_tool");
    REQUIRE(mcpTools[0].inputSchema.get("type")->asString() == "object");
  }
}

TEST_CASE("Builtin tools functionality", "[agents][builtin_tools]") {
  SECTION("Calculator tool evaluations") {
    auto calc = createCalculatorTool();
    REQUIRE(calc != nullptr);

    auto eval = [&](const std::string& expr) -> ToolResult {
      qorvix::api::json::Value args = qorvix::api::json::Value::object();
      args["expression"] = expr;
      return calc->execute(args);
    };

    REQUIRE(eval("2 + 2").output == "4");
    REQUIRE(eval("10 * (3 + 4)").output == "70");
    REQUIRE(eval("sqrt(144) + 8").output == "20");
    REQUIRE(eval("2^3 + 4 * 5").output == "28");
    REQUIRE_FALSE(eval("10 / 0").ok());
  }

  SECTION("File operations tool") {
    std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "qorvix_agent_test";
    std::filesystem::create_directories(tempDir);

    auto fileTool = createFileOpsTool(tempDir);

    // Write file
    qorvix::api::json::Value writeArgs = qorvix::api::json::Value::object();
    writeArgs["action"] = "write";
    writeArgs["path"] = "sub/hello.txt";
    writeArgs["content"] = "Hello from QorVix Multi-Agent!";
    auto writeRes = fileTool->execute(writeArgs);
    REQUIRE(writeRes.ok());

    // Read file
    qorvix::api::json::Value readArgs = qorvix::api::json::Value::object();
    readArgs["action"] = "read";
    readArgs["path"] = "sub/hello.txt";
    auto readRes = fileTool->execute(readArgs);
    REQUIRE(readRes.ok());
    REQUIRE(readRes.output == "Hello from QorVix Multi-Agent!");

    // List files
    qorvix::api::json::Value listArgs = qorvix::api::json::Value::object();
    listArgs["action"] = "list";
    listArgs["path"] = "sub";
    auto listRes = fileTool->execute(listArgs);
    REQUIRE(listRes.ok());
    REQUIRE(listRes.output.find("hello.txt") != std::string::npos);

    std::filesystem::remove_all(tempDir);
  }

  SECTION("System info tool") {
    auto sysTool = createSystemInfoTool();
    auto res = sysTool->execute(qorvix::api::json::Value());
    REQUIRE(res.ok());
    REQUIRE(res.output.find("QorVix Runtime") != std::string::npos);
  }
}

TEST_CASE("Blackboard state, tasks, messages, and artifacts", "[agents][blackboard]") {
  Blackboard bb;

  SECTION("Key-Value state operations") {
    bb.setState("project_name", "QorVix Core");
    bb.setState("version_major", 1);

    REQUIRE(bb.hasState("project_name"));
    REQUIRE(bb.getState("project_name").asString() == "QorVix Core");
    REQUIRE(bb.getState("version_major").asInt() == 1);

    auto keys = bb.listKeys();
    REQUIRE(keys.size() == 2);
  }

  SECTION("Task dependencies and ready tasks calculation") {
    Task t1;
    t1.id = "t1";
    t1.title = "Design Schema";
    t1.status = TaskStatus::Pending;

    Task t2;
    t2.id = "t2";
    t2.title = "Implement Backend";
    t2.status = TaskStatus::Pending;
    t2.dependencies = {"t1"};

    Task t3;
    t3.id = "t3";
    t3.title = "Deploy";
    t3.status = TaskStatus::Pending;
    t3.dependencies = {"t2"};

    bb.addTask(t1);
    bb.addTask(t2);
    bb.addTask(t3);

    auto ready = bb.readyTasks();
    REQUIRE(ready.size() == 1);
    REQUIRE(ready[0].id == "t1");

    bb.updateTaskStatus("t1", TaskStatus::Done, "Schema finalized");
    ready = bb.readyTasks();
    REQUIRE(ready.size() == 1);
    REQUIRE(ready[0].id == "t2");

    bb.updateTaskStatus("t2", TaskStatus::Done, "Backend complete");
    ready = bb.readyTasks();
    REQUIRE(ready.size() == 1);
    REQUIRE(ready[0].id == "t3");

    bb.updateTaskStatus("t3", TaskStatus::Done, "Deployed successfully");
    REQUIRE(bb.allTasksCompleted());
  }

  SECTION("Messages timeline and artifacts") {
    bb.postMessage("Planner", "Created breakdown", "plan");
    bb.postMessage("Coder", "Started implementation", "info");

    auto msgs = bb.messages();
    REQUIRE(msgs.size() == 2);
    REQUIRE(msgs[0].sender == "Planner");
    REQUIRE(msgs[1].sender == "Coder");

    Artifact art;
    art.name = "architecture_doc";
    art.content = "Multi-agent blackboard design specifications";
    art.createdBy = "Planner";
    bb.storeArtifact(art);

    auto retrieved = bb.getArtifact("architecture_doc");
    REQUIRE(retrieved.has_value());
    REQUIRE(retrieved->content == "Multi-agent blackboard design specifications");
  }

  SECTION("Blackboard JSON snapshot serialization and restore") {
    bb.setState("k1", "v1");
    Task t;
    t.id = "t1";
    t.title = "Snapshot task";
    bb.addTask(t);
    bb.postMessage("System", "Snapshot taken");

    auto jsonSnap = bb.toJson();
    REQUIRE(jsonSnap.isObject());

    Blackboard bb2;
    std::string err;
    REQUIRE(bb2.fromJson(jsonSnap, err));
    REQUIRE(bb2.hasState("k1"));
    REQUIRE(bb2.getState("k1").asString() == "v1");
    REQUIRE(bb2.listTasks().size() == 1);
    REQUIRE(bb2.messages().size() == 1);
  }
}

TEST_CASE("Agent ReAct loop and tool dispatch", "[agents][agent]") {
  auto registry = std::make_shared<ToolRegistry>();
  registerDefaultTools(*registry);
  auto blackboard = std::make_shared<Blackboard>();

  Agent agent("test_agent", createResearcherRole(), registry, blackboard);

  SECTION("Single-step direct final answer") {
    agent.setInferenceCallback([](const auto&, const auto&, const auto&, const auto&) {
      return "Final Answer: The answer is 42.";
    });

    auto res = agent.run("What is the ultimate answer?");
    REQUIRE(res.success);
    REQUIRE(res.finalAnswer == "The answer is 42.");
    REQUIRE(res.totalSteps == 1);
  }

  SECTION("Multi-step ReAct tool invocation") {
    int stepCount = 0;
    agent.setInferenceCallback([&stepCount](const std::vector<qorvix::api::ChatMessage>& history, const auto&, const auto&, const auto&) {
      ++stepCount;
      if (stepCount == 1) {
        return "I need to calculate 25 * 4.\nAction: calculator\nAction Input: 25 * 4";
      }
      // Step 2 receives the observation from step 1
      std::string lastObs;
      for (const auto& m : history) {
        if (m.role == "tool") lastObs = m.content;
      }
      return "Based on the calculation (" + lastObs + "), Final Answer: Result is 100.";
    });

    auto res = agent.run("Calculate 25 * 4");
    REQUIRE(res.success);
    REQUIRE(res.finalAnswer == "Result is 100.");
    REQUIRE(res.totalSteps == 2);
    REQUIRE(res.trace[0].toolCalls.size() == 1);
    REQUIRE(res.trace[0].toolCalls[0].name == "calculator");
    REQUIRE(res.trace[0].observations[0].output == "100");
  }
}

TEST_CASE("Multi-Agent Workflows execution", "[agents][workflow]") {
  auto tools = std::make_shared<ToolRegistry>();
  registerDefaultTools(*tools);
  auto bb = std::make_shared<Blackboard>();

  SECTION("Sequential workflow: Planner -> Coder -> Critic -> Synthesizer") {
    WorkflowOrchestrator orchestrator(tools, bb);

    // Mock inference engine that behaves according to agent role
    orchestrator.setInferenceCallback([](const std::vector<qorvix::api::ChatMessage>& history, const auto&, const auto&, const auto&) {
      std::string rolePrompt = history.empty() ? "" : history[0].content;
      if (rolePrompt.find("Strategic Planner") != std::string::npos) {
        return "Final Answer: Plan: 1. Parse input. 2. Process data. 3. Return result.";
      }
      if (rolePrompt.find("Software Engineer") != std::string::npos) {
        return "Final Answer: Code: struct Processor { void run() {} };";
      }
      if (rolePrompt.find("Quality Critic") != std::string::npos) {
        return "Final Answer: Review: Architecture is sound and verified.";
      }
      return "Final Answer: Complete solution synthesized with code and verification.";
    });

    auto workflow = orchestrator.createSoftwareDevTeam(WorkflowPattern::Sequential);
    auto res = workflow->run("Build a high-performance data processor");

    REQUIRE(res.success);
    REQUIRE(res.finalAnswer.find("Complete solution synthesized") != std::string::npos);
    REQUIRE(res.trace.size() == 4);
  }

  SECTION("Supervisor workflow: Supervisor delegates and synthesizes") {
    WorkflowOrchestrator orchestrator(tools, bb);

    int callCount = 0;
    orchestrator.setInferenceCallback([&callCount](const std::vector<qorvix::api::ChatMessage>&, const auto&, const auto&, const auto&) {
      ++callCount;
      if (callCount == 1) {
        return "Final Answer: Plan: Assign research to specialists.";
      } else if (callCount == 2) {
        return "Final Answer: Finding A: Quantum algorithms accelerate search.";
      } else if (callCount == 3) {
        return "Final Answer: Finding B: Error mitigation is essential.";
      }
      return "Final Answer: Synthesized Executive Summary: Quantum computing provides exponential advantages with error mitigation.";
    });

    auto workflow = orchestrator.createResearchTeam(WorkflowPattern::HierarchicalSupervisor);
    auto res = workflow->run("Analyze quantum computing state-of-the-art");

    REQUIRE(res.success);
    REQUIRE(res.finalAnswer.find("Synthesized Executive Summary") != std::string::npos);
  }

  SECTION("Consensus workflow: Proposers and Judge") {
    WorkflowOrchestrator orchestrator(tools, bb);

    int callCount = 0;
    orchestrator.setInferenceCallback([&callCount](const std::vector<qorvix::api::ChatMessage>&, const auto&, const auto&, const auto&) {
      ++callCount;
      if (callCount == 1) {
        return "Final Answer: Proposal 1: Use microkernel architecture.";
      } else if (callCount == 2) {
        return "Final Answer: Proposal 2: Use monolithic modular architecture.";
      }
      return "Final Answer: Consensus Verdict: Adopt modular architecture with microkernel principles for security.";
    });

    auto workflow = orchestrator.createConsensusTeam(2);
    auto res = workflow->run("Select the best OS kernel architecture");

    REQUIRE(res.success);
    REQUIRE(res.finalAnswer.find("Consensus Verdict") != std::string::npos);
  }
}

TEST_CASE("Agent Skills definition, registry, loader, and execution", "[agents][skill]") {
  SECTION("SkillDefinition markdown frontmatter parsing and formatting") {
    std::string mdContent = R"(---
name: custom-audit
version: 1.2.0
description: Custom vulnerability audit playbook
category: security
author: TestAuthor
tags: ["security", "audit", "cve"]
required_tools: ["file_ops", "calculator"]
---

### Security Audit Procedure:
1. Scan for OWASP Top 10 vulnerabilities.
2. Check memory safety and authentication guards.
)";

    SkillDefinition def;
    std::string err;
    REQUIRE(SkillDefinition::fromMarkdown(mdContent, def, err));
    REQUIRE(def.name == "custom-audit");
    REQUIRE(def.version == "1.2.0");
    REQUIRE(def.category == "security");
    REQUIRE(def.author == "TestAuthor");
    REQUIRE(def.tags.size() == 3);
    REQUIRE(def.requiredTools.size() == 2);
    REQUIRE(def.requiredTools[0] == "file_ops");
    REQUIRE(def.instructions.find("OWASP Top 10") != std::string::npos);

    // Markdown roundtrip
    std::string exportedMd = def.toMarkdown();
    SkillDefinition def2;
    REQUIRE(SkillDefinition::fromMarkdown(exportedMd, def2, err));
    REQUIRE(def2.name == "custom-audit");
    REQUIRE(def2.version == "1.2.0");
  }

  SECTION("SkillRegistry catalog, filtering, and prompt composition") {
    SkillRegistry registry;
    registerDefaultSkills(registry);

    REQUIRE(registry.listSkills().size() >= 5);
    REQUIRE(registry.hasSkill("code-review"));
    REQUIRE(registry.hasSkill("math-solver"));
    REQUIRE(registry.hasSkill("rag-search"));

    auto codingSkills = registry.searchSkills("", "", "coding");
    REQUIRE(codingSkills.size() >= 2);

    auto mathSkills = registry.searchSkills("calculator", "", "");
    REQUIRE(mathSkills.size() >= 1);

    std::string prompt = registry.composeSkillsPrompt({"code-review"});
    REQUIRE(prompt.find("Code Review Playbook") != std::string::npos);
    REQUIRE(prompt.find("math-solver") == std::string::npos);
  }

  SECTION("SkillLoader discovery from filesystem") {
    std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "qorvix_skills_test";
    std::filesystem::create_directories(tempDir);

    SkillDefinition customSkill;
    customSkill.name = "disk-skill";
    customSkill.version = "2.0.0";
    customSkill.description = "A skill loaded from disk";
    customSkill.category = "workflow";
    customSkill.instructions = "Step 1: Perform disk operation.";

    std::string err;
    REQUIRE(SkillLoader::saveToDirectory(customSkill, tempDir, err));

    SkillRegistry reg;
    std::size_t loaded = SkillLoader::loadFromDirectory(tempDir, reg, err);
    REQUIRE(loaded >= 1);
    REQUIRE(reg.hasSkill("disk-skill"));

    auto loadedSkill = reg.getSkill("disk-skill");
    REQUIRE(loadedSkill.has_value());
    REQUIRE(loadedSkill->version == "2.0.0");

    std::filesystem::remove_all(tempDir);
  }

  SECTION("Agent with active skills in system prompt and meta-tool invocation") {
    auto toolReg = std::make_shared<ToolRegistry>();
    registerDefaultTools(*toolReg);
    auto skillReg = std::make_shared<SkillRegistry>();
    registerDefaultSkills(*skillReg);

    // Register skills as callable meta-tools
    skillReg->registerAllAsTools(*toolReg);
    REQUIRE(toolReg->hasTool("skill_code-review"));
    REQUIRE(toolReg->hasTool("skill_math-solver"));

    RoleDefinition customRole = createResearcherRole();
    customRole.allowedSkills = {"code-review", "math-solver"};

    Agent agent("skilled_agent", customRole, toolReg, nullptr, skillReg);

    // System prompt should contain the activated skill playbooks
    const auto& history = agent.history();
    REQUIRE_FALSE(history.empty());
    REQUIRE(history[0].role == "system");
    REQUIRE(history[0].content.find("Code Review Playbook") != std::string::npos);
    REQUIRE(history[0].content.find("Math Reasoning Playbook") != std::string::npos);

    // Agent can invoke the skill meta-tool
    qorvix::api::json::Value args = qorvix::api::json::Value::object();
    ToolResult skillToolRes = toolReg->execute("skill_code-review", args);
    REQUIRE(skillToolRes.ok());
    REQUIRE(skillToolRes.output.find("Code Review Playbook") != std::string::npos);
  }
}

TEST_CASE("Agent Artifacts system, versioning, diff, store, and tools", "[agents][artifact]") {
  SECTION("Artifact creation, metadata, and markdown rendering") {
    ArtifactMetadata meta;
    meta.title = "Core Network Engine";
    meta.summary = "Asynchronous HTTP socket handler in C++23";
    meta.language = "cpp";
    meta.author = "CoderAgent";
    meta.tags = {"network", "cpp", "async"};

    Artifact art("net_engine", ArtifactType::Code, "class NetEngine {\npublic:\n  void listen(int port);\n};\n", meta);
    REQUIRE(art.name() == "net_engine");
    REQUIRE(art.type() == ArtifactType::Code);
    REQUIRE(art.currentVersion() == 1);
    REQUIRE(art.history().size() == 1);

    std::string md = art.toMarkdown();
    REQUIRE(md.find("Core Network Engine") == std::string::npos); // Summary or title
    REQUIRE(md.find("Asynchronous HTTP socket handler") != std::string::npos);
    REQUIRE(md.find("```cpp") != std::string::npos);

    // JSON round-trip
    auto jsonVal = art.toJson();
    Artifact art2;
    std::string err;
    REQUIRE(Artifact::fromJson(jsonVal, art2, err));
    REQUIRE(art2.name() == "net_engine");
    REQUIRE(art2.currentVersion() == 1);
    REQUIRE(art2.metadata().language == "cpp");
  }

  SECTION("Artifact version history and line diff") {
    Artifact art("algorithm", ArtifactType::Code, "int compute(int x) {\n  return x * 2;\n}\n");
    REQUIRE(art.currentVersion() == 1);

    // Update to v2
    art.updateContent("int compute(int x) {\n  // Optimized\n  return x << 1;\n}\n", "OptimizerAgent", "Use bitwise shift");
    REQUIRE(art.currentVersion() == 2);
    REQUIRE(art.history().size() == 2);

    auto contentV1 = art.getContentAtVersion(1);
    auto contentV2 = art.getContentAtVersion(2);
    REQUIRE(contentV1.has_value());
    REQUIRE(contentV2.has_value());
    REQUIRE(contentV1->find("return x * 2;") != std::string::npos);
    REQUIRE(contentV2->find("return x << 1;") != std::string::npos);

    ArtifactDiff diff = art.diff(1, 2);
    REQUIRE(diff.fromVersion == 1);
    REQUIRE(diff.toVersion == 2);
    REQUIRE(diff.additions > 0);
    REQUIRE(diff.deletions > 0);
    std::string diffStr = diff.toString();
    REQUIRE(diffStr.find("-  return x * 2;") != std::string::npos);
    REQUIRE(diffStr.find("+  return x << 1;") != std::string::npos);
  }

  SECTION("ArtifactStore query, filtering, and directory sync") {
    ArtifactStore store;

    Artifact a1("doc1", ArtifactType::Markdown, "# Title 1\nContent 1\n");
    a1.mutableMetadata().author = "AuthorA";
    a1.mutableMetadata().tags = {"spec", "v1"};

    Artifact a2("code1", ArtifactType::Code, "void fn() {}\n");
    a2.mutableMetadata().author = "AuthorB";
    a2.mutableMetadata().tags = {"cpp"};

    REQUIRE(store.createArtifact(std::move(a1)));
    REQUIRE(store.createArtifact(std::move(a2)));
    REQUIRE(store.listArtifacts().size() == 2);
    REQUIRE(store.hasArtifact("doc1"));
    REQUIRE(store.hasArtifact("code1"));

    auto codeArts = store.filterArtifacts("", ArtifactType::Code);
    REQUIRE(codeArts.size() == 1);
    REQUIRE(codeArts[0].name() == "code1");

    auto authorAArts = store.filterArtifacts("", std::nullopt, "AuthorA");
    REQUIRE(authorAArts.size() == 1);
    REQUIRE(authorAArts[0].name() == "doc1");

    // Test directory sync & load
    std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "qorvix_artifact_test";
    std::filesystem::create_directories(tempDir);

    std::string err;
    REQUIRE(store.syncToDirectory(tempDir, err));

    ArtifactStore store2;
    std::size_t loaded = store2.loadFromDirectory(tempDir, err);
    REQUIRE(loaded >= 2);
    REQUIRE(store2.hasArtifact("doc1"));
    REQUIRE(store2.hasArtifact("code1"));

    std::filesystem::remove_all(tempDir);
  }

  SECTION("Agent artifact tool execution (create, read, update, list, diff)") {
    auto store = std::make_shared<ArtifactStore>();
    ToolRegistry reg;
    registerArtifactTools(reg, store);

    REQUIRE(reg.hasTool("artifact_create"));
    REQUIRE(reg.hasTool("artifact_read"));
    REQUIRE(reg.hasTool("artifact_update"));
    REQUIRE(reg.hasTool("artifact_list"));
    REQUIRE(reg.hasTool("artifact_diff"));

    // 1. artifact_create
    auto createArgs = qorvix::api::json::Value::object();
    createArgs["name"] = "test_script";
    createArgs["content"] = "print('hello world')\n";
    createArgs["type"] = "code";
    createArgs["language"] = "python";
    createArgs["summary"] = "A simple hello world script";

    auto createRes = reg.execute("artifact_create", createArgs);
    REQUIRE(createRes.ok());
    REQUIRE(store->hasArtifact("test_script"));

    // 2. artifact_read
    auto readArgs = qorvix::api::json::Value::object();
    readArgs["name"] = "test_script";
    auto readRes = reg.execute("artifact_read", readArgs);
    REQUIRE(readRes.ok());
    REQUIRE(readRes.output.find("print('hello world')") != std::string::npos);

    // 3. artifact_update
    auto updateArgs = qorvix::api::json::Value::object();
    updateArgs["name"] = "test_script";
    updateArgs["content"] = "print('hello qorvix')\n";
    updateArgs["change_summary"] = "Greet QorVix instead of world";
    auto updateRes = reg.execute("artifact_update", updateArgs);
    REQUIRE(updateRes.ok());

    auto art = store->getArtifact("test_script");
    REQUIRE(art.has_value());
    REQUIRE(art->currentVersion() == 2);

    // 4. artifact_diff
    auto diffArgs = qorvix::api::json::Value::object();
    diffArgs["name"] = "test_script";
    diffArgs["from_version"] = 1;
    diffArgs["to_version"] = 2;
    auto diffRes = reg.execute("artifact_diff", diffArgs);
    REQUIRE(diffRes.ok());
    REQUIRE(diffRes.output.find("-print('hello world')") != std::string::npos);
    REQUIRE(diffRes.output.find("+print('hello qorvix')") != std::string::npos);

    // 5. artifact_list
    auto listArgs = qorvix::api::json::Value::object();
    auto listRes = reg.execute("artifact_list", listArgs);
    REQUIRE(listRes.ok());
    REQUIRE(listRes.output.find("test_script") != std::string::npos);
  }
}


