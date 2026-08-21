# QorVix Multi-Agent Runtime (`agents/`)

The `agents/` subsystem implements QorVix's native, zero-dependency C++23 multi-agent operating system and orchestration engine (SPEC "Multi-Agent Workflows", Phase 11b).

It enables collaborative, role-specialized agents to coordinate on complex tasks via tool execution and a shared blackboard memory architecture.

---

## 1. Core Architecture

```
                               ┌────────────────────────┐
                               │  Workflow Orchestrator │
                               └──────────┬─────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │                                             │
        ┌──────────▼──────────┐                       ┌──────────▼──────────┐
        │     Agent: Lead     │                       │   Agent: Specialist │
        │ (Role: Coordinator) │                       │    (Role: Coder)    │
        └──────────┬──────────┘                       └──────────┬──────────┘
                   │                                             │
                   └──────────────────────┬──────────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │                                 │                                 │
┌───────▼────────┐              ┌─────────▼────────┐              ┌─────────▼────────┐
│ Shared Memory  │              │    Task Graph    │              │  Tool Registry   │
│   Blackboard   │              │  & Dependencies  │              │  & MCP Bridges   │
└────────────────┘              └──────────────────┘              └──────────────────┘
```

---

## 2. Agent Roles & Personas (`role.hpp`)

QorVix provides pre-configured role personas with tailored system prompts, sampling parameters, tool permission sets, and reasoning step ceilings:

| Role | Name | Purpose | Default Sampling |
|------|------|---------|-------------------|
| `Coordinator` | Coordinator | Oversees multi-agent workflows, coordinates assignments, and ensures goal convergence | Temp: 0.3 |
| `Planner` | Planner | Decomposes complex user goals into dependency-aware subtasks | Temp: 0.2 |
| `Researcher` | Researcher | Gathers facts, searches vector index, and synthesizes findings | Temp: 0.3 |
| `Coder` | Coder | Writes, inspects, and refactors clean, high-performance code | Temp: 0.1 |
| `Critic` | Critic | Reviews logic, identifies flaws, and verifies output accuracy | Temp: 0.1 |
| `Executor` | Executor | Runs operational commands and records output states | Temp: 0.2 |
| `Synthesizer` | Synthesizer | Merges multi-agent contributions into a cohesive final answer | Temp: 0.4 |
| `Custom` | User-defined | Configurable role with custom prompts and permissions | Configurable |

---

## 3. Tool System & Registry (`tool.hpp`, `tool_registry.hpp`, `builtin_tools.hpp`)

The `ToolRegistry` allows agents to discover and invoke tools safely:
- **OpenAI & MCP Schema Export:** Automatically serializes tool definitions to standard OpenAI Function Call definitions and Anthropic Model Context Protocol (MCP) tool schemas.
- **Built-in Tools:**
  - `calculator`: Safe recursive-descent math parser (`+`, `-`, `*`, `/`, `^`, `%`, `sqrt`, `abs`, `sin`, `cos`, `tan`, `log`, `exp`, `pi`, `e`).
  - `file_ops`: Safe workspace file reader, writer, and directory lister.
  - `blackboard`: Mutates or inspects keys, tasks, and message timeline on the active blackboard.
  - `rag_search`: Hybrid lexical (BM25) and dense vector search via `qorvix::rag::Index`.
  - `system_info`: Inspects runtime version, CPU threads, and hardware features (AVX2, AVX-512).
  - `mcp_bridge`: Bridges external MCP server tools into native agent tools.

---

## 4. Shared Blackboard (`blackboard.hpp`)

The `Blackboard` serves as the centralized, thread-safe communication and memory backbone:
- **Key-Value State Store:** Dynamic typed values (`api::json::Value`) accessible across all agents.
- **Task Graph & Dependency Resolution:** Subtasks with dependencies (`t2` depends on `t1`), status tracking (`Pending`, `InProgress`, `Review`, `Done`, `Failed`), and automatic `readyTasks()` scheduling.
- **Message Timeline:** Inter-agent broadcast and direct messaging with sender attribution and timestamps.
- **Artifact Store:** Named artifacts (code blocks, analysis summaries, search citations) produced during execution.
- **Observers:** Pub/Sub callbacks for live telemetry when state, tasks, or messages update.
- **Snapshot Persistence:** Full JSON serialization and deserialization (`toJson()` / `fromJson()`).

---

## 5. Reasoning Engine (`agent.hpp`)

Each `Agent` executes a ReAct reasoning cycle:
1. **Thought:** Agent analyzes input and formulates next action.
2. **Action / Tool Call:** Identifies tool to invoke and formats JSON arguments.
3. **Observation:** Tool executes and feeds output back into message context.
4. **Final Answer:** Emitted when goal is satisfied, terminating the loop.

---

## 6. Multi-Agent Workflow Patterns (`workflow.hpp`, `orchestrator.hpp`)

Four orchestration topologies are supported:

1. **Sequential Pipeline:**
   Chains outputs sequentially (e.g., `Planner -> Coder -> Critic -> Synthesizer`).
2. **Hierarchical Supervisor:**
   A `Coordinator` agent creates a high-level plan, delegates subtasks to specialist workers (`Researcher`, `Coder`, `Critic`), collects results, and synthesizes a final verdict.
3. **Round-Robin Blackboard:**
   Autonomous agents continuously poll the shared blackboard for eligible tasks whose dependencies have been satisfied, execute them, write back results, until all tasks are complete.
4. **Consensus Voting:**
   Multiple independent proposer agents generate alternative solutions, which are subsequently evaluated and synthesized by a Judge/Critic agent.

---

## 7. Agent Skills System (`skill.hpp`, `skill_registry.hpp`, `skill_loader.hpp`, `builtin_skills.hpp`)

The **Skills System** packages high-level operational playbooks, procedural guidelines, tool combinations, and few-shot examples that agents can acquire and execute dynamically.

### Structure of a Skill Package (`SKILL.md`)
Skills are stored on disk in the `skills/` directory (or user-defined directories) with YAML/JSON frontmatter headers:
```markdown
---
name: code-review
version: 1.0.0
description: Comprehensive code review and vulnerability audit
category: coding
author: QorVix
tags: ["code", "review", "security", "cpp"]
required_tools: ["file_ops"]
---

# Code Review Playbook
1. Check for buffer overflows, use-after-free, and concurrency races.
2. Verify modern C++23 RAII idioms and span/view parameters.
3. Categorize findings into [CRITICAL], [MAJOR], [MINOR], [STYLE].
```

### Built-in Skills Catalog
- `code-review`: Static analysis, vulnerability detection, and C++23 best practices.
- `rag-search`: Strict grounded knowledge retrieval with explicit citation attribution.
- `math-solver`: Step-by-step problem decomposition and calculator verification.
- `cpp-refactor`: Modern C++23 refactoring (RAII, string_view, smart pointers).
- `git-workflow`: Conventional commit formulation and changelog synthesis.

### Dynamic Discovery & Meta-Tool Bridging
- **Dynamic Prompt Synthesis:** The `SkillRegistry` automatically formats and injects active skill playbooks into the agent's system prompt.
- **Meta-Tool Invocation:** Skills can be registered directly into the `ToolRegistry` as executable tools (e.g. `skill_code-review`), allowing agents to invoke playbooks explicitly.
- **Filesystem Loader:** `SkillLoader` recursively scans directories for `SKILL.md` files and hot-registers them.

---

## 8. Agent Artifacts System (`artifact.hpp`, `artifact_store.hpp`, `artifact_tools.hpp`)

Artifacts are first-class, structured digital objects created, iteratively edited, reviewed, versioned, and cited across agents and workflows:

### Supported Types & Features
- **Artifact Types:** `Code`, `Markdown`, `Document`, `Diff`, `Table`, `Json`, `Image`, `Binary`, `Custom`.
- **Revision History & Snapshots:** Every update creates an immutable historical version snapshot ([`ArtifactVersion`](file:///C:/Users/BN/Documents/QorVix/agents/include/qorvix/agents/artifact.hpp#L42-L50)) recording the timestamp, author, and change rationale.
- **Unified Line Diff Engine:** Computes line-by-line diffs ([`ArtifactDiff`](file:///C:/Users/BN/Documents/QorVix/agents/include/qorvix/agents/artifact.hpp#L59-L68)) comparing arbitrary historical revisions.
- **Thread-Safe ArtifactStore:** In-memory catalog with tag/type/author filtering and automatic filesystem directory synchronization.
- **Dedicated Agent Tools:**
  - `artifact_create`: Initializes a new named artifact with metadata.
  - `artifact_read`: Retrieves content and metadata (latest or specific revision).
  - `artifact_update`: Updates content, records commit message, and bumps version.
  - `artifact_diff`: Returns unified diff output between two versions.
  - `artifact_list`: Enumerates all active artifacts.

---

## 9. CLI & Correctness Gate

```bash
# Run the Phase 11b correctness gate (all 7 tiers verified)
qorvix agent-check

# Inspect agent capabilities
qorvix agent roles
qorvix agent tools
qorvix agent skills [optional_dir]
qorvix agent artifacts [list|show <name>|diff <name> <v1> [v2]]

# Execute multi-agent workflows from CLI
qorvix agent run --workflow sequential --goal "Design and implement a telemetry dashboard"
qorvix agent run --workflow supervisor --goal "Perform comprehensive security review"
qorvix agent run --workflow consensus --goal "Choose optimal cache eviction strategy"
```
