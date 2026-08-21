#include "qorvix/agents/builtin_skills.hpp"

#include "qorvix/agents/skill_loader.hpp"

namespace qorvix::agents {

SkillDefinition createCodeReviewSkill() {
  SkillDefinition s;
  s.name = "code-review";
  s.version = "1.0.0";
  s.description = "Performs exhaustive code review, vulnerability assessment, and performance optimization.";
  s.category = "coding";
  s.author = "QorVix";
  s.tags = {"code", "review", "security", "cpp", "performance"};
  s.requiredTools = {"file_ops"};
  s.instructions =
      "### Code Review Playbook:\n"
      "1. **Safety & Correctness**: Check for buffer overflows, use-after-free, null pointer dereferences, "
      "data races, and unhandled exception paths.\n"
      "2. **Modern C++23 Idioms**: Verify RAII resource management, use of std::string_view / std::span, "
      "proper const-correctness, and zero unnecessary dynamic allocations in hot loops.\n"
      "3. **Performance & Complexity**: Scrutinize time and space complexity, cache locality, and potential SIMD/GPU opportunities.\n"
      "4. **Format & Constructive Feedback**: Present issues categorized by Severity (Critical, Major, Minor, Style) "
      "with concrete diff suggestions.";

  SkillExample ex;
  ex.title = "Reviewing a raw pointer loop";
  ex.input = "char* buf = (char*)malloc(100); strcpy(buf, input);";
  ex.output = "Critical: Unbounded buffer copy (buffer overflow) and raw malloc. Replace with std::string or std::vector<char>.";
  s.examples.push_back(std::move(ex));
  return s;
}

SkillDefinition createRagSearchSkill() {
  SkillDefinition s;
  s.name = "rag-search";
  s.version = "1.0.0";
  s.description = "Executes grounded knowledge retrieval, multi-source synthesis, and factual citation.";
  s.category = "research";
  s.author = "QorVix";
  s.tags = {"rag", "search", "retrieval", "embeddings"};
  s.requiredTools = {"rag_search"};
  s.instructions =
      "### Grounded RAG Search Playbook:\n"
      "1. **Formulate Query**: Break down the user question into keyword-dense subqueries.\n"
      "2. **Retrieve Chunks**: Invoke `rag_search` with top_k >= 5.\n"
      "3. **Validate Grounding**: Ensure claims in the answer are strictly supported by the retrieved chunk snippets.\n"
      "4. **Cite Sources**: Include explicit bracketed citations [1], [2] referencing source filenames and chunk offsets.";

  return s;
}

SkillDefinition createMathSolverSkill() {
  SkillDefinition s;
  s.name = "math-solver";
  s.version = "1.0.0";
  s.description = "Decomposes complex mathematical problems and verifies arithmetic with calculator tool.";
  s.category = "math";
  s.author = "QorVix";
  s.tags = {"math", "calculator", "algebra", "arithmetic"};
  s.requiredTools = {"calculator"};
  s.instructions =
      "### Math Reasoning Playbook:\n"
      "1. **Deconstruct**: Identify given variables, equations, and target outputs.\n"
      "2. **Derive Formulation**: Formulate algebraic steps step-by-step.\n"
      "3. **Tool Verification**: Always execute intermediate calculations and final values through the `calculator` tool.\n"
      "4. **Sanity Check**: Verify dimensional consistency, boundary conditions, and sign.";

  return s;
}

SkillDefinition createCppRefactorSkill() {
  SkillDefinition s;
  s.name = "cpp-refactor";
  s.version = "1.0.0";
  s.description = "Refactors legacy C/C++ code into modern, clean, zero-cost C++23 standards.";
  s.category = "coding";
  s.author = "QorVix";
  s.tags = {"cpp", "refactor", "modern-cpp", "raii"};
  s.requiredTools = {"file_ops"};
  s.instructions =
      "### Modern C++23 Refactoring Playbook:\n"
      "1. **Eliminate Raw Resource Ownership**: Replace manual `new`/`delete`/`malloc` with `std::unique_ptr`, `std::shared_ptr`, or value semantics.\n"
      "2. **String & Buffer Views**: Replace `const std::string&` and pointer-length pairs with `std::string_view` and `std::span` where appropriate.\n"
      "3. **Standard Algorithms**: Replace error-prone raw index loops with `<algorithm>` ranges or structured bindings.\n"
      "4. **Const & Noexcept**: Apply `const`, `constexpr`, and `noexcept` to move operations and non-throwing functions.";

  return s;
}

SkillDefinition createGitWorkflowSkill() {
  SkillDefinition s;
  s.name = "git-workflow";
  s.version = "1.0.0";
  s.description = "Generates conventional commit messages, release changelogs, and branch review notes.";
  s.category = "workflow";
  s.author = "QorVix";
  s.tags = {"git", "commits", "changelog", "devops"};
  s.instructions =
      "### Conventional Git Workflow Playbook:\n"
      "1. **Commit Format**: Use `<type>(<scope>): <short description>`.\n"
      "   Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.\n"
      "2. **Body**: Explain *why* the change was made, rationale, and non-obvious design decisions.\n"
      "3. **Breaking Changes**: Explicitly mark `BREAKING CHANGE:` in the footer.";

  return s;
}

void registerDefaultSkills(SkillRegistry& registry, const std::filesystem::path& skillsDir) {
  registry.registerSkill(createCodeReviewSkill());
  registry.registerSkill(createRagSearchSkill());
  registry.registerSkill(createMathSolverSkill());
  registry.registerSkill(createCppRefactorSkill());
  registry.registerSkill(createGitWorkflowSkill());

  if (!skillsDir.empty() && std::filesystem::exists(skillsDir)) {
    std::string err;
    SkillLoader::loadFromDirectory(skillsDir, registry, err);
  }
}

}  // namespace qorvix::agents
