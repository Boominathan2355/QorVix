add_library(qorvix_agents STATIC
  src/role.cpp
  src/tool.cpp
  src/tool_registry.cpp
  src/builtin_tools.cpp
  src/skill.cpp
  src/skill_registry.cpp
  src/skill_loader.cpp
  src/builtin_skills.cpp
  src/artifact.cpp
  src/artifact_store.cpp
  src/artifact_tools.cpp
  src/blackboard.cpp
  src/agent.cpp
  src/workflow.cpp
  src/orchestrator.cpp
)

target_include_directories(qorvix_agents PUBLIC include)
target_compile_features(qorvix_agents PUBLIC cxx_std_23)

# Agents subsystem bridges tools across RAG, Embeddings, Tokenizer, Runtime, and API protocol layers.
target_link_libraries(qorvix_agents PUBLIC
  qorvix_api
  qorvix_runtime
  qorvix_rag
  qorvix_embeddings
  qorvix_tokenizer
)
