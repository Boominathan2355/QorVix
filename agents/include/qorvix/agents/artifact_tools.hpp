#pragma once

#include <memory>

#include "qorvix/agents/artifact_store.hpp"
#include "qorvix/agents/tool.hpp"
#include "qorvix/agents/tool_registry.hpp"

namespace qorvix::agents {

// Factory for artifact management tools
std::shared_ptr<ITool> createArtifactCreateTool(std::shared_ptr<ArtifactStore> store);
std::shared_ptr<ITool> createArtifactReadTool(std::shared_ptr<ArtifactStore> store);
std::shared_ptr<ITool> createArtifactUpdateTool(std::shared_ptr<ArtifactStore> store);
std::shared_ptr<ITool> createArtifactListTool(std::shared_ptr<ArtifactStore> store);
std::shared_ptr<ITool> createArtifactDiffTool(std::shared_ptr<ArtifactStore> store);

// Registers all artifact management tools into a ToolRegistry
void registerArtifactTools(ToolRegistry& registry, std::shared_ptr<ArtifactStore> store);

}  // namespace qorvix::agents
