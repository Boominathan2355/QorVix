#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "qorvix/agents/artifact.hpp"
#include "qorvix/agents/artifact_store.hpp"
#include "qorvix/api/json.hpp"

namespace qorvix::agents {

enum class TaskStatus {
  Pending,
  InProgress,
  Review,
  Done,
  Failed
};

std::string_view taskStatusName(TaskStatus status);
TaskStatus parseTaskStatus(std::string_view name);

struct Task {
  std::string id;
  std::string title;
  std::string description;
  std::string assignedAgent;
  TaskStatus status = TaskStatus::Pending;
  std::vector<std::string> dependencies;
  std::string result;
  std::vector<std::string> artifacts;
  std::int64_t createdTimestamp = 0;
  std::int64_t updatedTimestamp = 0;

  api::json::Value toJson() const;
  static bool fromJson(const api::json::Value& v, Task& out, std::string& error);
};

struct BlackboardMessage {
  std::string id;
  std::string sender;
  std::string recipient = "broadcast";
  std::string type = "info";  // "info", "plan", "thought", "action", "critique", "artifact", "error"
  std::string content;
  std::int64_t timestamp = 0;
  api::json::Value metadata;

  api::json::Value toJson() const;
  static bool fromJson(const api::json::Value& v, BlackboardMessage& out, std::string& error);
};

// Central shared knowledge repository and task blackboard for multi-agent workflows
class Blackboard {
 public:
  Blackboard();
  ~Blackboard() = default;

  // Non-copyable, movable
  Blackboard(const Blackboard&) = delete;
  Blackboard& operator=(const Blackboard&) = delete;
  Blackboard(Blackboard&&) noexcept;
  Blackboard& operator=(Blackboard&&) noexcept;

  // Key-Value State
  void setState(const std::string& key, api::json::Value value);
  api::json::Value getState(const std::string& key) const;
  bool hasState(const std::string& key) const;
  std::vector<std::string> listKeys() const;
  bool removeState(const std::string& key);

  // Task Board
  std::string addTask(Task task);
  bool updateTaskStatus(const std::string& taskId, TaskStatus status, const std::string& result = "");
  bool assignTask(const std::string& taskId, const std::string& agentName);
  std::optional<Task> getTask(const std::string& taskId) const;
  std::vector<Task> listTasks() const;
  std::vector<Task> readyTasks() const;  // tasks with all dependencies satisfied (Done)
  bool allTasksCompleted() const;

  // Message Timeline
  void postMessage(const std::string& sender, const std::string& content,
                   const std::string& type = "info", const std::string& recipient = "broadcast",
                   api::json::Value metadata = api::json::Value());
  std::vector<BlackboardMessage> messages() const;
  std::vector<BlackboardMessage> recentMessages(std::size_t count) const;

  // Artifact Store
  void storeArtifact(Artifact artifact);
  std::optional<Artifact> getArtifact(const std::string& name) const;
  std::vector<Artifact> listArtifacts() const;
  std::shared_ptr<ArtifactStore> artifactStore() const { return artifactStore_; }

  // Observers
  using StateObserver = std::function<void(const std::string& key, const api::json::Value& val)>;
  using TaskObserver = std::function<void(const Task& task)>;
  using MessageObserver = std::function<void(const BlackboardMessage& msg)>;

  void subscribeState(StateObserver obs);
  void subscribeTask(TaskObserver obs);
  void subscribeMessage(MessageObserver obs);

  // Snapshot serialization & restore
  api::json::Value toJson() const;
  bool fromJson(const api::json::Value& v, std::string& error);

  void clear();

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, api::json::Value> state_;
  std::unordered_map<std::string, Task> tasks_;
  std::vector<std::string> taskOrder_;
  std::vector<BlackboardMessage> messages_;
  std::shared_ptr<ArtifactStore> artifactStore_;

  std::vector<StateObserver> stateObservers_;
  std::vector<TaskObserver> taskObservers_;
  std::vector<MessageObserver> messageObservers_;
};

  std::vector<StateObserver> stateObservers_;
  std::vector<TaskObserver> taskObservers_;
  std::vector<MessageObserver> messageObservers_;
};

}  // namespace qorvix::agents
