#include "qorvix/agents/blackboard.hpp"

#include <chrono>
#include <sstream>

namespace qorvix::agents {

namespace {
std::int64_t currentEpochMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
}  // namespace

std::string_view taskStatusName(TaskStatus status) {
  switch (status) {
    case TaskStatus::Pending: return "pending";
    case TaskStatus::InProgress: return "in_progress";
    case TaskStatus::Review: return "review";
    case TaskStatus::Done: return "done";
    case TaskStatus::Failed: return "failed";
  }
  return "pending";
}

TaskStatus parseTaskStatus(std::string_view name) {
  if (name == "in_progress") return TaskStatus::InProgress;
  if (name == "review") return TaskStatus::Review;
  if (name == "done") return TaskStatus::Done;
  if (name == "failed") return TaskStatus::Failed;
  return TaskStatus::Pending;
}

api::json::Value Task::toJson() const {
  auto v = api::json::Value::object();
  v["id"] = id;
  v["title"] = title;
  v["description"] = description;
  v["assigned_agent"] = assignedAgent;
  v["status"] = std::string(taskStatusName(status));
  v["result"] = result;
  v["created_at"] = createdTimestamp;
  v["updated_at"] = updatedTimestamp;

  auto deps = api::json::Value::array();
  for (const auto& d : dependencies) deps.push(d);
  v["dependencies"] = std::move(deps);

  auto arts = api::json::Value::array();
  for (const auto& a : artifacts) arts.push(a);
  v["artifacts"] = std::move(arts);

  return v;
}

bool Task::fromJson(const api::json::Value& v, Task& out, std::string& error) {
  if (!v.isObject()) {
    error = "Task expects a JSON object";
    return false;
  }
  if (const auto* i = v.get("id"); i && i->isString()) out.id = i->asString();
  else { error = "Missing task id"; return false; }

  if (const auto* t = v.get("title"); t && t->isString()) out.title = t->asString();
  if (const auto* d = v.get("description"); d && d->isString()) out.description = d->asString();
  if (const auto* a = v.get("assigned_agent"); a && a->isString()) out.assignedAgent = a->asString();
  if (const auto* s = v.get("status"); s && s->isString()) out.status = parseTaskStatus(s->asString());
  if (const auto* r = v.get("result"); r && r->isString()) out.result = r->asString();
  if (const auto* c = v.get("created_at"); c && c->isNumber()) out.createdTimestamp = static_cast<std::int64_t>(c->asNumber());
  if (const auto* u = v.get("updated_at"); u && u->isNumber()) out.updatedTimestamp = static_cast<std::int64_t>(u->asNumber());

  out.dependencies.clear();
  if (const auto* deps = v.get("dependencies"); deps && deps->isArray()) {
    for (const auto& item : deps->items()) {
      if (item.isString()) out.dependencies.push_back(item.asString());
    }
  }

  out.artifacts.clear();
  if (const auto* arts = v.get("artifacts"); arts && arts->isArray()) {
    for (const auto& item : arts->items()) {
      if (item.isString()) out.artifacts.push_back(item.asString());
    }
  }

  return true;
}

api::json::Value BlackboardMessage::toJson() const {
  auto v = api::json::Value::object();
  v["id"] = id;
  v["sender"] = sender;
  v["recipient"] = recipient;
  v["type"] = type;
  v["content"] = content;
  v["timestamp"] = timestamp;
  if (!metadata.isNull()) v["metadata"] = metadata;
  return v;
}

bool BlackboardMessage::fromJson(const api::json::Value& v, BlackboardMessage& out, std::string& error) {
  if (!v.isObject()) {
    error = "BlackboardMessage expects a JSON object";
    return false;
  }
  if (const auto* i = v.get("id"); i && i->isString()) out.id = i->asString();
  if (const auto* s = v.get("sender"); s && s->isString()) out.sender = s->asString();
  if (const auto* r = v.get("recipient"); r && r->isString()) out.recipient = r->asString();
  if (const auto* t = v.get("type"); t && t->isString()) out.type = t->asString();
  if (const auto* c = v.get("content"); c && c->isString()) out.content = c->asString();
  if (const auto* ts = v.get("timestamp"); ts && ts->isNumber()) out.timestamp = static_cast<std::int64_t>(ts->asNumber());
  if (const auto* m = v.get("metadata"); m) out.metadata = *m;
  return true;
}

// Blackboard implementation
Blackboard::Blackboard()
    : artifactStore_(std::make_shared<ArtifactStore>()) {}

Blackboard::Blackboard(Blackboard&& other) noexcept {
  std::lock_guard<std::mutex> lock(other.mutex_);
  state_ = std::move(other.state_);
  tasks_ = std::move(other.tasks_);
  taskOrder_ = std::move(other.taskOrder_);
  messages_ = std::move(other.messages_);
  artifactStore_ = std::move(other.artifactStore_);
  stateObservers_ = std::move(other.stateObservers_);
  taskObservers_ = std::move(other.taskObservers_);
  messageObservers_ = std::move(other.messageObservers_);
}

Blackboard& Blackboard::operator=(Blackboard&& other) noexcept {
  if (this != &other) {
    std::scoped_lock lock(mutex_, other.mutex_);
    state_ = std::move(other.state_);
    tasks_ = std::move(other.tasks_);
    taskOrder_ = std::move(other.taskOrder_);
    messages_ = std::move(other.messages_);
    artifactStore_ = std::move(other.artifactStore_);
    stateObservers_ = std::move(other.stateObservers_);
    taskObservers_ = std::move(other.taskObservers_);
    messageObservers_ = std::move(other.messageObservers_);
  }
  return *this;
}

void Blackboard::setState(const std::string& key, api::json::Value value) {
  std::vector<StateObserver> observers;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    state_[key] = value;
    observers = stateObservers_;
  }
  for (const auto& obs : observers) {
    obs(key, value);
  }
}

api::json::Value Blackboard::getState(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = state_.find(key);
  if (it != state_.end()) return it->second;
  return api::json::Value();
}

bool Blackboard::hasState(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return state_.find(key) != state_.end();
}

std::vector<std::string> Blackboard::listKeys() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> keys;
  keys.reserve(state_.size());
  for (const auto& [k, _] : state_) keys.push_back(k);
  return keys;
}

bool Blackboard::removeState(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  return state_.erase(key) > 0;
}

std::string Blackboard::addTask(Task task) {
  if (task.id.empty()) {
    task.id = "task_" + std::to_string(currentEpochMs()) + "_" + std::to_string(tasks_.size() + 1);
  }
  if (task.createdTimestamp == 0) task.createdTimestamp = currentEpochMs();
  task.updatedTimestamp = currentEpochMs();

  std::vector<TaskObserver> observers;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_[task.id] = task;
    taskOrder_.push_back(task.id);
    observers = taskObservers_;
  }
  for (const auto& obs : observers) {
    obs(task);
  }
  return task.id;
}

bool Blackboard::updateTaskStatus(const std::string& taskId, TaskStatus status, const std::string& result) {
  Task updatedTask;
  std::vector<TaskObserver> observers;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tasks_.find(taskId);
    if (it == tasks_.end()) return false;
    it->second.status = status;
    if (!result.empty()) it->second.result = result;
    it->second.updatedTimestamp = currentEpochMs();
    updatedTask = it->second;
    observers = taskObservers_;
  }
  for (const auto& obs : observers) {
    obs(updatedTask);
  }
  return true;
}

bool Blackboard::assignTask(const std::string& taskId, const std::string& agentName) {
  Task updatedTask;
  std::vector<TaskObserver> observers;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tasks_.find(taskId);
    if (it == tasks_.end()) return false;
    it->second.assignedAgent = agentName;
    it->second.updatedTimestamp = currentEpochMs();
    updatedTask = it->second;
    observers = taskObservers_;
  }
  for (const auto& obs : observers) {
    obs(updatedTask);
  }
  return true;
}

std::optional<Task> Blackboard::getTask(const std::string& taskId) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = tasks_.find(taskId);
  if (it != tasks_.end()) return it->second;
  return std::nullopt;
}

std::vector<Task> Blackboard::listTasks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Task> res;
  res.reserve(taskOrder_.size());
  for (const auto& id : taskOrder_) {
    auto it = tasks_.find(id);
    if (it != tasks_.end()) res.push_back(it->second);
  }
  return res;
}

std::vector<Task> Blackboard::readyTasks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<Task> res;
  for (const auto& id : taskOrder_) {
    auto it = tasks_.find(id);
    if (it == tasks_.end()) continue;
    if (it->second.status != TaskStatus::Pending) continue;

    bool ready = true;
    for (const auto& depId : it->second.dependencies) {
      auto depIt = tasks_.find(depId);
      if (depIt == tasks_.end() || depIt->second.status != TaskStatus::Done) {
        ready = false;
        break;
      }
    }
    if (ready) {
      res.push_back(it->second);
    }
  }
  return res;
}

bool Blackboard::allTasksCompleted() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (tasks_.empty()) return true;
  for (const auto& [_, task] : tasks_) {
    if (task.status != TaskStatus::Done && task.status != TaskStatus::Failed) {
      return false;
    }
  }
  return true;
}

void Blackboard::postMessage(const std::string& sender, const std::string& content,
                            const std::string& type, const std::string& recipient,
                            api::json::Value metadata) {
  BlackboardMessage msg;
  msg.id = "msg_" + std::to_string(currentEpochMs()) + "_" + std::to_string(messages_.size() + 1);
  msg.sender = sender;
  msg.content = content;
  msg.type = type;
  msg.recipient = recipient;
  msg.timestamp = currentEpochMs();
  msg.metadata = std::move(metadata);

  std::vector<MessageObserver> observers;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    messages_.push_back(msg);
    observers = messageObservers_;
  }
  for (const auto& obs : observers) {
    obs(msg);
  }
}

std::vector<BlackboardMessage> Blackboard::messages() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return messages_;
}

std::vector<BlackboardMessage> Blackboard::recentMessages(std::size_t count) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (messages_.size() <= count) return messages_;
  return std::vector<BlackboardMessage>(messages_.end() - count, messages_.end());
}

void Blackboard::storeArtifact(Artifact artifact) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!artifactStore_) artifactStore_ = std::make_shared<ArtifactStore>();
  artifactStore_->createArtifact(std::move(artifact));
}

std::optional<Artifact> Blackboard::getArtifact(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!artifactStore_) return std::nullopt;
  return artifactStore_->getArtifact(name);
}

std::vector<Artifact> Blackboard::listArtifacts() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!artifactStore_) return {};
  return artifactStore_->listArtifacts();
}

void Blackboard::subscribeState(StateObserver obs) {
  std::lock_guard<std::mutex> lock(mutex_);
  stateObservers_.push_back(std::move(obs));
}

void Blackboard::subscribeTask(TaskObserver obs) {
  std::lock_guard<std::mutex> lock(mutex_);
  taskObservers_.push_back(std::move(obs));
}

void Blackboard::subscribeMessage(MessageObserver obs) {
  std::lock_guard<std::mutex> lock(mutex_);
  messageObservers_.push_back(std::move(obs));
}

api::json::Value Blackboard::toJson() const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto root = api::json::Value::object();

  auto st = api::json::Value::object();
  for (const auto& [k, v] : state_) st[k] = v;
  root["state"] = std::move(st);

  auto ts = api::json::Value::array();
  for (const auto& id : taskOrder_) {
    auto it = tasks_.find(id);
    if (it != tasks_.end()) ts.push(it->second.toJson());
  }
  root["tasks"] = std::move(ts);

  auto ms = api::json::Value::array();
  for (const auto& m : messages_) ms.push(m.toJson());
  root["messages"] = std::move(ms);

  if (artifactStore_) {
    root["artifacts"] = artifactStore_->toJson();
  } else {
    root["artifacts"] = api::json::Value::array();
  }

  return root;
}

bool Blackboard::fromJson(const api::json::Value& v, std::string& error) {
  if (!v.isObject()) {
    error = "Blackboard snapshot must be a JSON object";
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  clear();

  if (const auto* st = v.get("state"); st && st->isObject()) {
    for (const auto& [k, val] : st->members()) {
      state_[k] = val;
    }
  }

  if (const auto* ts = v.get("tasks"); ts && ts->isArray()) {
    for (const auto& item : ts->items()) {
      Task t;
      std::string err;
      if (Task::fromJson(item, t, err)) {
        tasks_[t.id] = t;
        taskOrder_.push_back(t.id);
      }
    }
  }

  if (const auto* ms = v.get("messages"); ms && ms->isArray()) {
    for (const auto& item : ms->items()) {
      BlackboardMessage m;
      std::string err;
      if (BlackboardMessage::fromJson(item, m, err)) {
        messages_.push_back(std::move(m));
      }
    }
  }

  if (const auto* as = v.get("artifacts"); as && as->isArray()) {
    if (!artifactStore_) artifactStore_ = std::make_shared<ArtifactStore>();
    artifactStore_->fromJson(*as, error);
  }

  return true;
}

void Blackboard::clear() {
  state_.clear();
  tasks_.clear();
  taskOrder_.clear();
  messages_.clear();
  if (artifactStore_) artifactStore_->clear();
}

}  // namespace qorvix::agents
