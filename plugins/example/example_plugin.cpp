#include <string>

#include "qorvix/plugin.hpp"

// Minimal reference plugin used to exercise the loader end-to-end (Phase 1). It performs no real
// inference — it exists to prove hot-load/unload and the C-ABI boundary work. Real architecture
// plugins (llama, qwen, ...) land alongside the text runtime in Phase 5.
namespace {

class ExamplePlugin final : public qorvix::IPlugin {
 public:
  bool load() override {
    loaded_ = true;
    return true;
  }

  bool unload() override {
    loaded_ = false;
    return true;
  }

  bool infer() override { return loaded_; }

  std::string architecture() override { return "example"; }

 private:
  bool loaded_ = false;
};

}  // namespace

QORVIX_REGISTER_PLUGIN(ExamplePlugin)
