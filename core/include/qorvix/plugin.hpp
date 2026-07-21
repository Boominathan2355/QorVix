#pragma once

#include <string>

namespace qorvix {

// Architecture plugin contract. Every model architecture (llama, qwen, gemma, ...) ships as a
// shared library exposing an implementation of this interface. Kept intentionally thin for
// Phase 1 — richer inference entry points arrive with the text runtime (Phase 5).
class IPlugin {
 public:
  virtual ~IPlugin() = default;

  virtual bool load() = 0;
  virtual bool unload() = 0;
  virtual bool infer() = 0;
  virtual std::string architecture() = 0;
};

// C-ABI factory symbols every plugin library must export (via QORVIX_REGISTER_PLUGIN). The
// registry looks these up by name after dlopen/LoadLibrary.
extern "C" {
using QorvixCreatePluginFn = IPlugin* (*)();
using QorvixDestroyPluginFn = void (*)(IPlugin*);
}

inline constexpr char kCreatePluginSymbol[] = "qorvix_create_plugin";
inline constexpr char kDestroyPluginSymbol[] = "qorvix_destroy_plugin";

}  // namespace qorvix

#if defined(_WIN32)
#define QORVIX_PLUGIN_EXPORT __declspec(dllexport)
#else
#define QORVIX_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

// Emits the two C-ABI factory functions the registry expects. Instance is allocated and freed
// on the plugin's side of the ABI boundary, so the host never news/deletes across it.
#define QORVIX_REGISTER_PLUGIN(PluginType)                                        \
  extern "C" QORVIX_PLUGIN_EXPORT qorvix::IPlugin* qorvix_create_plugin() {       \
    return new PluginType();                                                      \
  }                                                                               \
  extern "C" QORVIX_PLUGIN_EXPORT void qorvix_destroy_plugin(qorvix::IPlugin* p) { \
    delete p;                                                                     \
  }
