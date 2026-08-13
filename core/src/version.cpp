#include "qorvix/version.hpp"

#include <string>

namespace qorvix {

std::string_view startupBanner() {
  static const std::string banner =
      "qorvix AI core v" + std::string(kVersionString) +
      " — text generation (cpu/cuda/vulkan) + embeddings + RAG";
  return banner;
}

}  // namespace qorvix
