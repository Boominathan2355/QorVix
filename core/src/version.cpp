#include "qorvix/version.hpp"

#include <string>

namespace qorvix {

std::string_view startupBanner() {
  static const std::string banner =
      "qorvix AI core v" + std::string(kVersionString) +
      " — phase 0 scaffold, no inference runtime yet";
  return banner;
}

}  // namespace qorvix
