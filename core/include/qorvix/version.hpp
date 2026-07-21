#pragma once

#include <string_view>

namespace qorvix {

inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 1;
inline constexpr int kVersionPatch = 0;
inline constexpr std::string_view kVersionString = "0.1.0";

// One-line startup banner identifying the build. Kept in a .cpp (rather than inline
// in the header) so it's a real translation unit for the static lib to link against.
std::string_view startupBanner();

}  // namespace qorvix
