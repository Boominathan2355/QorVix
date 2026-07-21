#include <catch2/catch_test_macros.hpp>

#include "qorvix/version.hpp"

TEST_CASE("version string matches semver fields", "[core]") {
  const std::string expected = std::to_string(qorvix::kVersionMajor) + "." +
                                std::to_string(qorvix::kVersionMinor) + "." +
                                std::to_string(qorvix::kVersionPatch);
  REQUIRE(qorvix::kVersionString == expected);
}

TEST_CASE("startup banner mentions the version", "[core]") {
  REQUIRE(qorvix::startupBanner().find(qorvix::kVersionString) != std::string_view::npos);
}
