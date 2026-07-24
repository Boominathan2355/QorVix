#include <catch2/catch_test_macros.hpp>

#include <set>

#include "qorvix/ports.hpp"

using namespace qorvix;

// These pin the published port allocation. kRuntime in particular is SHIPPED — changing it breaks
// every deployed client config, so it is asserted by literal value on purpose: the test should
// fail loudly and force a deliberate decision rather than let a renumber slip through.

TEST_CASE("reserved port allocation is exactly as published", "[ports]") {
  REQUIRE(ports::kRuntime == 2005);
  REQUIRE(ports::kGateway == 2006);
  REQUIRE(ports::kDashboard == 2007);
  REQUIRE(ports::kAdminApi == 2008);
  REQUIRE(ports::kMetrics == 2009);
  REQUIRE(ports::kGrpc == 2010);
}

TEST_CASE("every reserved port is distinct and inside the declared range", "[ports]") {
  const int all[] = {ports::kRuntime,  ports::kGateway, ports::kDashboard,
                     ports::kAdminApi, ports::kMetrics, ports::kGrpc};
  const std::set<int> unique(std::begin(all), std::end(all));
  REQUIRE(unique.size() == std::size(all));  // no two services share a default

  for (int p : all) {
    REQUIRE(ports::inReservedRange(p));
    REQUIRE_FALSE(ports::serviceName(p).empty());
  }
  // The range is exactly the services, no slack that could be handed out twice.
  REQUIRE(ports::kRangeFirst == ports::kRuntime);
  REQUIRE(ports::kRangeLast == ports::kGrpc);
  REQUIRE(static_cast<int>(unique.size()) == ports::kRangeLast - ports::kRangeFirst + 1);
}

TEST_CASE("ports outside the block are not claimed", "[ports]") {
  REQUIRE_FALSE(ports::inReservedRange(2004));
  REQUIRE_FALSE(ports::inReservedRange(2011));
  REQUIRE_FALSE(ports::inReservedRange(8080));  // the old default is no longer ours
  REQUIRE(ports::serviceName(8080).empty());
  REQUIRE(ports::serviceName(0).empty());
}

TEST_CASE("port helpers are usable in constant expressions", "[ports]") {
  // constexpr so a service can static_assert its own binding at compile time.
  static_assert(ports::inReservedRange(ports::kMetrics));
  static_assert(!ports::inReservedRange(80));
  static_assert(ports::serviceName(ports::kDashboard) == "Qorvix Dashboard");
  SUCCEED();
}
