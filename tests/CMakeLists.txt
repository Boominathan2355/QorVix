find_package(Catch2 3 CONFIG QUIET)

if(Catch2_FOUND)
  add_executable(qorvix_tests core_smoke_test.cpp)
  target_link_libraries(qorvix_tests PRIVATE qorvix_core Catch2::Catch2WithMain)

  include(CTest)
  include(Catch)
  catch_discover_tests(qorvix_tests)
else()
  message(STATUS "Catch2 not found (vcpkg not bootstrapped yet) — skipping qorvix_tests. See docs/BUILDING.md.")
endif()
