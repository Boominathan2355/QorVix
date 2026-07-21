#include <iostream>

#include "qorvix/version.hpp"

int main() {
  std::cout << qorvix::startupBanner() << '\n';
  return 0;
}
