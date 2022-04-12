// RUN: %run  %s --compile_flags "-std=c++17" 2>&1 | %filecheck %s

#include "../../lib/support/System.h"

#include <cstdio>

struct Datastruct {
  int start;
  double middle;
  float end;
};

int main() {
  Datastruct d;

  const auto rss = typeart::system::Process::getMaxRSS();
  printf("Max RSS: %ld\n", rss);

  return 0;
}

// CHECK-NOT: Error
// CHECK: Max RSS: {{[1-9][0-9]+}}
