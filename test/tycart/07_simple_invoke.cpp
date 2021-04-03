// RUN: %scriptpath/applyAndRun-tycart.sh %s %pluginpath 2>&1 | FileCheck %s

#include "support.h"

struct S1 {
  int x;
  virtual ~S1() = default;
};

// CHECK: No assert 0.
// CHECK-NOT: No assert 0.
int main() {
  S1 s;
  make_assert(0, &s, 1, S1);
  make_assert(1, &s, 1, double);
  return 0;
}
