// RUN: %scriptpath/applyAndRun-tycart.sh %s %pluginpath 2>&1 | FileCheck %s

#include "support.h"

struct S1 {
  int x;
};

// CHECK: No assert 0.
// CHECK: No assert 1.
// CHECK: No assert 2.
// CHECK-NOT: No assert 3.
int main() {
  struct S1 s;
  int i;
  int* x = (int*)malloc(sizeof(int) * 10);
  make_assert(0, &s, 1, struct S1);
  make_assert(1, &i, 1, int);
  make_assert(2, x, 10, int);
  make_assert(3, x, 1, int);
  return 0;
}
