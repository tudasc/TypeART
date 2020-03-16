#include "support.h"

struct S1 {
  int x[10];
};

struct S2 {
  struct S1 s1;
};

// CHECK: No assert 0.
// CHECK: No assert 1.
// CHECK: No assert 2.
// CHECK-NOT: No assert 3
int main() {
  struct S1 s;
  struct S2 s2;
  make_assert(0, &s, 1, struct S1);
  make_assert(1, &s2, 1, struct S2);
  make_assert(2, &s2.s1.x, 10, int);
  make_assert(3, &s2.s1.x, 1, int);
  return 0;
}
