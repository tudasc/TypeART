#include "support.h"

struct S1 {
  int x[10];
};

struct S2 {
  struct S1 s1;
};

struct S3 {
  struct S2 s2;
};

// CHECK: No assert 0.
// CHECK: No assert 1.
// CHECK: No assert 2.
int main() {
  struct S1 s;
  struct S2 s2;
  struct S3 s3;
  make_assert(0, &s2.s1, 1, struct S1);
  make_assert(1, &s2.s1.x, 10, int);
  make_assert(2, &s3.s2.s1, 1, struct S1);
  return 0;
}
