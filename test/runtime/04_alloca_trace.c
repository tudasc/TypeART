// RUN: %run %s -typeart-filter-pointer-alloca=false 2>&1 | %filecheck %s

#include <stdlib.h>

int main(int argc, char** argv) {
  const int n = 42;
  // CHECK: [Trace] TypeART Runtime Trace

  // CHECK: [Trace] Alloc 0x{{.*}} int8 1 42
  char a[n];

  // CHECK: [Trace] Alloc 0x{{.*}} int16 2 42
  short b[n];

  // CHECK: [Trace] Alloc 0x{{.*}} int32 4 42
  int c[n];

  // CHECK: [Trace] Alloc 0x{{.*}} int64 8 42
  long d[n];

  // CHECK: [Trace] Alloc 0x{{.*}} float 4 42
  float e[n];

  // CHECK: [Trace] Alloc 0x{{.*}} double 8 42
  double f[n];

  // CHECK: [Trace] Alloc 0x{{.*}} pointer 8 42
  int* g[n];

  // CHECK: [Trace] Alloc 0x{{.*}} int32 4 1764
  int h[n][n];

  // CHECK: [Trace] Alloc 0x{{.*}} int32 4 74088
  int i[n][n][n];

  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}

  return 0;
}
