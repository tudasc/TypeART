// clang-format off
// RUN: %run %s --typeart-analysis-filter-pointer-alloca=false 2>&1 | %filecheck %s
// RUN: %run %s "--typeart-filter" 2>&1 | %filecheck %s --check-prefix CHECK-FILTER
// clang-format on

#include <stdlib.h>

void call() {
  const int n = 10;
  double a[n];
  double b[n];
}

int main(int argc, char** argv) {
  const int n = 42;
  // CHECK: [Trace] TypeART Runtime Trace
  // CHECK-FILTER-NOT: [Trace] TypeART Runtime Trace
  // CHECK-FILTER-NOT [Trace]

  // CHECK: [Trace] Alloc 0x{{.*}} {{(int8_t|char)}} 1 42
  char a[n];

  // CHECK: [Trace] Alloc 0x{{.*}} short 2 42
  short b[n];

  // CHECK: [Trace] Alloc 0x{{.*}} int 4 42
  int c[n];

  // CHECK: [Trace] Alloc 0x{{.*}} long int 8 42
  long d[n];

  // CHECK: [Trace] Alloc 0x{{.*}} float 4 42
  float e[n];

  // CHECK: [Trace] Alloc 0x{{.*}} double 8 42
  double f[n];

  // CHECK: [Trace] Alloc 0x{{.*}} ptr 8 42
  int* g[n];

  // CHECK: [Trace] Alloc 0x{{.*}} double 8 10
  // CHECK: [Trace] Alloc 0x{{.*}} double 8 10
  call();
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
