// RUN: %run %s 2>&1 | %filecheck %s

#include <stdlib.h>

void f(int n) {
  char a[n];
  char b[n][n];
  char c[2][n];
}

int main(int argc, char** argv) {
  // CHECK: [Trace] TypeART Runtime Trace

  // CHECK: [Trace] Alloc 0x{{.*}} int8 1 2
  // CHECK: [Trace] Alloc 0x{{.*}} int8 1 4
  // CHECK: [Trace] Alloc 0x{{.*}} int8 1 4
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  f(2);

  // CHECK: [Trace] Alloc 0x{{.*}} int8 1 8
  // CHECK: [Trace] Alloc 0x{{.*}} int8 1 64
  // CHECK: [Trace] Alloc 0x{{.*}} int8 1 16
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  f(8);

  return 0;
}
