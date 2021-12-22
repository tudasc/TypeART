// RUN: %run %s 2>&1 | %filecheck %s
// XFAIL: *
#include <stdlib.h>

int main(int argc, char** argv) {
  const int n = 42;
  // CHECK: [Trace] TypeART Runtime Trace

  // CHECK: [Trace] Alloc 0x{{.*}} uint8 1 42
  unsigned char a[n];

  // CHECK: [Trace] Alloc 0x{{.*}} uint16 2 42
  unsigned short b[n];

  // CHECK: [Trace] Alloc 0x{{.*}} uint32 4 42
  unsigned int c[n];

  // CHECK: [Trace] Alloc 0x{{.*}} uint64 8 42
  unsigned long d[n];

  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}

  return 0;
}
