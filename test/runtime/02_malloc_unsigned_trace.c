// RUN: %run %s 2>&1 | %filecheck %s
// XFAIL: *
#include <stdlib.h>

int main(int argc, char** argv) {
  const int n = 42;
  // CHECK: [Trace] TypeART Runtime Trace

  // CHECK: [Trace] Alloc 0x{{.*}} uint32 1 42
  unsigned char* a = (unsigned char*)malloc(n * sizeof(unsigned char));
  // CHECK: [Trace] Free 0x{{.*}}
  free(a);

  // CHECK: [Trace] Alloc 0x{{.*}} uint16 2 42
  unsigned short* b = (unsigned short*)malloc(n * sizeof(unsigned short));
  // CHECK: [Trace] Free 0x{{.*}}
  free(b);

  // CHECK: [Trace] Alloc 0x{{.*}} uint32 4 42
  unsigned int* c = (unsigned int*)malloc(n * sizeof(unsigned int));
  // CHECK: [Trace] Free 0x{{.*}}
  free(c);

  // CHECK: [Trace] Alloc 0x{{.*}} uint64 8 42
  unsigned long* d = (unsigned long*)malloc(n * sizeof(unsigned long));
  // CHECK: [Trace] Free 0x{{.*}}
  free(d);

  return 0;
}
