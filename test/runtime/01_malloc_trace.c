// RUN: %run %s 2>&1 | %filecheck %s

#include <stdlib.h>

int main(int argc, char** argv) {
  const int n = 42;
  // CHECK: [Trace] TypeART Runtime Trace

  // CHECK: [Trace] Alloc 0x{{.*}} {{(int8_t|char)}} 1 42
  char* a = malloc(n * sizeof(char));
  // CHECK: [Trace] Free 0x{{.*}}
  free(a);

  // CHECK: [Trace] Alloc 0x{{.*}} short 2 42
  short* b = malloc(n * sizeof(short));
  // CHECK: [Trace] Free 0x{{.*}}
  free(b);

  // CHECK: [Trace] Alloc 0x{{.*}} int 4 42
  int* c = malloc(n * sizeof(int));
  // CHECK: [Trace] Free 0x{{.*}}
  free(c);

  // CHECK: [Trace] Alloc 0x{{.*}} long int 8 42
  long* d = malloc(n * sizeof(long));
  // CHECK: [Trace] Free 0x{{.*}}
  free(d);

  // CHECK: [Trace] Alloc 0x{{.*}} float 4 42
  float* e = malloc(n * sizeof(float));
  // CHECK: Free 0x{{.*}}
  free(e);

  // CHECK: [Trace] Alloc 0x{{.*}} double 8 42
  double* f = malloc(n * sizeof(double));
  // CHECK: [Trace] Free 0x{{.*}}
  free(f);

  return 0;
}
