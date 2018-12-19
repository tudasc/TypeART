// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-typeart-alloca" %rtpath 2>&1 | FileCheck %s

#include <stdlib.h>

typedef int int2 __attribute__((ext_vector_type(2)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef double double3 __attribute__((ext_vector_type(3)));


void alloc_vector_arrays() {
  int2 i[2];
  float2 f[2];
  double3 d[2];
}

void alloc_vector_vlas(int n) {
    int2 i[n];
    float2 f[n];
    double3 d[n];
}

void malloc_vector(int n) {
    int2* i = malloc(n * sizeof(int2));
    float2* f = malloc(n * sizeof(float2));
    double3* d = malloc(n * sizeof(double3));
    printf("malloc size %d\n", sizeof(double3));
    free(i);
    free(f);
    free(d);
}

// TODO: Alignment of vector types

int main(int argc, char** argv) {
  // CHECK: [Trace] TypeART Runtime Trace

  // CHECK: [Trace] Alloc 0x{{.*}} int32 4 4
  // CHECK: [Trace] Alloc 0x{{.*}} float 4 4
  // CHECK: [Trace] Alloc 0x{{.*}} double 8 6
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  alloc_vector_arrays();

  // CHECK: [Trace] Alloc 0x{{.*}} int32 4 8
  // CHECK: [Trace] Alloc 0x{{.*}} float 4 8
  // CHECK: [Trace] Alloc 0x{{.*}} double 8 12
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  alloc_vector_vlas(4);


  // CHECK: [Trace] Alloc 0x{{.*}} int32 4 16
  // CHECK: [Trace] Alloc 0x{{.*}} float 4 16
  // CHECK: [Trace] Alloc 0x{{.*}} double 8 24
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  // CHECK: [Trace] Free 0x{{.*}}
  malloc_vector(8);

  return 0;
}
