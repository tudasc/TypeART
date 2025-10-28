// clang-format off
// RUN: export TYPEART_INSTRUMENTATION=1 
// RUN: %wrapper-cc -O1 %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

// REQUIRES: llvm-18 || llvm-19
// clang-format on

#include <stdlib.h>

struct DataHolder {
  double a;
  float b;
  int c;
};

int main(void) {
  struct DataHolder* p = (struct DataHolder*)malloc(2 * sizeof(struct DataHolder));
  free(p);
  return 0;
}

// CHECK: Allocation type detail (heap, stack, global)
// CHECK-NEXT: 256 :   1 ,    0 ,    0 , DataHolder
