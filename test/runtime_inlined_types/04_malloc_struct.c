// clang-format off
// RUN: export TYPEART_TYPE_SERIALIZATION=inline
// RUN: %wrapper-cc -O1 %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

// REQUIRES: llvm-18 || llvm-19
// clang-format on

#include <stdlib.h>

struct DataNested {
  struct DataNested* nested_pointer;
};

struct DataHolder {
  double a;
  float b;
  int c;
  struct DataNested nested;
};

int main(void) {
  struct DataHolder* p = (struct DataHolder*)malloc(2 * sizeof(struct DataHolder));
  free(p);
  return 0;
}

// CHECK: Allocation type detail (heap, stack, global)
// CHECK-NEXT: 256 :   1 ,    0 ,    0 , DataHolder
