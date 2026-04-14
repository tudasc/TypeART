// clang-format off
// RUN: export TYPEART_TYPE_SERIALIZATION=inline
// RUN: %wrapper-cc -fPIC -shared -O1 %s -DTYPEART_TU_ONE -o %s_1.so
// RUN: %wrapper-cc -O1 %s %s_1.so -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

// REQUIRES: !llvm-14
// clang-format on

#include <stdlib.h>

#ifdef TYPEART_TU_ONE
struct SharedStruct {
  int x;
  float y;
};
void* allocate() {
  struct SharedStruct* p = (struct SharedStruct*)malloc(33 * sizeof(struct SharedStruct));
  return (void*)p;
}
#else
struct BinaryStruct {
  double x;
  float y;
};
void* allocate();
int main(void) {
  void* lib_type         = allocate();
  struct BinaryStruct* p = (struct BinaryStruct*)malloc(42 * sizeof(struct BinaryStruct));
  free(p);
  free(lib_type);
  return 0;
}
#endif

// CHECK: Allocation type detail (heap, stack, global)
// CHECK-NEXT: 256 :   1 ,    0 ,    0 , SharedStruct
// CHECK-NEXT: 257 :   1 ,    0 ,    0 , BinaryStruct
