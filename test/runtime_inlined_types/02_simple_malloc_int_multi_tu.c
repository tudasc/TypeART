// clang-format off
// RUN: export TYPEART_INSTRUMENTATION=1 
// RUN: %wrapper-cc -c -O1 %s -DTYPEART_TU_ONE -o %s_1.o
// RUN: %wrapper-cc -c -O1 %s -o %s.o
// RUN: %wrapper-cc -O1 %s.o %s_1.o -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

// REQUIRES: llvm-18 || llvm-19
// clang-format on

#include <stdlib.h>
#ifdef TYPEART_TU_ONE
void allocate() {
  int* p = (int*)malloc(33 * sizeof(int));
  free(p);
}
#else
void allocate();
int main(void) {
  allocate();
  int* p = (int*)malloc(42 * sizeof(int));
  free(p);
  return 0;
}
#endif

// CHECK: Allocation type detail (heap, stack, global)
// CHECK-NEXT: 13 :   0 ,    2 ,    0 , int