// clang-format off
// RUN: %c-to-llvm -Xclang -disable-lifetime-markers %s | %opt -mem2reg -S | %apply-typeart --typeart-stack --typeart-filter --typeart-filter-implementation=acg --typeart-filter-cg-file=%p/27_cg.ipcg2 -S 2>&1 | %filecheck %s
// clang-format on

#include <stdlib.h>

#ifdef TYPEART_ACG_GENERATE_TEST
extern void lib_call_a(void*);
extern void lib_call_b(void*);
void f_one(int* a, int* b) {
  lib_call_a(a);
}
void f_two(int* a, int* b) {
  lib_call_b(b);
}
#else
extern void f_one(int* a, int* b);

extern void f_two(int* a, int* b);
#endif


void foo() {
  // only a2 may reach a lib
  int a2 = 3;
  int b2 = 4;
  f_one(&a2, &b2);

  // only b3 may reach a lib
  int a3 = 5;
  int b3 = 6;
  f_two(&a3, &b3);
}

// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :  4.00
// CHECK-NEXT: Stack call filtered %  : 50.00
