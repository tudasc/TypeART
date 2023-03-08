// clang-format off
// RUN: %c-to-llvm -Xclang -disable-lifetime-markers %s | %opt -mem2reg -S | %apply-typeart --typeart-stack --typeart-filter --typeart-filter-implementation=acg --typeart-filter-cg-file=%p/27_cg.ipcg2 -S 2>&1 | %filecheck %s
// clang-format on

#include <stdlib.h>

#ifdef TYPEART_ACG_GENERATE_TEST
extern void lib_call_a(void*);
extern void lib_call_b(void*);
extern void __typeart_alloc(const void*, int, size_t);
extern void pow(void*);
void f_one(int* a, int* b) {
  lib_call_a(a);
}
void f_two(int* a, int* b) {
  lib_call_b(b);
}
void f_three(int* a, int* b) {
  __typeart_alloc(a, 0, 0);
  __typeart_alloc(b, 0, 0);
}
#else
extern void f_one(int* a, int* b);

extern void f_two(int* a, int* b);

extern void f_three(int* a, int* b);
#endif

void f_four(int* a, int* b) {
}

static int a5 = 9;
static int b5 = 10;

void foo() {
  // only a1 may reach a lib
  int a1 = 1;
  int b1 = 2;
  f_one(&a1, &b1);

  // only b2 may reach a lib
  int a2 = 3;
  int b2 = 4;
  f_two(&a2, &b2);

  // a3 and b3 only reach irrelevant libs
  int a3 = 5;
  int b3 = 6;
  f_three(&a3, &b3);

  // a4 and b4 are never used
  int a4 = 7;
  int b4 = 8;
  f_four(&a4, &b4);

  // a5 and b5 are never used
  f_four(&a5, &b5);
}

// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :  8.00
// CHECK-NEXT: Stack call filtered %  : 75.00
