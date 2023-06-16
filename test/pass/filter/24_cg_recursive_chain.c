// clang-format off
// RUN: %c-to-llvm %s | %opt -mem2reg -S | %apply-typeart --typeart-stack --typeart-filter --typeart-filter-implementation=acg --typeart-filter-cg-file=%p/24_cg.ipcg2 -S 2>&1 | %filecheck %s
// clang-format on

#include <stdlib.h>

extern void MPI_sink(void* a);
extern int ring_index;

extern void ring(int* x, int* y, int* z, int counter);

// ring is implemented like the following function:
// void ring(int* x, int* y, int* z, int counter) {
//   if (counter == 0) {
//     MPI_sink(x);
//     return;
//   }
//
//   ring(y, z, x, counter - 1);
// }

void foo() {
  int a = 1;
  int b = 2;
  int c = 3;

  ring(&a, &b, &c, ring_index);
}

// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :  3.00
// CHECK-NEXT: Stack call filtered %  :  0.00