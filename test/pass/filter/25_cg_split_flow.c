// clang-format off
// RUN: %c-to-llvm %s | %opt -mem2reg -S | %apply-typeart --typeart-stack --typeart-filter --typeart-filter-implementation=acg --typeart-filter-cg-file=%p/25_cg.ipcg2 -S 2>&1 | %filecheck %s
// clang-format on

#include <stdlib.h>

extern void MPI_sink(void* a);

extern void split_flow(int* x, int* y, int* z);

// split_flow is implemented like the following function:
// void split_flow(int* x, int* y, int* z) {
//     MPI_sink(z);
// }

void foo() {
  int a = 1;
  int b = 1;

  split_flow(&a, &a, &a);
  split_flow(&b, &b, &a);
}

// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :  2.00
// CHECK-NEXT: Stack call filtered %  : 50.00
