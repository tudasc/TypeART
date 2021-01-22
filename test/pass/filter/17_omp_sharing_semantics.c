// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | opt -O2 -S | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s
// REQUIRES: openmp
// clang-format on

#include "omp.h"

extern void MPI_Send(void*, int);

void foo(int count) {
  int d = 3;
  int e = 4;
#pragma omp parallel for schedule(dynamic, 1) firstprivate(d) shared(e)
  for (int i = 0; i < count; ++i) {
    // Analysis should not filter d, but not e...
    MPI_Send((void*)&d, e);
  }
}

void bar(int count) {
  int d = 3;
  int e = 4;
#pragma omp parallel for schedule(dynamic, 1) lastprivate(d) shared(e)
  for (int i = 0; i < count; ++i) {
    // Analysis should not filter d, but not e...
    MPI_Send((void*)&d, e);
  }
}

void foo_bar(int count) {
  int d = 3;
  int e = 4;
#pragma omp parallel for schedule(dynamic, 1) private(d, e)
  for (int i = 0; i < count; ++i) {
    MPI_Send((void*)&d, e);
  }
}

// CHECK: TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc :   0
// CHECK-NEXT: Free   :   0
// CHECK-NEXT: Alloca :   3
// CHECK-NEXT: Global :   0
