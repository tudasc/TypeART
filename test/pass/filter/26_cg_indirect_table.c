// clang-format off
// RUN: %c-to-llvm -Xclang -disable-lifetime-markers %s | %opt -mem2reg -S | %apply-typeart --typeart-stack --typeart-filter --typeart-filter-implementation=acg --typeart-filter-cg-file=%p/26_cg.ipcg2 -S 2>&1 | %filecheck %s
// clang-format on

#include <stdlib.h>

#ifdef TYPEART_ACG_GENERATE_TEST
extern void MPI_sink(void* a);
void f_one(int* a, int* b) {
  f_three(a, a);
}
void f_two(int* a, int* b) {
  f_three(b, b);
}
void f_three(int* a, int* b) {
  MPI_sink(a);
}
void f_four(int* a, int* b) {
  MPI_sink(b);
}
#else
/// calls f_three(a, a);
extern void f_one(int* a, int* b);

/// calls f_three(b, b);
extern void f_two(int* a, int* b);

/// reaches a mpi target with 1st (a) argument
extern void f_three(int* a, int* b);

/// reaches a mpi target with 2nd (b) argument
extern void f_four(int* a, int* b);
#endif

typedef void function_entry(int*, int*);
extern int table_index;

function_entry* table[4] = {f_one, f_two, f_three, f_four};

void foo() {
  // both (a1 and b1) can reach a mpi function
  int a1 = 1;
  int b1 = 2;
  table[table_index](&a1, &b1);

  // only a2 can reach a mpi function
  int a2 = 3;
  int b2 = 4;
  f_one(&a2, &b2);

  // only b3 can reach a mpi function
  int a3 = 5;
  int b3 = 6;
  f_two(&a3, &b3);
}

// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :  6.00
// CHECK-NEXT: Stack call filtered %  : 33.33
