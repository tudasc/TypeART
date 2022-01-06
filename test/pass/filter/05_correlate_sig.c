// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-stack -typeart-call-filter -S 2>&1 | %filecheck %s --check-prefix=CHECK-exp-default
// RUN: %c-to-llvm %s | %apply-typeart -typeart-stack -typeart-call-filter -typeart-call-filter-impl=cg -typeart-call-filter-cg-file=%p/05_cg.ipcg -S 2>&1 | %filecheck %s --check-prefix=CHECK-exp-cg
// clang-format on

extern void MPI_Mock(int, int, int);
extern void MPI_Send(void*, int);
void foo() {
  int a = 0;
  int b = 1;
  int c = 2;
  int d = 3;
  int e = 4;
  // no (void*), so we assume benign (with deep analysis)
  MPI_Mock(a, b, c);
  // Analysis should not filter d, but e...
  MPI_Send((void*)d, e);
}

// Standard experimental filter
// CHECK-exp-default: > Stack Memory
// CHECK-exp-default-NEXT: Alloca                 :  5.00
// CHECK-exp-default-NEXT: Stack call filtered %  :  80.00

// CG experimental filter
// CHECK-exp-cg: > Stack Memory
// CHECK-exp-cg-NEXT: Alloca                 :  5.00
// CHECK-exp-cg-NEXT: Stack call filtered %  :  80.00