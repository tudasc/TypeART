// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-alloca -call-filter -filter-impl=deprecated::default -call-filter-deep=true -S 2>&1 | FileCheck %s --check-prefix=CHECK-default
// RUN: %c-to-llvm %s | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s --check-prefix=CHECK-exp-default
// RUN: %c-to-llvm %s | %apply-typeart -typeart-alloca -call-filter -filter-impl=cg -cg-file=%p/05_cg.ipcg -S 2>&1 | FileCheck %s --check-prefix=CHECK-exp-cg
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
  // Analysis should filter d, but not e...
  MPI_Send((void*)d, e);
}

// Standard filter
// CHECK-default: > Stack Memory
// CHECK-default-NEXT: Alloca                 :  5.00
// CHECK-default-NEXT: Stack call filtered %  :  60.00

// Standard experimental filter
// CHECK-exp-default: > Stack Memory
// CHECK-exp-default-NEXT: Alloca                 :  5.00
// CHECK-exp-default-NEXT: Stack call filtered %  :  80.00

// CG experimental filter
// CHECK-exp-cg: > Stack Memory
// CHECK-exp-cg-NEXT: Alloca                 :  5.00
// CHECK-exp-cg-NEXT: Stack call filtered %  :  80.00