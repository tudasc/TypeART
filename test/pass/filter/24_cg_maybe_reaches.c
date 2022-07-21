// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-stack -typeart-call-filter -typeart-call-filter-impl=cg -typeart-call-filter-cg-file=%p/24_cg.ipcg -S 2>&1 | %filecheck %s
// RUN: %c-to-llvm %s | %apply-typeart -typeart-stack -typeart-call-filter -S 2>&1 | %filecheck %s --check-prefix=CHECK-default
// clang-format on

extern int* bar(int* ptr);  // maybe reaches MPI, see 24_cg.ipcg
extern void aar(int* ptr);

void foo() {
  int a;             // won't be filtered with CG (a to bar follows p to aar), or std
  int* p = bar(&a);  // p* is "pointer alloca-filtered"
  aar(p);
}

// CG:
// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :  2.00
// CHECK-NEXT: Stack call filtered %  :  0.00

// Standard filter
// CHECK-default: > Stack Memory
// CHECK-default-NEXT: Alloca                 :  2.00
// CHECK-default-NEXT: Stack call filtered %  :  0.00
