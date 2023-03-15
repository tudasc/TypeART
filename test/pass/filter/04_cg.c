// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack --typeart-filter --typeart-filter-implementation=acg --typeart-filter-cg-file=%p/04_cg.ipcg2 -S 2>&1 | %filecheck %s --check-prefix=CHECK-default
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack --typeart-filter --typeart-filter-implementation=cg --typeart-filter-cg-file=%p/04_cg.ipcg -S 2>&1 | %filecheck %s
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack --typeart-filter -S 2>&1 | %filecheck %s --check-prefix=CHECK-default
// clang-format on

extern void bar(int* ptr);  // reaches MPI, see 04_cg.ipcg / 04_cg.ipcg2
extern void aar(int* ptr);  // does not reach MPI

void foo() {
  int a, b;
  bar(&a);
  aar(&b);
}
// CG:
// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :  2.00
// CHECK-NEXT: Stack call filtered %  : 50.00

// Standard filter  / ACG:
// CHECK-default: > Stack Memory
// CHECK-default-NEXT: Alloca                 :  2.00
// CHECK-default-NEXT: Stack call filtered %  :  0.00