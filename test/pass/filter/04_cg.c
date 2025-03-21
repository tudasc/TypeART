// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack=true --typeart-filter=true --typeart-filter-implementation=cg --typeart-filter-cg-file=%p/04_cg.ipcg -S 2>&1 | %filecheck %s
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack=true --typeart-filter=true -S 2>&1 | %filecheck %s --check-prefix=CHECK-default
// clang-format on

extern void bar(int* ptr);  // reaches MPI, see 04_cg.ipcg
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

// Standard filter
// CHECK-default: > Stack Memory
// CHECK-default-NEXT: Alloca                 :  2.00
// CHECK-default-NEXT: Stack call filtered %  :  0.00