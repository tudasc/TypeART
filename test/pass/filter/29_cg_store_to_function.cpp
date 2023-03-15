// clang-format off
// RUN: %cpp-to-llvm -Xclang -disable-lifetime-markers %s | %opt -mem2reg -S | %apply-typeart --typeart-stack --typeart-filter --typeart-filter-implementation=acg --typeart-filter-cg-file=%p/29_cg.ipcg2 -S 2>&1 | %filecheck %s
// clang-format on

#include <vector>

int& ref(int&) noexcept;

void foo(int r) {
  int i = r * 789;
  ref(r) += i;
}

// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                      :   1.00
// CHECK-NEXT: Stack call filtered %       : 100.00
