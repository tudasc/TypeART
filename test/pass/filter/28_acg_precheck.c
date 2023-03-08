// clang-format off
// RUN: %c-to-llvm -Xclang -disable-lifetime-markers %s | %apply-typeart --typeart-stack --typeart-filter --typeart-filter-implementation=acg --typeart-filter-cg-file=%p/28_cg.ipcg2 -S 2>&1 | %filecheck %s
// clang-format on

#include <stdlib.h>

void sink(void *a);

// precheck: no callsites
void foo() {
  int unused;
}

// precheck: alloca is temp
void bar(int *a) {
  sink(a);
}


static int g = 0;
void baz() {
  sink(&g);
}


// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                      :   2.00
// CHECK-NEXT: Stack call filtered %       : 100.00
// CHECK-NEXT: Alloca of pointer discarded :   1.00
