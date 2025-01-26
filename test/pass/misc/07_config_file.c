// RUN: %c-to-llvm %s | %apply-typeart --typeart-config=%S/07_typeart_config_stack.yml 2>&1 | %filecheck %s

// XFAIL: *

#include <stdlib.h>
void test() {
  int* p = (int*)malloc(42 * sizeof(int));
}

// CHECK: types:           {{.*}}
// CHECK-NEXT:  heap:            false
// CHECK-NEXT:  stack:           true
// CHECK-NEXT:  global:          false
// CHECK-NEXT:  stats:           {{.*}}
// CHECK-NEXT:  stack-lifetime:  false
// CHECK-NEXT:  typegen:         {{dimeta|ir}}
// CHECK-NEXT:  filter:          false
// CHECK-NEXT:  call-filter:
// CHECK-NEXT:    implementation:  std
// CHECK-NEXT:    glob:            '*MPI_*'
// CHECK-NEXT:    glob-deep:       'MPI_*'
// CHECK-NEXT:    cg-file:         'path/.../cg.file'
// CHECK-NEXT:  analysis:
// CHECK-NEXT:    filter-global:   true
// CHECK-NEXT:    filter-heap-alloca: true
// CHECK-NEXT:    filter-pointer-alloca: true
// CHECK-NEXT:    filter-non-array-alloca: false

// CHECK: TypeArtPass [Stack]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}0
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}0
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
