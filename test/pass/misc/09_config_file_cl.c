// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart --typeart-heap=true --typeart-stack=false --typeart-config=%S/07_typeart_config_stack.yml -S 2>&1 | %filecheck %s
// RUN: %c-to-llvm %s | %apply-typeart --typeart-heap=true --typeart-config=%S/07_typeart_config_stack.yml -S 2>&1 | %filecheck %s --check-prefix CHECK-HS
// RUN: %c-to-llvm %s | %apply-typeart --typeart-config=%S/07_typeart_config_stack.yml -S 2>&1 | %filecheck %s --check-prefix CHECK-S
// clang-format on

// Priority control with command line args vs. config file contents.

// XFAIL: *

#include <stdlib.h>
void test() {
  int x  = 0;
  int* p = (int*)malloc(42 * sizeof(int));
}

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}0
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

// CHECK-HS: TypeArtPass [Heap & Stack]
// CHECK-HS-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-HS-NEXT: Free{{[ ]*}}:{{[ ]*}}0
// CHECK-HS-NEXT: Alloca{{[ ]*}}:{{[ ]*}}1

// CHECK-S: TypeArtPass [Stack]
// CHECK-S-NEXT: Malloc{{[ ]*}}:{{[ ]*}}0
// CHECK-S-NEXT: Free{{[ ]*}}:{{[ ]*}}0
// CHECK-S-NEXT: Alloca{{[ ]*}}:{{[ ]*}}1
