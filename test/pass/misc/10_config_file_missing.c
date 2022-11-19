// RUN: %c-to-llvm %s | %apply-typeart -typeart-config=%S/missing_config.yml -S 2>&1 | %filecheck %s
// XFAIL: *
// CHECK: Fatal

#include <stdlib.h>
void test() {
  int* p = (int*)malloc(42 * sizeof(int));
}