// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack=true -S 2>&1 | %filecheck %s
// REQUIRES: llvm-14
// XFAIL: *

#include <stddef.h>
void __typeart_leave_scope(size_t alloca_count);

int main(void) {
  __typeart_leave_scope(0);
  return 0;
}