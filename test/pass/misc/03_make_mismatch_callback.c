// RUN: %c-to-llvm %s | %apply-typeart -typeart-stack -S 2>&1 | %filecheck %s
// XFAIL: *

#include <stddef.h>
void __typeart_leave_scope(size_t alloca_count);

int main(void) {
  __typeart_leave_scope(0);
  return 0;
}