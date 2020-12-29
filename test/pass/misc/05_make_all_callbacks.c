// RUN: %c-to-llvm %s | %apply-typeart -typeart-alloca -S 2>&1 | FileCheck %s

#include "../../../lib/runtime/CallbackInterface.h"

int main(void) {
  int count     = 0;
  int type_id   = 10;
  size_t extent = 0;
  void* addr    = NULL;
  __typeart_alloc(addr, type_id, extent);
  __typeart_alloc_global(addr, type_id, extent);
  __typeart_alloc_stack(addr, type_id, extent);
  __typeart_free(addr);
  __typeart_leave_scope(count);
  return 0;
}

// CHECK:      TypeArtPass [Heap & Stack]