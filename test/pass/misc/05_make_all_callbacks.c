// RUN: %c-to-llvm -Wimplicit-function-declaration %s -I%runtime_path -I%base_path/lib/runtime | %apply-typeart
// --typeart-stack=true -S 2>&1 | \ RUN: %filecheck %s

#include "CallbackInterface.h"

int main(void) {
  int count            = 0;
  int type_id          = 10;
  size_t extent        = 0;
  void* addr           = NULL;
  const void* info_ptr = NULL;
  const void* type_ptr = NULL;

  __typeart_alloc(addr, type_id, extent);
  __typeart_alloc_global(addr, type_id, extent);
  __typeart_alloc_stack(addr, type_id, extent);
  __typeart_free(addr);
  __typeart_leave_scope(count);

  // called (only) from OpenMP context:
  __typeart_alloc_omp(addr, type_id, extent);
  __typeart_alloc_stack_omp(addr, type_id, extent);
  __typeart_free_omp(addr);
  __typeart_leave_scope_omp(count);

  __typeart_alloc_mty(addr, info_ptr, count);
  __typeart_alloc_global_mty(addr, info_ptr, count);
  __typeart_alloc_stack_mty(addr, info_ptr, count);
  __typeart_register_type(type_ptr);
  __typeart_alloc_global_mty_omp(addr, info_ptr, count);
  __typeart_alloc_stack_mty_omp(addr, info_ptr, count);

  return 0;
}

// CHECK-NOT: error
// CHECK:      TypeArtPass [Heap & Stack]