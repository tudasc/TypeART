// RUN: %run %s --manual 2>&1 | %filecheck %s

#include "../../lib/runtime/CallbackInterface.h"

#include <stdlib.h>

int main(int argc, char** argv) {
  __typeart_alloc((const void*)0, 23, 1);
  __typeart_alloc((const void*)1, 22, 0);
  __typeart_alloc((const void*)0, 23, 0);
  __typeart_alloc((const void*)2, 24, 1);  // OK

  return 0;
}
// TODO disable Trace logs for early return?

// CHECK: [Error]{{.*}}:Nullptr allocation 0x0 23 {{.*}} 8 1
// CHECK: [Warning]{{.*}}:Zero-size allocation 0x1 22 {{.*}} 4 0
// CHECK: [Error]{{.*}}:Zero-size and nullptr allocation 0x0 23 {{.*}} 8 0
// CHECK: [Trace] Alloc 0x2 24 {{.*}} 16 1