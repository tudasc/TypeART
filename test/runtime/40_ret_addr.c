// RUN: %run %s --manual 2>&1 | %filecheck %s

#include "../../lib/runtime/CallbackInterface.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  const void* ret_check = NULL;
  const void* addr      = 1;

  typeart_get_return_address(addr, &ret_check);
  if (ret_check != NULL) {
    fprintf(stderr, "[Error] Ret address mismatch expected NULL but have %p\n", ret_check);
  }
  return 0;
}

// CHECK-NOT: [Error]
