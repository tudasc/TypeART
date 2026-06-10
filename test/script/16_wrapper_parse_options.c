// clang-format off
// RUN: %wrapper-cc -S -emit-llvm -O1 --typeart-stack=false --typeart-global=true --typeart-stack-lifetime %s -o - 2>&1 | %filecheck %s --check-prefix=ON
// RUN: TYPEART_WRAPPER=OFF %wrapper-cc -S -emit-llvm -O1 --typeart-stack=false --typeart-global=true --typeart-stack-lifetime %s -o - 2>&1 | %filecheck %s --check-prefix=OFF
// clang-format on

// REQUIRES: !legacywrapper

#include <stddef.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  volatile int local[4] = {0, 0, 0, 0};
  int* p                = (int*)malloc((size_t)argc * sizeof(int));
  local[0]              = argc;
  free(p);
  return 0;
}

// ON: __typeart_alloc
// ON-NOT: __typeart_alloc_stack
// OFF-NOT: __typeart_alloc
