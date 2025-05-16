// RUN: %run  %s --compile_flags "-fdebug-default-version=4" 2>&1 | %filecheck %s

#include "../../lib/runtime/RuntimeInterface.h"

#include <stdio.h>
#include <stdlib.h>

struct Datastruct {
  int start;
  double middle;
  float end;
};

const void* check_addr(void* ptr) {
  const void* addr;

  if (typeart_get_return_address(ptr, &addr) != TYPEART_OK) {
    fprintf(stderr, "Error getting return address.\n");
    return NULL;
  }

  if (addr == NULL) {
    fprintf(stderr, "Error return address NULL.\n");
    return NULL;
  }

  fprintf(stderr, "Address check OK.\n");

  return addr;
}

int main(int argc, char** argv) {
  struct Datastruct d;

  const void* addr = check_addr(&d);
  if (addr == NULL) {
    return 1;
  }

  typeart_source_location location;
  if (typeart_get_source_location(addr, &location) != TYPEART_OK) {
    fprintf(stderr, "Error getting source loc\n");
    return -1;
  }

  fprintf(stderr, "Loc File: %s\n", location.file);
  fprintf(stderr, "Loc Function: %s\n", location.function);
  fprintf(stderr, "Loc Line: %i\n", location.line);

  typeart_free_source_location(&location);

  if (location.file != NULL || location.function != NULL) {
    fprintf(stderr, "Error free'ing source loc\n");
    return -1;
  }

  return 0;
}

// CHECK-NOT: Error
// CHECK: Address check OK
// CHECK: Loc File:{{.*}}46_source_location.c
// CHECK: Loc Function: main
// CHECK: Loc Line: 3{{(3|5)}}
