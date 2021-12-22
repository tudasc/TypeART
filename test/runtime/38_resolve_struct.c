// RUN: %run %s --manual 2>&1 | %filecheck %s

#include "../../lib/runtime/CallbackInterface.h"
#include "util.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

struct Datastruct {
  int start;
  double middle;
  float end[2];
};

void type_check(const void* addr) {
  int id_result      = 0;
  size_t count_check = 0;
  typeart_status status;
  status = typeart_get_type(addr, &id_result, &count_check);

  if (status != TYPEART_OK) {
    fprintf(stderr, "[Error]: Status not OK: %i for %p\n", status, addr);
  } else {
    fprintf(stderr, "Status OK: %i %zu\n", id_result, count_check);
  }
}

void type_check_containing(const void* addr) {
  size_t offset         = 0;
  const void* base_adrr = NULL;
  int id_result         = 0;
  size_t count_check    = 0;
  typeart_status status;

  status = typeart_get_containing_type(addr, &id_result, &count_check, &base_adrr, &offset);

  if (status != TYPEART_OK) {
    fprintf(stderr, "[Error]: Status not OK: %i for %p\n", status, addr);
  } else {
    fprintf(stderr, "Status OK: %i %zu %zu %p\n", id_result, count_check, offset, base_adrr);
  }
}

int main(int argc, char** argv) {
  // CHECK-NOT: [Error]

  struct Datastruct data;
  __typeart_alloc((const void*)&data, 257, 1);

  // CHECK: Status OK: 6 1
  type_check((const void*)&data.middle);

  struct Datastruct daTYPEART_ar[3];
  // CHECK: [Trace] Alloc [[POINTER:0x[0-9a-f]+]] 257
  __typeart_alloc((const void*)&daTYPEART_ar[0], 257, 3);

  // CHECK: Status OK: 5 2
  type_check((const void*)&daTYPEART_ar[2].end);
  // CHECK: Status OK: 257 1 16 [[POINTER]]
  type_check_containing((const void*)&daTYPEART_ar[2].end);
  // CHECK: Status OK: 257 2 20 [[POINTER]]
  type_check_containing((const void*)&daTYPEART_ar[1].end[1]);

  return 0;
}
