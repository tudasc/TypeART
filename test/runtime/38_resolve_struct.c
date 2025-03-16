// RUN: %run %s --manual 2>&1 | %filecheck %s

// dimeta creates a different type id (258), making the test fail
// REQUIRES: !dimeta

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
  typeart_type_info info;
  typeart_status status;
  status = typeart_get_type(addr, &info);

  if (status != TYPEART_OK) {
    fprintf(stderr, "[Error]: Status not OK: %i for %p\n", status, addr);
  } else {
    fprintf(stderr, "Status OK: %i %zu\n", info.type_id, info.count);
  }
}

void type_check_containing(const void* addr) {
  size_t offset         = 0;
  const void* base_adrr = NULL;
  int id_result         = 0;
  size_t count_check    = 0;
  typeart_status status;

  typeart_type_info info;
  status = typeart_get_type(addr, &info);
  if (status != TYPEART_OK) {
    fprintf(stderr, "[Error]: get_type with containing type\n");
    return;
  }
  typeart_base_type_info containing;
  status = typeart_get_containing_type(info, &containing, &offset);

  if (status != TYPEART_OK) {
    fprintf(stderr, "[Error]: Status not OK: %i for %p\n", status, addr);
  } else {
    fprintf(stderr, "Status OK: %i %zu %zu %p\n", containing.type_id, containing.count, offset, containing.address);
  }
}

int main(int argc, char** argv) {
  // CHECK-NOT: [Error]

  struct Datastruct data;
  __typeart_alloc((const void*)&data, 259, 1);

  // CHECK: Status OK: 24 1
  type_check((const void*)&data.middle);

  struct Datastruct data_2[3];
  // CHECK: [Trace] Alloc [[POINTER:0x[0-9a-f]+]] 259
  __typeart_alloc((const void*)&data_2[0], 259, 3);

  // CHECK: Status OK: 23 2
  type_check((const void*)&data_2[2].end);
  // CHECK: Status OK: 259 1 16 [[POINTER]]
  type_check_containing((const void*)&data_2[2].end);
  // CHECK: Status OK: 259 2 20 [[POINTER]]
  type_check_containing((const void*)&data_2[1].end[1]);

  return 0;
}
