// RUN: %run %s --manual 2>&1 | FileCheck %s

#include "../../lib/runtime/CallbackInterface.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>


struct Datastruct {
  int start;
  double middle;
  float end;
};

int main(int argc, char** argv) {
  struct Datastruct  data;

  __typeart_alloc((const void*)&data, 257, 1);

  int id_result         = 0;
  size_t count_check    = 0;
  typeart_status status = typeart_get_type((const void*)&data.middle, &id_result, &count_check);

  if (status != TA_OK) {
    fprintf(stderr, "[Error]: Status not OK: %i\n", status);
  } else {
    if (1 != count_check) {
      fprintf(stderr, "[Error]: Count check failed %zu\n", count_check);
    }
    if (6 != id_result) {
      fprintf(stderr, "[Error]: ID check failed %i\n", id_result);
    }
    fprintf(stderr, "[Trace]: Status OK: %i %zu\n", id_result, count_check);
  }
  return 0;
}

// CHECK-NOT: [Error]