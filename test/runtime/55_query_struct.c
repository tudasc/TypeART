// RUN: %run %s 2>&1 | %filecheck %s

#include "../../lib/runtime/RuntimeInterface.h"

#include <stdbool.h>
#include <stdio.h>

struct Datastruct {
  int start;
  double middle;
  float end;
};

void print_layout(typeart_struct_layout* layout) {
  fprintf(stderr, "layout->id %i\n", layout->type_id);
  fprintf(stderr, "layout->name %s\n", layout->name);
  fprintf(stderr, "layout->num_members %zu\n", layout->num_members);
  fprintf(stderr, "layout->extent %zu\n", layout->extent);
}

int main(void) {
  struct Datastruct data = {0};

  typeart_struct_layout layout;
  typeart_type_info info;
  typeart_get_type(&data, &info);
  fprintf(stderr, "struct count %zu\n", info.count);

  // CHECK-NOT: [Error] status unexpectedly OK
  // CHECK: layout->id
  // CHECK: layout->name
  // CHECK: layout->num_members 0
  typeart_struct_layout layout_err;
  typeart_status status = typeart_resolve_type_id(777, &layout_err);
  if (status == TYPEART_OK) {
    fprintf(stderr, "[Error] status unexpectedly OK\n");
  }
  print_layout(&layout_err);

  return 0;
}
