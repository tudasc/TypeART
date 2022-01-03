// RUN: %run %s 2>&1 | %filecheck %s

#include "../../lib/runtime/RuntimeInterface.h"

#include <stdbool.h>
#include <stdio.h>

struct Datastruct {
  int start;
  double middle;
  float end;
};

struct Secondstruct {
  int start;
  float end;
};

void print_data(int type_id) {
  const char* typeart_name = typeart_get_type_name(type_id);
  size_t size              = typeart_get_type_size(type_id);
  bool is_builtin          = typeart_is_builtin_type(type_id);
  bool is_reserved         = typeart_is_reserved_type(type_id);
  bool is_struct           = typeart_is_struct_type(type_id);
  bool is_userdef          = typeart_is_userdefined_type(type_id);
  bool is_valid            = typeart_is_valid_type(type_id);
  bool is_vec              = typeart_is_vector_type(type_id);
  printf("Name: %s: %zu %i %i %i %i %i %i\n", typeart_name, size, is_builtin, is_reserved, is_struct, is_userdef,
         is_valid, is_vec);
}

int main(int argc, char** argv) {
  struct Datastruct data     = {0};
  struct Secondstruct data_2 = {0};

  int type_id = 0;
  typeart_get_type_id(&data, &type_id);
  print_data(type_id);

  type_id = 0;
  typeart_get_type_id(&data_2, &type_id);
  print_data(type_id);

  return data.start + data_2.start;
}

// CHECK: Name: struct.Datastruct: 24 0 0 1 1 1 0
// CHECK: Name: struct.Secondstruct: 8 0 0 1 1 1 0