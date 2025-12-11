// RUN: %c-to-llvm %s | %apply-typeart -typeart-type-serialization=inline -S | %filecheck --match-full-lines %s

// REQUIRES: !llvm-14

#include <stdlib.h>

struct DataNested {
  struct DataNested* nested_pointer;
};

struct DataHolder {
  double a;
  float b;
  int c;
  struct DataNested nested;
};

int main(void) {
  struct DataHolder* p = (struct DataHolder*)malloc(2 * sizeof(struct DataHolder));
  free(p);
  return 0;
}

// clang-format off
// CHECK-DAG: @_typeart_member_types_DataNested = private constant [1 x ptr] [ptr @_typeart_ptr], comdat($_typeart_DataNested)
// CHECK-DAG: @_typeart_DataHolder = linkonce_odr global %struct._typeart_struct_layout_t { i32 256, i32 24, i16 4, i16 1, ptr @_typeart_typename_DataHolder, ptr @_typeart_offsets_DataHolder, ptr @_typeart_counts_DataHolder, ptr @_typeart_member_types_DataHolder }, comdat
// CHECK-DAG: @_typeart_member_types_DataHolder = private constant [4 x ptr] [ptr @_typeart_double, ptr @_typeart_float, ptr @_typeart_int, ptr @_typeart_DataNested], comdat($_typeart_DataHolder)
// CHECK-DAG: @_typeart_typename_DataHolder = private unnamed_addr constant [11 x i8] c"DataHolder\00", comdat($_typeart_DataHolder), align 1
