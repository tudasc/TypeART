// RUN: %run %s 2>&1 | %filecheck %s

#include "util.h"

#include <stddef.h>
#include <stdio.h>

typedef struct {
  int a;
  double b;
  float c[2];
} DataStruct;

void print_layout() {
  // CHECK: Layout: 0 8 16
  fprintf(stderr, "Layout: %zu %zu %zu\n", offsetof(DataStruct, a), offsetof(DataStruct, b), offsetof(DataStruct, c));
}

void type_check(const void* addr) {
  int id_result         = 0;
  size_t count_check    = 0;
  typeart_status status = typeart_get_type(addr, &id_result, &count_check);

  if (status != TYPEART_OK) {
    fprintf(stderr, "[Error]: Status not OK: %i for %p\n", status, addr);
  } else {
    fprintf(stderr, "Status OK: type_id=%i count=%zu\n", id_result, count_check);
  }
}

void test_get_type() {
  DataStruct data[5];
  // CHECK: Status OK: type_id=5 count=2
  type_check(&data[1].c[0]);
}

void type_check_containing(const void* addr) {
  size_t offset         = 0;
  const void* base_adrr = NULL;
  int id_result         = 0;
  size_t count_check    = 0;

  typeart_status status = typeart_get_containing_type(addr, &id_result, &count_check, &base_adrr, &offset);

  if (status != TYPEART_OK) {
    fprintf(stderr, "[Error]: Status not OK: %i for %p\n", status, addr);
  } else {
    fprintf(stderr, "Status OK: type_id=%i count=%zu offset=%zu base=%p\n", id_result, count_check, offset, base_adrr);
  }
}

void test_get_containing() {
  DataStruct data[5];
  // CHECK: type_id=257 count=4 offset=16 base=
  type_check_containing(&data[1].c[0]);
}

void type_check_sub(const void* addr, size_t offset) {
  const void* base_adrr = NULL;
  size_t count_check    = 0;
  typeart_status status;

  typeart_struct_layout layout;
  {
    int type_id;
    size_t offset_containing;
    status = typeart_get_containing_type(addr, &type_id, &count_check, &base_adrr, &offset_containing);
    if (status != TYPEART_OK) {
      fprintf(stderr, "[Error]: with containing type\n");
      return;
    }
    status = typeart_resolve_type_id(type_id, &layout);
    if (status != TYPEART_OK) {
      fprintf(stderr, "[Error]: with resolving struct\n");
      return;
    }
  }

  int subtype_id;
  size_t subtype_byte_offset;
  status = typeart_get_subtype(base_adrr, offset, &layout, &subtype_id, &base_adrr, &subtype_byte_offset, &count_check);

  if (status != TYPEART_OK) {
    fprintf(stderr, "[Error]: Status not OK: %i for %p\n", status, addr);
  } else {
    fprintf(stderr, "Status OK: type_id=%i count=%zu offset=%zu addr(%p) base(%p)\n", subtype_id, count_check,
            subtype_byte_offset, addr, base_adrr);
  }
}

void test_get_subtype() {
  DataStruct data[5];
  // CHECK: type_id=5 count=1 offset=0
  type_check_sub(&data[1], offsetof(DataStruct, c[1]));
}

int main(void) {
  print_layout();

  test_get_type();

  test_get_containing();

  test_get_subtype();

  return 0;
}
