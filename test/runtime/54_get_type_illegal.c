// RUN: %run %s 2>&1 | %filecheck %s

#include "RuntimeInterface.h"
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
  int id_result      = 0;
  size_t count_check = 0;

  typeart_type_info info;
  typeart_status status = typeart_get_type(addr, &info);

  if (status != TYPEART_OK) {
    fprintf(stderr, "[Expected]: Status not OK: %s for %p\n", err_code_to_string(status), addr);
  } else {
    id_result   = info.type_id;
    count_check = info.count;
    fprintf(stderr, "Status OK: type_id=%i count=%zu\n", id_result, count_check);
  }
}
// void type_check_containing(const void* addr) {
//   size_t offset         = 0;
//   const void* base_adrr = NULL;
//   int id_result         = 0;
//   size_t count_check    = 0;

//   typeart_status status = typeart_get_containing_type(addr, &id_result, &count_check, &base_adrr, &offset);

//   if (status != TYPEART_OK) {
//     fprintf(stderr, "[Expected]: Status not OK: %s for %p\n", err_code_to_string(status), addr);
//   } else {
//     fprintf(stderr, "Status OK: type_id=%i count=%zu offset=%zu base=%p\n", id_result, count_check, offset,
//     base_adrr);
//   }
// }
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
  typeart_type_info info_base;
  status = typeart_get_containing_type(info, &info_base, &offset);

  if (status != TYPEART_OK) {
    fprintf(stderr, "[Expected]: Status not OK: %s for %p\n", err_code_to_string(status), addr);
  } else {
    fprintf(stderr, "Status OK: type_id=%i count=%zu offset=%zu base=%p\n", info_base.type_id, info_base.count, offset,
            info_base.address);
  }
}

void type_check_sub(const void* addr, size_t offset) {
  const void* base_adrr = NULL;
  typeart_status status;

  typeart_struct_layout layout;
  {
    // int type_id;
    size_t offset_containing;
    // typeart_type_info type, typeart_type_info* containing_type,
    //                                        size_t* byte_offset
    typeart_type_info info;
    status = typeart_get_type(addr, &info);
    if (status != TYPEART_OK) {
      fprintf(stderr, "[Error]: get_type with containing type\n");
      return;
    }
    typeart_type_info info_base;
    status = typeart_get_containing_type(info, &info_base, &offset_containing);
    if (status != TYPEART_OK) {
      fprintf(stderr, "[Error]: with containing type\n");
      return;
    }
    base_adrr = info_base.address;
    // count_check =
    status    = typeart_resolve_type_id(info.type_id, &layout);
    if (status != TYPEART_OK) {
      fprintf(stderr, "[Error]: with resolving struct\n");
      return;
    }
  }

  // const typeart_struct_layout* container_layout, const void* base_addr,
  //                                       size_t offset, typeart_type_info* subtype_info, size_t* subtype_byte_offset);
  size_t subtype_byte_offset;
  typeart_type_info subtype_info;
  status = typeart_get_subtype(&layout, base_adrr, offset, &subtype_info,
                               &subtype_byte_offset);  //(base_adrr, offset, &layout, &subtype_id, &base_adrr,
                                                       //&subtype_byte_offset, &count_check);

  if (status != TYPEART_OK) {
    fprintf(stderr, "[Expected]: Status not OK: %s for %p\n", err_code_to_string(status), addr);
  } else {
    int subtype_id     = subtype_info.type_id;
    size_t count_check = 0;
    count_check        = subtype_info.count;
    fprintf(stderr, "Status OK: type_id=%i count=%zu offset=%zu addr(%p) base(%p)\n", subtype_id, count_check,
            subtype_byte_offset, subtype_info.address, base_adrr);
  }
}

void test_get_type() {
  DataStruct data[5];
  void* illegal_addr = (char*)&data[0].a + sizeof(int);

  // CHECK: Diff(4)
  ptrdiff_t diff = (char*)illegal_addr - (char*)&data[0];
  fprintf(stderr, "Diff(%td)\n", diff);
  // CHECK: [Expected]: Status not OK: TYPEART_BAD_ALIGNMENT
  type_check(illegal_addr);
}

void test_get_containing() {
  DataStruct data[5];
  // Illegal address
  void* illegal_addr = (char*)&data[0].a + sizeof(int);  // goes into padding
  // CHECK: [Error]: get_type with containing type
  type_check_containing(illegal_addr);
}

void test_get_subtype() {
  DataStruct data[5];
  void* legal_addr = (char*)&data[0].a;

  // Legal address but illegal offset:
  // CHECK: [Expected]: Status not OK: TYPEART_BAD_ALIGNMENT
  type_check_sub(legal_addr, sizeof(int));

  // CHECK: [Expected]: Status not OK: TYPEART_BAD_OFFSET
  type_check_sub(legal_addr, 2 * sizeof(DataStruct));
}

void test_get_subtype_direct() {
  typeart_struct_layout layout;
  typeart_status status = typeart_resolve_type_id(777, &layout);
  int subtype_id;
  size_t subtype_byte_offset;
  const void* base_adrr = NULL;
  size_t count_check;
  typeart_type_info subtype_info;
  status = typeart_get_subtype(&layout, NULL, 0, &subtype_info, &subtype_byte_offset);

  // CHECK: [Expected]: Status not OK: TYPEART_ERROR
  fprintf(stderr, "[Expected]: Status not OK: %s\n", err_code_to_string(status));
}

int main(void) {
  print_layout();

  test_get_type();

  test_get_containing();

  test_get_subtype();

  test_get_subtype_direct();

  return 0;
}
