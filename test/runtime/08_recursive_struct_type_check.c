// RUN: %run %s -alloca-array-only=true 2>&1 | FileCheck %s

#include "../struct_defs.h"
#include "util.h"

#include <stdint.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  s_int s;

  // CHECK: [Trace] Alloc 0x{{.*}} struct.s_ptr_to_self_t 16 1
  s_ptr_to_self* a = malloc(sizeof(s_ptr_to_self));
  // CHECK: Ok
  check_struct(a, "struct.s_ptr_to_self_t", 1);
  // CHECK: Ok
  check(a, get_struct_id(0), 1, 0);
  // CHECK: Ok
  check(a, TA_PTR, 1, 1);
  // CHECK: Ok
  check(&a->b, TA_PTR, 1, 1);
  // CHECK: Error: Unknown address
  check(a + 1, get_struct_id(0), 1, 0);
  // CHECK: [Trace] Free 0x{{.*}}
  free(a);

  // CHECK: [Trace] Alloc 0x{{.*}} struct.s_struct_member_t 32 1
  s_struct_member* b = malloc(sizeof(s_struct_member));
  // CHECK: Ok
  check_struct(b, "struct.s_struct_member_t", 1);
  // CHECK: Ok
  check(b, get_struct_id(1), 1, 0);
  // CHECK: Ok
  check(b, TA_INT32, 1, 1);
  // CHECK: Ok
  check(&b->b, get_struct_id(0), 1, 0);
  // CHECK: Ok
  check(&b->b, TA_PTR, 1, 1);
  // CHECK: Ok
  check(&b->c, TA_PTR, 1, 1);
  // CHECK: Error: Unknown address
  check(b + 1, TA_INT32, 1, 1);
  // CHECK: [Trace] Free 0x{{.*}}
  free(b);

  // CHECK: [Trace] Alloc 0x{{.*}} struct.s_aos_t 96 1
  s_aos* c = malloc(sizeof(s_aos));
  // CHECK: Ok
  check_struct(c, "struct.s_aos_t", 1);
  // CHECK: Ok
  check(c, get_struct_id(2), 1, 0);
  // CHECK: Ok
  check(c, TA_INT32, 1, 1);
  // CHECK: Ok
  check(&c->b, get_struct_id(1), 2, 0);
  // CHECK: Ok
  check(&c->b[1], get_struct_id(1), 1, 0);
  // CHECK: Ok
  check(&c->c, TA_PTR, 3, 1);
  // CHECK: Error: Unknown address
  check(c + 1, TA_INT32, 1, 1);
  // CHECK: [Trace] Free 0x{{.*}}
  free(c);

  return 0;
}
