// RUN: %run %s -alloca-array-only=true 2>&1 | FileCheck %s

#include "../struct_defs.h"
#include "util.h"

#include <stdint.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  // CHECK: [Trace] Alloc 0x{{.*}} struct.s_int_t 4 1
  s_int* a = malloc(sizeof(s_int));
  // CHECK: Ok
  check_struct(a, "struct.s_int_t", 1);
  // CHECK: Ok
  check(a, get_struct_id(0), 1, 0);
  // CHECK: Ok
  check(a, TA_INT32, 1, 1);
  // CHECK: Error: Unknown address
  check(a + 1, get_struct_id(0), 1, 1);
  // CHECK: [Trace] Free 0x{{.*}}
  free(a);

  // CHECK: [Trace] Alloc 0x{{.*}} struct.s_builtins_t 16 1
  s_builtins* b = malloc(sizeof(s_builtins));
  // CHECK: Ok
  check_struct(b, "struct.s_builtins_t", 1);
  // CHECK: Ok
  check(b, get_struct_id(1), 1, 0);
  // CHECK: Ok
  check(b, TA_INT32, 1, 1);
  // CHECK: Error: Type mismatch
  check(b, TA_INT8, 1, 1);
  // CHECK: Error: Bad alignment
  check(((uint8_t*)b) + 2, TA_INT32, 1, 1);
  // CHECK: Ok
  check(&b->b, TA_INT8, 1, 1);
  // CHECK: Error: Bad alignment
  check(((uint8_t*)b) + 5, TA_INT64, 1, 1);
  // CHECK: Ok
  check(&b->c, TA_INT64, 1, 1);
  // CHECK: Error: Unknown address
  check(b + 1, get_struct_id(1), 1, 0);
  // CHECK: [Trace] Free 0x{{.*}}
  free(b);

  // CHECK: [Trace] Alloc 0x{{.*}} struct.s_arrays_t 72 1
  s_arrays* c = malloc(sizeof(s_arrays));
  // CHECK: Ok
  check_struct(c, "struct.s_arrays_t", 1);
  // CHECK: Ok
  check(c, get_struct_id(2), 1, 0);
  // CHECK: Ok
  check(c, TA_INT32, 3, 1);
  // CHECK: Ok
  check(((uint8_t*)c) + 4, TA_INT32, 2, 1);
  // CHECK: Ok
  check(((uint8_t*)c) + 8, TA_INT32, 1, 1);
  // CHECK: Bad alignment
  check(((uint8_t*)c) + 12, TA_INT64, 2, 1);
  // CHECK: Ok
  check(&c->b, TA_INT64, 2, 1);
  // CHECK: Ok
  check(&c->b[1], TA_INT64, 1, 1);
  // CHECK: Ok
  check(&c->e[2], TA_INT8, 3, 1);
  // CHECK: Error: Unknown address
  check(c + 1, get_struct_id(2), 1, 0);
  // CHECK: [Trace] Free 0x{{.*}}
  free(c);

  // CHECK: [Trace] Alloc 0x{{.*}} struct.s_ptrs_t 32 1
  s_ptrs* d = malloc(sizeof(s_ptrs));
  // CHECK: Ok
  check_struct(d, "struct.s_ptrs_t", 1);
  // CHECK: Ok
  check(d, get_struct_id(3), 1, 0);
  // CHECK: Ok
  check(d, TA_INT8, 1, 1);
  // CHECK: Ok
  check(&d->b, TA_PTR, 1, 1);
  // CHECK: Bad alignment
  check(((uint8_t*)d) + 12, TA_PTR, 1, 1);
  // CHECK: Ok
  check(&d->d, TA_PTR, 1, 1);
  // CHECK: Error: Unknown address
  check(d + 1, get_struct_id(3), 1, 0);
  // CHECK: [Trace] Free 0x{{.*}}
  free(d);

  // CHECK: [Trace] Alloc 0x{{.*}} struct.s_mixed_simple_t 48 1
  s_mixed_simple* e = malloc(sizeof(s_mixed_simple));
  // CHECK: Ok
  check_struct(e, "struct.s_mixed_simple_t", 1);
  // CHECK: Ok
  check(e, get_struct_id(4), 1, 0);
  // CHECK: Ok
  check(e, TA_INT32, 1, 1);
  // CHECK: Ok
  check(((uint8_t*)e) + 16, TA_DOUBLE, 2, 1);
  // CHECK: Ok
  check(&e->c, TA_PTR, 1, 1);
  // CHECK: Error: Unknown address
  check(e + 1, get_struct_id(4), 1, 0);
  // CHECK: [Trace] Free 0x{{.*}}
  free(e);

  return 0;
}
