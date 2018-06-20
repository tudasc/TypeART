// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-typeart-alloca" %rtpath 2>&1 | FileCheck %s

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
  check(a, S_INT_ID, 1, 0);
  // CHECK: Ok
  check(a, C_INT, 1, 1);
  // CHECK: Error: Unknown address
  check(a + 1, S_INT_ID, 1, 1);
  // CHECK: [Trace] Free 0x{{.*}}
  free(a);

  // CHECK: [Trace] Alloc 0x{{.*}} struct.s_builtins_t 16 1
  s_builtins* b = malloc(sizeof(s_builtins));
  // CHECK: Ok
  check_struct(b, "struct.s_builtins_t", 1);
  // CHECK: Ok
  check(b, S_BUILTINS_ID, 1, 0);
  // CHECK: Ok
  check(b, C_INT, 1, 1);
  // CHECK: Error: Type mismatch
  check(b, C_CHAR, 1, 1);
  // CHECK: Error: Bad alignment
  check(((uint8_t*)b) + 2, C_INT, 1, 1);
  // CHECK: Ok
  check(&b->b, C_CHAR, 1, 1);
  // CHECK: Error: Bad alignment
  check(((uint8_t*)b) + 5, C_LONG, 1, 1);
  // CHECK: Ok
  check(&b->c, C_LONG, 1, 1);
  // CHECK: Error: Unknown address
  check(b + 1, S_INT_ID, 1, 0);
  // CHECK: [Trace] Free 0x{{.*}}
  free(b);

  // CHECK: [Trace] Alloc 0x{{.*}} struct.s_arrays_t 72 1
  s_arrays* c = malloc(sizeof(s_arrays));
  // CHECK: Ok
  check_struct(c, "struct.s_arrays_t", 1);
  // CHECK: Ok
  check(c, S_ARRAYS_ID, 1, 0);
  // CHECK: Ok
  check(c, C_INT, 3, 1);
  // CHECK: Ok
  check(((uint8_t*)c) + 4, C_INT, 2, 1);
  // CHECK: Ok
  check(((uint8_t*)c) + 8, C_INT, 1, 1);
  // CHECK: Bad alignment
  check(((uint8_t*)c) + 12, C_LONG, 2, 1);
  // CHECK: Ok
  check(&c->b, C_LONG, 2, 1);
  // CHECK: Ok
  check(&c->b[1], C_LONG, 1, 1);
  // CHECK: Ok
  check(&c->e[2], C_CHAR, 3, 1);
  // CHECK: Error: Unknown address
  check(c + 1, S_ARRAYS_ID, 1, 0);
  // CHECK: [Trace] Free 0x{{.*}}
  free(c);

  // CHECK: [Trace] Alloc 0x{{.*}} struct.s_ptrs_t 32 1
  s_ptrs* d = malloc(sizeof(s_ptrs));
  // CHECK: Ok
  check_struct(d, "struct.s_ptrs_t", 1);
  // CHECK: Ok
  check(d, S_PTRS_ID, 1, 0);
  // CHECK: Ok
  check(d, C_CHAR, 1, 1);
  // CHECK: Ok
  check(&d->b, UNKNOWN, 1, 1);
  // CHECK: Bad alignment
  check(((uint8_t*)d) + 12, UNKNOWN, 1, 1);
  // CHECK: Ok
  check(&d->d, UNKNOWN, 1, 1);
  // CHECK: Error: Unknown address
  check(d + 1, S_PTRS_ID, 1, 0);
  // CHECK: [Trace] Free 0x{{.*}}
  free(d);

  // CHECK: [Trace] Alloc 0x{{.*}} struct.s_mixed_simple_t 48 1
  s_mixed_simple* e = malloc(sizeof(s_mixed_simple));
  // CHECK: Ok
  check_struct(e, "struct.s_mixed_simple_t", 1);
  // CHECK: Ok
  check(e, S_MIXED_SIMPLE_ID, 1, 0);
  // CHECK: Ok
  check(e, C_INT, 1, 1);
  // CHECK: Ok
  check(((uint8_t*)e) + 16, C_DOUBLE, 2, 1);
  // CHECK: Ok
  check(&e->c, UNKNOWN, 1, 1);
  // CHECK: Error: Unknown address
  check(e + 1, S_MIXED_SIMPLE_ID, 1, 0);
  // CHECK: [Trace] Free 0x{{.*}}
  free(e);

  return 0;
}
