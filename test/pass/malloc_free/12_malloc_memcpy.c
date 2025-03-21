// clang-format off
// RUN: %clang-cc -g -O2 -S -emit-llvm %s -o - | %apply-typeart -S 2>&1 | %filecheck %s --check-prefix CHECK-OPT
// clang-format on

// This is a dummy test illustrating problems with -Xclang approach and higher optimizations, losing infos about the
// malloc type
#include <stdlib.h>

typedef struct {
  int nvars;
  int* vartypes;
} struct_grid;

#define taFree(ptr) (ptr != NULL ? (free((char*)(ptr)), ptr = NULL) : (ptr = NULL))
#define taMalloc(type, count) \
  ((unsigned int)(count) * sizeof(type)) > 0 ? ((type*)malloc((unsigned int)(sizeof(type) * (count)))) : (type*)NULL

// clang-format off
// CHECK-OPT: tail call void @free
// CHECK-OPT-NEXT: call void @__typeart_free
// CHECK-OPT: call void @__typeart_alloc({{i8\*|ptr}} %{{[0-9a-z]+}}, i32 {{(11|13)}},
// CHECK-OPT: call void @llvm.memcpy.p0{{(i8)?}}.p0{{(i8)?}}.i64({{i8\*|ptr}} {{(align[[:space:]]?(4|16)[[:space:]])?}}%{{[0-9a-z]+}},
// clang-format on
void setVartypes(struct_grid* pgrid, int nvars, int* vartypes /* = i32 ptr */) {
  int* new_vartypes;
  // free(pgrid->vartypes);
  taFree(pgrid->vartypes);
  //  new_vartypes = (int*)malloc(nvars * sizeof(type_enum));
  new_vartypes = taMalloc(int, nvars);  // llvm does not use bitcast (with -O1 and higher)
  for (int i = 0; i < nvars; i++) {
    new_vartypes[i] = vartypes[i];  // this is a memcpy (with -O1 and higher)
  }
  pgrid->nvars    = nvars;
  pgrid->vartypes = new_vartypes;
}
