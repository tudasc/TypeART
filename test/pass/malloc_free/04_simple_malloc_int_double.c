// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on
#include <stdlib.h>
void test() {
  int* p     = (int*)malloc(42 * sizeof(int));
  double* pd = (double*)malloc(42 * sizeof(double));
}
// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}0
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} {{i8\*|ptr}} @malloc
// CHECK-NEXT: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER]],

// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} {{i8\*|ptr}} @malloc
// CHECK-NEXT: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER]], i32 24
