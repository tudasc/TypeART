// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on
#include <stdlib.h>

void test() {
  int* p = (int*)malloc(42 * sizeof(int));
  free(p);
}

// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} i8* @malloc
// CHECK-NEXT: call void @__typeart_alloc(i8* [[POINTER]],
// CHECK-NEXT: bitcast i8* [[POINTER]] to i32*

// CHECK: call void @free(i8*{{( noundef)?}} [[POINTER:%[0-9a-z]+]])
// CHECK-NEXT: call void @__typeart_free(i8* [[POINTER]])

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
