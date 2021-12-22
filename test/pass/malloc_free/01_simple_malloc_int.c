// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on
#include <stdlib.h>
void test() {
  int* p = (int*)malloc(42 * sizeof(int));
}

// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} i8* @malloc
// CHECK-NEXT: call void @__typeart_alloc(i8* [[POINTER]], i32 2, i64 42)
// CHECK-NEXT: bitcast i8* [[POINTER]] to i32*

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}0
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
