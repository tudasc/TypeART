// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-instumentation=true -S 2>&1 | %filecheck %s
// clang-format on

#include <stdlib.h>
void test() {
  int* p = (int*)malloc(42 * sizeof(int));
}
// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}0
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

// CHECK: %struct.typeart_struct_layout_t = type { i32, ptr, i64, i64, ptr, ptr, ptr }
// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} {{i8\*|ptr}} @malloc
// CHECK-NEXT: call void @__typeart_alloc_mty({{i8\*|ptr}} [[POINTER]], ptr {{.*}}, i64 42)
