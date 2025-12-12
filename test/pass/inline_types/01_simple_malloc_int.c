// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-type-serialization=inline -S 2>&1 | %filecheck %s

// RUN: %c-to-llvm %s | %apply-typeart -typeart-type-serialization=hybrid -S 2>&1 | %filecheck %s --check-prefix HYBRID

// RUN: %c-to-llvm %s | %apply-typeart -typeart-type-serialization=file -S 2>&1 | %filecheck %s --check-prefix FILE

// REQUIRES: !llvm-14
// clang-format on

#include <stdlib.h>
void test() {
  int* p = (int*)malloc(42 * sizeof(int));
}
// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}0
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

// CHECK: %struct._typeart_struct_layout_t = type { i32, i32, i16, i16, ptr, ptr, ptr, ptr }
// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} ptr @malloc
// CHECK-NEXT: call void @__typeart_alloc_mty(ptr [[POINTER]], ptr {{.*}}, i64 42)

// HYBRID-NOT: %struct._typeart_struct_layout_t = type {
// HYBRID: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} ptr @malloc
// HYBRID-NEXT: call void @__typeart_alloc(ptr [[POINTER]], i32 13, i64 42)

// FILE-NOT: %struct._typeart_struct_layout_t = type {
// FILE: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} ptr @malloc
// FILE-NEXT: call void @__typeart_alloc(ptr [[POINTER]], i32 13, i64 42)
