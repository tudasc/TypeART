// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-type-serialization=inline -S 2>&1 | %filecheck %s

// RUN: %c-to-llvm %s | %apply-typeart -typeart-type-serialization=hybrid -S 2>&1 | %filecheck %s --check-prefix HYBRID

// RUN: %c-to-llvm %s | %apply-typeart -typeart-type-serialization=file -S 2>&1 | %filecheck %s --check-prefix FILE

// REQUIRES: llvm-18 || llvm-19
// clang-format on

#include <stdlib.h>
void test() {
  int* p = (int*)malloc(42 * sizeof(int));
}
// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}0
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

// CHECK: %struct._typeart_struct_layout_t = type { i32, ptr, i64, i64, ptr, ptr, ptr, i32 }
// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} ptr @malloc
// CHECK-NEXT: call void @__typeart_alloc_mty(ptr [[POINTER]], ptr {{.*}}, i64 42)

// HYBRID-NOT: %struct._typeart_struct_layout_t = type { i32, ptr, i64, i64, ptr, ptr, ptr, i32 }
// HYBRID: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} ptr @malloc
// HYBRID-NEXT: call void @__typeart_alloc(ptr [[POINTER]], i32 13, i64 42)

// FILE-NOT: %struct._typeart_struct_layout_t = type { i32, ptr, i64, i64, ptr, ptr, ptr, i32 }
// FILE: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} ptr @malloc
// FILE-NEXT: call void @__typeart_alloc(ptr [[POINTER]], i32 13, i64 42)
