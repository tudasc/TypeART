// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-type-serialization=inline -S 2>&1 | %filecheck %s

// RUN: %c-to-llvm %s | %apply-typeart -typeart-type-serialization=hybrid -S 2>&1 | %filecheck %s --check-prefix HYBRID

// RUN: %c-to-llvm %s | %apply-typeart -typeart-type-serialization=file -S 2>&1 | %filecheck %s --check-prefix FILE

// REQUIRES: llvm-18 || llvm-19
// clang-format on
#include <stdlib.h>
typedef struct ms {
  int a;
  double b;
} mystruct;

void test() {
  mystruct* m = (mystruct*)malloc(sizeof(mystruct));
  free(m);
}
// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} ptr @malloc
// CHECK-NEXT: call void @__typeart_alloc_mty(ptr [[POINTER]], ptr {{.*}}, i64 1)

// CHECK: call void @free(ptr{{( noundef)?}} [[POINTER:%[0-9a-z]+]])
// CHECK-NEXT: call void @__typeart_free(ptr [[POINTER]])

// HYBRID: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} ptr @malloc
// HYBRID-NOT: call void @__typeart_alloc(ptr [[POINTER]]
// HYBRID: call void @free(ptr{{( noundef)?}} [[POINTER:%[0-9a-z]+]])
// HYBRID-NEXT: call void @__typeart_free(ptr [[POINTER]])

// FILE: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} ptr @malloc
// FILE: call void @__typeart_alloc(ptr [[POINTER]], i32
// FILE: call void @free(ptr{{( noundef)?}} [[POINTER:%[0-9a-z]+]])
// FILE-NEXT: call void @__typeart_free(ptr [[POINTER]])
