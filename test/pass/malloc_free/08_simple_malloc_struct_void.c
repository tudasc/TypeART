// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S | %filecheck %s
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s --check-prefix=PASS-OUT
// clang-format on
#include <stdlib.h>
typedef struct ms {
  int a;
  double b;
} mystruct;

void test() {
  void* m = malloc(sizeof(mystruct));
  free(m);
}

// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} i8* @malloc
// CHECK-NEXT: call void @__typeart_alloc(i8* [[POINTER]], i32 0, i64 16)
// CHECK-NOT: bitcast i8* [[POINTER]] to i32*

// CHECK: call void @free(i8*{{( noundef)?}} [[POINTER:%[0-9a-z]+]])
// CHECK-NEXT: call void @__typeart_free(i8* [[POINTER]])

// PASS-OUT: TypeArtPass [Heap]
// PASS-OUT-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// PASS-OUT-NEXT: Free{{[ ]*}}:{{[ ]*}}1
// PASS-OUT-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
