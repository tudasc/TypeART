// clang-format off
// RUN: %remove %tu_yaml && %c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
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

// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} {{i8\*|ptr}} @malloc
// CHECK-NEXT: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER]], i32 256, i64 1)

// CHECK: call void @free({{i8\*|ptr}}{{( noundef)?}} [[POINTER:%[0-9a-z]+]])
// CHECK-NEXT: call void @__typeart_free({{i8\*|ptr}} [[POINTER]])
