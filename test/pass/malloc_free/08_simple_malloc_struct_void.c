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

// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} {{i8\*|ptr}} @malloc
// CHECK-NEXT: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER]], i32 {{2|10}}, i64 16)

// CHECK: call void @free({{i8\*|ptr}}{{( noundef)?}} [[POINTER:%[0-9a-z]+]])
// CHECK-NEXT: call void @__typeart_free({{i8\*|ptr}} [[POINTER]])

// PASS-OUT: TypeArtPass [Heap]
// PASS-OUT-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// PASS-OUT-NEXT: Free{{[ ]*}}:{{[ ]*}}1
// PASS-OUT-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
