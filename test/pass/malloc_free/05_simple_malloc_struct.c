// clang-format off
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
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

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK: Free{{[ ]*}}:{{[ ]*}}1
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0
