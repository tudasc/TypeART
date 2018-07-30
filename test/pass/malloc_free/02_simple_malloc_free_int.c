// clang-format off
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
// clang-format on
#include <stdlib.h>
#include <stdalign.h>

struct jp {
int a;
alignas(8) int b;
double x;
};
void test() {
  int* p = (int*)malloc(42 * sizeof(int));
//  free(p);
struct  jp foo;
}

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK: Free{{[ ]*}}:{{[ ]*}}1
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0
