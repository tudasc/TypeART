// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
#include <stdlib.h>
typedef struct ms {
  int a;
  double b;
} mystruct;

void test() {
  void* m = malloc(sizeof(mystruct));
  free(m);
}

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK: Free{{[ ]*}}:{{[ ]*}}1
// Also required (TBD): Alloca{{[ ]*}}:{{[ ]*}}0