// RUN: %scriptpath/applyAndRun.sh %s %pluginname -must %pluginpath %rtpath | FileCheck %s
// UN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s


#include <stdlib.h>

int main(int argc, char** argv) {
  int a[64];
  return 0;
}

// CHECK: MUST Support Runtime Trace
// CHECK: Allocation    0x{{.*}}    {{[0-9]+}}    4   64