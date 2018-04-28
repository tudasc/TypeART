// RUN: %scriptpath/applyAndRun.sh %s %pluginname -must %pluginpath %rtpath | FileCheck %s

#include <stdlib.h>

int main(int argc, char** argv) {
  int* p = (int*)malloc(42 * sizeof(int));
  free(p);
  return 0;
}

// CHECK: MUST Support Runtime Trace
// CHECK: Alloc    0x{{.*}}    int    4   42
// CHECK: Free 0x{{.*}}