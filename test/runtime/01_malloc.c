// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-must-alloca" %rtpath | FileCheck %s

#include <stdlib.h>
#include "util.h"

int main(int argc, char** argv) {
  // CHECK: MUST Support Runtime Trace
  // CHECK: Alloc    0x{{.*}}    int    4   42
  int* p = (int*)malloc(42 * sizeof(int));
  check(p, 4);
  // CHECK: Ok
  check(p + 5, 4);
  // CHECK: Ok
  check(p + 41, 4);
  // CHECK: Ok
  check(p + 42, 4);
  // CHECK: Error: Unknown address
  free(p);
  // CHECK: Free 0x{{.*}}
  return 0;
}
