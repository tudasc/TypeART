// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-must-alloca" %rtpath | FileCheck %s

#include <stdlib.h>

int main(int argc, char** argv) {
  int a[64];
  return 0;
}

// CHECK: MUST Support Runtime Trace
// CHECK: Alloc    0x{{.*}}    int    4   64