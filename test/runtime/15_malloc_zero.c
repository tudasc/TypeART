// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-typeart-alloca" %rtpath 2>&1 | FileCheck %s

#include <stdlib.h>

int main(int argc, char** argv) {
  const int n = 42;
  // CHECK: [Trace] TypeART Runtime Trace

  // CHECK: [Warning]{{.*}}Zero-size allocation
  char* a = malloc(0);
  // CHECK-NOT: [Error]
  free(a);

  return 0;
}
