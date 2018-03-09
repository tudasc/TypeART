// RUN: %scriptpath/applyAndRun.sh %s %pluginname -must %pluginpath %rtpath | FileCheck %s

#include <stdlib.h>
#include <stdio.h>
int main(int argc, char** argv) {
    int* p = (int*)malloc(42 * sizeof(int));
    return 0;
}

// CHECK: MUST Support Runtime Trace
// CHECK: Allocation    0x{{.*}}    {{[0-9]+}}    4   42
