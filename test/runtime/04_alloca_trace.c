// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-must-alloca" %rtpath 2>&1 | FileCheck %s

#include <stdlib.h>

int main(int argc, char** argv) {
    const int n = 42;
    // CHECK: [Trace] TypeART Runtime Trace

    // CHECK: [Trace] Alloc 0x{{.*}} char 1 42
    char a[n];

    // CHECK: [Trace] Alloc 0x{{.*}} short 2 42
    short b[n];

    // CHECK: [Trace] Alloc 0x{{.*}} int 4 42
    int c[n];

    // CHECK: [Trace] Alloc 0x{{.*}} long 8 42
    long d[n];

    // CHECK: [Trace] Alloc 0x{{.*}} float 4 42
    float e[n];

    // CHECK: [Trace] Alloc 0x{{.*}} double 8 42
    double f[n];

    // CHECK: [Trace] Alloc 0x{{.*}} unknown 8 42
    int* g[n];

    // CHECK: [Trace] Free 0x{{.*}}
    // CHECK: [Trace] Free 0x{{.*}}
    // CHECK: [Trace] Free 0x{{.*}}
    // CHECK: [Trace] Free 0x{{.*}}
    // CHECK: [Trace] Free 0x{{.*}}
    // CHECK: [Trace] Free 0x{{.*}}
    // CHECK: [Trace] Free 0x{{.*}}

    return 0;
}
