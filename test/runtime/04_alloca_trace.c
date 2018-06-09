// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-must-alloca" %rtpath | FileCheck %s

#include <stdlib.h>

int main(int argc, char** argv) {
    const int n = 42;
    // CHECK: MUST Support Runtime Trace

    // CHECK: Alloc    0x{{.*}}    char   1   42
    char a[n];

    // CHECK: Alloc    0x{{.*}}    short    2   42
    short b[n];

    // CHECK: Alloc    0x{{.*}}    int    4   42
    int c[n];

    // CHECK: Alloc    0x{{.*}}    long    8   42
    long d[n];

    // CHECK: Alloc    0x{{.*}}    float    4   42
    float e[n];

    // CHECK: Alloc    0x{{.*}}    double    8   42
    double f[n];

    // CHECK: Alloc    0x{{.*}}    unknown    8   42
    int* g[n];

    // CHECK: Free 0x{{.*}}
    // CHECK: Free 0x{{.*}}
    // CHECK: Free 0x{{.*}}
    // CHECK: Free 0x{{.*}}
    // CHECK: Free 0x{{.*}}
    // CHECK: Free 0x{{.*}}
    // CHECK: Free 0x{{.*}}

    return 0;
}
