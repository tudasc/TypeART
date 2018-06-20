// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-must-alloca" %rtpath 2>&1 | FileCheck %s

#include <stdlib.h>

int main(int argc, char** argv) {
    const int n = 42;
    // CHECK: [Trace] TypeART Runtime Trace

    // CHECK: [Trace] Alloc 0x{{.*}} uchar 1 42
    unsigned char a[n];

    // CHECK: [Trace] Alloc 0x{{.*}} ushort 2 42
    unsigned short b[n];

    // CHECK: [Trace] Alloc 0x{{.*}} uint 4 42
    unsigned int c[n];

    // CHECK: [Trace] Alloc 0x{{.*}} ulong 8 42
    unsigned long d[n];

    // CHECK: [Trace] Free 0x{{.*}}
    // CHECK: [Trace] Free 0x{{.*}}
    // CHECK: [Trace] Free 0x{{.*}}
    // CHECK: [Trace] Free 0x{{.*}}

    return 0;
}
