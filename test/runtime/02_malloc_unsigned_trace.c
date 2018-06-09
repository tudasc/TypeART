// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-must-alloca" %rtpath | FileCheck %s

#include <stdlib.h>

int main(int argc, char** argv) {
    const int n = 42;
    // CHECK: MUST Support Runtime Trace

    // CHECK: Alloc    0x{{.*}}    uint    1   42
    unsigned char* a = (unsigned char*)malloc(n * sizeof(unsigned char));
    // CHECK: Free 0x{{.*}}
    free(a);

    // CHECK: Alloc    0x{{.*}}    ushort    2   42
    unsigned short* b = (unsigned short*)malloc(n * sizeof(unsigned short));
    // CHECK: Free 0x{{.*}}
    free(b);

    // CHECK: Alloc    0x{{.*}}    uint    4   42
    unsigned int* c = (unsigned int*)malloc(n * sizeof(unsigned int));
    // CHECK: Free 0x{{.*}}
    free(c);

    // CHECK: Alloc    0x{{.*}}    ulong    8   42
    unsigned long* d = (unsigned long*)malloc(n * sizeof(unsigned long));
    // CHECK: Free 0x{{.*}}
    free(d);

    return 0;
}
