// RUN: %scriptpath/applyAndRun.sh %s %pluginpath "-must-alloca" %rtpath | FileCheck %s

#include <stdlib.h>

int main(int argc, char** argv) {
    const int n = 42;
    // CHECK: MUST Support Runtime Trace

    // CHECK: Alloc    0x{{.*}}    char    1   42
    char* a = malloc(n * sizeof(char));
    // CHECK: Free 0x{{.*}}
    free(a);

    // CHECK: Alloc    0x{{.*}}    short    2   42
    short* b = malloc(n * sizeof(short));
    // CHECK: Free 0x{{.*}}
    free(b);

    // CHECK: Alloc    0x{{.*}}    int    4   42
    int* c = malloc(n * sizeof(int));
    // CHECK: Free 0x{{.*}}
    free(c);

    // CHECK: Alloc    0x{{.*}}    long    8   42
    long* d = malloc(n * sizeof(long));
    // CHECK: Free 0x{{.*}}
    free(d);

    // CHECK: Alloc    0x{{.*}}    float    4   42
    float* e = malloc(n * sizeof(float));
    // CHECK: Free 0x{{.*}}
    free(e);

    // CHECK: Alloc    0x{{.*}}    double    8   42
    double* f = malloc(n * sizeof(double));
    // CHECK: Free 0x{{.*}}
    free(e);

    return 0;
}
