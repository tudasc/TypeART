// RUN: rm musttypes | clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/MemInstFinderPass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1; cat musttypes | FileCheck %s

// Note: This test assumes standard alignment on a 64bit system. Non-standard alignment may lead to failure.

//typedef enum must_builtin_type_t {
//    C_CHAR = 0,
//    C_UCHAR = 1,
//    C_SHORT = 2,
//    C_USHORT = 3,
//    C_INT = 4,
//    C_UINT = 5,
//    C_LONG = 6,
//    C_ULONG = 7,
//    C_FLOAT = 8,
//    C_DOUBLE = 9,
//    INVALID = 10,
//    N_BUILTIN_TYPES
//} must_builtin_type;

#include <stdlib.h>

typedef struct s_t
{
    int a;
} s;

typedef struct s2_t
{
    int a; // 0
    char b; // 4
    long c; // 8
} s2;

typedef struct s3_t
{
    int a[3]; // 0
    long b[2]; // 16
    char c; // 32
    unsigned int d[3]; // 36
    char e[5]; // 48
    unsigned long f; // 56
} s3;

typedef struct s4_t
{
    int a; // 0
    double b[3]; // 8
    double c[3]; // 32
    struct s4_t* d; // 56
} s4;

int main(int argc, char** argv)
{
    s* a = malloc(sizeof(s));
    s2* b = malloc(sizeof(s2));
    s3* c = malloc(sizeof(s3));
    s4* d = malloc(sizeof(s4));
    free (d);
    free(c);
    free(b);
    free(a);
    return 0;
}

// CHECK: {{[0-9]*}}    struct.s_t  4   1   0,0,4,1
// CHECK: {{[0-9]*}}    struct.s2_t 16  3   0,0,4,1 4,0,0,1 8,0,6,1
// TODO: Replace the following line as soon as unsigned types are supported with "{{[0-9]*}}    struct.s3_t 64  6   0,0,4,3 16,0,6,2    32,0,0,1    36,0,5,3    48,0,0,5    56,0,7,1"
// CHECK: {{[0-9]*}}    struct.s3_t 64  6   0,0,4,3 16,0,6,2    32,0,0,1    36,0,4,3    48,0,0,5    56,0,6,1
// CHECK: {{[0-9]*}}    struct.s4_t 64  4   0,0,4,1 8,0,9,3 32,0,9,3    56,2,10,1
