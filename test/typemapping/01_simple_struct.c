// RUN: rm /tmp/musttypes | clang -S -emit-llvm %s -o - | opt -load %pluginpath/%pluginname %pluginargs -S 2>&1; cat /tmp/musttypes | FileCheck %s

// Note: This test assumes standard alignment on a 64bit system. Non-standard alignment may lead to failure.

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

// CHECK: {{[0-9]*}}    struct.s_t  4   1   0,0,0,1
// CHECK: {{[0-9]*}}    struct.s2_t 16  3   0,0,0,1 4,0,2,1 8,0,4,1
// CHECK: {{[0-9]*}}    struct.s3_t 64  6   0,0,0,3 16,0,4,2    32,0,2,1    36,0,0,3    48,0,2,5    56,0,4,1
// CHECK: {{[0-9]*}}    struct.s4_t 64  4   0,0,0,1 8,0,7,3 32,0,7,3    56,2,8,1