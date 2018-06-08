// clang-format off
// RUN: rm musttypes | clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/MemInstFinderPass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1; cat musttypes | FileCheck %s
// clang-format on

// Note: This test assumes standard alignment on a 64bit system. Non-standard alignment may lead to failure.

// typedef enum must_builtin_type_t {
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

typedef struct s_t {
  int a;
} s;

typedef struct s2_t {
  int a;   // 0
  char b;  // 4
  long c;  // 8
} s2;

typedef struct s3_t {
  int a[3];           // 0
  long b[2];          // 16
  char c;             // 32
  unsigned int d[3];  // 36
  char e[5];          // 48
  unsigned long f;    // 56
} s3;

typedef struct s4_t {
  int a;           // 0
  double b[3];     // 8
  double c[3];     // 32
  struct s4_t* d;  // 56
} s4;

int main(int argc, char** argv) {
  s* a = malloc(sizeof(s));
  s2* b = malloc(sizeof(s2));
  s3* c = malloc(sizeof(s3));
  s4* d = malloc(sizeof(s4));
  free(d);
  free(c);
  free(b);
  free(a);
  return 0;
}

// CHECK: - id:              11
// CHECK: name:            struct.s_t
// CHECK: extent:          4
// CHECK: member_count:    1
// CHECK: offsets:         [ 0 ]
// CHECK: types:
// CHECK: - id:              4
// CHECK: kind:            builtin
// CHECK: sizes:           [ 1 ]
// CHECK: - id:              12
// CHECK: name:            struct.s2_t
// CHECK:         extent:          16
// CHECK: member_count:    3
// CHECK: offsets:         [ 0, 4, 8 ]
// CHECK: types:
// CHECK: - id:              4
// CHECK: kind:            builtin
// CHECK: - id:              0
// CHECK: kind:            builtin
// CHECK: - id:              6
// CHECK: kind:            builtin
// CHECK: sizes:           [ 1, 1, 1 ]
// CHECK: - id:              13
// CHECK: name:            struct.s3_t
// CHECK:         extent:          64
// CHECK: member_count:    6
// CHECK: offsets:         [ 0, 16, 32, 36, 48, 56 ]
// CHECK: types:
// CHECK: - id:              4
// CHECK: kind:            builtin
// CHECK: - id:              6
// CHECK: kind:            builtin
// CHECK: - id:              0
// CHECK: kind:            builtin
// TODO: Change 4 -> 5 as soon as unsigned types are supported
// CHECK: - id:              4
// CHECK: kind:            builtin
// CHECK: - id:              0
// CHECK: kind:            builtin
// TODO: Change 6 -> 7 as soon as unsigned types are supported
// CHECK: - id:              6
// CHECK: kind:            builtin
// CHECK: sizes:           [ 3, 2, 1, 3, 5, 1 ]
// CHECK: - id:              14
// CHECK: name:            struct.s4_t
// CHECK:         extent:          64
// CHECK: member_count:    4
// CHECK: offsets:         [ 0, 8, 32, 56 ]
// CHECK: types:
// CHECK: - id:              4
// CHECK: kind:            builtin
// CHECK: - id:              9
// CHECK: kind:            builtin
// CHECK: - id:              9
// CHECK: kind:            builtin
// CHECK: - id:              10
// CHECK: kind:            pointer
// CHECK: sizes:           [ 1, 3, 3, 1 ]
