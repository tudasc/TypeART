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

typedef struct s1_t {
  char a[3];       // 0
  struct s1_t* b;  // 8
} s1;

typedef struct s2_t {
  s1 a;            // 0
  s1* b;           // 16
  struct s2_t* c;  // 24
} s2;

typedef struct s3_t {
  s1 a[2];   // 0
  char b;    // 32
  s2* c[3];  // 40
} s3;

int main(int argc, char** argv) {
  s1* a = malloc(sizeof(s1));
  s2* b = malloc(sizeof(s2));
  s3* c = malloc(sizeof(s3));
  free(c);
  free(b);
  free(a);
  return 0;
}

// CHECK: - id:              11
// CHECK: name:            struct.s1_t
// CHECK:         extent:          16
// CHECK: member_count:    2
// CHECK: offsets:         [ 0, 8 ]
// CHECK: types:
// CHECK: - id:              0
// CHECK: kind:            builtin
// CHECK: - id:              10
// CHECK: kind:            pointer
// CHECK: sizes:           [ 3, 1 ]
// CHECK: - id:              12
// CHECK: name:            struct.s2_t
// CHECK:         extent:          32
// CHECK: member_count:    3
// CHECK: offsets:         [ 0, 16, 24 ]
// CHECK: types:
// CHECK: - id:              11
// CHECK: kind:            struct
// CHECK: - id:              10
// CHECK: kind:            pointer
// CHECK: - id:              10
// CHECK: kind:            pointer
// CHECK: sizes:           [ 1, 1, 1 ]
// CHECK: - id:              13
// CHECK: name:            struct.s3_t
// CHECK:         extent:          64
// CHECK: member_count:    3
// CHECK: offsets:         [ 0, 32, 40 ]
// CHECK: types:
// CHECK: - id:              11
// CHECK: kind:            struct
// CHECK: - id:              0
// CHECK: kind:            builtin
// CHECK: - id:              10
// CHECK: kind:            pointer
// CHECK: sizes:           [ 2, 1, 3 ]
