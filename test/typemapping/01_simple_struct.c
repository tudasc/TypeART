// RUN: %remove %tu_yaml
// RUN: %c-to-llvm %s | %apply-typeart
// RUN: cat %tu_yaml | %filecheck %s

// Note: This test assumes standard alignment on a 64bit system. Non-standard alignment may lead to failure.

#include <stdlib.h>

// TODO: Use structs from struct_defs.h

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
  char e[6];          // 48
  unsigned long f;    // 56
} s3;

typedef struct s4_t {
  int a;           // 0
  double b[3];     // 8
  double c[3];     // 32
  struct s4_t* d;  // 56
} s4;

int main(int argc, char** argv) {
  s* a  = malloc(sizeof(s));
  s2* b = malloc(sizeof(s2));
  s3* c = malloc(sizeof(s3));
  s4* d = malloc(sizeof(s4));
  free(d);
  free(c);
  free(b);
  free(a);
  return 0;
}

// CHECK: - id:              256
// CHECK: name:            {{(struct.)?}}s_t
// CHECK: extent:          4
// CHECK: member_count:    1
// CHECK: offsets:         [ 0 ]
// CHECK: types:           [ 13 ]
// CHECK: sizes:           [ 1 ]

// CHECK: - id:              257
// CHECK: name:            {{(struct.)?}}s2_t
// CHECK:         extent:          16
// CHECK: member_count:    3
// CHECK: offsets:         [ 0, 4, 8 ]
// CHECK: types:           [ 13, {{(11|6)}}, 14 ]
// CHECK: sizes:           [ 1, 1, 1 ]

// CHECK: - id:              258
// CHECK: name:            {{(struct.)?}}s3_t
// CHECK:         extent:          64
// CHECK: member_count:    6
// CHECK: offsets:         [ 0, 16, 32, 36, 48, 56 ]
// CHECK: types:           [ 13, 14, {{(11|6)}}, {{(13|18)}}, {{(11|6)}}, {{(14|19)}} ]
// CHECK: sizes:           [ 3, 2, 1, 3, 6, 1 ]

// CHECK: - id:              259
// CHECK: name:            {{(struct.)?}}s4_t
// CHECK:         extent:          64
// CHECK: member_count:    4
// CHECK: offsets:         [ 0, 8, 32, 56 ]
// CHECK: types:           [ 13, 24, 24, 1 ]
// CHECK: sizes:           [ 1, 3, 3, 1 ]
