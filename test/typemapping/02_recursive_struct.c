// RUN: %remove %tu_yaml
// RUN: %c-to-llvm %s | %apply-typeart
// RUN: cat %tu_yaml | %filecheck %s

// Note: This test assumes standard alignment on a 64bit system. Non-standard alignment may lead to failure.

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

// CHECK: - id:              256
// CHECK: name:            {{(struct.)?}}s1_t
// CHECK:         extent:          16
// CHECK: member_count:    2
// CHECK: offsets:         [ 0, 8 ]
// CHECK: types:           [ 10, 1 ]
// CHECK: sizes:           [ 3, 1 ]

// CHECK: - id:              257
// CHECK: name:            {{(struct.)?}}s2_t
// CHECK:         extent:          32
// CHECK: member_count:    3
// CHECK: offsets:         [ 0, 16, 24 ]
// CHECK: types:           [ 256, 1, 1 ]
// CHECK: sizes:           [ 1, 1, 1 ]

// CHECK: - id:              258
// CHECK: name:            {{(struct.)?}}s3_t
// CHECK:         extent:          64
// CHECK: member_count:    3
// CHECK: offsets:         [ 0, 32, 40 ]
// CHECK: types:           [ 256, 10, 1 ]
// CHECK: sizes:           [ 2, 1, 3 ]
