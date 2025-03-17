// RUN: %remove %tu_yaml
// RUN: %c-to-llvm %s | %apply-typeart
// RUN: cat %tu_yaml | %filecheck %s

#include <stdlib.h>

typedef struct {
  float real;
  float imag;
} complex;

typedef struct {
  complex e[4][4];
} su2_matrix;

void foo() {
  su2_matrix* matrix = malloc(sizeof(su2_matrix));
}

// CHECK:   name:            {{.*}}complex
// CHECK-NEXT:   extent:          8
// CHECK-NEXT:   member_count:    2
// CHECK-NEXT:   offsets:         [ 0, 4 ]
// CHECK-NEXT:   types:           [ 23, 23 ]
// CHECK-NEXT:   sizes:           [ 1, 1 ]