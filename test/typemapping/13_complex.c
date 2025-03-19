// RUN: %remove %tu_yaml
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack=true
// RUN: cat %tu_yaml | %filecheck %s

// REQUIRES: llvm-18 || llvm-19

#include <complex.h>

typedef struct CplxY {
  float complex f_cplx;
  double complex d_cplx;
  long double complex ld_cplx;
} Y;

void foo() {
  Y struct_y;
}

// CHECK: name:            {{(CplxY|struct.CplxY)}}
// CHECK-NEXT: extent:          64
// CHECK-NEXT: member_count:    3
// CHECK-NEXT: offsets:         [ 0, 8, 32 ]
// CHECK-NEXT: types:           [ 26, 27, 28 ]
// CHECK-NEXT: sizes:           [ 1, 1, 1 ]
