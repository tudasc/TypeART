// RUN: %remove %tu_yaml
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack=true
// RUN: cat %tu_yaml | %filecheck %s

// REQUIRES: llvm-18 || llvm-19

union UnionTy {
  float a;
  int b;
};

void foo() {
  union UnionTy union_stack;
}

// CHECK: offsets:         [ 0, 0 ]
// CHECK-NEXT: types:           [ 23, 13 ]
// CHECK-NEXT: sizes:           [ 1, 1 ]
// CHECK-NEXT: flags:           8
