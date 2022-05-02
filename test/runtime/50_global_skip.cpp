// RUN: %run %s 2>&1 | %filecheck %s
// RUN: %run %s -typeart-global=false 2>&1 | %filecheck %s --check-prefix CHECK-SKIP

// REQUIRES: softcounter

double global_d;

void f() {
  static int global_t{0};
  char c[7];
  double d = 5;
}

int main(int argc, char** argv) {
  f();
  f();
  return 0;
}

// CHECK: Allocation type detail (heap, stack, global)
// CHECK: 0 :   0 ,    {{[0-9]}} ,    0 , int8
// CHECK: 2 :   0 ,    {{[0-9]}} ,    1 , int32
// CHECK: 6 :   0 ,    {{[0-9]}} ,    1 , double

// CHECK-SKIP: Allocation type detail (heap, stack, global)
// CHECK-SKIP: 0 :   0 ,    {{[0-9]}} ,    0 , int8
// CHECK-SKIP: 2 :   0 ,    {{[0-9]}} ,    0 , int32
// CHECK-SKIP: 6 :   0 ,    {{[0-9]}} ,    0 , double
