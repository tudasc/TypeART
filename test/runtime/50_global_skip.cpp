// RUN: %run %s 2>&1 | %filecheck %s
// RUN: %run %s --typeart-global=false 2>&1 | %filecheck %s --check-prefix CHECK-SKIP

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
// CHECK: {{(10|5)}} :   0 ,    {{[0-9]}} ,    0 , {{(int8_t|char)}}
// CHECK: 12 :   0 ,    {{[0-9]}} ,    1 , int
// CHECK: 23 :   0 ,    {{[0-9]}} ,    1 , double

// CHECK-SKIP: Allocation type detail (heap, stack, global)
// CHECK-SKIP: {{(10|5)}} :   0 ,    {{[0-9]}} ,    0 , {{(int8_t|char)}}
// CHECK-SKIP: 12 :   0 ,    {{[0-9]}} ,    0 , int
// CHECK-SKIP: 23 :   0 ,    {{[0-9]}} ,    0 , double
