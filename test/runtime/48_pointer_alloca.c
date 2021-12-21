// clang-format off
// RUN: %run %s -typeart-filter-pointer-alloca=true 2>&1 | FileCheck %s
// RUN: %run %s -typeart-filter-pointer-alloca=false 2>&1 | FileCheck %s --check-prefix CHECK-pointer
// clang-format on

int main(int argc, char** argv) {
  int n = argc * 2;
  int* x;
  int* y = &n;
  return 0;
}

// CHECK-NOT: 10 :   0 ,    {{[1-9]+}} ,    0 , pointer
// CHECK-pointer: 10 :   0 ,    {{[1-9]+}} ,    0 , pointer