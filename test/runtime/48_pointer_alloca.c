// clang-format off
// RUN: %run %s --typeart-analysis-filter-pointer-alloca=true 2>&1 | %filecheck %s
// RUN: %run %s --typeart-analysis-filter-pointer-alloca=false 2>&1 | %filecheck %s --check-prefix CHECK-pointer
// clang-format on

// REQUIRES: softcounter

int main(int argc, char** argv) {
  int n = argc * 2;
  int* x;
  int* y = &n;
  return 0;
}

// CHECK-NOT: 1 :   0 ,    {{[1-9]+}} ,    0 , ptr
// CHECK-pointer: 1 :   0 ,    {{[1-9]+}} ,    0 , ptr
