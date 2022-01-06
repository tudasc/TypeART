// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-stack -typeart-call-filter  -S 2>&1 | %filecheck %s
// clang-format on

extern int d;

void empty() {
  int a = 1;
  int b = 2;
  int c = 3;

  if (d > c) {
    b = a * c;
  } else {
    b = c * c;
  }
}

// Standard filter
// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :  3.00
// CHECK-NEXT: Stack call filtered %  :  100.00