// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-stack -typeart-filter-pointer-alloca=true -S 2>&1 | %filecheck %s
// clang-format on

int main(int argc, char** argv) {
  int n = argc * 2;
  int* x;
  int* y = &n;
  return 0;
}

// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                      :   6.00
// CHECK-NEXT: Stack call filtered %       :   0.00
// CHECK-NEXT: Alloca of pointer discarded :   3.00
