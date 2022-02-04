// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-stack -typeart-call-filter  -S 2>&1 | %filecheck %s
// clang-format on

int a;
double x[3];

extern void bar(int* v);
void foo() {
  bar(&a);
}

// CHECK: MemInstFinderPass
// CHECK: Global                 :     2
// CHECK: Global filter total    :     1
// CHECK: Global call filtered % : 50.00
// CHECK: Global filtered %      : 50.00