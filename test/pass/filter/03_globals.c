// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-alloca -call-filter -call-filter-impl=deprecated::default -S 2>&1 | %filecheck %s
// RUN: %c-to-llvm %s | %apply-typeart -typeart-alloca -call-filter  -S 2>&1 | %filecheck %s
// clang-format on

int a;
double x[3];

extern void bar(int* v);
void foo() {
  bar(&a);
}

// CHECK: MemInstFinderPass
// Global                 :     2
// Global filter total    :     1
// Global call filtered % : 50.00
// Global filtered %      : 50.00