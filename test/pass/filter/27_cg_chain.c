// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack=true --typeart-filter=true --typeart-filter-implementation=acg --typeart-filter-cg-file=%p/27_cg.mcg 2>&1 | %filecheck %s
// clang-format on
void MPI_mock(void* buf) {
}

int forward_mock(int* a) {
  float c = 2.f;
  MPI_mock(&c);
  return *a;
}
__attribute__((always_inline)) void bar(int* ptr, double* bptr) {
  forward_mock(ptr);
  MPI_mock(bptr);
}

void foo() {
  int a    = 0;   // a -> bar(a,..) -> forward_mock(a) -> OK
  double b = 1.;  // b -> bar(...,b) -> MPI_mock(b)
  bar(&a, &b);
}
// ACG:
// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :
// CHECK-NEXT: Stack call filtered %  : 33
