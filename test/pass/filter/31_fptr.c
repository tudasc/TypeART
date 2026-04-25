// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack=true --typeart-filter=true --typeart-filter-implementation=acg --typeart-filter-cg-file=%p/31_fptr.mcg 2>&1 | %filecheck %s
// clang-format on

void MPI_mock(void* buf) {
}

int never_called(int* never) {
  MPI_mock(never);
  return 0;
}

int forward_mock(int* a) {
  float c = 2.f;
  MPI_mock(&c);
  return *a;
}

int (*forward_mock_ptr)(int*) = forward_mock;

void bar(int* ptr, double* bptr) {
  forward_mock_ptr(ptr);  // in mcg: can be (forward_mock | never_called) -> cannot filter ptr
  MPI_mock(bptr);
}

void foo() {
  int a    = 0;   // a -> bar(a,..) -> forward_mock_ptr(a)
  double b = 1.;  // b -> bar(...,b) -> MPI_mock(b)
  bar(&a, &b);
}

// ACG:
// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :
// CHECK-NEXT: Stack call filtered %  : 0