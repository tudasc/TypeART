// clang-format off
// RUN: %run %s 2>&1 | %filecheck %s
// clang-format on

#include <new>

struct S1 {
  int x;
  virtual ~S1() = default;
};

void foo() {
  S1* b{nullptr};
  try {
    b = new S1;
  } catch (...) {
  }
  if (b != nullptr) {
    delete b;
  }
}

int main() {
  try {
    S1* ss = new S1;
    delete ss;  // LLVM does not call _ZdaPv here, but in destructor @_ZN2S1D0Ev
  } catch (...) {
  }

  foo();

  return 0;
}
// main()
// CHECK: [Trace] Alloc [[POINTER:0x[0-9a-f]+]] [[typeid:2[5-9][0-9]]] struct.S1 16 1 ([[ALLOC_FROM:0x[0-9a-f]+]]) H
// CHECK: [Trace] Free [[POINTER]] [[typeid]] struct.S1 16 1 ([[ALLOC_FROM]])
// foo()
// CHECK: [Trace] Alloc [[POINTER:0x[0-9a-f]+]] [[typeid:2[5-9][0-9]]] struct.S1 16 1 ([[ALLOC_FROM:0x[0-9a-f]+]]) H
// CHECK: [Trace] Free [[POINTER]] [[typeid]] struct.S1 16 1 ([[ALLOC_FROM]])
