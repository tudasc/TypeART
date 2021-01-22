// clang-format off
// RUN: %run %s 2>&1 | FileCheck %s
// clang-format on

#include <stdlib.h>

#include <thread>
#include <stdio.h>

void f() {
  char c[4];
  double d = 5;
}

int main(int argc, char** argv) {

  constexpr unsigned n = 4;

  // CHECK: [Trace] TypeART Runtime Trace

  std::thread t1(f);
  std::thread t2(f);

  t1.join();
  t2.join();

  // CHECK-NOT: [Error]

  // CHECK: [Trace] Free 0x{{.*}} 0 int8 1 4
  // CHECK-NEXT: [Trace] Free 0x{{.*}} 6 double 8 1
  // CHECK-NEXT: [Trace] Stack after free

  // CHECK: [Trace] Free 0x{{.*}} 0 int8 1 4
  // CHECK-NEXT: [Trace] Free 0x{{.*}} 6 double 8 1
  // CHECK-NEXT: [Trace] Stack after free


  return 0;
}
