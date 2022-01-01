// clang-format off
// RUN: %run %s --thread 2>&1 | %filecheck %s --check-prefix=CHECK-TSAN
// RUN: %run %s --thread 2>&1 | %filecheck %s
// REQUIRES: thread
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <thread>

void f() {
  char c[7];
  double d = 5;
}

int main(int argc, char** argv) {
  constexpr unsigned n = 4;

  // CHECK: [Trace] TypeART Runtime Trace

  std::thread t1(f);
  std::thread t2(f);

  t1.join();
  t2.join();

  // CHECK-TSAN-NOT: ThreadSanitizer

  // CHECK-NOT: Error

  // CHECK: [Trace] Free 0x{{.*}} 0 int8 1 7
  // CHECK: [Trace] Free 0x{{.*}} 6 double 8 1

  // CHECK: [Trace] Free 0x{{.*}} 0 int8 1 7
  // CHECK: [Trace] Free 0x{{.*}} 6 double 8 1

  return 0;
}
