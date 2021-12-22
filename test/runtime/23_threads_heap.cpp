// clang-format off
// RUN: %run %s --thread 2>&1 | %filecheck %s --check-prefix=CHECK-TSAN
// RUN: %run %s --thread 2>&1 | %filecheck %s
// REQUIRES: thread && softcounter
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <thread>

void repeat_alloc_free(unsigned n) {
  for (int i = 0; i < n; i++) {
    double* d = (double*)malloc(sizeof(double) * n);
    free(d);
  }
}

int main(int argc, char** argv) {
  constexpr unsigned n = 1000;

  // CHECK: [Trace] TypeART Runtime Trace

  std::thread t1(repeat_alloc_free, n);
  std::thread t2(repeat_alloc_free, n);
  std::thread t3(repeat_alloc_free, n);

  t1.join();
  t2.join();
  t3.join();

  // CHECK-TSAN-NOT: ThreadSanitizer

  // CHECK-NOT: Error
  // CHECK: Allocation type detail (heap, stack, global)
  // CHECK: 6   : 3000 ,    0 ,    0 , double
  // CHECK: Free allocation type detail (heap, stack)
  // CHECK: 6   : 3000 ,    0 , double

  return 0;
}
