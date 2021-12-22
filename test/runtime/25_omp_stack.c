// clang-format off
// RUN: %run %s --omp 2>&1 | %filecheck %s --check-prefix=CHECK-TSAN
// RUN: %run %s --omp 2>&1 | %filecheck %s
// REQUIRES: openmp
// clang-format on

void f() {
  char c[4];
  double d = 5;
}

int main(int argc, char** argv) {
  // CHECK: [Trace] TypeART Runtime Trace
#pragma omp parallel sections
  {
#pragma omp section
    f();
#pragma omp section
    f();
  }

  // CHECK-TSAN-NOT: ThreadSanitizer

  // CHECK-NOT: Error

  // CHECK: [Trace] Free 0x{{.*}} 0 int8 1 4
  // CHECK-DAG: [Trace] Free 0x{{.*}} 6 double 8 1

  // CHECK-DAG: [Trace] Free 0x{{.*}} 0 int8 1 4
  // CHECK-DAG: [Trace] Free 0x{{.*}} 6 double 8 1

  return 0;
}
