// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe"
// RUN: cat "%s.log/1/rank.0/stderr" "%s.log/1/rank.1/stderr" | FileCheck %s
// clang-format on

// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan

#include "Util.hpp"

#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  // clang-format off
  // CHECK: R[{{0|1}}][Info]ID[0] MPI_Send at 0x{{.*}}: checking send-buffer {{.*}} of type "" against MPI type "MPI_DOUBLE"
  // CHECK: R[{{0|1}}][Warning]ID[0] buffer is NULL
  // CHECK: Null_Buf: 1
  // clang-format on

  double* null_buffer = nullptr;
  run_test(null_buffer, 0, MPI_DOUBLE);

  MPI_Finalize();
  return 0;
}
