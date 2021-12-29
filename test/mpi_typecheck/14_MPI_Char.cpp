// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe"
// RUN: cat "%s.log/1/rank.0/stderr" | FileCheck %s
// RUN: cat "%s.log/1/rank.1/stderr" | FileCheck %s
// clang-format on

// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan

#include "Util.hpp"

#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  // CHECK: [1 x int8] against 1 element of MPI type "MPI_CHAR"
  // CHECK: Type_Error: 0
  char data{'A'};
  run_test(&data, 1, MPI_CHAR);

  MPI_Finalize();
  return 0;
}
