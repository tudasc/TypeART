// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe"
// RUN: cat "%s.log/1/rank.0/stderr" | FileCheck %s
// RUN: cat "%s.log/1/rank.1/stderr" | FileCheck %s
// clang-format on

// REQUIRES: mpi
// UNSUPPORTED: asan

// XFAIL: *

#include "Util.hpp"

#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  // CHECK: Type_Error: 0
  float data{0.0f};
  run_test(&data, sizeof(float), MPI_BYTE);

  MPI_Finalize();
  return 0;
}
