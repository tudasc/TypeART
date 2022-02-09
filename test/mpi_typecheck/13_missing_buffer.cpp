// clang-format off
// RUN: %run %s --manual --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe"
// RUN: cat "%s.log/1/rank.0/stderr" | %filecheck %s
// RUN: cat "%s.log/1/rank.1/stderr" | %filecheck %s
// clang-format on

// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan

#include "Util.hpp"

#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  double f[4];

  // CHECK-DAG: Buffer not registered
  // CHECK-DAG: MCounter { Error: 1
  run_test(f, 1, MPI_DOUBLE);

  MPI_Finalize();
  return 0;
}
