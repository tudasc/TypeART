// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe" -typeart-call-filter
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

  // CHECK: Type_Error: 0
  float data{0.0f};
  run_test(&data, sizeof(float), MPI_BYTE);

  MPI_Finalize();
  return 0;
}
