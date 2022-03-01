// clang-format off
// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe" -typeart-call-filter
// RUN: cat "%s.log/1/rank.0/stderr" "%s.log/1/rank.1/stderr" | %filecheck %s
// clang-format on

// REQUIRES: mpi
// UNSUPPORTED: asan
// UNSUPPORTED: tsan

#include "Util.hpp"

#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // clang-format off
  // CHECK: R[{{0|1}}][Warning]T[{{[0-9]*}}] at 0x{{.*}}: MPI_Send: {{send|recv}}-buffer is NULL
  // CHECK: Null_Buf: 1
  // clang-format on

  double* null_buffer = nullptr;
  run_test(null_buffer, 0, MPI_DOUBLE);

  MPI_Finalize();
  return 0;
}
