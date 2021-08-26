// REQUIRES: mpi
// UNSUPPORTED: asan
// clang-format off
//
// Note:
// In this test we want to check the console output of the MPI interceptor for
// nullptr buffers. With a non-zero count, the MPI function will abort the
// program. Therefore we add `|| true` to let the test succeed even if the
// run command fails as long as the expected output is produced.

// RUN: %run %s --mpi_intercept --compile_flags "-g" --executable %s.exe --command "%mpi-exec -n 2 --output-filename %s.log %s.exe" || true
// RUN: cat "%s.log/1/rank.0/stderr" "%s.log/1/rank.1/stderr" | FileCheck %s
// clang-format on

#include "Util.hpp"

#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  // clang-format off
  //
  // Here, we only check that the expected output exists in at least one of the
  // processes as the other one may be aborted even before it reaches the MPI call.
  //
  // CHECK: R[{{0|1}}][Info]ID[0] run_test(void*, int, {{.*}}[0x{{.*}}] at {{(/.*)*/.*\..*}}:{{[0-9]+}}: MPI_Send: checking send-buffer {{.*}} of type "" against MPI type "MPI_DOUBLE"
  // CHECK: R[{{0|1}}][Error]ID[0] buffer is NULL
  // clang-format on
  run_test(MPI_BOTTOM, 4, MPI_DOUBLE);

  MPI_Finalize();
  return 0;
}
