// clang-format off
// RUN: %run %s --mpi_intercept --mpi_output_filename "%s.log" && cat "%s.log/1/rank.0/stderr" | FileCheck --check-prefixes CHECK,RANK0 %s && cat "%s.log/1/rank.1/stderr" | FileCheck --check-prefixes CHECK,RANK1 %s
// clang-format on

#include <mpi.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  const auto n = 16;
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  // RANK0: [Trace] Alloc 0x{{.*}} double 8 16
  auto f = new double[n];
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    // RANK0: R[0][Info][1] MPI_Send: buffer 0x{{.*}} has type double, MPI type is double
    MPI_Send(f, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  } else {
    // RANK1: R[1][Info][0] MPI_Recv: buffer 0x{{.*}} has type double, MPI type is double
    MPI_Recv(f, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // RANK0: [Trace] Free 0x{{.*}}
  delete[] f;

  MPI_Finalize();
  return 0;
}
