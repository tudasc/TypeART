// RUN: %run %s --mpi_intercept --mpi_output_filename "%s.log" 2>&1 && find . -type f -exec cat {} \; | FileCheck %s

#include <mpi.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  // CHECK: [Trace] Alloc 0x{{.*}} double 8 16
  const auto n = 16;
  auto f       = new double[n];

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    // CHECK: R[0][Info][1] MPI_Send: buffer 0x{{.*}} has type double, MPI type is double
    MPI_Send(f, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  } else {
    // CHECK: R[1][Info][0] MPI_Recv: buffer 0x{{.*}} has type double, MPI type is double
    MPI_Recv(f, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // CHECK: [Trace] Free 0x{{.*}}
  delete[] f;

  MPI_Finalize();
  return 0;
}
