// RUN: %run %s --mpi_intercept 2>&1 | FileCheck %s

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
    MPI_Send(f, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(f, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // CHECK: [Trace] Free 0x{{.*}}
  delete[] f;

  MPI_Finalize();
  return 0;
}
