// clang-format off
// RUN: %run %s --mpi_intercept --mpi_output_filename "%s.log" && cat "%s.log/1/rank.0/stderr" | FileCheck --check-prefixes CHECK,RANK0 %s && cat "%s.log/1/rank.1/stderr" | FileCheck --check-prefixes CHECK,RANK1 %s
// clang-format on

#include <mpi.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // CHECK: [Trace] TypeART Runtime Trace

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Datatype mpi_double_vec;
  MPI_Type_contiguous(3, MPI_DOUBLE, &mpi_double_vec);
  MPI_Type_commit(&mpi_double_vec);

  // RANK0: [Trace] Alloc 0x{{.*}} double 8 9
  const auto n = 9;
  auto f       = new double[n];

  if (rank == 0) {
    // RANK0: R[0][Info][1] MPI_Send: buffer 0x{{.*}} has type double, MPI type is double
    MPI_Send(f, 3, mpi_double_vec, 1, 0, MPI_COMM_WORLD);
  } else {
    // RANK1: R[1][Info][0] MPI_Recv: buffer 0x{{.*}} has type double, MPI type is double
    MPI_Recv(f, 3, mpi_double_vec, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // RANK0: [Trace] Free 0x{{.*}}
  delete[] f;

  // RANK0: [Trace] Alloc 0x{{.*}} double 8 8
  auto too_small = new double[n - 1];
  if (rank == 0) {
    // clang-format off
    // RANK0: R[0][Info][1] MPI_Send: buffer 0x{{.*}} has type double, MPI type is double
    // RANK0: R[0][Error][1] MPI_Send: buffer 0x{{.*}} too small. The buffer can only hold 8 elements (9 required)
    // clang-format on
    MPI_Send(too_small, 3, mpi_double_vec, 1, 0, MPI_COMM_WORLD);
  } else {
    // clang-format off
    // RANK1: R[1][Info][0] MPI_Recv: buffer 0x{{.*}} has type double, MPI type is double
    // RANK1: R[1][Error][0] MPI_Recv: buffer 0x{{.*}} too small. The buffer can only hold 8 elements (9 required)
    // clang-format on
    MPI_Recv(too_small, 3, mpi_double_vec, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // RANK0: [Trace] Free 0x{{.*}}
  delete[] too_small;

  MPI_Type_free(&mpi_double_vec);
  MPI_Finalize();
  return 0;
}
